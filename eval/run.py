"""Run the eval harness and write metrics + calibration curve.

Usage:
    python -m eval.run                      # uses configured LLM / mock
    python -m eval.run --samples 3
    python -m eval.run --samples 5 --repeats 3
"""
import argparse
import json
import os

import numpy as np

from app.config import settings
from app.grading.engine import GradingEngine
from eval.dataset import ITEMS
from eval.harness import calibration_curve, run_eval

ABLATION_KEYS = ("agree", "variance", "semantic", "cross", "combined")


def _mean_std(values):
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None}
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _summarize(runs):
    ms = [r["metrics"] for r in runs]

    def col(key):
        return _mean_std([m.get(key) for m in ms])

    def ablation(key):
        return {k: _mean_std([m[key].get(k) for m in ms]) for k in ABLATION_KEYS}

    flag = [m["flagged_at_threshold"] for m in ms]
    return {
        "repeats": len(runs),
        "mae": col("mae"),
        "mae_before_adjudication": col("mae_before_adjudication"),
        "agreement_rate": col("agreement_rate"),
        "agreement_rate_before_adjudication": col("agreement_rate_before_adjudication"),
        "auroc": col("auroc"),
        "auroc_pre_adjudication": col("auroc_pre_adjudication"),
        "n_adjudicated": col("n_adjudicated"),
        "ablation": ablation("ablation"),
        "ablation_pre_adjudication": ablation("ablation_pre_adjudication"),
        "flagged_at_threshold": {
            "threshold": flag[0]["threshold"],
            "error_rate_flagged_post": _mean_std([f["post"]["error_rate_flagged"] for f in flag]),
            "error_rate_unflagged_post": _mean_std([f["post"]["error_rate_unflagged"] for f in flag]),
            "error_rate_flagged_pre": _mean_std([f["pre"]["error_rate_flagged"] for f in flag]),
            "error_rate_unflagged_pre": _mean_std([f["pre"]["error_rate_unflagged"] for f in flag]),
        },
    }


def _plot(curve, auroc, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        centers = curve["bin_centers"]
        rates = curve["error_rates"]
        ax.plot([0, 1], [1, 0], "--", color="gray", label="perfect calibration (1 - confidence)")
        if centers:
            ax.plot(centers, rates, "o-", color="#6c8cff", linewidth=2, label="observed error rate")
        for x, y, c in zip(centers, rates, curve["counts"]):
            ax.annotate(str(c), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=9)
        ax.set_xlabel("Predicted confidence (bin)")
        ax.set_ylabel("Error rate (|pred - truth| >= margin)")
        title = "Confidence calibration" + (f" (AUROC={auroc:.3f})" if auroc is not None else "")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {path}")
    except Exception as exc:
        print(f"[warn] could not render plot: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=None, help="grading samples per item")
    parser.add_argument("--repeats", type=int, default=1, help="number of eval repetitions")
    parser.add_argument("--out", type=str, default=os.path.join("eval", "results"), help="output dir")
    args = parser.parse_args()

    repeats = max(1, args.repeats)
    os.makedirs(args.out, exist_ok=True)
    print(
        f"Running eval on {len(ITEMS)} items x {repeats} repeat(s) "
        f"(mock={settings.using_mock}, model={settings.llm_model})..."
    )

    engine = GradingEngine()
    runs, pooled_conf, pooled_err = [], [], []
    for r in range(repeats):
        metrics, curve, details = run_eval(engine, ITEMS, n_samples=args.samples)
        runs.append({"metrics": metrics, "details": details})
        pooled_conf += [d["confidence"] for d in details]
        pooled_err += [d["error"] for d in details]
        print(
            f"[run {r + 1}/{repeats}] MAE(post)={metrics['mae']:.4f} "
            f"agreement(post)={metrics['agreement_rate']:.3f} "
            f"AUROC(post)={metrics['auroc']:.3f} n_adj={metrics['n_adjudicated']}"
        )

    summary = _summarize(runs)
    pooled_curve = calibration_curve(pooled_conf, pooled_err, settings.error_margin)
    payload = {
        "repeats": repeats,
        "summary": summary,
        "runs": runs,
        "metrics": runs[0]["metrics"],  # first run, kept for backwards compatibility
        "calibration": pooled_curve,
        "details": runs[0]["details"],
    }
    out_json = os.path.join(args.out, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _plot(pooled_curve, summary["auroc"]["mean"], os.path.join(args.out, "calibration_curve.png"))

    print(json.dumps(summary, indent=2))
    print(f"Saved metrics -> {out_json}")


if __name__ == "__main__":
    main()
