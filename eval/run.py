"""Run the eval harness and write metrics + calibration curve.

Usage:
    python -m eval.run                      # uses configured LLM / mock
    python -m eval.run --samples 3
"""
import argparse
import json
import os

from app.config import settings
from app.grading.engine import GradingEngine
from eval.dataset import ITEMS
from eval.harness import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=None, help="grading samples per item")
    parser.add_argument("--out", type=str, default=os.path.join("eval", "results"), help="output dir")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Running eval on {len(ITEMS)} items (mock={settings.using_mock}, model={settings.llm_model})...")

    engine = GradingEngine()
    metrics, curve, details = run_eval(engine, ITEMS, n_samples=args.samples)

    out_json = os.path.join(args.out, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "calibration": curve, "details": details}, f, indent=2)

    # Calibration curve plot
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
        ax.set_title(f"Confidence calibration (AUROC={metrics['auroc']:.3f}, n={metrics['n_items']})")
        ax.legend()
        ax.grid(alpha=0.3)
        plot_path = os.path.join(args.out, "calibration_curve.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {plot_path}")
    except Exception as exc:
        print(f"[warn] could not render plot: {exc}")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics -> {out_json}")


if __name__ == "__main__":
    main()
