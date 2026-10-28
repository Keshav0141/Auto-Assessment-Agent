"""Generate the README evaluation tables from eval/results/*.json.

    python -m eval.report           # rewrite the generated blocks in README.md
    python -m eval.report --check   # exit 1 if README.md is out of date

Each table lives between ``<!-- report:NAME -->`` and ``<!-- /report:NAME -->`` in
the README and is produced only from the result files, so the numbers cannot drift
from the runs. A test runs the check, so CI fails when they do.
"""
import argparse
import json
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

RESULTS_DIR = os.path.join("eval", "results")
README = "README.md"
LANG_NAMES = {"en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "en_rerun": "English, graded again"}


def load_results(results_dir: str = RESULTS_DIR) -> Dict[str, Any]:
    """Load every result file the report uses.

    Args:
        results_dir: Folder with metrics.json, baseline_2026-10-14.json,
            adversarial.json, multilingual.json and ocr_metrics.json.

    Returns:
        {name: parsed JSON} keyed by file name without the extension.
    """
    names = ("metrics", "baseline_2026-10-14", "adversarial", "multilingual", "ocr_metrics")
    out = {}
    for name in names:
        with open(os.path.join(results_dir, name + ".json"), encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


# --- formatting -------------------------------------------------------------------


def _f(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _ms(stat: Optional[Dict[str, Any]], digits: int = 2) -> str:
    if not stat or stat.get("mean") is None:
        return "-"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def _span(values: List[int]) -> str:
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _weights(w: Dict[str, float]) -> str:
    return " / ".join(f"{k} {v:g}" for k, v in w.items())


# --- blocks -----------------------------------------------------------------------


def grading_block(metrics: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    """Grading accuracy and calibration: this run against the Oct 14 baseline."""
    cfg = metrics.get("config", {})
    s, b = metrics["summary"], baseline["summary"]
    runs = [r["metrics"] for r in metrics["runs"]]
    pearson = [r["pearson_r"] for r in runs]
    pearson_stat = {"mean": statistics.fmean(pearson), "std": statistics.pstdev(pearson)}
    flagged = [r["flagged_at_threshold"]["post"]["n_flagged"] for r in runs]
    ft, bft = s["flagged_at_threshold"], b["flagged_at_threshold"]
    changed = s.get("n_referee_changed")
    rows = [
        ["MAE (marks)", _ms(s["mae"]), _ms(b["mae"])],
        ["MAE before adjudication", _ms(s["mae_before_adjudication"]), _ms(b["mae_before_adjudication"])],
        ["Agreement rate (error < 0.5)", _ms(s["agreement_rate"]), _ms(b["agreement_rate"])],
        ["Pearson r", _ms(pearson_stat), _ms(baseline["pearson_r"])],
        ["Calibration AUROC (combined confidence)", _ms(s["auroc"]), _ms(b["auroc"])],
        ["AUROC against pre-adjudication errors", _ms(s["auroc_pre_adjudication"]), _ms(b["auroc_pre_adjudication"])],
        [f"Error rate, flagged at {ft['threshold']:g}", _ms(ft["error_rate_flagged_post"]),
         _ms(bft["error_rate_flagged_post"])],
        ["Error rate, not flagged", _ms(ft["error_rate_unflagged_post"]), _ms(bft["error_rate_unflagged_post"])],
        ["Items flagged per run", _span(flagged), _span(baseline["n_flagged_per_run"])],
        ["Items refereed per run", _ms(s["n_adjudicated"]), _ms(b["n_adjudicated"])],
        ["Marks changed by the referee per run", _ms(changed), "not recorded"],
    ]
    intro = (
        f"Run of {cfg.get('date', '?')}: primary `{cfg.get('primary_model')}`, second grader "
        f"`{cfg.get('second_grader')}`, referee `{cfg.get('referee_model')}`; {cfg.get('samples')} samples, "
        f"{metrics['repeats']} repeats, {runs[0]['n_items']} items; confidence weights "
        f"{_weights(cfg.get('weights', {}))}, threshold {cfg.get('threshold')}."
    )
    return intro + "\n\n" + _table(["Metric", cfg.get("date", "this run"), "Oct 14 baseline"], rows)


def ablation_block(metrics: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    """AUROC of each confidence component on its own."""
    keys = (("agree", "Agreement"), ("variance", "Variance"), ("semantic", "Semantic"),
            ("cross", "Cross-model"), ("combined", "Combined"))
    s, b = metrics["summary"]["ablation"], baseline["summary"]["ablation"]
    rows = [[label, _ms(s[key]), _ms(b[key])] for key, label in keys]
    return _table(["Component", metrics.get("config", {}).get("date", "this run"), "Oct 14 baseline"], rows)


def adversarial_block(adv: Dict[str, Any]) -> str:
    """Pass count and any failed cases."""
    cfg, summary = adv["config"], adv["summary"]
    text = (
        f"**Result: {summary['passed']}/{summary['total']} passed** (primary `{cfg['primary_model']}`, "
        f"second grader `{cfg['second_grader']}`, referee `{cfg['referee_model']}`, fallback "
        f"`{cfg['fallback_model']}`, {cfg['samples']} samples)."
    )
    failed = [c for c in adv["cases"] if not c.get("passed")]
    if failed:
        rows = [[c["id"], c["expectation"], _f(c.get("mark")), ", ".join(c.get("integrity_flags") or []) or "-"]
                for c in failed]
        text += "\n\nFailed cases:\n\n" + _table(["Case", "Expectation", "Mark", "Flags"], rows)
    return text


def multilingual_block(ml: Dict[str, Any]) -> str:
    """Per-language summary and per-item marks."""
    cfg, summary = ml["config"], ml["summary"]
    rows = []
    for lang, s in summary.items():
        rows.append([
            LANG_NAMES.get(lang, lang), str(s["n"]), _f(s["mae"]), _f(s["mean_abs_diff_vs_en"]),
            _f(s["max_abs_diff_vs_en"], 1), _f(s["mean_confidence"]), str(s["n_flagged"]),
            str(s.get("n_script_warnings", "-")),
        ])
    intro = (
        f"Primary `{cfg['primary_model']}`, second grader `{cfg['second_grader']}`, referee "
        f"`{cfg['referee_model']}`; {cfg['samples']} samples; `FEEDBACK_LANGUAGE={cfg['feedback_language']}`."
    )
    header = ["Language", "Items", "MAE vs ground truth", "Mean \\|mark − English\\|", "Max", "Mean confidence",
              "Flagged", "Feedback-script warnings"]
    marks = {(r["id"], r["lang"]): r["mark"] for r in ml["rows"]}
    gts = {r["id"]: r["ground_truth"] for r in ml["rows"]}
    langs = list(summary)
    item_rows = [[item, f"{gts[item]:g}"] + [_f(marks.get((item, lang)), 1) for lang in langs]
                 for item in cfg["items"]]
    item_header = ["Item", "Ground truth"] + [LANG_NAMES.get(lang, lang) for lang in langs]
    return intro + "\n\n" + _table(header, rows) + "\n\nMarks per item:\n\n" + _table(item_header, item_rows)


def ocr_block(ocr: Dict[str, Any]) -> str:
    """CER/WER per language and capture style."""
    cfg = ocr["ocr_config"]
    host = urlparse(cfg.get("base_url") or "").netloc or cfg.get("base_url")
    blocks = dict(ocr.get("per_language_style") or {})
    for label in ("scanned PDF", "real photo"):
        if label in ocr.get("per_style", {}):
            blocks[f"English / {label}"] = ocr["per_style"][label]
    blocks["All images"] = ocr["overall"]
    rows = [[name, str(b["n_images"]), str(b["n_failures"]), _ms(b["cer"], 3), _ms(b["wer"], 3),
             _f(b["latency_s"]["mean"], 1)] for name, b in blocks.items()]
    intro = (
        f"Run of {ocr['generated_at'][:10]}: `{cfg['model']}` at `{host}` (backend `{cfg['backend']}`), "
        f"{ocr['n_images']} images, {ocr['repeats']} read each, no grading step."
    )
    return intro + "\n\n" + _table(["Language / style", "Images", "Failures", "CER", "WER", "Latency (s)"], rows)


def render(results: Dict[str, Any]) -> Dict[str, str]:
    """Render every README block from loaded results.

    Args:
        results: Output of `load_results`.

    Returns:
        {block name: markdown}.
    """
    metrics, baseline = results["metrics"], results["baseline_2026-10-14"]
    return {
        "grading": grading_block(metrics, baseline),
        "ablation": ablation_block(metrics, baseline),
        "adversarial": adversarial_block(results["adversarial"]),
        "multilingual": multilingual_block(results["multilingual"]),
        "ocr": ocr_block(results["ocr_metrics"]),
    }


def apply(text: str, blocks: Dict[str, str]) -> str:
    """Replace each `<!-- report:NAME -->` block in `text` with the rendered markdown.

    Raises:
        KeyError: If a block has no markers in the text.
    """
    for name, body in blocks.items():
        pattern = re.compile(rf"(<!-- report:{name} -->\n)(?:.*?\n)?(<!-- /report:{name} -->)", re.S)
        if not pattern.search(text):
            raise KeyError(f"README has no <!-- report:{name} --> block")
        text = pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
    return text


def main() -> None:
    """Rewrite (or with --check, verify) the generated blocks in README.md."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 if README.md is out of date")
    args = parser.parse_args()
    with open(README, encoding="utf-8") as f:
        current = f.read()
    updated = apply(current, render(load_results()))
    if args.check:
        if updated != current:
            print("README.md evaluation tables are out of date; run python -m eval.report")
            sys.exit(1)
        print("README.md evaluation tables are up to date.")
        return
    with open(README, "w", encoding="utf-8", newline="\n") as f:
        f.write(updated)
    print("Updated README.md evaluation tables.")


if __name__ == "__main__":
    main()
