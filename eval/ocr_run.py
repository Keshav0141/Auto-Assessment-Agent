"""OCR benchmark runner.

Measures transcription quality (CER, WER, question-label recall), latency and
failures per capture style, and the downstream grading impact of OCR noise.

Usage:
    python -m eval.make_ocr_images                     # build eval/ocr_data/
    python -m eval.ocr_run --provider goedel --limit 8
    python -m eval.ocr_run --provider openrouter --limit 8
    python -m eval.ocr_run --provider tesseract

Provider selection: set OCR_BASE_URL / OCR_API_KEY / OCR_MODEL in the shell (they
win over .env for this run) and pass a name with --provider. `tesseract` forces the
local Tesseract backend; any other name uses the vision-LLM backend (`vision`) at the
configured endpoint. Results are written to eval/results/ocr_metrics.json.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime, timezone
from typing import Optional

# Snapshot shell-provided OCR overrides BEFORE app.config calls load_dotenv(override=True),
# so `export OCR_BASE_URL=...` on the command line wins over .env for this run.
_SHELL_OCR = {
    key: os.environ.get(key)
    for key in (
        "OCR_PROVIDER",
        "OCR_BASE_URL",
        "OCR_API_KEY",
        "OCR_MODEL",
        "OCR_EXTRA_BODY",
        "OCR_DISABLE_THINKING",
    )
}

from app.config import settings  # noqa: E402 - must run after the snapshot above


def _apply_shell_overrides():
    s = _SHELL_OCR
    if s.get("OCR_PROVIDER"):
        settings.ocr_provider = s["OCR_PROVIDER"].strip().lower()
    if s.get("OCR_BASE_URL"):
        settings.ocr_base_url = s["OCR_BASE_URL"].strip().rstrip("/")
    if s.get("OCR_API_KEY"):
        settings.ocr_api_key = s["OCR_API_KEY"].strip()
    if s.get("OCR_MODEL"):
        settings.ocr_model = s["OCR_MODEL"].strip()
    if s.get("OCR_EXTRA_BODY"):
        try:
            body = json.loads(s["OCR_EXTRA_BODY"])
            if isinstance(body, dict):
                settings.ocr_extra_body = body
        except ValueError:
            pass
    if s.get("OCR_DISABLE_THINKING"):
        settings.ocr_disable_thinking = s["OCR_DISABLE_THINKING"].lower() in ("1", "true", "yes")


_apply_shell_overrides()

from app.grading.engine import GradingEngine  # noqa: E402
from app.ingestion.ocr import OCRError, ocr_pages, render_pdf_pages  # noqa: E402
from app.ingestion.parser import default_criterion, parse_rubric  # noqa: E402
from app.models import QuestionSpec, StudentAnswer  # noqa: E402
from eval.dataset import ITEMS  # noqa: E402
from eval.ocr_dataset import STYLE_LABELS, STYLES, all_entries  # noqa: E402

LANG_LABELS = {"en": "English", "hi": "Hindi", "ta": "Tamil"}
ANCHOR_RE = re.compile(r"\b(?:question|q\.?)\s*(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- metrics
def _normalize(text):
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ground_truth: str, hypothesis: str) -> float:
    """Character error rate: Levenshtein distance / ground-truth length (after normalising)."""
    gt, hy = _normalize(ground_truth), _normalize(hypothesis)
    return _levenshtein(gt, hy) / max(1, len(gt))


def wer(ground_truth: str, hypothesis: str) -> float:
    """Word error rate: word-level Levenshtein distance / ground-truth word count."""
    gt, hy = _normalize(ground_truth).split(), _normalize(hypothesis).split()
    return _levenshtein(gt, hy) / max(1, len(gt))


def label_recall(ground_truth: str, hypothesis: str) -> Optional[float]:
    """Share of "Question N" labels in the ground truth that survive OCR (None if there are none)."""
    labels = set(ANCHOR_RE.findall(ground_truth))
    if not labels:
        return None
    found = set(ANCHOR_RE.findall(hypothesis))
    return len(labels & found) / len(labels)


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {"mean": float(mean), "std": float(std), "n": len(vals)}


def _block(observations):
    ok = [o for o in observations if o["ok"]]
    return {
        "n_images": len(observations),
        "n_failures": len(observations) - len(ok),
        "cer": _agg([o["cer"] for o in ok]),
        "wer": _agg([o["wer"] for o in ok]),
        "label_recall": _agg([o["label_recall"] for o in ok]),
        "latency_s": _agg([o["latency_s"] for o in observations]),
    }


# --------------------------------------------------------------------------- provider
def _select_provider(name):
    n = (name or "").strip().lower()
    if n == "tesseract":
        settings.ocr_provider = "tesseract"
    elif n == "auto":
        settings.ocr_provider = "auto"
    else:
        # Any other label = vision-LLM backend; endpoint comes from the env.
        settings.ocr_provider = "vision"


def _tesseract_available():
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- OCR + e2e
def _ocr_entry(entry):
    with open(entry["image"], "rb") as f:
        data = f.read()
    if entry["image"].lower().endswith(".pdf"):
        return ocr_pages(render_pdf_pages(data))
    return ocr_pages([data])


def _grade(engine, item, text, samples, threshold):
    rubric_map = parse_rubric(item["rubric_text"])
    criteria = (
        rubric_map.get(item["id"])
        or rubric_map.get("1")
        or [default_criterion(item["max_marks"])]
    )
    question = QuestionSpec(
        question_id=item["id"],
        question_text=item["question"],
        max_marks=float(item["max_marks"]),
        rubric_criteria=criteria,
        model_answer=item["model_answer"],
    )
    student = StudentAnswer(question_id=item["id"], student_answer_text=text)
    representative, _samples, conf = engine.grade_with_confidence(
        question, student, n_samples=samples, threshold=threshold
    )
    return representative.total_awarded, conf.combined_confidence


def _end_to_end(results, samples):
    by_id = {it["id"]: it for it in ITEMS}
    engine = GradingEngine()
    gt_cache = {}
    rows = []
    for rec in results:
        if rec["kind"] != "item" or rec.get("lang", "en") != "en":
            continue  # translated items have no English ground-truth answer to compare
        item = by_id.get(rec.get("item_id"))
        ocr_text = next((o["text"] for o in rec["reacts"] if o["ok"]), None)
        if item is None or ocr_text is None:
            continue
        if item["id"] not in gt_cache:
            gt_cache[item["id"]] = _grade(engine, item, item["answer"], samples, settings.confidence_threshold)
        gt_marks, gt_conf = gt_cache[item["id"]]
        ocr_marks, ocr_conf = _grade(engine, item, ocr_text, samples, settings.confidence_threshold)
        rows.append(
            {
                "id": rec["id"],
                "item_id": item["id"],
                "style": rec["style"],
                "gt_marks": gt_marks,
                "ocr_marks": ocr_marks,
                "abs_diff": abs(ocr_marks - gt_marks),
                "gt_confidence": round(gt_conf, 3),
                "ocr_confidence": round(ocr_conf, 3),
            }
        )
    if not rows:
        return None
    diffs = [r["abs_diff"] for r in rows]
    margin = settings.error_margin
    return {
        "summary": {
            "n": len(rows),
            "mean_abs_mark_diff": float(statistics.fmean(diffs)),
            "share_changed_ge_margin": float(sum(1 for d in diffs if d >= margin) / len(diffs)),
            "error_margin": margin,
            "mean_confidence_ground_truth": float(statistics.fmean([r["gt_confidence"] for r in rows])),
            "mean_confidence_ocr": float(statistics.fmean([r["ocr_confidence"] for r in rows])),
            "samples_per_grading": samples,
        },
        "rows": rows,
    }


# --------------------------------------------------------------------------- reporting
def _fmt(block, key, digits=3):
    a = block[key]
    if not a["n"]:
        return "-"
    return f"{a['mean']:.{digits}f}+/-{a['std']:.{digits}f}"


def _print_table(by_style, overall):
    header = (
        f"{'style':<28}{'n':>4}{'fail':>6}  {'CER':<18}{'WER':<18}"
        f"{'label-recall':<16}{'latency(s)':<16}"
    )
    print(header)
    print("-" * len(header))
    for style, block in list(by_style.items()) + [("overall", overall)]:
        print(
            f"{style:<28}{block['n_images']:>4}{block['n_failures']:>6}  "
            f"{_fmt(block, 'cer'):<18}{_fmt(block, 'wer'):<18}"
            f"{_fmt(block, 'label_recall'):<16}{_fmt(block, 'latency_s', 2):<16}"
        )


def main() -> None:
    """Run the OCR benchmark from the command line."""
    parser = argparse.ArgumentParser(description="OCR benchmark")
    parser.add_argument("--provider", required=True, help="report label; 'tesseract' forces local OCR")
    parser.add_argument("--limit", type=int, default=None, help="max images to process")
    parser.add_argument("--repeats", type=int, default=1, help="OCR repeats per image")
    parser.add_argument("--samples", type=int, default=2, help="grading samples for the end-to-end check")
    parser.add_argument("--no-grade", action="store_true", help="skip the end-to-end grading comparison")
    parser.add_argument("--out", default=os.path.join("eval", "results"), help="output directory")
    args = parser.parse_args()

    _select_provider(args.provider)
    if settings.ocr_provider == "tesseract" and not _tesseract_available():
        print("Tesseract is not installed/available. Install it, or use another --provider.")
        sys.exit(2)

    try:
        entries = all_entries()
    except FileNotFoundError:
        print("No manifest found. Run: python -m eval.make_ocr_images")
        sys.exit(2)
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print("No images to process.")
        sys.exit(2)

    base_url = settings.ocr_base_url.split("?")[0]
    print(
        f"provider={args.provider}  backend={settings.ocr_provider}  model={settings.ocr_model}  "
        f"base_url={base_url}  key_set={bool(settings.ocr_api_key)}"
    )
    if settings.ocr_provider != "tesseract" and not settings.ocr_api_key:
        print("[warn] no OCR_API_KEY set - the vision backend will likely fail.")

    repeats = max(1, args.repeats)
    results = []
    for entry in entries:
        rec = {
            "id": entry["id"],
            "doc_id": entry.get("doc_id"),
            "item_id": entry.get("item_id"),
            "style": entry["style"],
            "kind": entry.get("kind"),
            "lang": entry.get("lang", "en"),
            "image": os.path.basename(entry["image"]),
            "reacts": [],
        }
        for _ in range(repeats):
            started = time.perf_counter()
            try:
                text, backend = _ocr_entry(entry)
                latency = time.perf_counter() - started
                rec["reacts"].append(
                    {
                        "ok": True,
                        "backend": backend,
                        "cer": cer(entry["ground_truth"], text),
                        "wer": wer(entry["ground_truth"], text),
                        "label_recall": label_recall(entry["ground_truth"], text),
                        "latency_s": latency,
                        "text": text,
                    }
                )
            except (OCRError, Exception) as exc:  # noqa: BLE001 - report, don't abort
                latency = time.perf_counter() - started
                rec["reacts"].append(
                    {"ok": False, "error": str(exc)[:300], "latency_s": latency}
                )
        ok = [o for o in rec["reacts"] if o["ok"]]
        if ok:
            rec["cer"] = statistics.fmean([o["cer"] for o in ok])
            rec["wer"] = statistics.fmean([o["wer"] for o in ok])
            rec["latency_s"] = statistics.fmean([o["latency_s"] for o in rec["reacts"]])
        results.append(rec)
        status = "ok" if ok else "FAIL"
        first = ok[0] if ok else None
        detail = f"CER={first['cer']:.3f} WER={first['wer']:.3f}" if first else rec["reacts"][0]["error"][:60]
        print(f"  [{status}] {rec['id']:<22} {detail}")

    observations = [o for rec in results for o in rec["reacts"]]
    by_style = {}
    for style in STYLES + ("pdf", "real"):
        obs = [o for rec in results if rec["style"] == style for o in rec["reacts"]]
        if obs:
            by_style[STYLE_LABELS.get(style, style)] = _block(obs)
    overall = _block(observations)
    by_language = {}
    for lang in sorted({rec["lang"] for rec in results}, key=lambda x: (x != "en", x)):
        for style in STYLES:
            obs = [o for rec in results if rec["lang"] == lang and rec["style"] == style for o in rec["reacts"]]
            if obs:
                by_language[f"{LANG_LABELS.get(lang, lang)} / {STYLE_LABELS.get(style, style)}"] = _block(obs)

    total_latency = sum(o["latency_s"] for o in observations)
    print()
    print(
        f"Images={len(results)}  OCR calls={len(observations)}  repeats={repeats}  "
        f"total OCR={total_latency:.1f}s  mean={total_latency / max(1, len(observations)):.2f}s/image"
    )
    print(
        f"Cost proxy: {len(observations)} OCR request(s) - actual cost depends on provider "
        "pricing (free tiers may be $0)."
    )
    print()
    _print_table(by_style, overall)
    print()
    _print_table(by_language, overall)

    end_to_end = None
    if not args.no_grade:
        print("\nRunning end-to-end grading comparison (ground truth vs OCR)...")
        end_to_end = _end_to_end(results, max(1, args.samples))
        if end_to_end:
            s = end_to_end["summary"]
            print(
                f"n={s['n']} items  mean|dmarks|={s['mean_abs_mark_diff']:.3f}  "
                f"changed>={s['error_margin']}: {s['share_changed_ge_margin'] * 100:.0f}%  "
                f"confidence GT={s['mean_confidence_ground_truth']:.3f} vs OCR={s['mean_confidence_ocr']:.3f}"
            )

    os.makedirs(args.out, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "ocr_config": {
            "backend": settings.ocr_provider,
            "model": settings.ocr_model,
            "base_url": base_url,
            "api_key_set": bool(settings.ocr_api_key),
            "extra_body_set": bool(settings.ocr_extra_body),
        },
        "limit": args.limit,
        "repeats": repeats,
        "n_images": len(results),
        "n_ocr_calls": len(observations),
        "total_latency_s": total_latency,
        "per_style": by_style,
        "per_language_style": by_language,
        "overall": overall,
        "end_to_end": end_to_end,
        "entries": results,
    }
    out_path = os.path.join(args.out, "ocr_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results -> {out_path}")


if __name__ == "__main__":
    main()
