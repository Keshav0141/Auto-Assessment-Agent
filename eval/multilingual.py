"""Multilingual eval: does the mark change when the same answer is in another language?

Eight items from eval/dataset.py are graded with the student's answer in English and
in Hindi, Tamil and Telugu. The question, rubric, model answer and ground truth stay
the same (English), so a language-blind grader should give the same mark in every
language. Reported per language:

  * MAE against the ground truth;
  * cross-language consistency: mean |mark_lang - mark_en|;
  * mean confidence and the number of flagged items.

The English answers are graded twice; mean |mark_en_rerun - mark_en| is the noise
floor from sampling alone, to read the cross-language numbers against.

The Hindi, Tamil and Telugu answers are MACHINE-TRANSLATED from the English answers
in eval/dataset.py and have not been reviewed by native speakers. Formulas, numbers
and units are kept as written.

Usage:
    python -m eval.multilingual               # uses the configured LLMs
    python -m eval.multilingual --samples 3
"""
import argparse
import concurrent.futures
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional

from app.config import settings
from app.grading.engine import GradingEngine
from app.models import StudentAnswer
from eval.dataset import ITEMS
from eval.harness import question_for

ITEM_IDS = ("P2", "P5", "M1", "M4", "E1", "E2", "C2", "P8")
LANGUAGES = {"en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu"}
RERUN = "en_rerun"

# Machine-translated from the English answers in eval/dataset.py (not reviewed by a
# native speaker). Same meaning, same mistakes: P5 still omits the distance, M4 still
# has the wrong roots, P8 still gives the wrong unit.
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "P2": {
        "hi": "प्रत्येक क्रिया की एक समान और विपरीत प्रतिक्रिया होती है। उदाहरण के लिए, जब आप दीवार को "
              "धक्का देते हैं, तो दीवार भी आपको पीछे की ओर धक्का देती है।",
        "ta": "ஒவ்வொரு வினைக்கும் சமமான மற்றும் எதிரான எதிர்வினை உண்டு. உதாரணமாக, நீங்கள் ஒரு சுவரைத் "
              "தள்ளும்போது, சுவரும் உங்களைப் பின்னோக்கித் தள்ளுகிறது.",
        "te": "ప్రతి చర్యకు సమానమైన మరియు వ్యతిరేకమైన ప్రతిచర్య ఉంటుంది. ఉదాహరణకు, మీరు గోడను "
              "నెట్టినప్పుడు, గోడ కూడా మిమ్మల్ని వెనక్కి నెడుతుంది.",
    },
    "P5": {
        "hi": "त्वरण a = F/m = 20/5 = 4 m/s^2 है।",
        "ta": "முடுக்கம் a = F/m = 20/5 = 4 m/s^2 ஆகும்.",
        "te": "త్వరణం a = F/m = 20/5 = 4 m/s^2 అవుతుంది.",
    },
    "M1": {
        "hi": "x^2 का अवकलज 2x है, और x का अवकलज 1 है। इसलिए f'(x) = 6x + 2।",
        "ta": "x^2 இன் வகைக்கெழு 2x, மேலும் x இன் வகைக்கெழு 1. எனவே f'(x) = 6x + 2.",
        "te": "x^2 యొక్క అవకలజం 2x, మరియు x యొక్క అవకలజం 1. కాబట్టి f'(x) = 6x + 2.",
    },
    "M4": {
        "hi": "x = 6 और x = -1, क्योंकि 6 और -1 का गुणनफल -6 है और उनका योग 5 है।",
        "ta": "x = 6 மற்றும் x = -1, ஏனெனில் 6 மற்றும் -1 இன் பெருக்கற்பலன் -6, அவற்றின் கூட்டுத்தொகை 5.",
        "te": "x = 6 మరియు x = -1, ఎందుకంటే 6 మరియు -1 ల గుణకారం -6, వాటి మొత్తం 5.",
    },
    "E1": {
        "hi": "प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा पौधे सूर्य के प्रकाश को रासायनिक ऊर्जा में बदलते "
              "हैं, और पानी तथा कार्बन डाइऑक्साइड से ग्लूकोज बनाकर ऑक्सीजन छोड़ते हैं।",
        "ta": "ஒளிச்சேர்க்கை என்பது தாவரங்கள் சூரிய ஒளியை வேதி ஆற்றலாக மாற்றும் செயல்முறை ஆகும்; அவை நீர் "
              "மற்றும் கார்பன் டை ஆக்சைடைப் பயன்படுத்தி குளுக்கோஸை உருவாக்கி ஆக்சிஜனை வெளியிடுகின்றன.",
        "te": "కిరణజన్య సంయోగక్రియ అనేది మొక్కలు సూర్యరశ్మిని రసాయన శక్తిగా మార్చే ప్రక్రియ; అవి నీరు "
              "మరియు కార్బన్ డయాక్సైడ్‌ను ఉపయోగించి గ్లూకోజ్‌ను తయారు చేసి ఆక్సిజన్‌ను విడుదల చేస్తాయి.",
    },
    "E2": {
        "hi": "यह वह है जो पौधे भोजन बनाने के लिए करते हैं।",
        "ta": "தாவரங்கள் உணவு தயாரிக்கச் செய்வது இதுதான்.",
        "te": "మొక్కలు ఆహారం తయారు చేసుకోవడానికి చేసేది ఇదే.",
    },
    "C2": {
        "hi": "ऐरे तत्वों को लगातार मेमोरी में रखते हैं और O(1) रैंडम एक्सेस देते हैं, लेकिन उनका आकार "
              "निश्चित होता है। लिंक्ड लिस्ट पॉइंटर वाले नोड्स का उपयोग करती है, O(1) में इंसर्शन/डिलीशन "
              "देती है, लेकिन एक्सेस O(n) होता है।",
        "ta": "அணிகள் உறுப்புகளைத் தொடர்ச்சியான நினைவகத்தில் சேமிக்கின்றன, O(1) சீரற்ற அணுகலை "
              "அனுமதிக்கின்றன, ஆனால் அவற்றின் அளவு நிலையானது. இணைக்கப்பட்ட பட்டியல்கள் சுட்டிகளுடன் கூடிய "
              "கணுக்களைப் பயன்படுத்துகின்றன, O(1) செருகல்/நீக்கலை அனுமதிக்கின்றன, ஆனால் அணுகல் O(n).",
        "te": "అరేలు మూలకాలను వరుస మెమరీలో నిల్వ చేస్తాయి, O(1) యాదృచ్ఛిక యాక్సెస్‌ను అనుమతిస్తాయి, కానీ "
              "వాటి పరిమాణం స్థిరంగా ఉంటుంది. లింక్డ్ లిస్ట్‌లు పాయింటర్లు ఉన్న నోడ్‌లను ఉపయోగిస్తాయి, O(1) "
              "చొప్పింపు/తొలగింపును అనుమతిస్తాయి, కానీ యాక్సెస్ O(n).",
    },
    "P8": {
        "hi": "चाल = दूरी / समय = 100 / 2 = 50 m/s।",
        "ta": "வேகம் = தூரம் / நேரம் = 100 / 2 = 50 m/s.",
        "te": "వేగం = దూరం / సమయం = 100 / 2 = 50 m/s.",
    },
}


def cases() -> List[Dict[str, Any]]:
    """Return one grading case per (item, language), plus the English rerun.

    Returns:
        Dicts with item, lang and answer.
    """
    items = {it["id"]: it for it in ITEMS}
    out = []
    for item_id in ITEM_IDS:
        item = items[item_id]
        for lang in list(LANGUAGES) + [RERUN]:
            answer = item["answer"] if lang in ("en", RERUN) else TRANSLATIONS[item_id][lang]
            out.append({"item": item, "lang": lang, "answer": answer})
    return out


def _grade(engine: GradingEngine, case: Dict[str, Any], n_samples: Optional[int]) -> Dict[str, Any]:
    item = case["item"]
    row = {"id": item["id"], "lang": case["lang"], "ground_truth": float(item["ground_truth"]),
           "max": float(item["max_marks"]), "mark": None, "error": None}
    started = time.time()
    try:
        student = StudentAnswer(question_id=item["id"], student_answer_text=case["answer"])
        rep, samples, conf = engine.grade_with_confidence(question_for(item), student, n_samples=n_samples)
        row.update(
            mark=rep.total_awarded,
            sampled_totals=[s.total_awarded for s in samples],
            judge2_total=conf.judge2_total,
            adjudicated=conf.adjudicated,
            confidence=round(conf.combined_confidence, 3),
            flagged=conf.flagged,
            integrity_flags=conf.integrity_flags,
            answer_language=conf.answer_language,
            feedback_language=conf.feedback_language,
            feedback=rep.feedback,
            graded_by=conf.graded_by,
        )
    except Exception as exc:  # noqa: BLE001 - one failed case must not stop the run
        row["error"] = str(exc)
    row["seconds"] = round(time.time() - started, 1)
    return row


def _mean(values: List[float]) -> Optional[float]:
    return round(statistics.fmean(values), 3) if values else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-language metrics.

    Args:
        rows: Graded rows from `evaluate`.

    Returns:
        {lang: {n, n_errors, mae, mean_abs_diff_vs_en, max_abs_diff_vs_en,
        mean_confidence, n_flagged}} for each language and the English rerun.
    """
    english = {r["id"]: r["mark"] for r in rows if r["lang"] == "en" and r["mark"] is not None}
    out = {}
    for lang in list(LANGUAGES) + [RERUN]:
        done = [r for r in rows if r["lang"] == lang and r["mark"] is not None]
        diffs = [abs(r["mark"] - english[r["id"]]) for r in done if r["id"] in english] if lang != "en" else []
        out[lang] = {
            "n": len(done),
            "n_errors": sum(1 for r in rows if r["lang"] == lang and r["mark"] is None),
            "mae": _mean([abs(r["mark"] - r["ground_truth"]) for r in done]),
            "mean_abs_diff_vs_en": _mean(diffs),
            "max_abs_diff_vs_en": max(diffs) if diffs else None,
            "mean_confidence": _mean([r["confidence"] for r in done]),
            "n_flagged": sum(1 for r in done if r["flagged"]),
        }
    return out


def evaluate(engine: GradingEngine, n_samples: Optional[int] = None, workers: int = 4) -> Dict[str, Any]:
    """Grade every case and summarise.

    Args:
        engine: The grading engine.
        n_samples: Gradings per answer (default: CONFIDENCE_SAMPLES).
        workers: Cases graded in parallel.

    Returns:
        {"summary": ..., "rows": [...]} with rows ordered by item, then language.
    """
    all_cases = cases()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(lambda c: _grade(engine, c, n_samples), all_cases))
    return {"summary": summarize(rows), "rows": rows}


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def main() -> None:
    """Run the multilingual eval, print the table and write eval/results/multilingual.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=None, help="grading samples per answer")
    parser.add_argument("--workers", type=int, default=int(os.getenv("GRADE_WORKERS", "4")))
    parser.add_argument("--out", default=os.path.join("eval", "results"))
    args = parser.parse_args()
    n = args.samples or settings.n_samples
    if settings.using_mock:
        print("WARNING: mock mode (no API key); the marks below are a keyword heuristic.")
    print(f"Multilingual eval: {len(ITEM_IDS)} items x {len(LANGUAGES)} languages + English rerun, "
          f"{n} samples each (model={settings.llm_model})...")
    started = time.time()
    result = evaluate(GradingEngine(), n_samples=n, workers=args.workers)
    judge2 = settings.judge2_provider()
    report = {
        "config": {
            "mock": settings.using_mock,
            "primary_model": settings.llm_model,
            "second_grader": judge2.model if judge2 else None,
            "referee_model": settings.referee_provider().model,
            "samples": n,
            "feedback_language": settings.feedback_language,
            "items": list(ITEM_IDS),
            "translations": "machine-translated from the English answers, not reviewed by native speakers",
        },
        "seconds": round(time.time() - started, 1),
        **result,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "multilingual.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'language':<10} {'n':>3} {'MAE':>6} {'|Δ en|':>7} {'max|Δ|':>7} {'conf':>6} {'flagged':>8}")
    for lang, s in result["summary"].items():
        name = LANGUAGES.get(lang, "English (rerun)")
        print(f"{name:<10} {s['n']:>3} {_fmt(s['mae']):>6} {_fmt(s['mean_abs_diff_vs_en']):>7} "
              f"{_fmt(s['max_abs_diff_vs_en']):>7} {_fmt(s['mean_confidence']):>6} {s['n_flagged']:>8}")
    print("\nPer item (mark: en / hi / ta / te / en rerun; ground truth):")
    marks = {(r["id"], r["lang"]): r["mark"] for r in result["rows"]}
    gts = {r["id"]: r["ground_truth"] for r in result["rows"]}
    for item_id in ITEM_IDS:
        cells = " / ".join(_fmt(marks.get((item_id, lang))) for lang in list(LANGUAGES) + [RERUN])
        print(f"  {item_id}: {cells}  (gt {gts[item_id]:g})")
    errors = [r for r in result["rows"] if r["error"]]
    for r in errors:
        print(f"  ERROR {r['id']} {r['lang']}: {r['error'][:200]}")
    print(f"\nWrote {path} ({report['seconds']} s)")


if __name__ == "__main__":
    main()
