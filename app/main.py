"""Confidence-Aware Auto-Assessment: the FastAPI application.

Serves the JSON API under /api and the static frontend at /.
"""
import concurrent.futures
import json
import os
import queue
import threading
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.grading.engine import GradingEngine
from app.grading.prompts import FeedbackLanguageError, normalize_language
from app.ingestion.ocr import backend_label as ocr_backend_label
from app.ingestion.ocr import provider_name as ocr_provider_name
from app.ingestion.parser import (
    NoTextError,
    ParseError,
    apply_rubric,
    extract_text_meta,
    parse_answer_sheet,
    parse_model_answers,
    parse_question_paper,
    parse_rubric,
)
from app.models import PipelineResult, QuestionResult, StudentAnswer

app = FastAPI(title="Confidence-Aware Auto-Assessment", version="0.1.0")
engine = GradingEngine()

_MAX_UPLOAD_BYTES = 4 * 1024 * 1024  # 4 MB per uploaded file (Vercel caps request bodies at ~4.5 MB)


def _warm_embeddings() -> None:
    """Load the embedding model up front so the first request doesn't pay for it."""
    try:
        from app.embeddings import embed_texts
        embed_texts(["warmup"])
    except Exception:
        pass


@app.on_event("startup")
def _startup_warmup() -> None:
    """Warm the embedding model and LLM provider in the background.

    This keeps the first grading from paying for the cold start. The embedding model
    is only loaded when the semantic weight is non-zero. No-op on serverless
    (Vercel), where background threads don't survive past the current invocation.
    """
    if settings.serverless:
        return
    if settings.uses_embeddings:
        threading.Thread(target=_warm_embeddings, daemon=True).start()
    if settings.using_mock:
        return
    threading.Thread(target=engine.llm.warmup, daemon=True).start()

_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
_SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")


def _second_examiner_active() -> bool:
    return bool(settings.judge2_api_key) and not settings.using_mock


@app.get("/api/health", response_model=None)
def health() -> Dict[str, Any]:
    """Report status, mock mode, models and confidence settings (never keys)."""
    fallback = settings.fallback_provider()
    return {
        "status": "ok",
        "mock": settings.using_mock,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "samples_per_question": settings.n_samples,
        "confidence_threshold": settings.confidence_threshold,
        "ocr_provider": settings.ocr_provider,
        "ocr_backend": ocr_provider_name(),
        "ocr_model": settings.ocr_model,
        "judge2": bool(settings.judge2_api_key),
        "judge2_model": settings.judge2_model if settings.judge2_api_key else None,
        "second_examiner_active": _second_examiner_active(),
        "adjudicate": settings.adjudicate,
        "feedback_language": settings.feedback_language,
        "referee_model": settings.referee_provider().model,
        "fallback_model": fallback.model if fallback else None,
        "weights": {
            "agreement": settings.w_agree,
            "variance": settings.w_variance,
            "semantic": settings.w_semantic,
            "cross": settings.w_cross,
        },
    }


def _sample_files(folder: str = "", answer_sheet: str = "answer_sheet.txt") -> Dict[str, str]:
    """File names (relative to sample_data/) for one sample set."""
    names = {
        "question_paper_text": "question_paper.txt",
        "answer_sheet_text": answer_sheet,
        "rubric_text": "rubric.txt",
        "model_answers_text": "model_answers.txt",
    }
    return {key: f"{folder}/{name}" if folder else name for key, name in names.items()}


# Bundled sample sets. The English ones share the paper, rubric and model answers and
# differ in the answer sheet; each language has its own folder.
SAMPLES = {
    "good": ("English (good)", _sample_files(answer_sheet="answer_sheet_good.txt")),
    "weak": ("English (weak)", _sample_files(answer_sheet="answer_sheet_poor.txt")),
    "adversarial": ("Adversarial (prompt injection)", _sample_files(answer_sheet="answer_sheet_adversarial.txt")),
    "hindi": ("Hindi", _sample_files("hindi")),
    "tamil": ("Tamil", _sample_files("tamil")),
}


@app.get("/api/samples", response_model=None)
def samples() -> List[Dict[str, str]]:
    """List the bundled sample sets."""
    return [{"name": k, "label": v[0]} for k, v in SAMPLES.items()]


@app.get("/api/sample", response_model=None)
def sample(name: str = "good") -> Dict[str, str]:
    """Return one bundled sample set (used by the sample picker).

    Args:
        name: Sample set name, one of SAMPLES.

    Returns:
        The set's name, label and the text of each input.

    Raises:
        HTTPException: 404 for an unknown name.
    """
    if name not in SAMPLES:
        raise HTTPException(404, f"Unknown sample '{name}'. Choose one of: {', '.join(SAMPLES)}.")
    label, files = SAMPLES[name]
    out = {"name": name, "label": label}
    for key, fname in files.items():
        try:
            with open(os.path.join(_SAMPLE_DIR, fname), "r", encoding="utf-8") as f:
                out[key] = f.read()
        except OSError:
            out[key] = ""
    return out


def _upload_bytes(upload: UploadFile) -> bytes:
    data = upload.file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        mb = len(data) / (1024 * 1024)
        raise HTTPException(
            413,
            f"'{upload.filename}' is {mb:.1f} MB; the maximum is "
            f"{_MAX_UPLOAD_BYTES // (1024 * 1024)} MB per file.",
        )
    return data


@app.post("/api/extract", response_model=None)
def extract(
    question_paper: Optional[UploadFile] = File(None),
    answer_sheet: Optional[UploadFile] = File(None),
    rubric: Optional[UploadFile] = File(None),
    model_answers: Optional[UploadFile] = File(None),
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Extract the text of uploaded files without grading anything.

    Text and text-layer PDFs are read directly; images and scans go through OCR. The
    text is returned so it can be checked and corrected before grading. A file that
    cannot be read gives an `error` for that input instead of failing the request.

    Returns:
        {"inputs": {key: {filename, text, ocr, warning, error}}} for each uploaded
        input.
    """
    uploads = {
        "question_paper": question_paper,
        "answer_sheet": answer_sheet,
        "rubric": rubric,
        "model_answers": model_answers,
    }
    out = {}
    for key, upload in uploads.items():
        if upload is None or not upload.filename:
            continue
        entry = {"filename": upload.filename, "text": "", "ocr": None, "warning": None, "error": None}
        try:
            text, ocr = extract_text_meta(upload.filename, _upload_bytes(upload))
            entry.update(text=text, ocr=ocr)
            if ocr:
                entry["warning"] = (f"Read by OCR ({ocr_backend_label(ocr)}). "
                                    "Check the text for mistakes before marking.")
        except NoTextError as exc:
            entry["warning"] = str(exc)
        except ParseError as exc:
            entry["error"] = str(exc)
        except HTTPException as exc:
            entry["error"] = exc.detail
        out[key] = entry
    return {"inputs": out}


def _read(upload: Optional[UploadFile], text: Optional[str], warnings: List[str], label: str,
          required: bool = False):
    """Return (content, source_name) from either an uploaded file or raw text.

    Unreadable or unsupported files are a 400 for the required inputs (question
    paper, answer sheet) and a warning for optional ones (rubric, model answers),
    which are then ignored. A readable file with no text yields "" plus a warning,
    except for the question paper, which must contain questions.
    """
    if upload is not None and upload.filename:
        data = _upload_bytes(upload)
        try:
            content, ocr = extract_text_meta(upload.filename, data)
        except NoTextError as exc:
            if label == "question paper":
                raise HTTPException(400, f"{exc} The question paper must contain the questions.")
            warnings.append(f"{exc} The {label} was treated as empty.")
            return "", f"file:{upload.filename}"
        except ParseError as exc:
            if required:
                raise HTTPException(400, str(exc))
            warnings.append(f"{exc} The {label} was ignored.")
            return None, None
        if ocr:
            warnings.append(f"'{upload.filename}' was read via OCR ({ocr_backend_label(ocr)}); "
                            "check the extracted text below.")
        return content, f"file:{upload.filename}"
    return text, ("text" if text else None)


def _prepare(question_paper, answer_sheet, rubric, model_answers, question_paper_text,
             answer_sheet_text, rubric_text, model_answers_text, structured_json):
    """Ingest and parse the inputs.

    Returns:
        (questions, answer_map, warnings).

    Raises:
        HTTPException: 400 with a clear message when there is nothing to grade.
    """
    warnings: List[str] = []
    if settings.using_mock:
        warnings.append(
            "No LLM API key configured - using MOCK grading. Set OPENAI_API_KEY (and "
            "OPENAI_BASE_URL if needed) in .env for real grading."
        )

    q_text, _ = _read(question_paper, question_paper_text or structured_json, warnings,
                      "question paper", required=True)
    a_text, a_src = _read(answer_sheet, answer_sheet_text, warnings, "answer sheet", required=True)
    r_text, _ = _read(rubric, rubric_text, warnings, "rubric")
    m_text, _ = _read(model_answers, model_answers_text, warnings, "model answers")

    if not q_text or not q_text.strip():
        raise HTTPException(400, "A question paper is required (file or pasted text).")
    try:
        questions = parse_question_paper(q_text, warnings)
    except ParseError as exc:
        raise HTTPException(400, str(exc))
    if not questions:
        raise HTTPException(400, "No questions could be parsed from the question paper.")

    rubric_map = parse_rubric(r_text, warnings) if r_text else {}
    model_map = parse_model_answers(m_text, warnings) if m_text else {}
    question_ids = [q.question_id for q in questions]
    for name, mapping in (("rubric", rubric_map), ("model answers", model_map)):
        extra = [qid for qid in mapping if qid not in question_ids]
        if extra and len(mapping) > 1:
            warnings.append(
                f"The {name} include {', '.join('Q' + x for x in extra)}, which are not "
                "in the question paper; they were ignored."
            )
    for q in questions:
        # A structured-JSON paper may carry its own rubric; an uploaded rubric wins.
        warnings.extend(apply_rubric(q, rubric_map.get(q.question_id) or q.rubric_criteria))
        q.model_answer = model_map.get(q.question_id, q.model_answer or "")

    if not a_text or not a_text.strip():
        if a_src is None:
            warnings.append("No answer sheet was provided; every question is graded as blank.")
        a_text = ""
    answer_map = None
    if a_text:
        answer_map, sheet_warnings = parse_answer_sheet(a_text, question_ids)
        warnings.extend(sheet_warnings)
    if answer_map is None:
        answer_map = {
            question_ids[0]: StudentAnswer(question_id=question_ids[0], student_answer_text=a_text or "")
        }
        for qid in question_ids:
            answer_map.setdefault(qid, StudentAnswer(question_id=qid, student_answer_text=""))
    return questions, answer_map, warnings


def _grade_all(questions, answer_map, warnings, n, thr, emit=None, feedback_language=None) -> PipelineResult:
    """Grade every question in parallel and aggregate the results.

    `emit`, if given, receives progress events:
    {"type": "stage", "stage": ..., "question": id}. `feedback_language` is passed
    to the engine (None: FEEDBACK_LANGUAGE).
    """

    def _grade_one(q):
        student = answer_map.get(q.question_id, StudentAnswer(question_id=q.question_id, student_answer_text=""))
        warns = []
        if not student.student_answer_text.strip():
            warns.append(f"Q{q.question_id}: no student answer found for this question.")
        progress = None
        if emit is not None:
            def progress(stage, qid=q.question_id):
                emit({"type": "stage", "stage": stage, "question": qid})
        try:
            representative, _samples, confidence = engine.grade_with_confidence(
                q, student, n_samples=n, threshold=thr, progress=progress,
                feedback_language=feedback_language,
            )
        except Exception as exc:
            raise HTTPException(502, f"Grading failed for Q{q.question_id}: {exc}")
        if emit is not None:
            emit({"type": "stage", "stage": "done", "question": q.question_id,
                  "llm": confidence.grader_role != "none"})
        warns.extend(confidence.warnings)
        result = QuestionResult(
            grading=representative,
            confidence=confidence,
            sampled_totals=[round(r.total_awarded, 1) for r in _samples],
            question_text=q.question_text,
            student_answer_text=student.student_answer_text,
        )
        return result, warns

    workers = int(os.getenv("GRADE_WORKERS", "4"))
    results: List[QuestionResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result, warns in pool.map(_grade_one, questions):
            warnings.extend(warns)
            results.append(result)

    total_awarded = sum(r.grading.total_awarded for r in results)
    total_max = sum(r.grading.max_marks for r in results)
    conf = [r.confidence.combined_confidence for r in results]
    overall = sum(conf) / len(conf) if conf else 0.0
    flagged = [r.grading.question_id for r in results if r.confidence.flagged]

    return PipelineResult(
        results=results,
        total_awarded=round(total_awarded, 1),
        total_max=round(total_max, 1),
        overall_confidence=round(overall, 3),
        flagged=flagged,
        warnings=warnings,
        using_mock=settings.using_mock,
    )


def _stream(questions, answer_map, warnings, n, thr, feedback_language=None):
    """Return an NDJSON stream of real progress events, then the result (or an error)."""
    events: "queue.Queue" = queue.Queue()

    def worker():
        try:
            result = _grade_all(questions, answer_map, warnings, n, thr, emit=events.put,
                                feedback_language=feedback_language)
            events.put({"type": "result", "result": jsonable_encoder(result)})
        except HTTPException as exc:
            events.put({"type": "error", "status": exc.status_code, "detail": exc.detail})
        except Exception as exc:  # noqa: BLE001 - reported to the client, never swallowed
            events.put({"type": "error", "status": 500, "detail": str(exc)})
        finally:
            events.put(None)

    def gen():
        yield json.dumps({
            "type": "start",
            "questions": [q.question_id for q in questions],
            "samples": n,
            "second_examiner": _second_examiner_active(),
            "moderation": settings.adjudicate and _second_examiner_active(),
        }) + "\n"
        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/grade", response_model=None)
def grade(
    question_paper: Optional[UploadFile] = File(None),
    answer_sheet: Optional[UploadFile] = File(None),
    rubric: Optional[UploadFile] = File(None),
    model_answers: Optional[UploadFile] = File(None),
    question_paper_text: Optional[str] = Form(None),
    answer_sheet_text: Optional[str] = Form(None),
    rubric_text: Optional[str] = Form(None),
    model_answers_text: Optional[str] = Form(None),
    structured_json: Optional[str] = Form(None),
    n_samples: Optional[int] = Form(None),
    threshold: Optional[float] = Form(None),
    stream: Optional[bool] = Form(False),
    feedback_language: Optional[str] = Form(None),
) -> Union[PipelineResult, StreamingResponse]:
    """Grade an answer sheet.

    Inputs are files or the matching `*_text` form fields. `n_samples` is clamped to
    1-10 and `threshold` to 0-1. `feedback_language` is auto (the student's answer
    language), a language code such as `en` or `hi`, or a name such as `Tamil`
    (default: FEEDBACK_LANGUAGE).

    Returns:
        A PipelineResult, or with `stream=true` an NDJSON stream: a `start` event,
        `stage` events as each question moves through grading, second examiner and
        moderation, then a `result` (or `error`) event.

    Raises:
        HTTPException: 4xx for input errors (in both modes), 502 when grading fails
            (non-streaming).
    """
    if feedback_language is not None:
        try:
            feedback_language = normalize_language(feedback_language)
        except FeedbackLanguageError as exc:
            raise HTTPException(400, str(exc))
    questions, answer_map, warnings = _prepare(
        question_paper, answer_sheet, rubric, model_answers, question_paper_text,
        answer_sheet_text, rubric_text, model_answers_text, structured_json,
    )
    n = settings.n_samples if n_samples is None else max(1, min(10, n_samples))
    thr = settings.confidence_threshold if threshold is None else max(0.0, min(1.0, threshold))
    if stream:
        return _stream(questions, answer_map, warnings, n, thr, feedback_language)
    return _grade_all(questions, answer_map, warnings, n, thr, feedback_language=feedback_language)


# API routes first, then serve the frontend at /
app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="static")
