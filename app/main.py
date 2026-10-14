"""Confidence-Aware Auto-Assessment - FastAPI application."""
import concurrent.futures
import os
import threading
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.grading.engine import GradingEngine
from app.ingestion.parser import (
    ParseError,
    default_criterion,
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


def _warm_embeddings():
    """Load the embedding model up front so the first request doesn't pay for it."""
    try:
        from app.embeddings import embed_texts
        embed_texts(["warmup"])
    except Exception:
        pass


@app.on_event("startup")
def _startup_warmup():
    """Warm the embedding model and LLM provider in the background so the first
    user grading isn't hit with the cold start. No-op on serverless (Vercel), where
    background threads don't survive past the current invocation."""
    if settings.serverless:
        return
    threading.Thread(target=_warm_embeddings, daemon=True).start()
    if settings.using_mock:
        return
    threading.Thread(target=engine.llm.warmup, daemon=True).start()

_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
_SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mock": settings.using_mock,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "samples_per_question": settings.n_samples,
        "confidence_threshold": settings.confidence_threshold,
        "ocr_provider": settings.ocr_provider,
        "ocr_model": settings.ocr_model,
        "judge2": bool(settings.judge2_api_key),
        "judge2_model": settings.judge2_model if settings.judge2_api_key else None,
        "adjudicate": settings.adjudicate,
    }


@app.get("/api/sample")
def sample():
    """Return the bundled sample inputs (used by the 'Try with sample data' button)."""
    files = {
        "question_paper_text": "question_paper.txt",
        "answer_sheet_text": "answer_sheet_good.txt",
        "rubric_text": "rubric.txt",
        "model_answers_text": "model_answers.txt",
    }
    out = {}
    for key, fname in files.items():
        try:
            with open(os.path.join(_SAMPLE_DIR, fname), "r", encoding="utf-8") as f:
                out[key] = f.read()
        except OSError:
            out[key] = ""
    return out


def _read(upload: Optional[UploadFile], text: Optional[str], warnings: Optional[List[str]] = None):
    """Return (content, source_name) from either an uploaded file or raw text."""
    if upload is not None and upload.filename:
        data = upload.file.read()
        if len(data) > _MAX_UPLOAD_BYTES:
            mb = len(data) / (1024 * 1024)
            raise HTTPException(
                413,
                f"'{upload.filename}' is {mb:.1f} MB; the maximum is "
                f"{_MAX_UPLOAD_BYTES // (1024 * 1024)} MB per file.",
            )
        try:
            content, ocr = extract_text_meta(upload.filename, data)
        except ParseError as exc:
            raise HTTPException(400, str(exc))
        if ocr and warnings is not None:
            warnings.append(f"'{upload.filename}' was read via OCR ({ocr}); check the extracted text below.")
        return content, f"file:{upload.filename}"
    return text, ("text" if text else None)


@app.post("/api/grade")
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
):
    warnings: List[str] = []
    if settings.using_mock:
        warnings.append(
            "No LLM API key configured - using MOCK grading. Set OPENAI_API_KEY (and "
            "OPENAI_BASE_URL if needed) in .env for real grading."
        )

    # 1. Ingestion
    q_text, q_src = _read(question_paper, question_paper_text or structured_json, warnings)
    a_text, _ = _read(answer_sheet, answer_sheet_text, warnings)
    r_text, _ = _read(rubric, rubric_text, warnings)
    m_text, _ = _read(model_answers, model_answers_text, warnings)

    if not q_text:
        raise HTTPException(400, "A question paper is required (file or pasted text).")
    try:
        questions = parse_question_paper(q_text)
    except ParseError as exc:
        raise HTTPException(400, str(exc))
    if not questions:
        raise HTTPException(400, "No questions could be parsed from the question paper.")

    rubric_map = parse_rubric(r_text) if r_text else {}
    model_map = parse_model_answers(m_text) if m_text else {}
    for q in questions:
        q.rubric_criteria = rubric_map.get(q.question_id) or [default_criterion(q.max_marks)]
        q.model_answer = model_map.get(q.question_id, q.model_answer or "")

    question_ids = [q.question_id for q in questions]
    answer_map = None
    if a_text:
        answer_map, warn = parse_answer_sheet(a_text, question_ids)
        if warn:
            warnings.append(warn)
    if answer_map is None:
        answer_map = {
            question_ids[0]: StudentAnswer(question_id=question_ids[0], student_answer_text=a_text or "")
        }
        for qid in question_ids:
            answer_map.setdefault(qid, StudentAnswer(question_id=qid, student_answer_text=""))

    # 2 & 3. Grade each question with confidence sampling
    n = settings.n_samples if n_samples is None else max(1, min(10, n_samples))
    thr = settings.confidence_threshold if threshold is None else max(0.0, min(1.0, threshold))

    def _grade_one(q):
        student = answer_map.get(q.question_id, StudentAnswer(question_id=q.question_id, student_answer_text=""))
        warns = []
        if not student.student_answer_text.strip():
            warns.append(f"Q{q.question_id}: no student answer found for this question.")
        try:
            representative, _samples, confidence = engine.grade_with_confidence(
                q, student, n_samples=n, threshold=thr
            )
        except Exception as exc:
            raise HTTPException(502, f"Grading failed for Q{q.question_id}: {exc}")
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

    # 4. Output aggregation
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


# API routes first, then serve the frontend at /
app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="static")