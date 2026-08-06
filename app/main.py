"""Confidence-Aware Auto-Assessment - FastAPI application."""
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
    extract_text,
    parse_answer_sheet,
    parse_model_answers,
    parse_question_paper,
    parse_rubric,
)
from app.models import PipelineResult, QuestionResult, StudentAnswer

app = FastAPI(title="Confidence-Aware Auto-Assessment", version="0.1.0")
engine = GradingEngine()


@app.on_event("startup")
def _startup_warmup():
    """Warm the LLM provider in the background so the first user grading isn't
    hit with the free-tier cold start."""
    if settings.using_mock:
        return
    threading.Thread(target=engine.llm.warmup, daemon=True).start()

_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mock": settings.using_mock,
        "model": settings.llm_model,
        "samples_per_question": settings.n_samples,
        "confidence_threshold": settings.confidence_threshold,
    }


def _read(upload: Optional[UploadFile], text: Optional[str]):
    """Return (content, source_name) from either an uploaded file or raw text."""
    if upload is not None and upload.filename:
        data = upload.file.read()
        return extract_text(upload.filename, data), f"file:{upload.filename}"
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
    q_text, q_src = _read(question_paper, question_paper_text or structured_json)
    a_text, _ = _read(answer_sheet, answer_sheet_text)
    r_text, _ = _read(rubric, rubric_text)
    m_text, _ = _read(model_answers, model_answers_text)

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
    results: List[QuestionResult] = []
    for q in questions:
        student = answer_map.get(q.question_id, StudentAnswer(question_id=q.question_id, student_answer_text=""))
        if not student.student_answer_text.strip():
            warnings.append(f"Q{q.question_id}: no student answer found for this question.")
        try:
            representative, _samples, confidence = engine.grade_with_confidence(q, student)
        except Exception as exc:
            raise HTTPException(502, f"Grading failed for Q{q.question_id}: {exc}")
        results.append(
            QuestionResult(
                grading=representative,
                confidence=confidence,
                sampled_totals=[round(r.total_awarded, 1) for r in _samples],
                question_text=q.question_text,
                student_answer_text=student.student_answer_text,
            )
        )

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