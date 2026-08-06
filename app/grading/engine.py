"""Grading engine: LLM-based per-question grading with JSON validation and a
deterministic mock fallback for API-key-free testing."""
import random
import re
import time
from collections import Counter
from typing import List, Tuple

from app.config import settings
from app.ingestion.parser import default_criterion
from app.llm import LLMClient
from app.models import (
    CriterionScore,
    GradingResult,
    QuestionSpec,
    StudentAnswer,
)

GRADE_SYSTEM = (
    "You are an expert, strict, fair examiner. Grade the student's answer strictly "
    "against the rubric criteria and the model answer. Award marks only where a "
    "criterion is genuinely met. Give feedback that cites exactly which criterion "
    "was or was not met and how the student can improve. Return JSON only, no prose."
)

GRADE_SCHEMA_HINT = (
    'Return ONLY JSON with this exact schema: {"criterion_scores": '
    '[{"criterion": "<rubric criterion text>", "awarded": <number>, "max": <number>, '
    '"justification": "<why this mark / no mark>"}], "total_awarded": <number>, '
    '"feedback": "<2-4 sentence feedback citing specific criteria met/missed and how to improve>"}'
)


def build_grade_prompt(q: QuestionSpec, student: StudentAnswer) -> str:
    if q.rubric_criteria:
        rubric_lines = "\n".join(
            f"- {c.criterion} [{c.marks} marks]" for c in q.rubric_criteria
        )
    else:
        rubric_lines = f"- Overall answer quality [{q.max_marks} marks]"
    model_answer = q.model_answer or "Not provided - grade on rubric and correctness."
    return f"""QUESTION: {q.question_text}
MAX MARKS: {q.max_marks}
RUBRIC CRITERIA:
{rubric_lines}
MODEL ANSWER:
{model_answer}
STUDENT ANSWER:
{student.student_answer_text}

{GRADE_SCHEMA_HINT}"""


def _num(value) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


class GradingEngine:
    def __init__(self):
        self.llm = LLMClient()
        self.mock = settings.using_mock
        self._last_call = 0.0

    def _maybe_warmup(self):
        """Ping the provider if the last LLM call was a while ago (free-tier cold
        starts make the first call after idle very slow)."""
        if self.mock:
            return
        now = time.time()
        if now - self._last_call > settings.idle_warmup_seconds:
            self.llm.warmup()
        self._last_call = now

    def grade_once(self, q: QuestionSpec, student: StudentAnswer, temperature=None, jitter=0.0):
        if self.mock:
            return self._mock_grade(q, student, jitter)
        raw = self.llm.chat_json(GRADE_SYSTEM, build_grade_prompt(q, student), temperature=temperature)
        return self._validate(q, raw)

    def grade_with_confidence(
        self, q: QuestionSpec, student: StudentAnswer
    ) -> Tuple[GradingResult, List[GradingResult], object]:
        """Sample N gradings and return (representative, all_samples, confidence)."""
        from app.confidence.scorer import compute_confidence

        self._maybe_warmup()
        results = []
        for i in range(settings.n_samples):
            results.append(self.grade_once(q, student, jitter=i))
            # Delay between samples to respect free-tier rate limits
            if not self.mock and settings.request_pacing and i < settings.n_samples - 1:
                time.sleep(settings.request_pacing)
        totals = [r.total_awarded for r in results]
        representative_total = Counter(totals).most_common(1)[0][0]
        representative = next(r for r in results if r.total_awarded == representative_total)
        confidence = compute_confidence(results)
        return representative, results, confidence

    # -- JSON validation -----------------------------------------------------

    def _validate(self, q: QuestionSpec, raw: dict) -> GradingResult:
        raw = raw or {}
        scores = []
        for item in raw.get("criterion_scores", []) or []:
            max_ = _num(item.get("max"))
            if max_ <= 0:
                max_ = q.max_marks
            awarded = min(max(_num(item.get("awarded")), 0.0), max_)
            scores.append(
                CriterionScore(
                    criterion=str(item.get("criterion", "") or "Overall"),
                    awarded=awarded,
                    max=max_,
                    justification=str(item.get("justification", "")),
                )
            )
        if not scores:
            awarded = min(max(_num(raw.get("total_awarded")), 0.0), q.max_marks)
            scores = [
                CriterionScore(
                    criterion=f"Q{q.question_id} overall",
                    awarded=awarded,
                    max=q.max_marks,
                    justification="Awarded overall (no per-criterion breakdown given).",
                )
            ]
        total = min(max(_num(raw.get("total_awarded", sum(s.awarded for s in scores))), 0.0), q.max_marks)
        feedback = str(raw.get("feedback", "") or "No detailed feedback provided.")
        return GradingResult(
            question_id=q.question_id,
            criterion_scores=scores,
            total_awarded=round(total, 1),
            max_marks=q.max_marks,
            feedback=feedback,
        )

    # -- Mock fallback -------------------------------------------------------

    def _mock_grade(self, q: QuestionSpec, student: StudentAnswer, jitter=0.0) -> GradingResult:
        rng = random.Random(hash((q.question_id, student.student_answer_text)) & 0xFFFFFFFF)
        answer = student.student_answer_text.lower()
        criteria = q.rubric_criteria or [default_criterion(q.max_marks)]
        scores = []
        for c in criteria:
            keywords = set(re.findall(r"[a-z]{4,}", str(c.criterion).lower()))
            keywords |= set(re.findall(r"[a-z]{4,}", q.model_answer.lower()))
            hits = sum(1 for k in keywords if k in answer)
            ratio = hits / max(1, len(keywords))
            awarded = min(round(c.marks * ratio * (0.8 + 0.4 * rng.random()) + jitter * 0.1, 1), c.marks)
            scores.append(
                CriterionScore(
                    criterion=c.criterion,
                    awarded=round(max(awarded, 0.0), 1),
                    max=c.marks,
                    justification="Mock heuristic grading (keyword overlap) - configure an LLM API key for real grading.",
                )
            )
        total = round(min(sum(s.awarded for s in scores), q.max_marks), 1)
        feedback = (
            f"Mock feedback for Q{q.question_id}: {total}/{q.max_marks} marks. "
            "Run with an LLM API key (see .env) for rubric-defensible feedback."
        )
        return GradingResult(
            question_id=q.question_id,
            criterion_scores=scores,
            total_awarded=total,
            max_marks=q.max_marks,
            feedback=feedback,
        )
