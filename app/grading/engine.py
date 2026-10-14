"""Grading engine: LLM-based per-question grading with JSON validation and a
deterministic mock fallback for API-key-free testing."""
import hashlib
import logging
import random
import re
import time
from typing import List, Tuple

from app.config import settings
from app.ingestion.parser import default_criterion
from app.llm import LLMClient, LLMError, _is_auth_error
from app.models import (
    CriterionScore,
    GradingResult,
    QuestionSpec,
    StudentAnswer,
)

logger = logging.getLogger(__name__)

GRADE_SYSTEM = (
    "You are an expert, strict, fair examiner. Grade the student's answer strictly "
    "against the rubric criteria and the model answer. Award marks only where a "
    "criterion is genuinely met. Award partial credit for each rubric criterion that "
    "is partially met; do not give 0 to an answer that correctly addresses any "
    "criterion. Give feedback that cites exactly which criterion was or was not met "
    "and how the student can improve. Return JSON only, no prose."
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


def _format_grading(g: GradingResult) -> str:
    lines = "\n".join(
        f"  - {s.criterion}: {s.awarded}/{s.max} - {s.justification}"
        for s in g.criterion_scores
    )
    return f"  Total: {g.total_awarded}/{g.max_marks}\n{lines}\n  Feedback: {g.feedback}"


def build_referee_prompt(q: QuestionSpec, student: StudentAnswer, primary: GradingResult, second: GradingResult) -> str:
    return f"""{build_grade_prompt(q, student)}

TWO INDEPENDENT GRADINGS OF THIS ANSWER DISAGREE. As the senior examiner, adjudicate and decide the FINAL mark for each rubric criterion, then give your final total.
GRADER A:
{_format_grading(primary)}
GRADER B:
{_format_grading(second)}

Reconcile the two gradings above and return the final mark per criterion with a one-line reason. Keep the "feedback" field to a single line stating your adjudication reason. {GRADE_SCHEMA_HINT}"""


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
        starts make the first call after idle very slow). Disabled on serverless,
        where each invocation is isolated and a warm-up only adds latency."""
        if self.mock or settings.serverless:
            return
        now = time.time()
        if now - self._last_call > settings.idle_warmup_seconds:
            self.llm.warmup()
        self._last_call = now

    def grade_once(self, q: QuestionSpec, student: StudentAnswer, temperature=None, jitter=0.0):
        if self.mock:
            return self._mock_grade(q, student, jitter)
        raw = self.llm.chat_json(GRADE_SYSTEM, build_grade_prompt(q, student), temperature=temperature, max_tokens=800)
        return self._validate(q, raw)

    def grade_with_confidence(
        self, q: QuestionSpec, student: StudentAnswer, n_samples: int = None, threshold: float = None
    ) -> Tuple[GradingResult, List[GradingResult], object]:
        """Sample N gradings and return (representative, all_samples, confidence)."""
        from app.confidence.scorer import compute_confidence, representative_total

        n_samples = settings.n_samples if n_samples is None else n_samples
        warnings: List[str] = []
        self._maybe_warmup()
        fallback_used = False
        try:
            results = self._sample_gradings(q, student, n_samples)
        except Exception as exc:
            # Auth errors are real config problems - never silently fall back.
            if _is_auth_error(exc) or self.mock or not settings.judge2_api_key:
                raise
            logger.warning("Primary grading failed for Q%s; using second grader: %s", q.question_id, exc)
            results = [self._judge2_grade(q, student)]
            fallback_used = True
            warnings.append(
                f"Q{q.question_id}: the primary model failed after retries, so this "
                "question was graded by the second grader instead."
            )

        totals = [r.total_awarded for r in results]
        rep_total = representative_total(totals)
        representative = next(r for r in results if r.total_awarded == rep_total)

        judge2_grade = None
        judge2_total = None
        conf_cross = None
        if fallback_used:
            # Already graded by the second grader - no independent cross-check.
            judge2_total = rep_total
        else:
            judge2_grade = self._second_grader(q, student)
            judge2_total = judge2_grade.total_awarded if judge2_grade is not None else None
            if judge2_total is not None:
                denom = max(1.0, 0.25 * q.max_marks)
                conf_cross = 1.0 - min(1.0, abs(judge2_total - rep_total) / denom)

        if fallback_used:
            adjudicated, pre_total, referee_reason = False, rep_total, None
        else:
            representative, adjudicated, pre_total, referee_reason = self._maybe_adjudicate(
                q, student, representative, judge2_grade, rep_total
            )

        confidence = compute_confidence(
            results, threshold=threshold, conf_cross=conf_cross, representative=rep_total
        )
        confidence.judge2_total = judge2_total
        confidence.adjudicated = adjudicated
        confidence.pre_adjudication_total = pre_total
        confidence.referee_reason = referee_reason
        confidence.warnings = warnings
        return representative, results, confidence

    def _sample_gradings(self, q: QuestionSpec, student: StudentAnswer, n_samples: int) -> List[GradingResult]:
        """Sample N gradings from the primary model (native-n if enabled, else a
        per-sample loop). Raises if the provider fails after retries."""
        results = []
        if settings.llm_native_n and not self.mock:
            try:
                raw_list = self.llm.chat_json_n(
                    GRADE_SYSTEM,
                    build_grade_prompt(q, student),
                    n=n_samples,
                    temperature=settings.grading_temperature,
                    max_tokens=800,
                )
            except Exception:
                raw_list = []
            for raw in raw_list:
                try:
                    results.append(self._validate(q, raw))
                except Exception:
                    continue
        if len(results) < 2:
            # Fewer than 2 valid native-n results: fall back to the per-sample loop.
            results = []
            for i in range(n_samples):
                results.append(self.grade_once(q, student, jitter=i))
                # Delay between samples to respect free-tier rate limits
                if not self.mock and settings.request_pacing and i < n_samples - 1:
                    time.sleep(settings.request_pacing)
        return results

    def _second_grader(self, q: QuestionSpec, student: StudentAnswer):
        """Grade once with an independent second model (best-effort; None if unavailable)."""
        if self.mock or not settings.judge2_api_key:
            return None
        try:
            return self._judge2_grade(q, student)
        except Exception as exc:  # noqa: BLE001 - judge 2 is best-effort only
            logger.warning("Second-grader check failed for Q%s (skipped): %s", q.question_id, exc)
            return None

    def _maybe_adjudicate(self, q, student, primary, judge2_grade, rep_total):
        """Referee the two gradings when they disagree materially.

        Returns (final_grading, adjudicated, pre_adjudication_total, referee_reason).
        On any failure the original grading is kept unchanged.
        """
        if not settings.adjudicate or judge2_grade is None:
            return primary, False, rep_total, None
        if abs(judge2_grade.total_awarded - rep_total) < 0.25 * q.max_marks:
            return primary, False, rep_total, None
        try:
            referee = self._referee_grade(q, student, primary, judge2_grade)
        except Exception as exc:  # noqa: BLE001 - keep the original grading on failure
            logger.warning("Adjudication failed for Q%s (keeping original): %s", q.question_id, exc)
            return primary, False, rep_total, None
        return referee, True, rep_total, referee.feedback

    def _judge2_grade(self, q: QuestionSpec, student: StudentAnswer) -> GradingResult:
        """Single temperature-0 grading by the independent second model."""
        from openai import OpenAI

        from app.llm import extract_json

        client = OpenAI(base_url=settings.judge2_base_url, api_key=settings.judge2_api_key)
        resp = client.chat.completions.create(
            model=settings.judge2_model,
            messages=[
                {"role": "system", "content": GRADE_SYSTEM},
                {"role": "user", "content": build_grade_prompt(q, student)},
            ],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        return self._validate(q, extract_json(content))

    def _referee_grade(self, q, student, primary, second) -> GradingResult:
        """One temperature-0 referee call by the primary LLM deciding the final mark."""
        raw = self.llm.chat_json(
            GRADE_SYSTEM,
            build_referee_prompt(q, student, primary, second),
            temperature=0.0,
            max_tokens=800,
        )
        return self._validate(q, raw)

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
        seed = int(hashlib.md5(f"{q.question_id}|{student.student_answer_text}".encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed & 0xFFFFFFFF)
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
