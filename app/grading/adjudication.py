"""Cross-model check and referee adjudication.

When an independent second grader is configured, its mark is compared with the
primary's representative mark. A material disagreement (at least a quarter of the
question's marks) sends both gradings to a referee, whose criterion-by-criterion
grading becomes the final one. Adjudication never raises confidence.
"""
import logging
from typing import Callable, Optional, Tuple

from app.config import settings
from app.grading.prompts import build_referee_prompt, grade_system, new_tag
from app.llm import LLMClient
from app.models import GradingResult, QuestionSpec, StudentAnswer

logger = logging.getLogger(__name__)

# Disagreement (as a share of max marks) that triggers the referee and that drives
# cross-model confidence to 0.
DISAGREEMENT_SHARE = 0.25

Validator = Callable[[QuestionSpec, dict], GradingResult]


def cross_model_confidence(rep_total: float, judge2_total: float, max_marks: float) -> float:
    """Score how closely the second grader agrees with the primary.

    Args:
        rep_total: The primary's representative total.
        judge2_total: The second grader's total.
        max_marks: The question's maximum marks.

    Returns:
        1.0 for identical totals, falling linearly to 0.0 at a gap of a quarter of
        the marks (at least 1 mark).
    """
    denom = max(1.0, DISAGREEMENT_SHARE * max_marks)
    return 1.0 - min(1.0, abs(judge2_total - rep_total) / denom)


def needs_adjudication(rep_total: float, judge2_total: float, max_marks: float) -> bool:
    """Return True when the two graders disagree materially.

    Args:
        rep_total: The primary's representative total.
        judge2_total: The second grader's total.
        max_marks: The question's maximum marks.

    Returns:
        Whether the gap is at least a quarter of the question's marks.
    """
    return abs(judge2_total - rep_total) >= DISAGREEMENT_SHARE * max_marks


def referee_grade(q: QuestionSpec, student: StudentAnswer, primary: GradingResult,
                  second: GradingResult, validate: Validator,
                  feedback_language: Optional[str] = None) -> GradingResult:
    """Ask the referee for the final grading.

    One temperature-0 call to the REFEREE_* provider (default: the primary LLM).

    Args:
        q: The question.
        student: The (guarded) student answer.
        primary: The primary's representative grading.
        second: The second grader's grading.
        validate: Turns the referee's raw JSON into a GradingResult.
        feedback_language: Language for feedback and justifications (None: English).

    Returns:
        The referee's grading.

    Raises:
        LLMError: If the referee call fails after retries.
    """
    tag = new_tag()
    raw = LLMClient(settings.referee_provider()).chat_json(
        grade_system(tag, referee=True),
        build_referee_prompt(q, student, primary, second, tag, feedback_language=feedback_language),
        temperature=0.0,
        max_tokens=800,
    )
    return validate(q, raw)


def maybe_adjudicate(
    q: QuestionSpec,
    student: StudentAnswer,
    primary: GradingResult,
    judge2_grade: Optional[GradingResult],
    rep_total: float,
    validate: Validator,
    notify: Optional[Callable[[str], None]] = None,
    feedback_language: Optional[str] = None,
) -> Tuple[GradingResult, bool, float, Optional[str]]:
    """Referee the two gradings when they disagree materially.

    Args:
        q: The question.
        student: The (guarded) student answer.
        primary: The primary's representative grading.
        judge2_grade: The second grader's grading, or None if it did not run.
        rep_total: The primary's representative total.
        validate: Turns the referee's raw JSON into a GradingResult.
        notify: Optional progress hook, called with "moderation" and
            "moderation_done".
        feedback_language: Language for the referee's feedback (None: English).

    Returns:
        (final_grading, adjudicated, pre_adjudication_total, referee_reason). On
        any referee failure the primary grading is kept unchanged.
    """
    if not settings.adjudicate or judge2_grade is None:
        return primary, False, rep_total, None
    if not needs_adjudication(rep_total, judge2_grade.total_awarded, q.max_marks):
        return primary, False, rep_total, None
    if notify:
        notify("moderation")
    try:
        referee = referee_grade(q, student, primary, judge2_grade, validate, feedback_language)
    except Exception as exc:  # noqa: BLE001 - keep the original grading on failure
        logger.warning("Adjudication failed for Q%s (keeping original): %s", q.question_id, exc)
        return primary, False, rep_total, None
    finally:
        if notify:
            notify("moderation_done")
    return referee, True, rep_total, referee.feedback
