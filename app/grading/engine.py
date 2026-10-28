"""Grading engine: per-question LLM grading with confidence and fallbacks.

For each question the engine runs the guards, samples N gradings from the primary
model (falling back to other providers on failure), asks an optional second grader,
referees material disagreements, then scores confidence and applies integrity flags.
Without an API key it uses a deterministic mock grader, so the whole pipeline runs
offline.
"""
import hashlib
import logging
import random
import re
import time
from typing import Callable, List, Optional, Tuple

from app.config import settings
from app.grading.adjudication import cross_model_confidence, maybe_adjudicate, referee_grade
from app.grading.guards import CONFIDENCE_CAP, GuardReport, check_answer, merge_flags, parse_flags

# Re-exported so existing `from app.grading.engine import ...` imports keep working.
from app.grading.prompts import (  # noqa: F401
    GRADE_SCHEMA_HINT,
    build_grade_messages,
    build_grade_prompt,
    build_referee_prompt,
    grade_system,
    new_tag,
    resolve_feedback_language,
)
from app.grading.prompts import fence as _fence  # noqa: F401
from app.grading.scripts import foreign_scripts
from app.ingestion.parser import default_criterion
from app.llm import LLMClient, LLMError, OutputBudgetError, _is_auth_error
from app.models import (
    ConfidenceScore,
    CriterionScore,
    GradingResult,
    QuestionSpec,
    StudentAnswer,
)

logger = logging.getLogger(__name__)

ProgressHook = Callable[[str], None]


def _num(value: object) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


class GradingEngine:
    """Grades one question at a time and attaches a confidence score.

    Args:
        include_semantic: Compute the embedding-based semantic-consistency
            component even when its weight is 0 (the eval harness sets this for
            its ablation). None follows the configured weight.
    """

    def __init__(self, include_semantic: Optional[bool] = None):
        self.llm = LLMClient()
        self.mock = settings.using_mock
        self.include_semantic = include_semantic
        self._last_call = 0.0

    def _maybe_warmup(self) -> None:
        """Ping the provider if the last LLM call was a while ago.

        Free-tier cold starts make the first call after idle very slow. Disabled on
        serverless, where each invocation is isolated and a warm-up only adds latency.
        """
        if self.mock or settings.serverless:
            return
        now = time.time()
        if now - self._last_call > settings.idle_warmup_seconds:
            self.llm.warmup()
        self._last_call = now

    def grade_once(self, q: QuestionSpec, student: StudentAnswer, temperature: Optional[float] = None,
                   jitter: float = 0.0, llm: Optional[LLMClient] = None,
                   feedback_language: Optional[str] = None) -> GradingResult:
        """Grade an answer once.

        Args:
            q: The question.
            student: The student's answer.
            temperature: Sampling temperature (default: GRADING_TEMPERATURE).
            jitter: Mock-mode perturbation so repeated mock samples differ.
            llm: Client to use instead of the primary.
            feedback_language: Language for feedback and justifications (None: English).

        Returns:
            The validated grading.
        """
        if self.mock:
            return self._mock_grade(q, student, jitter)

        def ask():
            system, user = build_grade_messages(q, student, feedback_language)
            raw = (llm or self.llm).chat_json(system, user, temperature=temperature, max_tokens=800)
            return self._validate(q, raw)

        return self._script_checked(q, student, feedback_language, ask(), ask)

    @staticmethod
    def _foreign(q: QuestionSpec, student: StudentAnswer, lang: Optional[str], g: GradingResult) -> List[str]:
        """Scripts in g's feedback that are neither Latin nor the feedback language's."""
        text = " ".join([g.feedback] + [s.justification for s in g.criterion_scores])
        context = [q.question_text, q.model_answer, student.student_answer_text]
        context += [c.criterion for c in q.rubric_criteria]
        return foreign_scripts(text, lang or "English", context)

    def _script_checked(self, q: QuestionSpec, student: StudentAnswer, lang: Optional[str],
                        grading: GradingResult, regenerate: Callable[[], GradingResult]) -> GradingResult:
        """Regenerate a grading once if its feedback drifted into an unrelated script.

        Feedback may use Latin, Greek symbols, the feedback language's script and any
        script that appears in the question, rubric, model answer or student answer.
        Anything else (e.g. Japanese inside Telugu feedback) triggers one retry; if the
        retry fails or still drifts, the grading is kept and the scripts are recorded
        for a warning.
        """
        foreign = self._foreign(q, student, lang, grading)
        if not foreign:
            return grading
        logger.warning("Q%s feedback contains %s text; regenerating once", q.question_id, ", ".join(foreign))
        try:
            retry = regenerate()
        except Exception as exc:  # noqa: BLE001 - keep the first grading
            logger.warning("Regeneration failed for Q%s: %s", q.question_id, exc)
            retry = None
        if retry is not None:
            still = self._foreign(q, student, lang, retry)
            if not still:
                return retry
            grading, foreign = retry, still
        grading._foreign_scripts = foreign
        return grading

    def grade_with_confidence(
        self, q: QuestionSpec, student: StudentAnswer, n_samples: Optional[int] = None,
        threshold: Optional[float] = None, progress: Optional[ProgressHook] = None,
        feedback_language: Optional[str] = None,
    ) -> Tuple[GradingResult, List[GradingResult], ConfidenceScore]:
        """Sample N gradings and score their confidence.

        Args:
            q: The question.
            student: The student's answer.
            n_samples: Gradings to sample (default: CONFIDENCE_SAMPLES).
            threshold: Flagging threshold (default: CONFIDENCE_THRESHOLD).
            progress: Optional hook called with a stage name as the work proceeds
                (grading, graded, second, second_done, moderation, moderation_done).
                For display only; it never affects grading.
            feedback_language: "auto", a language code or a name (default:
                FEEDBACK_LANGUAGE). auto writes feedback in the answer's language.

        Returns:
            (representative, all_samples, confidence). The representative is the
            sample closest to the median total, or the referee's grading.

        Raises:
            LLMError: If every configured grader fails, or on an auth error.
        """
        from app.confidence.scorer import compute_confidence, representative_total

        def notify(stage):
            if progress is not None:
                try:
                    progress(stage)
                except Exception:  # noqa: BLE001 - a display hook must never break grading
                    pass

        n_samples = settings.n_samples if n_samples is None else n_samples
        threshold = settings.confidence_threshold if threshold is None else threshold

        # Cheap pre-checks: blank -> 0 without an LLM call; injections redacted;
        # over-long answers truncated.
        rubric_text = "; ".join(c.criterion for c in q.rubric_criteria)
        report = check_answer(student.student_answer_text, q.question_text, rubric_text)
        warnings: List[str] = [f"Q{q.question_id}: {w}" for w in report.warnings]
        if report.blank:
            return self._blank_result(q, report, warnings, threshold)
        graded = StudentAnswer(question_id=q.question_id, student_answer_text=report.text)
        requested = settings.feedback_language if feedback_language is None else feedback_language
        lang = resolve_feedback_language(requested, report.language)

        notify("grading")
        self._maybe_warmup()
        if self.mock:
            role, graded_by = "mock", "mock"
        else:
            role, graded_by = "primary", self.llm.provider.model
        try:
            results = self._sample_gradings(q, graded, n_samples, self.llm, settings.llm_native_n, lang)
        except Exception as exc:
            # Auth errors are real config problems - never silently fall back.
            if _is_auth_error(exc) or self.mock:
                raise
            results, role, graded_by, warn = self._fallback_grade(q, graded, n_samples, exc, lang)
            warnings.append(warn)
        notify("graded")

        totals = [r.total_awarded for r in results]
        rep_total = representative_total(totals)
        representative = next(r for r in results if r.total_awarded == rep_total)

        judge2_grade = None
        judge2_total = None
        conf_cross = None
        if role == "judge2":
            # Already graded by the second grader - no independent cross-check.
            judge2_total = rep_total
        else:
            if not self.mock and settings.judge2_api_key:
                notify("second")
            judge2_grade = self._second_grader(q, graded, lang)
            if not self.mock and settings.judge2_api_key:
                notify("second_done")
            judge2_total = judge2_grade.total_awarded if judge2_grade is not None else None
            if judge2_total is not None:
                conf_cross = cross_model_confidence(rep_total, judge2_total, q.max_marks)

        referee_model = None
        if role == "judge2":
            adjudicated, pre_total, referee_reason = False, rep_total, None
        else:
            representative, adjudicated, pre_total, referee_reason = self._maybe_adjudicate(
                q, graded, representative, judge2_grade, rep_total, notify, lang
            )
            if adjudicated:
                referee_model = settings.referee_provider().model

        confidence = compute_confidence(
            results, threshold=threshold, conf_cross=conf_cross, representative=rep_total,
            include_semantic=self.include_semantic,
        )
        confidence.judge2_total = judge2_total
        confidence.adjudicated = adjudicated
        confidence.pre_adjudication_total = pre_total
        confidence.referee_reason = referee_reason
        confidence.referee_model = referee_model
        confidence.graded_by = graded_by
        confidence.grader_role = role
        confidence.warnings = warnings
        confidence.feedback_language = lang

        voters = list(results) + [judge2_grade] + ([representative] if adjudicated else [])
        unique = {id(g): g for g in voters if g is not None}.values()  # the referee may repeat
        drifted = [g for g in unique if g._foreign_scripts]
        if drifted:
            scripts = sorted({s for g in drifted for s in g._foreign_scripts})
            confidence.warnings.append(
                f"Q{q.question_id}: {len(drifted)} marking(s) wrote feedback containing "
                f"{', '.join(scripts)} text instead of {lang} even after a retry; it was kept as is."
            )
        flags, reasons = merge_flags(report, voters)
        self._apply_flags(confidence, flags, reasons, report.language)
        return representative, results, confidence

    @staticmethod
    def _apply_flags(confidence: ConfidenceScore, flags: List[str], reasons: List[str],
                     language: Optional[str] = None) -> None:
        """Record integrity flags: they force review and cap confidence, never touch marks.

        The detected answer language is informational only.
        """
        confidence.integrity_flags = list(flags)
        confidence.integrity_reasons = list(reasons)
        confidence.answer_language = language
        if flags:
            confidence.combined_confidence = min(confidence.combined_confidence, CONFIDENCE_CAP)
            confidence.flagged = True

    def _blank_result(self, q: QuestionSpec, report: GuardReport, warnings: List[str], threshold: float):
        """Give 0 marks for a blank answer, without any LLM call."""
        criteria = q.rubric_criteria or [default_criterion(q.max_marks)]
        grading = GradingResult(
            question_id=q.question_id,
            criterion_scores=[
                CriterionScore(criterion=c.criterion, awarded=0.0, max=c.marks,
                               justification="No answer given.")
                for c in criteria
            ],
            total_awarded=0.0,
            max_marks=q.max_marks,
            feedback=(
                "No answer was found for this question, so it received 0 marks without "
                "calling the grader. If the student did answer, check the answer sheet "
                "(or its OCR text) and the question numbering."
            ),
            integrity_flags=list(report.flags),
        )
        confidence = ConfidenceScore(
            n_samples=0,
            score_variance=1.0,
            semantic_consistency=1.0,
            agreement_confidence=1.0,
            combined_confidence=1.0,
            threshold=threshold,
            flagged=False,
            pre_adjudication_total=0.0,
            grader_role="none",
            warnings=warnings,
        )
        self._apply_flags(confidence, report.flags, report.reasons)
        return grading, [], confidence

    def _sample_gradings(self, q: QuestionSpec, student: StudentAnswer, n_samples: int,
                         llm: LLMClient, native_n: bool, lang: Optional[str] = None) -> List[GradingResult]:
        """Sample N gradings from `llm`.

        Uses one native-n request when enabled, else a per-sample loop. Raises if
        the provider fails after retries.
        """
        results = []
        if native_n and not self.mock:
            try:
                system, user = build_grade_messages(q, student, lang)
                raw_list = llm.chat_json_n(
                    system,
                    user,
                    n=n_samples,
                    temperature=settings.grading_temperature,
                    max_tokens=800,
                )
            except Exception as exc:
                # Auth and output-budget errors would repeat in the per-sample loop.
                if _is_auth_error(exc) or isinstance(exc, OutputBudgetError):
                    raise
                raw_list = []
            for raw in raw_list:
                try:
                    grading = self._validate(q, raw)
                except Exception:
                    continue
                results.append(self._script_checked(
                    q, student, lang, grading,
                    lambda: self.grade_once(q, student, llm=llm, feedback_language=lang),
                ))
        if len(results) < 2:
            # Fewer than 2 valid native-n results: fall back to the per-sample loop.
            results = []
            for i in range(n_samples):
                results.append(self.grade_once(q, student, jitter=i, llm=llm, feedback_language=lang))
                # Delay between samples to respect free-tier rate limits
                if not self.mock and settings.request_pacing and i < n_samples - 1:
                    time.sleep(settings.request_pacing)
        return results

    def _fallback_grade(self, q: QuestionSpec, student: StudentAnswer, n_samples: int, primary_exc: Exception,
                        lang: Optional[str] = None):
        """Grade with the next provider in the chain after the primary failed.

        The chain is the fallback provider (per-sample loop, never native n), then
        the second grader. Returns (results, role, model, warning). Re-raises when
        nothing else is configured or every provider fails.
        """
        errors = [f"primary {settings.llm_model}: {primary_exc}"]
        fallback = settings.fallback_provider()
        if fallback is not None:
            logger.warning("Primary grading failed for Q%s; trying fallback %s: %s",
                           q.question_id, fallback.model, primary_exc)
            try:
                results = self._sample_gradings(q, student, n_samples, LLMClient(fallback), False, lang)
                return results, "fallback", fallback.model, (
                    f"Q{q.question_id}: the primary model failed after retries, so this "
                    f"question was graded by the fallback model ({fallback.model})."
                )
            except Exception as exc:  # noqa: BLE001 - move on to the next provider
                logger.warning("Fallback grading failed for Q%s: %s", q.question_id, exc)
                errors.append(f"fallback {fallback.model}: {exc}")
        judge2 = settings.judge2_provider()
        if judge2 is not None:
            logger.warning("Grading Q%s with the second grader %s", q.question_id, judge2.model)
            try:
                results = [self._judge2_grade(q, student, lang)]
                also = " (and so did the fallback)" if fallback is not None else ""
                return results, "judge2", judge2.model, (
                    f"Q{q.question_id}: the primary model failed after retries{also}, so this "
                    f"question was graded once by the second grader ({judge2.model})."
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"second grader {judge2.model}: {exc}")
        if len(errors) == 1:
            raise primary_exc
        raise LLMError("all graders failed - " + "; ".join(errors))

    def _second_grader(self, q: QuestionSpec, student: StudentAnswer,
                       lang: Optional[str] = None) -> Optional[GradingResult]:
        """Grade once with an independent second model (best-effort; None if unavailable)."""
        if self.mock or not settings.judge2_api_key:
            return None
        try:
            return self._judge2_grade(q, student, lang)
        except Exception as exc:  # noqa: BLE001 - judge 2 is best-effort only
            logger.warning("Second-grader check failed for Q%s (skipped): %s", q.question_id, exc)
            return None

    def _maybe_adjudicate(self, q, student, primary, judge2_grade, rep_total, notify=None, lang=None):
        """Referee the two gradings when they disagree materially (see `maybe_adjudicate`)."""
        return maybe_adjudicate(
            q, student, primary, judge2_grade, rep_total, self._validate, notify, lang,
            check=lambda g, ask: self._script_checked(q, student, lang, g, ask),
        )

    def _judge2_grade(self, q: QuestionSpec, student: StudentAnswer, lang: Optional[str] = None) -> GradingResult:
        """Single temperature-0 grading by the independent second model."""
        def ask():
            system, user = build_grade_messages(q, student, lang)
            raw = LLMClient(settings.judge2_provider()).chat_json(
                system, user, temperature=0.0, max_tokens=800, max_attempts=2
            )
            return self._validate(q, raw)

        return self._script_checked(q, student, lang, ask(), ask)

    def _referee_grade(self, q, student, primary, second, lang=None) -> GradingResult:
        """One referee call deciding the final mark (see `referee_grade`)."""
        return referee_grade(q, student, primary, second, self._validate, lang)

    # -- JSON validation -----------------------------------------------------

    def _validate(self, q: QuestionSpec, raw: dict) -> GradingResult:
        """Turn a model's raw JSON into a GradingResult, clamping marks to valid bounds."""
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
            integrity_flags=parse_flags(raw.get("integrity_flags")),
        )

    # -- Mock fallback -------------------------------------------------------

    def _mock_grade(self, q: QuestionSpec, student: StudentAnswer, jitter: float = 0.0) -> GradingResult:
        """Deterministic keyword-overlap grading used when no API key is configured."""
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
                    justification="Mock heuristic grading (keyword overlap) - configure an LLM API key "
                                  "for real grading.",
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
