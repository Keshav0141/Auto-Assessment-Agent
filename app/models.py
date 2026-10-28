"""Pydantic schemas shared by the parser, grading engine, scorer and API."""
from typing import List, Optional

from pydantic import BaseModel


class RubricCriterion(BaseModel):
    """One rubric line: what earns marks and how many."""

    criterion: str
    marks: float


class QuestionSpec(BaseModel):
    """A question with its marks, rubric criteria and model answer."""

    question_id: str
    question_text: str
    max_marks: float
    rubric_criteria: List[RubricCriterion] = []
    model_answer: str = ""
    # False when the question paper gave no marks for this question (max_marks is
    # then derived from the rubric, or defaults to 1).
    marks_specified: bool = True


class StudentAnswer(BaseModel):
    """The student's answer text for one question."""

    question_id: str
    student_answer_text: str


class CriterionScore(BaseModel):
    """Marks awarded for one rubric criterion, with the reason."""

    criterion: str
    awarded: float
    max: float
    justification: str


class GradingResult(BaseModel):
    """One grading of one answer: per-criterion marks, total and feedback."""

    question_id: str
    criterion_scores: List[CriterionScore]
    total_awarded: float
    max_marks: float
    feedback: str
    integrity_flags: List[str] = []


class ConfidenceScore(BaseModel):
    """Confidence components, the combined score and review status for one question."""

    n_samples: int
    score_variance: float
    # None when not computed (W_SEMANTIC is 0, so no embeddings are made).
    semantic_consistency: Optional[float] = None
    agreement_confidence: Optional[float] = None
    cross_model_confidence: Optional[float] = None
    judge2_total: Optional[float] = None
    adjudicated: bool = False
    pre_adjudication_total: Optional[float] = None
    referee_reason: Optional[str] = None
    referee_model: Optional[str] = None
    graded_by: Optional[str] = None      # model that produced the final samples
    grader_role: Optional[str] = None    # primary | fallback | judge2 | mock | none (no LLM call)
    integrity_flags: List[str] = []
    integrity_reasons: List[str] = []
    # Detected language of the answer (e.g. "English", "Hindi", "Hinglish"). Informational:
    # never a flag unless the rubric explicitly requires a language.
    answer_language: Optional[str] = None
    eigv_confidence: Optional[float] = None
    warnings: List[str] = []
    combined_confidence: float
    threshold: float
    flagged: bool


class QuestionResult(BaseModel):
    """The final grading, confidence and sampled totals for one question."""

    grading: GradingResult
    confidence: ConfidenceScore
    sampled_totals: List[float]
    question_text: str = ""
    student_answer_text: str = ""


class PipelineResult(BaseModel):
    """The response of /api/grade: every question plus totals and warnings."""

    results: List[QuestionResult]
    total_awarded: float
    total_max: float
    overall_confidence: float
    flagged: List[str]
    warnings: List[str] = []
    using_mock: bool = False
