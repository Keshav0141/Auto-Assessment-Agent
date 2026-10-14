"""Pydantic schemas for the whole pipeline."""
from typing import List, Optional

from pydantic import BaseModel


class RubricCriterion(BaseModel):
    criterion: str
    marks: float


class QuestionSpec(BaseModel):
    question_id: str
    question_text: str
    max_marks: float
    rubric_criteria: List[RubricCriterion] = []
    model_answer: str = ""


class StudentAnswer(BaseModel):
    question_id: str
    student_answer_text: str


class CriterionScore(BaseModel):
    criterion: str
    awarded: float
    max: float
    justification: str


class GradingResult(BaseModel):
    question_id: str
    criterion_scores: List[CriterionScore]
    total_awarded: float
    max_marks: float
    feedback: str


class ConfidenceScore(BaseModel):
    n_samples: int
    score_variance: float
    semantic_consistency: float
    agreement_confidence: Optional[float] = None
    cross_model_confidence: Optional[float] = None
    judge2_total: Optional[float] = None
    adjudicated: bool = False
    pre_adjudication_total: Optional[float] = None
    referee_reason: Optional[str] = None
    eigv_confidence: Optional[float] = None
    warnings: List[str] = []
    combined_confidence: float
    threshold: float
    flagged: bool


class QuestionResult(BaseModel):
    grading: GradingResult
    confidence: ConfidenceScore
    sampled_totals: List[float]
    question_text: str = ""
    student_answer_text: str = ""


class PipelineResult(BaseModel):
    results: List[QuestionResult]
    total_awarded: float
    total_max: float
    overall_confidence: float
    flagged: List[str]
    warnings: List[str] = []
    using_mock: bool = False
