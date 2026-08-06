"""Evaluation harness: run grading + confidence on the labeled set and compute
grading accuracy and confidence-calibration metrics (the core MTP deliverable)."""
from typing import Dict, List

import numpy as np

from app.config import settings
from app.ingestion.parser import default_criterion, parse_rubric
from app.models import QuestionSpec, StudentAnswer


def _correlation(x, y):
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def auroc(confidences, correct_flags):
    """Calibration AUROC: probability that a correct grading has HIGHER confidence
    than an erroneous one. Higher = better (low-confidence gradings are the ones
    that actually went wrong). Tie ranks are averaged.

    confidences: predicted confidence (higher -> more confident).
    correct_flags: 1 where |pred - ground_truth| < margin (grading was correct).
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = (np.asarray(correct_flags, dtype=int) == 1)
    n_pos = int(correct.sum())
    n_neg = int((~correct).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(confidences, kind="mergesort")  # ascending confidence
    sorted_conf = confidences[order]
    is_tie_start = np.empty(len(confidences) + 1, dtype=bool)
    is_tie_start[0] = True
    is_tie_start[-1] = True
    is_tie_start[1:-1] = sorted_conf[1:] != sorted_conf[:-1]
    starts = np.flatnonzero(is_tie_start)

    ranks = np.empty(len(confidences), dtype=float)
    for i, s in enumerate(starts[:-1]):
        e = starts[i + 1]
        avg_rank = (s + 1 + e) / 2.0  # mean of 1-indexed ranks s+1 .. e
        ranks[order[s:e]] = avg_rank

    pos_ranks = ranks[correct]
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_metrics(predictions, ground_truths, confidences, errors, margin):
    predictions = np.asarray(predictions, dtype=float)
    ground_truths = np.asarray(ground_truths, dtype=float)
    errors = np.asarray(errors, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    error_flags = (errors >= margin).astype(int)
    return {
        "n_items": int(len(predictions)),
        "mae": float(np.mean(np.abs(errors))),
        "pearson_r": _correlation(predictions, ground_truths),
        "agreement_rate": float(np.mean(error_flags == 0)),
        "auroc": auroc(confidences, error_flags == 0),
        "error_count": int(error_flags.sum()),
        "mean_confidence": float(np.mean(confidences)),
        "mean_confidence_errors": float(np.mean(confidences[error_flags == 1])) if error_flags.sum() else None,
        "mean_confidence_correct": float(np.mean(confidences[error_flags == 0])) if (error_flags == 0).sum() else None,
        "margin": margin,
    }


def calibration_curve(confidences, errors, margin, bins=None):
    confidences = np.asarray(confidences, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if bins is None:
        bins = np.arange(0.0, 1.05, 0.1)
    centers, error_rates, counts = [], [], []
    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        error_rates.append(float(np.mean(errors[mask] >= margin)))
        counts.append(int(mask.sum()))
    return {"bin_centers": centers, "error_rates": error_rates, "counts": counts, "bins": bins.tolist()}


def run_eval(engine, items: List[Dict], n_samples: int = None, threshold: float = None):
    n_samples = n_samples or settings.n_samples
    threshold = threshold if threshold is not None else settings.confidence_threshold
    predictions, ground_truths, confidences, errors, details = [], [], [], [], []
    for it in items:
        rubric_map = parse_rubric(it["rubric_text"])
        criteria = (
            rubric_map.get(it["id"])
            or rubric_map.get("1")  # single-block rubric without question headers
            or [default_criterion(it["max_marks"])]
        )
        q = QuestionSpec(
            question_id=it["id"],
            question_text=it["question"],
            max_marks=float(it["max_marks"]),
            rubric_criteria=criteria,
            model_answer=it["model_answer"],
        )
        student = StudentAnswer(question_id=it["id"], student_answer_text=it["answer"])
        representative, samples, conf = engine.grade_with_confidence(q, student)
        pred = representative.total_awarded
        gt = float(it["ground_truth"])
        error = abs(pred - gt)
        predictions.append(pred)
        ground_truths.append(gt)
        confidences.append(conf.combined_confidence)
        errors.append(error)
        details.append(
            {
                "id": it["id"],
                "predicted": pred,
                "ground_truth": gt,
                "error": round(error, 3),
                "confidence": round(conf.combined_confidence, 3),
                "flagged": conf.flagged,
                "sampled_totals": [round(s.total_awarded, 1) for s in samples],
            }
        )

    metrics = compute_metrics(predictions, ground_truths, confidences, errors, settings.error_margin)
    curve = calibration_curve(confidences, errors, settings.error_margin)
    return metrics, curve, details
