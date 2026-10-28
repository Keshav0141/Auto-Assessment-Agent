"""Evaluation harness for grading accuracy and confidence calibration.

Grades every labelled item with the full pipeline and computes MAE, Pearson r,
agreement rate, calibration AUROC (overall and per confidence component), flagging
error rates and a calibration curve.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import settings
from app.ingestion.parser import default_criterion, parse_rubric
from app.models import QuestionSpec, StudentAnswer


def _correlation(x, y):
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def auroc(confidences: Sequence[float], correct_flags: Sequence[int]) -> float:
    """Compute the calibration AUROC.

    The probability that a correct grading has higher confidence than an erroneous
    one, so higher is better (low-confidence gradings are the ones that actually went
    wrong). Tie ranks are averaged.

    Args:
        confidences: Predicted confidence (higher means more confident).
        correct_flags: 1 where |pred - ground_truth| < margin (grading was correct).

    Returns:
        The AUROC, or NaN when every item is correct or every item is wrong.
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


def compute_metrics(predictions: Sequence[float], ground_truths: Sequence[float], confidences: Sequence[float],
                    errors: Sequence[float], margin: float) -> Dict[str, Any]:
    """Compute grading-accuracy and calibration metrics.

    Args:
        predictions: Predicted marks.
        ground_truths: Ground-truth marks.
        confidences: Combined confidence per item.
        errors: |prediction - ground truth| per item.
        margin: An error of at least this many marks counts as wrong.

    Returns:
        n_items, mae, pearson_r, agreement_rate, auroc, error counts and mean
        confidences.
    """
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


def _safe_auroc(values, errors, margin):
    """AUROC for one confidence component, ignoring items where it is missing."""
    pairs = [(v, e) for v, e in zip(values, errors) if v is not None]
    if len(pairs) < 2:
        return None
    vs = [p[0] for p in pairs]
    errs = [p[1] for p in pairs]
    correct = [e < margin for e in errs]
    if all(correct) or not any(correct):
        return None
    return auroc(vs, correct)


def low_conf_threshold_analysis(confidences: Sequence[float], errors: Sequence[float], margin: float,
                                frac: float = 0.2) -> Dict[str, Any]:
    """Flag the lowest-confidence `frac` of items and compare error rates.

    Returns:
        The threshold that flags that fraction, the flagged/unflagged counts, and the
        error rate in each group.
    """
    conf = np.asarray(confidences, dtype=float)
    err = np.asarray(errors, dtype=float)
    thr = float(np.quantile(conf, frac))
    flagged = conf <= thr
    unflagged = ~flagged
    return {
        "fraction": frac,
        "threshold": thr,
        "n_flagged": int(flagged.sum()),
        "n_unflagged": int(unflagged.sum()),
        "error_rate_flagged": float(np.mean(err[flagged] >= margin)) if flagged.sum() else None,
        "error_rate_unflagged": float(np.mean(err[unflagged] >= margin)) if unflagged.sum() else None,
    }


def threshold_error_rates(confidences: Sequence[float], errors: Sequence[float], margin: float,
                          threshold: float) -> Dict[str, Any]:
    """Error rate among items flagged (confidence < threshold) vs unflagged."""
    conf = np.asarray(confidences, dtype=float)
    err = np.asarray(errors, dtype=float)
    flagged = conf < threshold
    unflagged = ~flagged
    return {
        "threshold": threshold,
        "n_flagged": int(flagged.sum()),
        "n_unflagged": int(unflagged.sum()),
        "error_rate_flagged": float(np.mean(err[flagged] >= margin)) if flagged.sum() else None,
        "error_rate_unflagged": float(np.mean(err[unflagged] >= margin)) if unflagged.sum() else None,
    }


def calibration_curve(confidences: Sequence[float], errors: Sequence[float], margin: float,
                      bins: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
    """Bin items by confidence and report the observed error rate per bin.

    Args:
        confidences: Combined confidence per item.
        errors: |prediction - ground truth| per item.
        margin: An error of at least this many marks counts as wrong.
        bins: Bin edges (default: 0.0 to 1.0 in steps of 0.1).

    Returns:
        bin_centers, error_rates and counts for the non-empty bins, and the edges.
    """
    confidences = np.asarray(confidences, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if bins is None:
        bins = np.arange(0.0, 1.05, 0.1)
    centers, error_rates, counts = [], [], []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            # final bin is inclusive of 1.0
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        error_rates.append(float(np.mean(errors[mask] >= margin)))
        counts.append(int(mask.sum()))
    return {"bin_centers": centers, "error_rates": error_rates, "counts": counts, "bins": bins.tolist()}


def run_eval(engine: Any, items: List[Dict], n_samples: Optional[int] = None,
             threshold: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, List[float]], List[Dict]]:
    """Grade every item and compute all metrics.

    Args:
        engine: A GradingEngine.
        items: Labelled items from eval.dataset.
        n_samples: Gradings per item (default: CONFIDENCE_SAMPLES).
        threshold: Flagging threshold (default: CONFIDENCE_THRESHOLD).

    Returns:
        (metrics, calibration_curve, per_item_details).
    """
    n_samples = n_samples or settings.n_samples
    threshold = threshold if threshold is not None else settings.confidence_threshold
    predictions, ground_truths, confidences, errors, details = [], [], [], [], []
    pre_predictions = []
    agrees, variances, semantics, crosses = [], [], [], []
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
        representative, samples, conf = engine.grade_with_confidence(
            q, student, n_samples=n_samples, threshold=threshold
        )
        pred = representative.total_awarded
        gt = float(it["ground_truth"])
        error = abs(pred - gt)
        pre_pred = conf.pre_adjudication_total if conf.pre_adjudication_total is not None else pred
        predictions.append(pred)
        pre_predictions.append(pre_pred)
        ground_truths.append(gt)
        confidences.append(conf.combined_confidence)
        agrees.append(conf.agreement_confidence)
        variances.append(conf.score_variance)
        semantics.append(conf.semantic_consistency)
        crosses.append(conf.cross_model_confidence)
        errors.append(error)
        details.append(
            {
                "id": it["id"],
                "predicted": pred,
                "pre_adjudication_total": pre_pred,
                "adjudicated": conf.adjudicated,
                "referee_reason": conf.referee_reason,
                "ground_truth": gt,
                "error": round(error, 3),
                "confidence": round(conf.combined_confidence, 3),
                "agreement": None if conf.agreement_confidence is None else round(conf.agreement_confidence, 3),
                "cross_model": None if conf.cross_model_confidence is None else round(conf.cross_model_confidence, 3),
                "judge2_total": conf.judge2_total,
                "referee_model": conf.referee_model,
                "graded_by": conf.graded_by,
                "integrity_flags": conf.integrity_flags,
                "flagged": conf.flagged,
                "sampled_totals": [round(s.total_awarded, 1) for s in samples],
            }
        )

    margin = settings.error_margin
    metrics = compute_metrics(predictions, ground_truths, confidences, errors, margin)
    pre_errors = [abs(p - g) for p, g in zip(pre_predictions, ground_truths)]
    metrics["mae_before_adjudication"] = float(np.mean(pre_errors))
    metrics["agreement_rate_before_adjudication"] = float(np.mean(np.asarray(pre_errors) < margin))
    metrics["n_adjudicated"] = int(sum(1 for d in details if d["adjudicated"]))
    metrics["n_referee_changed"] = int(sum(
        1 for d in details if d["adjudicated"] and abs(d["predicted"] - d["pre_adjudication_total"]) > 1e-9
    ))
    metrics["referee_models"] = sorted({d["referee_model"] for d in details if d.get("referee_model")})
    metrics["n_integrity_flagged"] = int(sum(1 for d in details if d.get("integrity_flags")))
    post_flags = (np.asarray(errors, dtype=float) >= margin)
    pre_flags = (np.asarray(pre_errors, dtype=float) >= margin)
    metrics["auroc_pre_adjudication"] = auroc(confidences, pre_flags == 0)
    metrics["ablation"] = {
        "agree": _safe_auroc(agrees, errors, margin),
        "variance": _safe_auroc(variances, errors, margin),
        "semantic": _safe_auroc(semantics, errors, margin),
        "cross": _safe_auroc(crosses, errors, margin),
        "combined": auroc(confidences, post_flags == 0),
    }
    metrics["ablation_pre_adjudication"] = {
        "agree": _safe_auroc(agrees, pre_errors, margin),
        "variance": _safe_auroc(variances, pre_errors, margin),
        "semantic": _safe_auroc(semantics, pre_errors, margin),
        "cross": _safe_auroc(crosses, pre_errors, margin),
        "combined": auroc(confidences, pre_flags == 0),
    }
    metrics["low_conf_20pct"] = low_conf_threshold_analysis(confidences, errors, margin, frac=0.2)
    thr = settings.confidence_threshold
    metrics["flagged_at_threshold"] = {
        "threshold": thr,
        "post": threshold_error_rates(confidences, errors, margin, thr),
        "pre": threshold_error_rates(confidences, pre_errors, margin, thr),
    }
    curve = calibration_curve(confidences, errors, margin)
    return metrics, curve, details
