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


def low_conf_threshold_analysis(confidences, errors, margin, frac=0.2):
    """Threshold that flags the lowest-confidence `frac` of items, and the error
    rate among flagged vs unflagged items at that threshold."""
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


def threshold_error_rates(confidences, errors, margin, threshold):
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


def calibration_curve(confidences, errors, margin, bins=None):
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


def run_eval(engine, items: List[Dict], n_samples: int = None, threshold: float = None):
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
