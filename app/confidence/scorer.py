"""Black-box confidence scoring for grading decisions.

Given N sampled grading results for the same question, compute a 0-1 confidence
score using only the outputs (no logprobs/logits), following the black-box
uncertainty quantification family of methods from Lin et al. 2023
("Generating with Confidence"):

  * Score variance      : spread of `total_awarded` across samples.
  * Semantic consistency: greedy clustering of the N feedback texts by embedding
                          cosine similarity -> NumSet / Deg-style measures.
  * EigV (optional)     : eigenvalue spread on the similarity-graph Laplacian.

The two embedding-based measures are only computed when they can affect the result
(semantic weight > 0, or EigV requested), so with the default weights no embedding
model is loaded at all.
"""
from typing import List, Optional

import numpy as np

from app.config import settings
from app.embeddings import cosine_similarity, embed_texts
from app.models import ConfidenceScore, GradingResult


def _score_variance_confidence(totals: List[float], max_marks: float) -> float:
    """Return 1 - std(totals) / max_marks, clipped to [0, 1]."""
    if max_marks <= 0:
        return 1.0
    arr = np.asarray(totals, dtype=float)
    std = arr.std()
    return float(np.clip(1.0 - std / max_marks, 0.0, 1.0))


def representative_total(totals: List[float]) -> float:
    """Pick the representative mark for a set of sampled totals.

    Args:
        totals: Sampled `total_awarded` values.

    Returns:
        The median, snapped to the nearest sampled value (0.0 when empty).
    """
    arr = np.asarray(totals, dtype=float)
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    idx = int(np.argmin(np.abs(arr - med)))
    return float(arr[idx])


def _agreement_confidence(totals: List[float], representative: float, tol: float = 0.25) -> float:
    """Fraction of sample totals within `tol` marks of the representative total."""
    arr = np.asarray(totals, dtype=float)
    if arr.size == 0:
        return 1.0
    return float(np.mean(np.abs(arr - representative) <= tol))


def _max_linkage_clusters(vectors, threshold):
    """Cluster vectors by max-linkage.

    Each vector joins the cluster containing its most similar member (cosine >=
    threshold). Robust to rephrased-but-identical feedback.
    """
    clusters = []
    for v in vectors:
        best_i, best_sim = -1, -1.0
        for i, members in enumerate(clusters):
            s = max(cosine_similarity(v, m) for m in members)
            if s > best_sim:
                best_sim, best_i = s, i
        if best_i >= 0 and best_sim >= threshold:
            clusters[best_i].append(v)
        else:
            clusters.append([v])
    return clusters


def _semantic_consistency(feedbacks: List[str]) -> float:
    """Average of NumSet- and Deg-style scores over clustered feedback embeddings."""
    n = len(feedbacks)
    if n <= 1:
        return 1.0
    if not any(feedbacks):
        return 0.5
    vectors = embed_texts(feedbacks)
    clusters = _max_linkage_clusters(list(vectors), settings.cluster_threshold)
    n_clusters = len(clusters)
    # NumSet-style: fewer distinct clusters -> more confident
    numset = 1.0 - (n_clusters - 1) / (n - 1)
    # Deg-style: fraction of samples in the largest cluster
    largest = max(len(c) for c in clusters)
    deg = largest / n
    return float(0.5 * numset + 0.5 * deg)


def _eigv_confidence(feedbacks: List[str]) -> float:
    """Eigenvalue-based confidence (EigV-ish) on the similarity graph Laplacian."""
    n = len(feedbacks)
    if n <= 1:
        return 1.0
    vectors = embed_texts(feedbacks)
    sims = np.array([[cosine_similarity(a, b) for b in vectors] for a in vectors])
    sims = np.clip(sims, 0.0, 1.0)
    # Normalized Laplacian; use algebraic connectivity (second smallest eigenvalue).
    deg = sims.sum(axis=1)
    deg[deg == 0] = 1e-9
    d_inv = np.diag(1.0 / np.sqrt(deg))
    lap = np.eye(n) - d_inv @ sims @ d_inv
    eigvals = np.linalg.eigvalsh(lap)
    algebraic = float(np.clip(eigvals[1], 0.0, None))
    # For a fully-connected (identical) set this approaches 1; for disconnected
    # clusters it approaches 0. Map to 0-1 via saturation at 1.0.
    return float(np.clip(algebraic, 0.0, 1.0))


def compute_confidence(
    results: List[GradingResult],
    threshold: Optional[float] = None,
    include_eigv: bool = False,
    conf_cross: Optional[float] = None,
    representative: Optional[float] = None,
    include_semantic: Optional[bool] = None,
) -> ConfidenceScore:
    """Combine the confidence components for one question.

    Missing components (e.g. no cross-model judge) and zero-weight ones are skipped
    and the remaining weights renormalised.

    Args:
        results: The sampled gradings (at least one).
        threshold: Flagging threshold (default: CONFIDENCE_THRESHOLD).
        include_eigv: Also compute the EigV component (fixed weight 0.2).
        conf_cross: Cross-model agreement in [0, 1], or None without a second grader.
        representative: The representative total (default: `representative_total`).
        include_semantic: Compute semantic consistency. None computes it only when
            W_SEMANTIC > 0; when it is not computed, `semantic_consistency` is None
            and no embeddings are made.

    Returns:
        A ConfidenceScore with `flagged` set when the combined score is below the
        threshold.

    Raises:
        ValueError: If `results` is empty.
    """
    if not results:
        raise ValueError("compute_confidence requires at least one grading result")
    threshold = settings.confidence_threshold if threshold is None else threshold
    totals = [r.total_awarded for r in results]
    max_marks = max(r.max_marks for r in results)
    feedbacks = [r.feedback for r in results]
    if representative is None:
        representative = representative_total(totals)

    conf_agree = _agreement_confidence(totals, representative)
    conf_var = _score_variance_confidence(totals, max_marks)
    if include_semantic is None:
        include_semantic = settings.w_semantic > 0
    conf_sem = _semantic_consistency(feedbacks) if include_semantic else None
    conf_eig = _eigv_confidence(feedbacks) if include_eigv else None

    # Weighted average of the available components; missing components (e.g. no
    # cross-model judge) are skipped and the remaining weights renormalised.
    components = [
        (settings.w_agree, conf_agree),
        (settings.w_variance, conf_var),
        (settings.w_semantic, conf_sem),
        (settings.w_cross, conf_cross),
    ]
    if include_eigv and conf_eig is not None:
        components.append((0.2, conf_eig))
    used = [(w, v) for (w, v) in components if v is not None and w > 0]
    weight_sum = sum(w for w, _ in used)
    combined = float(sum(w * v for w, v in used) / weight_sum) if weight_sum > 0 else 0.0
    combined = float(np.clip(combined, 0.0, 1.0))
    return ConfidenceScore(
        n_samples=len(results),
        score_variance=conf_var,
        semantic_consistency=conf_sem,
        agreement_confidence=conf_agree,
        cross_model_confidence=conf_cross,
        eigv_confidence=conf_eig,
        combined_confidence=combined,
        threshold=threshold,
        flagged=combined < threshold,
    )
