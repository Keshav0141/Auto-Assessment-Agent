"""Black-box confidence scoring for grading decisions.

Given N sampled grading results for the same question, compute a 0-1 confidence
score using only the outputs (no logprobs/logits), following the black-box
uncertainty quantification family of methods from Lin et al. 2023
("Generating with Confidence"):

  * Score variance      : spread of `total_awarded` across samples.
  * Semantic consistency: greedy clustering of the N feedback texts by embedding
                          cosine similarity -> NumSet / Deg-style measures.
  * EigV (optional)     : eigenvalue spread on the similarity-graph Laplacian.
"""
from typing import List

import numpy as np

from app.config import settings
from app.embeddings import cosine_similarity, embed_texts
from app.models import ConfidenceScore, GradingResult


def _score_variance_confidence(totals: List[float], max_marks: float) -> float:
    if max_marks <= 0:
        return 1.0
    arr = np.asarray(totals, dtype=float)
    std = arr.std()
    return float(np.clip(1.0 - std / max_marks, 0.0, 1.0))


def _max_linkage_clusters(vectors, threshold):
    """Cluster vectors by max-linkage: join the cluster containing the most similar
    member (cosine >= threshold). Robust to rephrased-but-identical feedback."""
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
    threshold: float | None = None,
    include_eigv: bool = False,
) -> ConfidenceScore:
    if not results:
        raise ValueError("compute_confidence requires at least one grading result")
    threshold = settings.confidence_threshold if threshold is None else threshold
    totals = [r.total_awarded for r in results]
    max_marks = max(r.max_marks for r in results)
    feedbacks = [r.feedback for r in results]

    conf_var = _score_variance_confidence(totals, max_marks)
    conf_sem = _semantic_consistency(feedbacks)
    conf_eig = _eigv_confidence(feedbacks) if include_eigv else None

    if include_eigv and conf_eig is not None:
        combined = (
            settings.w_variance * conf_var
            + settings.w_semantic * conf_sem
            + 0.2 * conf_eig
        ) / (settings.w_variance + settings.w_semantic + 0.2)
    else:
        combined = (
            settings.w_variance * conf_var + settings.w_semantic * conf_sem
        ) / (settings.w_variance + settings.w_semantic)

    combined = float(np.clip(combined, 0.0, 1.0))
    return ConfidenceScore(
        n_samples=len(results),
        score_variance=conf_var,
        semantic_consistency=conf_sem,
        eigv_confidence=conf_eig,
        combined_confidence=combined,
        threshold=threshold,
        flagged=combined < threshold,
    )
