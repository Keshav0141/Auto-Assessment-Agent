"""Semantic embeddings with a dependency-free fallback.

Prefers sentence-transformers (EMBEDDING_MODEL, default all-MiniLM-L6-v2) when
installed; otherwise falls back to hashed character n-gram vectors so the
confidence pipeline always runs. The model is loaded lazily on first use.
"""
import hashlib
import re
from typing import Iterable

import numpy as np

_sent_model = None
_FEATURE_SPACE = 512


def _load_sentence_transformer():
    global _sent_model
    if _sent_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            from app.config import settings

            _sent_model = SentenceTransformer(settings.embedding_model)
        except Exception:
            _sent_model = False
    return _sent_model


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _char_ngrams(text, n=3):
    text = re.sub(r"\s+", " ", str(text).lower())
    grams = set()
    for tok in re.split(r"\W+", text):
        wrapped = "<" + tok + ">"
        for i in range(max(1, len(wrapped) - n + 1)):
            grams.add(wrapped[i:i + n])
    return grams


def _fallback_vec(text):
    vec = np.zeros(_FEATURE_SPACE, dtype=np.float32)
    for g in _char_ngrams(text):
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
        vec[h % _FEATURE_SPACE] += 1.0
    return _norm(vec)


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    """Embed texts as unit-length vectors.

    Args:
        texts: The texts to embed.

    Returns:
        A (len(texts), dim) array of L2-normalised embeddings.
    """
    model = _load_sentence_transformer()
    if model:
        return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.array([_fallback_vec(t) for t in texts], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity of two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
