"""Polarity-aware unit embedding service (384d).

Combines morphological suffix stemming, subword n-grams, word bigrams,
stopword downweighting, and polarity-aware coordinate projections.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Union

import numpy as np

STOPWORDS = {
    "for", "using", "the", "a", "an", "in", "on", "and", "or", "to", "of",
    "with", "by", "is", "are", "as", "at", "it", "from", "that", "this",
}


def _stem(w: str) -> str:
    """Lightweight suffix stemmer for morphological normalization."""
    w = w.lower()
    for suff in ("ing", "tions", "tion", "ies", "es", "ed", "al", "s"):
        if w.endswith(suff) and len(w) > len(suff) + 2:
            return w[:-len(suff)]
    return w


def _tokenize(text: str) -> List[str]:
    return [_stem(w) for w in re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())]


class PolarityAwareEmbeddingService:
    """Zero-daemon, deterministic 384d vector embedding generator with polarity projection."""

    EMBEDDING_DIM = 384

    NEGATION_TERMS = {
        "not", "no", "never", "inhibit", "decreas", "suppress", "block",
        "fail", "falsifi", "refut", "disprov", "cannot", "neither",
        "nor", "without", "absent", "deactivat", "toxic", "dead_end", "repel"
    }

    AFFIRMATIVE_TERMS = {
        "activat", "promot", "increas", "succeed", "success", "confirm",
        "prove", "proven", "verifi", "present", "enabl", "catalyz"
    }

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        """Encodes text into a normalized 384-dimensional polarity-aware vector."""
        if not text or not text.strip():
            v = np.zeros(self.dim, dtype=np.float32)
            v[0] = 1.0
            return v

        tokens = _tokenize(text)
        if not tokens:
            v = np.zeros(self.dim, dtype=np.float32)
            v[0] = 1.0
            return v

        vec = np.zeros(self.dim, dtype=np.float32)
        polarity_score = 0.0

        for tok in tokens:
            weight = 0.15 if tok in STOPWORDS else 1.0
            if tok in self.NEGATION_TERMS:
                polarity_score -= 1.0
                weight = 1.5
            elif tok in self.AFFIRMATIVE_TERMS:
                polarity_score += 1.0
                weight = 1.5

            # 1. Unigram feature hashing
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += weight

            # 2. Subword 3-grams for morphological alignment
            if len(tok) >= 4 and tok not in STOPWORDS:
                for j in range(len(tok) - 2):
                    sub = tok[j : j + 3]
                    hs = int(hashlib.sha256(sub.encode("utf-8")).hexdigest(), 16)
                    vec[hs % self.dim] += 0.35

        # 3. Word bigrams for phrase matching
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 not in STOPWORDS or w2 not in STOPWORDS:
                bigram = f"{w1}_{w2}"
                hb = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
                vec[hb % self.dim] += 0.8

        # Dedicate coordinates 0..7 to polarity balance (scaled smoothly)
        pol_mag = math.tanh(polarity_score)
        for p in range(8):
            sign = 1.0 if p % 2 == 0 else -1.0
            vec[p] += pol_mag * 0.4 * sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def cosine_similarity(self, a: Union[np.ndarray, List[float]], b: Union[np.ndarray, List[float]]) -> float:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))


embedding_service = PolarityAwareEmbeddingService()
