"""Offline distilled paraphrase-relevance embedder for decomposition ranking.

This is the OFFLINE counterpart of the optional nomic-embed-text semantic
scorer. It is a deterministic hashed-n-gram feature extractor plus a learned
bilinear projection, distilled from nomic-embed-text's cosine similarities on
(parent, paraphrase/distractor/unrelated) triples.

Why it exists: the deterministic hashing polarity embedder cannot rank a
zero-surface-overlap paraphrase above an off-topic lexical distractor (proven).
A local nomic-embed-text daemon fixes that (CAR 1.000) but needs a daemon.
Distilling nomic into this student recovers the same ranking OFFLINE:
  - 58 disjoint training parents (zero overlap with the eval probes): CAR 1.000
  - 16 disjoint training parents: CAR 0.875
  - polarity baseline: CAR 0.750
  (metric = correct-above-irrelevant rate on an 8-item held-out probe set.)

Contract: deterministic (fixed seed, pure NumPy), no network, artifact is a
single .npz of the projection matrix. Falls back gracefully (returns None) when
the weights file is absent, so callers keep the polarity path.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional, Union

import numpy as np

_DIM_DEFAULT = 512
_DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "distilled_pid_large.npz"
)

_STOP = {
    "for", "using", "the", "a", "an", "in", "on", "and", "or", "to", "of",
    "with", "by", "is", "are", "as", "at", "it", "from", "that", "this",
    "how", "why", "we", "do", "does", "when", "so", "no", "its",
}


def _hash(s: str, dim: int) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % dim


def features(text: str, dim: int = _DIM_DEFAULT) -> np.ndarray:
    """Hashed bag-of-ngrams: word uni/bigrams + char 4-grams (roots)."""
    v = np.zeros(dim, dtype=np.float32)
    low = text.lower()
    words = re.findall(r"[a-z0-9]+", low)
    for w in words:
        v[_hash("w:" + w, dim)] += 0.15 if w in _STOP else 1.0
    for i in range(len(words) - 1):
        v[_hash("b:" + words[i] + "_" + words[i + 1], dim)] += 0.8
    for j in range(len(low) - 3):
        if low[j] == " ":
            continue
        v[_hash("c4:" + low[j:j + 4], dim)] += 0.35
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class DistilledPIDEmbedder:
    """Deterministic offline student embedder (bilinear projection)."""

    name = "distilled_pid"

    def __init__(self, weights_path: Optional[str] = None, dim: int = _DIM_DEFAULT):
        self.dim = dim
        self.W: Optional[np.ndarray] = None
        path = weights_path or os.getenv("SUPER_SOLVER_PID_WEIGHTS", _DEFAULT_WEIGHTS)
        try:
            data = np.load(path)
            self.W = data["W"].astype(np.float32)
            self.dim = int(data["dim"])
        except Exception:
            self.W = None

    @property
    def available(self) -> bool:
        return self.W is not None

    def encode(self, text: str) -> np.ndarray:
        f = features(text, self.dim)
        if self.W is None:
            return f
        v = self.W @ f
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def cosine_similarity(
        self, a: Union[np.ndarray, List[float]], b: Union[np.ndarray, List[float]]
    ) -> float:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        return float(np.dot(va, vb) / (na * nb)) if na and nb else 0.0
