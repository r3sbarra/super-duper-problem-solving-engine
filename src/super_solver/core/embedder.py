"""Pluggable custom embedder for super-solver.

Provides a single ``Embedder`` interface with multiple backends so problems,
solutions, and discovery paths can be encoded into dense vectors:

- ``polarity`` (default): the existing deterministic 384d
  :class:`PolarityAwareEmbeddingService` (no network, reproducible).
- ``ollama``: local Ollama ``nomic-embed-text`` (or any configured model) via
  HTTP. Higher quality semantic vectors; requires a running Ollama.
- ``hybrid``: concatenate polarity + ollama (when available) for a richer
  representation, falling back to polarity-only if Ollama is unreachable.

The active backend is selected by ``SUPER_SOLVER_EMBEDDER`` (default
``polarity``). All backends expose the same ``encode(text) -> np.ndarray`` and
``cosine_similarity(a, b) -> float`` API, so the rest of the engine is
backend-agnostic.
"""

from __future__ import annotations

import os
from typing import List, Optional, Union

import numpy as np

from super_solver.core.embeddings import PolarityAwareEmbeddingService, embedding_service

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


class EmbedderBackend:
    """Base protocol for an embedder backend."""

    name: str = "base"
    dim: int = 0

    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def cosine_similarity(
        self, a: Union[np.ndarray, List[float]], b: Union[np.ndarray, List[float]]
    ) -> float:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))


class PolarityBackend(EmbedderBackend):
    """Deterministic 384d polarity-aware hashing embedder (no network)."""

    name = "polarity"
    dim = 384

    def __init__(self, service: Optional[PolarityAwareEmbeddingService] = None):
        self._service = service or embedding_service

    def encode(self, text: str) -> np.ndarray:
        return self._service.encode(text)


class OllamaBackend(EmbedderBackend):
    """Local Ollama embedder via HTTP (nomic-embed-text by default).

    Falls back to a deterministic hashing vector if Ollama is unreachable so
    the engine never hard-fails on a missing daemon.
    """

    name = "ollama"
    dim = 768  # nomic-embed-text default; overridden by the model's actual dim

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 5.0,
    ):
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.model = model or os.getenv("SUPER_SOLVER_OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.timeout = timeout
        self._fallback = PolarityBackend()
        self._available: Optional[bool] = None

    def _probe(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import httpx

            resp = httpx.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def encode(self, text: str) -> np.ndarray:
        if not self._probe():
            return self._fallback.encode(text)
        try:
            import httpx

            resp = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embedding")
            if not emb:
                return self._fallback.encode(text)
            vec = np.asarray(emb, dtype=np.float32)
            self.dim = vec.shape[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        except Exception:
            return self._fallback.encode(text)


class HybridBackend(EmbedderBackend):
    """Concatenate polarity + ollama vectors for a richer representation.

    Falls back to polarity-only when Ollama is unreachable.
    """

    name = "hybrid"

    def __init__(self):
        self._polarity = PolarityBackend()
        self._ollama = OllamaBackend()
        self.dim = self._polarity.dim + self._ollama.dim

    def encode(self, text: str) -> np.ndarray:
        vp = self._polarity.encode(text)
        vo = self._ollama.encode(text)
        vec = np.concatenate([vp, vo]).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


_BACKENDS: dict[str, type[EmbedderBackend]] = {
    "polarity": PolarityBackend,
    "ollama": OllamaBackend,
    "hybrid": HybridBackend,
}


def get_backend(name: Optional[str] = None) -> EmbedderBackend:
    """Resolve an embedder backend by name (env ``SUPER_SOLVER_EMBEDDER``)."""
    key = (name or os.getenv("SUPER_SOLVER_EMBEDDER", "polarity")).lower()
    cls = _BACKENDS.get(key, PolarityBackend)
    return cls()


# ---------------------------------------------------------------------------
# Typed vector encoders for problems, solutions, and paths
# ---------------------------------------------------------------------------


class ProblemVectorizer:
    """Encodes a problem specification (plus optional boundary/goal) into a vector."""

    def __init__(self, backend: Optional[EmbedderBackend] = None):
        self.backend = backend or get_backend()

    def encode(
        self,
        specification: str,
        title: str = "",
        goal_criteria: Optional[List[str]] = None,
        boundary_is: Optional[List[str]] = None,
        boundary_is_not: Optional[List[str]] = None,
    ) -> np.ndarray:
        parts = []
        if title:
            parts.append(f"PROBLEM: {title}")
        parts.append(f"SPEC: {specification}")
        if goal_criteria:
            parts.append("GOAL: " + " ".join(goal_criteria))
        if boundary_is:
            parts.append("IS: " + " ".join(boundary_is))
        if boundary_is_not:
            parts.append("IS_NOT: " + " ".join(boundary_is_not))
        return self.backend.encode("\n".join(parts))


class SolutionVectorizer:
    """Encodes a solution / breakthrough into a vector."""

    def __init__(self, backend: Optional[EmbedderBackend] = None):
        self.backend = backend or get_backend()

    def encode(
        self,
        solution_text: str,
        method: str = "",
        domain: str = "",
        operators: Optional[List[str]] = None,
    ) -> np.ndarray:
        parts = []
        if domain:
            parts.append(f"DOMAIN: {domain}")
        if method:
            parts.append(f"METHOD: {method}")
        parts.append(f"SOLUTION: {solution_text}")
        if operators:
            parts.append("OPERATORS: " + ", ".join(operators))
        return self.backend.encode("\n".join(parts))


class PathVectorizer:
    """Encodes a discovery path (sequence of reasoning steps) into a vector.

    Produces a mean-pooled vector over the step vectors plus a structural
    component capturing the operator-type sequence, so two paths that reach
    the same place via different operators are distinguishable.
    """

    def __init__(self, backend: Optional[EmbedderBackend] = None):
        self.backend = backend or get_backend()

    def encode(
        self,
        steps: List[str],
        operator_types: Optional[List[str]] = None,
        final_breakthrough: str = "",
    ) -> np.ndarray:
        if not steps:
            return self.backend.encode(final_breakthrough or "empty path")
        step_vecs = np.stack([self.backend.encode(s) for s in steps])
        mean_vec = step_vecs.mean(axis=0)
        # Structural component: encode the operator-type sequence as a string
        if operator_types:
            struct = self.backend.encode(" -> ".join(operator_types))
            # Blend: 70% content mean, 30% structure
            vec = 0.7 * mean_vec + 0.3 * struct
        else:
            vec = mean_vec
        if final_breakthrough:
            fb = self.backend.encode("BREAKTHROUGH: " + final_breakthrough)
            vec = 0.8 * vec + 0.2 * fb
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# Singleton default vectorizers
problem_vectorizer = ProblemVectorizer()
solution_vectorizer = SolutionVectorizer()
path_vectorizer = PathVectorizer()
