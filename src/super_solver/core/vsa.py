"""Vector Symbolic Architecture (VSA) / Hyperdimensional Computing (HDC) Engine.

Implements algebraic operators over high-dimensional vector representations:
1. Binding (\\otimes) via Circular Convolution / Complex Multiply (Plate HRR / Gayler)
2. Bundling (\\oplus) via Vector Superposition & Normalization
3. Permutation (\\Pi) via Cyclic Coordinate Permutation
4. Cleanup Memory (Associative Codebook) for noise-resistant symbol recovery
5. Analogical Query: A : B :: C : ? solved via structural algebra
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class CleanupMemory:
    """Associative codebook that projects noisy hypervectors back to exact canonical symbols."""

    def __init__(self, vsa_dim: int):
        self.dim = vsa_dim
        self.codebook: Dict[str, np.ndarray] = {}

    def register(self, symbol: str, vector: np.ndarray):
        """Registers a canonical symbol hypervector."""
        norm = np.linalg.norm(vector)
        self.codebook[symbol] = vector / (norm + 1e-12) if norm > 0 else vector

    def clean(self, noisy_vector: np.ndarray, min_similarity: float = 0.20) -> Tuple[Optional[str], float]:
        """Finds the nearest canonical symbol in the codebook."""
        if not self.codebook:
            return None, 0.0

        best_symbol = None
        best_sim = -1.0
        norm_n = np.linalg.norm(noisy_vector)
        if norm_n == 0:
            return None, 0.0
        unit_n = noisy_vector / norm_n

        for sym, vec in self.codebook.items():
            sim = float(np.dot(unit_n, vec))
            if sim > best_sim:
                best_sim = sim
                best_symbol = sym

        if best_sim >= min_similarity:
            return best_symbol, best_sim
        return None, best_sim


class VSAEngine:
    """Hyperdimensional Computing / Vector Symbolic Architecture algebra engine."""

    def __init__(self, dim: int = 2048, seed: Optional[int] = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.cleanup = CleanupMemory(vsa_dim=dim)

    def random_hypervector(self, symbol: Optional[str] = None) -> np.ndarray:
        """Generates a random zero-mean, unit-norm hypervector."""
        v = self.rng.normal(0, 1.0, size=self.dim)
        unit_v = v / (np.linalg.norm(v) + 1e-12)
        if symbol:
            self.cleanup.register(symbol, unit_v)
        return unit_v

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Binding operator (\\otimes) via Circular Convolution using FFT."""
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        conv = np.fft.irfft(fa * fb, n=self.dim)
        norm = np.linalg.norm(conv)
        return conv / (norm + 1e-12) if norm > 0 else conv

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Unbinding operator: retrieves filler given bound vector and role key."""
        inv_key = np.zeros_like(key)
        inv_key[0] = key[0]
        inv_key[1:] = key[:0:-1]
        return self.bind(bound, inv_key)

    def bundle(self, vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """Bundling operator (\\oplus) via weighted superposition."""
        if not vectors:
            return np.zeros(self.dim)
        if weights is None:
            weights = [1.0] * len(vectors)

        acc = np.zeros(self.dim)
        for w, v in zip(weights, vectors):
            acc += w * v
        norm = np.linalg.norm(acc)
        return acc / (norm + 1e-12) if norm > 0 else acc

    def permute(self, v: np.ndarray, shifts: int = 1) -> np.ndarray:
        """Permutation operator (\\Pi^k): encodes sequence or structural hierarchy."""
        return np.roll(v, shifts)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two hypervectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def solve_analogy(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Solves analogical proportion A : B :: C : ?
        
        Relation R = B \\otimes A^(-1)
        Target D = R \\otimes C
        """
        rel = self.unbind(b, a)
        d = self.bind(rel, c)
        return d
