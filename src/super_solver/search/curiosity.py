"""Curiosity and Anomaly Discovery Engine.

Detects low-density, high-entropy frontiers between known scientific clusters
to autonomously suggest breakthrough hypotheses.
"""

from __future__ import annotations

from typing import List

import numpy as np

from super_solver.core.embeddings import embedding_service


class CuriosityEngine:
    """Exploration bonus and vector frontier anomaly detector."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def compute_frontier_novelty(
        self,
        candidate_vector: np.ndarray,
        known_finding_vectors: List[np.ndarray],
    ) -> float:
        """Computes epistemic novelty: distance to nearest known cluster of findings.

        High novelty means the candidate explores an under-researched latent region.
        """
        if not known_finding_vectors:
            return 1.0

        max_sim = -1.0
        for kv in known_finding_vectors:
            sim = embedding_service.cosine_similarity(candidate_vector, kv)
            if sim > max_sim:
                max_sim = sim

        # Distance = 1.0 - max_similarity
        novelty = max(0.0, min(1.0, 1.0 - max_sim))
        return float(novelty)
