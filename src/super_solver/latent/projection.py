"""Dual-Representation Projection Engine.

Projects continuous latent thought vectors back into verifiable symbolic checkpoints,
hypotheses, and natural language statements for scientific auditability.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from super_solver.core.embeddings import embedding_service


class SymbolicProjector:
    """Projects latent vectors down to discrete concepts and human-interpretable claims."""

    def __init__(self):
        pass

    def project_to_nearest_claim(
        self,
        latent_vector: np.ndarray,
        candidate_claims: List[str],
    ) -> Tuple[str, float]:
        """Finds the most semantically aligned discrete claim for a latent state vector."""
        if not candidate_claims:
            return "", 0.0

        best_claim = candidate_claims[0]
        best_sim = -1.0

        for claim in candidate_claims:
            v_claim = embedding_service.encode(claim)
            sim = embedding_service.cosine_similarity(latent_vector, v_claim)
            if sim > best_sim:
                best_sim = sim
                best_claim = claim

        return best_claim, float(best_sim)

    def generate_checkpoint_summary(
        self,
        step_index: int,
        operator_name: str,
        latent_vector: np.ndarray,
        reference_concepts: Dict[str, str],
    ) -> str:
        """Constructs an auditable checkpoint text from latent state and operator."""
        nearest_concept = "General Reasoning Space"
        best_sim = -1.0

        for concept_name, concept_desc in reference_concepts.items():
            v_c = embedding_service.encode(f"{concept_name}: {concept_desc}")
            sim = embedding_service.cosine_similarity(latent_vector, v_c)
            if sim > best_sim:
                best_sim = sim
                nearest_concept = concept_name

        summary = f"Step {step_index} [{operator_name}]: Aligned with '{nearest_concept}' (latent coherence: {best_sim:.2f})"
        return summary
