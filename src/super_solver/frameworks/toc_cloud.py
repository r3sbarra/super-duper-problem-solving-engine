"""Theory of Constraints (TOC) & Goldratt's Evaporating Cloud Conflict Resolver."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import numpy as np

from super_solver.core.embeddings import embedding_service


class EvaporatingCloud(BaseModel):
    objective: str
    requirement_a: str
    requirement_b: str
    prerequisite_d: str
    prerequisite_d_prime: str
    underlying_assumptions: List[str] = Field(default_factory=list)


class TOCEngine:
    """Theory of Constraints bottleneck analyzer & Evaporating Cloud resolver."""

    def __init__(self):
        pass

    def formulate_cloud(
        self,
        objective: str,
        requirement_a: str,
        requirement_b: str,
        prerequisite_d: str,
        prerequisite_d_prime: str,
        assumptions: Optional[List[str]] = None,
    ) -> EvaporatingCloud:
        return EvaporatingCloud(
            objective=objective,
            requirement_a=requirement_a,
            requirement_b=requirement_b,
            prerequisite_d=prerequisite_d,
            prerequisite_d_prime=prerequisite_d_prime,
            underlying_assumptions=assumptions or [],
        )

    def evaporate(
        self,
        cloud: EvaporatingCloud,
        candidate_injections: List[str],
    ) -> Dict[str, Any]:
        """Evaluates candidate injections to resolve the conflict between Requirements A and B.
        
        A valid injection must satisfy BOTH Requirement A and Requirement B simultaneously,
        penalizing one-sided collapse into either extreme.
        """
        v_a = embedding_service.encode(cloud.requirement_a)
        v_b = embedding_service.encode(cloud.requirement_b)
        v_obj = embedding_service.encode(cloud.objective)

        best_injection = None
        best_score = -float("inf")
        rankings = []

        for inj in candidate_injections:
            v_inj = embedding_service.encode(inj)
            sim_a = embedding_service.cosine_similarity(v_inj, v_a)
            sim_b = embedding_service.cosine_similarity(v_inj, v_b)
            sim_obj = embedding_service.cosine_similarity(v_inj, v_obj)

            # Balanced synthesis: requires both requirements, rewards objective alignment
            harmonic_min = min(sim_a, sim_b)
            score = (1.2 * harmonic_min) + (0.4 * (sim_a + sim_b)) + (0.3 * sim_obj)

            rankings.append({
                "injection": inj,
                "score": float(score),
                "sim_to_requirement_a": float(sim_a),
                "sim_to_requirement_b": float(sim_b),
                "sim_to_objective": float(sim_obj),
            })

            if score > best_score:
                best_score = score
                best_injection = inj

        rankings.sort(key=lambda x: x["score"], reverse=True)
        return {
            "resolved": bool(best_score > 0.05),
            "winning_injection": best_injection,
            "evaporation_score": float(best_score),
            "rankings": rankings,
        }
