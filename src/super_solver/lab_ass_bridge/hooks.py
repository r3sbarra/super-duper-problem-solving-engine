"""Autonomous hooks and adapters connecting SuperDuperProblemSolvingEngine to lab-ass."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.core.types import KTBoundary, CrucialExperiment, Hypothesis


class LabAssSuperSolverBridge:
    """Provides advanced problem-solving capabilities directly to lab-ass sessions."""

    def __init__(self, engine: Optional[SuperDuperProblemSolvingEngine] = None):
        self.engine = engine or SuperDuperProblemSolvingEngine()

    def decompose_topic_with_polya_and_triz(
        self,
        topic: str,
        initial_subproblems: List[str],
    ) -> Dict[str, Any]:
        """Decomposes a difficult research topic into auxiliary angles using Polya & TRIZ."""
        aux_angles = self.engine.polya.decompose_auxiliary_problems(
            problem_specification=topic,
            known_subproblems=initial_subproblems,
        )
        principles = self.engine.triz.suggest_principles(topic, top_k=3)

        return {
            "auxiliary_angles": aux_angles,
            "suggested_triz_operators": principles,
        }

    def check_dead_end_deflection(
        self,
        proposed_content: str,
        known_dead_ends: List[str],
    ) -> Dict[str, Any]:
        """Checks if a proposed finding or angle approaches a dead end and computes deflection."""
        from super_solver.core.embeddings import embedding_service
        import numpy as np

        rep = self.engine.repulsor
        for de in known_dead_ends:
            rep.register_dead_end(de)

        v_prop = embedding_service.encode(proposed_content)
        is_near, max_sim, closest = rep.check_proximity(v_prop)

        deflected_text = None
        if is_near:
            v_def = rep.deflect_trajectory(v_prop)
            deflected_text = "Vector trajectory deflected away from toxic dead end."

        return {
            "hazard_detected": is_near,
            "max_similarity_to_dead_end": float(max_sim),
            "closest_dead_end": closest,
            "deflected": bool(is_near),
        }
