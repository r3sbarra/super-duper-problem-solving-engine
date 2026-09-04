"""Kepner-Tregoe (KT) Problem Analysis Framework.

Establishes strict boundary conditions using the IS / IS NOT matrix across
Identity, Location, Timing, and Extent, projecting a high-margin separating hyperplane
with full dimensional diagnostics.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from super_solver.core.embeddings import embedding_service
from super_solver.core.types import KTBoundary


class KepnerTregoeEngine:
    """Kepner-Tregoe 4-Dimensional Boundary and Root Cause Analyzer."""

    def __init__(self):
        pass

    def build_boundary_hyperplane(self, boundary: KTBoundary) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds positive centroid c_IS, negative centroid c_IS_NOT, and normal vector w."""
        is_text, is_not_text = boundary.to_contrastive_text()
        v_is = embedding_service.encode(is_text)
        v_not = embedding_service.encode(is_not_text)

        w = v_is - v_not
        norm_w = np.linalg.norm(w)
        if norm_w > 0:
            w = w / norm_w
        return v_is, v_not, w

    def evaluate_candidate_cause(
        self,
        candidate_cause: str,
        boundary: KTBoundary,
        leakage_penalty: float = 1.5,
    ) -> Dict[str, Any]:
        """Evaluates how well a candidate cause fits the KT Boundary with 4D breakdown."""
        v_is, v_not, w = self.build_boundary_hyperplane(boundary)
        v_cause = embedding_service.encode(candidate_cause)

        sim_is = embedding_service.cosine_similarity(v_cause, v_is)
        sim_not = embedding_service.cosine_similarity(v_cause, v_not)

        projection = float(np.dot(v_cause, w))
        unwarranted_leakage = max(0.0, sim_not - 0.25)
        net_score = sim_is - (leakage_penalty * unwarranted_leakage)

        # 4-Dimensional breakdown analysis
        dimensions = {
            "identity": (boundary.identity_is, boundary.identity_is_not),
            "location": (boundary.location_is, boundary.location_is_not),
            "timing": (boundary.timing_is, boundary.timing_is_not),
            "extent": (boundary.extent_is, boundary.extent_is_not),
        }

        dim_scores = {}
        violated_dims = []

        for dim_name, (d_is, d_not) in dimensions.items():
            vd_is = embedding_service.encode(d_is)
            vd_not = embedding_service.encode(d_not)
            s_is = embedding_service.cosine_similarity(v_cause, vd_is)
            s_not = embedding_service.cosine_similarity(v_cause, vd_not)
            passed = bool(s_is >= s_not or (s_not < 0.20))
            if not passed:
                violated_dims.append(dim_name)
            dim_scores[dim_name] = {
                "sim_to_is": float(s_is),
                "sim_to_is_not": float(s_not),
                "dimension_passed": passed,
            }

        return {
            "net_score": float(net_score),
            "sim_to_is": float(sim_is),
            "sim_to_is_not": float(sim_not),
            "hyperplane_projection": float(projection),
            "valid_boundary_fit": bool(net_score > 0.15 and sim_is > sim_not and len(violated_dims) <= 1),
            "dimensional_breakdown": dim_scores,
            "violated_dimensions": violated_dims,
        }
