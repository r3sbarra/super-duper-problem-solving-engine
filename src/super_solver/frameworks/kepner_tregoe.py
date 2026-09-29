"""Kepner-Tregoe (KT) Problem Analysis Framework.

Establishes strict boundary conditions using the IS / IS NOT matrix across
Identity, Location, Timing, and Extent, projecting a high-margin separating hyperplane
with full dimensional diagnostics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from super_solver.core.embeddings import embedding_service
from super_solver.core.types import KTBoundary


class KepnerTregoeEngine:
    """Kepner-Tregoe 4-Dimensional Boundary and Root Cause Analyzer."""

    def __init__(self):
        pass

    def build_boundary_hyperplane(
        self, boundary: KTBoundary
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            "valid_boundary_fit": bool(
                net_score > 0.15 and sim_is > sim_not and len(violated_dims) <= 1
            ),
            "dimensional_breakdown": dim_scores,
            "violated_dimensions": violated_dims,
        }

    def generate_candidate_causes(
        self,
        is_facts: List[str],
        is_not_facts: List[str],
        max_candidates: int = 6,
    ) -> List[str]:
        """Autonomously propose candidate root causes from an IS / IS-NOT fact set.

        A valid root cause must EXPLAIN what IS and be EXCLUDED by what IS-NOT.
        The distinguishing signal lives in the contrast between the two lists:
        content present in IS but absent from IS-NOT (and vice versa) is what a
        cause must account for. This generates candidates by pairing the
        distinctive IS content with the distinctive IS-NOT content, so each
        candidate names a factor that is present in the IS regime and absent in
        the IS-NOT regime (the shape of a real cause).

        Deterministic and offline; no LLM required.
        """
        from super_solver.frameworks.problem_identification import (
            _content_tokens,
        )

        is_text = " ; ".join(is_facts).lower()
        not_text = " ; ".join(is_not_facts).lower()
        is_tokens = {t for t in _content_tokens(is_text)}
        not_tokens = {t for t in _content_tokens(not_text)}
        # Drop generic process verbs that are not cause FACTORS (e.g. 'happens',
        # 'sparks', 'occurs') — a root cause is a noun-like condition, not the
        # event verb describing the symptom.
        _NON_FACTOR = {
            "happens", "happen", "occurs", "occur", "sparks", "spark",
            "appears", "appear", "shows", "show", "seems", "seem",
            "sometimes", "often", "always", "never", "only", "just",
        }
        # Tokens unique to the IS regime are what a cause must explain;
        # tokens unique to the IS-NOT regime are what a cause must NOT invoke.
        distinguishing = sorted(
            t for t in (is_tokens - not_tokens) if t not in _NON_FACTOR
        )
        excluded = sorted(
            t for t in (not_tokens - is_tokens) if t not in _NON_FACTOR
        )

        candidates: List[str] = []
        # 1. Single distinguishing factors (present in IS, absent in IS-NOT).
        for tok in distinguishing:
            candidates.append(
                f"a cause tied to '{tok}' (present in the IS regime, absent in IS-NOT)"
            )
            if len(candidates) >= max_candidates:
                return candidates
        # 2. Pairwise factor-interaction candidates.
        for i in range(len(distinguishing)):
            for j in range(i + 1, len(distinguishing)):
                candidates.append(
                    f"an interaction between '{distinguishing[i]}' and "
                    f"'{distinguishing[j]}' specific to the IS regime"
                )
                if len(candidates) >= max_candidates:
                    return candidates
        # 3. Contrastive candidates ruling out the excluded regime.
        if not candidates and excluded:
            candidates.append(
                f"'{excluded[0]}' is ruled out (appears in IS-NOT too)"
            )
        return candidates[:max_candidates]

    def analyze_boundary(
        self,
        is_facts: List[str],
        is_not_facts: List[str],
        candidate_causes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyzes an IS / IS-NOT fact list, builds hyperplane, evaluates causes.

        When ``candidate_causes`` is not supplied, candidate root causes are
        GENERATED autonomously from the contrast between the IS and IS-NOT facts
        (see ``generate_candidate_causes``), then ranked by boundary fit, so the
        analysis is never empty. Returns the ranked candidates plus a
        recommended leading cause.
        """
        is_text = " IS: " + "; ".join(is_facts)
        is_not_text = " IS NOT: " + "; ".join(is_not_facts)
        v_is = embedding_service.encode(is_text)
        v_not = embedding_service.encode(is_not_text)

        w = v_is - v_not
        norm_w = float(np.linalg.norm(w))
        if norm_w > 0:
            w = w / norm_w

        generated = False
        causes = list(candidate_causes or [])
        if not causes:
            causes = self.generate_candidate_causes(is_facts, is_not_facts)
            generated = True

        cause_evals = []
        for cause in causes:
            vc = embedding_service.encode(cause)
            sim_is = float(embedding_service.cosine_similarity(vc, v_is))
            sim_not = float(embedding_service.cosine_similarity(vc, v_not))
            proj = float(np.dot(vc, w))
            cause_evals.append({
                "cause": cause,
                "sim_to_is": sim_is,
                "sim_to_is_not": sim_not,
                "hyperplane_projection": proj,
                "margin": float(sim_is - sim_not),
                "is_consistent": bool(sim_is > sim_not and proj > 0),
            })

        # Rank: consistent causes first, then by projection margin.
        cause_evals.sort(
            key=lambda c: (c["is_consistent"], c["margin"]), reverse=True
        )
        leading = cause_evals[0] if cause_evals else None

        return {
            "is_facts_count": len(is_facts),
            "is_not_facts_count": len(is_not_facts),
            "hyperplane_norm": norm_w,
            "candidate_evaluations": cause_evals,
            "causes_generated_autonomously": generated,
            "leading_cause": leading["cause"] if leading else None,
            "leading_cause_consistent": bool(leading["is_consistent"]) if leading else False,
        }
