"""Theory of Constraints (TOC) & Goldratt's Evaporating Cloud Conflict Resolver."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from super_solver.core.embeddings import embedding_service

# Thresholds for the conflict-engagement screen in TOCEngine.evaporate.
# Calibrated on the worked example in the docstring below; they are deliberately
# coarse because the underlying signal (hashed bag-of-words cosine) is coarse.
_POLE_ENGAGE_FLOOR = 0.05   # below this to BOTH prerequisites == alien idea
_POLE_RESTATE_CEIL = 0.40   # above this to ONE prerequisite == restating a pole
_POLE_RESTATE_RATIO = 0.5   # ...while being this much weaker on the other pole


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
        # The two conflicting means. The original score looked only at the two
        # *requirements*, so nothing anchored a candidate to the actual conflict,
        # and the winner was decided by hashed-cosine noise (~0.01-0.1 across the
        # board). Measured on the cloud below, "Use a completely unrelated
        # cooking recipe with basil and tomatoes" won at 0.096 over the genuine
        # CI/CD injection at 0.047.
        #
        # The prerequisites do carry a usable signal, though (measured):
        #   cooking recipe  -> sim(D)=0.034, sim(D')=0.007   engages NEITHER pole
        #   freeze dev      -> sim(D)=0.088, sim(D')=0.496   RESTATES one pole
        #   CI/CD pipeline  -> sim(D)=0.171, sim(D')=0.008   engages a pole
        #   decoupling plan -> sim(D)=0.126, sim(D')=0.240   engages both
        # So a candidate is screened on whether it engages the conflict at all:
        # an injection touching neither conflicting means is about something else
        # entirely, and one dominated by a single pole is a collapse, not a
        # resolution -- exactly the one-sided outcome the evaporating cloud is
        # meant to dissolve.
        v_d = embedding_service.encode(cloud.prerequisite_d)
        v_dp = embedding_service.encode(cloud.prerequisite_d_prime)

        best_injection = None
        best_score = -float("inf")
        rankings = []

        for inj in candidate_injections:
            v_inj = embedding_service.encode(inj)
            sim_a = embedding_service.cosine_similarity(v_inj, v_a)
            sim_b = embedding_service.cosine_similarity(v_inj, v_b)
            sim_obj = embedding_service.cosine_similarity(v_inj, v_obj)
            sim_d = embedding_service.cosine_similarity(v_inj, v_d)
            sim_dp = embedding_service.cosine_similarity(v_inj, v_dp)

            # Balanced synthesis: requires both requirements, rewards objective alignment
            harmonic_min = min(sim_a, sim_b)
            score = (1.2 * harmonic_min) + (0.4 * (sim_a + sim_b)) + (0.3 * sim_obj)

            pole_max, pole_min = max(sim_d, sim_dp), min(sim_d, sim_dp)
            engages_conflict = pole_max >= _POLE_ENGAGE_FLOOR
            restates_pole = (
                pole_max >= _POLE_RESTATE_CEIL
                and pole_min < _POLE_RESTATE_RATIO * pole_max
            )
            eligible = engages_conflict and not restates_pole

            rankings.append(
                {
                    "injection": inj,
                    "score": float(score),
                    "eligible": eligible,
                    "engages_conflict": engages_conflict,
                    "restates_pole": restates_pole,
                    "sim_to_prerequisite_d": float(sim_d),
                    "sim_to_prerequisite_d_prime": float(sim_dp),
                    "sim_to_requirement_a": float(sim_a),
                    "sim_to_requirement_b": float(sim_b),
                    "sim_to_objective": float(sim_obj),
                }
            )

            if eligible and score > best_score:
                best_score = score
                best_injection = inj

        rankings.sort(key=lambda x: x["score"], reverse=True)
        return {
            # "Resolved" now means: a candidate survived the conflict screen. The
            # old `score > 0.05` absolute floor was measured against a noise
            # distribution, so it accepted noise and rejected real answers
            # depending on where the cloud's wording happened to land.
            "resolved": best_injection is not None,
            "winning_injection": best_injection,
            "evaporation_score": float(best_score) if best_injection is not None else 0.0,
            "rankings": rankings,
        }
