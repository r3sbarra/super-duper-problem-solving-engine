"""Diffusion of Thoughts (DoT inspired, Ye et al. NeurIPS 2024).

Implements iterative latent trajectory optimization with Classifier-Free Guidance (CFG)
heuristics pulling toward goal attractors and repelling from negative dead-end manifolds.

Implementation Note:
This module implements an artificial potential-field optimization in vector space
(combining temporal smoothness, centroid attraction, and inverse-distance dead-end repulsion)
serving as an analytical, zero-daemon analogue of continuous score-based diffusion denoising.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class ThoughtDiffusionRefiner:
    """Iterative latent denoising engine for reasoning trajectories with CFG."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def denoise_trajectory(
        self,
        trajectory: List[np.ndarray],
        constraint_attractors: List[np.ndarray],
        negative_repulsors: Optional[List[np.ndarray]] = None,
        diffusion_steps: int = 4,
        step_size: float = 0.25,
        guidance_scale: float = 1.4,
        repulsion_scale: float = 1.0,
    ) -> List[np.ndarray]:
        """Iteratively refines a latent reasoning trajectory toward constraint attractors
        and away from negative repulsors using Classifier-Free Guidance (CFG).
        """
        if not trajectory:
            return []

        refined = [v.copy() for v in trajectory]
        neg_repulsors = negative_repulsors or []

        for _step in range(diffusion_steps):
            for i in range(len(refined)):
                # 1. Temporal smoothness
                temporal_target = refined[i].copy()
                if i > 0 and i < len(refined) - 1:
                    temporal_target = 0.5 * (refined[i - 1] + refined[i + 1])

                # 2. Positive goal attraction (CFG positive direction)
                attractor_pull = np.zeros(self.dim)
                if constraint_attractors:
                    for att in constraint_attractors:
                        diff = att - refined[i]
                        attractor_pull += diff / len(constraint_attractors)

                # 3. Negative dead-end repulsion (CFG negative direction)
                repulsor_push = np.zeros(self.dim)
                if neg_repulsors:
                    for rep in neg_repulsors:
                        diff_neg = refined[i] - rep
                        dist_sq = float(np.sum(diff_neg**2)) + 1e-4
                        repulsor_push += (diff_neg / dist_sq) / len(neg_repulsors)

                # Net guided update step
                guided_gradient = (
                    0.4 * temporal_target
                    + (guidance_scale * attractor_pull)
                    + (repulsion_scale * repulsor_push)
                ) - refined[i]

                refined[i] = refined[i] + (step_size * guided_gradient)

                norm = np.linalg.norm(refined[i])
                if norm > 0:
                    refined[i] = refined[i] / norm

        return refined

    # ------------------------------------------------------------------
    # Backward / self-consistency pass (self-inspire win on super-solver).
    # The forward refiner is unidirectional; add a reverse pass that
    # reconstructs the premise region a solution likely originated from and
    # compares it to the ACTUAL problem premise, quantifying solution drift
    # / hallucinated assumptions (mirrors drift-tolerant verification).
    # ------------------------------------------------------------------

    def reverse_denoise(
        self,
        solution: np.ndarray,
        constraint_attractors: Optional[List[np.ndarray]] = None,
        negative_repulsors: Optional[List[np.ndarray]] = None,
        diffusion_steps: int = 4,
        step_size: float = 0.25,
    ) -> np.ndarray:
        """Reverse-diffuse: given a solution, estimate the premise region it would
        have needed to produce it by inverting the forward attractor/repulsor field.

        Forward pushed solution toward attractors and away from repulsors, so from a
        solution we SUBTRACT the attractor pull and ADD the repulsor push to walk
        back toward the implied premise. Kept for completeness; self_consistency uses
        the more robust forward-prediction residual.
        """
        attrs = constraint_attractors or []
        reps = negative_repulsors or []
        v = np.array(solution, dtype=np.float64).copy()

        for _ in range(diffusion_steps):
            if attrs:
                pull = np.zeros(self.dim)
                for a in attrs:
                    pull += (a - v) / len(attrs)
                v = v - (step_size * pull)
            if reps:
                push = np.zeros(self.dim)
                for r in reps:
                    diff = v - r
                    dsq = float(np.sum(diff**2)) + 1e-4
                    push += (diff / dsq) / len(reps)
                v = v + (step_size * push)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v

    def _forward_predict(
        self,
        start: np.ndarray,
        constraint_attractors: Optional[List[np.ndarray]] = None,
        negative_repulsors: Optional[List[np.ndarray]] = None,
        diffusion_steps: int = 4,
        step_size: float = 0.25,
    ) -> np.ndarray:
        """Run the forward denoise from a start vector; returns final (unit) vector.
        Mirrors denoise_trajectory on a single point (no temporal neighbor term)."""
        attrs = constraint_attractors or []
        reps = negative_repulsors or []
        v = np.array(start, dtype=np.float64).copy()
        for _ in range(diffusion_steps):
            attractor_pull = np.zeros(self.dim)
            if attrs:
                for a in attrs:
                    attractor_pull += (a - v) / len(attrs)
            repulsor_push = np.zeros(self.dim)
            if reps:
                for r in reps:
                    diff = v - r
                    dsq = float(np.sum(diff**2)) + 1e-4
                    repulsor_push += (diff / dsq) / len(reps)
            grad = (attractor_pull + repulsor_push) - v
            v = v + (step_size * grad)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v

    def self_consistency(
        self,
        solution: np.ndarray,
        problem_premise: np.ndarray,
        constraint_attractors: Optional[List[np.ndarray]] = None,
        negative_repulsors: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Self-consistency via FORWARD-PREDICTION residual.

        A solution is self-consistent iff forward-diffusing the ACTUAL problem
        premise (same attractors/repulsors) lands near the solution. If the premise
        predicts the solution well, it is traceable to the problem; a large residual
        means the solution drifted / hallucinated assumptions the premise can't
        support. This is the verification a forward-only refiner lacks.
        """
        predicted = self._forward_predict(
            problem_premise, constraint_attractors, negative_repulsors
        )
        s_norm = np.linalg.norm(solution)
        p_norm = np.linalg.norm(predicted)
        if s_norm == 0 or p_norm == 0:
            return {"score": 0.0, "drift": 1.0, "verdict": "HALLUCINATED"}
        cos = float(np.dot(solution / s_norm, predicted / p_norm))
        cos = max(-1.0, min(1.0, cos))
        score = max(0.0, cos)
        drift = 1.0 - score
        if score >= 0.75:
            verdict = "CONSISTENT"
        elif score >= 0.45:
            verdict = "DRIFTED"
        else:
            verdict = "HALLUCINATED"
        return {"score": round(score, 4), "drift": round(drift, 4), "verdict": verdict}
