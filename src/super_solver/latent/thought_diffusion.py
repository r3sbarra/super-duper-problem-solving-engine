"""Diffusion of Thoughts (DoT inspired, Ye et al. NeurIPS 2024).

Implements iterative latent trajectory optimization with Classifier-Free Guidance (CFG)
heuristics pulling toward goal attractors and repelling from negative dead-end manifolds.

Implementation Note:
This module implements an artificial potential-field optimization in vector space
(combining temporal smoothness, centroid attraction, and inverse-distance dead-end repulsion)
serving as an analytical, zero-daemon analogue of continuous score-based diffusion denoising.
"""

from __future__ import annotations

from typing import List, Optional

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
