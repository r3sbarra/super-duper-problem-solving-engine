"""Negative Manifold Repulsor (NMR).

Models verified dead ends and falsified hypotheses as repulsive potential fields
in vector space, mathematically deflecting active search trajectories away from failure modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from super_solver.core.embeddings import embedding_service

if TYPE_CHECKING:
    from super_solver.memory.episodic_store import EpisodicVectorStore


class NegativeManifoldRepulsor:
    """Repulsive potential field engine over dead-end hazard manifolds."""

    PROXIMITY_ALERT_THRESHOLD = 0.48

    def __init__(
        self,
        repulsor_weight: float = 0.5,
        epsilon: float = 1e-4,
        store: Optional[EpisodicVectorStore] = None,
    ):
        self.repulsor_weight = repulsor_weight
        self.epsilon = epsilon
        self.store = store
        self.dead_ends: List[np.ndarray] = []
        self.dead_end_descriptions: List[str] = []

        if self.store is not None:
            loaded = self.store.load_dead_ends()
            for desc, vec in loaded:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self.dead_ends.append(vec)
                self.dead_end_descriptions.append(desc)

    def register_dead_end(self, description: str, vector: Optional[np.ndarray] = None) -> np.ndarray:
        """Registers a verified dead end or falsified trajectory."""
        if vector is None:
            vector = embedding_service.encode(description)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        self.dead_ends.append(vector)
        self.dead_end_descriptions.append(description)

        if self.store is not None:
            self.store.insert_dead_end(description=description, vector=vector)

        return vector

    def check_proximity(
        self,
        state_vector: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, Optional[str]]:
        """Checks if a state vector is within the hazard radius of any known dead end."""
        if not self.dead_ends:
            return False, 0.0, None

        thresh = threshold if threshold is not None else self.PROXIMITY_ALERT_THRESHOLD
        max_sim = -1.0
        closest_desc = None

        for d_vec, desc in zip(self.dead_ends, self.dead_end_descriptions, strict=False):
            sim = embedding_service.cosine_similarity(state_vector, d_vec)
            if sim > max_sim:
                max_sim = sim
                closest_desc = desc

        is_near = max_sim >= thresh
        return is_near, float(max_sim), closest_desc

    def compute_repulsive_force(self, state_vector: np.ndarray) -> np.ndarray:
        """Computes the net repulsive gradient vector steering state away from dead ends."""
        if not self.dead_ends:
            return np.zeros_like(state_vector)

        f_net = np.zeros_like(state_vector)
        for d_vec in self.dead_ends:
            diff = state_vector - d_vec
            dist_sq = float(np.sum(diff**2)) + self.epsilon
            magnitude = self.repulsor_weight / dist_sq
            f_net += magnitude * diff

        norm = np.linalg.norm(f_net)
        if norm > 0:
            f_net = f_net / norm
        return f_net

    def deflect_trajectory(
        self,
        candidate_vector: np.ndarray,
        goal_vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Deflects a candidate trajectory step away from toxic dead ends."""
        is_near, max_sim, desc = self.check_proximity(candidate_vector)
        if not is_near:
            return candidate_vector

        f_rep = self.compute_repulsive_force(candidate_vector)
        deflected = candidate_vector + (0.45 * f_rep)

        if goal_vector is not None:
            deflected = deflected + (0.35 * goal_vector)

        norm = np.linalg.norm(deflected)
        if norm > 0:
            deflected = deflected / norm
        return deflected

    def consolidate_memory(self, similarity_threshold: float = 0.75) -> int:
        """Consolidates clusters of near-duplicate dead ends into unified centroid repulsors (synaptic consolidation)."""
        if len(self.dead_ends) <= 1:
            return 0

        initial_count = len(self.dead_ends)
        consolidated_vecs: List[np.ndarray] = []
        consolidated_descs: List[str] = []
        merged_indices = set()

        for i in range(len(self.dead_ends)):
            if i in merged_indices:
                continue

            cluster = [self.dead_ends[i]]
            cluster_descs = [self.dead_end_descriptions[i]]
            merged_indices.add(i)

            for j in range(i + 1, len(self.dead_ends)):
                if j in merged_indices:
                    continue
                sim = embedding_service.cosine_similarity(self.dead_ends[i], self.dead_ends[j])
                if sim >= similarity_threshold:
                    cluster.append(self.dead_ends[j])
                    cluster_descs.append(self.dead_end_descriptions[j])
                    merged_indices.add(j)

            # Compute normalized centroid
            centroid = np.mean(cluster, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            if len(cluster) > 1:
                desc = f"Consolidated Hazard Cluster ({len(cluster)} items): {cluster_descs[0]}"
            else:
                desc = cluster_descs[0]

            consolidated_vecs.append(centroid)
            consolidated_descs.append(desc)

        self.dead_ends = consolidated_vecs
        self.dead_end_descriptions = consolidated_descs

        if self.store is not None:
            self.store.sync_dead_ends(list(zip(self.dead_end_descriptions, self.dead_ends, strict=False)))

        pruned_count = initial_count - len(self.dead_ends)
        return pruned_count

