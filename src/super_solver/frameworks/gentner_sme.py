"""Dedre Gentner's Structure-Mapping Engine (SME) for Analogical Problem Solving.

Discovers deep relational isomorphisms between disparate scientific domains,
aligning relational structures while abstracting away surface entities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel
import numpy as np

from super_solver.core.vsa import VSAEngine
from super_solver.core.embeddings import embedding_service


class RelationalStatement(BaseModel):
    """A higher-order relational proposition: Relation(EntityA, EntityB)."""
    relation: str
    entity_a: str
    entity_b: str
    domain: str


class StructureMappingEngine:
    """Computational analogical reasoning engine based on Gentner's SME."""

    def __init__(self, vsa_dim: int = 2048):
        self.vsa = VSAEngine(dim=vsa_dim)
        self.symbol_table: Dict[str, np.ndarray] = {}

    def _get_symbol_vector(self, name: str) -> np.ndarray:
        if name not in self.symbol_table:
            seed = int(name.encode("utf-8").hex()[:8], 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.normal(0, 1.0, size=self.vsa.dim)
            self.symbol_table[name] = v / (np.linalg.norm(v) + 1e-12)
        return self.symbol_table[name]

    def align_systems(
        self,
        base_domain_relations: List[RelationalStatement],
        target_domain_relations: List[RelationalStatement],
    ) -> Dict[str, Any]:
        """Aligns relational structures between base and target domains (Gentner SME).
        
        Evaluates relational correspondence and induces entity bindings.
        """
        if not base_domain_relations or not target_domain_relations:
            return {"structural_alignment_score": 0.0, "matches": [], "isomorphic_transfer_viable": False}

        matches = []
        entity_mappings: Dict[str, str] = {}
        relational_scores = []

        for b_stmt in base_domain_relations:
            v_b_rel = self._get_symbol_vector(b_stmt.relation)
            best_target = None
            best_sim = -1.0

            for t_stmt in target_domain_relations:
                v_t_rel = self._get_symbol_vector(t_stmt.relation)
                # Compare relational predicates (identical relations have sim=1.0)
                sim = self.vsa.similarity(v_b_rel, v_t_rel)
                if sim > best_sim:
                    best_sim = sim
                    best_target = t_stmt

            if best_target and best_sim > 0.40:
                relational_scores.append(best_sim)
                matches.append({
                    "base_relation": f"{b_stmt.relation}({b_stmt.entity_a}, {b_stmt.entity_b})",
                    "target_relation": f"{best_target.relation}({best_target.entity_a}, {best_target.entity_b})",
                    "relational_match_score": float(best_sim),
                })
                # Induce entity correspondences
                entity_mappings[b_stmt.entity_a] = best_target.entity_a
                entity_mappings[b_stmt.entity_b] = best_target.entity_b

        avg_score = sum(relational_scores) / len(relational_scores) if relational_scores else 0.0
        return {
            "structural_alignment_score": float(avg_score),
            "systematic_match_count": len(matches),
            "entity_mappings": entity_mappings,
            "matches": matches,
            "isomorphic_transfer_viable": bool(avg_score > 0.35 and len(matches) >= 1),
        }
