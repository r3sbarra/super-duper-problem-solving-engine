"""Dedre Gentner's Structure-Mapping Engine (SME) for Analogical Problem Solving.

Discovers deep relational isomorphisms between disparate scientific domains,
aligning relational structures while abstracting away surface entities.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel

from super_solver.core.vsa import VSAEngine


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
            return {
                "structural_alignment_score": 0.0,
                "matches": [],
                "isomorphic_transfer_viable": False,
            }

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
                matches.append(
                    {
                        "base_relation": f"{b_stmt.relation}({b_stmt.entity_a}, {b_stmt.entity_b})",
                        "target_relation": f"{best_target.relation}({best_target.entity_a}, {best_target.entity_b})",
                        "relational_match_score": float(best_sim),
                    }
                )
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

    def retrieve_analogs(
        self,
        specification: str,
        candidate_systems: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieves and ranks structural analogies for a target problem specification.

        Extracts relational/functional primitives from the target and evaluates alignment
        against candidate or canonical base systems.
        """
        if candidate_systems is not None:
            results = []
            target_relations = self._extract_relations(specification, domain="target")
            for cand in candidate_systems:
                if isinstance(cand, str):
                    cand_name = cand
                    cand_domain = "General"
                    cand_text = cand
                    base_stmts = self._extract_relations(cand_text, domain=cand_domain)
                elif isinstance(cand, dict):
                    cand_name = cand.get("name", "Unknown System")
                    cand_domain = cand.get("domain", "General")
                    base_relations = cand.get("relations", [])
                    if isinstance(base_relations, list) and base_relations and isinstance(base_relations[0], dict):
                        base_stmts = [RelationalStatement(**r) for r in base_relations]
                    elif not base_relations and "description" in cand:
                        base_stmts = self._extract_relations(cand["description"], domain=cand_domain)
                    else:
                        base_stmts = base_relations
                else:
                    continue

                alignment = self.align_systems(base_stmts, target_relations)
                score = alignment["structural_alignment_score"]
                if score == 0.0:
                    v_target = embedding_service.encode(specification)
                    v_cand = embedding_service.encode(cand if isinstance(cand, str) else cand.get("description", cand_name))
                    score = float(embedding_service.cosine_similarity(v_target, v_cand))

                if score >= min_score:
                    results.append({
                        "name": cand_name,
                        "candidate": cand if isinstance(cand, str) else cand_name,
                        "domain": cand_domain,
                        "score": score,
                        "alignment_score": score,
                        "entity_mappings": alignment.get("entity_mappings", {}),
                        "matches": alignment.get("matches", []),
                        "transfer_viable": alignment.get("isomorphic_transfer_viable", False),
                    })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        # Default fallback: use vectorization service primitive analogize if available
        try:
            from super_solver.vectorize import vectorization_service
            return vectorization_service.primitive_analogize(specification, top_k=top_k, min_overlap=min_score)
        except Exception:
            return []

    def _extract_relations(self, text: str, domain: str = "general") -> List[RelationalStatement]:
        """Heuristically extracts relational statements from text."""
        tokens = text.lower().split()
        relations = []
        verbs = ["attracts", "orbits", "causes", "regulates", "contains", "flows_through", "grips", "blocks", "reduces"]
        for i, word in enumerate(tokens):
            for verb in verbs:
                if verb in word and i > 0 and i < len(tokens) - 1:
                    relations.append(
                        RelationalStatement(
                            relation=verb,
                            entity_a=tokens[i - 1],
                            entity_b=tokens[i + 1],
                            domain=domain,
                        )
                    )
        if not relations:
            # Fallback relation for general structure
            relations.append(
                RelationalStatement(
                    relation="operates_on",
                    entity_a=tokens[0] if tokens else "subject",
                    entity_b=tokens[-1] if len(tokens) > 1 else "object",
                    domain=domain,
                )
            )
        return relations
