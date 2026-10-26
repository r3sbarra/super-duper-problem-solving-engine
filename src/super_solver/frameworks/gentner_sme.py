"""Dedre Gentner's Structure-Mapping Engine (SME) for Analogical Problem Solving.

Discovers deep relational isomorphisms between disparate scientific domains,
aligning relational structures while abstracting away surface entities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from super_solver.core.embeddings import embedding_service
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

    @staticmethod
    def _mapping_compatible(
        b_stmt: RelationalStatement,
        t_stmt: RelationalStatement,
        mapping: Dict[str, str],
        reverse: Dict[str, str],
    ) -> bool:
        """Whether this candidate match respects the one-to-one constraint.

        Gentner's structure-mapping requires a *consistent, injective* mapping:
        one base entity maps to exactly one target entity, and no two base
        entities share a target entity. Without this check the previous
        implementation simply overwrote entries, so ``sun`` could be bound to
        two different targets in the same alignment and the reported mapping was
        just whichever statement happened to be processed last.
        """
        for base_ent, target_ent in (
            (b_stmt.entity_a, t_stmt.entity_a),
            (b_stmt.entity_b, t_stmt.entity_b),
        ):
            if base_ent in mapping and mapping[base_ent] != target_ent:
                return False  # base entity already committed elsewhere
            if target_ent in reverse and reverse[target_ent] != base_ent:
                return False  # target entity already claimed (injectivity)
        return True

    def align_systems(
        self,
        base_domain_relations: List[RelationalStatement],
        target_domain_relations: List[RelationalStatement],
    ) -> Dict[str, Any]:
        """Aligns relational structures between base and target domains (Gentner SME).

        Evaluates relational correspondence and induces entity bindings under
        the one-to-one constraint: each target statement is consumed by at most
        one base statement, and the induced entity map stays consistent and
        injective. A candidate that would contradict an established binding is
        rejected rather than silently overwriting it, so the alignment does not
        depend on statement order.
        """
        if not base_domain_relations or not target_domain_relations:
            return {
                "structural_alignment_score": 0.0,
                "matches": [],
                "isomorphic_transfer_viable": False,
            }

        matches = []
        entity_mappings: Dict[str, str] = {}
        reverse_mappings: Dict[str, str] = {}
        used_target: set = set()
        relational_scores = []

        for b_stmt in base_domain_relations:
            v_b_rel = self._get_symbol_vector(b_stmt.relation)
            best_idx = None
            best_sim = -1.0

            for t_idx, t_stmt in enumerate(target_domain_relations):
                if t_idx in used_target:
                    continue  # one-to-one: a target statement matches once
                v_t_rel = self._get_symbol_vector(t_stmt.relation)
                # Compare relational predicates (identical relations have sim=1.0)
                sim = self.vsa.similarity(v_b_rel, v_t_rel)
                if sim <= 0.40:
                    continue
                if not self._mapping_compatible(
                    b_stmt, t_stmt, entity_mappings, reverse_mappings
                ):
                    continue
                if sim > best_sim:
                    best_sim = sim
                    best_idx = t_idx

            if best_idx is not None:
                t_stmt = target_domain_relations[best_idx]
                used_target.add(best_idx)
                relational_scores.append(best_sim)
                matches.append(
                    {
                        "base_relation": f"{b_stmt.relation}({b_stmt.entity_a}, {b_stmt.entity_b})",
                        "target_relation": f"{t_stmt.relation}({t_stmt.entity_a}, {t_stmt.entity_b})",
                        "relational_match_score": float(best_sim),
                    }
                )
                # Induce entity correspondences (now guaranteed consistent).
                entity_mappings[b_stmt.entity_a] = t_stmt.entity_a
                entity_mappings[b_stmt.entity_b] = t_stmt.entity_b
                reverse_mappings[t_stmt.entity_a] = b_stmt.entity_a
                reverse_mappings[t_stmt.entity_b] = b_stmt.entity_b

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
