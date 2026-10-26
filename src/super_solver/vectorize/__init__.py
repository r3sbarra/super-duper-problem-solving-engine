"""Problem / Solution / Path vectorizing subsystem.

Exposes typed vector encoders (``ProblemVectorizer``, ``SolutionVectorizer``,
``PathVectorizer``), a pluggable custom embedder (``core.embedder``), and a
persistent ``VectorIndex`` for similarity search and analogical transfer.

The orchestrator (:class:`VectorizationService`) is the single entry point
used by the CLI, the lab-ass bridge, and the falsci adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from super_solver.core.embedder import (
    EmbedderBackend,
    PathVectorizer,
    ProblemVectorizer,
    SolutionVectorizer,
    get_backend,
)
from super_solver.vectorize.index import VectorIndex


class VectorizationService:
    """High-level service: encode problems/solutions/paths and index them."""

    def __init__(
        self,
        backend: Optional[EmbedderBackend] = None,
        db_path: Optional[str] = None,
    ):
        self.backend = backend or get_backend()
        self.problem_vec = ProblemVectorizer(self.backend)
        self.solution_vec = SolutionVectorizer(self.backend)
        self.path_vec = PathVectorizer(self.backend)
        self.index = VectorIndex(db_path=db_path, backend=self.backend)

    def ensure_seeded(self, force: bool = False) -> Dict[str, Any]:
        """Seed the committed domain-knowledge fact base into an empty index.

        The engine shipped with an empty brain: ``build_domain_knowledge_index``
        was only ever called from ``scripts/`` and tests, never from the runtime
        or MCP path, and the index itself lives in a gitignored directory.  So a
        fresh install answered every ``analogize``/``primitive_analogize`` query
        with nothing at all, and the only indexes that ever had content were
        built ad hoc by offline scripts (which is why the one on the dev machine
        held 738 rows of coding/DB/self-improvement work and no other domain).

        This makes the committed fact base the default prior.  Idempotent:
        entries are stored under content-derived ids, so re-running replaces
        rather than duplicates.  Returns a small JSON-safe report.
        """
        if self.index.db_path != ":memory:" or force:
            if force or self.index.count("problem") == 0:
                from super_solver.vectorize.domain_knowledge import (
                    build_domain_knowledge_index,
                )

                build_domain_knowledge_index(self)
                return {
                    "seeded": True,
                    "problems": self.index.count("problem"),
                    "solutions": self.index.count("solution"),
                }
        existing = self.index.count("problem")
        return {"seeded": False, "reason": "index already populated", "problems": existing}

    # -- encode -------------------------------------------------------------

    def encode_problem(
        self,
        specification: str,
        title: str = "",
        goal_criteria: Optional[List[str]] = None,
        boundary_is: Optional[List[str]] = None,
        boundary_is_not: Optional[List[str]] = None,
    ) -> np.ndarray:
        return self.problem_vec.encode(
            specification,
            title=title,
            goal_criteria=goal_criteria,
            boundary_is=boundary_is,
            boundary_is_not=boundary_is_not,
        )

    def encode_solution(
        self,
        solution_text: str,
        method: str = "",
        domain: str = "",
        operators: Optional[List[str]] = None,
    ) -> np.ndarray:
        return self.solution_vec.encode(
            solution_text, method=method, domain=domain, operators=operators
        )

    def encode_path(
        self,
        steps: List[str],
        operator_types: Optional[List[str]] = None,
        final_breakthrough: str = "",
    ) -> np.ndarray:
        return self.path_vec.encode(steps, operator_types=operator_types, final_breakthrough=final_breakthrough)

    # -- index --------------------------------------------------------------

    def store_problem(
        self,
        specification: str,
        title: str = "",
        goal_criteria: Optional[List[str]] = None,
        boundary_is: Optional[List[str]] = None,
        boundary_is_not: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        vec = self.encode_problem(
            specification, title, goal_criteria, boundary_is, boundary_is_not
        )
        return self.index.add_problem(specification, vec, title=title, metadata=metadata)

    def store_solution(
        self,
        solution_text: str,
        method: str = "",
        domain: str = "",
        operators: Optional[List[str]] = None,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        vec = self.encode_solution(solution_text, method, domain, operators)
        return self.index.add_solution(solution_text, vec, title=title, metadata=metadata)

    def store_path(
        self,
        steps: List[str],
        operator_types: Optional[List[str]] = None,
        final_breakthrough: str = "",
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        vec = self.encode_path(steps, operator_types, final_breakthrough)
        summary = " -> ".join(steps) if steps else final_breakthrough
        return self.index.add_path(summary, vec, title=title, metadata=metadata)

    # -- query --------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        kind: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        return self.index.search(query_vector, kind=kind, top_k=top_k, min_similarity=min_similarity)

    def analogize(
        self,
        problem_vector: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        return self.index.analogize(problem_vector, top_k=top_k, min_similarity=min_similarity)

    def hybrid_analogize(
        self,
        problem_vector: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.0,
        solution_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        return self.index.hybrid_analogize(
            problem_vector, top_k=top_k, min_similarity=min_similarity, solution_weight=solution_weight
        )

    def primitive_analogize(
        self,
        target_text: str,
        top_k: int = 3,
        min_overlap: float = 0.0,
        semantic_weight: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Cross-domain analogical transfer via structural primitives.

        Strips the target problem into structural atoms (action + object-type
        primitives) and matches against all stored solutions by containment
        overlap. Robust to surface differences: two problems in different
        domains share a structure when they share primitives, even if their
        surface entities differ. This is the cross-domain path that surface
        vector retrieval fails at.

        ``semantic_weight`` in [0,1] composes the containment score with the
        OFFLINE semantic embedder (distilled_pid): measured, 32% of problem->
        solution pairs share no specific atom, so containment is 0 for them by
        construction and NO re-weighting can retrieve them; a semantic bridge is
        required (see docs/IMPROVEMENT_PLAN.md R1). Default 0.0 = pure
        containment (unchanged behaviour).
        """
        from super_solver.vectorize.primitives import match_by_primitives

        # The index accumulates duplicates (measured: 738 rows, 186 distinct
        # contents — up to 8 copies of a solution). Dedupe by solution text so
        # retrieval returns DISTINCT analogs instead of the same one repeated.
        seen = set()
        sources = []
        for row in self.index._rows("solution"):
            sol = row[3]
            if sol in seen:
                continue
            seen.add(sol)
            sources.append({"title": row[2] or "", "solution": sol})
        # Widen the candidate net when a semantic term is used, so disjoint
        # (zero-containment) pairs the semantic bridge can rescue are not cut
        # off before scoring.
        fetch_k = top_k if semantic_weight <= 0 else max(top_k, min(len(sources), 50))
        hits = match_by_primitives(target_text, sources, top_k=fetch_k)

        if semantic_weight > 0:
            try:
                from super_solver.core.distilled_pid import DistilledPIDEmbedder

                emb = DistilledPIDEmbedder()
                if emb.available:
                    vt = emb.encode(target_text)
                    w = max(0.0, min(1.0, semantic_weight))
                    for h in hits:
                        sem = emb.cosine_similarity(vt, emb.encode(h["content"]))
                        # semantic cosine ~0.2-0.6 band; rescale for blending.
                        sem_scaled = max(0.0, min(1.0, (sem - 0.2) / 0.5))
                        # Composition, not replacement: containment captures
                        # structural atoms, semantic captures paraphrase. Measured
                        # best: 0.5 containment + 0.5 semantic -> top-1 0.476 vs
                        # containment-only 0.286 and semantic-only 0.429. A LEARNED
                        # ranker over the same features did NOT beat hand-set (0.286
                        # both): the semantic feature is the entire gain.
                        h["containment_score"] = h["primitive_overlap"]
                        h["semantic_score"] = sem
                        h["primitive_overlap"] = (
                            (1.0 - w) * h["primitive_overlap"] + w * sem_scaled
                        )
                    hits.sort(key=lambda x: x["primitive_overlap"], reverse=True)
            except Exception:
                pass

        return [
            {
                "content": h["content"],
                "title": h["title"],
                "primitive_overlap": h["primitive_overlap"],
                "target_primitives": h["target_primitives"],
                "source_primitives": h["source_primitives"],
                # Propagate the no-structural-signal sentinel flag. Dropping it
                # here made every downstream `if h.get("empty_match"): continue`
                # guard dead code, so a pseudo-hit whose content is the literal
                # "<no structural primitives extracted>" was reported as if it
                # were a retrieved analog (see problem_identification.py).
                "empty_match": bool(h.get("empty_match", False)),
            }
            for h in hits[:top_k]
            if h["primitive_overlap"] >= min_overlap
        ]

    def close(self):
        self.index.close()


# Singleton default service
vectorization_service = VectorizationService()
