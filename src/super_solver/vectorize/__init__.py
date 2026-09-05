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

    def close(self):
        self.index.close()


# Singleton default service
vectorization_service = VectorizationService()
