"""Core primitives, types, and vector representations."""

from super_solver.core.embeddings import PolarityAwareEmbeddingService, embedding_service
from super_solver.core.types import (
    CrucialExperiment,
    DiscoveryPath,
    Hypothesis,
    HypothesisStatus,
    KTBoundary,
    OperatorType,
    ProblemState,
    ReasoningStep,
)
from super_solver.core.vsa import VSAEngine

__all__ = [
    "ProblemState",
    "Hypothesis",
    "HypothesisStatus",
    "CrucialExperiment",
    "KTBoundary",
    "OperatorType",
    "ReasoningStep",
    "DiscoveryPath",
    "VSAEngine",
    "PolarityAwareEmbeddingService",
    "embedding_service",
]
