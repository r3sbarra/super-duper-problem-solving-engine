"""Core primitives, types, and vector representations."""

from super_solver.core.types import (
    ProblemState,
    Hypothesis,
    HypothesisStatus,
    CrucialExperiment,
    KTBoundary,
    OperatorType,
    ReasoningStep,
    DiscoveryPath,
)
from super_solver.core.vsa import VSAEngine
from super_solver.core.embeddings import PolarityAwareEmbeddingService, embedding_service

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
