"""Proven problem-solving frameworks and scientific engines."""

from super_solver.frameworks.gentner_sme import RelationalStatement, StructureMappingEngine
from super_solver.frameworks.kepner_tregoe import KepnerTregoeEngine
from super_solver.frameworks.math_verification import LakatosProofVerificationEngine
from super_solver.frameworks.peirce_inquiry import PeirceanInquiryEngine
from super_solver.frameworks.platt_inference import StrongInferenceEngine
from super_solver.frameworks.polya_heuristics import PolyaHeuristicsEngine
from super_solver.frameworks.toc_cloud import EvaporatingCloud, TOCEngine
from super_solver.frameworks.triz_engine import TRIZ_PRINCIPLES, TRIZEngine
from super_solver.frameworks.universal_vector_solver import (
    UniversalVectorizedSolver,
    VectorizedSolution,
)

__all__ = [
    "StrongInferenceEngine",
    "KepnerTregoeEngine",
    "TRIZEngine",
    "TRIZ_PRINCIPLES",
    "TOCEngine",
    "EvaporatingCloud",
    "PeirceanInquiryEngine",
    "PolyaHeuristicsEngine",
    "StructureMappingEngine",
    "RelationalStatement",
    "LakatosProofVerificationEngine",
    "UniversalVectorizedSolver",
    "VectorizedSolution",
]
