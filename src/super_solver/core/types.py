"""Core formal types and data models for Super-Duper-Problem-Solving-Engine."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np


class HypothesisStatus(str, Enum):
    PROPOSED = "PROPOSED"
    TESTING = "TESTING"
    CONFIRMED = "CONFIRMED"
    FALSIFIED = "FALSIFIED"
    REFINED = "REFINED"


class OperatorType(str, Enum):
    TRIZ_TRANSFORMATION = "TRIZ_TRANSFORMATION"
    TOC_EVAPORATING_CLOUD = "TOC_EVAPORATING_CLOUD"
    POLYA_AUXILIARY_DECOMP = "POLYA_AUXILIARY_DECOMP"
    POLYA_WORKING_BACKWARDS = "POLYA_WORKING_BACKWARDS"
    GENTNER_ANALOGICAL_MAP = "GENTNER_ANALOGICAL_MAP"
    PLATT_CRUCIAL_EXPERIMENT = "PLATT_CRUCIAL_EXPERIMENT"
    PEIRCE_ABDUCTIVE_LEAP = "PEIRCE_ABDUCTIVE_LEAP"
    CONTINUOUS_LATENT_STEP = "CONTINUOUS_LATENT_STEP"


class KTBoundary(BaseModel):
    """Kepner-Tregoe 4-Dimensional IS / IS NOT Boundary Condition specification."""
    identity_is: str
    identity_is_not: str
    location_is: str
    location_is_not: str
    timing_is: str
    timing_is_not: str
    extent_is: str
    extent_is_not: str

    def to_contrastive_text(self) -> Tuple[str, str]:
        """Returns concatenated (IS_text, IS_NOT_text)."""
        is_text = f"IS: {self.identity_is}. In location: {self.location_is}. At timing: {self.timing_is}. Scope extent: {self.extent_is}."
        is_not_text = f"IS NOT: {self.identity_is_not}. In location: {self.location_is_not}. At timing: {self.timing_is_not}. Scope extent: {self.extent_is_not}."
        return is_text, is_not_text


class Hypothesis(BaseModel):
    """Formal scientific hypothesis representation."""
    id: str
    title: str
    description: str
    prior_confidence: float = 0.5
    current_confidence: float = 0.5
    falsification_criteria: List[str] = Field(default_factory=list)
    predictions: List[str] = Field(default_factory=list)
    vector: Optional[List[float]] = None
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    evidence_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrucialExperiment(BaseModel):
    """Platt Strong Inference crucial discriminating experiment."""
    id: str
    name: str
    description: str
    target_hypotheses: List[str] = Field(default_factory=list)
    exclusory_predictions: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps hypothesis_id -> expected outcome if hypothesis is true"
    )
    actual_outcome: Optional[str] = None
    falsified_hypotheses: List[str] = Field(default_factory=list)
    information_gain: float = 0.0


class ProblemState(BaseModel):
    """The 5-Tuple problem state: (P, O, S, G, D)."""
    id: str
    title: str
    specification: str
    boundary: Optional[KTBoundary] = None
    goal_criteria: List[str] = Field(default_factory=list)
    current_step: int = 0
    state_vector: Optional[List[float]] = None
    goal_vector: Optional[List[float]] = None
    known_dead_ends: List[List[float]] = Field(default_factory=list)
    active_hypotheses: List[Hypothesis] = Field(default_factory=list)
    resolved: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReasoningStep(BaseModel):
    """A single discrete or continuous reasoning transition."""
    step_index: int
    operator_type: OperatorType
    operator_name: str
    description: str
    latent_vector: Optional[List[float]] = None
    cost_to_goal: float = 1.0
    dead_end_margin: float = 1.0
    confidence: float = 0.5
    symbolic_summary: Optional[str] = None


class DiscoveryPath(BaseModel):
    """A complete reconstructed or discovered path from problem to breakthrough."""
    problem_id: str
    problem_title: str
    initial_abduction: str
    steps: List[ReasoningStep] = Field(default_factory=list)
    crucial_experiments: List[CrucialExperiment] = Field(default_factory=list)
    falsified_paths: List[str] = Field(default_factory=list)
    final_breakthrough: str
    total_steps: int = 0
    confidence: float = 1.0


class VerificationStatus(str, Enum):
    VERIFIED_SOUND = "VERIFIED_SOUND"
    REFUTED_FLAWED = "REFUTED_FLAWED"
    GAP_DETECTED = "GAP_DETECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class MathematicalLemma(BaseModel):
    """A formal mathematical lemma or theorem step in a paper's proof."""
    lemma_id: str
    statement: str
    technique: str
    assumptions: List[str] = Field(default_factory=list)
    claimed_bound: Optional[str] = None
    status: str = "PENDING"
    barrier_conflicts: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MathematicalBarrier(BaseModel):
    """A known proven obstruction or no-go theorem in mathematics."""
    barrier_id: str
    name: str
    field: str
    description: str
    obstruction_criterion: str
    evasion_criterion: str = ""
    counterexample_template: str = ""
    established_reference: str


class ProofVerificationResult(BaseModel):
    """Formal verification report for an unproven or contested mathematical paper."""
    paper_id: str
    paper_title: str
    target_conjecture: str
    verdict: VerificationStatus
    confidence: float
    barrier_violations: List[str] = Field(default_factory=list)
    counterexamples: List[str] = Field(default_factory=list)
    flawed_lemmas: List[str] = Field(default_factory=list)
    sound_lemmas: List[str] = Field(default_factory=list)
    reasoning_trace: List[str] = Field(default_factory=list)
    formal_summary: str


class SuggestedPath(BaseModel):
    """An actionable, ranked alternative discovery path or repair trajectory."""
    path_id: str
    strategy_type: str
    title: str
    description: str
    rationale: str
    feasibility_score: float = 0.5
    novelty_score: float = 0.5
    dead_end_safety_margin: float = 1.0
    recommended_next_action: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


