"""Adversarial Dialectical Tournament Engine.

Pits candidate hypotheses against an adversarial Red Team (Refuter) and Blue Team (Advocate)
in a multi-round debate tournament, cross-examining claims against impossibility barriers,
dead-end manifolds, and logical consistency before expensive downstream exploration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from super_solver.core.dpll_solver import DPLLSolver, dpll_solver
from super_solver.core.embeddings import embedding_service
from super_solver.frameworks.math_verification import (
    LakatosProofVerificationEngine,
    VerificationStatus,
)
from super_solver.search.negative_manifold import NegativeManifoldRepulsor


class TournamentVerdict(str, Enum):
    SURVIVED_ROBUST = "SURVIVED_ROBUST"
    FALSIFIED_BY_BARRIER = "FALSIFIED_BY_BARRIER"
    CONCEDED_DEAD_END = "CONCEDED_DEAD_END"
    WEAKENED_INCONSISTENT = "WEAKENED_INCONSISTENT"


@dataclass
class TournamentRound:
    round_number: int
    advocate_argument: str
    refuter_counter: str
    barrier_detected: Optional[str] = None
    dead_end_conflict: Optional[str] = None
    round_score: float = 1.0  # 1.0 = advocate prevailed, 0.0 = refuter crushed hypothesis


@dataclass
class TournamentMatchResult:
    hypothesis: str
    verdict: TournamentVerdict
    robustness_score: float
    survived: bool
    rounds: List[TournamentRound] = field(default_factory=list)
    barrier_violations: List[str] = field(default_factory=list)
    dead_end_warnings: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    summary: str = ""
    # Which adversarial checks actually had traction on this hypothesis. A
    # SURVIVED_ROBUST verdict with an empty list means "nothing tested it", not
    # "it withstood testing" -- read alongside robustness_score, never alone.
    checks_applied: List[str] = field(default_factory=list)
    evidence_coverage: float = 0.0


class DialecticalTournamentEngine:
    """Multi-agent adversarial tournament engine evaluating scientific & mathematical claims."""

    def __init__(
        self,
        repulsor: Optional[NegativeManifoldRepulsor] = None,
        lakatos: Optional[LakatosProofVerificationEngine] = None,
        dpll: Optional[DPLLSolver] = None,
    ):
        self.repulsor = repulsor or NegativeManifoldRepulsor()
        self.lakatos = lakatos or LakatosProofVerificationEngine()
        self.dpll = dpll or dpll_solver

    def run_match(
        self,
        hypothesis: str,
        context: Optional[str] = None,
        max_rounds: int = 3,
        boundary_is_not: Optional[str] = None,
    ) -> TournamentMatchResult:
        """Runs a multi-round adversarial debate on a candidate hypothesis."""
        rounds: List[TournamentRound] = []
        barrier_violations: List[str] = []
        dead_end_warnings: List[str] = []
        counterexamples: List[str] = []

        hyp_vec = embedding_service.encode(hypothesis)
        current_robustness = 1.0
        traction: List[str] = []

        # Round 1: Dead-End & Negative Manifold Audit (Quicksand Check)
        is_near, max_sim, closest_desc = self.repulsor.check_proximity(hyp_vec, threshold=0.62)
        if self.repulsor.dead_ends:
            traction.append("negative_manifold_audit")
        if is_near and closest_desc:
            dead_end_warnings.append(f"Near dead-end (sim={max_sim:.2f}): {closest_desc}")
            current_robustness *= 1.0 - max_sim * 0.7
            r1 = TournamentRound(
                round_number=1,
                advocate_argument=f"Hypothesis proposes mechanism: {hypothesis}",
                refuter_counter=f"Refuter flags severe proximity to known falsified trajectory: {closest_desc}",
                dead_end_conflict=closest_desc,
                round_score=max(0.0, 1.0 - max_sim),
            )
        else:
            r1 = TournamentRound(
                round_number=1,
                advocate_argument=f"Hypothesis proposes mechanism: {hypothesis}",
                refuter_counter="Refuter checks negative manifold: no proximity to known toxic dead ends detected.",
                round_score=1.0,
            )
        rounds.append(r1)

        # Round 2: Mathematical / Physical Barrier & Impossibility Audit
        # Check against Lakatos barriers
        lakatos_res = self.lakatos.verify_mathematical_paper(
            paper_id="hypothesis_claim",
            paper_title=hypothesis,
            abstract_text=hypothesis,
            target_conjecture=hypothesis,
        )

        if (
            lakatos_res.verdict == VerificationStatus.REFUTED_FLAWED
            and lakatos_res.barrier_violations
        ):
            for b in lakatos_res.barrier_violations:
                barrier_violations.append(b)
            for c in lakatos_res.counterexamples:
                counterexamples.append(c)
            current_robustness *= 0.15
            r2 = TournamentRound(
                round_number=2,
                advocate_argument="Advocate claims structural novelty and formal breakthrough.",
                refuter_counter=f"Refuter identifies barrier violation: {', '.join(lakatos_res.barrier_violations)}",
                barrier_detected=lakatos_res.barrier_violations[0],
                round_score=0.1,
            )
        else:
            r2 = TournamentRound(
                round_number=2,
                advocate_argument="Advocate asserts evasion of known mathematical/physical impossibility barriers.",
                refuter_counter="Refuter verifies technique avoids canonical impossibility barriers.",
                round_score=0.95,
            )
        if lakatos_res.verdict != VerificationStatus.INCONCLUSIVE:
            traction.append("barrier_audit")
        rounds.append(r2)

        # Round 3: Boundary & Consistency Refutation
        contradiction = None
        if max_rounds >= 3:
            advocate_claim = "Hypothesis holds under constraints with high explanatory coherence."
            refuter_response = "Refuter verifies internal constraint satisfaction."
            round_score = 0.90

            # Internal contradiction check: a hypothesis asserting both a
            # predicate and its opposite ('increase X and also reduce X') is
            # self-refuting and must not survive. Without this the tournament
            # rubber-stamped such claims at 1.00.
            from super_solver.frameworks.problem_identification import (
                _antonym_flip,
                _antonym_variant_pattern,
            )

            hyp_low = hypothesis.lower()
            antonym_self = _antonym_flip(hypothesis, hypothesis)

            def _present(word: str) -> bool:
                # Inflection-aware: a plain `"increase" in hyp_low` misses
                # "increases"/"increasing", so "Increasing the dose reduces
                # survival" was NOT flagged as self-contradictory while the
                # uninflected wording was.
                return re.search(_antonym_variant_pattern(word), hyp_low) is not None

            both_dirs = any(
                (_present(a) and _present(b))
                for a, b in (
                    ("increase", "reduce"), ("increase", "decrease"),
                    ("add", "remove"), ("raise", "lower"),
                    ("enable", "disable"), ("always", "never"),
                    ("maximize", "minimize"),
                )
            )
            if antonym_self or both_dirs:
                contradiction = (
                    "Hypothesis asserts mutually exclusive predicates "
                    "(both a claim and its opposite)"
                )
                current_robustness *= 0.20
                refuter_response = f"Refuter exposes internal contradiction: {contradiction}."
                round_score = 0.10

            if boundary_is_not:
                b_vec = embedding_service.encode(boundary_is_not)
                boundary_leak_sim = embedding_service.cosine_similarity(hyp_vec, b_vec)
                if boundary_leak_sim >= 0.65:
                    refuter_response = f"Refuter exposes boundary contamination: overlaps with IS NOT condition (sim={boundary_leak_sim:.2f})."
                    current_robustness *= 0.50
                    round_score = 0.35

            if contradiction is not None or boundary_is_not:
                traction.append("boundary_consistency")

            r3 = TournamentRound(
                round_number=3,
                advocate_argument=advocate_claim,
                refuter_counter=refuter_response,
                round_score=round_score,
            )
            rounds.append(r3)

        # Final verdict determination
        current_robustness = float(np.clip(current_robustness, 0.0, 1.0))
        if barrier_violations:
            verdict = TournamentVerdict.FALSIFIED_BY_BARRIER
            survived = False
            summary = f"Refuted by Red Team: Violates {len(barrier_violations)} barrier(s) ({barrier_violations[0]})."
        elif contradiction:
            verdict = TournamentVerdict.WEAKENED_INCONSISTENT
            survived = False
            summary = f"Refuted by Red Team: {contradiction}."
        elif dead_end_warnings and current_robustness < 0.45:
            verdict = TournamentVerdict.CONCEDED_DEAD_END
            survived = False
            summary = (
                f"Conceded to Red Team: Trapped in known failure manifold ({dead_end_warnings[0]})."
            )
        elif current_robustness < 0.60:
            verdict = TournamentVerdict.WEAKENED_INCONSISTENT
            survived = False
            summary = "Weakened during debate: High vulnerability to boundary counterexamples."
        else:
            verdict = TournamentVerdict.SURVIVED_ROBUST
            survived = True
            # A verdict may only claim what the checks actually established. When
            # no check had traction the honest reading is "nothing tested this",
            # not "this withstood testing" -- the earlier wording asserted the
            # latter for a claim no round could engage with.
            if traction:
                summary = (
                    f"Not falsified: survived {len(traction)}/{len(rounds)} adversarial "
                    f"check(s) that had traction ({', '.join(traction)})."
                )
            else:
                summary = (
                    "NOT FALSIFIED, BUT UNTESTED: no adversarial check had traction "
                    "(no dead ends registered, claim outside the barrier auditor's "
                    "scope, no boundary condition supplied). Absence of evidence, "
                    "not evidence of robustness."
                )

        return TournamentMatchResult(
            hypothesis=hypothesis,
            verdict=verdict,
            robustness_score=round(current_robustness, 3),
            survived=survived,
            rounds=rounds,
            barrier_violations=barrier_violations,
            dead_end_warnings=dead_end_warnings,
            counterexamples=counterexamples,
            summary=summary,
            checks_applied=traction,
            evidence_coverage=round(len(traction) / max(1, len(rounds)), 3),
        )

    def run_tournament(
        self,
        candidate_hypotheses: List[str],
        context: Optional[str] = None,
        boundary_is_not: Optional[str] = None,
        top_k: int = 2,
    ) -> List[TournamentMatchResult]:
        """Pits multiple candidate hypotheses against the Red Team and ranks survivors by robustness."""
        results: List[TournamentMatchResult] = []
        for hyp in candidate_hypotheses:
            res = self.run_match(
                hypothesis=hyp,
                context=context,
                boundary_is_not=boundary_is_not,
            )
            results.append(res)

        # Rank by robustness score descending
        results.sort(key=lambda r: (r.survived, r.robustness_score), reverse=True)
        return results[:top_k]
