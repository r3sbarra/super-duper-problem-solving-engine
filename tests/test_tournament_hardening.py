"""Tests for the hardened dialectical tournament engine and barrier audit.

Regression guard for the fixes to the rubber-stamp behaviour: the tournament must
NOT mark every hypothesis SURVIVED_ROBUST. Flawed math claims are falsified by
barrier; self-contradictory claims are refuted as inconsistent; sound practical
hypotheses survive.
"""
from __future__ import annotations

from super_solver.frameworks.math_verification import LakatosProofVerificationEngine
from super_solver.frameworks.tournament import (
    DialecticalTournamentEngine,
    TournamentVerdict,
)


def test_barrier_audit_flags_monotone_circuit_pnp_proof():
    e = LakatosProofVerificationEngine()
    r = e.verify_mathematical_paper(
        paper_id="x",
        paper_title="P vs NP",
        abstract_text=(
            "We apply the Berg-Ulfberg monotone approximation method to general "
            "Boolean circuits computing Clique to establish an exponential lower bound."
        ),
        target_conjecture="P versus NP",
    )
    assert r.barrier_violations, "monotone-circuit P!=NP proof should trigger Tardos barrier"


def test_barrier_audit_does_not_false_positive_on_sound_collatz():
    """Tao's sound Collatz paper names the evasion technique (2-adic/renewal) in a
    later sentence; it must NOT be flagged."""
    e = LakatosProofVerificationEngine()
    r = e.verify_mathematical_paper(
        paper_id="t",
        paper_title="Almost all orbits of the Collatz map attain almost bounded values",
        abstract_text=(
            "Let Collatz denote the Syracuse mapping. We prove almost all integers "
            "attain bounded values, modeling 2-adic valuations as a skew random walk "
            "with strictly negative logarithmic drift and proving non-concentration "
            "of the associated renewal measures."
        ),
        target_conjecture="Collatz 3x+1 Conjecture",
    )
    assert not r.barrier_violations, f"false positive: {r.barrier_violations}"


def test_tournament_falsifies_flawed_math_claim():
    e = DialecticalTournamentEngine()
    r = e.run_match("A proof that P != NP via monotone circuit approximations", max_rounds=3)
    assert r.verdict == TournamentVerdict.FALSIFIED_BY_BARRIER
    assert r.survived is False
    assert r.robustness_score < 0.5


def test_tournament_refutes_self_contradictory_claim():
    e = DialecticalTournamentEngine()
    r = e.run_match(
        "Increase the cache size and also reduce it at the same time", max_rounds=3
    )
    assert r.survived is False
    assert r.verdict == TournamentVerdict.WEAKENED_INCONSISTENT


def test_tournament_survives_sound_practical_hypothesis():
    e = DialecticalTournamentEngine()
    r = e.run_match(
        "Add a cache in front of the database to reduce read latency", max_rounds=3
    )
    assert r.survived is True
    assert r.verdict == TournamentVerdict.SURVIVED_ROBUST


def test_tournament_does_not_rubber_stamp_everything():
    """The core regression: not every hypothesis may be SURVIVED_ROBUST."""
    e = DialecticalTournamentEngine()
    hyps = [
        "A proof that P != NP via monotone circuit approximations",
        "Increase the cache size and also reduce it at the same time",
        "Add a cache in front of the database to reduce read latency",
    ]
    results = [e.run_match(h) for h in hyps]
    verdicts = {r.verdict for r in results}
    assert len(verdicts) >= 2, f"tournament gave identical verdicts: {verdicts}"
    survived = [r for r in results if r.survived]
    assert len(survived) < len(results), "everything survived (rubber-stamp)"
