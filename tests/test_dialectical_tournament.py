"""Tests for the Adversarial Dialectical Tournament Engine (Red Team vs Blue Team)."""

from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.frameworks.tournament import TournamentVerdict


def test_tournament_debate_eliminates_barrier_violating_hypothesis():
    """Validates that Red Team detects impossibility barriers and refutes flawed claims."""
    engine = SuperDuperProblemSolvingEngine()

    hypotheses = [
        "Proving P != NP by monotone circuit approximation bounds on general Boolean circuits",
        "Solving bounded prime gaps via smooth moduli in Bombieri-Vinogradov theorem",
    ]

    results = engine.run_tournament(
        candidate_hypotheses=hypotheses,
        context="Computational complexity circuit lower bounds and analytic number theory.",
        top_k=2,
    )

    assert len(results) == 2

    # Monotone circuit approximation should be refuted by Tardos Barrier
    monotone_res = next(r for r in results if "monotone circuit" in r.hypothesis)
    assert monotone_res.survived is False
    assert monotone_res.verdict == TournamentVerdict.FALSIFIED_BY_BARRIER
    assert len(monotone_res.barrier_violations) >= 1
    assert "Tardos" in monotone_res.barrier_violations[0]

    # Prime gaps breakthrough should survive unrefuted
    prime_res = next(r for r in results if "prime gaps" in r.hypothesis)
    assert prime_res.survived is True
    assert prime_res.verdict == TournamentVerdict.SURVIVED_ROBUST
    assert prime_res.robustness_score > 0.80


def test_tournament_detects_dead_end_manifold_proximity():
    """Validates that Red Team detects proximity to known negative dead ends."""
    engine = SuperDuperProblemSolvingEngine()

    # Register known dead end
    dead_end_hazard = "Chemical vapor deposition with uncontrolled oxygen stoichiometry"
    engine.repulsor.register_dead_end(dead_end_hazard)

    match = engine.tournament.run_match(
        hypothesis="Chemical vapor deposition with uncontrolled oxygen stoichiometry and high gas flow",
    )

    assert match.survived is False
    assert match.verdict == TournamentVerdict.CONCEDED_DEAD_END
    assert len(match.dead_end_warnings) >= 1
