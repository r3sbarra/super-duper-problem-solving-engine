"""Recursive self-application: the engine reviewing itself.

Guards the recursive self-review harness (it must produce a coherent review of
its own measured weaknesses) and locks in the two REAL self-detected limitations,
so they are not silently "fixed" into dishonesty:

  1. The crucible (tournament) is a SANITY FILTER, not an option ranker: it
     passes all good-faith engineering proposals at 1.00. Ranking must use Pareto.
  2. Retrieval is CORPUS-BOUND: the primitive matcher cannot retrieve a solution
     for an abstract/meta problem with no domain vocabulary.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_recursive_self_review_runs():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import importlib

    mod = importlib.import_module("recursive_self_review")
    # Just build the self-description (cheap, no full review) and assert shape.
    desc = mod.build_self_description()
    assert desc["n_subsystems"] > 0
    assert desc["solve_loop_idea_coverage"] is not None


def test_crucible_does_not_rank_good_faith_proposals():
    """Locked-in finding: the tournament passes ALL good-faith proposals, so it is
    a sanity filter, not a ranker."""
    from super_solver.engine import SuperDuperProblemSolvingEngine

    e = SuperDuperProblemSolvingEngine()
    proposals = [
        "Add a solution-synthesis stage before retrieval",
        "Rank whole-problem solutions instead of per-subproblem",
        "Expand the vector corpus with solution ideas",
    ]
    results = [e.tournament.run_match(p) for p in proposals]
    survivors = [r for r in results if r.survived]
    # Documented behaviour: all survive (no discrimination)
    assert len(survivors) == len(results), (
        "crucible now discriminates good-faith proposals — update the self-review finding"
    )


def test_tournament_still_falsifies_flawed_math():
    """The sanity-filter role must remain intact."""
    from super_solver.engine import SuperDuperProblemSolvingEngine

    e = SuperDuperProblemSolvingEngine()
    r = e.tournament.run_match("A proof that P != NP via monotone circuit approximations")
    assert not r.survived
