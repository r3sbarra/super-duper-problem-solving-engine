"""T1.3 — property/fuzz tests for the core reasoning modules.

Deterministic generators (seeded) exercise edge cases that unit tests miss:
empty/whitespace/very-long/unicode inputs, degenerate numeric inputs, and
invariant properties that must always hold.
"""
from __future__ import annotations

import random

import pytest

from super_solver.frameworks.pareto import Candidate, Objective, resolve
from super_solver.frameworks.problem_identification import decompose, identify_problem

WEIRD_TEXTS = [
    "",
    " ",
    "\n\t",
    "a",
    "?",
    "!?!?",
    "x" * 5000,
    "日本語のテキストです",
    "emoji 🚀🔥 test",
    "make it better without making it worse or better",
    "the the the the the",
    "NULL\x00byte",
    "'; DROP TABLE users; --",
    "<script>alert(1)</script>",
]


@pytest.mark.parametrize("text", WEIRD_TEXTS)
def test_identify_never_crashes(text):
    st = identify_problem(text)
    assert st.problem_type in {
        "optimization", "proof", "search", "estimation", "design",
        "diagnosis", "explanation",
    }
    for e in st.entities:
        assert isinstance(e, str)
        assert len(e.split()) <= 3  # clause-merge invariant


@pytest.mark.parametrize("text", WEIRD_TEXTS)
def test_decompose_never_crashes(text):
    out = decompose(text)
    assert "structure" in out and "subproblems" in out
    assert 0.0 <= out["metrics"]["slot_coverage"] <= 1.0
    # no duplicate subproblem texts
    texts = [s["subproblem"] for s in out["subproblems"]]
    assert len(texts) == len(set(texts))


def test_decompose_is_deterministic():
    a = decompose("Reduce latency without increasing cost.")
    b = decompose("Reduce latency without increasing cost.")
    assert [s["subproblem"] for s in a["subproblems"]] == [
        s["subproblem"] for s in b["subproblems"]
    ]


def _rand_pareto(rng):
    n_obj = rng.randint(1, 4)
    n_cand = rng.randint(1, 8)
    objs = [
        Objective(f"o{i}", rng.choice(["minimize", "maximize"]))
        for i in range(n_obj)
    ]
    cands = [
        Candidate(f"c{j}", {f"o{i}": rng.uniform(-10, 10) for i in range(n_obj)})
        for j in range(n_cand)
    ]
    return objs, cands


def test_pareto_invariants_hold_under_fuzz():
    rng = random.Random(11)
    for _ in range(200):
        objs, cands = _rand_pareto(rng)
        r = resolve(objs, cands)
        # front + dominated partition the candidates exactly
        assert set(r.pareto_front) | set(r.dominated) == {c.name for c in cands}
        assert not (set(r.pareto_front) & set(r.dominated))
        # knee (if any) is on the front
        if r.knee is not None:
            assert r.knee in r.pareto_front
        # a frontier member is never strictly dominated by another frontier member
        for a in r.pareto_front:
            for b in r.pareto_front:
                if a != b:
                    assert not _strictly_dominates(b, a, r, objs, cands)


def _strictly_dominates(b, a, r, objs, cands):
    """b strictly dominates a on all objectives (normalized lower-is-better)."""
    names = [o.name for o in objs]
    lo = r.ideal
    hi = r.nadir

    def norm(c, o):
        span = hi[o] - lo[o]
        if span <= 0:
            return 0.0
        v = (c.values[o] - lo[o]) / span
        if next(x for x in objs if x.name == o).direction == "maximize":
            v = 1 - v
        return v

    bv = next(c for c in cands if c.name == b)
    av = next(c for c in cands if c.name == a)
    strictly = False
    for n in names:
        nb, na = norm(bv, n), norm(av, n)
        if nb > na:
            return False
        if nb < na:
            strictly = True
    return strictly


def test_pareto_empty_and_single():
    assert resolve([], []).knee is None
    r = resolve([Objective("x", "minimize")], [Candidate("only", {"x": 5})])
    assert r.pareto_front == ["only"]
    assert r.knee == "only"
