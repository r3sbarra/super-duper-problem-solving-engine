"""Regression guard for primitive breakdown (strip_to_primitives) quality.

Locks in the generic action + quality-attribute layers: problems must strip to a
non-empty primitive set carrying at least one action atom, AND the generic atoms
must NOT inflate cross-domain match scores (they are excluded from ranking).
"""
from __future__ import annotations

from super_solver.vectorize.primitives import (
    GENERIC_ACTION_PRIMITIVES,
    QUALITY_ATTRIBUTES,
    match_by_primitives,
    strip_to_primitives,
)

PROBES = [
    "make a climbing pad that grips smooth walls without glue",
    "reduce p99 latency during cache invalidation",
    "prove there are infinitely many primes",
    "design a battery that charges fast without overheating",
    "detect duplicate invoices across two billing systems",
    "explain why a recommendation engine becomes less diverse over time",
    "find the cheapest way to connect all offices with no cycles",
    "estimate how long a queue takes to drain",
    "improve the engine so its decompositions lead to solved problems",
]


def test_generic_layers_exist():
    assert len(GENERIC_ACTION_PRIMITIVES) >= 15
    assert len(QUALITY_ATTRIBUTES) >= 10


def test_problems_are_not_empty_and_carry_an_action():
    empty = 0
    with_action = 0
    for p in PROBES:
        prims = strip_to_primitives(p)
        if not prims:
            empty += 1
        if any(x.startswith("act_") for x in prims):
            with_action += 1
    assert empty == 0, f"{empty}/{len(PROBES)} problems stripped to empty"
    assert with_action >= len(PROBES) - 2, (
        f"only {with_action}/{len(PROBES)} problems carry an action atom"
    )


def test_generic_atoms_do_not_inflate_match_scores():
    """Two problems that share ONLY generic atoms must not score as a strong
    match (generic atoms are excluded from ranking)."""
    sources = [
        {"title": "unrelated A", "solution": "reduce cost by sourcing cheaper materials"},
        {"title": "unrelated B", "solution": "improve reliability by adding redundancy"},
    ]
    # Query shares generic act_reduce/act_improve but no specific primitive.
    hits = match_by_primitives("improve the quality of something vague", sources, top_k=1)
    assert hits
    assert hits[0]["primitive_overlap"] < 0.5, (
        "generic-only overlap produced a strong match (generic atoms leaking)"
    )


def test_meta_problem_now_has_primitives():
    """The engine's own meta-problem used to strip to EMPTY."""
    prims = strip_to_primitives(
        "improve the engine so its decompositions lead to solved problems"
    )
    assert prims, "meta-problem still strips to empty"
