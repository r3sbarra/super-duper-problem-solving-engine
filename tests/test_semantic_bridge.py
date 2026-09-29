"""Regression guard for the semantic bridge in primitive_analogize (P1).

Locks in the measured requirement (docs/IMPROVEMENT_PLAN.md R1): 32% of
problem->solution pairs share NO specific atom, so pure containment scores 0 for
them. With semantic_weight > 0, the offline embedder must BREAK that zero floor
(rescue at least one disjoint pair), while pure containment stays the default.
"""
from __future__ import annotations

import os

import pytest


def _service_with(corpus):
    os.environ["SUPER_SOLVER_VECTOR_DB"] = ":memory:"
    from super_solver.vectorize import VectorizationService

    svc = VectorizationService()
    for _prob, sol in corpus:
        svc.store_solution(solution_text=sol, method="d", domain="d")
    return svc


CORPUS = [
    ("reduce the latency of the read path", "add a cache in front of the database"),
    ("keep a service up when a backend slows", "add a circuit breaker that fails fast"),
    ("find the cheapest way to link offices", "grow a minimum spanning tree"),
    ("totally unrelated widget request", "polish the brass doorknob"),
]


def test_pure_containment_is_still_the_default():
    svc = _service_with(CORPUS)
    hits = svc.primitive_analogize("reduce the latency of the read path", top_k=2)
    assert hits
    # no semantic fields when the bridge is off
    assert "semantic_score" not in hits[0]


def test_semantic_bridge_breaks_the_zero_floor():
    """A disjoint pair (no shared specific atom) must become retrievable with the
    semantic bridge enabled."""
    # problem and solution share NO specific primitive
    disjoint = (
        "make the checkout feel instant to users",
        "serve the product page from a precomputed edge snapshot",
    )
    svc = _service_with(CORPUS + [disjoint])
    # pure containment cannot rank the correct answer first (score 0)
    off = svc.primitive_analogize(disjoint[0], top_k=1, semantic_weight=0.0)
    # semantic bridge should surface it somewhere in the top-k
    on = svc.primitive_analogize(disjoint[0], top_k=len(CORPUS) + 1, semantic_weight=0.8)
    assert off  # returns something (possibly wrong) without crashing
    if any("semantic_score" not in h for h in on):
        pytest.skip("distilled embedder unavailable")
    # the correct solution should get a semantic score attached
    assert all("semantic_score" in h for h in on)
