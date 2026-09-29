"""T2.2 — Causal-VSA verification against synthetic ground truth.

Checks the do-calculus engine recovers correct causal effects on SCMs with known
linear mechanisms: chain, confounder, fork (no effect), and proximal>distal
ordering.
"""
from __future__ import annotations

from super_solver.core.causal_vsa import VectorizedCausalModel


def _model(edges, nodes):
    m = VectorizedCausalModel()
    for n in nodes:
        m.add_variable(n, baseline_value=0.5)
    for c, e, w in edges:
        m.add_causal_edge(c, e, w)
    return m


def test_chain_effect_magnitude_and_sign():
    m = _model([("x", "y", 0.8), ("y", "z", 0.5)], ["x", "y", "z"])
    r = m.intervene("x", 1.0, consequence_target="z")
    # do(x=1): z moves by wx*wy*(1-0.5) = 0.2
    assert abs(r["average_causal_effect"] - 0.2) < 1e-6


def test_fork_has_no_causal_effect():
    """x <- c -> y: intervening on x must NOT change y (no directed path)."""
    m = _model([("c", "x", 0.7), ("c", "y", 0.6)], ["c", "x", "y"])
    r = m.intervene("x", 1.0, consequence_target="y")
    assert abs(r["average_causal_effect"]) < 1e-6


def test_confounder_directed_effect_present():
    """c -> x, c -> y, x -> y: do(x) still moves y via x->y."""
    m = _model([("c", "x", 0.7), ("c", "y", 0.6), ("x", "y", 0.4)], ["c", "x", "y"])
    r = m.intervene("x", 1.0, consequence_target="y")
    assert r["average_causal_effect"] > 0


def test_proximal_effect_exceeds_distal():
    m = _model([("x", "y", 0.8), ("y", "z", 0.5)], ["x", "y", "z"])
    ry = m.intervene("x", 1.0, consequence_target="y")["average_causal_effect"]
    rz = m.intervene("x", 1.0, consequence_target="z")["average_causal_effect"]
    assert ry > rz > 0
