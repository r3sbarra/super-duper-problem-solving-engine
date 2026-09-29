"""Tests for the multi-objective Pareto resolver."""
from __future__ import annotations

from super_solver.frameworks.pareto import Candidate, Objective, resolve


def test_pareto_front_excludes_dominated():
    objs = [Objective("latency", "minimize"), Objective("cost", "minimize")]
    cands = [
        Candidate("a", {"latency": 10, "cost": 90}),
        Candidate("b", {"latency": 50, "cost": 50}),
        Candidate("c", {"latency": 90, "cost": 10}),
        Candidate("dominated", {"latency": 95, "cost": 95}),
    ]
    r = resolve(objs, cands)
    assert set(r.pareto_front) == {"a", "b", "c"}
    assert r.dominated == ["dominated"]


def test_knee_is_the_compromise_point():
    objs = [Objective("x", "minimize"), Objective("y", "minimize")]
    cands = [
        Candidate("extreme_x", {"x": 0, "y": 100}),
        Candidate("extreme_y", {"x": 100, "y": 0}),
        Candidate("middle", {"x": 50, "y": 50}),
    ]
    r = resolve(objs, cands)
    assert r.knee == "middle"


def test_conflict_detected_on_pure_tradeoff():
    objs = [Objective("strength", "maximize"), Objective("weight", "minimize")]
    cands = [Candidate(f"c{i}", {"strength": i * 10, "weight": i * 10}) for i in range(1, 6)]
    r = resolve(objs, cands)
    assert len(r.pareto_front) == 5
    assert any(abs(c[2] + 1.0) < 1e-6 for c in r.conflicts)


def test_maximize_direction_is_respected():
    # maximizing 'gain' should prefer higher gain
    objs = [Objective("gain", "maximize")]
    cands = [Candidate("low", {"gain": 1}), Candidate("high", {"gain": 10})]
    r = resolve(objs, cands)
    ranking = dict(r.scalarized_ranking)
    assert ranking["high"] < ranking["low"]  # lower scalarized = better


def test_weights_shift_the_scalarized_ranking():
    objs = [Objective("x", "minimize"), Objective("y", "minimize")]
    cands = [
        Candidate("x_win", {"x": 0, "y": 100}),
        Candidate("y_win", {"x": 100, "y": 0}),
    ]
    heavy_x = resolve(objs, cands, weights={"x": 0.9, "y": 0.1})
    assert dict(heavy_x.scalarized_ranking)["x_win"] < dict(heavy_x.scalarized_ranking)["y_win"]
    heavy_y = resolve(objs, cands, weights={"x": 0.1, "y": 0.9})
    assert dict(heavy_y.scalarized_ranking)["y_win"] < dict(heavy_y.scalarized_ranking)["x_win"]


def test_empty_inputs_do_not_crash():
    r = resolve([], [])
    assert r.pareto_front == []
    assert r.knee is None
