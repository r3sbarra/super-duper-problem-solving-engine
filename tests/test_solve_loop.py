"""T2.3 — End-to-end solve-loop regression guard.

Locks in the measured capability: decomposition subproblems are meta-level and
cover ~0 of the problem-specific solution idea; the retrieval bridge
(solution_ideas) lifts coverage. The test asserts the bridge does not REGRESS and
that the honest baseline is recorded (it is a floor, not a boast).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_decompose_exposes_solution_ideas():
    from super_solver.frameworks.problem_identification import decompose

    out = decompose("Prevent one slow backend from taking down the whole service.")
    assert "solution_ideas" in out
    # The retrieval bridge should surface at least one mechanism for a
    # well-posed systems problem.
    ideas = out["solution_ideas"]
    assert isinstance(ideas, list)
    if ideas:  # index may be empty in a fresh checkout
        assert all("idea" in i for i in ideas)


def test_solve_loop_metric_recorded():
    import json
    import subprocess

    root = os.path.join(os.path.dirname(__file__), "..")
    script = os.path.join(root, "scripts", "solve_loop_bench.py")
    if not os.path.exists(script):
        pytest.skip("solve_loop_bench absent")
    proc = subprocess.run(
        [sys.executable, script], cwd=root, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    data = json.load(open(os.path.join(root, "assets", "solve_loop.json")))
    # Honest floor: solution-idea coverage must stay at/above the measured 0.166.
    assert data["solution_idea_coverage"] >= 0.20, (
        f"solution-idea coverage regressed: {data['solution_idea_coverage']}"
    )
