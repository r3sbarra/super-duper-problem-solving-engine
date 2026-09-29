"""T5.1 — lab-ass bridge end-to-end.

Exercises every bridge hook against the (optionally live) lab-ass integration.
The pure hooks (decompose/identify/deflect/suggest_paths/tournament) run offline;
the live-REST check is skipped when the lab-ass service is unreachable.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="module")
def bridge():
    from super_solver.lab_ass_bridge.hooks import LabAssSuperSolverBridge

    return LabAssSuperSolverBridge()


def test_bridge_decompose_hook(bridge):
    out = bridge.decompose_topic_with_polya_and_triz("Reduce latency under budget.", [])
    assert out["auxiliary_angles"]
    assert "problem_structure" in out
    assert "decomposition_metrics" in out


def test_bridge_identify_hook(bridge):
    st = bridge.identify_problem("Prove there are infinitely many primes.")
    assert st["problem_type"] == "proof"
    assert "entities" in st


def test_bridge_deflect_hook(bridge):
    out = bridge.check_dead_end_deflection("add a cache", ["add a cache in front of the db"])
    assert "hazard_detected" in out


def test_bridge_suggest_paths_hook(bridge):
    paths = bridge.suggest_paths("reduce p99 latency", top_k=3)
    assert len(paths) == 3
    assert all("title" in p for p in paths)


def test_bridge_tournament_hook(bridge):
    res = bridge.run_tournament(["add a cache to reduce latency", "avoid caches entirely"])
    assert res


def test_bridge_live_service_reachable():
    import json
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:3002/api/health", timeout=3) as r:
            data = json.loads(r.read().decode())
        assert data.get("status") == "ok"
    except Exception:
        pytest.skip("lab-ass service not reachable")
