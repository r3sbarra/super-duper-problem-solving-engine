"""Smoke tests: every advertised MCP tool is callable without crashing.

Guards against wiring bugs found in the wild (e.g. suggest_paths passing a list
into top_k; triz_solve calling a non-existent method).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from super_solver.mcp import server


def _run(name, args):
    return asyncio.run(server.call_tool(name, args))


def test_all_mcp_tools_listed():
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    for expected in [
        "super_solver_identify",
        "super_solver_decompose",
        "super_solver_deflect",
        "super_solver_analogize",
        "super_solver_suggest_paths",
        "super_solver_triz_solve",
        "super_solver_kt_boundary",
        "super_solver_next_experiment",
        "super_solver_primitive_analogize",
    ]:
        assert expected in names


@pytest.mark.parametrize(
    "name,args",
    [
        ("super_solver_identify", {"specification": "Prove Fermat's little theorem."}),
        ("super_solver_decompose", {"topic": "Shortest-path routing under time windows."}),
        ("super_solver_deflect", {"proposed_content": "add a cache", "known_dead_ends": ["add a cache"]}),
        ("super_solver_analogize", {"specification": "grip smooth walls without glue"}),
        ("super_solver_suggest_paths", {"current_goal": "reduce p99 latency"}),
        ("super_solver_triz_solve", {"improving_parameter": "speed", "worsening_parameter": "accuracy"}),
        ("super_solver_kt_boundary", {"is_facts": ["sparks in bedroom"], "is_not_facts": ["not in kitchen"]}),
        ("super_solver_next_experiment", {
            "hypothesis_priors": {"h1": 0.5, "h2": 0.5},
            "candidate_designs": [{"design_id": "d1", "name": "a", "cost": 1}],
        }),
        ("super_solver_primitive_analogize", {"specification": "make a climbing pad grip walls"}),
    ],
)
def test_mcp_tool_callable(name, args):
    out = _run(name, args)
    assert out and out[0].text
    # Output must be valid JSON (all tools serialize dicts).
    json.loads(out[0].text)


def test_suggest_paths_tool_accepts_dead_end_list():
    """Regression: a dead-end list must not be mistaken for top_k (integer)."""
    out = _run("super_solver_suggest_paths", {
        "current_goal": "reduce latency",
        "explored_paths": ["add a cache"],
        "dead_ends": ["polymer coat the anode"],
    })
    data = json.loads(out[0].text)
    assert isinstance(data, list)
