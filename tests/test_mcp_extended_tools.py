"""Regression tests for the extended super-solver MCP tool surface.

The engine had strong internal methods (SMT, tournament, boundary audit, paper
verification, self-audit, memory consolidation) that were not reachable by
agents. These tests pin the MCP exposure and its basic behaviour.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.mcp import server


def _names():
    tools = asyncio.run(server.list_tools())
    return {t.name for t in tools}


def test_extended_tools_exposed():
    names = _names()
    for t in (
        "super_solver_smt_lra",
        "super_solver_smt_euf",
        "super_solver_tournament",
        "super_solver_coalesce_hypotheses",
        "super_solver_audit_boundaries",
        "super_solver_verify_math_paper",
        "super_solver_self_audit",
        "super_solver_consolidate_memory",
    ):
        assert t in names, f"MCP tool {t} not exposed"


def _call(name, args):
    r = asyncio.run(server.call_tool(name, args))
    return r[0].text


def test_smt_lra_sat_and_unsat():
    sat = _call("super_solver_smt_lra", {"constraints": ["x + y <= 10", "x >= 0", "y >= 0"]})
    assert "SAT" in sat
    unsat = _call("super_solver_smt_lra", {"constraints": ["x >= 5", "x <= 3"]})
    assert "UNSAT" in unsat


def test_smt_euf_consistency():
    out = _call("super_solver_smt_euf", {"equalities": [["a", "b"], ["b", "c"]], "disequalities": [["a", "c"]]})
    assert "UNSAT" in out  # a=b=c but a!=c is inconsistent


def test_tournament_runs():
    out = _call("super_solver_tournament", {"candidate_hypotheses": ["more memory", "faster cpu"]})
    assert "SURVIVED" in out


def test_verify_math_paper_refutes_flawed_claim():
    out = _call("super_solver_verify_math_paper", {
        "paper_id": "P1",
        "paper_title": "T",
        "abstract_text": "A short proof that P equals NP using a novel technique.",
        "target_conjecture": "P = NP",
    })
    assert "REFUTED" in out


def test_self_audit_returns_status():
    out = _call("super_solver_self_audit", {})
    assert "status" in out.lower() or "AUDIT" in out
