"""T5.2 — MCP schema stability snapshot.

Records each MCP tool's name + inputSchema so an unannounced schema change fails
CI. Regenerate the snapshot intentionally when a tool changes.
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from super_solver.mcp import server

SNAPSHOT = os.path.join(os.path.dirname(__file__), "..", "assets", "mcp_schema.json")


def _current():
    tools = asyncio.run(server.list_tools())
    return {
        t.name: {"description": t.description, "inputSchema": t.inputSchema}
        for t in tools
    }


def test_mcp_schema_matches_snapshot():
    cur = _current()
    if not os.path.exists(SNAPSHOT):
        os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
        json.dump(cur, open(SNAPSHOT, "w"), indent=2, sort_keys=True)
        pytest.skip("schema snapshot created")
    snap = json.load(open(SNAPSHOT))
    # Tool set must match exactly (added/removed tools are breaking changes).
    assert set(cur) == set(snap), (
        f"tool set changed: added={set(cur) - set(snap)} removed={set(snap) - set(cur)}"
    )
    for name in cur:
        assert cur[name]["inputSchema"] == snap[name]["inputSchema"], (
            f"inputSchema changed for {name}"
        )
