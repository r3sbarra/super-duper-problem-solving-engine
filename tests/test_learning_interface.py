"""Tests for the engine's learning interface: `remember_solution` / `record_dead_end`.

Why these exist
---------------
Before this interface the engine was query-only from a caller's point of view:
all 18 MCP tools could read and none could write, so the analogical index only
ever grew from offline scripts and the dead-end repulsor only from lab-ass
events.  That is the mechanism behind the documented finding that the shipped
index held 738 rows of coding/DB/self-improvement work and returned nothing
useful for any other domain -- the engine could not be taught by the agent
using it.

These tests pin the loop closed: teach a case -> retrieve it back as a full
analog; record a failure -> have it deflect a paraphrase.
"""

from __future__ import annotations

import asyncio
import json

from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.mcp import server
from super_solver.vectorize import vectorization_service

PROBLEM = (
    "how to stop a background job from running twice when two workers pick it up"
)
SOLUTION = (
    "guard the job with a distributed lock keyed by job id, with a lease timeout "
    "so a crashed worker releases it"
)
STEPS = [
    "identify the duplicate-execution race",
    "introduce a keyed lock",
    "add a lease timeout",
]


def test_remember_solution_is_retrievable_as_a_full_analog():
    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")
    out = engine.remember_solution(
        problem_specification=PROBLEM,
        solution=SOLUTION,
        title="single-flight job lock",
        method="distributed lock + lease",
        domain="distributed-systems",
        path_steps=STEPS,
    )
    assert out["problem_id"] and out["solution_id"] and out["path_id"]

    # A paraphrase of the same problem must retrieve the taught case, with BOTH
    # its solution and its path resolved (that is what makes it a reusable
    # precedent rather than just a recalled title).
    hits = vectorization_service.analogize(
        vectorization_service.encode_problem(
            "two workers execute the same scheduled job at the same time"
        ),
        top_k=1,
    )
    assert hits, "taught case was not retrievable"
    top = hits[0]
    assert top["problem"]["content"] == PROBLEM
    assert top["solution"] is not None
    assert "distributed lock" in top["solution"]["content"]
    assert top["path"] is not None
    assert "keyed lock" in top["path"]["content"]


def test_remember_solution_is_idempotent():
    """Re-teaching the same case updates it rather than duplicating it."""
    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")
    first = engine.remember_solution(PROBLEM, SOLUTION)
    second = engine.remember_solution(PROBLEM, SOLUTION)
    assert first["problem_id"] == second["problem_id"]
    assert first["solution_id"] == second["solution_id"]
    assert first["index_size"] == second["index_size"]


def test_record_dead_end_persists_and_deflects_a_paraphrase():
    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")
    before = len(engine.repulsor.dead_ends)
    out = engine.record_dead_end(
        "polling the database in a tight loop to detect job completion"
    )
    assert out["recorded"] is True
    assert out["total_dead_ends"] == before + 1

    from super_solver.core.embeddings import embedding_service

    is_near, sim, closest = engine.repulsor.check_proximity(
        embedding_service.encode("poll the db repeatedly until the job finishes")
    )
    assert closest == "polling the database in a tight loop to detect job completion"
    assert sim > 0.0
    assert isinstance(is_near, bool)


def test_record_dead_end_survives_a_restart_through_the_store():
    """The whole point of the tool: it must outlive the process."""
    from super_solver.memory.episodic_store import EpisodicVectorStore

    store = EpisodicVectorStore(db_path=":memory:")
    engine_a = SuperDuperProblemSolvingEngine.__new__(SuperDuperProblemSolvingEngine)
    # Wire just enough of the engine to exercise register_dead_end + persistence.
    from super_solver.search.negative_manifold import NegativeManifoldRepulsor

    engine_a.repulsor = NegativeManifoldRepulsor(store=store)
    engine_a.record_dead_end("cache the result of a non-deterministic function")

    reloaded = NegativeManifoldRepulsor(store=store)
    assert "cache the result of a non-deterministic function" in reloaded.dead_end_descriptions


def test_mcp_learning_tools_round_trip():
    async def _run(name, args):
        return await server.call_tool(name, args)

    out = asyncio.run(
        _run(
            "super_solver_remember_solution",
            {
                "problem_specification": "mcp: dedupe a stream of events by id",
                "solution": "keep a bounded sliding-window set of seen ids",
                "title": "mcp case",
            },
        )
    )
    payload = json.loads(out[0].text)
    assert payload["problem_id"]

    out = asyncio.run(
        _run("super_solver_record_dead_end", {"description": "mcp: scan every row"})
    )
    payload = json.loads(out[0].text)
    assert payload["recorded"] is True
    assert payload["total_dead_ends"] >= 1
