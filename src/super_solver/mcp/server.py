"""Standalone MCP Server for Super-Duper-Problem-Solving-Engine."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.lab_ass_bridge.hooks import LabAssSuperSolverBridge

server = Server("super-solver")
_engine: SuperDuperProblemSolvingEngine | None = None
_bridge: LabAssSuperSolverBridge | None = None


def get_engine() -> SuperDuperProblemSolvingEngine:
    global _engine
    if _engine is None:
        _engine = SuperDuperProblemSolvingEngine()
    return _engine


def get_bridge() -> LabAssSuperSolverBridge:
    global _bridge
    if _bridge is None:
        _bridge = LabAssSuperSolverBridge(get_engine())
    return _bridge


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="super_solver_decompose",
            description="Decomposes a difficult research or engineering problem into auxiliary angles using Polya heuristics and TRIZ inventive principles.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The problem or research topic to decompose"},
                    "initial_subproblems": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Known subproblems or aspects already identified",
                    },
                },
                "required": ["topic"],
            },
        ),
        types.Tool(
            name="super_solver_deflect",
            description="Evaluates if a proposed approach or finding is in the attraction basin of known dead ends, and computes geometric deflection in continuous representation space.",
            inputSchema={
                "type": "object",
                "properties": {
                    "proposed_content": {"type": "string", "description": "The idea, angle, or finding to test"},
                    "known_dead_ends": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of known failed approaches / dead ends",
                    },
                },
                "required": ["proposed_content"],
            },
        ),
        types.Tool(
            name="super_solver_analogize",
            description="Retrieves cross-domain structural analogies using Vector Symbolic Architecture (VSA) / Hyperdimensional Computing (HDC) and Gentner's Structure-Mapping Engine.",
            inputSchema={
                "type": "object",
                "properties": {
                    "specification": {"type": "string", "description": "Description of the target system or challenge to find analogies for"},
                    "top_k": {"type": "integer", "default": 3, "description": "Number of analogies to retrieve"},
                },
                "required": ["specification"],
            },
        ),
        types.Tool(
            name="super_solver_suggest_paths",
            description="Suggests next optimal exploratory or reasoning paths avoiding dead ends using continuous thought guidance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "current_goal": {"type": "string", "description": "The current problem goal"},
                    "explored_paths": {"type": "array", "items": {"type": "string"}, "description": "Paths already tried"},
                    "dead_ends": {"type": "array", "items": {"type": "string"}, "description": "Known dead ends"},
                },
                "required": ["current_goal"],
            },
        ),
        types.Tool(
            name="super_solver_triz_solve",
            description="Uses Altshuller's TRIZ contradiction matrix to resolve trade-offs between improving and worsening parameters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "improving_parameter": {"type": "string", "description": "What property or parameter needs to be improved"},
                    "worsening_parameter": {"type": "string", "description": "What property deteriorates as a side-effect"},
                },
                "required": ["improving_parameter", "worsening_parameter"],
            },
        ),
        types.Tool(
            name="super_solver_kt_boundary",
            description="Applies Kepner-Tregoe IS / IS-NOT 4D boundary separation to isolate root cause or system failure envelope.",
            inputSchema={
                "type": "object",
                "properties": {
                    "is_facts": {"type": "array", "items": {"type": "string"}, "description": "Facts that ARE true / observed"},
                    "is_not_facts": {"type": "array", "items": {"type": "string"}, "description": "Facts that COULD be true but ARE NOT"},
                },
                "required": ["is_facts", "is_not_facts"],
            },
        ),
        types.Tool(
            name="super_solver_next_experiment",
            description="Ranks candidate experiment designs using Bayesian Optimal Experimental Design (BOED) by cost-penalized Expected Information Gain.",
            inputSchema={
                "type": "object",
                "properties": {
                    "hypothesis_priors": {
                        "type": "object",
                        "description": "Mapping from hypothesis ID to prior probability float",
                    },
                    "candidate_designs": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of experiment design dicts with design_id, name, parameters, cost",
                    },
                    "likelihood_matrix": {
                        "type": "object",
                        "description": "Nested mapping: design_id -> outcome -> hypothesis_id -> float",
                    },
                },
                "required": ["hypothesis_priors", "candidate_designs"],
            },
        ),
        types.Tool(
            name="super_solver_primitive_analogize",
            description="Performs cross-domain analogical transfer by extracting structural primitives (action + object atoms) and computing overlap with stored solutions. Overcomes surface cosine similarity limits.",
            inputSchema={
                "type": "object",
                "properties": {
                    "specification": {"type": "string", "description": "Description of the target system to analogize"},
                    "top_k": {"type": "integer", "default": 3, "description": "Number of cross-domain analogs to retrieve"},
                    "min_overlap": {"type": "number", "default": 0.0, "description": "Minimum structural primitive overlap ratio"},
                },
                "required": ["specification"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    engine = get_engine()
    bridge = get_bridge()

    if name == "super_solver_decompose":
        topic = arguments["topic"]
        subproblems = arguments.get("initial_subproblems", [])
        result = bridge.decompose_topic_with_polya_and_triz(topic, subproblems)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "super_solver_deflect":
        content = arguments["proposed_content"]
        dead_ends = arguments.get("known_dead_ends", [])
        result = bridge.check_dead_end_deflection(content, dead_ends)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name in ("super_solver_analogize", "super_solver_primitive_analogize"):
        spec = arguments["specification"]
        top_k = arguments.get("top_k", 3)
        min_overlap = arguments.get("min_overlap", 0.0)
        # Wire to the cross-domain structural path. The previous call,
        # engine.gentner.retrieve_analogs, does not exist (StructureMappingEngine
        # only defines align_systems) and raised AttributeError at runtime, so the
        # analogize tool was dead code. primitive_analogize strips the target to
        # action/object primitives and matches stored solutions by containment
        # overlap, which transfers across domains (vs. surface cosine noise).
        from super_solver.vectorize import vectorization_service

        analogs = vectorization_service.primitive_analogize(spec, top_k=top_k, min_overlap=min_overlap)
        return [types.TextContent(type="text", text=json.dumps({"specification": spec, "analogs": analogs}, indent=2, default=str))]

    elif name == "super_solver_suggest_paths":
        goal = arguments["current_goal"]
        explored = arguments.get("explored_paths", [])
        dead_ends = arguments.get("dead_ends", [])
        result = bridge.suggest_paths(goal, explored, dead_ends)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    elif name == "super_solver_triz_solve":
        imp = arguments["improving_parameter"]
        wors = arguments["worsening_parameter"]
        principles = engine.triz.resolve_contradiction(imp, wors)
        return [types.TextContent(type="text", text=json.dumps({"improving": imp, "worsening": wors, "recommended_principles": principles}, indent=2, default=str))]

    elif name == "super_solver_kt_boundary":
        is_facts = arguments["is_facts"]
        is_not = arguments["is_not_facts"]
        causes = arguments.get("candidate_causes")
        boundary = engine.kt.analyze_boundary(is_facts=is_facts, is_not_facts=is_not, candidate_causes=causes)
        return [types.TextContent(type="text", text=json.dumps({"boundary": boundary}, indent=2, default=str))]

    elif name == "super_solver_next_experiment":
        priors = arguments["hypothesis_priors"]
        designs = arguments["candidate_designs"]
        likelihoods = arguments.get("likelihood_matrix")
        ranked = engine.boed.rank_experiments(priors, designs, likelihoods)
        return [types.TextContent(type="text", text=json.dumps({"ranked_experiments": ranked}, indent=2, default=str))]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
