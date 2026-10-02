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
            name="super_solver_identify",
            description=("Identifies a problem's structure: problem type (optimization/proof/search/"
                         "estimation/design/diagnosis/explanation), entities, givens, unknowns, "
                         "constraints, goal and success criteria. Use BEFORE decomposition to see "
                         "how the engine understands the problem."),
            inputSchema={
                "type": "object",
                "properties": {
                    "specification": {"type": "string", "description": "The problem statement to identify"},
                    "title": {"type": "string", "description": "Optional short title"},
                },
                "required": ["specification"],
            },
        ),
        types.Tool(
            name="super_solver_decompose",
            description=("Decomposes a difficult research or engineering problem into auxiliary angles. "
                         "First IDENTIFIES the problem (type/entities/givens/unknowns/constraints/goal), "
                         "then generates subproblems anchored on those slots via Polya heuristics and "
                         "TRIZ inventive principles. Returns problem_structure, ranked auxiliary_angles, "
                         "suggested_triz_operators and decomposition_metrics (slot coverage, relevance spread)."),
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
            name="super_solver_pareto",
            description=("Resolves a multi-objective trade-off: given objectives (with"
                           " direction) and candidate solutions scored on each, returns the"
                           " non-dominated Pareto frontier, the knee/compromise point, a"
                           " scalarized ranking, and which objective pairs actually trade off."),
            inputSchema={
                "type": "object",
                "properties": {
                    "objectives": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "[{name, direction: 'minimize'|'maximize'}]"
                    },
                    "candidates": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "[{name, values: {objective_name: float}}]"
                    },
                    "weights": {
                        "type": "object",
                        "description": "Optional objective weights for the scalarized ranking"
                    },
                },
                "required": ["objectives", "candidates"],
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
        types.Tool(
            name="super_solver_smt_lra",
            description="Checks satisfiability of linear real-arithmetic constraints (QF_LRA) with an exact Simplex/DPLL(T) verdict (SAT / UNSAT with conflict).",
            inputSchema={
                "type": "object",
                "properties": {
                    "constraints": {"type": "array", "items": {"type": "string"},
                                    "description": "Linear constraints, e.g. ['x + y <= 10', 'x >= 0']"},
                },
                "required": ["constraints"],
            },
        ),
        types.Tool(
            name="super_solver_smt_euf",
            description="Checks satisfiability of equality logic with uninterpreted functions: congruences over equalities and disequalities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "equalities": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                                   "description": "Pairs asserted equal, e.g. [['a','b'],['b','c']]"},
                    "disequalities": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                                      "description": "Pairs asserted unequal, e.g. [['a','c']]"},
                },
                "required": ["equalities"],
            },
        ),
        types.Tool(
            name="super_solver_tournament",
            description="Runs an adversarial Red-Team vs Blue-Team dialectical debate tournament across candidate hypotheses and returns the surviving robust ones.",
            inputSchema={
                "type": "object",
                "properties": {
                    "candidate_hypotheses": {"type": "array", "items": {"type": "string"}},
                    "context": {"type": "string"},
                    "boundary_is_not": {"type": "string"},
                    "top_k": {"type": "integer", "default": 2},
                },
                "required": ["candidate_hypotheses"],
            },
        ),
        types.Tool(
            name="super_solver_coalesce_hypotheses",
            description="Coalesces redundant or near-continuous candidate hypotheses into coarse equivalence classes via invariant-signature compression.",
            inputSchema={
                "type": "object",
                "properties": {
                    "hypotheses": {"type": "array", "items": {"type": "string"}},
                    "domain_context": {"type": "string"},
                },
                "required": ["hypotheses"],
            },
        ),
        types.Tool(
            name="super_solver_audit_boundaries",
            description="Audits a candidate hypothesis or design against universal adversarial boundary regimes; returns resilience, fatal flaws and strengthening advice.",
            inputSchema={
                "type": "object",
                "properties": {
                    "candidate_description": {"type": "string"},
                    "problem_specification": {"type": "string"},
                    "domain_context": {"type": "string"},
                },
                "required": ["candidate_description", "problem_specification"],
            },
        ),
        types.Tool(
            name="super_solver_verify_math_paper",
            description="Audits, tests, and verifies or refutes an unverified mathematical preprint against its target conjecture via Lakatos falsification and proof search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "paper_title": {"type": "string"},
                    "abstract_text": {"type": "string"},
                    "target_conjecture": {"type": "string"},
                },
                "required": ["paper_id", "paper_title", "abstract_text", "target_conjecture"],
            },
        ),
        types.Tool(
            name="super_solver_self_audit",
            description="Runs the engine's self-audit and optimization: DPLL invariant verification, health gate, and self-evaluation status.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="super_solver_consolidate_memory",
            description="Consolidates the episodic memory store, merging near-duplicate cases below a similarity threshold; returns count merged.",
            inputSchema={
                "type": "object",
                "properties": {"similarity_threshold": {"type": "number", "default": 0.75}},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    engine = get_engine()
    bridge = get_bridge()

    if name == "super_solver_identify":
        spec = arguments["specification"]
        title = arguments.get("title", "")
        result = bridge.identify_problem(spec, title)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

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
        # Build a context string from explored paths / dead ends and pass top_k
        # explicitly. (The previous positional call put the dead-end list into
        # top_k, a crash: 'slice indices must be integers'.)
        ctx_parts = []
        if explored:
            ctx_parts.append("Already explored: " + "; ".join(map(str, explored)))
        if dead_ends:
            ctx_parts.append("Dead ends to avoid: " + "; ".join(map(str, dead_ends)))
        context = " | ".join(ctx_parts) or None
        result = bridge.suggest_paths(goal, context=context, top_k=3)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    elif name == "super_solver_triz_solve":
        imp = arguments["improving_parameter"]
        wors = arguments["worsening_parameter"]
        principles = engine.triz.resolve_contradiction_matrix(imp, wors)
        return [types.TextContent(type="text", text=json.dumps({"improving": imp, "worsening": wors, "recommended_principles": principles}, indent=2, default=str))]

    elif name == "super_solver_pareto":
        objectives = arguments["objectives"]
        candidates = arguments["candidates"]
        weights = arguments.get("weights")
        result = engine.resolve_pareto(objectives, candidates, weights)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

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

    elif name == "super_solver_smt_lra":
        result = engine.check_smt_lra(arguments["constraints"])
        return [types.TextContent(type="text", text=json.dumps({"verdict": result.verdict, "is_sat": result.is_sat, "details": result.details}, indent=2, default=str))]

    elif name == "super_solver_smt_euf":
        eq = arguments.get("equalities", [])
        neq = arguments.get("disequalities", [])
        result = engine.check_smt_euf([tuple(e) for e in eq], [tuple(d) for d in neq])
        return [types.TextContent(type="text", text=json.dumps({"verdict": result.verdict, "is_sat": result.is_sat, "details": result.details}, indent=2, default=str))]

    elif name == "super_solver_tournament":
        results = engine.run_tournament(
            candidate_hypotheses=arguments["candidate_hypotheses"],
            context=arguments.get("context"),
            boundary_is_not=arguments.get("boundary_is_not"),
            top_k=arguments.get("top_k", 2),
        )
        return [types.TextContent(type="text", text=json.dumps({"matches": [
            {"hypothesis": m.hypothesis, "verdict": getattr(m.verdict, "value", str(m.verdict)),
             "robustness_score": getattr(m, "robustness_score", None)}
            for m in results
        ]}, indent=2, default=str))]

    elif name == "super_solver_coalesce_hypotheses":
        classes = engine.coalesce_hypotheses(arguments["hypotheses"], arguments.get("domain_context"))
        return [types.TextContent(type="text", text=json.dumps({"classes": classes}, indent=2, default=str))]

    elif name == "super_solver_audit_boundaries":
        report = engine.audit_adversarial_boundaries(
            candidate_description=arguments["candidate_description"],
            problem_specification=arguments["problem_specification"],
            domain_context=arguments.get("domain_context"),
        )
        return [types.TextContent(type="text", text=json.dumps(report, indent=2, default=str))]

    elif name == "super_solver_verify_math_paper":
        result = engine.verify_math_paper(
            paper_id=arguments["paper_id"],
            paper_title=arguments["paper_title"],
            abstract_text=arguments["abstract_text"],
            target_conjecture=arguments["target_conjecture"],
        )
        return [types.TextContent(type="text", text=json.dumps({
            "paper_id": getattr(result, "paper_id", arguments["paper_id"]),
            "verdict": getattr(result, "verdict", None),
            "details": getattr(result, "to_dict", lambda: result)() if hasattr(result, "to_dict") else str(result),
        }, indent=2, default=str))]

    elif name == "super_solver_self_audit":
        result = engine.run_self_audit_and_optimization()
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    elif name == "super_solver_consolidate_memory":
        merged = engine.consolidate_memory(arguments.get("similarity_threshold", 0.75))
        return [types.TextContent(type="text", text=json.dumps({"merged": merged}, indent=2, default=str))]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
