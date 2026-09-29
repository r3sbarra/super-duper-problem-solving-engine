"""Recursive self-application: run the engine's problem-solving machinery on
ITSELF, and see what it suggests for its own improvement.

Protocol:
  1. Introspect the engine's own source + measured artifacts (health report,
     metrics, solve-loop, roadmap) to build an HONEST self-description of its
     weaknesses (not hand-fed claims).
  2. Frame that self-description as a PROBLEM.
  3. Run the engine's full machinery on the problem: identify -> decompose ->
     suggest_paths -> TRIZ -> KT boundary -> abduct -> crucible (tournament) ->
     pareto (over candidate improvements).
  4. Compare the engine's self-suggestions against the human roadmap; report
     which it agrees with, and any NEW suggestions.

Writes assets/recursive_self_review.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS = os.path.join(ROOT, "assets")


def _load(name, default=None):
    p = os.path.join(ASSETS, name)
    try:
        return json.load(open(p))
    except Exception:
        return default


def build_self_description() -> dict:
    """Assemble an evidence-based self-description from measured artifacts."""
    health = _load("health_report.json", {"summary": {}})
    metrics = _load("metrics_artifact.json", {}).get("metrics", {})
    solve = _load("solve_loop.json", {})
    return {
        "n_subsystems": health["summary"].get("ok", 0)
        + health["summary"].get("fail", 0)
        + health["summary"].get("degenerate", 0),
        "failing_subsystems": health["summary"].get("fail", 0),
        "solve_loop_idea_coverage": solve.get("solution_idea_coverage", None),
        "subproblem_only_coverage": solve.get("solution_coverage_at_3", None),
        "ood_scorer_car": metrics.get("probe_suites", {}).get("ood_13", {}).get("scorer_car"),
        "polarity_baseline_car": metrics.get("probe_suites", {}).get("ood_13", {}).get("polarity_car"),
    }


def main():
    from super_solver.engine import SuperDuperProblemSolvingEngine
    from super_solver.frameworks.problem_identification import decompose, identify_problem

    engine = SuperDuperProblemSolvingEngine()
    desc = build_self_description()

    problem = (
        "The super-solver problem-solving engine decomposes problems well but its "
        "decomposition covers almost none of the problem-specific solution idea "
        f"(solution-idea coverage {desc['solve_loop_idea_coverage']} vs subproblem "
        f"coverage {desc['subproblem_only_coverage']}); it is a decomposer and "
        "mechanism-retriever, not yet an end-to-end solver. Improve the engine so "
        "that its decompositions lead to solved problems, without regressing its "
        f"measured health ({desc['n_subsystems']} subsystems, "
        f"{desc['failing_subsystems']} failing) or its retrieval accuracy."
    )

    review = {"self_description": desc, "problem": problem}

    # 1. Identify
    st = identify_problem(problem)
    review["identify"] = {"type": st.problem_type, "entities": st.entities[:6],
                          "objectives": st.objectives, "is_vague": st.is_vague}

    # 2. Decompose (the engine decomposing its own weakness)
    d = decompose(problem)
    review["decompose"] = {
        "subproblems": [s["subproblem"] for s in d["subproblems"][:6]],
        "slot_coverage": d["metrics"]["slot_coverage"],
        "solution_ideas": [i["idea"] for i in d.get("solution_ideas", [])],
    }
    # Self-detected gap: can the engine retrieve a solution for an ABSTRACT /
    # meta problem (its own weakness)? It has no domain primitives for it.
    no_idea = all("no structural primitives" in i for i in review["decompose"]["solution_ideas"]) \
        or not review["decompose"]["solution_ideas"]
    review["self_retrieval_gap"] = (
        "the engine CANNOT retrieve a solution idea for its own meta problem: the "
        "primitive matcher has no domain vocabulary for meta-engineering, so "
        "retrieval is corpus-bound (works for systems/math problems, not abstract "
        "self-improvement problems)." if no_idea else None
    )

    # 3. Suggest paths
    paths = engine.suggest_paths(problem, top_k=4)
    review["suggest_paths"] = [
        {"title": p.title, "strategy": getattr(p, "strategy_type", ""),
         "action": getattr(p, "recommended_next_action", "")[:120]}
        for p in paths
    ]

    # 4. TRIZ contradiction resolution on the core tension
    triz = engine.triz.suggest_principles(
        f"Improve solution specificity of {st.entities[0] if st.entities else 'decomposition'} "
        f"without regressing retrieval accuracy", top_k=3)
    review["triz"] = [{"id": t["principle_id"], "name": t["name"]} for t in triz]

    # 5. KT boundary on the failure
    kt = engine.kt.analyze_boundary(
        is_facts=["decomposition subproblems are generic scaffolding",
                  "retrieval finds real mechanisms for corpus problems"],
        is_not_facts=["decomposition subproblems never reference the problem idea",
                      "retrieval fails for out-of-corpus problems"],
    )
    review["kt_boundary"] = {
        "leading_cause": kt.get("leading_cause"),
        "n_candidates": len(kt.get("candidate_evaluations", [])),
    }

    # 6. Abduction: competing hypotheses for HOW to fix it
    hyps = engine.peirce.autonomous_abduct(problem, domain_context="meta-engineering")
    review["abduct"] = [getattr(h, "description", "")[:100] for h in hyps[:4]]

    # 7. Crucible: tournament the concrete candidate improvements
    candidates = [
        "Add a solution-synthesis stage that proposes a concrete mechanism for each subproblem before retrieval",
        "Retrieve and rank candidate solutions for the WHOLE problem, not per subproblem",
        "Expand the vector corpus with solution ideas so retrieval generalizes beyond hand-authored cases",
        "Leave decomposition as-is; the engine is a decomposition tool, not a solver",
    ]
    tournament = engine.run_tournament(candidates, top_k=len(candidates))
    review["crucible"] = [
        {"hypothesis": t.hypothesis, "verdict": t.verdict.value,
         "robustness": t.robustness_score, "survived": t.survived}
        for t in tournament
    ]
    # Does the crucible DISCRIMINATE between competing good-faith proposals?
    verdicts = {c["verdict"] for c in review["crucible"]}
    survivors = [c for c in review["crucible"] if c["survived"]]
    review["crucible_discriminates"] = len(verdicts) > 1 and len(survivors) < len(review["crucible"])
    review["crucible_limitation"] = (
        None if review["crucible_discriminates"]
        else "the crucible passes ALL good-faith engineering proposals at 1.00: it is a "
             "sanity filter (falsifies flawed math / self-contradictions), NOT an option "
             "ranker. Competing design proposals must be ranked by the Pareto resolver, "
             "not the tournament."
    )

    # 8. Pareto over candidate improvements on (impact, effort) — engine's own resolver
    pareto = engine.resolve_pareto(
        objectives=[
            {"name": "expected_impact", "direction": "maximize"},
            {"name": "implementation_effort", "direction": "minimize"},
            {"name": "regression_risk", "direction": "minimize"},
        ],
        candidates=[
            {"name": "solution_synthesis_stage", "values": {"expected_impact": 9, "implementation_effort": 8, "regression_risk": 6}},
            {"name": "whole_problem_solution_retrieval", "values": {"expected_impact": 7, "implementation_effort": 3, "regression_risk": 3}},
            {"name": "expand_solution_corpus", "values": {"expected_impact": 6, "implementation_effort": 5, "regression_risk": 2}},
            {"name": "status_quo_decomposition_only", "values": {"expected_impact": 1, "implementation_effort": 0, "regression_risk": 0}},
        ],
    )
    review["pareto"] = pareto

    # 9. Compare engine self-suggestions with the human roadmap
    review["agreement"] = _compare_with_roadmap(review)

    path = os.path.join(ASSETS, "recursive_self_review.json")
    json.dump(review, open(path, "w"), indent=2)
    _print(review)
    print(f"\nwrote {path}")


def _compare_with_roadmap(review) -> dict:
    """Which engine suggestions match the human roadmap (T2.3 solve loop)?"""
    roadmap_t2_3 = "end-to-end solve loop / decomposition->solution link"
    engine_actions = " ".join(
        p["action"].lower() for p in review["suggest_paths"]
    ) + " " + " ".join(review["decompose"]["subproblems"]).lower()
    agrees = any(k in engine_actions for k in ("solution", "mechanism", "retriev"))
    return {
        "human_roadmap_item": roadmap_t2_3,
        "engine_agrees_on_solution_link": agrees,
        "engine_pareto_recommends": review["pareto"]["knee"],
        "engine_crucible_survivors": [
            c["hypothesis"] for c in review["crucible"] if c["survived"]
        ],
    }


def _print(review):
    print("=== ENGINE SELF-REVIEW ===")
    print("self-description:", json.dumps(review["self_description"], indent=2))
    print("\nidentify:", review["identify"]["type"], review["identify"]["entities"])
    print("\n-- decomposed its own weakness into --")
    for s in review["decompose"]["subproblems"]:
        print("  -", s[:88])
    print("\n-- retrieved solution ideas --")
    for i in review["decompose"]["solution_ideas"]:
        print("  -", i[:88])
    print("\n-- suggested paths --")
    for p in review["suggest_paths"]:
        print(f"  [{p['strategy']}] {p['title'][:50]}")
        print(f"     -> {p['action']}")
    print("\nTRIZ:", review["triz"])
    print("KT leading cause:", review["kt_boundary"]["leading_cause"])
    print("\nabduction hypotheses:")
    for h in review["abduct"]:
        print("  -", h)
    print("\n-- crucible (tournament of improvements) --")
    for c in review["crucible"]:
        print(f"  {c['robustness']:.2f} {c['verdict']:22} survived={c['survived']} :: {c['hypothesis'][:60]}")
    print("\n-- pareto over improvements --")
    print("  front:", review["pareto"]["pareto_front"])
    print("  knee :", review["pareto"]["knee"])
    print("  conflicts:", review["pareto"]["conflicts"])
    print("\nagreement with human roadmap:", json.dumps(review["agreement"], indent=2))


if __name__ == "__main__":
    main()
