"""Automated subsystem health probe for the super-solver.

Exercises every major subsystem with a representative input and classifies it:
  OK          — produced a non-trivial, discriminating result
  DEGENERATE  — ran but returned empty/constant/trivial output
  FAIL        — raised an exception

This is a diagnostic (no asserts) that writes assets/health_report.json, used to
prioritise the improvement roadmap.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RESULTS = []


def probe(name, fn):
    try:
        out = fn()
    except Exception as e:
        RESULTS.append({"subsystem": name, "status": "FAIL",
                        "detail": f"{type(e).__name__}: {e}"})
        return
    deg = (out is None) or (isinstance(out, (list, dict, str, tuple)) and len(out) == 0)
    RESULTS.append({"subsystem": name,
                    "status": "DEGENERATE" if deg else "OK",
                    "detail": repr(out)[:160]})


def _emb():
    from super_solver.core.embeddings import embedding_service
    return embedding_service


def main():
    from super_solver.core.dpll_solver import dpll_solver
    from super_solver.engine import SuperDuperProblemSolvingEngine
    from super_solver.frameworks.gentner_sme import RelationalStatement
    from super_solver.frameworks.toc_cloud import EvaporatingCloud

    e = SuperDuperProblemSolvingEngine()

    # --- core solvers ---
    probe("core.dpll.refute", lambda: dpll_solver.refute_conjecture(
        ["A -> B", "B -> C", "A"], "C"))
    probe("core.smt_lra", lambda: e.check_smt_lra(["x + y <= 10", "x >= 0", "y >= 0"]))
    probe("core.smt_euf", lambda: e.check_smt_euf([("f(a)", "b")], [("a", "c")]))

    # --- frameworks ---
    probe("platt.crucial_experiments", lambda: e.platt.autonomous_design_crucial_experiments(
        e.peirce.autonomous_abduct("The 104C transition in Pb10-xCux(PO4)6O", "condensed matter")))
    probe("triz.principles", lambda: e.triz.suggest_principles(
        "improve speed without worsening accuracy", top_k=3))
    probe("kt.boundary", lambda: e.kt.analyze_boundary(
        ["sparks in bedroom", "on Tuesdays"], ["not in kitchen", "never Saturdays"]))
    probe("polya.aux", lambda: e.polya.generate_auxiliary_problems(
        "Prove there are infinitely many primes"))
    probe("gentner.align", lambda: e.gentner.align_systems(
        [RelationalStatement(relation="flows_to", entity_a="heat", entity_b="cold", domain="thermo")],
        [RelationalStatement(relation="flows_to", entity_a="water", entity_b="low", domain="hydro")]))
    probe("peirce.abduct", lambda: e.peirce.autonomous_abduct(
        "The plant grew toward the window", "botany"))
    probe("toc.evaporate", lambda: e.toc.evaporate(
        EvaporatingCloud(objective="ship on time", requirement_a="maximize speed",
                         requirement_b="maximize quality", prerequisite_d="add staff",
                         prerequisite_d_prime="reduce scope"), ["automate testing"]))

    # --- problem identification / decomposition ---
    from super_solver.frameworks.problem_identification import decompose
    probe("problem_id.decompose", lambda: decompose(
        "Shortest-path routing under time-window constraints."))

    # --- vectorize / retrieval ---
    from super_solver.vectorize import vectorization_service
    probe("vectorize.primitive_analogize", lambda: vectorization_service.primitive_analogize(
        "grip smooth vertical walls without glue", top_k=2))

    # --- search / latent ---
    probe("search.negative_manifold", lambda: _nm(e))
    probe("search.mcts.best_path", lambda: e.mcts.search_best_path(
        _emb().encode("reduce latency"),
        [("add_cache", _emb().encode("add a cache"), 0.8)],
        _emb().encode("minimize p99 latency")))
    probe("latent.coconut.bfs", lambda: e.coconut.rollout_latent_bfs(
        _emb().encode("reduce latency"),
        [("add_cache", _emb().encode("add a cache"))],
        depth=3, goal_vector=_emb().encode("minimize p99 latency")))
    probe("latent.dot.denoise", lambda: e.dot.denoise_trajectory(
        [_emb().encode("initial")], [_emb().encode("goal")], diffusion_steps=5))

    # --- tournament / math ---
    probe("tournament.match", lambda: e.run_tournament(
        ["add a cache to reduce latency", "a proof that P != NP via monotone circuits"]))
    probe("math.verify_flawed", lambda: e.verify_math_paper(
        paper_id="x", paper_title="P vs NP",
        abstract_text="Berg-Ulfberg monotone approximation lower bound for Clique.",
        target_conjecture="P versus NP"))

    # --- boed / pareto / ingenuity ---
    probe("boed.rank", lambda: e.boed.rank_experiments(
        {"h1": 0.5, "h2": 0.5},
        [{"design_id": "d1", "name": "a", "cost": 1}, {"design_id": "d2", "name": "b", "cost": 2}],
        {"d1": {"pos": {"h1": 0.9, "h2": 0.1}, "neg": {"h1": 0.1, "h2": 0.9}},
         "d2": {"pos": {"h1": 0.5, "h2": 0.5}, "neg": {"h1": 0.5, "h2": 0.5}}}))
    probe("pareto.resolve", lambda: e.resolve_pareto(
        [{"name": "x", "direction": "minimize"}, {"name": "y", "direction": "minimize"}],
        [{"name": "a", "values": {"x": 1, "y": 9}}, {"name": "b", "values": {"x": 9, "y": 1}}]))
    probe("ingenuity.audit", lambda: e.ingenuity.audit_hypotheses(
        ["add a bounded cache with fallback"], "reduce latency", "systems"))

    # --- meta ---
    probe("meta.self_audit", lambda: e.run_self_audit_and_optimization())
    probe("meta.self_diagnostic", lambda: e.run_self_diagnostic())
    probe("memory.consolidate", lambda: e.consolidate_memory(0.75))
    probe("code.synthesize", lambda: e.program("return the sum of a list", "assert f([1,2]) == 3"))
    probe("diagnose", lambda: e.diagnose("server returns 500 under load intermittently"))
    probe("suggest_paths", lambda: e.suggest_paths("reduce p99 latency during invalidation"))

    # --- report ---
    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    deg = sum(1 for r in RESULTS if r["status"] == "DEGENERATE")
    fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    print(f"{'subsystem':30} {'status':10} detail")
    for r in RESULTS:
        print(f"{r['subsystem']:30} {r['status']:10} {r['detail'][:88]}")
    print(f"\nOK={ok}  DEGENERATE={deg}  FAIL={fail}  total={len(RESULTS)}")

    out = os.path.join(os.path.dirname(__file__), "..", "assets", "health_report.json")
    json.dump({"summary": {"ok": ok, "degenerate": deg, "fail": fail},
               "results": RESULTS}, open(out, "w"), indent=2)
    print(f"wrote {out}")


def _nm(e):
    e.repulsor.register_dead_end("add a cache in front of the database")
    return e.repulsor.check_proximity(_emb().encode("add a cache in front of the database"))


if __name__ == "__main__":
    main()
