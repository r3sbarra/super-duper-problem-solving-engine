#!/usr/bin/env python3
"""Novel-problem self-improvement: does learning from its own successes help
on genuinely NEW cross-domain problems?

Setup: start with a PARTIAL index (only half the source solutions). Present a
NOVEL cross-domain problem (not in the index). The engine generates a solution
from what it has. If the generated solution is VALID (passes the validation
gate) and NOVEL (better than what's in the index), learn it. Then present a
SECOND novel problem of the same structure and check whether the learned
solution helps.

This is the honest test of "using itself to improve itself" for cross-domain:
the engine only improves if it learns something NEW it didn't already have.

Usage:
  .venv/bin/python scripts/novel_self_improve.py [--embedder ollama]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from super_solver.core.embedder import get_backend
from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.vectorize import VectorizationService
from super_solver.vectorize.domain_knowledge import build_domain_knowledge_index
from super_solver.vectorize.primitives import validate_solution

from test_cross_domain import CROSS_DOMAIN_CASES


def cosine(a, b):
    a = np.asarray(a, float).reshape(1, -1)
    b = np.asarray(b, float).reshape(1, -1)
    d = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    d[d == 0] = 1e-9
    return float((a @ b.T)[0, 0] / d[0])


def run_generation(engine, dk, problem, title):
    """Run generation on a single problem. Returns (generated, sim_to_expected)."""
    p = engine.solve_engineering_problem(
        title=title, specification=problem, top_principles=4,
        ground_truth_outcomes=None, domain_knowledge=dk, concrete_top_k=3,
    )
    g = p.final_breakthrough.split(": ", 1)[-1] if ": " in p.final_breakthrough else p.final_breakthrough
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="ollama", choices=["polarity", "ollama"])
    args = ap.parse_args()
    get_backend(args.embedder)

    # Two novel problems of the SAME structure (adhesion/mimicry), neither in
    # the index. The engine should learn the first and apply it to the second.
    NOVEL_PROBLEMS = [
        {
            "title": "Novel A: gecko climbing",
            "problem": "How to make a reusable wall anchor that holds on smooth painted drywall without screws or adhesive residue",
            "expected": "Mimic the microscopic setae structure of gecko feet that adhere to smooth surfaces",
        },
        {
            "title": "Novel B: octopus suction",
            "problem": "How to make a temporary camera mount that grips a wet glass window without leaving marks",
            "expected": "Mimic the suction-cup structure of octopus tentacles that grip smooth wet surfaces",
        },
    ]

    # PARTIAL index: only the first 4 source solutions (not the adhesion ones).
    dk = VectorizationService(db_path=":memory:")
    build_domain_knowledge_index(dk)
    for case in CROSS_DOMAIN_CASES[:4]:
        pid = dk.store_problem(specification=case["source_problem"], title=case["title"], metadata={"kind": "source"})
        sid = dk.store_solution(solution_text=case["source_solution"], method="source", domain="engineering", title=case["title"], metadata={"problem_context": case["source_problem"]})
        dk.index._conn.execute("UPDATE vectors SET metadata_json=? WHERE id=?", (json.dumps({"solution_id": sid, "kind": "source"}), pid))
    dk.index._conn.commit()

    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")

    print("Novel-problem self-improvement (partial index, adhesion cases ABSENT)")
    print("=" * 78)

    # Problem A: generate, validate, learn.
    gA = run_generation(engine, dk, NOVEL_PROBLEMS[0]["problem"], NOVEL_PROBLEMS[0]["title"])
    simA = cosine(get_backend().encode(gA), get_backend().encode(NOVEL_PROBLEMS[0]["expected"]))
    vA = validate_solution(gA, NOVEL_PROBLEMS[0]["problem"])
    print(f"Problem A ({NOVEL_PROBLEMS[0]['title']}):")
    print(f"  generated: {gA[:80]}")
    print(f"  sim_to_expected: {simA:.3f}  valid: {vA['valid']} ({vA['reason']})")

    learned = False
    if vA["valid"]:
        pid = dk.store_problem(specification=NOVEL_PROBLEMS[0]["problem"], title=NOVEL_PROBLEMS[0]["title"], metadata={"kind": "self_learned"})
        sid = dk.store_solution(solution_text=gA, method="self_learned", domain="engineering", title=NOVEL_PROBLEMS[0]["title"], metadata={"problem_context": NOVEL_PROBLEMS[0]["problem"]})
        dk.index._conn.execute("UPDATE vectors SET metadata_json=? WHERE id=?", (json.dumps({"solution_id": sid, "kind": "self_learned"}), pid))
        dk.index._conn.commit()
        learned = True
        print(f"  -> learned into index")
    else:
        print(f"  -> NOT learned (invalid)")

    # Problem B: generate with the enriched index.
    gB = run_generation(engine, dk, NOVEL_PROBLEMS[1]["problem"], NOVEL_PROBLEMS[1]["title"])
    simB = cosine(get_backend().encode(gB), get_backend().encode(NOVEL_PROBLEMS[1]["expected"]))
    print(f"\nProblem B ({NOVEL_PROBLEMS[1]['title']}):")
    print(f"  generated: {gB[:80]}")
    print(f"  sim_to_expected: {simB:.3f}")

    print("-" * 78)
    if learned:
        print(f"Learned problem A's solution; problem B sim = {simB:.3f}")
        print("Verdict: self-improvement transferred the learned solution to a novel problem" if simB >= 0.5 else "Verdict: learned solution did NOT transfer to the novel problem")
    else:
        print("Problem A's solution was invalid; nothing learned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
