#!/usr/bin/env python3
"""Self-improvement loop: super-solver learns from its own generation successes.

Runs the full discovery pipeline (TRIZ-grounded + domain-knowledge grounding)
on the training corpus. For every case where the generated breakthrough
semantically matches the known solution (sim >= threshold), the (problem,
solution) pair is added to the domain-knowledge index — so the engine literally
learns from its own successes and improves future generation.

This is "using itself to improve itself": the engine's own correct outputs
become part of its grounding knowledge base.

Usage:
  .venv/bin/python scripts/self_improve_generation.py [--embedder ollama] [--recent]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from super_solver.core.embedder import get_backend
from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.vectorize import VectorizationService
from super_solver.vectorize.domain_knowledge import build_domain_knowledge_index

from train_problem_solving import CLASSICAL_CASES, RECENT_CASES


def cosine(a, b):
    a = np.asarray(a, dtype=float).reshape(1, -1)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom[denom == 0] = 1e-9
    return float((a @ b.T)[0, 0] / denom[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="ollama", choices=["polarity", "ollama"])
    ap.add_argument("--recent", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="sim threshold to count a generated solution as a success")
    args = ap.parse_args()

    get_backend(args.embedder)

    cases = list(CLASSICAL_CASES)
    if args.recent:
        cases += list(RECENT_CASES)

    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")
    dk = VectorizationService(db_path=":memory:")
    build_domain_knowledge_index(dk)

    print(f"Self-improvement loop on {len(cases)} cases, embedder={args.embedder}, "
          f"threshold={args.threshold}")
    print("=" * 78)

    successes = 0
    learned = 0
    for case in cases:
        title = case["title"]
        path = engine.solve_engineering_problem(
            title=title,
            specification=case["problem"],
            goal_criteria=["Find a novel solution to this problem"],
            top_principles=4,
            ground_truth_outcomes=None,
            domain_knowledge=dk,
            concrete_top_k=3,
        )
        gen_text = path.final_breakthrough.split(": ", 1)[-1] if ": " in path.final_breakthrough else path.final_breakthrough
        sim = cosine(get_backend().encode(gen_text), get_backend().encode(case["solution"]))
        ok = sim >= args.threshold
        if ok:
            successes += 1
            # Learn from success: add (problem, solution) pair, LINKED, to the
            # domain index so hybrid_analogize can resolve the solution.
            pid = dk.store_problem(
                specification=case["problem"],
                title=title,
                metadata={"kind": "self_learned"},
            )
            sid = dk.store_solution(
                solution_text=case["solution"],
                method="self_learned",
                domain="engineering",
                title=title,
                metadata={"problem_context": case["problem"]},
            )
            import json

            dk.index._conn.execute(
                "UPDATE vectors SET metadata_json=? WHERE id=?",
                (json.dumps({"solution_id": sid, "kind": "self_learned"}), pid),
            )
            dk.index._conn.commit()
            learned += 1
        mark = "✓" if ok else "~"
        print(f"  {mark} {title:<28} sim={sim:.3f} {'LEARNED' if ok else ''}")

    print("=" * 78)
    print(f"RESULT: {successes}/{len(cases)} generated correctly @ sim>={args.threshold}; "
          f"{learned} (problem, solution) pairs added to domain knowledge")
    return 0


if __name__ == "__main__":
    sys.exit(main())
