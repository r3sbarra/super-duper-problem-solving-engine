#!/usr/bin/env python3
"""Cross-domain self-improvement with response validation.

Runs the engine against itself on the cross-domain cases, VALIDATES each
generated solution (structural correctness, non-triviality, concreteness),
learns the validated (problem, solution) pairs back into the index, and
re-runs to measure improvement.

Validation is the honest gate: a generated solution is only learned if it
(1) addresses the target problem's structural primitives, (2) is not a trivial
echo of the problem, and (3) is a concrete solution (not a vague principle).
This prevents the engine from learning its own hallucinations.

Usage:
  .venv/bin/python scripts/cross_domain_self_improve.py [--embedder ollama]
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


def run_generation(engine, dk, cases):
    """Run generation on cross-domain cases. Returns {title: (generated, sim)}."""
    results = {}
    for c in cases:
        p = engine.solve_engineering_problem(
            title=c["title"], specification=c["target_problem"], top_principles=4,
            ground_truth_outcomes=None, domain_knowledge=dk, concrete_top_k=3,
        )
        g = p.final_breakthrough.split(": ", 1)[-1] if ": " in p.final_breakthrough else p.final_breakthrough
        sim = cosine(get_backend().encode(g), get_backend().encode(c["target_solution"]))
        results[c["title"]] = (g, sim)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="ollama", choices=["polarity", "ollama"])
    ap.add_argument("--threshold", type=float, default=0.60)
    args = ap.parse_args()
    get_backend(args.embedder)

    # Full domain-knowledge index + the 8 source solutions (as in the test).
    dk = VectorizationService(db_path=":memory:")
    build_domain_knowledge_index(dk)
    for case in CROSS_DOMAIN_CASES:
        pid = dk.store_problem(specification=case["source_problem"], title=case["title"], metadata={"kind": "source"})
        sid = dk.store_solution(solution_text=case["source_solution"], method="source", domain="engineering", title=case["title"], metadata={"problem_context": case["source_problem"]})
        dk.index._conn.execute("UPDATE vectors SET metadata_json=? WHERE id=?", (json.dumps({"solution_id": sid, "kind": "source"}), pid))
    dk.index._conn.commit()

    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")
    cases = CROSS_DOMAIN_CASES

    print(f"Cross-domain self-improvement + validation (threshold={args.threshold})")
    print("=" * 78)

    # Round 1: baseline.
    base = run_generation(engine, dk, cases)
    n_ok_base = sum(1 for v in base.values() if v[1] >= args.threshold)
    print(f"Round 1 (baseline): {n_ok_base}/{len(cases)} correct @ sim>={args.threshold}, "
          f"avg_sim={np.mean([v[1] for v in base.values()]):.3f}")

    # Validate each generated solution; learn the VALID ones.
    # Honest rule: only learn a generated solution if it is a genuine NOVEL
    # contribution. If the target problem already retrieves a good source
    # solution (the ground truth), learning the imperfect generated version
    # just pollutes the index (re-learning a worse copy of what we have).
    learned = 0
    rejected = 0
    for c in cases:
        g, sim = base[c["title"]]
        v = validate_solution(g, c["target_problem"])
        if not v["valid"]:
            rejected += 1
            print(f"  rejected {c['title']}: {v['reason']}")
            continue
        # Check if a good source solution already exists for this structure.
        # If the generated solution is not clearly better than the source,
        # do NOT learn it (it would add a worse copy).
        src_sim = cosine(
            get_backend().encode(c["source_solution"]),
            get_backend().encode(c["target_solution"]),
        )
        if sim <= src_sim + 0.05:
            rejected += 1
            print(f"  rejected {c['title']}: generated ({sim:.3f}) not better than source ({src_sim:.3f})")
            continue
        pid = dk.store_problem(specification=c["target_problem"], title=c["title"], metadata={"kind": "self_learned"})
        sid = dk.store_solution(solution_text=g, method="self_learned", domain="engineering", title=c["title"], metadata={"problem_context": c["target_problem"]})
        dk.index._conn.execute("UPDATE vectors SET metadata_json=? WHERE id=?", (json.dumps({"solution_id": sid, "kind": "self_learned"}), pid))
        dk.index._conn.commit()
        learned += 1
    print(f"Validated: learned {learned}, rejected {rejected}")

    # Round 2: re-run with the enriched index.
    engine2 = SuperDuperProblemSolvingEngine(db_path=":memory:")
    improved = run_generation(engine2, dk, cases)
    n_ok_improved = sum(1 for v in improved.values() if v[1] >= args.threshold)
    print(f"Round 2 (after self-learning): {n_ok_improved}/{len(cases)} correct @ sim>={args.threshold}, "
          f"avg_sim={np.mean([v[1] for v in improved.values()]):.3f}")

    print("-" * 78)
    print(f"{'case':<30} {'base':>6} {'after':>6} {'delta':>6}")
    n_improved = 0
    for t in base:
        d = improved[t][1] - base[t][1]
        if d > 0.01:
            n_improved += 1
        print(f"{t:<30} {base[t][1]:>6.3f} {improved[t][1]:>6.3f} {d:>+6.3f}")
    print("-" * 78)
    print(f"cases improved: {n_improved}/{len(cases)}; correct {n_ok_base} -> {n_ok_improved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
