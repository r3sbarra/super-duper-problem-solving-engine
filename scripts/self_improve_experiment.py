#!/usr/bin/env python3
"""Honest self-improvement experiment: does learning from its own successes help?

Setup: start with a PARTIAL domain-knowledge index (only half the fact-base
entries). Run generation — some cases fail because no close fact-base entry
exists. Then self-learn the SUCCESSFUL (problem, solution) pairs into the index
and re-run. Cases whose solution was learned should now retrieve the exact
solution and improve.

This is the honest test of "using itself to improve itself": the engine only
improves if it learns something NEW (a solution not already in its knowledge).

Usage:
  .venv/bin/python scripts/self_improve_experiment.py [--embedder ollama]
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
from super_solver.vectorize.domain_knowledge import (
    DOMAIN_KNOWLEDGE,
    build_domain_knowledge_index,
)

from train_problem_solving import CLASSICAL_CASES


def cosine(a, b):
    a = np.asarray(a, float).reshape(1, -1)
    b = np.asarray(b, float).reshape(1, -1)
    d = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    d[d == 0] = 1e-9
    return float((a @ b.T)[0, 0] / d[0])


def run_generation(engine, dk, cases):
    sims = {}
    for c in cases:
        p = engine.solve_engineering_problem(
            title=c["title"], specification=c["problem"], top_principles=4,
            ground_truth_outcomes=None, domain_knowledge=dk, concrete_top_k=3,
        )
        g = p.final_breakthrough.split(": ", 1)[-1] if ": " in p.final_breakthrough else p.final_breakthrough
        sims[c["title"]] = cosine(get_backend().encode(g), get_backend().encode(c["solution"]))
    return sims


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="ollama", choices=["polarity", "ollama"])
    ap.add_argument("--threshold", type=float, default=0.60)
    args = ap.parse_args()
    get_backend(args.embedder)

    # PARTIAL domain knowledge: only the first half of the fact base.
    partial = DOMAIN_KNOWLEDGE[: len(DOMAIN_KNOWLEDGE) // 2]
    dk = VectorizationService(db_path=":memory:")
    for entry in partial:
        pid = dk.store_problem(specification=entry["problem_context"], title=entry["problem_context"][:60], metadata={"kind": "domain_knowledge"})
        sid = dk.store_solution(solution_text=entry["solution_direction"], method="domain_knowledge", domain="engineering", title=entry["problem_context"][:60], metadata={"problem_context": entry["problem_context"]})
        import json
        dk.index._conn.execute("UPDATE vectors SET metadata_json=? WHERE id=?", (json.dumps({"solution_id": sid, "kind": "domain_knowledge"}), pid))
    dk.index._conn.commit()

    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")
    cases = CLASSICAL_CASES

    print(f"Self-improvement experiment: partial knowledge ({len(partial)}/{len(DOMAIN_KNOWLEDGE)} entries), "
          f"threshold={args.threshold}")
    print("=" * 78)

    # Round 1: baseline with partial knowledge.
    base = run_generation(engine, dk, cases)
    n_ok_base = sum(1 for v in base.values() if v >= args.threshold)
    print(f"Round 1 (partial knowledge): {n_ok_base}/{len(cases)} correct @ sim>={args.threshold}, "
          f"avg_sim={np.mean(list(base.values())):.3f}")

    # Self-learn the successes into the index (LINKED so hybrid_analogize resolves).
    import json

    learned = 0
    for c in cases:
        if base[c["title"]] >= args.threshold:
            pid = dk.store_problem(specification=c["problem"], title=c["title"], metadata={"kind": "self_learned"})
            sid = dk.store_solution(solution_text=c["solution"], method="self_learned", domain="engineering", title=c["title"], metadata={"problem_context": c["problem"]})
            dk.index._conn.execute("UPDATE vectors SET metadata_json=? WHERE id=?", (json.dumps({"solution_id": sid, "kind": "self_learned"}), pid))
            dk.index._conn.commit()
            learned += 1
    print(f"Self-learned {learned} (problem, solution) pairs from successes")

    # Round 2: re-run with the enriched index.
    engine2 = SuperDuperProblemSolvingEngine(db_path=":memory:")
    improved = run_generation(engine2, dk, cases)
    n_ok_improved = sum(1 for v in improved.values() if v >= args.threshold)
    print(f"Round 2 (after self-learning): {n_ok_improved}/{len(cases)} correct @ sim>={args.threshold}, "
          f"avg_sim={np.mean(list(improved.values())):.3f}")

    print("-" * 78)
    print(f"{'case':<26} {'base':>6} {'after':>6} {'delta':>6}")
    n_improved = 0
    for t in base:
        d = improved[t] - base[t]
        if d > 0.01:
            n_improved += 1
        print(f"{t:<26} {base[t]:>6.3f} {improved[t]:>6.3f} {d:>+6.3f}")
    print("-" * 78)
    print(f"cases improved: {n_improved}/{len(cases)}; correct {n_ok_base} -> {n_ok_improved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
