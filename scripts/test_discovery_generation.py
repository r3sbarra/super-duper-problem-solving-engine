#!/usr/bin/env python3
"""Test super-solver's FULL DISCOVERY pipeline (generation, not retrieval).

For each training case, run the complete discovery pipeline
(Peirce abduction -> KT boundary -> latent rollout -> MCTS -> breakthrough)
with NO ground truth, so the engine must GENERATE a solution from scratch
rather than retrieve a known one. Then check whether the generated
breakthrough semantically matches the known novel solution.

This is the honest generation test: the engine never sees the solution text.

Usage:
  .venv/bin/python scripts/test_discovery_generation.py [--embedder ollama] [--recent] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.core.embedder import get_backend  # noqa: E402
from super_solver.engine import SuperDuperProblemSolvingEngine  # noqa: E402

from train_problem_solving import CLASSICAL_CASES, RECENT_CASES  # noqa: E402


def cosine(a, b):
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom[denom == 0] = 1e-9
    return float((a @ b.T)[0, 0] / denom[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="ollama", choices=["polarity", "ollama"])
    ap.add_argument("--recent", action="store_true", help="include recent 2024-26 cases")
    ap.add_argument("--limit", type=int, default=0, help="max cases to run (0 = all)")
    args = ap.parse_args()

    get_backend(args.embedder)  # select embedder before engine init

    cases = list(CLASSICAL_CASES)
    if args.recent:
        cases += list(RECENT_CASES)
    if args.limit:
        cases = cases[: args.limit]

    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")

    print(f"Testing FULL DISCOVERY (generation) on {len(cases)} cases, embedder={args.embedder}")
    print("=" * 78)

    correct = 0
    total = 0
    sims = []
    for case in cases:
        title = case["title"]
        # Run the full pipeline with TRIZ-grounded abduction and NO ground truth
        # -> genuine generation of an engineering solution, never sees the answer.
        path = engine.solve_engineering_problem(
            title=title,
            specification=case["problem"],
            goal_criteria=["Find a novel solution to this problem"],
            top_principles=4,
            ground_truth_outcomes=None,  # NO ground truth -> UNVERIFIED, honest
        )
        breakthrough = path.final_breakthrough
        # Strip the UNVERIFIED/CONFIRMED prefix to get the generated claim.
        gen_text = breakthrough.split(": ", 1)[-1] if ": " in breakthrough else breakthrough

        # Semantic match: generated breakthrough vs known solution.
        sim = cosine(
            get_backend().encode(gen_text),
            get_backend().encode(case["solution"]),
        )
        sims.append(sim)
        total += 1
        ok = sim >= 0.80
        if ok:
            correct += 1
        mark = "✓" if ok else "~"
        print(f"  {mark} {title:<28} sim={sim:.3f}  steps={path.total_steps}")
        print(f"      generated: {gen_text[:110]}")
        print(f"      known:     {case['solution'][:110]}")

    print("=" * 78)
    avg = sum(sims) / len(sims) if sims else 0.0
    print(f"RESULT: generated-match {correct}/{total} correct @ sim>=0.80, avg_sim={avg:.3f}")
    print("(generation, not retrieval: engine never saw the solution text)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
