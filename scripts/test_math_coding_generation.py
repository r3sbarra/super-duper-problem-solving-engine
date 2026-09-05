#!/usr/bin/env python3
"""Test math + coding GENERATION (not just retrieval).

Runs the full discovery pipeline (solve_engineering_problem) on NOVEL math and
coding problems — problems NOT in the training corpus — and checks whether the
engine GENERATES the correct solution via domain-knowledge grounding + primitive
matching. This is the honest test of "can it solve math/coding problems", not
just retrieve known ones.

Usage:
  .venv/bin/python scripts/test_math_coding_generation.py [--embedder ollama]
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

# Novel math/coding problems NOT in the training corpus. Each has the expected
# solution direction (the algorithm/technique it should generate).
NOVEL_MATH_CODING = [
    {
        "title": "Novel: sum of squares",
        "problem": "How to compute the sum of the squares of the first n integers without adding each one",
        "expected": "Gauss",
    },
    {
        "title": "Novel: find in rotated array",
        "problem": "How to search for a value in a sorted array that has been rotated at an unknown point",
        "expected": "binary search",
    },
    {
        "title": "Novel: two-sum",
        "problem": "How to find two numbers in an array that add up to a given target in one pass",
        "expected": "hash",
    },
    {
        "title": "Novel: merge sorted",
        "problem": "How to combine two already-sorted lists into one sorted list efficiently",
        "expected": "merge",
    },
    {
        "title": "Novel: max flow",
        "problem": "How to find the maximum amount of data that can flow from a source server to a sink through a network with capacity limits",
        "expected": "augmenting path",
    },
    {
        "title": "Novel: kth smallest",
        "problem": "How to find the 5th smallest number in a large unsorted list without sorting the whole thing",
        "expected": "quickselect",
    },
    {
        "title": "Novel: modular inverse",
        "problem": "How to find the modular inverse of a number under a prime modulus for a cryptographic key",
        "expected": "extended euclid",
    },
    {
        "title": "Novel: islands",
        "problem": "How to count the number of separate islands in a grid of land and water cells",
        "expected": "flood-fill",
    },
]


def cosine(a, b):
    a = np.asarray(a, float).reshape(1, -1)
    b = np.asarray(b, float).reshape(1, -1)
    d = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    d[d == 0] = 1e-9
    return float((a @ b.T)[0, 0] / d[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="ollama", choices=["polarity", "ollama"])
    args = ap.parse_args()
    get_backend(args.embedder)

    dk = VectorizationService(backend=get_backend(args.embedder), db_path=":memory:")
    build_domain_knowledge_index(dk)
    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")

    print(f"Math/coding GENERATION test ({args.embedder})")
    print("=" * 78)
    correct = 0
    for case in NOVEL_MATH_CODING:
        p = engine.solve_engineering_problem(
            title=case["title"], specification=case["problem"], top_principles=4,
            ground_truth_outcomes=None, domain_knowledge=dk, concrete_top_k=3,
        )
        g = p.final_breakthrough.split(": ", 1)[-1] if ": " in p.final_breakthrough else p.final_breakthrough
        ok = case["expected"].lower() in g.lower()
        if ok:
            correct += 1
        print(f'{"✓" if ok else "~"} {case["title"]:<28}')
        print(f'    generated: {g[:90]}')
        print(f'    expected:  {case["expected"]}')
    print("-" * 78)
    print(f"RESULT: {correct}/{len(NOVEL_MATH_CODING)} correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
