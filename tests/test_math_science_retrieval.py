"""Math/science primitive-retrieval coverage test.

Proves the extension added for suuri-chan coverage integration: novel,
rephrased math and science problems must retrieve the structurally correct
concrete solution from the domain-knowledge fact base via the primitive
matcher (not string matching).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.vectorize.domain_knowledge import DOMAIN_KNOWLEDGE
from super_solver.vectorize.primitives import match_by_primitives, strip_to_primitives

_SOURCES = [
    {"solution": e["solution_direction"], "title": e["problem_context"][:50]}
    for e in DOMAIN_KNOWLEDGE
]

# (rephrased problem, substrings expected in the retrieved solution)
MATH_SCIENCE_CASES = [
    ("Work out the total of the first hundred whole numbers without a calculator", ["gauss", "arithmetic", "pair"]),
    ("How do I check whether a huge number has any divisors other than 1 and itself", ["miller-rabin", "witness", "primal"]),
    ("Find a common measure shared by two big whole numbers", ["euclidean", "gcd"]),
    ("I have noisy measurements and want a line that fits them best", ["least-squares", "regression", "best-fit"]),
    ("Tell me if two groups really differ rather than by chance", ["t-test", "p-value"]),
    ("Prove something is true for every whole number by stepping up from a base", ["induction", "base case"]),
    ("Which conserved quantity follows from a system's symmetry", ["noether", "conserved"]),
    ("Model how an infection moves through a population", ["sir", "compartmental"]),
    ("How does a chemical reaction speed up when things get hotter", ["arrhenius"]),
    ("How many points lie on a cubic curve over a finite field", ["point counting", "hasse"]),
    ("Find all the places a polynomial crosses zero on an interval", ["sturm", "isolate"]),
    ("Show a fact about triangles using algebra", ["wu", "pseudodivide"]),
    ("Decide if an identity is actually true by hunting for counterexamples", ["counterexample", "falsif"]),
    ("Fit a curve through measurement points", ["least-squares", "regression"]),
]


def test_math_science_retrieval_high_accuracy():
    hits = 0
    misses = []
    for query, expected in MATH_SCIENCE_CASES:
        matches = match_by_primitives(query, _SOURCES)
        top = matches[0]["content"].lower() if matches else ""
        if any(k in top for k in expected):
            hits += 1
        else:
            misses.append(query)
    accuracy = hits / len(MATH_SCIENCE_CASES)
    assert accuracy >= 0.85, f"math/science retrieval {hits}/{len(MATH_SCIENCE_CASES)}; misses={misses}"


def test_new_math_science_solutions_are_self_describing():
    """Every math/science entry added for coverage must strip to at least one
    action primitive (an empty solution never matches and silently fails)."""
    markers = ("gcd", "prime", "integral", "induction", "statistic", "regression",
               "sturm", "noether", "sir model", "arrhenius", "elliptic", "knot",
               "geometry", "least-squares", "hypothesis", "binomial", "counterexample",
               "schrodinger", "logistic", "point counting", "descriptive", "algebraic")
    checked = 0
    for entry in DOMAIN_KNOWLEDGE:
        text = (entry["problem_context"] + " " + entry["solution_direction"]).lower()
        if not any(m in text for m in markers):
            continue
        checked += 1
        prims = strip_to_primitives(entry["solution_direction"])
        assert prims, f"math/science solution stripped to nothing: {entry['solution_direction'][:60]}"
    assert checked >= 15
