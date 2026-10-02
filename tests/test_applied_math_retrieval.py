"""Applied-math / science retrieval coverage (wave 2).

Proves the extended domain knowledge + primitives retrieve the correct
concrete solution for standard applied-mathematics, discrete-math, information-
theory, logic, and science-math problems that were previously uncovered
(linear-algebra algorithms, optimization, graph theory, signal processing,
estimation, termination proofs, symbolic algebra).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.vectorize.domain_knowledge import DOMAIN_KNOWLEDGE
from super_solver.vectorize.primitives import match_by_primitives

_SOURCES = [
    {"solution": e["solution_direction"], "title": e["problem_context"][:50]}
    for e in DOMAIN_KNOWLEDGE
]

CASES = [
    ("Prove that a program halts using a well-founded measure", ["termination", "well-founded", "ranking"]),
    ("Find the shortest route visiting every city once", ["tsp", "travelling", "held-karp", "hamiltonian"]),
    ("Compute the determinant of a large matrix exactly", ["bareiss", "determinant", "fraction-free"]),
    ("Find a Groebner basis for a polynomial ideal", ["groebner", "buchberger"]),
    ("Determine if a graph is bipartite", ["bipartite", "two-coloring", "2-color"]),
    ("Compute the entropy of a probability distribution", ["entropy", "shannon", "bits"]),
    ("Find the eigenvalues of a matrix", ["eigenvalue", "characteristic polynomial"]),
    ("Solve a linear program", ["simplex", "linear program", "polytope"]),
    ("Estimate a parameter from noisy observations", ["likelihood", "mle", "posterior"]),
    ("Compute a Fourier transform", ["fourier"]),
    ("Balance a chemical reaction", ["stoichiometric", "conservation"]),
    ("Decide whether a Boolean formula is satisfiable", ["sat", "dpll", "satisfying"]),
    ("Find the maximum flow in a network", ["flow", "augmenting"]),
    ("Fit a nonlinear model to data", ["least squares", "gauss-newton", "levenberg"]),
]


def test_applied_math_science_retrieval():
    hits = 0
    misses = []
    for query, expected in CASES:
        matches = match_by_primitives(query, _SOURCES)
        top = matches[0]["content"].lower() if matches else ""
        if any(k in top for k in expected):
            hits += 1
        else:
            misses.append(query)
    accuracy = hits / len(CASES)
    assert accuracy >= 0.85, f"applied retrieval {hits}/{len(CASES)}; misses={misses}"


def test_domain_knowledge_grew():
    assert len(DOMAIN_KNOWLEDGE) >= 155
