"""Verify the recursive loop is closed: the engine retrieves its own fix."""
import os
import sys

os.environ['SUPER_SOLVER_VECTOR_DB'] = ':memory:'
sys.path.insert(0, 'src')
from super_solver.vectorize import vectorization_service  # noqa: E402
from super_solver.vectorize.domain_knowledge import DOMAIN_KNOWLEDGE

for e in DOMAIN_KNOWLEDGE:
    vectorization_service.store_solution(
        solution_text=e['solution_direction'], method='d', domain='d',
        title=e['problem_context'][:50])
from super_solver.frameworks.problem_identification import decompose  # noqa: E402

d = decompose(
    "The engine decomposes problems well but its decomposition covers almost none "
    "of the problem-specific solution idea. Improve the engine so its decompositions "
    "lead to solved problems.")
ideas = d['solution_ideas']
assert ideas, "engine retrieved no solution for its own problem"
assert ideas[0]['overlap'] >= 0.5, f"top idea overlap too low: {ideas[0]['overlap']}"
print("PASS: engine retrieves its own fix ->", ideas[0]['idea'][:70])
