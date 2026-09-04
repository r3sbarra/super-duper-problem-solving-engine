#!/usr/bin/env python3
"""Example: Formulating and deducing the LK-99 superconductivity anomaly."""

from super_solver.core.types import KTBoundary
from super_solver.engine import SuperDuperProblemSolvingEngine


def main():
    print("=== Super-Duper-Problem-Solving-Engine: LK-99 Anomaly Deduction ===")
    engine = SuperDuperProblemSolvingEngine()

    # 1. Formulate problem with Kepner-Tregoe 4D Boundary Analysis
    boundary = KTBoundary(
        identity_is="Resistivity drop occurs sharply at 104 C (377 K)",
        identity_is_not="Does not reach true zero electrical resistance",
        location_is="Multiphase sintered pellets containing Cu2S secondary phases",
        location_is_not="Pure transparent stoichiometric single crystals",
        timing_is="Occurs strictly during 104 C heating/cooling thermal cycle",
        timing_is_not="Continuously variable with copper substitution concentration",
        extent_is="Only certain inhomogeneous fragments show half-levitation",
        extent_is_not="Bulk homogeneous superconductivity throughout entire ingot",
    )

    problem = engine.formulate_problem(
        title="LK-99 Ambient-Temperature Superconductivity Anomaly",
        specification="Reported room-temperature superconductivity in modified lead apatite Pb10-x Cux (PO4)6O.",
        boundary=boundary,
        goal_criteria=["Identify true physical mechanism of 104 C transition"],
    )

    print(f"Formulated Problem: {problem.title}")
    print(f"Goal Criteria: {problem.goal_criteria}")

    # 2. Query TRIZ inventive principles for contradiction resolution
    triz_matches = engine.triz.suggest_principles(problem.specification, top_k=3)
    print("\nSuggested TRIZ Principles:")
    for match in triz_matches:
        print(f"  - Principle #{match['principle_id']} ({match['name']}): {match['similarity_score']:.3f}")

    # 3. Deduce discovery path with Strong Inference falsification
    path = engine.deduce_discovery_path(
        problem=problem,
        candidate_hypotheses=[
            "Room-temperature ambient-pressure room-temperature superconductor",
            "Cu2S structural first-order phase transition artifact at 104 C",
            "Diamagnetic half-levitation via ferromagnetism artifact",
        ],
        known_dead_ends=[
            "Room-temperature ambient-pressure room-temperature superconductor",
        ],
        ground_truth_outcomes={
            "exp_auto_purity_control": "anomaly_vanishes_in_pure_sample",
        },
    )

    print(f"\nFinal Breakthrough: {path.final_breakthrough}")
    print(f"Falsified Paths: {path.falsified_paths}")
    print(f"Total Steps: {path.total_steps}")


if __name__ == "__main__":
    main()
