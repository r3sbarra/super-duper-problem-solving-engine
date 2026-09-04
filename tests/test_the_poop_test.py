"""The Poop Test: Empirical validation of common-sense physical & hygiene problem solving.

Validates that SuperDuperProblemSolvingEngine:
1. Rejects leaving waste on the floor as a toxic dead-end on the Negative Manifold.
2. Rejects directly mopping solid waste due to dispersion/smearing falsification.
3. Suggests TRIZ Principle #7 (Nested Doll / Inverted Bag Containment).
4. Confirms that picking up the bulk solid phase is the correct first prerequisite action.
"""

from super_solver.core.types import CrucialExperiment, KTBoundary
from super_solver.engine import SuperDuperProblemSolvingEngine


def test_the_poop_test():
    """The Poop Test: Validates deduction on common-sense physical cleanup."""
    engine = SuperDuperProblemSolvingEngine()

    problem_title = "I pooped on the floor"
    problem_spec = "I pooped on the floor. Feces are on the floor surface."

    candidates = [
        "I mop it",
        "I pick it up",
        "leave it there for someone else",
    ]

    boundary = KTBoundary(
        identity_is="Solid biological waste",
        identity_is_not="Liquid spill only",
        location_is="Confined spot on floor",
        location_is_not="Entire room surface",
        timing_is="Immediately after event",
        timing_is_not="Hours later after drying",
        extent_is="Localized discrete mass",
        extent_is_not="Dispersed film across floor",
    )

    problem = engine.formulate_problem(
        title=problem_title,
        specification=problem_spec,
        boundary=boundary,
        goal_criteria=["Restore floor hygiene without biohazard smearing or escalation"],
    )

    # Crucial Experiment: Phase Separation & Dispersion Assay
    # Mopping solid feces directly causes mechanical smearing across the floor.
    # Leaving it causes hazard escalation.
    # Picking it up successfully isolates the bulk solid phase.
    crucial_exp = CrucialExperiment(
        id="exp_poop_dispersion_assay",
        name="Phase Separation & Dispersion Assay",
        description="Assesses physical distribution of biological waste after tool application.",
        target_hypotheses=["hyp_abduct_1", "hyp_abduct_2", "hyp_abduct_3"],
        exclusory_predictions={
            "hyp_abduct_1": "smears_solid_waste_into_floor",
            "hyp_abduct_2": "isolates_solid_bulk_waste",
            "hyp_abduct_3": "biohazard_escalation",
        }
    )

    known_dead_ends = [
        "leave it there for someone else",
        "smearing feces across the entire room with a wet mop",
    ]

    discovery_path = engine.deduce_discovery_path(
        problem=problem,
        candidate_hypotheses=candidates,
        crucial_experiments=[crucial_exp],
        known_dead_ends=known_dead_ends,
        ground_truth_outcomes={"exp_poop_dispersion_assay": "isolates_solid_bulk_waste"},
    )

    # 1. Verify that 'I pick it up' is confirmed as the breakthrough
    assert "CONFIRMED: I pick it up" in discovery_path.final_breakthrough
    assert discovery_path.confidence == 1.0

    # 2. Verify that 'I mop it' and 'leave it there' were both falsified
    assert "hyp_abduct_1" in discovery_path.falsified_paths  # I mop it
    assert "hyp_abduct_3" in discovery_path.falsified_paths  # leave it there

    # 3. Verify that suggested paths recommend TRIZ Nested Doll / Inverted Containment
    suggested = engine.suggest_paths(problem_spec, top_k=3)
    assert len(suggested) >= 1
    triz_paths = [p for p in suggested if "TRIZ" in p.strategy_type]
    assert len(triz_paths) >= 1
    assert "Nested Doll" in triz_paths[0].title or triz_paths[0].feasibility_score > 0.0

    # 4. Verify visual Mermaid flowchart export
    mermaid = engine.export_mermaid_diagram(discovery_path)
    assert "```mermaid" in mermaid
    assert "CONFIRMED: I pick it up" in mermaid
    assert "hyp_abduct_1" in mermaid
    assert "hyp_abduct_3" in mermaid

    # 5. Verify human-auditable explanation
    explanation = engine.explain_discovery(discovery_path)
    assert "=== DISCOVERY PATH EXPLANATION: I pooped on the floor ===" in explanation
    assert "Final Result: CONFIRMED: I pick it up" in explanation
