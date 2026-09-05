"""Tests for advanced features: TRIZ Contradiction Matrix, KT 4D Diagnostics,
VSA Cleanup Memory, Bayesian updating, and Mermaid Diagram export.
"""

from super_solver.core.types import CrucialExperiment, Hypothesis, KTBoundary
from super_solver.core.vsa import VSAEngine
from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.frameworks.kepner_tregoe import KepnerTregoeEngine
from super_solver.frameworks.platt_inference import StrongInferenceEngine
from super_solver.frameworks.triz_engine import TRIZEngine


def test_triz_contradiction_matrix_lookup():
    engine = TRIZEngine()
    results = engine.resolve_contradiction_matrix("speed", "accuracy")
    assert len(results) >= 3
    assert results[0]["source"] == "altshuller_matrix"
    principle_names = [p["name"] for p in results]
    assert "Mechanics Substitution" in principle_names or "Preliminary Action" in principle_names


def test_kepner_tregoe_4d_dimensional_diagnostics():
    engine = KepnerTregoeEngine()
    boundary = KTBoundary(
        identity_is="Laser diode failure on spectrometer unit 4",
        identity_is_not="Other diodes or units",
        location_is="Optical bench room 102",
        location_is_not="Storage or outdoor testing",
        timing_is="Occurs strictly between 2pm and 4pm",
        timing_is_not="Morning or night shifts",
        extent_is="Optical intensity drops by 80%",
        extent_is_not="Total electrical burnout",
    )

    cause = "Direct afternoon sunlight reflecting through room 102 window between 2pm and 4pm heating laser diode"
    res = engine.evaluate_candidate_cause(cause, boundary)
    assert res["valid_boundary_fit"] is True
    assert "dimensional_breakdown" in res
    assert "timing" in res["dimensional_breakdown"]
    assert res["dimensional_breakdown"]["timing"]["dimension_passed"] is True


def test_vsa_cleanup_memory():
    vsa = VSAEngine(dim=2048, seed=999)
    v_dog = vsa.random_hypervector(symbol="DOG")
    vsa.random_hypervector(symbol="CAT")

    # Add 15% unit noise
    noise = vsa.random_hypervector()
    noisy_dog = v_dog + (0.15 * noise)
    symbol, sim = vsa.cleanup.clean(noisy_dog)

    assert symbol == "DOG"
    assert sim > 0.85


def test_platt_noisy_bayesian_update():
    engine = StrongInferenceEngine()
    h1 = Hypothesis(id="h1", title="H1", description="Theory 1", current_confidence=0.5)
    h2 = Hypothesis(id="h2", title="H2", description="Theory 2", current_confidence=0.5)

    exp = CrucialExperiment(
        id="e1",
        name="Noisy test",
        description="Assay with 10% measurement error",
        target_hypotheses=["h1", "h2"],
        exclusory_predictions={"h1": "pos", "h2": "neg"},
    )

    updated, falsified = engine.execute_and_prune(
        exp, [h1, h2], observed_outcome="pos", error_rate=0.1
    )
    assert updated[0].current_confidence > 0.85
    assert updated[1].current_confidence < 0.15


def test_mermaid_and_explanation_export():
    engine = SuperDuperProblemSolvingEngine()
    prob = engine.formulate_problem(
        title="Superconducting Transition Demarcation",
        specification="104C drop in resistivity in Pb-Cu-P-O apatite",
    )
    path = engine.deduce_discovery_path(
        problem=prob,
        candidate_hypotheses=["Superconductor", "Cu2S phase transition"],
        crucial_experiments=[
            CrucialExperiment(
                id="e1",
                name="Pure crystal test",
                description="Test without Cu2S",
                target_hypotheses=["hyp_abduct_1", "hyp_abduct_2"],
                exclusory_predictions={
                    "hyp_abduct_1": "superconducts",
                    "hyp_abduct_2": "insulator",
                },
            )
        ],
        ground_truth_outcomes={"e1": "insulator"},
    )

    mermaid_code = engine.export_mermaid_diagram(path)
    assert "```mermaid" in mermaid_code
    assert "graph TD" in mermaid_code
    assert "Cu2S phase transition" in mermaid_code

    explanation = engine.explain_discovery(path)
    assert "DISCOVERY PATH EXPLANATION" in explanation
    assert "Final Confidence:" in explanation
