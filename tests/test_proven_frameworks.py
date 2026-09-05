"""Unit tests for the 7 proven problem-solving frameworks in super_solver."""

from super_solver.core.types import CrucialExperiment, Hypothesis, HypothesisStatus, KTBoundary
from super_solver.frameworks.kepner_tregoe import KepnerTregoeEngine
from super_solver.frameworks.peirce_inquiry import PeirceanInquiryEngine
from super_solver.frameworks.platt_inference import StrongInferenceEngine
from super_solver.frameworks.polya_heuristics import PolyaHeuristicsEngine
from super_solver.frameworks.toc_cloud import TOCEngine
from super_solver.frameworks.triz_engine import TRIZEngine


def test_platt_strong_inference_entropy_and_pruning():
    engine = StrongInferenceEngine()

    h1 = Hypothesis(
        id="h1",
        title="Hypothesis A",
        description="Enzyme is substrate-inhibited",
        current_confidence=0.5,
    )
    h2 = Hypothesis(
        id="h2",
        title="Hypothesis B",
        description="Allosteric feedback loop",
        current_confidence=0.5,
    )

    initial_entropy = engine.compute_entropy([h1, h2])
    assert 0.95 <= initial_entropy <= 1.05

    exp = CrucialExperiment(
        id="exp1",
        name="Allosteric site mutation assay",
        description="Mutate site X. If H1, activity unchanged; if H2, feedback disabled.",
        target_hypotheses=["h1", "h2"],
        exclusory_predictions={"h1": "unchanged", "h2": "feedback_disabled"},
    )

    info_gain = engine.evaluate_crucial_experiment(exp, [h1, h2])
    assert info_gain > 0.8

    updated_h, falsified = engine.execute_and_prune(
        exp, [h1, h2], observed_outcome="feedback_disabled"
    )
    assert "h1" in falsified
    assert updated_h[0].status == HypothesisStatus.FALSIFIED
    assert updated_h[1].status == HypothesisStatus.CONFIRMED
    assert updated_h[1].current_confidence == 1.0


def test_kepner_tregoe_boundary_filtering():
    engine = KepnerTregoeEngine()

    boundary = KTBoundary(
        identity_is="High-temperature sensor failure on Reactor Core 3",
        identity_is_not="Standard temperature sensors or other reactors",
        location_is="North quadrant piping",
        location_is_not="South quadrant or control room",
        timing_is="Occurs solely during peak pressurization cycles",
        timing_is_not="During idle or steady-state operation",
        extent_is="3 sensors burned out simultaneously",
        extent_is_not="Gradual drift in all sensors",
    )

    good_cause = "Transient localized pressure-induced thermal arc in North piping during peak pressurization"
    eval_good = engine.evaluate_candidate_cause(good_cause, boundary)
    assert eval_good["valid_boundary_fit"] is True
    assert eval_good["net_score"] > 0.30

    bad_cause = "Global firmware clock bug causing all sensors across all reactors to fail continuously at all times"
    eval_bad = engine.evaluate_candidate_cause(bad_cause, boundary)
    assert eval_bad["sim_to_is_not"] > 0.35
    assert eval_bad["net_score"] < eval_good["net_score"]


def test_triz_contradiction_matching():
    engine = TRIZEngine()

    contradiction = (
        "Increasing mechanical strength of the beam causes excessive mass and weight burden"
    )
    principles = engine.suggest_principles(contradiction, top_k=5)

    assert len(principles) == 5
    principle_names = [p["name"] for p in principles]
    assert any(
        p
        in ["Anti-Weight", "Mechanical Vibration", "Mechanics Substitution", "Composite Materials"]
        for p in principle_names
    )


def test_toc_evaporating_cloud_resolution():
    engine = TOCEngine()

    cloud = engine.formulate_cloud(
        objective="Produce reliable scientific software",
        requirement_a="Innovate rapidly with cutting-edge features",
        requirement_b="Maintain zero bugs and rock-solid stability",
        prerequisite_d="Deploy code immediately without manual review",
        prerequisite_d_prime="Lock down codebase and mandate weeks of manual regression testing",
    )

    injections = [
        "Ignore testing completely and push broken code",
        "Freeze all feature development forever",
        "Implement automated continuous integration and testing for rapid stable delivery",
    ]

    res = engine.evaporate(cloud, injections)
    assert res["resolved"] is True
    assert "automated continuous integration" in res["winning_injection"]


def test_peirce_abductive_deductive_inductive_cycle():
    engine = PeirceanInquiryEngine()

    anomaly = (
        "Bacterial culture survives normally lethal antibiotic dose in presence of metabolite X"
    )
    context = "Microbiology cellular metabolism and antibiotic resistance mechanisms"
    candidates = [
        "Metabolite X acts as an enzymatic decoy or competitive inhibitor for the antibiotic",
        "The bacteria instantaneously underwent thousands of simultaneous random mutations in seconds",
        "The antibiotic spontaneously vanished from the solution without interacting",
    ]

    # Abduction
    hypotheses = engine.abduct(anomaly, context, candidates)
    assert len(hypotheses) == 3
    assert "enzymatic decoy" in hypotheses[0].description

    # Deduction
    predictions = engine.deduce_predictions(
        hypotheses[0],
        candidate_predictions=[
            "Binding affinity assay will show direct molecular complex between metabolite X and antibiotic",
            "Temperature of the room will drop by 20 degrees",
        ],
    )
    assert len(predictions) >= 1
    assert "Binding affinity" in predictions[0]

    # Induction
    conf = engine.induct_calibrate(
        hypotheses[0],
        observed_evidence=[
            "Surface plasmon resonance verifies high-affinity binding between metabolite X and antibiotic"
        ],
    )
    assert conf > 0.45
    assert hypotheses[0].status in [HypothesisStatus.CONFIRMED, HypothesisStatus.TESTING]


def test_polya_heuristics():
    engine = PolyaHeuristicsEngine()

    spec = "Calculate the exact structural stability of a complex 3D folded macromolecule"
    subproblems = [
        "Predict secondary structure helices and stability of folded domains",
        "Analyze stock market price trends of pharmaceutical retail stores",
    ]

    aux = engine.decompose_auxiliary_problems(spec, subproblems)
    assert (
        aux[0]["subproblem"]
        == "Predict secondary structure helices and stability of folded domains"
    )
    assert (
        aux[1]["subproblem"] == "Analyze stock market price trends of pharmaceutical retail stores"
    )
    assert aux[0]["relevance_score"] > aux[1]["relevance_score"]


def test_polya_generate_auxiliary_problems():
    """generate_auxiliary_problems synthesizes NEW subproblems, not echoes."""
    engine = PolyaHeuristicsEngine()

    spec = (
        "Find new falsifiable, numerically-testable approaches to the Riemann "
        "Hypothesis. RH states all non-trivial zeros of the Riemann zeta function "
        "lie on the critical line Re(s)=1/2."
    )
    known = ["Lagarias", "Robin", "Li criterion", "GUE spacing", "Mertens"]

    res = engine.generate_auxiliary_problems(spec, known_subproblems=known, max_new=8)

    # It must return something, all entries carry the expected keys, and at
    # least one genuinely-new subproblem is produced (not just re-ranked knowns).
    assert len(res) > 0
    assert len(res) <= 8
    for r in res:
        assert "subproblem" in r
        assert "heuristic" in r
        assert "relevance_score" in r
        assert "is_new" in r
        assert isinstance(r["relevance_score"], float)
    assert any(r["is_new"] for r in res)
    # Generated subproblems should be distinct from the caller-supplied knowns.
    known_lower = {k.lower() for k in known}
    assert not any(r["subproblem"].lower() in known_lower for r in res)
