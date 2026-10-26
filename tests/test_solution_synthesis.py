"""Test Solution Mechanism Synthesizer (SMS) and engine integration."""
from __future__ import annotations

from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.frameworks.solution_synthesis import SolutionMechanismSynthesizer


def test_synthesizer_categories():
    sms = SolutionMechanismSynthesizer()
    assert sms.infer_category("Reduce cache invalidation spikes in p99 latency") == "latency"
    assert sms.infer_category("Detect a data race in multi-threaded execution") == "concurrency"
    assert sms.infer_category("Shortest-path routing for delivery fleet under bounds") == "optimization"
    assert sms.infer_category("Detect duplicate invoices across two billing systems") == "consistency"
    assert sms.infer_category("Prevent one slow backend from crashing the whole cluster") == "resilience"


def test_synthesizer_category_matching_is_word_boundary_aware():
    """Regression: category cues used to be raw substring tests, so ``"bound"``
    fired on the word *"boundary"* and a "Characterize the boundary/special
    cases of ..." subproblem was classified as an *optimisation* problem and
    handed a branch-and-bound template -- producing breakthroughs like
    "Characterize the boundary ... to prune non-optimal subtrees in
    branch-and-bound" for a concrete-canoe problem."""
    sms = SolutionMechanismSynthesizer()
    assert (
        sms.infer_category(
            "Characterize the boundary/special cases of 'dense concrete canoe hull' "
            "(degenerate inputs, limits)"
        )
        == "generic"
    )
    # "race" must not fire on "trace", "lock" must not fire on "clock".
    assert sms.infer_category("Trace the clock skew across nodes") == "generic"


def test_synthesizer_falls_back_to_context_only_when_subproblem_is_silent():
    sms = SolutionMechanismSynthesizer()
    # Subproblem carries the cue -> wins.
    assert (
        sms.category_from("Reduce p99 latency spikes", "Shortest-path routing problem")
        == "latency"
    )
    # Subproblem is uninformative -> inherits the problem's category.
    assert (
        sms.category_from("Handle arrival penalties", "Shortest-path routing problem")
        == "optimization"
    )


def test_unrecognised_problem_keeps_triz_grounded_breakthrough():
    """When the synthesizer does not recognise the problem's category it must
    NOT inject domain-specific mechanism templates as if they were a concrete
    solution; the coherent TRIZ-grounded abduction is kept instead."""
    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")
    path = engine.solve_engineering_problem(
        title="Concrete Canoe",
        specification="How to make a canoe hull that floats despite being made of dense concrete",
        top_principles=4,
        ground_truth_outcomes=None,
    )
    assert "branch-and-bound" not in path.final_breakthrough
    assert "TRIZ principle" in path.final_breakthrough


def test_synthesize_mechanisms():
    sms = SolutionMechanismSynthesizer()
    mechs = sms.synthesize_mechanisms_for_subproblem(
        subproblem_text="Reduce cache invalidation spikes in p99 latency",
        problem_context="High-throughput distributed API gateway",
        max_mechanisms=2,
    )
    assert len(mechs) == 2
    assert any("cache" in m["description"].lower() for m in mechs)
    assert any(m["action_primitive"] == "act_cache" for m in mechs)
    assert "triz_principle" in mechs[0]


def test_engine_synthesize_integration():
    engine = SuperDuperProblemSolvingEngine()
    spec = "Shortest-path routing for a delivery fleet under time-window constraints"
    subproblems = [
        {"subproblem": "Formulate routing objective with vehicle capacities"},
        {"subproblem": "Handle non-linear time-window arrival penalties"},
    ]
    mechs = engine.synthesize_solution_mechanisms(spec, subproblems=subproblems)
    assert len(mechs) >= 2
    for m in mechs:
        assert "mechanism" in m
        assert "source_subproblem" in m
