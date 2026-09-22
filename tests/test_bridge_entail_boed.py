"""Unit tests for the lab-ass bridge: DPLL entail + BOED next_experiment."""

from super_solver.lab_ass_bridge.hooks import LabAssSuperSolverBridge


def test_bridge_dpll_entail_proves():
    bridge = LabAssSuperSolverBridge()
    result = bridge.dpll_entail(
        premises=["rain => wet", "wet => slippery", "rain"],
        target_claim="slippery",
    )
    assert result["proved"] is True
    assert result["verdict"] == "LOGICALLY_SOUND_PROOF"
    assert result["clauses_evaluated"] > 0


def test_bridge_dpll_entail_countermodel():
    bridge = LabAssSuperSolverBridge()
    result = bridge.dpll_entail(
        premises=["rain => wet"],
        target_claim="slippery",
    )
    assert result["proved"] is False
    assert result["verdict"] == "COUNTERMODEL_EXISTS"
    assert result["countermodel"] is not None


def test_bridge_decompose_merges_generated_subproblems():
    """decompose_topic_with_polya_and_triz must merge genuinely NEW generated
    subproblems (is_new=True) with the caller-supplied re-ranking, instead of only
    re-ranking the caller's own list. Regression for the finding that the Polya
    generator was never wired into the decompose path."""
    bridge = LabAssSuperSolverBridge()
    known = ["Background & Context", "Current State & Recent Developments"]
    result = bridge.decompose_topic_with_polya_and_triz(
        "Find a new proof that there are infinitely many primes",
        known,
    )
    aux = result["auxiliary_angles"]
    # The caller's subproblems must be present (re-ranked), AND new ones appended.
    titles = {a.get("subproblem", "") for a in aux}
    for k in known:
        assert k in titles
    assert any(a.get("is_new", False) for a in aux)
    # All entries carry the keys the lab-ass plugin consumes.
    for a in aux:
        assert "subproblem" in a and "is_essential" in a and "relevance_score" in a
    # The output is richer than the caller's two inputs.
    assert len(aux) > len(known)


def test_bridge_next_experiment_ranks():
    bridge = LabAssSuperSolverBridge()
    priors = {"H1": 0.5, "H2": 0.5}
    designs = [
        {
            "design_id": "exp_squid",
            "name": "SQUID",
            "description": "direct",
            "parameters": {},
            "cost": 2.0,
        },
        {
            "design_id": "exp_dc",
            "name": "DC probe",
            "description": "confounded",
            "parameters": {},
            "cost": 1.0,
        },
    ]
    likelihood = {
        "exp_squid": {
            "flux": {"H1": 0.95, "H2": 0.02},
            "no_flux": {"H1": 0.05, "H2": 0.98},
        },
        "exp_dc": {
            "drop": {"H1": 0.85, "H2": 0.85},
            "no_drop": {"H1": 0.15, "H2": 0.15},
        },
    }
    ranked = bridge.next_experiment(priors, designs, likelihood)
    assert len(ranked) == 2
    # informative SQUID beats uninformative DC probe
    assert ranked[0]["design_id"] == "exp_squid"
    assert ranked[0]["expected_information_gain"] > 0.40
    assert ranked[1]["design_id"] == "exp_dc"
    assert ranked[1]["expected_information_gain"] < 0.05
