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


def test_bridge_next_experiment_ranks():
    bridge = LabAssSuperSolverBridge()
    priors = {"H1": 0.5, "H2": 0.5}
    designs = [
        {"design_id": "exp_squid", "name": "SQUID", "description": "direct", "parameters": {}, "cost": 2.0},
        {"design_id": "exp_dc", "name": "DC probe", "description": "confounded", "parameters": {}, "cost": 1.0},
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
