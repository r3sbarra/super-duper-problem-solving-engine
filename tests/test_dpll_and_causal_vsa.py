"""Unit tests for DPLL SAT Solver, Causal VSA Graph Surgery, and BOED Experimental Design."""

from super_solver.core.causal_vsa import VectorizedCausalModel
from super_solver.core.dpll_solver import DPLLSolver
from super_solver.frameworks.boed_designer import BOEDDesigner, CandidateExperimentDesign


def test_dpll_sat_and_refutation():
    """Validates DPLL SAT solving, unit propagation, and proof by resolution refutation."""
    solver = DPLLSolver()

    # 1. Simple Satisfiable CNF: (A or B) and (~A or B) and (~B or C)
    sat, model = solver.solve_cnf([[1, 2], [-1, 2], [-2, 3]])
    assert sat is True
    assert model[2] is True  # B must be True
    assert model[3] is True  # C must be True

    # 2. Unsatisfiable CNF: A and ~A
    unsat, _ = solver.solve_cnf([[1], [-1]])
    assert unsat is False

    # 3. Proof by Refutation: Rain => Wet, Wet => Slippery, Rain |= Slippery
    premises = [
        "rain => wet",
        "wet => slippery",
        "rain",
    ]
    refute_res = solver.refute_conjecture(premises=premises, target_claim="slippery")
    assert refute_res["proved"] is True
    assert refute_res["verdict"] == "LOGICALLY_SOUND_PROOF"

    # 4. Countermodel detection for invalid claim
    non_entailed = solver.refute_conjecture(premises=["rain => wet"], target_claim="slippery")
    assert non_entailed["proved"] is False
    assert non_entailed["verdict"] == "COUNTERMODEL_EXISTS"
    assert non_entailed["countermodel"] is not None


def test_causal_vsa_graph_surgery_and_confounding_resolution():
    """Validates Pearl's Level 2 do-calculus intervention in VSA hypervector space."""
    causal = VectorizedCausalModel(vsa_dim=2048)

    # Model the LK-99 dilemma:
    # Temperature -> Cu2S Phase Transition -> Resistivity Drop
    # True Superconductivity -> Meissner Expulsion
    # True Superconductivity -> Resistivity Drop
    causal.add_variable("cu2s_phase_transition", baseline_value=0.5)
    causal.add_variable("true_superconductivity", baseline_value=0.1)
    causal.add_variable("resistivity_drop", baseline_value=0.5)
    causal.add_variable("meissner_flux_expulsion", baseline_value=0.1)

    causal.add_causal_edge("cu2s_phase_transition", "resistivity_drop", strength=0.9)
    causal.add_causal_edge("true_superconductivity", "resistivity_drop", strength=0.9)
    causal.add_causal_edge("true_superconductivity", "meissner_flux_expulsion", strength=0.95)

    # Topology hypervector check
    g_vec = causal.get_graph_hypervector()
    assert g_vec.shape == (2048,)

    # 1. Intervene do(cu2s_phase_transition = 0.9): should strongly affect resistivity_drop
    int_cu2s = causal.intervene(
        "cu2s_phase_transition", clamped_value=0.9, consequence_target="resistivity_drop"
    )
    assert int_cu2s["is_causally_effective"] is True
    assert int_cu2s["average_causal_effect"] > 0.20
    assert "cu2s_phase_transition" in int_cu2s["intervention"]

    # 2. Intervene do(cu2s_phase_transition = 0.9) on meissner effect: should have zero effect!
    int_meissner = causal.intervene(
        "cu2s_phase_transition", clamped_value=0.9, consequence_target="meissner_flux_expulsion"
    )
    assert abs(int_meissner["average_causal_effect"]) < 0.05
    assert int_meissner["is_causally_effective"] is False


def test_boed_expected_information_gain_ranking():
    """Validates Bayesian Optimal Experimental Design selecting the most informative trial."""
    designer = BOEDDesigner()

    hypothesis_priors = {
        "H1_Superconductivity": 0.5,
        "H2_ArtifactualPhaseTransition": 0.5,
    }

    # Candidate Experiment 1: SQUID Magnetometer (Crucial Test)
    exp_squid = CandidateExperimentDesign(
        design_id="exp_squid",
        name="SQUID Magnetometry Flux Expulsion",
        description="Measures direct magnetic susceptibility chi_v at 400K",
        parameters={"field_gauss": 100, "temp_kelvin": 380},
        cost=2.0,
    )

    # Candidate Experiment 2: DC 4-Probe Resistance (Ambiguous / Confounded)
    exp_dc = CandidateExperimentDesign(
        design_id="exp_dc",
        name="Standard 4-Probe DC Resistivity",
        description="Measures bulk voltage drop across electrical contacts",
        parameters={"current_ma": 10},
        cost=1.0,
    )

    likelihood_matrix = {
        "exp_squid": {
            "flux_expulsion_detected": {
                "H1_Superconductivity": 0.95,
                "H2_ArtifactualPhaseTransition": 0.02,
            },
            "no_flux_expulsion": {
                "H1_Superconductivity": 0.05,
                "H2_ArtifactualPhaseTransition": 0.98,
            },
        },
        "exp_dc": {
            "voltage_drop": {
                "H1_Superconductivity": 0.85,
                "H2_ArtifactualPhaseTransition": 0.85,  # Completely uninformative!
            },
            "no_voltage_drop": {
                "H1_Superconductivity": 0.15,
                "H2_ArtifactualPhaseTransition": 0.15,
            },
        },
    }

    ranked = designer.rank_optimal_experiments(
        hypothesis_priors=hypothesis_priors,
        candidate_designs=[exp_squid, exp_dc],
        likelihood_matrix=likelihood_matrix,
    )

    assert len(ranked) == 2
    # SQUID magnetometry must yield significantly higher EIG than uninformative DC test
    assert ranked[0]["design_id"] == "exp_squid"
    assert ranked[0]["expected_information_gain"] > 0.40
    assert ranked[1]["design_id"] == "exp_dc"
    assert ranked[1]["expected_information_gain"] < 0.05
