"""Integration tests for the 6-Stage Cascading Discovery Trajectory Pipeline."""

from super_solver.engine import SuperDuperProblemSolvingEngine


def test_end_to_end_cascading_discovery_pipeline():
    """Validates the complete 6-stage multi-disciplinary discovery trajectory."""
    engine = SuperDuperProblemSolvingEngine()

    problem_title = "Solid-State Battery Dendrite Suppression and High-Rate Interfacial Transport"
    domain = "Materials Science & Electrochemistry"

    # Stage 1: Boundary Diagnostic
    is_manifestations = [
        "Catastrophic short circuit through intergranular ceramic grain boundaries during charging",
        "Localized mechanical cracking and metallic lithium dendrite propagation",
        "Abrupt voltage drop to zero observed across solid electrolyte pellet",
    ]
    is_not_manifestations = [
        "Stable reversible cycling without shorting observed under low current",
        "Uniform lithium ion flux distribution across 3D porous scaffold",
        "Conformal viscoelastic contact without intergranular penetration",
    ]
    candidate_causes = [
        "Grain boundary localized electric field concentration triggering metallic dendrite propagation",
        "Random thermal heating in ambient laboratory environment",
        "Minor contact measurement impedance artifact",
    ]

    # Stage 2: Causal Model & Intervention
    causal_variables = [
        ("grain_boundary_stress", 0.7),
        ("interfacial_viscoelasticity", 0.2),
        ("lithium_dendrite_penetration", 0.8),
        ("cycle_retention", 0.3),
    ]
    causal_edges = [
        ("grain_boundary_stress", "lithium_dendrite_penetration", 0.85),
        ("interfacial_viscoelasticity", "lithium_dendrite_penetration", -0.75),
        ("lithium_dendrite_penetration", "cycle_retention", -0.90),
    ]
    intervention_target = "interfacial_viscoelasticity"
    intervention_value = 0.95

    # Stage 3: Contradiction Tensor
    improving_objective = "Maximize energy density and ionic conductivity"
    worsening_penalty = "Interfacial mechanical fracture and rigid ceramic brittleness"

    # Stage 4: Null-Space Assumption Inversion
    requirement_a = "Maintain high shear modulus to mechanically block dendrites"
    requirement_b = "Maintain high interfacial compliance to prevent void formation"
    assumptions = [
        "The solid electrolyte must be a single monolithic, homogeneous ceramic phase",
        "High mechanical stiffness is incompatible with viscoelastic plastic deformation",
    ]
    candidate_injections = [
        "Functionally graded hybrid with rigid inorganic backbone and self-healing viscoelastic polymer interlayer",
        "Unmodified thick monolithic garnet LLZO pellet",
        "Pure liquid electrolyte fallback",
    ]

    # Stage 5: Structural Analogy from Biology
    source_domain = "Neurobiology (Voltage-Gated Ion Channel Conduits)"
    source_relations = [
        {
            "relation": "selectively_transports",
            "subject": "filter_cavity",
            "object": "potassium_cation",
        },
        {
            "relation": "accommodates_strain",
            "subject": "lipid_bilayer",
            "object": "conformation_change",
        },
    ]
    target_substitutions = {
        "filter_cavity": "sub_nanometer_zeolitic_pore",
        "potassium_cation": "lithium_cation",
        "lipid_bilayer": "polymeric_matrix_elastomer",
        "conformation_change": "electrode_volume_expansion",
    }

    # Stage 6: Logical Verification
    logical_premises = [
        "viscoelastic_interlayer -> conformal_contact",
        "conformal_contact -> suppress_dendrite",
        "suppress_dendrite -> stable_high_capacity",
        "viscoelastic_interlayer",
    ]
    target_logical_claim = "stable_high_capacity"

    result = engine.execute_cascading_discovery(
        problem_title=problem_title,
        domain=domain,
        is_manifestations=is_manifestations,
        is_not_manifestations=is_not_manifestations,
        candidate_causes=candidate_causes,
        causal_variables=causal_variables,
        causal_edges=causal_edges,
        intervention_target=intervention_target,
        intervention_value=intervention_value,
        improving_objective=improving_objective,
        worsening_penalty=worsening_penalty,
        requirement_a=requirement_a,
        requirement_b=requirement_b,
        assumptions=assumptions,
        candidate_injections=candidate_injections,
        source_domain=source_domain,
        source_relations=source_relations,
        target_entity_substitutions=target_substitutions,
        logical_premises=logical_premises,
        target_logical_claim=target_logical_claim,
    )

    # 1. Verify structure and confidence
    assert result.problem_title == problem_title
    assert result.domain == domain
    assert result.overall_confidence > 0.65

    # 2. Verify all 6 stages executed successfully
    assert "grain boundary" in result.stage1_boundary_diagnostic["primary_operator"].lower()
    assert result.stage2_causal_intervention["is_causally_effective"] is True
    assert result.stage3_contradiction_solution["score"] > 0.0
    assert "Functionally graded hybrid" in result.stage4_nullspace_evaporation["primary_operator"]
    assert len(result.stage5_analogical_transfer["details"]["transferred_statements"]) == 2
    assert result.stage6_logical_verification["proved"] is True

    # 3. Verify synthesis text
    assert "DISCOVERY SYNTHESIS" in result.synthesized_breakthrough
    assert "LOGICALLY_SOUND" in result.synthesized_breakthrough
