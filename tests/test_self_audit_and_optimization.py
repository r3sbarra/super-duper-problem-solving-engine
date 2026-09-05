"""Tests validating the engine running its new capabilities against itself."""

from super_solver.engine import SuperDuperProblemSolvingEngine


def test_engine_runs_against_itself():
    """Validates the engine evaluating and optimizing its own reasoning substrate using DPLL, Causal VSA, and BOED."""
    engine = SuperDuperProblemSolvingEngine()

    # 1. Run the comprehensive self-audit and optimization
    audit = engine.run_self_audit_and_optimization()

    assert audit["self_evaluation_status"] == "AUDIT_PASSED_HEALTHY"

    # 2. Verify DPLL verified the engine's core safety invariants
    dpll_res = audit["dpll_invariant_verification"]
    assert dpll_res["proved"] is True
    assert dpll_res["verdict"] == "LOGICALLY_SOUND_PROOF"

    # 3. Verify Causal SCM intervention on internal parameters
    causal_res = audit["causal_intervention_audit"]
    assert causal_res["is_effective"] is True
    assert causal_res["average_causal_effect"] > 0.20
    assert "negative_manifold_repulsion" in causal_res["intervention"]

    # 4. Verify BOED optimal experimental design for parameter tuning
    boed_res = audit["boed_tuning_utility"]
    assert boed_res["expected_information_gain"] > 0.20
    assert boed_res["cost_penalized_utility"] > 0.20

    # 5. Verify the engine successfully executed its self-improvement loop
    discovery_path = engine.improve_self()
    assert discovery_path.total_steps >= 2
    assert "CONFIRMED" in discovery_path.final_breakthrough


def test_synaptic_memory_consolidation():
    """Validates that redundant negative manifold repulsors are clustered and consolidated into centroids."""
    engine = SuperDuperProblemSolvingEngine()

    # Register multiple near-duplicate dead ends
    similar_hazards = [
        "Rigid static heuristic parameter tuning without adaptation",
        "Rigid static heuristic parameter tuning with no adaptation",
        "Rigid static heuristic parameter tuning lacking adaptation",
        "Completely unrelated chemical synthesis protocol failure",
    ]
    for hazard in similar_hazards:
        engine.repulsor.register_dead_end(hazard)

    initial_count = len(engine.repulsor.dead_ends)
    assert initial_count == 4

    # Run consolidation with threshold 0.75
    pruned = engine.consolidate_memory(similarity_threshold=0.75)
    assert pruned == 2  # 3 similar hazards merged into 1 centroid
    assert len(engine.repulsor.dead_ends) == 2  # 1 centroid + 1 chemical synthesis

    # Verify proximity hazard detection still works against the consolidated centroid
    from super_solver.core.embeddings import embedding_service

    query_vec = embedding_service.encode(
        "Rigid static heuristic parameter tuning without adaptation"
    )
    is_near, max_sim, desc = engine.repulsor.check_proximity(query_vec, threshold=0.60)
    assert is_near is True
    assert desc is not None
    assert "Consolidated Hazard Cluster" in desc
