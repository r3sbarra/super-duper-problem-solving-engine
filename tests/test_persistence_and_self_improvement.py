"""Tests validating persistent memory (episodes, dead ends, parameters) and functional self-improvement."""

from super_solver.core.embeddings import embedding_service
from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.memory.episodic_store import get_default_db_path


def test_default_db_path_resolution(tmp_path, monkeypatch):
    """Verifies default DB path resolution respects env vars, tests, and persistence."""
    # 1. When SUPER_SOLVER_DB_PATH is set
    custom_path = str(tmp_path / "custom.db")
    monkeypatch.setenv("SUPER_SOLVER_DB_PATH", custom_path)
    assert get_default_db_path() == custom_path

    # 2. When in pytest without custom env var
    monkeypatch.delenv("SUPER_SOLVER_DB_PATH", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_foo")
    assert get_default_db_path() == ":memory:"

    # 3. When running normally (not in pytest, no env var)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    default_path = get_default_db_path()
    assert ".super_solver_data" in default_path
    assert default_path.endswith("super_solver.db")


def test_memory_persists_across_engine_instances(tmp_path):
    """Verifies that dead ends and episodic breakthroughs survive process restarts."""
    db_file = str(tmp_path / "persistent_test.db")

    # Instance 1: Register dead ends and solve a problem
    engine1 = SuperDuperProblemSolvingEngine(db_path=db_file)
    engine1.repulsor.register_dead_end("Toxic local minima in chemical reaction step A")
    engine1.repulsor.register_dead_end("Unstable catalyst precipitation at low pH")

    assert len(engine1.repulsor.dead_ends) == 2

    # Solve a problem to harvest episodic memory
    prob = engine1.formulate_problem(
        title="Catalyst Stability Assay",
        specification="Seeking optimal pH window for organometallic catalyst",
    )
    path = engine1.deduce_discovery_path(
        problem=prob,
        candidate_hypotheses=["Maintain neutral pH buffer", "Acidic flash rinse"],
        known_dead_ends=["Acidic flash rinse"],
    )
    assert path.confidence > 0
    assert engine1.store.count() >= 1

    # Close engine 1
    engine1.store.close()
    del engine1

    # Instance 2: Connect to the same database
    engine2 = SuperDuperProblemSolvingEngine(db_path=db_file)

    # 1. Dead ends must be automatically loaded into the repulsor
    assert len(engine2.repulsor.dead_ends) >= 2
    assert any("Toxic local minima" in desc for desc in engine2.repulsor.dead_end_descriptions)
    assert any("Unstable catalyst" in desc for desc in engine2.repulsor.dead_end_descriptions)

    # 2. Proximity detection must recognize pre-existing dead ends
    q_vec = embedding_service.encode("Toxic local minima in chemical reaction step A")
    is_near, sim, desc = engine2.repulsor.check_proximity(q_vec, threshold=0.80)
    assert is_near is True
    assert desc is not None
    assert "Toxic local minima" in desc

    # 3. Episodic memory must be retained
    assert engine2.store.count() >= 1
    similar_cases = engine2.store.search_similar(
        embedding_service.encode("Catalyst Stability Assay")
    )
    assert len(similar_cases) >= 1
    assert "Catalyst Stability Assay" in similar_cases[0]["title"]

    engine2.store.close()


def test_synaptic_consolidation_persists(tmp_path):
    """Verifies that consolidated centroid repulsors persist to disk."""
    db_file = str(tmp_path / "consolidation_test.db")

    engine1 = SuperDuperProblemSolvingEngine(db_path=db_file)
    engine1.repulsor.register_dead_end("Extreme thermal degradation at 900C")
    engine1.repulsor.register_dead_end("Extreme thermal degradation at 910C")
    engine1.repulsor.register_dead_end("Extreme thermal degradation at 920C")
    engine1.repulsor.register_dead_end("Completely separate vacuum leak")

    assert len(engine1.repulsor.dead_ends) == 4
    pruned = engine1.consolidate_memory(similarity_threshold=0.75)
    assert pruned == 2
    assert len(engine1.repulsor.dead_ends) == 2
    engine1.store.close()
    del engine1

    # Instance 2: Check consolidated state is preserved in DB
    engine2 = SuperDuperProblemSolvingEngine(db_path=db_file)
    assert len(engine2.repulsor.dead_ends) == 2
    assert any(
        "Consolidated Hazard Cluster" in desc for desc in engine2.repulsor.dead_end_descriptions
    )
    assert any("vacuum leak" in desc for desc in engine2.repulsor.dead_end_descriptions)
    engine2.store.close()


def test_self_improvement_parameter_persistence_and_wiring(tmp_path):
    """Verifies that self-improvement parameter adjustments persist and are wired into engine execution."""
    db_file = str(tmp_path / "self_improvement_test.db")

    # Instance 1: Run auto-tune and self-audit
    engine1 = SuperDuperProblemSolvingEngine(db_path=db_file)

    # Initial defaults
    assert engine1.self_improver.params["guidance_scale"] == 1.4

    # Run auto-tuning and audit
    record = engine1.self_improver.auto_tune_parameters(target_goal_alignment=0.99)
    assert (
        "guidance_scale" in record["updates_applied"]
        or engine1.self_improver.params["guidance_scale"] > 1.4
    )
    tuned_guidance = engine1.self_improver.params["guidance_scale"]
    tuned_repulsion = engine1.self_improver.params["repulsion_scale"]
    tuned_threshold = engine1.self_improver.params["proximity_threshold"]

    # Repulsor should reflect tuned parameters
    assert engine1.repulsor.PROXIMITY_ALERT_THRESHOLD == tuned_threshold
    assert engine1.repulsor.repulsor_weight == tuned_repulsion

    engine1.store.close()
    del engine1

    # Instance 2: Load new engine on same DB
    engine2 = SuperDuperProblemSolvingEngine(db_path=db_file)

    # Loaded parameters should match tuned values
    assert engine2.self_improver.params["guidance_scale"] == tuned_guidance
    assert engine2.self_improver.params["repulsion_scale"] == tuned_repulsion
    assert engine2.self_improver.params["proximity_threshold"] == tuned_threshold
    assert engine2.repulsor.PROXIMITY_ALERT_THRESHOLD == tuned_threshold
    assert engine2.repulsor.repulsor_weight == tuned_repulsion

    # History records must be loaded
    assert len(engine2.self_improver.improvement_history) >= 1

    # Execute discovery path and verify it runs smoothly with tuned parameters
    prob = engine2.formulate_problem(
        title="Parameter Verification",
        specification="Ensure thought diffusion denoising runs with tuned scales",
    )
    discovery = engine2.deduce_discovery_path(problem=prob)
    assert discovery.total_steps >= 2
    assert any("Thought Diffusion" in s.operator_name for s in discovery.steps)

    engine2.store.close()
