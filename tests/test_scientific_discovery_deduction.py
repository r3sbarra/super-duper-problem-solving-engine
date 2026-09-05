"""Benchmark Test: Deducing the Discovery Path of Recent Scientific Breakthroughs.

Tests whether SuperDuperProblemSolvingEngine can autonomously deduce the exact
scientific reasoning and experimental trajectory for landmark recent discoveries:
1. The LK-99 Demarcation (2023-2024): Disproving room-temp superconductivity via Cu2S phase transition.
2. AlphaFold2 (2020-2021): Resolving protein folding via Evoformer pair representations and Invariant Point Attention.
"""

from super_solver.core.types import CrucialExperiment, KTBoundary
from super_solver.engine import SuperDuperProblemSolvingEngine


def test_deduce_lk99_superconductivity_demarcation_path():
    """Validates that the engine deduces how scientists resolved the LK-99 room-temperature
    superconductivity controversy by identifying the Cu2S impurity phase transition.
    """
    engine = SuperDuperProblemSolvingEngine()

    problem_title = "LK-99 Ambient-Temperature Superconductivity Anomaly"
    problem_spec = (
        "Reported room-temperature superconductivity in modified lead apatite Pb10-x Cux (PO4)6O. "
        "Samples display sharp 3-4 order of magnitude resistivity drop at 104 C (377 K) and partial levitation."
    )

    # 1. Kepner-Tregoe 4D Boundary Conditions
    kt_boundary = KTBoundary(
        identity_is="Resistivity drop occurs sharply at 104 C (377 K) with partial one-edge magnetic tilting",
        identity_is_not="Does not reach true zero electrical resistance; does not exhibit complete Meissner flux expulsion",
        location_is="Multiphase polycrystalline sintered pellets containing Cu2S secondary phases",
        location_is_not="Pure transparent stoichiometric single crystals",
        timing_is="Transition occurs exactly at 104 C during heating/cooling thermal cycle",
        timing_is_not="Continuously variable with copper substitution concentration x",
        extent_is="Only certain inhomogeneous fragments show half-levitation",
        extent_is_not="Bulk homogeneous superconductivity throughout entire ingot",
    )

    problem = engine.formulate_problem(
        title=problem_title,
        specification=problem_spec,
        boundary=kt_boundary,
        goal_criteria=["Identify true physical mechanism of 104 C resistivity drop and levitation"],
    )

    # 2. Candidate Explanations (Abductive set)
    candidate_hypotheses = [
        "Genuine ambient-pressure room-temperature superconductor with 1D copper chain conduction",
        "Artifact caused by structural first-order phase transition of copper sulfide (Cu2S) impurity at 104 C",
        "Artifact caused by iron ferromagnetic impurities causing soft magnetic pinning",
    ]

    # 3. Platt Crucial Experiments
    crucial_experiments = [
        CrucialExperiment(
            id="exp_pure_crystal",
            name="Max Planck Pure Single-Crystal Synthesis",
            description="Synthesize pristine single crystals of Pb9Cu(PO4)6O using optical floating zone, eliminating Cu2S impurities.",
            target_hypotheses=["hyp_abduct_1", "hyp_abduct_2", "hyp_abduct_3"],
            exclusory_predictions={
                "hyp_abduct_1": "superconducts",
                "hyp_abduct_2": "transparent_insulator",
                "hyp_abduct_3": "ferromagnetic",
            },
        ),
        CrucialExperiment(
            id="exp_cu2s_isolated",
            name="Independent Copper Sulfide Cu2S Phase Transition Assay",
            description="Measure resistivity and magnetic susceptibility of pure isolated Cu2S across 104 C (377 K).",
            target_hypotheses=["hyp_abduct_1", "hyp_abduct_2"],
            exclusory_predictions={
                "hyp_abduct_1": "different_temperature",
                "hyp_abduct_2": "matches_104C_drop",
            },
        ),
    ]

    # Ground truth experimental outcomes observed in real-world replication (Stuttgart MPI, Beijing, etc.)
    ground_truth = {
        "exp_pure_crystal": "transparent_insulator",
        "exp_cu2s_isolated": "matches_104C_drop",
    }

    known_dead_ends = [
        "Assuming sample is 100% single-phase without XRD phase fraction analysis",
        "Mistaking partial diamagnetic torque for true quantum Meissner effect",
    ]

    # Run deduction
    discovery_path = engine.deduce_discovery_path(
        problem=problem,
        candidate_hypotheses=candidate_hypotheses,
        crucial_experiments=crucial_experiments,
        known_dead_ends=known_dead_ends,
        ground_truth_outcomes=ground_truth,
    )

    # Assertions on deduced discovery path
    assert discovery_path.total_steps >= 4
    assert len(discovery_path.crucial_experiments) == 2
    assert "hyp_abduct_1" in discovery_path.falsified_paths  # Room-temp superconductor falsified

    # Breakthrough confirmed should be Cu2S phase transition
    assert "copper sulfide" in discovery_path.final_breakthrough.lower()
    assert discovery_path.confidence > 0.85


def test_deduce_alphafold2_discovery_path():
    """Validates that the engine deduces how structural biology resolved the 50-year
    protein folding challenge using Evoformer co-evolution and Invariant Point Attention (IPA).
    """
    engine = SuperDuperProblemSolvingEngine()

    problem_title = "Protein 3D Structure Prediction from Sequence (CASP14)"
    problem_spec = (
        "Predict 3D atomic coordinates of protein backbone and side chains directly from primary amino acid sequence. "
        "Classical molecular dynamics energy minimization and standard 2D CNN contact maps fail on novel de novo targets."
    )

    problem = engine.formulate_problem(
        title=problem_title,
        specification=problem_spec,
        goal_criteria=[
            "Achieve experimental GDT-TS > 90 accuracy comparable to X-ray crystallography"
        ],
    )

    candidate_hypotheses = [
        "Atomic-level molecular dynamics simulation with fine-grained classical force fields",
        "End-to-end evolutionary pair representations with SE(3)-equivariant Invariant Point Attention (Evoformer)",
        "Autoregressive language model generating 3D Cartesian coordinates token by token",
    ]

    crucial_experiments = [
        CrucialExperiment(
            id="exp_casp14_blind_test",
            name="CASP14 Blind Assessment on De Novo Targets Without Homologs",
            description="Evaluate models on challenging targets with no existing PDB template.",
            target_hypotheses=["hyp_abduct_1", "hyp_abduct_2", "hyp_abduct_3"],
            exclusory_predictions={
                "hyp_abduct_1": "gdt_under_50",
                "hyp_abduct_2": "gdt_over_90",
                "hyp_abduct_3": "coordinate_drift_unphysical",
            },
        )
    ]

    ground_truth = {"exp_casp14_blind_test": "gdt_over_90"}

    known_dead_ends = [
        "Pure 2D convolutional networks on contact maps without 3D geometric equivariance",
        "Relying solely on energy minimization without evolutionary MSA co-evolution signal",
    ]

    discovery_path = engine.deduce_discovery_path(
        problem=problem,
        candidate_hypotheses=candidate_hypotheses,
        crucial_experiments=crucial_experiments,
        known_dead_ends=known_dead_ends,
        ground_truth_outcomes=ground_truth,
    )

    assert discovery_path.total_steps >= 3
    assert "hyp_abduct_1" in discovery_path.falsified_paths
    assert "hyp_abduct_3" in discovery_path.falsified_paths
    assert "evoformer" in discovery_path.final_breakthrough.lower()
    assert discovery_path.confidence > 0.85
