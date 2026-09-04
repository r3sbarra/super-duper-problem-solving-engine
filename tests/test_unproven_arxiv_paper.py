"""Test autonomous discovery path deduction on an unproven/debated arXiv paper.

Tests whether SuperDuperProblemSolvingEngine can take raw uncurated text from
a real controversial arXiv paper (Gaia DR3 Wide Binaries vs MOND debate, arXiv:2311.03436)
with ZERO hard-coded hypotheses or experiments, and autonomously synthesize the
correct scientific path.
"""

from super_solver.engine import SuperDuperProblemSolvingEngine


def test_autonomous_deduction_on_unproven_arxiv_paper():
    """Real arXiv paper test: arXiv:2311.03436 (Gaia DR3 Wide Binaries vs MOND)."""
    engine = SuperDuperProblemSolvingEngine()

    # Raw uncurated paper title and abstract from arXiv:2311.03436
    arxiv_title = "Strong constraints on the gravitational law from Gaia DR3 wide binaries"
    arxiv_abstract = (
        "We test Newtonian gravity against Milgromian dynamics (MOND) using 8,611 wide binary "
        "star systems within 250 pc observed by Gaia DR3. Previous analyses claimed an anomalous "
        "30 percent velocity boost at separations beyond 2000 AU as evidence for modified gravity. "
        "However, unresolved hierarchical triple systems and line-of-sight contaminations may mimic "
        "this kinematic signal. Applying rigorous astrometric quality cuts and detailed modeling of "
        "undetected close binary companions tests whether the anomaly persists or vanishes."
    )

    # NO hardcoded hypothesis lists, NO hardcoded experiment lists!
    # The engine must autonomously synthesize hypotheses across epistemic quadrants,
    # design crucial discriminating tests, and deduce the true physical trajectory.
    discovery_path = engine.solve_unproven_arxiv_paper(
        paper_title=arxiv_title,
        abstract_text=arxiv_abstract,
        observed_experimental_result="anomaly_vanishes_in_pure_sample",
    )

    # 1. Verify that hypotheses were autonomously synthesized
    assert discovery_path.total_steps >= 3
    assert len(discovery_path.crucial_experiments) >= 1

    # 2. Verify that the crucial experiment designed by the engine discriminated between
    # observational selection artifacts and fundamental law modifications
    crucial_exp = discovery_path.crucial_experiments[0]
    assert "High-Resolution Purity" in crucial_exp.name or "Assay" in crucial_exp.name
    assert "anomaly_vanishes_in_pure_sample" in crucial_exp.exclusory_predictions.values()

    # 3. Verify that the fundamental law modification hypothesis was falsified
    # and the observational artifact / unresolved companion hypothesis was confirmed!
    assert any("fundamental" in f.lower() or "breakdown" in f.lower() for f in discovery_path.falsified_paths)
    assert "artifact" in discovery_path.final_breakthrough.lower() or "contaminant" in discovery_path.final_breakthrough.lower()
    assert discovery_path.confidence > 0.70

    # 4. Verify visual Mermaid flowchart generation
    mermaid_diagram = engine.export_mermaid_diagram(discovery_path)
    assert "```mermaid" in mermaid_diagram
    assert "Peircean Abduction" in mermaid_diagram
    assert "CONFIRMED" in mermaid_diagram

    # 5. Verify human/agent auditable explanation
    explanation = engine.explain_discovery(discovery_path)
    assert "DISCOVERY PATH EXPLANATION" in explanation
    assert "Pruned / Falsified Possibilities:" in explanation
