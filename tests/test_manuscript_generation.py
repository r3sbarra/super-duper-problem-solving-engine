"""Tests for Scientific Manuscript Generator (LaTeX & Markdown)."""

import os
import tempfile
from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.core.types import DiscoveryPath, KTBoundary, ProblemState, ReasoningStep, OperatorType


def test_manuscript_generation_markdown_and_latex():
    """Validates converting a solved DiscoveryPath into publication-grade Markdown and LaTeX."""
    engine = SuperDuperProblemSolvingEngine()

    boundary = KTBoundary(
        identity_is="Sharp resistivity anomaly at 104 C",
        identity_is_not="True zero electrical resistance",
        location_is="Multiphase sintered pellets with Cu2S impurities",
        location_is_not="Pure stoichiometric Pb9Cu(PO4)6O crystals",
        timing_is="Occurs during 104 C first-order phase transition",
        timing_is_not="Persistent room temperature state",
        extent_is="Partial flux pinning and ferromagnetism",
        extent_is_not="Bulk Meissner expulsion",
    )

    problem = engine.formulate_problem(
        title="LK-99 Demarcation Investigation",
        specification="Anomalous resistance and half-levitation in modified lead apatite.",
        boundary=boundary,
    )

    path = DiscoveryPath(
        problem_id=problem.id,
        problem_title=problem.title,
        initial_abduction="Abduced candidate mechanisms",
        steps=[
            ReasoningStep(
                step_index=1,
                operator_type=OperatorType.PEIRCE_ABDUCTIVE_LEAP,
                operator_name="Peircean Abduction",
                description="Abduced competing hypotheses: Room-temp superconductor vs Cu2S phase transition.",
                confidence=0.50,
            ),
            ReasoningStep(
                step_index=2,
                operator_type=OperatorType.PLATT_CRUCIAL_EXPERIMENT,
                operator_name="Platt Crucial Experiment",
                description="Synthesized pure single crystal without Cu2S; observed transparent insulator.",
                confidence=0.98,
            ),
        ],
        final_breakthrough="CONFIRMED: Artifact caused by Cu2S first-order structural phase transition at 104 C",
        total_steps=2,
    )

    manuscript = engine.generate_manuscript(path, problem=problem)

    md = manuscript["markdown"]
    assert "# LK-99 Demarcation Investigation" in md
    assert "Kepner-Tregoe 4D Demarcation Boundary" in md
    assert "CONFIRMED: Artifact caused by Cu2S" in md
    assert "Strong Inference" in md

    latex = manuscript["latex"]
    assert "\\documentclass" in latex
    assert "\\title{LK-99 Demarcation Investigation}" in latex
    assert "\\begin{abstract}" in latex
    assert "CONFIRMED: Artifact caused by Cu2S" in latex
    assert "\\end{document}" in latex


def test_manuscript_save_to_disk():
    """Validates saving both LaTeX and Markdown manuscripts directly to disk."""
    engine = SuperDuperProblemSolvingEngine()
    problem = engine.formulate_problem(title="BICEP2 Cosmic Inflation Dust Demarcation", specification="Cosmic B-mode polarization.")
    path = DiscoveryPath(
        problem_id=problem.id,
        problem_title=problem.title,
        initial_abduction="Abduced candidate mechanisms",
        steps=[
            ReasoningStep(
                step_index=1,
                operator_type=OperatorType.PLATT_CRUCIAL_EXPERIMENT,
                operator_name="Platt Crucial Experiment",
                description="Planck 353 GHz dust frequency comparison",
                confidence=0.99,
            ),
        ],
        final_breakthrough="CONFIRMED: Galactic thermal dust polarization accounts for observed B-modes",
        total_steps=1,
    )


    with tempfile.TemporaryDirectory() as tmpdir:
        res = engine.generate_manuscript(path, problem=problem, output_dir=tmpdir)
        assert os.path.exists(res["markdown_path"])
        assert os.path.exists(res["latex_path"])

        with open(res["markdown_path"], "r") as f:
            content = f.read()
            assert "BICEP2" in content
            assert "Galactic thermal dust" in content
