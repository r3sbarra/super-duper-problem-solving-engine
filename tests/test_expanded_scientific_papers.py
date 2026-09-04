"""Tests expanded landmark papers and self-referential improvement loop.

Validates:
1. Astrophysics: BICEP2 Cosmic Inflation B-Mode Anomaly (arXiv:1403.3985 / 1502.00612)
2. Biomedicine: STAP Acid-Induced Pluripotency Controversy (Obokata et al. 2014)
3. Mathematics: Yitang Zhang's Bounded Prime Gaps Parity Evasion (arXiv:1305.5589)
4. Self-Referential Engine Improvement & Auto-Tuning Loop
"""

import pytest
from super_solver.core.types import VerificationStatus
from super_solver.engine import SuperDuperProblemSolvingEngine


def test_bicep2_cosmic_inflation_dust_falsification():
    """Validates autonomous path deduction on the BICEP2 cosmic inflation claim (arXiv:1403.3985 / 1502.00612)."""
    engine = SuperDuperProblemSolvingEngine()

    arxiv_title = "Detection of B-Mode Polarization at Degree Angular Scales by BICEP2"
    arxiv_abstract = (
        "We report the detection of degree-scale B-mode polarization of the cosmic microwave "
        "background at 150 GHz. The observed excess corresponds to a tensor-to-scalar ratio "
        "r = 0.20, interpreted as direct evidence for primordial gravitational waves from "
        "cosmic inflation. However, galactic interstellar thermal dust emission may contribute "
        "polarized foreground contamination. Multi-frequency cross-correlation with high-frequency "
        "dust channels tests whether the signal is cosmological or astrophysical foreground."
    )

    discovery_path = engine.solve_unproven_arxiv_paper(
        paper_title=arxiv_title,
        abstract_text=arxiv_abstract,
        observed_experimental_result="anomaly_vanishes_in_pure_sample",
    )

    # 1. Fundamental inflation modification hypothesis must be falsified
    assert any("fundamental" in f.lower() or "breakdown" in f.lower() for f in discovery_path.falsified_paths)

    # 2. Interstellar dust / foreground selection artifact must be confirmed
    assert "artifact" in discovery_path.final_breakthrough.lower() or "contaminant" in discovery_path.final_breakthrough.lower()
    assert discovery_path.confidence > 0.70

    # 3. Verify Mermaid diagram generation
    mermaid = engine.export_mermaid_diagram(discovery_path)
    assert "CONFIRMED" in mermaid
    assert "BICEP2" in mermaid


def test_stap_cells_acid_pluripotency_falsification():
    """Validates autonomous path deduction on the STAP acid-induced pluripotency claim (Obokata et al. 2014)."""
    engine = SuperDuperProblemSolvingEngine()

    paper_title = "Stimulus-triggered fate conversion of somatic cells into pluripotency"
    paper_abstract = (
        "We report that mammalian somatic cells undergo epigenetic reprogramming into a pluripotent state "
        "when exposed to transient sub-lethal low-pH acid stress. Spleen cells treated at pH 5.7 express "
        "Oct4-GFP and form chimeric embryos. Skeptics propose that the observed pluripotency may instead be "
        "an artifact of laboratory cross-contamination with existing embryonic stem cell lines. Strict genetic "
        "sequencing and pure isolated culture assays test whether stress-induced reprogramming is genuine or artifact."
    )

    discovery_path = engine.solve_unproven_arxiv_paper(
        paper_title=paper_title,
        abstract_text=paper_abstract,
        observed_experimental_result="anomaly_vanishes_in_pure_sample",
    )

    # 1. Verify that the novel epigenetic reprogramming hypothesis was falsified
    assert any("fundamental" in f.lower() or "breakdown" in f.lower() for f in discovery_path.falsified_paths)

    # 2. Verify that ES cell cross-contamination / artifact was confirmed
    assert "artifact" in discovery_path.final_breakthrough.lower() or "contaminant" in discovery_path.final_breakthrough.lower()
    assert discovery_path.confidence > 0.70


def test_yitang_zhang_bounded_prime_gaps_verification():
    """Validates mathematical verification of Yitang Zhang's bounded prime gaps breakthrough (arXiv:1305.5589)."""
    engine = SuperDuperProblemSolvingEngine()

    paper_id = "arXiv:1305.5589"
    paper_title = "Bounded gaps between primes"
    paper_abstract = (
        "We prove that liminf (p_{n+1} - p_n) < 70,000,000 as n -> infinity. The proof overcomes the Selberg "
        "sieve parity barrier by establishing an extension of the Bombieri-Vinogradov theorem on prime distributions "
        "to level theta = 1/2 + 1/1168 for smooth moduli, decomposing error terms into Type I and Type II bilinear sums "
        "bounded by Deligne's Kloosterman estimates."
    )
    target_conjecture = "Bounded Prime Gaps / Twin Prime"

    result = engine.verify_math_paper(
        paper_id=paper_id,
        paper_title=paper_title,
        abstract_text=paper_abstract,
        target_conjecture=target_conjecture,
    )
    assert result.verdict == VerificationStatus.VERIFIED_SOUND
    assert result.confidence >= 0.90
    assert len(result.barrier_violations) == 0
    assert len(result.counterexamples) == 0

    # 2. Sound lemmas must record the bilinear decomposition
    assert any("bilinear" in s.lower() or "smooth" in s.lower() for s in result.sound_lemmas)


def test_self_improvement_auto_tuning_loop():
    """Validates the engine using its own problem-solving frameworks to optimize itself."""
    engine = SuperDuperProblemSolvingEngine()

    # 1. Run baseline diagnostic
    diag_before = engine.run_self_diagnostic()
    assert diag_before["status"] == "HEALTHY"
    initial_proximity = diag_before["active_parameters"]["proximity_threshold"]

    # 2. Engine solves the problem of improving itself
    improvement_path = engine.improve_self()
    assert improvement_path.total_steps >= 2
    assert "CONFIRMED" in improvement_path.final_breakthrough

    # 3. Verify auto-tuned parameter updates
    diag_after = engine.run_self_diagnostic()
    assert len(engine.self_improver.improvement_history) >= 1
    latest_update = engine.self_improver.improvement_history[-1]
    assert "updates_applied" in latest_update

    # 4. Verify that falsified dead ends from the self-improvement run were harvested
    assert len(engine.repulsor.dead_ends) >= 1
    assert any("Static rigid" in desc or "Auto-harvested" in desc for desc in engine.repulsor.dead_end_descriptions)
