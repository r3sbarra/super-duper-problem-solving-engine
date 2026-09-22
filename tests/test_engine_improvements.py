"""Unit tests for newly added engine capabilities and CLI commands."""

import pytest
from click.testing import CliRunner

from super_solver.cli import main
from super_solver.core.types import KTBoundary, MathematicalLemma
from super_solver.frameworks.boed_designer import BOEDDesigner
from super_solver.frameworks.gentner_sme import StructureMappingEngine
from super_solver.frameworks.kepner_tregoe import KepnerTregoeEngine
from super_solver.frameworks.math_verification import LakatosProofVerificationEngine


def test_structure_mapping_engine_retrieve_analogs():
    sme = StructureMappingEngine()
    spec = "A cooling fin transfers excess heat away from a central processing unit"
    candidates = [
        "Radiator dissipates thermal energy from an internal combustion engine",
        "A ledger tracks financial accounting credits and debits",
        "Heat sink with micro-channels conducts heat to ambient air",
    ]
    matches = sme.retrieve_analogs(spec, candidates, top_k=2)
    assert len(matches) <= 2
    assert len(matches) > 0
    # Top match should be thermal / heat transfer related
    top = matches[0]
    assert "score" in top
    assert any(term in top["candidate"].lower() for term in ["radiator", "heat sink", "thermal", "heat"])


def test_kepner_tregoe_analyze_boundary():
    kt = KepnerTregoeEngine()
    is_facts = ["Tail latency spikes only on worker nodes during DB flush"]
    is_not_facts = ["Leader node latency spikes", "Spikes during normal read queries"]
    causes = [
        "Synchronous fsync blocking thread pool on workers",
        "Cosmic ray bit flips on memory bus",
    ]
    res = kt.analyze_boundary(is_facts, is_not_facts, candidate_causes=causes)
    assert res["is_facts_count"] == 1
    assert res["is_not_facts_count"] == 2
    assert res["hyperplane_norm"] > 0
    evals = res["candidate_evaluations"]
    assert len(evals) == 2
    # fsync on worker should score higher sim to IS than cosmic rays
    assert evals[0]["sim_to_is"] > evals[1]["sim_to_is"]


def test_boed_designer_rank_experiments_alias():
    boed = BOEDDesigner()
    priors = {"h1": 0.5, "h2": 0.5}
    designs = [
        {"design_id": "exp1", "name": "DirectAssay", "cost": 1.0},
        {"design_id": "exp2", "name": "ControlAssay", "cost": 2.0},
    ]
    likelihoods = {
        "exp1": {"pos": {"h1": 0.9, "h2": 0.1}, "neg": {"h1": 0.1, "h2": 0.9}},
        "exp2": {"pos": {"h1": 0.5, "h2": 0.5}, "neg": {"h1": 0.5, "h2": 0.5}},
    }
    ranked = boed.rank_experiments(priors, designs, likelihoods)
    assert len(ranked) == 2
    assert ranked[0]["design_id"] == "exp1"
    assert ranked[0]["expected_information_gain"] > ranked[1]["expected_information_gain"]


def test_lakatos_suuri_symbolic_integration():
    lakatos = LakatosProofVerificationEngine()
    # Test symbolic query directly
    res = lakatos.verify_symbolic_identity("integrate x^2 dx")
    assert res is not None
    assert res["solved"] is True
    assert "1/3*x^3" in res["value"]


def test_cli_solve_full():
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["solve", "Tail Latency Spike", "--spec", "P99 latency jumps during cache invalidation", "--full"],
    )
    assert result.exit_code == 0
    assert "Discovery Trajectory Result" in result.output
    assert "Breakthrough:" in result.output


def test_cli_kt_boundary():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "kt-boundary",
            "-i", "Worker node memory high",
            "-n", "Leader node memory high",
            "-c", "Memory leak in worker connection pool",
        ],
    )
    assert result.exit_code == 0
    assert "Kepner-Tregoe Boundary Hyperplane" in result.output
    assert "Candidate Cause Evaluation" in result.output


def test_cli_decompose():
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["decompose", "Reduce memory footprint of distributed key-value store"],
    )
    assert result.exit_code == 0
    assert "Polya Auxiliary Problem Decomposition" in result.output
    assert "Auxiliary Angles" in result.output


def test_cli_boed():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "boed",
            "--hyp", "HypA",
            "--hyp", "HypB",
            "--design", "AssayA:1.0",
            "--design", "AssayB:2.0",
        ],
    )
    assert result.exit_code == 0
    assert "BOED Experiment Ranking" in result.output
    assert "AssayA" in result.output
