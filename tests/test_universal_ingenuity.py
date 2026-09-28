"""Unit tests for UniversalIngenuityEngine and domain-general ingenuity upgrades."""

import numpy as np
import pytest
from click.testing import CliRunner

from super_solver.cli import main
from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.frameworks.boed_designer import BOEDDesigner
from super_solver.frameworks.universal_ingenuity import (
    AuxiliaryPlateauUnblocker,
    BoundaryProbeType,
    InvariantQuotientEngine,
    UniversalBoundaryAuditor,
    UniversalIngenuityEngine,
)


def test_universal_boundary_auditor():
    auditor = UniversalBoundaryAuditor()
    spec = "High-throughput distributed cache experiencing latency spikes under failover"

    # Candidate with defensive guards
    safe_candidate = (
        "Implement idempotent quorum-based writes with bounded timeout and fallback to read-only replica"
    )
    report_safe = auditor.audit_candidate(safe_candidate, spec, domain_context="Distributed Systems")
    assert report_safe.overall_resilience > 0.5
    assert len(report_safe.probes) == 5

    # Fragile candidate that ignores scale and race conditions
    fragile_candidate = "Unbounded broadcast without sequence checks or timeout limits"
    report_fragile = auditor.audit_candidate(fragile_candidate, spec, domain_context="Distributed Systems")
    assert len(report_fragile.fatal_flaws) > 0
    assert report_fragile.overall_resilience < report_safe.overall_resilience


def test_invariant_quotient_coalescence():
    quotient = InvariantQuotientEngine(similarity_threshold=0.65)
    candidates = [
        "Increase worker connection pool to 50 threads",
        "Increase worker connection pool to 100 threads",
        "Scale worker thread pool dynamically to 200 threads",
        "Implement circuit breaker on database connections",
        "Introduce rate limiting on client requests",
    ]
    classes = quotient.coalesce_hypotheses(candidates, domain_context="Systems Infrastructure")
    # The three thread pool sizing hypotheses should coalesce into a single class
    assert len(classes) < len(candidates)
    pool_class = next(c for c in classes if "pool" in c.representative.lower())
    assert len(pool_class.members) >= 2


def test_auxiliary_plateau_unblocker():
    unblocker = AuxiliaryPlateauUnblocker()
    dim = 768
    current = np.ones(dim) / np.sqrt(dim)
    goal = np.zeros(dim)
    goal[0] = 1.0

    ops = unblocker.generate_unblocking_operators(
        current_latent=current,
        goal_vector=goal,
        problem_specification="Deadlock in multi-agent consensus protocol",
    )
    assert len(ops) == 4
    names = [op[0] for op in ops]
    assert any("Working Backwards" in n for n in names)
    assert any("Stepping-Stone" in n for n in names)
    assert any("Constraint Relaxation" in n for n in names)
    assert any("Analogy" in n for n in names)


def test_boed_structural_class_ranking():
    boed = BOEDDesigner()
    priors = {"h1": 0.5, "h2": 0.5}
    designs = [
        {
            "design_id": "exp_param",
            "name": "AdjustBufferTimeout",
            "description": "Increase timeout from 10ms to 15ms",
            "cost": 1.0,
        },
        {
            "design_id": "exp_invariant",
            "name": "VerifyConservedBoundaryInvariant",
            "description": "Test conservation law and boundary condition under network partition",
            "cost": 1.0,
        },
    ]
    likelihoods = {
        "exp_param": {"pos": {"h1": 0.8, "h2": 0.2}, "neg": {"h1": 0.2, "h2": 0.8}},
        "exp_invariant": {"pos": {"h1": 0.8, "h2": 0.2}, "neg": {"h1": 0.2, "h2": 0.8}},
    }
    ranked = boed.rank_experiments_by_structural_class(priors, designs, likelihoods)
    # The invariant/boundary test should be boosted to the top rank
    assert ranked[0]["design_id"] == "exp_invariant"
    assert ranked[0]["is_structural_test"] is True
    assert ranked[0]["cost_penalized_utility"] > ranked[1]["cost_penalized_utility"]


def test_engine_deduce_discovery_path_with_ingenuity():
    engine = SuperDuperProblemSolvingEngine()
    prob = engine.formulate_problem(
        title="Distributed Transaction Deadlock",
        specification="Microservices experience cyclic locks when executing multi-shard updates",
    )
    candidates = [
        "Increase retry timeout to 30 seconds",
        "Implement strict two-phase locking with global coordinator",
        "Adopt deterministic timestamp ordering with wait-die deadlock prevention",
    ]
    path = engine.deduce_discovery_path(prob, candidate_hypotheses=candidates)
    assert len(path.steps) > 0
    step_names = [s.operator_name for s in path.steps]
    assert any("Universal Adversarial Boundary Audit" in name for name in step_names)


def test_cli_ingenuity_audit_and_coalesce():
    runner = CliRunner()
    res_audit = runner.invoke(
        main,
        [
            "ingenuity-audit",
            "Unbounded queue buffer without backpressure",
            "--spec", "Memory exhaustion under burst traffic",
            "--domain", "Network Engineering",
        ],
    )
    assert res_audit.exit_code == 0
    assert "Universal Adversarial Boundary Audit" in res_audit.output

    res_coalesce = runner.invoke(
        main,
        [
            "coalesce",
            "Buffer size 10MB",
            "Buffer size 20MB",
            "Buffer size 50MB",
            "Switch to UDP protocol",
        ],
    )
    assert res_coalesce.exit_code == 0
    assert "Coalescence Class" in res_coalesce.output
