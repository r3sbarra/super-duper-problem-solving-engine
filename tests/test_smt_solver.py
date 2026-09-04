"""Tests for Pure-Python SMT Solver (EUF + QF_LRA)."""

from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.core.smt_solver import SMTSolver


def test_smt_euf_congruence_closure():
    """Validates equality with uninterpreted functions and congruence propagation."""
    solver = SMTSolver()

    # Case 1: Simple transitivity conflict (a == b, b == c, a != c -> UNSAT)
    res1 = solver.check_euf_satisfiability(
        equalities=[("a", "b"), ("b", "c")],
        disequalities=[("a", "c")],
    )
    assert res1.is_sat is False
    assert res1.verdict == "UNSAT"

    # Case 2: Congruence application: a == b implies f(a) == f(b)
    res2 = solver.check_euf_satisfiability(
        equalities=[("a", "b")],
        disequalities=[("f(a)", "f(b)")],
    )
    assert res2.is_sat is False
    assert res2.verdict == "UNSAT"

    # Case 3: Satisfiable structure
    res3 = solver.check_euf_satisfiability(
        equalities=[("a", "b"), ("c", "d")],
        disequalities=[("f(a)", "f(c)")],
    )
    assert res3.is_sat is True
    assert res3.verdict == "SAT"


def test_smt_qf_lra_linear_arithmetic():
    """Validates linear real arithmetic feasibility and inconsistency detection."""
    solver = SMTSolver()

    # Case 1: Direct contradictory bounds: x > 5 and x < 3 -> UNSAT
    res1 = solver.check_lra_satisfiability(["x > 5", "x < 3"])
    assert res1.is_sat is False
    assert res1.verdict == "UNSAT"

    # Case 2: System contradiction: x + y <= 10, x >= 6, y >= 5 -> UNSAT
    res2 = solver.check_lra_satisfiability(["x + y <= 10", "x >= 6", "y >= 5"])
    assert res2.is_sat is False
    assert res2.verdict == "UNSAT"

    # Case 3: Feasible system: x >= 0, y >= 0, 2*x + 3*y <= 12 -> SAT
    res3 = solver.check_lra_satisfiability(["x >= 0", "y >= 0", "2*x + 3*y <= 12"])
    assert res3.is_sat is True
    assert res3.verdict == "SAT"


def test_engine_smt_integration():
    """Validates SMT solver access through the master engine."""
    engine = SuperDuperProblemSolvingEngine()

    lra_sat = engine.check_smt_lra(["temperature >= 77", "temperature <= 300", "resistance >= 0"])
    assert lra_sat.is_sat is True

    lra_unsat = engine.check_smt_lra(["temperature > 100", "temperature < 50"])
    assert lra_unsat.is_sat is False

    euf_res = engine.check_smt_euf(
        equalities=[("p1", "p2")],
        disequalities=[("resistivity(p1)", "resistivity(p2)")],
    )
    assert euf_res.is_sat is False
