"""Regression tests for the precedence-aware propositional parser and its
integration into ``DPLLSolver.refute_conjecture``.

Background (measured, before the fix): the solver extracted clauses from premise
strings with flat regex splits that ignored parentheses, so every grouped
formula was silently mis-parsed into garbage atoms and reported as
``COUNTERMODEL_EXISTS`` -- a *confident, wrong* "not entailed" verdict on
arguments that are plainly valid::

    ["(A AND B) -> C", "A", "B"] |= "C"     -> countermodel {"(a": ..., "b)": ...}
    ["P iff (Q AND R)", "Q", "R"] |= "P"    -> countermodel {"(q": ..., "r)": ...}

These tests lock in the corrected behaviour, and -- just as importantly -- lock
in that soundness was *not* traded away for it.
"""

from __future__ import annotations

import pytest

from super_solver.core.dpll_solver import DPLLSolver
from super_solver.core.prop_parser import (
    FormulaTooLarge,
    ParseError,
    parse_cnf,
    tokenize,
)

# ---------------------------------------------------------------------------
# Tokenizer / CNF unit level
# ---------------------------------------------------------------------------


def test_tokenizer_splits_operators_and_keeps_atoms_intact():
    kinds = [k for k, _ in tokenize("A and B")]
    assert kinds == ["ATOM", "AND", "ATOM"]

    # "greater_or_equal" must not be read as a disjunction: the word operator
    # is only recognised on non-word boundaries.
    atoms = [v for k, v in tokenize("x is greater_or_equal zero") if k == "ATOM"]
    assert atoms == ["x is greater_or_equal zero"]


def test_tokenizer_rejects_unbalanced_parentheses():
    with pytest.raises(ParseError):
        tokenize("(A AND B")
    with pytest.raises(ParseError):
        tokenize("A AND B)")


def test_parse_cnf_conjunction_and_disjunction():
    # A AND B  ->  two unit clauses
    assert parse_cnf("A and B") == [[(False, "A")], [(False, "B")]]
    # A OR B  ->  one two-literal clause
    assert parse_cnf("A or B") == [[(False, "A"), (False, "B")]]
    # A -> B  ->  ~A OR B
    assert parse_cnf("A -> B") == [[(True, "A"), (False, "B")]]


def test_parse_cnf_biconditional_matches_hand_rolled_encoding():
    """A <-> B must produce exactly the two clauses the solver used before."""
    cnf = parse_cnf("A iff B")
    as_sets = {frozenset(c) for c in cnf}
    assert as_sets == {
        frozenset({(True, "A"), (False, "B")}),
        frozenset({(False, "A"), (True, "B")}),
    }


def test_parse_cnf_drops_tautological_clauses():
    # A iff A is a tautology and collapses to no clauses at all.
    assert parse_cnf("A iff A") == []


def test_parse_cnf_respects_precedence():
    # NOT binds tighter than AND, which binds tighter than OR.
    # ~A and B  ==  (~A) AND B
    cnf = parse_cnf("not A and B")
    assert {frozenset(c) for c in cnf} == {
        frozenset({(True, "A")}),
        frozenset({(False, "B")}),
    }


def test_parse_cnf_clause_budget_is_enforced():
    # A chain of disjunctions distributes to an exponential number of clauses;
    # the cap must surface as a clear error rather than hanging.
    formula = " or ".join(f"A{i} and B{i}" for i in range(40))
    with pytest.raises(FormulaTooLarge):
        parse_cnf(formula, clause_cap=64)


# ---------------------------------------------------------------------------
# End-to-end refutation on grouped formulas (the actual bug)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "premises,target",
    [
        # The canonical report of the bug: split into atoms "(a", "b)", "c".
        (["(A AND B) -> C", "A", "B"], "C"),
        # Nested biconditional with a grouped conjunction.
        (["P iff (Q AND R)", "Q", "R"], "P"),
        # Distribution over a grouped disjunction.
        (["A OR (B AND C)", "NOT B"], "A"),
        # De Morgan over a grouped negation.
        (["NOT (A OR B)"], "NOT A"),
        # Nested implication with grouping on both sides.
        (["(A OR B) -> (C AND D)", "A"], "C"),
        # Double negation around a group.
        (["NOT NOT (A AND B)"], "B"),
    ],
)
def test_grouped_formulas_are_proved_structurally(premises, target):
    res = DPLLSolver().refute_conjecture(premises=premises, target_claim=target)
    assert res["proved"] is True, res["summary"]
    assert res["verdict"] == "LOGICALLY_SOUND_PROOF"
    assert res["parser"] == "regex_propositional"


@pytest.mark.parametrize(
    "premises,target",
    [
        # Affirming the consequent remains invalid -- grouping must not make
        # the solver credulous.
        (["(A AND B) -> C", "C"], "A"),
        # A group that does not entail the target.
        (["A OR (B AND C)"], "B"),
        # Only one direction of a biconditional is available.
        (["P -> (Q AND R)", "Q", "R"], "P"),
        # De Morgan must not be applied in the wrong direction.
        (["NOT A OR NOT B"], "NOT (A OR B)"),
    ],
)
def test_grouped_formulas_stay_sound(premises, target):
    res = DPLLSolver().refute_conjecture(premises=premises, target_claim=target)
    assert res["proved"] is False, res["summary"]
    assert res["verdict"] == "COUNTERMODEL_EXISTS"
    assert res["countermodel"] is not None


def test_countermodel_contains_no_fragment_atoms():
    """The tell-tale of the old bug was atoms carrying stray parentheses."""
    res = DPLLSolver().refute_conjecture(
        premises=["(A AND B) -> C", "C"], target_claim="A"
    )
    for atom in res["countermodel"]:
        assert "(" not in atom and ")" not in atom, res["countermodel"]


def test_flat_formulas_are_unaffected():
    """The pre-existing flat cases must keep their exact verdicts."""
    solver = DPLLSolver()
    assert solver.refute_conjecture(["A -> B", "B -> C", "A"], "C")["proved"] is True
    assert solver.refute_conjecture(["A iff B"], "A -> B")["proved"] is True
    assert solver.refute_conjecture(["A AND B"], "A")["proved"] is True
    assert solver.refute_conjecture(["A -> B", "B"], "A")["proved"] is False


def test_prose_still_degrades_to_the_semantic_fallback():
    """Ordinary prose is not a formula: it must still take the honest,
    fail-closed semantic path rather than being forced through the parser."""
    res = DPLLSolver().refute_conjecture(
        premises=["All men are mortal", "Socrates is a man"],
        target_claim="Socrates is mortal",
    )
    assert res["parser"] == "semantic_embedding_fallback"
    assert res["inconclusive"] is True
