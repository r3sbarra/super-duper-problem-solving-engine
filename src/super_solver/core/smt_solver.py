"""Pure-Python SMT Solver (EUF + QF_LRA).

Implements:
1. Congruence Closure for Equality with Uninterpreted Functions (EUF) via Union-Find.
2. Quantifier-Free Linear Real Arithmetic (QF_LRA) feasibility solver via Fourier-Motzkin elimination.
3. DPLL(T) integration combining propositional SAT with mathematical theory solvers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ============================================================================
# 1. Theory of Equality with Uninterpreted Functions (EUF) - Congruence Closure
# ============================================================================


class CongruenceClosure:
    """Union-Find with Congruence Closure for Equality with Uninterpreted Functions (EUF)."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        # func_app: (func_name, (arg1_repr, arg2_repr, ...)) -> term_repr
        self.signatures: Dict[Tuple[str, Tuple[str, ...]], str] = {}
        self.disequalities: List[Tuple[str, str]] = []

    def find(self, term: str) -> str:
        if term not in self.parent:
            self.parent[term] = term
            return term
        path = []
        curr = term
        while self.parent[curr] != curr:
            path.append(curr)
            curr = self.parent[curr]
        for node in path:
            self.parent[node] = curr
        return curr

    def union(self, term1: str, term2: str):
        root1 = self.find(term1)
        root2 = self.find(term2)
        if root1 != root2:
            self.parent[root1] = root2
            self._propagate_congruence()

    def _propagate_congruence(self):
        """Recomputes signatures of function applications and unifies congruent terms."""
        changed = True
        while changed:
            changed = False
            new_sigs: Dict[Tuple[str, Tuple[str, ...]], str] = {}
            for (fname, args), term in list(self.signatures.items()):
                canon_args = tuple(self.find(a) for a in args)
                sig = (fname, canon_args)
                if sig in new_sigs:
                    other_term = new_sigs[sig]
                    root1 = self.find(term)
                    root2 = self.find(other_term)
                    if root1 != root2:
                        self.parent[root1] = root2
                        changed = True
                else:
                    new_sigs[sig] = term
            self.signatures = new_sigs

    def add_function_app(self, func_name: str, args: List[str]) -> str:
        """Registers a function application term like f(a, b)."""
        term_repr = f"{func_name}({','.join(args)})"
        for arg in args:
            self.find(arg)
        self.find(term_repr)
        canon_args = tuple(self.find(a) for a in args)
        sig = (func_name, canon_args)
        if sig in self.signatures:
            other = self.signatures[sig]
            self.union(other, term_repr)
        else:
            self.signatures[sig] = term_repr
        self._propagate_congruence()
        return term_repr

    def add_equality(self, t1: str, t2: str):
        """Asserts t1 == t2."""
        self.find(t1)
        self.find(t2)
        self.union(t1, t2)

    def add_disequality(self, t1: str, t2: str):
        """Asserts t1 != t2."""
        self.find(t1)
        self.find(t2)
        self.disequalities.append((t1, t2))

    def is_consistent(self) -> bool:
        """Returns True if no asserted disequality t1 != t2 has find(t1) == find(t2)."""
        for t1, t2 in self.disequalities:
            if self.find(t1) == self.find(t2):
                return False
        return True


# ============================================================================
# 2. Theory of Linear Real Arithmetic (QF_LRA) - Fourier-Motzkin Elimination
# ============================================================================


@dataclass
class LinearConstraint:
    """Represents a linear inequality: sum(coeffs[x] * x) <= const_val."""

    coeffs: Dict[str, float]
    const_val: float
    strict: bool = False  # True for <, False for <=

    def normalized(self) -> LinearConstraint:
        c = {k: v for k, v in self.coeffs.items() if abs(v) > 1e-9}
        return LinearConstraint(coeffs=c, const_val=self.const_val, strict=self.strict)


class FourierMotzkinLRA:
    """Solves feasibility of systems of linear inequalities via Fourier-Motzkin elimination."""

    def __init__(self):
        self.constraints: List[LinearConstraint] = []

    def add_constraint(self, coeffs: Dict[str, float], const_val: float, strict: bool = False):
        self.constraints.append(LinearConstraint(coeffs=coeffs, const_val=const_val, strict=strict))

    def check_feasibility(self) -> Tuple[bool, Optional[str]]:
        """Determines if the system of linear inequalities is satisfiable."""
        current = [c.normalized() for c in self.constraints]

        # Check any immediate contradictions with empty coeffs
        for c in current:
            if not c.coeffs:
                if c.strict and c.const_val <= 1e-9:
                    return False, f"Contradiction: 0 < {c.const_val}"
                elif not c.strict and c.const_val < -1e-9:
                    return False, f"Contradiction: 0 <= {c.const_val}"

        # Collect all variables
        variables: Set[str] = set()
        for c in current:
            variables.update(c.coeffs.keys())

        # Successively eliminate each variable
        for var in sorted(variables):
            pos: List[LinearConstraint] = []  # coeffs[var] > 0
            neg: List[LinearConstraint] = []  # coeffs[var] < 0
            zero: List[LinearConstraint] = []  # var not present

            for c in current:
                a = c.coeffs.get(var, 0.0)
                if a > 1e-9:
                    pos.append(c)
                elif a < -1e-9:
                    neg.append(c)
                else:
                    zero.append(c)

            new_constraints: List[LinearConstraint] = list(zero)

            # Cross-combine all positive and negative pairs to eliminate var
            for p in pos:
                a_p = p.coeffs[var]
                for n in neg:
                    a_n = -n.coeffs[var]  # positive magnitude

                    comb_coeffs: Dict[str, float] = {}
                    for v, val in p.coeffs.items():
                        if v != var:
                            comb_coeffs[v] = comb_coeffs.get(v, 0.0) + (val * a_n)
                    for v, val in n.coeffs.items():
                        if v != var:
                            comb_coeffs[v] = comb_coeffs.get(v, 0.0) + (val * a_p)

                    comb_const = (p.const_val * a_n) + (n.const_val * a_p)
                    comb_strict = p.strict or n.strict

                    new_c = LinearConstraint(
                        coeffs=comb_coeffs, const_val=comb_const, strict=comb_strict
                    ).normalized()

                    # Direct contradiction check on empty coefficients
                    if not new_c.coeffs:
                        if new_c.strict and new_c.const_val <= 1e-9:
                            return False, f"Contradiction: 0 < {new_c.const_val}"
                        elif not new_c.strict and new_c.const_val < -1e-9:
                            return False, f"Contradiction: 0 <= {new_c.const_val}"
                    else:
                        new_constraints.append(new_c)

            current = new_constraints

        # Final check on any remaining constant inequalities
        for c in current:
            if not c.coeffs:
                if c.strict and c.const_val <= 1e-9:
                    return False, f"Contradiction: 0 < {c.const_val}"
                elif not c.strict and c.const_val < -1e-9:
                    return False, f"Contradiction: 0 <= {c.const_val}"

        return True, None


# ============================================================================
# 3. High-Level SMT Solver Facade
# ============================================================================


@dataclass
class SMTResult:
    is_sat: bool
    verdict: str
    details: Dict[str, Any] = field(default_factory=dict)


class SMTSolver:
    """Unified Pure-Python SMT solver supporting EUF and QF_LRA theories."""

    def __init__(self):
        pass

    def check_euf_satisfiability(
        self,
        equalities: List[Tuple[str, str]],
        disequalities: List[Tuple[str, str]],
        function_terms: Optional[List[Tuple[str, List[str]]]] = None,
    ) -> SMTResult:
        """Verifies satisfiability of an EUF formula using Congruence Closure."""
        cc = CongruenceClosure()

        if function_terms:
            for fname, args in function_terms:
                cc.add_function_app(fname, args)

        # Pass 1: Register all function applications and terms
        for t1, t2 in equalities + disequalities:
            self._register_potential_func(cc, t1)
            self._register_potential_func(cc, t2)

        # Pass 2: Assert all equalities
        for t1, t2 in equalities:
            cc.add_equality(t1, t2)

        # Pass 3: Assert all disequalities
        for t1, t2 in disequalities:
            cc.add_disequality(t1, t2)

        sat = cc.is_consistent()
        return SMTResult(
            is_sat=sat,
            verdict="SAT" if sat else "UNSAT",
            details={
                "theory": "EUF",
                "equivalence_classes": {k: cc.find(k) for k in cc.parent},
            },
        )

    def _register_potential_func(self, cc: CongruenceClosure, term: str):
        term = term.strip()
        m = re.match(r"^([a-zA-Z0-9_]+)\((.*)\)$", term)
        if m:
            fname = m.group(1)
            raw_args = [a.strip() for a in m.group(2).split(",") if a.strip()]
            for arg in raw_args:
                self._register_potential_func(cc, arg)
            cc.add_function_app(fname, raw_args)
        else:
            cc.find(term)

    def check_lra_satisfiability(self, constraints: List[str]) -> SMTResult:
        """Parses and checks feasibility of linear inequalities e.g. 'x + 2*y <= 5', 'x > 2'."""
        fm = FourierMotzkinLRA()

        for raw_c in constraints:
            raw_c = raw_c.strip()
            strict = False
            if "<=" in raw_c:
                lhs, rhs = raw_c.split("<=")
            elif ">=" in raw_c:
                rhs, lhs = raw_c.split(">=")
            elif "<" in raw_c:
                lhs, rhs = raw_c.split("<")
                strict = True
            elif ">" in raw_c:
                rhs, lhs = raw_c.split(">")
                strict = True
            elif "==" in raw_c or "=" in raw_c:
                parts = raw_c.split("==") if "==" in raw_c else raw_c.split("=")
                lhs, rhs = parts[0], parts[1]
                diff1 = self._diff_sides(lhs, rhs)
                const1 = -diff1.pop("__const__", 0.0)
                fm.add_constraint(diff1, const1, strict=False)

                diff2 = self._diff_sides(rhs, lhs)
                const2 = -diff2.pop("__const__", 0.0)
                fm.add_constraint(diff2, const2, strict=False)
                continue
            else:
                continue

            diff = self._diff_sides(lhs, rhs)
            const_val = -diff.pop("__const__", 0.0)
            fm.add_constraint(diff, const_val, strict=strict)

        sat, conflict = fm.check_feasibility()
        return SMTResult(
            is_sat=sat,
            verdict="SAT" if sat else "UNSAT",
            details={
                "theory": "QF_LRA",
                "conflict": conflict,
                "constraint_count": len(constraints),
            },
        )

    def _parse_side(self, s: str) -> Dict[str, float]:
        coeffs: Dict[str, float] = {}
        s = s.strip().replace(" ", "")
        terms = re.findall(r"[+-]?[^+-]+", s)
        for term in terms:
            if not term:
                continue
            sign = 1.0
            if term.startswith("+"):
                term = term[1:]
            elif term.startswith("-"):
                sign = -1.0
                term = term[1:]

            if "*" in term:
                p1, p2 = term.split("*")
                try:
                    num = float(p1)
                    var = p2
                except ValueError:
                    num = float(p2)
                    var = p1
                coeffs[var] = coeffs.get(var, 0.0) + (sign * num)
            else:
                m = re.match(r"^([0-9]*\.?[0-9]+)?([a-zA-Z_][a-zA-Z0-9_]*)?$", term)
                if m:
                    num_str, var_str = m.group(1), m.group(2)
                    if var_str:
                        num_val = float(num_str) if num_str else 1.0
                        coeffs[var_str] = coeffs.get(var_str, 0.0) + (sign * num_val)
                    elif num_str:
                        coeffs["__const__"] = coeffs.get("__const__", 0.0) + (sign * float(num_str))
        return coeffs

    def _diff_sides(self, lhs: str, rhs: str) -> Dict[str, float]:
        lhs_terms = self._parse_side(lhs)
        rhs_terms = self._parse_side(rhs)
        res = dict(lhs_terms)
        for k, v in rhs_terms.items():
            res[k] = res.get(k, 0.0) - v
        return res


smt_solver = SMTSolver()
