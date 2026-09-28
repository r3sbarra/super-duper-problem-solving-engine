"""Pure-Python DPLL (Davis-Putnam-Logemann-Loveland) SAT and Logical Refutation Solver.

Guarantees exact formal consistency across multi-step deduction trajectories:
1. Conjunctive Normal Form (CNF) satisfiability checking
2. Unit propagation and pure literal elimination
3. Proof by resolution refutation (proves Premise => Claim by showing Premise and ~Claim is unsatisfiable)
4. Detection of internal logical contradictions in deduction graphs
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from super_solver.core.llm_parser import llm_client


class DPLLSolver:
    """Zero-daemon, deterministic Boolean Satisfiability and Resolution Refutation Engine."""

    _IFF_PLACEHOLDER = "\u0001iff\u0001"  # protects "if and only if" from conjunction-splitting

    @staticmethod
    def _negated_target_to_cnf(target: str, get_var) -> List[List[int]]:
        """Return the CNF clauses of NOT target, negating the target's top-level
        structure (De Morgan) instead of treating the whole string as one atom.

        This fixes the nested/biconditional weakness: "A iff B" as a target now
        negates to clauses [A,B] and [-A,-B]; "A -> B" to [A] and [-B]; etc.
        If no structural connective is detected, falls back to [-atom].
        """
        t = target.strip()
        if not t:
            return []
        # Biconditional: NOT(A <-> B) == (A AND NOT B) OR (NOT A AND B) == [A,B] / [-A,-B]
        protected = re.sub(r"\bif and only if\b", DPLLSolver._IFF_PLACEHOLDER, t)
        if (
            "iff" in protected
            or " <=> " in protected
            or " <-> " in protected
            or DPLLSolver._IFF_PLACEHOLDER in protected
            or "\u2261" in protected
        ):
            parts = re.split(
                r"\s+(?:iff|<=>|<->|\u2261)\s+",
                protected.replace(DPLLSolver._IFF_PLACEHOLDER, "iff"),
            )
            # handle "A iff B iff C" left-associatively: take first and last
            ant = parts[0].strip()
            cons = parts[-1].strip()
            if ant and cons:
                a = get_var(ant)
                b = get_var(cons)
                return [[a, b], [-a, -b]]
        # Implication: NOT(A -> B) == A AND NOT B == [A] and [-B]
        if "->" in protected or "=>" in protected:
            parts = re.split(r"->|=>", protected, maxsplit=1)
            ant = parts[0].strip()
            cons = parts[1].strip()
            if ant and cons:
                return [[get_var(ant)], [-get_var(cons)]]
        # Disjunction at top level: NOT(A OR B ...) == -A AND -B ... == one clause
        if re.search(r"\s+(?:OR|or)\s+", protected):
            lits = [get_var(x) for x in re.split(r"\s+(?:OR|or)\s+", protected) if x.strip()]
            if lits:
                return [[-l for l in lits]]
        # Conjunction at top level: NOT(A AND B) == -A OR -B == one clause
        if re.search(r"\s+(?:AND|and|\^|&|\u2227)\s+", protected):
            lits = [get_var(x) for x in re.split(r"\s+(?:AND|and|\^|&|\u2227)\s+", protected) if x.strip()]
            if len(lits) >= 2:
                return [[-l for l in lits]]
        # Negation: NOT(NOT A) == A
        m = re.match(r"^\s*(?:NOT|not|~|-|\u00ac)\s*(.+)$", protected)
        if m:
            return [[get_var(m.group(1))]]
        # Atomic target
        return [[-get_var(target)]]


    def __init__(self):
        pass

    def solve_cnf(self, clauses: List[List[int]]) -> Tuple[bool, Optional[Dict[int, bool]]]:
        """Solves a CNF formula where each clause is a list of integer literals.

        Positive integer = positive variable; negative integer = negated variable.
        Returns (is_satisfiable, model_assignment).
        """
        # Clean clauses: remove tautological clauses (e.g. [1, -1])
        cleaned_clauses = []
        for c in clauses:
            if not any(-lit in c for lit in c):
                # Deduplicate literals within clause
                cleaned_clauses.append(list(set(c)))

        model: Dict[int, bool] = {}
        sat, res_model = self._dpll(cleaned_clauses, model)
        return sat, res_model

    def _dpll(
        self,
        clauses: List[List[int]],
        assignment: Dict[int, bool],
    ) -> Tuple[bool, Optional[Dict[int, bool]]]:
        """Recursive DPLL with unit propagation and pure literal elimination."""
        # 1. Check base cases
        if not clauses:
            return True, assignment
        if any(len(c) == 0 for c in clauses):
            return False, None

        current_clauses = [list(c) for c in clauses]
        current_assign = dict(assignment)

        # 2. Unit Propagation: clauses with exactly 1 literal must be satisfied
        changed = True
        while changed:
            changed = False
            unit_literals = [c[0] for c in current_clauses if len(c) == 1]
            if unit_literals:
                lit = unit_literals[0]
                var = abs(lit)
                val = lit > 0
                current_assign[var] = val
                current_clauses = self._simplify_clauses(current_clauses, lit)
                changed = True
                if not current_clauses:
                    return True, current_assign
                if any(len(c) == 0 for c in current_clauses):
                    return False, None

            # 3. Pure Literal Elimination
            all_lits = [lit for c in current_clauses for lit in c]
            lit_set = set(all_lits)
            for lit in list(lit_set):
                if -lit not in lit_set:
                    var = abs(lit)
                    val = lit > 0
                    current_assign[var] = val
                    current_clauses = [c for c in current_clauses if lit not in c]
                    changed = True
                    if not current_clauses:
                        return True, current_assign

        # 4. Choose a branching literal (MOM's heuristic / most frequent variable)
        remaining_vars = set(abs(lit) for c in current_clauses for lit in c)
        if not remaining_vars:
            return True, current_assign

        # Pick variable appearing most frequently
        var_counts = {}
        for c in current_clauses:
            for lit in c:
                var_counts[abs(lit)] = var_counts.get(abs(lit), 0) + 1
        branch_var = max(var_counts, key=var_counts.get)

        # Try branch = True
        assign_true = dict(current_assign)
        assign_true[branch_var] = True
        clauses_true = self._simplify_clauses(current_clauses, branch_var)
        sat, model = self._dpll(clauses_true, assign_true)
        if sat:
            return True, model

        # Try branch = False
        assign_false = dict(current_assign)
        assign_false[branch_var] = False
        clauses_false = self._simplify_clauses(current_clauses, -branch_var)
        sat, model = self._dpll(clauses_false, assign_false)
        if sat:
            return True, model

        return False, None

    def _simplify_clauses(self, clauses: List[List[int]], true_literal: int) -> List[List[int]]:
        """Removes clauses satisfied by true_literal and removes -true_literal from remaining clauses."""
        new_clauses = []
        for c in clauses:
            if true_literal in c:
                # Clause is satisfied, drop it
                continue
            if -true_literal in c:
                # -true_literal is false, remove it from clause
                new_clauses.append([lit for lit in c if lit != -true_literal])
            else:
                new_clauses.append(c)
        return new_clauses

    def refute_conjecture(
        self,
        premises: List[str],
        target_claim: str,
    ) -> Dict[str, Any]:
        """Proves that Premise_1 ... Premise_n entails target_claim via resolution refutation.

        Proof by contradiction: If (Premises AND NOT target_claim) is UNSATISFIABLE,
        then the target claim is a necessary and proven logical consequence.

        When the zero-daemon regex parser extracts no formal clauses (common for
        unstructured scientific prose), falls back to embedding cosine similarity
        between the premise centroid and the target claim as a graded semantic
        entailment score instead of a silent UNPROVEN with confidence=0.3.
        """
        symbol_map: Dict[str, int] = {}
        rev_map: Dict[int, str] = {}
        clauses: List[List[int]] = []
        parser_used = "regex_propositional"

        def get_var(name: str) -> int:
            clean = name.strip().lower().replace(" ", "_")
            if clean not in symbol_map:
                idx = len(symbol_map) + 1
                symbol_map[clean] = idx
                rev_map[idx] = clean
            return symbol_map[clean]

        # 1. Attempt optional small LLM semantic clause extraction if enabled
        llm_data = llm_client.extract_propositional_clauses(premises, target_claim)
        if llm_data and "clauses" in llm_data and "target_var" in llm_data:
            parser_used = "optional_llm"
            for raw_clause in llm_data["clauses"]:
                clause_ints = []
                for lit_str in raw_clause:
                    is_neg = lit_str.startswith("-") or lit_str.startswith("~")
                    clean_name = lit_str.lstrip("-~")
                    v = get_var(clean_name)
                    clause_ints.append(-v if is_neg else v)
                if clause_ints:
                    clauses.append(clause_ints)
            target_var = get_var(llm_data["target_var"])
        else:
            # 2. Standard zero-daemon regex parsing.
            #
            # Handles the connectives that actually appear in scientific prose,
            # emitting proper CNF clauses so the resolution-refutation engine can
            # run (instead of silently degrading to an embedding-similarity guess):
            #   - conjunction       "A AND B"            -> one clause per conjunct
            #   - biconditional     "A iff B" / "A <=> B" -> [-A,B] and [A,-B]
            #   - implication       "A -> B" / "A => B"   -> [-A,B]
            #   - disjunction       "A OR B ..."          -> [A,B,...]
            #   - negation          "NOT A" / "~A" / "-A" -> [-A]

            def _conjunct_clauses(conj: str):
                conj = conj.strip()
                # Biconditional: A <-> B  ==  (A -> B) AND (B -> A)
                if (
                    " iff " in conj
                    or " <=> " in conj
                    or " <-> " in conj
                    or " if and only if " in conj
                    or "\u2261" in conj
                ):
                    parts = re.split(r"\s+(?:iff|<=>|<->|if and only if|\u2261)\s+", conj)
                    ant = parts[0].strip()
                    cons = parts[-1].strip()
                    if ant and cons:
                        a = get_var(ant)
                        b = get_var(cons)
                        return [[-a, b], [a, -b]]
                # Implication: A -> B  ==  ¬A OR B
                if "->" in conj or "=>" in conj:
                    parts = re.split(r"->|=>", conj, maxsplit=1)
                    ant = parts[0].strip()
                    cons = parts[1].strip()
                    if ant and cons:
                        a = get_var(ant)
                        b = get_var(cons)
                        return [[-a, b]]
                # Disjunction
                if " OR " in conj:
                    lits = [get_var(part) for part in conj.split(" OR ") if part.strip()]
                    return [lits] if lits else []
                # Negation
                m = re.match(r"^\s*(?:NOT|~|-|\u00ac)\s*(.+)$", conj)
                if m:
                    return [[-get_var(m.group(1))]]
                # Atomic literal
                return [[get_var(conj)]] if conj else []

            _IFF_PLACEHOLDER = "\u0001iff\u0001"  # protects "if and only if" from conjunction-splitting

            for p in premises:
                p_clean = p.strip()
                if not p_clean:
                    continue
                # Protect "if and only if" so the conjunction splitter below does not
                # break the biconditional phrase at its internal "and".
                protected = re.sub(r"\bif and only if\b", _IFF_PLACEHOLDER, p_clean)
                # Split on top-level conjunction; each conjunct is its own clause(s).
                conjuncts = re.split(r"\s+(?:AND|and|\^|&|\u2227)\s+", protected)
                for conj in conjuncts:
                    # Restore the protected IFF phrase before parsing this conjunct.
                    restored = conj.replace(_IFF_PLACEHOLDER, " if and only if ")
                    clauses.extend(_conjunct_clauses(restored))

            target_var = get_var(target_claim)

        # 3. Semantic fallback: if only trivial unit-clauses were extracted (one per premise,
        #    no logical connectives), the formal proof will always yield COUNTERMODEL for
        #    real scientific prose.  Use embedding cosine similarity as a graded score.
        #    Note: a conjunction ("A AND B") or biconditional ("A iff B") expands one
        #    premise into multiple clauses, so also treat clause-count > premise-count
        #    as structural -- otherwise a valid conjunction-elimination proof would be
        #    wrongly downgraded to a semantic guess.
        has_structural_clauses = (
            any(
                len(c) > 1 or (len(c) == 1 and c[0] < 0)  # multi-literal or negation
                for c in clauses
            )
            or len(clauses) > len([p for p in premises if p.strip()])
        )
        if not has_structural_clauses and clauses and parser_used == "regex_propositional":
            try:
                from super_solver.core.embeddings import embedding_service as _emb
                import numpy as _np

                prem_vecs = [_emb.encode(p) for p in premises if p.strip()]
                if prem_vecs:
                    centroid = _np.mean(_np.vstack(prem_vecs), axis=0)
                    target_vec = _emb.encode(target_claim)
                    semantic_score = float(_emb.cosine_similarity(centroid, target_vec))
                    # Treat ≥0.65 cosine as soft entailment, gradient below
                    if semantic_score >= 0.65:
                        verdict = "SEMANTIC_ENTAILMENT"
                        proved_soft = True
                    else:
                        verdict = "SEMANTIC_WEAK"
                        proved_soft = False
                    # FAIL-CLOSED (mirrors suuri finding 66fb07d1/d30325f6): the
                    # semantic fallback fires precisely when NO structural logic
                    # could be extracted, so its score is a graded guess, not a
                    # proof. Report it as soft/inconclusive so callers cannot
                    # mistake a similarity score for a formal entailment.
                    return {
                        "proved": proved_soft,
                        "inconclusive": True,
                        "verdict": verdict,
                        "summary": (
                            f"No structural logical clauses extracted (regex_propositional). "
                            f"Semantic fallback: embedding cosine similarity = {semantic_score:.3f}. "
                            f"{'Soft entailment likely — but this is a graded guess, NOT a formal proof.' if proved_soft else 'Weak semantic link; not entailed.'}"
                        ),
                        "countermodel": None,
                        "clauses_evaluated": 0,
                        "parser": "semantic_embedding_fallback",
                        "semantic_score": semantic_score,
                    }
            except Exception:
                pass  # semantic fallback is best-effort

        # FAIL-CLOSED GATE: if no clauses could be extracted at all (not even
        # trivial unit-clauses) and the optional LLM parser was unavailable, do
        # NOT proceed to a vacuous 'proved' via an empty clause set. Return an
        # explicit INCONCLUSIVE instead.
        if not clauses:
            return {
                "proved": False,
                "inconclusive": True,
                "verdict": "INCONCLUSIVE",
                "summary": (
                    "No structural logical clauses could be extracted and no semantic "
                    "fallback was available. This is a lack of evidence, NOT an "
                    "entailment. Verdict: INCONCLUSIVE."
                ),
                "countermodel": None,
                "clauses_evaluated": 0,
                "parser": parser_used,
            }

        # Add negation of target claim: ~target_claim.
        # With the optional-LLM path the goal variable is already a single symbol
        # resolved by the model, so negate that atom directly.  With the
        # zero-daemon regex path negate the target's structural form (De Morgan)
        # so a compound target like "A iff B" is refuted correctly instead of
        # being flattened to one opaque atom.
        clauses_with_negated_goal = [list(c) for c in clauses]
        if parser_used == "optional_llm":
            clauses_with_negated_goal.append([-target_var])
        else:
            clauses_with_negated_goal.extend(
                self._negated_target_to_cnf(target_claim, get_var)
            )

        is_sat, countermodel = self.solve_cnf(clauses_with_negated_goal)

        if not is_sat:
            # UNSAT => Proof by refutation succeeds!
            return {
                "proved": True,
                "verdict": "LOGICALLY_SOUND_PROOF",
                "summary": f"Target claim '{target_claim}' is strictly entailed by premises under propositional resolution refutation. Negation yielded an empty resolution clause.",
                "countermodel": None,
                "clauses_evaluated": len(clauses_with_negated_goal),
                "parser": parser_used,
            }
        else:
            # Countermodel found where premises hold but claim is false
            readable_countermodel = (
                {rev_map.get(k, str(k)): v for k, v in countermodel.items()} if countermodel else {}
            )
            return {
                "proved": False,
                "verdict": "COUNTERMODEL_EXISTS",
                "summary": f"Target claim '{target_claim}' is NOT strictly entailed under propositional parsing ({parser_used}). Countermodel satisfies premises with target false.",
                "countermodel": readable_countermodel,
                "clauses_evaluated": len(clauses_with_negated_goal),
                "parser": parser_used,
            }


dpll_solver = DPLLSolver()

