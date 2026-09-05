"""George Pólya's Heuristics ('How to Solve It', 1945).

1. Auxiliary Problem Decomposition (Divide & Conquer / Problem Reduction)
2. Working Backwards (Teleological goal regression)
3. Analogy to a simpler related problem
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from super_solver.core.embeddings import embedding_service


class PolyaHeuristicsEngine:
    """Pólya's Mathematical Discovery Heuristics Engine."""

    def __init__(self):
        pass

    def decompose_auxiliary_problems(
        self,
        problem_specification: str,
        known_subproblems: List[str],
    ) -> List[Dict[str, Any]]:
        """Identifies auxiliary subproblems whose solution unlocks the parent problem.

        Ranks the provided known_subproblems by cosine similarity to the
        problem statement (relevance) and flags the essential ones.  This is
        a re-ranking of the caller-supplied subproblems; for genuinely NEW
        subproblems generated from the problem text, use
        generate_auxiliary_problems().
        """
        v_prob = embedding_service.encode(problem_specification)

        sub_scores = []
        for sub in known_subproblems:
            v_sub = embedding_service.encode(sub)
            relevance = embedding_service.cosine_similarity(v_prob, v_sub)
            sub_scores.append(
                {
                    "subproblem": sub,
                    "relevance_score": float(relevance),
                    "is_essential": bool(relevance > 0.45),
                }
            )

        sub_scores.sort(key=lambda x: x["relevance_score"], reverse=True)
        return sub_scores

    def generate_auxiliary_problems(
        self,
        problem_specification: str,
        known_subproblems: Optional[List[str]] = None,
        max_new: int = 8,
    ) -> List[Dict[str, Any]]:
        """Generate GENUINELY NEW auxiliary subproblems from the problem text.

        Unlike decompose_auxiliary_problems (which only re-ranks caller-
        supplied subproblems), this applies Polya's decomposition heuristics
        to the problem statement itself and synthesizes new subproblems via
        structured templates.  Each generated subproblem is scored for
        relevance to the parent problem and de-duplicated against the
        known_subproblems so the caller gets fresh angles, not echoes.

        Returns a list of dicts:
            {
              "subproblem": str,
              "heuristic": "divide_and_conquer" | "working_backwards" |
                           "analogy" | "special_case" | "generalization" |
                           "invariant" | "counterexample_search" | "asymptotic",
              "relevance_score": float,
              "is_essential": bool,
              "is_new": bool,
            }
        """
        import re
        from collections import Counter

        known = set(known_subproblems or [])
        v_prob = embedding_service.encode(problem_specification)

        # --- Extract candidate domain terms (substantive tokens) ---
        _STOP = {
            "the", "and", "that", "this", "with", "from", "have", "has",
            "for", "are", "was", "were", "will", "would", "could", "should",
            "their", "there", "which", "where", "when", "what", "into",
            "over", "under", "about", "between", "through", "during", "after",
            "before", "then", "than", "them", "they", "these", "those",
            "such", "each", "other", "also", "only", "more", "most", "some",
            "any", "all", "both", "new", "find", "test", "using",
            "used", "use", "via", "can", "may", "must", "not", "non",
            "its", "our", "your", "their", "been", "being", "does", "done",
            "numerically", "testable", "falsifiable", "approaches", "approach",
            "hypothesis", "property", "target", "structure", "structural",
            "different", "mathematical", "domain", "large", "parameter",
            "strongest", "necessary", "consequence", "constituent", "parts",
            "independently", "analogous", "preserved", "generalization",
            "invariant", "monotone", "transformations", "extremal", "behavior",
            "counterexample", "violation", "asymptotic", "predicted", "growth",
            "rate", "boundary", "values", "extremal", "inputs", "special",
            "family", "larger", "whether", "holds", "within", "scanning",
        }
        # Split on hyphens too so "numerically-testable" -> "numerically","testable".
        raw = re.sub(r"[-_]", " ", problem_specification)
        tokens = [
            t.lower()
            for t in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", raw)
            if t.lower() not in _STOP
        ]
        freq = Counter(tokens)
        anchors = [t for t, _ in freq.most_common(6)]
        if not anchors:
            anchors = ["the problem"]

        # --- Polya decomposition templates ---
        templates = [
            ("divide_and_conquer",
             "Decompose {anchor} into its constituent structural parts and test each part independently for the target property."),
            ("working_backwards",
             "Assume the target property holds for {anchor}; derive the strongest necessary consequence and test whether it is numerically satisfied."),
            ("analogy",
             "Find an analogous structure to {anchor} in a different mathematical domain and test whether the analogous target property holds there."),
            ("special_case",
             "Restrict {anchor} to a special case (small parameters, boundary values, extremal inputs) and test the target property there."),
            ("generalization",
             "Generalize {anchor} to a larger family and test whether the target property is preserved under the generalization."),
            ("invariant",
             "Identify a quantity that is invariant or monotone under the transformations of {anchor} and test its extremal behavior."),
            ("counterexample_search",
             "Search for a counterexample to the target property within {anchor} by scanning a large parameter range for a violation."),
            ("asymptotic",
             "Test the asymptotic behavior of {anchor} in the large-parameter limit and compare against the predicted growth rate."),
        ]

        generated = []
        seen = set()
        for anchor in anchors:
            for heuristic, tmpl in templates:
                sub = tmpl.format(anchor=anchor)
                if sub in seen:
                    continue
                seen.add(sub)
                v_sub = embedding_service.encode(sub)
                relevance = embedding_service.cosine_similarity(v_prob, v_sub)
                is_new = sub not in known
                generated.append(
                    {
                        "subproblem": sub,
                        "heuristic": heuristic,
                        "relevance_score": float(relevance),
                        "is_essential": bool(relevance > 0.45),
                        "is_new": is_new,
                    }
                )

        # Rank: prefer new + relevant; cap at max_new.
        generated.sort(
            key=lambda x: (x["is_new"], x["relevance_score"]), reverse=True
        )
        return generated[:max_new]

    def plan_working_backwards(
        self,
        current_state_spec: str,
        desired_goal_spec: str,
        intermediate_milestones: List[str],
    ) -> List[str]:
        """Working Backwards: Plans backward from desired goal state to current state."""
        v_curr = embedding_service.encode(current_state_spec)
        v_goal = embedding_service.encode(desired_goal_spec)

        milestone_data = []
        for m in intermediate_milestones:
            vm = embedding_service.encode(m)
            dist_to_goal = 1.0 - embedding_service.cosine_similarity(vm, v_goal)
            dist_to_curr = 1.0 - embedding_service.cosine_similarity(vm, v_curr)
            milestone_data.append((m, dist_to_goal, dist_to_curr))

        # Order milestones: closest to goal -> furthest from goal (closer to current)
        milestone_data.sort(key=lambda x: x[1])
        ordered_steps = [m for m, d_g, d_c in reversed(milestone_data)]
        return ordered_steps
