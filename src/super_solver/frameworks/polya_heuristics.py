"""George Pólya's Heuristics ('How to Solve It', 1945).

1. Auxiliary Problem Decomposition (Divide & Conquer / Problem Reduction)
2. Working Backwards (Teleological goal regression)
3. Analogy to a simpler related problem
"""

from __future__ import annotations

from typing import Any, Dict, List

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
        """Identifies auxiliary subproblems whose solution unlocks the parent problem."""
        v_prob = embedding_service.encode(problem_specification)

        sub_scores = []
        for sub in known_subproblems:
            v_sub = embedding_service.encode(sub)
            relevance = embedding_service.cosine_similarity(v_prob, v_sub)
            sub_scores.append({
                "subproblem": sub,
                "relevance_score": float(relevance),
                "is_essential": bool(relevance > 0.45),
            })

        sub_scores.sort(key=lambda x: x["relevance_score"], reverse=True)
        return sub_scores

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
