"""Monte Carlo Tree Search with Process Reward Model (PRM).

Directs search across latent reasoning states using value-guided rollouts,
Platt exclusory information gain, and negative manifold hazard penalties.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from super_solver.core.embeddings import embedding_service
from super_solver.search.negative_manifold import NegativeManifoldRepulsor


class MCTSNode:
    """A node in the reasoning trajectory search tree."""

    def __init__(
        self,
        state_vector: np.ndarray,
        step_index: int = 0,
        operator_name: str = "ROOT",
        parent: Optional[MCTSNode] = None,
        op_vec: Optional[np.ndarray] = None,
        exclusory_gain: float = 0.0,
    ):
        self.state_vector = state_vector
        self.step_index = step_index
        self.operator_name = operator_name
        self.parent = parent
        self.op_vec = op_vec
        self.exclusory_gain = exclusory_gain
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.total_value: float = 0.0
        self.prm_step_score: float = 0.0
        # Track which operator indices have already been expanded
        self._expanded_op_indices: set = set()

    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0


class ProcessRewardModel:
    """Evaluates the intermediate quality of a latent reasoning transition."""

    def __init__(self, repulsor: Optional[NegativeManifoldRepulsor] = None):
        self.repulsor = repulsor or NegativeManifoldRepulsor()

    def score_transition(
        self,
        current_state: np.ndarray,
        next_state: np.ndarray,
        goal_vector: np.ndarray,
        exclusory_gain: float = 0.0,
    ) -> float:
        """Computes step-wise reward:
        Reward = 0.5 * ProgressToGoal + 0.3 * ExclusoryGain - 0.5 * DeadEndProximity.
        """
        sim_goal = embedding_service.cosine_similarity(next_state, goal_vector)
        progress = max(0.0, sim_goal)

        is_near, dead_end_sim, _ = self.repulsor.check_proximity(next_state)
        dead_end_penalty = dead_end_sim if is_near else 0.0

        step_reward = (0.5 * progress) + (0.3 * exclusory_gain) - (0.5 * dead_end_penalty)
        # Clamp to [-1, 1] (not [0.01, 1]) so that states near a dead end can carry a
        # NEGATIVE reward and are correctly avoided. The old [0.01, 1.0] floor clamped
        # every dead-end-adjacent state to 0.01, erasing the dead-end signal the
        # negative-manifold is meant to supply (everything near a dead end scored
        # identically). UCB1 exploitation is total_value/visits and works fine with
        # negative values, so discouraging dead-end branches is safe.
        return float(max(-1.0, min(1.0, step_reward)))


class LatentMCTSEngine:
    """Monte Carlo Tree Search over continuous/discrete reasoning graphs.

    Fix (was degenerate): previously, expand_and_evaluate() created ALL children
    in a single call, making N simulations equivalent to 1. Now each simulation
    expands exactly ONE previously-unexplored operator child (UCB1-guided), so
    num_simulations=N actually explores N distinct reasoning trajectories.
    """

    def __init__(self, prm: Optional[ProcessRewardModel] = None, c_puct: float = 1.414):
        self.prm = prm or ProcessRewardModel()
        self.c_puct = c_puct

    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Selects child with maximum Upper Confidence Bound (UCB1)."""
        best_child = None
        best_score = -float("inf")

        for child in node.children:
            if child.visits == 0:
                return child
            # UCB1 formula
            exploitation = child.value
            exploration = self.c_puct * math.sqrt(math.log(node.visits) / child.visits)
            ucb = exploitation + exploration
            if ucb > best_score:
                best_score = ucb
                best_child = child

        return best_child or node.children[0]

    def expand_one_child(
        self,
        node: MCTSNode,
        candidate_actions: List[Tuple[str, np.ndarray, float]],
        goal_vector: np.ndarray,
    ) -> Optional[MCTSNode]:
        """Expands exactly ONE unexplored operator child from the candidate list.

        Picks the first operator not yet tried from this node (in original order).
        Returns the new child, or None if all operators have been expanded.
        """
        for idx, (op_name, op_vec, exclusory_gain) in enumerate(candidate_actions):
            if idx in node._expanded_op_indices:
                continue
            # Expand this operator
            node._expanded_op_indices.add(idx)
            next_state = 0.7 * node.state_vector + 0.3 * op_vec
            norm = np.linalg.norm(next_state)
            if norm > 0:
                next_state = next_state / norm

            child = MCTSNode(
                state_vector=next_state,
                step_index=node.step_index + 1,
                operator_name=op_name,
                parent=node,
                op_vec=op_vec,
                exclusory_gain=exclusory_gain,
            )
            reward = self.prm.score_transition(
                node.state_vector,
                next_state,
                goal_vector,
                exclusory_gain=exclusory_gain,
            )
            child.prm_step_score = reward
            child.total_value = reward
            child.visits = 1
            node.children.append(child)
            return child

        return None  # All operators already expanded from this node

    def backpropagate(self, node: MCTSNode, value: float):
        """Backpropagates value up to root."""
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.total_value += value
            curr = curr.parent

    def search_best_path(
        self,
        initial_state: np.ndarray,
        candidate_operator_generators: List[Tuple[str, np.ndarray, float]],
        goal_vector: np.ndarray,
        num_simulations: int = 15,
    ) -> List[Tuple[str, np.ndarray, float]]:
        """Conducts MCTS and returns the highest-value trajectory.

        Each simulation expands one new operator child (fix for previously degenerate
        all-at-once expansion). N simulations now explore up to N distinct operators,
        with backpropagation updating UCB1 scores between simulations.
        """
        root = MCTSNode(state_vector=initial_state, step_index=0)

        for _ in range(num_simulations):
            curr = root

            # Selection: walk down already-expanded nodes using UCB1
            while curr.children and len(curr._expanded_op_indices) >= len(candidate_actions := candidate_operator_generators):
                curr = self.select_child(curr)

            # Expansion: add exactly one new child to the selected node
            new_child = self.expand_one_child(curr, candidate_operator_generators, goal_vector)
            if new_child is not None:
                # Simulation & backprop from the new child
                self.backpropagate(new_child, new_child.prm_step_score)
            elif curr.children:
                # All operators expanded at this node; backprop from best child
                best = max(curr.children, key=lambda c: c.prm_step_score)
                self.backpropagate(best, best.prm_step_score)

        # Extract best path by following highest-visited children
        path = []
        curr = root
        while curr.children:
            best_child = max(curr.children, key=lambda c: c.visits)
            path.append(
                (best_child.operator_name, best_child.state_vector, best_child.prm_step_score)
            )
            curr = best_child

        return path
