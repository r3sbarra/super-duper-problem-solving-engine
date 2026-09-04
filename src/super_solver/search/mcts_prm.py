"""Monte Carlo Tree Search with Process Reward Model (PRM).

Directs search across latent reasoning states using value-guided rollouts,
Platt exclusory information gain, and negative manifold hazard penalties.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from super_solver.core.types import ReasoningStep, OperatorType
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
    ):
        self.state_vector = state_vector
        self.step_index = step_index
        self.operator_name = operator_name
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.total_value: float = 0.0
        self.prm_step_score: float = 0.0

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

        step_reward = (0.5 * progress) + (0.3 * exclusory_gain) - (0.6 * dead_end_penalty)
        return float(max(0.01, min(1.0, step_reward)))


class LatentMCTSEngine:
    """Monte Carlo Tree Search over continuous/discrete reasoning graphs."""

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

    def expand_and_evaluate(
        self,
        node: MCTSNode,
        candidate_actions: List[Tuple[str, np.ndarray, float]],
        goal_vector: np.ndarray,
    ) -> MCTSNode:
        """Expands node with candidate actions and scores them via PRM."""
        for op_name, op_vec, exclusory_gain in candidate_actions:
            next_state = 0.7 * node.state_vector + 0.3 * op_vec
            norm = np.linalg.norm(next_state)
            if norm > 0:
                next_state = next_state / norm

            child = MCTSNode(
                state_vector=next_state,
                step_index=node.step_index + 1,
                operator_name=op_name,
                parent=node,
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

        return node.children[0] if node.children else node

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
        """Conducts MCTS and returns the highest-value trajectory."""
        root = MCTSNode(state_vector=initial_state, step_index=0)

        for _ in range(num_simulations):
            curr = root
            # Selection
            while curr.children:
                curr = self.select_child(curr)

            # Expansion & Evaluation
            if curr.visits == 0 or not curr.children:
                leaf = self.expand_and_evaluate(curr, candidate_operator_generators, goal_vector)
                value = leaf.prm_step_score
                self.backpropagate(leaf, value)

        # Extract best path by following highest-visited children
        path = []
        curr = root
        while curr.children:
            best_child = max(curr.children, key=lambda c: c.visits)
            path.append((best_child.operator_name, best_child.state_vector, best_child.prm_step_score))
            curr = best_child

        return path
