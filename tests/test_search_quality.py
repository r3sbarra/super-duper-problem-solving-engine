"""T2.1 — Search quality: guided search must beat random operator selection.

Guards the MCTS engine against silently degrading to random. On a latent goal
navigation task (2 goal-directed ops, 8 noise), the guided search must select
the goal-directed operators far more often than the 0.2 random baseline.
"""
from __future__ import annotations

import numpy as np

from super_solver.search.mcts_prm import LatentMCTSEngine

DIM = 64
TRIALS = 10
SEED = 7


def _make(rng):
    goal = rng.standard_normal(DIM).astype(np.float32)
    goal /= np.linalg.norm(goal)
    ops = []
    for i in range(2):
        v = goal + 0.15 * rng.standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        ops.append((f"good_{i}", v, 0.9))
    for i in range(8):
        v = rng.standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        ops.append((f"noise_{i}", v, 0.3))
    return goal, ops


def test_mcts_selects_goal_directed_operators():
    mcts = LatentMCTSEngine()
    rng = np.random.default_rng(SEED)
    fracs = []
    for _ in range(TRIALS):
        goal, ops = _make(rng)
        start = -goal.copy()
        path = mcts.search_best_path(start, ops, goal, num_simulations=15)
        assert path, "MCTS returned an empty path"
        good = sum(1 for name, _, _ in path if name.startswith("good"))
        fracs.append(good / len(path))
    mean_good = sum(fracs) / len(fracs)
    # Random baseline is 0.2; require a clear margin above it.
    assert mean_good > 0.6, f"guided search barely beats random: {mean_good:.2f}"


def test_mcts_improves_proximity_over_start():
    mcts = LatentMCTSEngine()
    rng = np.random.default_rng(SEED + 1)
    goal, ops = _make(rng)
    start = -goal.copy()
    start_prox = float(np.dot(start, goal))
    path = mcts.search_best_path(start, ops, goal, num_simulations=15)
    end_prox = float(np.dot(path[-1][1], goal))
    assert end_prox > start_prox, "MCTS did not improve goal proximity"
