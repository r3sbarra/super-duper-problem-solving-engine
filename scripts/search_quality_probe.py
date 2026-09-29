"""T2.1 — Search quality probe: does guided search beat random?

The health probe shows MCTS/DoT/coconut RUN, but not whether they find GOOD paths.
This applies the model-vs-bruteforce methodology: on a task with a known goal,
does the guided search reach the goal in fewer steps / higher value than random
operator selection?

Task: a synthetic latent navigation where each "operator" moves the state toward
or away from a known goal by a fixed amount. Guided search should reach the goal;
random selection should not (much).

Writes assets/search_quality.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DIM = 64
TRIALS = 20
SEED = 7


def make_problem(rng):
    goal = rng.standard_normal(DIM).astype(np.float32)
    goal /= np.linalg.norm(goal)
    # 10 operators: 2 move toward goal, 8 are noise/away.
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


def guided_value(mcts, start, goal, ops, rng):
    """Run MCTS search; return (final proximity, fraction of GOOD ops chosen).

    Proximity alone is confounded by step size; the decisive signal is whether
    the search SELECTS the goal-directed operators over the noise ones.
    """
    try:
        path = mcts.search_best_path(start, ops, goal, num_simulations=15)
    except Exception:
        return -1.0, 0.0
    if not path:
        return -1.0, 0.0
    good = sum(1 for name, _, _ in path if name.startswith("good"))
    frac_good = good / len(path)
    last = path[-1][1]
    return float(np.dot(last, goal)), frac_good


def random_value(start, goal, ops, rng):
    """Pick a random operator repeatedly; return best proximity reached."""
    best = float(np.dot(start, goal))
    state = start.copy()
    for _ in range(15):
        _, v, _ = ops[rng.integers(len(ops))]
        state = state + 0.1 * v
        state = state / (np.linalg.norm(state) + 1e-8)
        best = max(best, float(np.dot(state, goal)))
    return best


def main():
    from super_solver.search.mcts_prm import LatentMCTSEngine

    mcts = LatentMCTSEngine()
    rng = np.random.default_rng(SEED)

    guided, random_ = [], []
    guided_goodfrac, random_goodfrac = [], []
    for _ in range(TRIALS):
        goal, ops = make_problem(rng)
        start = -goal.copy()  # start far from goal
        gp, gf = guided_value(mcts, start, goal, ops, rng)
        guided.append(gp)
        guided_goodfrac.append(gf)
        # random baseline: expected good-op fraction = 2/10 = 0.2
        random_.append(random_value(start, goal, ops, rng))
        random_goodfrac.append(2 / 10)

    g_mean = float(np.mean(guided))
    r_mean = float(np.mean(random_))
    g_good = float(np.mean(guided_goodfrac))
    r_good = float(np.mean(random_goodfrac))
    wins = sum(1 for g, r in zip(guided, random_, strict=False) if g > r)
    # The decisive capability signal is operator-selection accuracy vs the
    # 0.2 random baseline.
    if g_good > r_good + 0.15 and g_mean > r_mean:
        verdict = "GUIDED_BEATS_RANDOM"
    elif abs(g_good - r_good) <= 0.15:
        verdict = "NO_ADVANTAGE"
    else:
        verdict = "GUIDED_WORSE"
    out = {
        "task": "latent goal navigation (2 good ops, 8 noise)",
        "trials": TRIALS,
        "guided_mean_proximity": round(g_mean, 4),
        "random_mean_proximity": round(r_mean, 4),
        "guided_good_op_fraction": round(g_good, 4),
        "random_good_op_fraction": round(r_good, 4),
        "guided_win_rate": round(wins / TRIALS, 4),
        "verdict": verdict,
        "note": "proximity is step-size-confounded; good-op fraction is the \
capability signal (random baseline 0.2)",
    }
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "search_quality.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
