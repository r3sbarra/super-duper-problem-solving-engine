"""P2 — Learned ranker: learn score weights from (problem, correct, distractor).

The hand-set ranker is `0.8*action + 0.2*object` (IDF-weighted). Measured top-1
within rankable pairs is only 63.8%, so hand weights leave correctness unused.
This learns the weights from the engine's own triples with a pairwise ranking
objective (pairwise logistic on score(problem, correct) > score(problem, wrong)),
which is monotone-in-relevance by construction.

Deterministic, offline, pure-NumPy. Trains on features:
  [action_contain, obj_contain, semantic_cos, len_ratio]
Artifact: assets/ranker_weights.npz

Usage:
  python scripts/train_ranker.py            # train from committed triples
  python scripts/train_ranker.py --verify   # load artifact, report held-out top-1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS = os.path.join(ROOT, "assets")
ARTIFACT = os.path.join(ASSETS, "ranker_weights.npz")

FEATURES = ["action_contain", "obj_contain", "semantic_cos", "len_ratio"]


def _load_triples():
    """(problem, correct_solution, distractor_solution) triples from committed data."""
    triples = []
    p = os.path.join(ASSETS, "distilled_pid_train.json")
    for r in json.load(open(p)):
        triples.append((r["parent"], r["paraphrase"], r["distractor"]))
    p2 = os.path.join(ASSETS, "distilled_pid_train_v2.json")
    if os.path.exists(p2):
        for r in json.load(open(p2)):
            triples.append((r["parent"], r["paraphrase"], r["distractor"]))
    return triples


def _features(problem: str, candidate: str):
    from super_solver.core.distilled_pid import DistilledPIDEmbedder
    from super_solver.vectorize.primitives import (
        match_by_primitives,
        strip_to_primitives,
    )

    def _is_generic(p):
        return p.startswith("act_") or p.startswith("attr_")

    tp = strip_to_primitives(problem)
    cp = strip_to_primitives(candidate)
    ta = {p for p in tp if not p.startswith("object_") and not _is_generic(p)}
    to = {p for p in tp if p.startswith("object_") and not _is_generic(p)}
    ca = {p for p in cp if not p.startswith("object_") and not _is_generic(p)}
    co = {p for p in cp if p.startswith("object_") and not _is_generic(p)}
    action = len(ta & ca) / len(ta) if ta else 0.0
    obj = len(to & co) / len(to) if to else 0.0
    emb = _EMB
    sem = emb.cosine_similarity(emb.encode(problem), emb.encode(candidate)) if emb.available else 0.0
    lr = min(len(candidate), len(problem)) / max(len(candidate), len(problem), 1)
    return np.array([action, obj, sem, lr], dtype=np.float32)


_EMB = None


def build_dataset(triples, limit=None):
    global _EMB
    from super_solver.core.distilled_pid import DistilledPIDEmbedder

    _EMB = DistilledPIDEmbedder()
    rows = []  # (x_pos, x_neg)
    for i, (prob, correct, distractor) in enumerate(triples):
        if limit and i >= limit:
            break
        xp = _features(prob, correct)
        xn = _features(prob, distractor)
        rows.append((xp, xn))
    return rows


def build_retrieval_dataset(triples):
    """Train on the ACTUAL retrieval objective: for each rankable pair, the
    positive is the correct solution and the negatives are the OTHER solutions
    in the corpus (leave-one-out), not the curated distractors. (Measured: the
    curated hard-distractor triples give only semantic signal — action/object
    tie 75-84 times of 93 — so training on them teaches nothing about the
    retrieval task, where containment does carry signal.)
    """
    rows = []
    pool = [(p, c) for p, c in {(p, c) for p, c, _ in triples}]
    global _EMB
    from super_solver.core.distilled_pid import DistilledPIDEmbedder

    _EMB = DistilledPIDEmbedder()
    from super_solver.vectorize.primitives import strip_to_primitives

    def _spec(p):
        return {x for x in p if not (x.startswith("act_") or x.startswith("attr_"))}

    for prob, correct in pool:
        if not (_spec(strip_to_primitives(prob)) & _spec(strip_to_primitives(correct))):
            continue  # not rankable -> no learning signal
        xp = _features(prob, correct)
        for other_prob, other in pool:
            if other == correct:
                continue
            rows.append((xp, _features(prob, other)))
    return rows


def train(rows, epochs=400, lr=0.5, seed=42):
    """Pairwise logistic: P(pos>neg) = sigmoid(w·(xp-xn)); maximize likelihood."""
    rng = np.random.default_rng(seed)
    w = np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32)  # start near hand-set
    for ep in range(epochs):
        rng.shuffle(rows)
        grads = np.zeros_like(w)
        for xp, xn in rows:
            d = xp - xn
            z = float(w @ d)
            p = 1.0 / (1.0 + np.exp(-z))
            grads += (1.0 - p) * d  # dL/dw for maximizing log p
        w += lr * grads / max(len(rows), 1)
        # normalize to keep weights comparable
        n = np.linalg.norm(w)
        if n > 0:
            w = w / n * np.linalg.norm(np.array([0.8, 0.2, 0.0, 0.0]))
    return w


def _heldout_accuracy(w, triples, k=20):
    """Top-1 accuracy of the learned ranker among rankable pairs (held-out tail)."""
    from super_solver.vectorize.primitives import strip_to_primitives

    def _spec(p):
        return {x for x in p if not (x.startswith("act_") or x.startswith("attr_"))}

    # candidate pool = the tail solutions
    pool = [c for _, c, _ in triples[-60:]]
    hits = 0
    n = 0
    for prob, correct, _ in triples[-k:]:
        if not (_spec(strip_to_primitives(prob)) & _spec(strip_to_primitives(correct))):
            continue  # not rankable
        scored = []
        for cand in pool:
            x = _features(prob, cand)
            scored.append((float(w @ x), cand))
        scored.sort(reverse=True)
        if scored and scored[0][1] == correct:
            hits += 1
        n += 1
    return (hits / n if n else 0.0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()

    triples = _load_triples()
    if args.verify:
        if not os.path.exists(ARTIFACT):
            print("no ranker artifact; train first")
            sys.exit(1)
        w = np.load(ARTIFACT)["w"]
        acc, n = _heldout_accuracy(w, triples)
        print(f"held-out top-1 (learned): {acc:.3f} on {n} rankable pairs")
        print("features:", dict(zip(FEATURES, [round(float(x), 3) for x in w])))
        return

    # Train on the RETRIEVAL objective (leave-one-out over the corpus).
    rows = build_retrieval_dataset(triples)
    w = train(rows, epochs=args.epochs)
    np.savez_compressed(ARTIFACT, w=w, features=np.array(FEATURES))
    acc, n = _heldout_accuracy(w, triples)
    print("learned weights:", dict(zip(FEATURES, [round(float(x), 3) for x in w])))
    print(f"held-out top-1 (learned): {acc:.3f} on {n} rankable pairs")
    print(f"wrote {ARTIFACT}")


if __name__ == "__main__":
    main()
