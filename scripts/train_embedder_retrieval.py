"""Rest of phase — train the distilled embedder on the RETRIEVAL objective.

The P1 semantic lift was small because the distilled embedder was trained on
paraphrase-vs-curated-distractor pairs, where the embedding alone scores 0.203
top-1 on the actual retrieval corpus. This retrains the student so that, for a
problem, the CORRECT solution scores above the OTHER solutions in the corpus
(leave-one-out negatives) — the objective retrieval actually measures.

Deterministic, offline. Writes assets/distilled_pid_large.npz (same artifact
contract) and reports held-out top-1 before/after.

Usage: python scripts/train_embedder_retrieval.py [--epochs 1200]
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
DIM = 512
WEIGHTS = os.path.join(ASSETS, "distilled_pid_large.npz")


def _features(text, dim=DIM):
    from super_solver.core.distilled_pid import features

    return features(text, dim)


def build_corpus():
    from super_solver.vectorize.domain_knowledge import DOMAIN_KNOWLEDGE

    pairs = [(e["problem_context"], e["solution_direction"]) for e in DOMAIN_KNOWLEDGE]
    for fn in ("distilled_pid_train.json", "distilled_pid_train_v2.json"):
        p = os.path.join(ASSETS, fn)
        if os.path.exists(p):
            for r in json.load(open(p)):
                pairs.append((r["parent"], r["paraphrase"]))
    # dedupe, keep order
    seen = set()
    out = []
    for pr, so in pairs:
        k = (pr, so)
        if k in seen:
            continue
        seen.add(k)
        out.append((pr, so))
    return out


def train(W, F, anchors, targets, negs, epochs, lr, seed=42):
    """Contrastive margin ranking, fully vectorized.

    For each problem i: positive = its correct solution, negatives = other
    solutions. Score = cosine in the projected space. Hinge loss
    max(0, m - (sim_pos - sim_neg)); gradient pushes pos up / neg down.
    """
    # Flatten (anchor, pos, neg) triples once.
    P_list, N_list = [], []
    for i in range(len(anchors)):
        for j in negs[i]:
            P_list.append(i)
            N_list.append(j)
    P_idx = np.asarray(P_list, dtype=int)
    N_idx = np.asarray(N_list, dtype=int)
    a_idx = anchors[P_idx]
    p_idx = targets[P_idx]
    n_idx = targets[N_idx]
    m = 0.2
    n = len(a_idx)
    for ep in range(epochs):
        A = F[a_idx] @ W.T
        P = F[p_idx] @ W.T
        Ng = F[n_idx] @ W.T
        Ah = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        Ph = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        Nh = Ng / (np.linalg.norm(Ng, axis=1, keepdims=True) + 1e-8)
        sim_p = np.sum(Ah * Ph, axis=1)
        sim_n = np.sum(Ah * Nh, axis=1)
        margin = sim_p - sim_n
        active = margin < m
        loss = float(np.sum(np.maximum(0.0, m - margin)) / n)
        if not np.any(active):
            if (ep + 1) % max(1, epochs // 4) == 0:
                print(f"  ep {ep+1}: hinge={loss:.4f} (converged)")
            continue
        # gradient on active triples only
        fa = F[a_idx[active]]
        fp = F[p_idx[active]]
        fn = F[n_idx[active]]
        ah = Ah[active]
        ph = Ph[active]
        nh = Nh[active]
        # d/dW (sim_p - sim_n) = fa⊗ph + ah⊗fp - (fa⊗nh + ah⊗fn)
        G = (
            fa.T @ ph + ah.T @ fp - fa.T @ nh - ah.T @ fn
        )
        W += lr * (G / max(int(active.sum()), 1))
        if (ep + 1) % max(1, epochs // 4) == 0:
            print(f"  ep {ep+1}: hinge={loss:.4f} active={int(active.sum())}")
    return W


def top1(W, corpus, F, dim=DIM):
    """Leave-one-out top-1 in PAIR space.

    For problem i, rank all OTHER solutions (j != i) and check the correct one is
    first. Earlier versions leaked (query's own entry in the pool) or misaligned
    pair-space with text-space; this uses one row per PAIR.
    """
    # one row per pair: anchor features (problem) and target features (solution)
    A = np.stack([_features(pr) for pr, _ in corpus]).astype(np.float32) @ W.T
    S = np.stack([_features(so) for _, so in corpus]).astype(np.float32) @ W.T
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
    n = len(corpus)
    hits = 0
    for i, (_, correct) in enumerate(corpus):
        sims = S @ A[i]
        sims[i] = -np.inf  # exclude the query's own (problem,solution) row
        best = int(np.argmax(sims))
        if corpus[best][1] == correct:
            hits += 1
    return hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    corpus = build_corpus()
    print(f"retrieval corpus: {len(corpus)} pairs")
    texts = list(dict.fromkeys([t for pr, so in corpus for t in (pr, so)]))
    idx = {t: i for i, t in enumerate(texts)}
    F = np.stack([_features(t) for t in texts]).astype(np.float32)

    anchors = np.array([idx[pr] for pr, _ in corpus])
    targets = np.array([idx[so] for _, so in corpus])
    rng = np.random.default_rng(0)
    negs = []
    for i in range(len(corpus)):
        pool = [j for j in range(len(corpus)) if j != i]
        negs.append(rng.choice(pool, size=min(4, len(pool)), replace=False))

    before = np.load(WEIGHTS)["W"].astype(np.float32) if os.path.exists(WEIGHTS) else np.eye(DIM, dtype=np.float32)
    print(f"before top-1: {top1(before, corpus, F):.3f}")
    W = before.copy()
    W = train(W, F, anchors, targets, negs, args.epochs, args.lr)
    after = top1(W, corpus, F)
    print(f"after  top-1: {after:.3f}")
    np.savez_compressed(WEIGHTS, W=W, dim=DIM)
    print(f"wrote {WEIGHTS}")


if __name__ == "__main__":
    main()
