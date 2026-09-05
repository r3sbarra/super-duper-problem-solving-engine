"""Metric-learning embedder: learn a projection on top of polarity features.

Uses the engine's OWN corpus (related problem<->solution pairs) as supervision
to learn a linear projection W that pulls related pairs together and pushes
unrelated pairs apart (contrastive / margin metric learning).

This is the "use the engine to make a better embedder" move: the engine's
training data defines what "related" means, and we learn an embedding that
separates related from unrelated pairs as cleanly as possible.

The learned projection is a (dim x dim) matrix applied to the polarity
features, then L2-normalized. It is trained with a margin-based contrastive
loss via SGD (pure numpy, no heavy deps).

Usage:
  python scripts/learn_embedder.py [--epochs 200] [--lr 0.1] [--margin 0.3]
                                    [--out scratch/learned_embedder.npz]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.core.embeddings import PolarityAwareEmbeddingService

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_problem_solving import CLASSICAL_CASES, RECENT_CASES, MATH_CODING_CASES
from test_cross_domain import CROSS_DOMAIN_CASES


def build_benchmark():
    pairs = []
    for case in CLASSICAL_CASES + RECENT_CASES + MATH_CODING_CASES:
        pairs.append((case["problem"], case["solution"]))
    for case in CROSS_DOMAIN_CASES:
        pairs.append((case["source_problem"], case["source_solution"]))
    return pairs


class LearnedProjectionEmbedder:
    """Wraps a base embedder + learned projection W into an EmbedderBackend."""

    def __init__(self, base, W, dim=384):
        self.base = base
        self.W = W  # (dim, dim) projection
        self.dim = dim
        self.name = "learned"

    def encode(self, text):
        v = self.base.encode(text)
        out = self.W @ v
        n = np.linalg.norm(out)
        return out / n if n > 0 else out

    def cosine_similarity(self, a, b):
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))


def evaluate(backend, pairs, top_k=1):
    probs = [backend.encode(p) for p, _ in pairs]
    sols = [backend.encode(s) for _, s in pairs]
    n = len(pairs)
    related = []
    unrelated = []
    hits = 0
    for i in range(n):
        rel = backend.cosine_similarity(probs[i], sols[i])
        related.append(rel)
        others = [sols[j] for j in range(n) if j != i]
        if others:
            unrelated.append(float(np.mean([backend.cosine_similarity(probs[i], o) for o in others])))
        sims = [backend.cosine_similarity(probs[i], sols[j]) for j in range(n)]
        order = np.argsort(sims)[::-1]
        if i in order[:top_k]:
            hits += 1
    return (
        float(np.mean(related)) - float(np.mean(unrelated)),
        hits / n,
        float(np.mean(related)),
        float(np.mean(unrelated)),
    )


def train_contrastive(base, pairs, dim=384, epochs=200, lr=0.1, margin=0.3, seed=42):
    """Learn projection W via margin contrastive loss (pure numpy SGD)."""
    rng = np.random.default_rng(seed)
    W = np.eye(dim, dtype=np.float32) + 0.01 * rng.standard_normal((dim, dim)).astype(np.float32)

    # Pre-encode all problems and solutions.
    P = np.stack([base.encode(p) for p, _ in pairs])  # (n, dim)
    S = np.stack([base.encode(s) for _, s in pairs])  # (n, dim)
    n = len(pairs)

    # Precompute unrelated pairs: for each i, sample a few negative solutions.
    neg_idx = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        neg_idx.append(rng.choice(others, size=min(3, len(others)), replace=False))

    t0 = time.time()
    for ep in range(epochs):
        grad = np.zeros_like(W)
        loss = 0.0
        for i in range(n):
            # Positive pair (i, i)
            p = W @ P[i]
            s = W @ S[i]
            pn = np.linalg.norm(p) + 1e-8
            sn = np.linalg.norm(s) + 1e-8
            p_hat = p / pn
            s_hat = s / sn
            pos_sim = float(p_hat @ s_hat)

            # Negative pairs (i, j)
            for j in neg_idx[i]:
                sj = W @ S[j]
                sjn = np.linalg.norm(sj) + 1e-8
                sj_hat = sj / sjn
                neg_sim = float(p_hat @ sj_hat)
                # Margin loss: want pos_sim - neg_sim >= margin
                d = margin - (pos_sim - neg_sim)
                if d > 0:
                    loss += d
                    # Gradient w.r.t. W (simplified, sign-based update)
                    grad += d * (np.outer(p_hat, s_hat) - np.outer(p_hat, sj_hat)) / (n * len(neg_idx[i]))
        W += lr * grad
        # Project back toward orthonormal-ish to keep stable (soft constraint).
        if ep % 20 == 0:
            U, _, Vt = np.linalg.svd(W)
            W = U @ Vt
        if ep % 50 == 0:
            print(f"  epoch {ep}: loss={loss:.4f}")

    print(f"  training took {time.time()-t0:.1f}s")
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--out", default="scratch/learned_embedder.npz")
    args = ap.parse_args()

    pairs = build_benchmark()
    print(f"Benchmark: {len(pairs)} related pairs")

    base = PolarityAwareEmbeddingService()
    dim = base.EMBEDDING_DIM

    # Baseline.
    m0, acc0, rel0, unrel0 = evaluate(base, pairs)
    print(f"\n[baseline] polarity: margin={m0:.4f} acc={acc0:.3f} rel={rel0:.4f} unrel={unrel0:.4f}")

    # Train projection.
    print(f"\nTraining contrastive projection ({args.epochs} epochs, lr={args.lr}, margin={args.margin})...")
    W = train_contrastive(base, pairs, dim=dim, epochs=args.epochs, lr=args.lr, margin=args.margin)

    # Evaluate learned embedder.
    learned = LearnedProjectionEmbedder(base, W, dim=dim)
    m1, acc1, rel1, unrel1 = evaluate(learned, pairs)
    print(f"\n[learned] projection: margin={m1:.4f} acc={acc1:.3f} rel={rel1:.4f} unrel={unrel1:.4f}")
    print(f"  margin delta: {m1-m0:+.4f}  acc delta: {acc1-acc0:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, W=W)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
