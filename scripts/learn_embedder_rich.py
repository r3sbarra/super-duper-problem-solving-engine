"""Richer-feature + learned-projection embedder, with honest train/test split.

The plain polarity hashing embedder has weak features (margin ~0.06, retrieval
~12.5% on its own corpus). A linear projection on those features barely helps
because the base features carry little semantic signal.

This script builds a RICHER base feature extractor (character n-grams + word
n-grams + subword 3-grams + bigrams, all hashed into a larger dim), then learns
a projection on top via contrastive loss. Crucially it evaluates on a HELD-OUT
split (train on some pairs, test on others) so we can see whether the learned
embedder generalizes or just memorizes the training pairs.

Usage:
  python scripts/learn_embedder_rich.py [--epochs 300] [--dim 512]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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


STOPWORDS = {"for", "using", "the", "a", "an", "in", "on", "and", "or", "to", "of", "with", "by", "is", "are", "as", "at", "it", "from", "that", "this"}


def _hash_feat(text, dim):
    """Hash a string into a dim-dim vector (feature hashing)."""
    v = np.zeros(dim, dtype=np.float32)
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
    v[h % dim] += 1.0
    return v


def rich_features(text, dim=512):
    """Extract a rich bag-of-ngrams feature vector (char + word + subword)."""
    v = np.zeros(dim, dtype=np.float32)
    low = text.lower()
    words = re.findall(r"\b[a-z0-9_]+\b", low)

    # Word unigrams (stopwords downweighted).
    for w in words:
        wgt = 0.15 if w in STOPWORDS else 1.0
        v += wgt * _hash_feat("w:" + w, dim)

    # Word bigrams.
    for i in range(len(words) - 1):
        v += 0.8 * _hash_feat("b:" + words[i] + "_" + words[i + 1], dim)

    # Character 3-grams (captures morphology/semantic roots).
    for j in range(len(low) - 2):
        v += 0.4 * _hash_feat("c:" + low[j : j + 3], dim)

    # Character 4-grams (captures longer roots like "reduc", "latenc").
    for j in range(len(low) - 3):
        v += 0.3 * _hash_feat("c4:" + low[j : j + 4], dim)

    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class RichEmbedder:
    def __init__(self, W=None, dim=512):
        self.dim = dim
        self.W = W  # (dim, dim) or None for identity
        self.name = "rich"

    def encode(self, text):
        v = rich_features(text, self.dim)
        if self.W is not None:
            v = self.W @ v
            n = np.linalg.norm(v)
            v = v / n if n > 0 else v
        return v

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


def train_contrastive(backend, pairs, dim=512, epochs=300, lr=0.05, margin=0.3, seed=42):
    rng = np.random.default_rng(seed)
    W = np.eye(dim, dtype=np.float32) + 0.01 * rng.standard_normal((dim, dim)).astype(np.float32)
    P = np.stack([backend.encode(p) for p, _ in pairs])
    S = np.stack([backend.encode(s) for _, s in pairs])
    n = len(pairs)
    neg_idx = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        neg_idx.append(rng.choice(others, size=min(4, len(others)), replace=False))

    t0 = time.time()
    for ep in range(epochs):
        grad = np.zeros_like(W)
        loss = 0.0
        for i in range(n):
            p = W @ P[i]
            s = W @ S[i]
            pn = np.linalg.norm(p) + 1e-8
            sn = np.linalg.norm(s) + 1e-8
            p_hat = p / pn
            s_hat = s / sn
            pos_sim = float(p_hat @ s_hat)
            for j in neg_idx[i]:
                sj = W @ S[j]
                sjn = np.linalg.norm(sj) + 1e-8
                sj_hat = sj / sjn
                neg_sim = float(p_hat @ sj_hat)
                d = margin - (pos_sim - neg_sim)
                if d > 0:
                    loss += d
                    grad += d * (np.outer(p_hat, s_hat) - np.outer(p_hat, sj_hat)) / (n * len(neg_idx[i]))
        W += lr * grad
        if ep % 20 == 0:
            U, _, Vt = np.linalg.svd(W)
            W = U @ Vt
        if ep % 100 == 0:
            print(f"  epoch {ep}: loss={loss:.4f}")
    print(f"  training took {time.time()-t0:.1f}s")
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--out", default="scratch/rich_embedder.npz")
    args = ap.parse_args()

    pairs = build_benchmark()
    print(f"Benchmark: {len(pairs)} related pairs")

    # Baseline: rich features, no projection.
    base = RichEmbedder(W=None, dim=args.dim)
    m0, acc0, rel0, unrel0 = evaluate(base, pairs)
    print(f"\n[baseline] rich features (no projection): margin={m0:.4f} acc={acc0:.3f} rel={rel0:.4f} unrel={unrel0:.4f}")

    # Train/test split: hold out 20% of pairs for testing generalization.
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(pairs))
    n_test = max(1, len(pairs) // 5)
    test_idx = set(idx[:n_test].tolist())
    train_pairs = [p for i, p in enumerate(pairs) if i not in test_idx]
    test_pairs = [p for i, p in enumerate(pairs) if i in test_idx]
    print(f"  train={len(train_pairs)} test={len(test_pairs)}")

    # Train on train split.
    print(f"\nTraining contrastive projection ({args.epochs} epochs)...")
    W = train_contrastive(base, train_pairs, dim=args.dim, epochs=args.epochs, lr=args.lr, margin=args.margin)

    learned = RichEmbedder(W=W, dim=args.dim)

    # Evaluate on train (memorization check) and test (generalization check).
    m_tr, acc_tr, _, _ = evaluate(learned, train_pairs)
    m_te, acc_te, rel_te, unrel_te = evaluate(learned, test_pairs)
    print(f"\n[train] learned: margin={m_tr:.4f} acc={acc_tr:.3f}")
    print(f"[test ] learned: margin={m_te:.4f} acc={acc_te:.3f} rel={rel_te:.4f} unrel={unrel_te:.4f}")

    # Baseline on test for comparison.
    m0_te, acc0_te, _, _ = evaluate(base, test_pairs)
    print(f"[test ] baseline: margin={m0_te:.4f} acc={acc0_te:.3f}")
    print(f"  test margin delta: {m_te-m0_te:+.4f}  test acc delta: {acc_te-acc0_te:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, W=W)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
