"""Train the neural embedder on the full 41k problem->solution corpus.

Uses the corpus structure directly as supervision: each (problem, solution)
pair is RELATED (pull together), and cross-case pairs are UNRELATED (push
apart). This is contrastive metric learning scaled to the full corpus — no
ollama needed, so it's fast.

The 41k pairs give the neural embedder far more semantic signal than the
192-pair distillation, so it should generalize much better than the small
corpus version.

Usage:
  python scripts/train_neural_embedder_large.py --corpus scratch/large_corpus.json
                                                --epochs 3 --dim 256
                                                --out scratch/neural_large.npz
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_neural_embedder import NeuralEmbedder, tokenize, evaluate, build_benchmark


def load_corpus(path, limit=None):
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def train_contrastive_large(emb, corpus, epochs=3, lr=0.05, margin=0.3, seed=42,
                            batch_neg=3, max_pairs=None):
    """Train with margin contrastive loss over the corpus (in-batch negatives)."""
    rng = np.random.default_rng(seed)
    if max_pairs:
        corpus = corpus[:max_pairs]
    n = len(corpus)
    # Pre-tokenize all problems and solutions.
    P = [emb._text_to_ids(e["problem"]) for e in corpus]
    S = [emb._text_to_ids(e["solution"]) for e in corpus]
    print(f"  {n} pairs, {len(emb._vocab)} vocab")

    t0 = time.time()
    for ep in range(epochs):
        total_loss = 0.0
        order = rng.permutation(n)
        for i in order:
            p_ids, p_wts = P[i]
            s_ids, s_wts = S[i]
            if not p_ids or not s_ids:
                continue
            p = emb._forward(p_ids, p_wts)
            s = emb._forward(s_ids, s_wts)
            pos_sim = float(p @ s)
            # In-batch negatives: random other solutions.
            negs = rng.choice([j for j in range(n) if j != i], size=min(batch_neg, n - 1), replace=False)
            for j in negs:
                sj_ids, sj_wts = S[j]
                if not sj_ids:
                    continue
                sj = emb._forward(sj_ids, sj_wts)
                neg_sim = float(p @ sj)
                d = margin - (pos_sim - neg_sim)
                if d > 0:
                    total_loss += d
                    grad_p = -d * (s - sj)
                    h = np.tanh(p @ emb.W1 + emb.b1)
                    emb.W2 -= lr * np.outer(h, grad_p)
                    emb.b2 -= lr * grad_p
                    grad_h = grad_p @ emb.W2.T
                    grad_pooled = grad_h * (1 - h**2) @ emb.W1.T
                    for k, wid in enumerate(p_ids):
                        emb.E[wid] -= lr * grad_pooled * p_wts[k] / (np.sum(p_wts) + 1e-8)
        print(f"  epoch {ep}: loss={total_loss:.4f} ({time.time()-t0:.0f}s)")
    print(f"  training took {time.time()-t0:.1f}s")
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="scratch/large_corpus.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--max-vocab", type=int, default=20000)
    ap.add_argument("--out", default="scratch/neural_large.npz")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus, args.max_pairs)
    print(f"Corpus: {len(corpus)} pairs")

    # Build vocab from the corpus, capped to the most frequent words.
    from collections import Counter
    counts = Counter()
    for e in corpus:
        counts.update(tokenize(e["problem"]))
        counts.update(tokenize(e["solution"]))
    top = counts.most_common(args.max_vocab)
    vocab = {w: i for i, (w, _) in enumerate(top)}
    print(f"Vocab: {len(vocab)} words (capped at {args.max_vocab})")

    emb = NeuralEmbedder(len(vocab), dim=args.dim, seed=42)
    emb._vocab = vocab

    # Baseline on the engine benchmark (before training).
    bench = build_benchmark()
    m0, acc0, _, _ = evaluate(emb, bench)
    print(f"\n[before] neural: margin={m0:.4f} acc={acc0:.3f}")

    # Train.
    print(f"\nTraining contrastive on {len(corpus)} pairs ({args.epochs} epochs)...")
    emb = train_contrastive_large(emb, corpus, epochs=args.epochs, lr=args.lr,
                                  margin=args.margin, max_pairs=args.max_pairs)

    # Evaluate on the engine benchmark.
    m1, acc1, rel1, unrel1 = evaluate(emb, bench)
    print(f"\n[after] neural: margin={m1:.4f} acc={acc1:.3f} rel={rel1:.4f} unrel={unrel1:.4f}")
    print(f"  margin delta: {m1-m0:+.4f}  acc delta: {acc1-acc0:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, E=emb.E, W1=emb.W1, b1=emb.b1, W2=emb.W2, b2=emb.b2,
             vocab=np.array(sorted(vocab.items())))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
