"""Distill ollama's semantic knowledge into a fast local neural embedder.

Uses ollama (nomic-embed-text) as a TEACHER: it scores a set of (text_a,
text_b) pairs with cosine similarity. We then train a small local neural
embedder to REPRODUCE those similarity scores (regression loss), transferring
ollama's semantic knowledge into a fast, offline, deterministic model.

This is the classic knowledge-distillation path for making a small model match
a big one. The teacher provides the semantic signal; the student learns to
compress it.

Usage:
  python scripts/distill_embedder.py --scores /tmp/distill_scores.json
                                     --epochs 600 --dim 128
                                     --out scratch/distilled_embedder.npz
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


def load_scores(path):
    with open(path) as f:
        return json.load(f)


def train_distill(emb, scored_pairs, epochs=600, lr=0.05, seed=42):
    """Train the neural embedder to match teacher (ollama) similarities."""
    rng = np.random.default_rng(seed)
    # Pre-tokenize all pairs.
    data = []
    for rec in scored_pairs:
        a_ids, a_wts = emb._text_to_ids(rec["a"])
        b_ids, b_wts = emb._text_to_ids(rec["b"])
        if not a_ids or not b_ids:
            continue
        data.append((a_ids, a_wts, b_ids, b_wts, float(rec["sim"])))
    print(f"  {len(data)} trainable pairs")

    t0 = time.time()
    for ep in range(epochs):
        total_loss = 0.0
        rng.shuffle(data)
        for a_ids, a_wts, b_ids, b_wts, target in data:
            va = emb._forward(a_ids, a_wts)
            vb = emb._forward(b_ids, b_wts)
            pred = float(va @ vb)
            err = pred - target
            loss = err * err
            total_loss += loss
            # Gradient: dL/dpred = 2*err; dL/dva = 2*err*vb, dL/dvb = 2*err*va
            grad_va = 2 * err * vb
            grad_vb = 2 * err * va
            # Backprop through MLP + pooling (approximate).
            ha = np.tanh(va @ emb.W1 + emb.b1)
            hb = np.tanh(vb @ emb.W1 + emb.b1)
            emb.W2 -= lr * (np.outer(ha, grad_va) + np.outer(hb, grad_vb))
            emb.b2 -= lr * (grad_va + grad_vb)
            grad_ha = grad_va @ emb.W2.T
            grad_hb = grad_vb @ emb.W2.T
            grad_pa = grad_ha * (1 - ha**2) @ emb.W1.T
            grad_pb = grad_hb * (1 - hb**2) @ emb.W1.T
            for k, wid in enumerate(a_ids):
                emb.E[wid] -= lr * grad_pa * a_wts[k] / (np.sum(a_wts) + 1e-8)
            for k, wid in enumerate(b_ids):
                emb.E[wid] -= lr * grad_pb * b_wts[k] / (np.sum(b_wts) + 1e-8)
        if ep % 100 == 0:
            print(f"  epoch {ep}: loss={total_loss:.4f}")
    print(f"  training took {time.time()-t0:.1f}s")
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="/tmp/distill_scores.json")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="scratch/distilled_embedder.npz")
    args = ap.parse_args()

    scored = load_scores(args.scores)
    print(f"Loaded {len(scored)} teacher-scored pairs")

    # Build vocab from all texts in the scored pairs.
    vocab = {}
    for rec in scored:
        for t in tokenize(rec["a"]) + tokenize(rec["b"]):
            if t not in vocab:
                vocab[t] = len(vocab)
    print(f"Vocab: {len(vocab)} words")

    emb = NeuralEmbedder(len(vocab), dim=args.dim, seed=42)
    emb._vocab = vocab

    # Baseline on the engine benchmark (before distillation).
    pairs = build_benchmark()
    m0, acc0, _, _ = evaluate(emb, pairs)
    print(f"\n[before] neural: margin={m0:.4f} acc={acc0:.3f}")

    # Distill.
    print(f"\nDistilling from teacher ({args.epochs} epochs)...")
    emb = train_distill(emb, scored, epochs=args.epochs, lr=args.lr)

    # Evaluate on the engine benchmark.
    m1, acc1, rel1, unrel1 = evaluate(emb, pairs)
    print(f"\n[after] neural: margin={m1:.4f} acc={acc1:.3f} rel={rel1:.4f} unrel={unrel1:.4f}")
    print(f"  margin delta: {m1-m0:+.4f}  acc delta: {acc1-acc0:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, E=emb.E, W1=emb.W1, b1=emb.b1, W2=emb.W2, b2=emb.b2,
             vocab=np.array(sorted(vocab.items())))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
