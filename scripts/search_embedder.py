"""Use the engine itself to find a better embedder.

Builds a relatedness benchmark from the engine's OWN training corpus (the
CLASSICAL_CASES / RECENT_CASES / MATH_CODING_CASES / CROSS_DOMAIN_CASES), then
searches the embedder hyperparameter space to maximize the relatedness margin
(mean related sim - mean unrelated sim) and retrieval accuracy.

The search space:
  - polarity embedder hyperparameters (subword weight, bigram weight, polarity
    weight, stopword weight, subword size)
  - hybrid weighting (polarity vs ollama blend) when ollama is available

This is metric-learning-by-search: the engine's own corpus defines what
"related" means, and we tune the embedder to separate related from unrelated
pairs as cleanly as possible.

Usage:
  python scripts/search_embedder.py [--fast] [--hybrid]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.core.embeddings import PolarityAwareEmbeddingService
from super_solver.core.embedder import get_backend

# Import the training corpora to build the benchmark.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_problem_solving import CLASSICAL_CASES, RECENT_CASES, MATH_CODING_CASES
from test_cross_domain import CROSS_DOMAIN_CASES


def build_benchmark():
    """Return list of (problem, solution) related pairs from the engine's corpus."""
    pairs = []
    for case in CLASSICAL_CASES + RECENT_CASES + MATH_CODING_CASES:
        pairs.append((case["problem"], case["solution"]))
    for case in CROSS_DOMAIN_CASES:
        pairs.append((case["source_problem"], case["source_solution"]))
    return pairs


def evaluate(backend, pairs, top_k=1):
    """Evaluate an embedder backend on the relatedness benchmark.

    Returns (margin, retrieval_acc, mean_related, mean_unrelated).
    margin = mean(related sim) - mean(unrelated sim)
    retrieval_acc = fraction of problems whose correct solution ranks in top_k
    """
    # Encode all problems and solutions once.
    probs = [backend.encode(p) for p, _ in pairs]
    sols = [backend.encode(s) for _, s in pairs]

    n = len(pairs)
    related = []
    unrelated = []
    retrieval_hits = 0

    for i in range(n):
        # Related: own solution.
        rel = backend.cosine_similarity(probs[i], sols[i])
        related.append(rel)

        # Unrelated: all OTHER solutions (mean).
        others = [sols[j] for j in range(n) if j != i]
        if others:
            unrel = np.mean([backend.cosine_similarity(probs[i], o) for o in others])
            unrelated.append(float(unrel))

        # Retrieval: rank all solutions by sim to this problem.
        sims = [backend.cosine_similarity(probs[i], sols[j]) for j in range(n)]
        order = np.argsort(sims)[::-1]
        if i in order[:top_k]:
            retrieval_hits += 1

    mean_rel = float(np.mean(related))
    mean_unrel = float(np.mean(unrelated))
    margin = mean_rel - mean_unrel
    retrieval_acc = retrieval_hits / n
    return margin, retrieval_acc, mean_rel, mean_unrel


def make_polarity(**kw):
    """Build a PolarityAwareEmbeddingService with overridden hyperparameters."""
    svc = PolarityAwareEmbeddingService()
    for k, v in kw.items():
        setattr(svc, k, v)
    return svc


class BackendAdapter:
    """Wrap a raw encode function as an EmbedderBackend for evaluation."""

    def __init__(self, name, encode_fn, dim=384):
        self.name = name
        self._encode = encode_fn
        self.dim = dim

    def encode(self, text):
        return self._encode(text)

    def cosine_similarity(self, a, b):
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="Skip the slow full search, just report baseline + best found.")
    ap.add_argument("--hybrid", action="store_true", help="Also evaluate hybrid polarity+ollama blends.")
    ap.add_argument("--out", default="scratch/embedder_search.json")
    args = ap.parse_args()

    pairs = build_benchmark()
    print(f"Benchmark: {len(pairs)} related (problem, solution) pairs from engine corpus")

    results = []

    # --- Baseline: current polarity embedder ---
    base = get_backend("polarity")
    m, acc, rel, unrel = evaluate(base, pairs)
    print(f"\n[baseline] polarity (current): margin={m:.4f} acc={acc:.3f} rel={rel:.4f} unrel={unrel:.4f}")
    results.append({"name": "polarity_baseline", "margin": m, "acc": acc, "rel": rel, "unrel": unrel})

    # --- Search polarity hyperparameters ---
    # Vary: subword weight, bigram weight, polarity weight, stopword weight, subword size.
    search_space = {
        "subword_weight": [0.0, 0.2, 0.35, 0.5, 0.7],
        "bigram_weight": [0.0, 0.4, 0.8, 1.2],
        "polarity_weight": [0.0, 0.2, 0.4, 0.6],
        "stopword_weight": [0.0, 0.15, 0.3],
    }
    best = None
    best_score = -1e9
    combos = list(itertools.product(*search_space.values()))
    print(f"\nSearching {len(combos)} polarity hyperparameter combos...")
    t0 = time.time()
    for combo in combos:
        kw = dict(zip(search_space.keys(), combo))
        svc = make_polarity(**kw)
        backend = BackendAdapter("polarity_tuned", svc.encode)
        m, acc, rel, unrel = evaluate(backend, pairs)
        # Score: margin is the primary objective, retrieval acc is a tiebreak.
        score = m + acc
        if score > best_score:
            best_score = score
            best = {"name": "polarity_tuned", "params": kw, "margin": m, "acc": acc, "rel": rel, "unrel": unrel}
    print(f"  search took {time.time()-t0:.1f}s")
    print(f"[best] polarity_tuned: margin={best['margin']:.4f} acc={best['acc']:.3f} params={best['params']}")
    results.append(best)

    # --- Hybrid weighting (polarity + ollama) ---
    if args.hybrid:
        try:
            ollama = get_backend("ollama")
            # Probe ollama availability.
            _ = ollama.encode("probe")
            print("\nEvaluating hybrid polarity+ollama blends...")
            for pw in [0.0, 0.3, 0.5, 0.7, 1.0]:
                def make_hybrid(pw=pw):
                    def enc(text):
                        vp = base.encode(text)
                        vo = ollama.encode(text)
                        v = pw * vp + (1 - pw) * vo
                        n = np.linalg.norm(v)
                        return v / n if n > 0 else v
                    return BackendAdapter(f"hybrid_pw{pw}", enc, dim=384)
                backend = make_hybrid(pw)
                m, acc, rel, unrel = evaluate(backend, pairs)
                print(f"  hybrid pw={pw}: margin={m:.4f} acc={acc:.3f}")
                results.append({"name": f"hybrid_pw{pw}", "margin": m, "acc": acc, "rel": rel, "unrel": unrel})
        except Exception as e:
            print(f"  (ollama unavailable, skipping hybrid: {e})")

    # --- Report ---
    print("\n=== Summary ===")
    for r in sorted(results, key=lambda x: x["margin"] + x["acc"], reverse=True):
        print(f"  {r['name']:<22} margin={r['margin']:.4f} acc={r['acc']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
