"""Refined finding: WHERE the retrieval bottleneck actually lives.

P6 concluded "the hashed embedder family is bounded." This refines it by
measuring the PRETRAINED teacher (nomic-embed-text) on the same retrieval corpus:

  - mean sim(problem, own solution)   = 0.658
  - mean sim(problem, random solution) = 0.424   -> margin +0.235 (signal EXISTS)
  - mean RANK of the correct solution over 197 = ~194; recall@20 = 0.01

So the encoder is NOT the sole bottleneck: even the strong pretrained encoder
cannot rank the correct solution out of ~196 DISTINCT competing mechanisms that
share vocabulary. The bottleneck is:
  (a) corpus discriminability — 196 distinct mechanisms, heavy lexical overlap; and
  (b) metric harshness — top-1 over a 197-way near-duplicate pool.

Actionable path: retrieval must be a CANDIDATE GENERATOR + a discriminative
reranker (or a cross-encoder that reads problem AND candidate together), not a
bi-encoder top-1. A bi-encoder alone cannot separate this pool.

Writes assets/retrieval_bottleneck_finding.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    from train_embedder_retrieval import build_corpus

    from super_solver.core.embedder import get_backend

    c = build_corpus()
    be = get_backend("ollama")
    P = [be.encode(p) for p, _ in c]
    S = [be.encode(s) for _, s in c]

    ranks = []
    own, rnd = [], []
    rng = np.random.default_rng(0)
    for i, (_, sol) in enumerate(c):
        sims = np.array([be.cosine_similarity(P[i], S[j]) for j in range(len(c))])
        own.append(float(sims[i]))
        rnd.append(float(sims[int(rng.integers(len(c)))]))
        sims[i] = -1e9
        order = np.argsort(sims)[::-1]
        r = next((k for k, idx in enumerate(order) if c[idx][1] == sol), len(order))
        ranks.append(r)
    ranks = np.array(ranks)

    out = {
        "corpus_size": len(c),
        "distinct_solutions": len({s for _, s in c}),
        "teacher": "nomic-embed-text (ollama)",
        "mean_sim_own": round(float(np.mean(own)), 3),
        "mean_sim_random": round(float(np.mean(rnd)), 3),
        "margin": round(float(np.mean(own) - np.mean(rnd)), 3),
        "mean_rank_of_correct": int(ranks.mean()),
        "recall_at_1": round(float((ranks < 1).mean()), 3),
        "recall_at_20": round(float((ranks < 20).mean()), 3),
        "bottleneck": "corpus discriminability + top-1-over-197 harshness, NOT "
                      "solely the encoder",
        "actionable_path": "bi-encoder as candidate generator + discriminative "
                           "reranker (or cross-encoder reading problem and candidate "
                           "together); a bi-encoder alone cannot separate this pool",
    }
    path = os.path.join(ROOT, "assets", "retrieval_bottleneck_finding.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
