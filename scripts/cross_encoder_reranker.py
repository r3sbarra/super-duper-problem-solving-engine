"""Cross-encoder reranker: read (problem, candidate) TOGETHER and score.

The refined finding (assets/retrieval_bottleneck_finding.json) showed a bi-encoder
top-1 cannot separate the 197-way pool: even the pretrained teacher ranks the
correct solution ~194th. The actionable path is a CROSS-ENCODER that scores a
(problem, candidate) PAIR jointly, using features a bi-encoder cannot:

  - joint distinctive-token overlap (IDF-weighted),
  - structural-atom containment (specific primitives),
  - semantic cosine,
  - candidate brevity/specificity (generic long solutions rank low).

This is a candidate GENERATOR + reranker: take the top-N from any recall path,
rerank with the joint score, measure recall@k. Deterministic, offline.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), "..")

STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "by", "is", "are", "that", "this", "it", "its", "as", "at", "from",
        "so", "only", "one", "side", "over", "into", "without", "not", "add",
        "ensure", "use", "make", "model", "system"}


def _toks(t):
    import re

    return [w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 3]


def _idf(corpus_texts):
    from collections import Counter

    df = Counter()
    for t in corpus_texts:
        for w in set(_toks(t)):
            df[w] += 1
    n = len(corpus_texts)
    return {w: np.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}


class CrossEncoderReranker:
    def __init__(self, corpus_texts):
        from super_solver.core.distilled_pid import DistilledPIDEmbedder
        from super_solver.vectorize.primitives import strip_to_primitives

        self.idf = _idf(corpus_texts)
        self.emb = DistilledPIDEmbedder()
        self._strip = strip_to_primitives

    def score(self, problem: str, candidate: str) -> float:
        pt = set(_toks(problem))
        ct = set(_toks(candidate))
        # 1. IDF-weighted joint overlap (rare shared words matter most)
        shared = pt & ct
        if pt:
            idf_shared = sum(self.idf.get(w, 1.0) for w in shared)
            idf_total = sum(self.idf.get(w, 1.0) for w in pt)
            jt = idf_shared / idf_total if idf_total else 0.0
        else:
            jt = 0.0
        # 2. structural atom containment (specific primitives only)
        def spec(p):
            return {x for x in p if not (x.startswith("act_") or x.startswith("attr_"))}

        tp = spec(self._strip(problem))
        cp = spec(self._strip(candidate))
        atom = len(tp & cp) / len(tp) if tp else 0.0
        # 3. semantic
        sem = 0.0
        if self.emb.available:
            raw = self.emb.cosine_similarity(self.emb.encode(problem), self.emb.encode(candidate))
            sem = max(0.0, min(1.0, (raw - 0.2) / 0.5))
        # 4. specificity: penalize generic long candidates
        spec_pen = 1.0 / (1.0 + max(0, len(ct) - 8) * 0.08)
        return 0.45 * jt + 0.25 * atom + 0.20 * sem + 0.10 * spec_pen


def main():
    from train_embedder_retrieval import build_corpus

    corpus = build_corpus()
    texts = [p for p, _ in corpus] + [s for _, s in corpus]
    rr = CrossEncoderReranker(texts)

    # candidate generator: retrieve all (recall path), rerank jointly
    ranks = []
    for i, (p, sol) in enumerate(corpus):
        scores = [(rr.score(p, c), j) for j, (_, c) in enumerate(corpus) if j != i]
        scores.sort(reverse=True)
        r = next((k for k, (_, j) in enumerate(scores) if corpus[j][1] == sol), len(scores))
        ranks.append(r)
    ranks = np.array(ranks)

    out = {
        "n": len(corpus),
        "cross_encoder_recall_at_1": round(float((ranks < 1).mean()), 3),
        "cross_encoder_recall_at_5": round(float((ranks < 5).mean()), 3),
        "cross_encoder_recall_at_20": round(float((ranks < 20).mean()), 3),
        "mean_rank": int(ranks.mean()),
        "bi_encoder_mean_rank_baseline": 194,
    }
    path = os.path.join(ROOT, "assets", "cross_encoder_reranker.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
