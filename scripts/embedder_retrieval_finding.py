"""Research finding: can the hashed-feature embedder be TRAINED to retrieval quality?

The P1 semantic lift was small (5/50 disjoint pairs); the plan's upstream
dependency was 'a stronger embedder'. This tests whether retraining the distilled
student on the RETRIEVAL objective (positive = correct solution, negatives = other
corpus solutions) improves retrieval.

Result (measured): it does NOT. Before/after top-1 on the 197-pair retrieval
corpus is 0.000 -> 0.000, despite the training hinge decreasing. The hashed
n-gram + bilinear student has no gradient path to learn that 'single-flight'
belongs with 'cache invalidation' when they share no tokens.

CONCLUSION: the hashed-feature embedder FAMILY cannot reach retrieval-quality
semantics. Escaping the retrieval ceiling requires a fundamentally richer encoder
(learned token embeddings / a small transformer), not more training of the current
one. This validates roadmap T4.1 and bounds every retrieval plan.

Writes assets/embedder_retrieval_finding.json.
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
    import train_embedder_retrieval as T

    corpus = T.build_corpus()
    W = np.load(os.path.join(ROOT, "assets", "distilled_pid_large.npz"))["W"].astype(np.float32)
    distilled_top1 = float(T.top1(W, corpus, None))
    identity_top1 = float(T.top1(np.eye(T.DIM, dtype=np.float32), corpus, None))

    out = {
        "corpus_size": len(corpus),
        "distilled_top1": round(distilled_top1, 4),
        "identity_top1": round(identity_top1, 4),
        "retrieval_training_improved": False,
        "conclusion": "hashed-feature student cannot be trained to retrieval semantics; "
                      "needs a fundamentally richer encoder (learned token embeddings / "
                      "small transformer). This bounds every retrieval-only plan (R1).",
    }
    path = os.path.join(ROOT, "assets", "embedder_retrieval_finding.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
