"""Reproduce the offline distilled paraphrase-relevance embedder.

End-to-end: build training data (teacher = nomic-embed-text), train the offline
student, evaluate CAR on the held-out probe set, and write the artifact to
assets/distilled_pid_large.npz.

Requires a local Ollama with nomic-embed-text and a generation model. Determinism:
training uses a fixed seed; the ARTIFACT it writes is stable given the same
training data (data generation via the LLM is not bit-reproducible, which is why
the generated training set is committed as assets/distilled_pid_train.json).

Usage:
  python scripts/build_distilled_pid_embedder.py --train   # regenerate data+model
  python scripts/build_distilled_pid_embedder.py --verify  # load artifact, assert CAR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scratch"))

ROOT = os.path.join(os.path.dirname(__file__), "..")
ARTIFACT = os.path.join(ROOT, "assets", "distilled_pid_large.npz")
TRAIN_JSON = os.path.join(ROOT, "assets", "distilled_pid_train.json")

PROBES = [
    ("Shortest-path routing for a delivery fleet under time-window constraints.",
     "Model the road network as a weighted graph and relax edges",
     "Paint the delivery van blue"),
    ("How do we keep the service responding when one backend becomes slow?",
     "Add a circuit breaker that fails fast on the slow dependency",
     "Repaint the office walls"),
    ("Determine the maximum flow between source and sink in a capacity network.",
     "Find augmenting paths and push residual capacity along them",
     "Choose a name for the project"),
    ("Why does the cache invalidation cause p99 latency spikes?",
     "Trace the thundering-herd of simultaneous misses on expiry",
     "Increase the font size of the dashboard"),
    ("Prove that the sum of the first n odd numbers equals n squared.",
     "Use induction, pairing each new odd number with a growing square",
     "Book a flight to Berlin"),
    ("Design a scheme so a message is processed exactly once despite retries.",
     "Attach a unique id and deduplicate on the receiver side",
     "Water the office plants"),
    ("Estimate how long a queue takes to drain after a traffic spike.",
     "Model arrival and service rates and integrate over the backlog",
     "Order more printer paper"),
    ("Find the cheapest way to connect all offices with links so no cycles form.",
     "Grow a minimum spanning tree by repeatedly taking the cheapest safe edge",
     "Hire a new accountant"),
]


def car(embedder) -> float:
    wins = 0
    for parent, rel, irr in PROBES:
        r = embedder.cosine_similarity(embedder.encode(parent), embedder.encode(rel))
        i = embedder.cosine_similarity(embedder.encode(parent), embedder.encode(irr))
        wins += int(r > i)
    return wins / len(PROBES)


def do_train():
    from distill_pid_student_fast import build_xy, train_fast
    from distill_pid_student import Student

    if not os.path.exists(TRAIN_JSON):
        print(f"missing {TRAIN_JSON}; run the data builder first")
        sys.exit(1)
    records = json.load(open(TRAIN_JSON))
    s = Student(dim=512, seed=42)
    F, pairs = build_xy(records, 512)
    train_fast(s.W, F, pairs, epochs=1500, lr=0.5)
    np.savez_compressed(ARTIFACT, W=s.W, dim=512)
    print(f"wrote {ARTIFACT}")


def do_verify():
    from super_solver.core.distilled_pid import DistilledPIDEmbedder

    e = DistilledPIDEmbedder(weights_path=ARTIFACT)
    assert e.available, "artifact not loadable"
    c = car(e)
    print(f"CAR = {c:.3f} ({int(c*len(PROBES))}/{len(PROBES)})")
    assert c >= 0.875, f"regression: CAR {c} below floor 0.875"
    print("OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.train:
        do_train()
    if a.verify or not a.train:
        do_verify()
