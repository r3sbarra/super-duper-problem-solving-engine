#!/usr/bin/env python3
"""Test that the EXPANDED domain-knowledge fact base (ops/networking/security/
data) retrieves correct concrete solutions for NOVEL rephrased problems.

Method (honest test, mirrors scripts/test_analogical_transfer.py):
1. Seed the vector index with the expanded DOMAIN_KNOWLEDGE fact base.
2. Query with a NOVEL, rephrased formulation of each problem — wording the
   solver has NOT seen (different words, same underlying problem).
3. Check whether analogical transfer retrieves the CORRECT solution as the
   top hit. This tests genuine semantic relatedness, not string matching.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.vectorize import VectorizationService
from super_solver.vectorize.domain_knowledge import (
    DOMAIN_KNOWLEDGE,
    build_domain_knowledge_index,
    retrieve_concrete_solutions,
)
from super_solver.vectorize.primitives import match_by_primitives

# Novel rephrased problem -> expected solution keyword (substring that must appear)
NOVEL_CASES = [
    {
        "title": "Leader failover",
        "query": "My primary database server just died; how do I keep the app running?",
        "expect": "replica",
    },
    {
        "title": "Consensus",
        "query": "Several machines must all settle on the same transaction order even if some are slow",
        "expect": "consensus",
    },
    {
        "title": "Exactly-once",
        "query": "A message got sent twice over a flaky link; I only want it processed once",
        "expect": "deduplicat",
    },
    {
        "title": "Load balancing",
        "query": "Too many users are hitting one web server and it is slowing down",
        "expect": "load balancer",
    },
    {
        "title": "Circuit breaker",
        "query": "The payment API is timing out and dragging down my whole checkout page",
        "expect": "circuit breaker",
    },
    {
        "title": "Backoff retry",
        "query": "My client keeps retrying instantly and making the flaky service worse",
        "expect": "backoff",
    },
    {
        "title": "Auth",
        "query": "Strangers are getting into my admin panel and changing things",
        "expect": "authenticat",
    },
    {
        "title": "Encryption",
        "query": "Someone could sniff the traffic between my app and the server",
        "expect": "encrypt",
    },
    {
        "title": "Backup recovery",
        "query": "I lost my data after a crash and need to get it back fast",
        "expect": "backup",
    },
    {
        "title": "Graceful degradation",
        "query": "One microservice is broken but I do not want the whole site to go down",
        "expect": "degrad",
    },
    {
        "title": "Indexing",
        "query": "My search query crawls the whole table and takes forever",
        "expect": "index",
    },
    {
        "title": "Checkpointing",
        "query": "My batch job restarts from scratch every time it gets killed halfway",
        "expect": "checkpoint",
    },
    {
        "title": "Canary deploy",
        "query": "I want to ship a new version without risking a full outage if it is buggy",
        "expect": "canary",
    },
    {
        "title": "Root cause tracing",
        "query": "The bug only happens sometimes and I cannot figure out where it comes from",
        "expect": "tracing",
    },
]


def main():
    print("=" * 78)
    print("EXPANDED domain-knowledge analogical-transfer test (novel formulations)")
    print("=" * 78)
    print(f"fact base entries: {len(DOMAIN_KNOWLEDGE)}")

    service = VectorizationService()
    build_domain_knowledge_index(service)

    # Build the source-solution list once for primitive matching.
    source_solutions = [
        {"title": row[2] or "", "solution": row[3]}
        for row in service.index._rows("solution")
    ]

    correct = 0
    for case in NOVEL_CASES:
        # Path 1: hybrid analogical transfer (vector).
        hits = retrieve_concrete_solutions(service, case["query"], top_k=3)
        top = hits[0]["content"] if hits else ""
        ok = case["expect"].lower() in top.lower()
        path = "vector"
        # Path 2: primitive matching (structural) — often bridges surface gaps.
        if not ok:
            prim_hits = match_by_primitives(case["query"], source_solutions, top_k=3)
            if prim_hits:
                prim_top = prim_hits[0]["content"]
                if case["expect"].lower() in prim_top.lower():
                    top = prim_top
                    ok = True
                    path = "primitive"
        correct += 1 if ok else 0
        mark = "✓" if ok else "✗"
        print(f"  {mark} {case['title']:<22} [{path:<9}] expect='{case['expect']}' -> {top[:70]}")

    print(f"\nRESULT: {correct}/{len(NOVEL_CASES)} correct")
    return 0 if correct == len(NOVEL_CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
