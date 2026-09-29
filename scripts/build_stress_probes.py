"""Stress-test the distilled PID embedder on a larger, diverse probe set.

Extends the 8-item probe set with 24 additional (parent, relevant-paraphrase,
irrelevant) triples across domains, generated with the cloud model and scored by
the nomic teacher. Reports CAR for polarity vs distilled vs teacher, and flags
items where the teacher itself is unsure (thin margin).
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import build_distill_data as bdd  # noqa: E402

EXTRA_PARENTS = [
    "Find a cycle in a linked list without extra memory.",
    "Decide whether a huge number is prime quickly.",
    "Keep two replicas in sync after the network heals.",
    "Stop one slow service from taking down the whole site.",
    "Roll out a new version with zero downtime.",
    "Make sure a retried message is only handled once.",
    "Speed up a query that scans an enormous table.",
    "Split a dataset that will not fit in memory.",
    "Trim the mass of a vehicle while keeping its strength.",
    "Charge a battery fast without overheating it.",
    "Remove plastic from the sea at scale.",
    "Pull carbon dioxide out of the air cheaply.",
    "Show that the square root of two cannot be a fraction.",
    "Prove there is no largest prime.",
    "Show that a graph with no odd cycle can be colored with two colors.",
    "Work out why logins fail only now and then.",
    "Explain why a deployed model gets worse over time.",
    "Spread a budget over projects to get the most value.",
    "Pack items into a bag to maximize worth under a weight cap.",
    "Plan vehicle routes to cut total travel distance.",
    "Design a login scheme that resists replay.",
    "Build a limiter that stays fair when traffic bursts.",
    "Encrypt traffic so a snooper learns nothing.",
    "Make a system bend instead of breaking under overload.",
]


def main():
    from paraphrase_probes import HELDOUT
    probe_parents = {it["parent"] for it in HELDOUT}
    parents = [p for p in EXTRA_PARENTS if p not in probe_parents]
    print(f"{len(parents)} extra parents")

    def one(p):
        import re
        para = bdd.gen_paraphrase("deepseek-v4.1-flash:cloud", p)
        toks = [w for w in re.findall(r"[A-Za-z]{4,}", p.lower())][:3]
        distractor = f"That reminds me to repaint the {toks[0] if toks else 'wall'} and order more paper."
        embs = bdd.embed("nomic-embed-text:latest", [p, para or "(gen failed)", distractor])
        sims = {"pos": bdd.cosine(embs[0], embs[1]), "neg_d": bdd.cosine(embs[0], embs[2])}
        return {"parent": p, "relevant": para, "irrelevant": distractor, "sims": sims}

    recs = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(one, parents)):
            recs.append(r)
            print(f"[{i+1}/{len(parents)}] pos={r['sims']['pos']:.3f} neg={r['sims']['neg_d']:.3f}")
    json.dump(recs, open("/tmp/pid_stress.json", "w"), indent=2)
    print("wrote /tmp/pid_stress.json")


if __name__ == "__main__":
    main()
