"""Out-of-distribution (OOD) probe suite for the distilled PID embedder.

Tests generalization along three shifts the training set under-covers:
  1. DOMAIN shift  — humanities, biology, law, everyday reasoning.
  2. LENGTH shift  — very short and very long parents.
  3. ADVERSARIAL negation — a distractor that shares a token AND is
     semantically opposite (highest-confusion case).

Each item scores the embedder's correct-above-irrelevant rate (CAR). Teacher
(nomic) scores are included where available to separate "student fails" from
"teacher also fails".
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

# domain-shifted probes (humanities / bio / law / everyday)
DOMAIN_OOD = [
    ("Explain why the Roman Republic collapsed despite its military strength.",
     "Trace the concentration of land and army loyalty away from civic institutions",
     "Buy a new set of curtains for the forum"),
    ("Why do two closely related bird species stop interbreeding?",
     "Accumulate isolating barriers such as differing songs and timing",
     "Order a new bird feeder online"),
    ("Interpret whether a late rent payment waives the landlord's right to evict.",
     "Apply the doctrine that accepting late payment can modify the lease terms",
     "Paint the apartment ceiling green"),
    ("Why does bread dough rise more slowly in winter?",
     "Slow the yeast's metabolic rate as ambient temperature drops",
     "Replace the kitchen wallpaper"),
    ("Explain how a rumor spreads faster than a fact.",
     "Model transmission as a function of emotional salience and novelty",
     "Sharpen the pencils on the desk"),
    ("Why did the market overreact to a minor earnings miss?",
     "Amplify the move through leveraged positioning and stop-loss cascades",
     "Rearrange the office chairs"),
]

# length-shift probes (very short, very long)
LENGTH_OOD = [
    ("Cache stampede?", "Coordinate misses so only one worker recomputes an expired entry",
     "Pick a font"),
    ("Lose weight?", "Sustain a caloric deficit through portion control and activity",
     "Buy a hat"),
    ("Prove a limit exists.",
     "Show the sequence is monotone and bounded so it must converge",
     "Feed the cat"),
    ("Design, for a hospital that must keep operating through regional power failures "
     "and network outages lasting days, a system that preserves patient records, "
     "coordinates staff, and never loses a critical update.",
     "Replicate the record store across sites with an offline-first conflict-free merge",
     "Repaint the ambulance"),
]

# adversarial negation probes: distractor shares a token but is negated/opposite
NEGATION_OOD = [
    ("Reduce the latency of the read path.", "Add a cache in front of the database",
     "Do not add any cache and instead increase the latency of the read path"),
    ("Prove the theorem holds for all n.",
     "Use induction with a verified base case",
     "Disprove that the theorem holds for all n"),
    ("Make the structure lighter without weakening it.",
     "Replace solid members with trusses that carry the same load",
     "Make the structure heavier and weaker"),
]


def all_probes():
    items = []
    for group, data in [("domain", DOMAIN_OOD), ("length", LENGTH_OOD), ("negation", NEGATION_OOD)]:
        for parent, rel, irr in data:
            items.append({"group": group, "parent": parent, "relevant": rel, "irrelevant": irr})
    return items


def car(items, embed):
    wins = 0
    rows = []
    for it in items:
        r = embed.cosine_similarity(embed.encode(it["parent"]), embed.encode(it["relevant"]))
        i = embed.cosine_similarity(embed.encode(it["parent"]), embed.encode(it["irrelevant"]))
        ok = r > i
        wins += int(ok)
        rows.append((it, ok, r, i))
    return wins / len(items), rows


def main():
    from super_solver.core.distilled_pid import DistilledPIDEmbedder
    from super_solver.core.embedder import get_backend
    from super_solver.frameworks.problem_identification import relevance_score

    items = all_probes()
    distilled = DistilledPIDEmbedder()
    polarity = get_backend("polarity")
    assert distilled.available

    def scorer(embed):
        def s(parent, child):
            return embed.cosine_similarity(embed.encode(parent), embed.encode(child))
        return s

    print(f"{len(items)} OOD probes\n")
    # Raw embedder cosines (no guard) for reference.
    for name, emb in [("polarity(raw)", polarity), ("distilled(raw)", distilled)]:
        c, rows = car(items, emb)
        print(f"{name:16} OOD CAR = {c:.3f} ({int(c*len(items))}/{len(items)})")

    # Full scorer (semantic + negation guard).
    class _ScorerAdapter:
        def __init__(self, embed):
            self.dim = getattr(embed, "dim", 0)
            self._embed = embed
        def encode(self, t):
            return self._embed.encode(t)
        def cosine_similarity(self, a, b):
            return self._embed.cosine_similarity(a, b)

    class _RelevanceAsEmbedder:
        """Wraps relevance_score so car() can use it."""
        def __init__(self):
            self.dim = 1
        def encode(self, t):
            return t
        def cosine_similarity(self, a, b):
            if isinstance(a, str) and isinstance(b, str):
                return relevance_score(a, b)[0]
            return 0.0

    rs = _RelevanceAsEmbedder()
    c, rows = car(items, rs)
    print(f"{'scorer+guard':16} OOD CAR = {c:.3f} ({int(c*len(items))}/{len(items)})")

    for group in ["domain", "length", "negation"]:
        gi = [it for it in items if it["group"] == group]
        c, _ = car(gi, rs)
        print(f"  scorer[{group:8}] CAR = {c:.3f} ({int(c*len(gi))}/{len(gi)})")


if __name__ == "__main__":
    main()
