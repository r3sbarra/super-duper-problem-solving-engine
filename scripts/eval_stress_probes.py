"""Score polarity vs distilled vs teacher CAR on the combined stress probe set.

Uses the 8 held-out probes + 24 extra stress probes (all parents disjoint from
the distilled training set). Reports CAR and margin, and shows where teacher and
student disagree.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from paraphrase_probes import HELDOUT  # noqa: E402
from super_solver.core.distilled_pid import DistilledPIDEmbedder  # noqa: E402
from super_solver.core.embedder import get_backend  # noqa: E402


def load_probes():
    items = []
    for it in HELDOUT:
        items.append((it["parent"], it["relevant"], it["irrelevant"], None, None))
    stress = json.load(open("/tmp/pid_stress.json"))
    for r in stress:
        items.append((r["parent"], r["relevant"], r["irrelevant"],
                      r["sims"]["pos"], r["sims"]["neg_d"]))
    return items


def car_for(name, embed):
    items = load_probes()
    wins = 0
    rows = []
    for parent, rel, irr, tp, tn in items:
        r = embed.cosine_similarity(embed.encode(parent), embed.encode(rel))
        i = embed.cosine_similarity(embed.encode(parent), embed.encode(irr))
        ok = r > i
        wins += int(ok)
        rows.append((ok, r, i, parent[:42], tp, tn))
    return wins / len(items), rows


def main():
    items = load_probes()
    print(f"{len(items)} probes\n")
    polarity = get_backend("polarity")
    distilled = DistilledPIDEmbedder()
    assert distilled.available

    pc, prows = car_for("polarity", polarity)
    dc, drows = car_for("distilled", distilled)
    print(f"polarity  CAR = {pc:.3f} ({int(pc*len(items))}/{len(items)})")
    print(f"distilled CAR = {dc:.3f} ({int(dc*len(items))}/{len(items)})")

    # Teacher CAR on the items where teacher scores exist (stress set).
    tw = 0
    tn = 0
    for it in items:
        if it[4] is not None:
            tn += 1
            tw += int(it[4] > it[5])
    print(f"teacher   CAR = {tw/tn:.3f} ({tw}/{tn})  [on stress subset only]")

    print("\nDistilled misses:")
    for ok, r, i, p, tp, tn_ in drows:
        if not ok:
            print(f"  XX rel={r:.3f} irr={i:.3f}  {p}   (teacher pos={tp} neg={tn_})")


if __name__ == "__main__":
    main()
