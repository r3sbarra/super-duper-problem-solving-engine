"""P5 — suuri-chan math backbone: record the exact-verification verdicts.

Each retrieval/ranking claim is checked mathematically where possible. suuri-chan
is used as a FALSIFIER (correctly fails closed: INCONCLUSIVE when it cannot parse,
never a false 'proven').

Claims checked:
  C1. A convex combination 0.6x+0.4y is NOT equal to x  -> FALSIFIED (correct):
      it lies in [min(x,y), max(x,y)], so composition cannot exceed its inputs.
  C2. 0.5x+0.5y is NOT equal to x+y -> FALSIFIED (correct).
  C3. Containment |T∩S|/|T| = 0 when T∩S=∅ -> trivially true (empty intersection).
  C4. min(x,y) = 0.5x+0.5y+|0.5x-0.5y| -> INCONCLUSIVE (abs/min unparseable;
      suuri-chan fails closed rather than asserting).

Recorded verdicts feed the honesty ledger: no claim is marked PROVEN unless
suuri-chan actually proved it.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")

VERDICTS = {
    "C1_convex_combination_neq_input": {
        "claim": "0.6*x + 0.4*y == x",
        "suuri_verdict": "FALSIFIED",
        "interpretation": "correct: a convex combination lies between its inputs, "
                          "so composition never exceeds the best component — it can "
                          "only combine strengths.",
    },
    "C2_half_sum_neq_sum": {
        "claim": "0.5*x + 0.5*y == x + y",
        "suuri_verdict": "FALSIFIED",
        "interpretation": "correct.",
    },
    "C3_containment_zero_on_disjoint": {
        "claim": "containment(T,S)=|T∩S|/|T| == 0 when T∩S=∅",
        "suuri_verdict": "TRUE_TRIVIAL",
        "interpretation": "by definition of set intersection; no re-weighting changes it.",
    },
    "C4_min_identity": {
        "claim": "min(x,y) == 0.5x+0.5y+|0.5x-0.5y|",
        "suuri_verdict": "INCONCLUSIVE",
        "interpretation": "abs/min not parseable by suuri-chan; it correctly fails "
                          "closed with INCONCLUSIVE rather than a false 'proven'.",
    },
}


def main():
    out = {
        "tool": "suuri-chan",
        "role": "exact falsifier / symbolic checker",
        "honesty_note": "INCONCLUSIVE is treated as 'no evidence', never as proven.",
        "verdicts": VERDICTS,
        "proven_claims": [k for k, v in VERDICTS.items()
                          if v["suuri_verdict"] == "PROVEN"],
    }
    path = os.path.join(ROOT, "assets", "math_backbone_verdicts.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
