"""Parse the TRIZ contradiction matrix into problem->solution pairs.

The classic Altshuller 39x39 contradiction matrix maps each (improving
parameter, worsening parameter) pair to recommended Inventive Principles.
Each cell is a structured problem->solution pair:
  problem: "How to improve X without worsening Y"
  solution: the recommended inventive principle(s)

This produces a large, structured, cross-domain corpus of engineering
problem->solution pairs (up to ~1300 cells) to feed the embedder.

Usage:
  python scripts/build_triz_corpus.py --xls /tmp/triz_matrix.xls
                                      --out scratch/triz_corpus.json
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd


def parse_matrix(xls_path):
    df = pd.read_excel(xls_path, header=None)
    # Parameters: header row 0, columns 2..40 hold the 39 parameter names
    # (column 1 is an empty spacer). Parameter N lives at column N+1.
    params = {}
    for c in range(2, 41):
        name = df.iloc[0, c]
        if pd.notna(name) and str(name).strip():
            params[c - 1] = str(name).strip()
    # Principles: column 43, rows 2..41.
    principles = {}
    for r in range(2, 42):
        name = df.iloc[r, 43]
        if pd.notna(name) and str(name).strip() and not str(name).startswith("Inventive"):
            num = int(str(name).split(".")[0])
            principles[num] = str(name).strip()
    return df, params, principles


def build_corpus(xls_path):
    df, params, principles = parse_matrix(xls_path)
    entries = []
    # Rows 2..40 = improving parameter (39 params), cols 1..39 = worsening.
    for r in range(2, 41):
        improving = df.iloc[r, 0]
        if pd.notna(improving):
            imp_num = int(float(str(improving).split(".")[0]))
            imp_name = params.get(imp_num, f"parameter {imp_num}")
            for c in range(1, 40):
                cell = df.iloc[r, c]
                if pd.notna(cell) and str(cell).strip() and str(cell).strip() != "+":
                    nums = [int(x) for x in re.findall(r"\d+", str(cell))]
                    if not nums:
                        continue
                    wname = params.get(c, f"parameter {c}")
                    prin_names = [principles.get(n, f"principle {n}") for n in nums]
                    problem = f"How to improve {imp_name} without worsening {wname}"
                    solution = "Apply the inventive principle(s): " + "; ".join(prin_names)
                    entries.append({
                        "problem": problem,
                        "solution": solution,
                        "improving": imp_name,
                        "worsening": wname,
                        "principles": nums,
                    })
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xls", default="/tmp/triz_matrix.xls")
    ap.add_argument("--out", default="scratch/triz_corpus.json")
    args = ap.parse_args()

    entries = build_corpus(args.xls)
    print(f"Parsed {len(entries)} contradiction->principle pairs")
    # Dedupe by (problem, solution).
    seen = set()
    uniq = []
    for e in entries:
        key = (e["problem"], e["solution"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    print(f"Unique: {len(uniq)}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(uniq, f, indent=2)
    print(f"Wrote {args.out}")
    # Sample.
    for e in uniq[:3]:
        print(f"  {e['problem']} -> {e['solution'][:60]}")


if __name__ == "__main__":
    main()
