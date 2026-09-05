"""Build a large cross-domain problem->solution corpus from multiple sources.

Sources:
  1. TRIZ contradiction matrix (1212 pairs) - engineering contradictions.
  2. AskNature biomimicry dataset (2037 entries) - nature-inspired solutions.

Each entry becomes a {problem, solution} pair suitable for feeding the
embedder / domain-knowledge index.

Usage:
  python scripts/build_large_corpus.py --out scratch/large_corpus.json
"""

from __future__ import annotations

import argparse
import json
import os
import re


def load_triz(path):
    with open(path) as f:
        return json.load(f)


def load_asknature(path):
    with open(path) as f:
        return json.load(f)


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def build_asknature_entries(data):
    entries = []
    for e in data:
        app = clean(e.get("Application", ""))
        strat = clean(e.get("Strategy", ""))
        src = clean(e.get("Source", ""))
        fn = clean(e.get("Function1", ""))
        if not app or not strat:
            continue
        # Problem: the human design challenge. Solution: nature's strategy.
        problem = f"How to {app}"
        solution = f"Nature-inspired solution from {src}: {strat}"
        entries.append({
            "problem": problem,
            "solution": solution,
            "source": "asknature",
            "organism": src,
            "function": fn,
        })
    return entries


def build_leetcode_entries(data):
    """LeetCode problems: description is the problem, topics+approach is the solution."""
    entries = []
    questions = data.get("questions", []) if isinstance(data, dict) else data
    for q in questions:
        title = clean(q.get("title", ""))
        desc = clean(q.get("description", ""))
        topics = q.get("topics", []) or []
        sol = clean(q.get("solution", ""))
        if not title or not desc:
            continue
        # Problem: title + description. Solution: topics + first approach.
        problem = f"{title}: {desc}"
        # Extract the first approach heading from the solution article.
        approach = ""
        if sol:
            m = re.search(r"Approach \d+[^\n]*\n+(.*?)(?=\n###|\Z)", sol, re.S)
            if m:
                approach = clean(m.group(1))[:300]
        solution = f"Algorithm: {', '.join(topics)}. " + (approach or "Use the standard algorithm for this problem type.")
        entries.append({
            "problem": problem,
            "solution": solution,
            "source": "leetcode",
            "title": title,
            "difficulty": clean(q.get("difficulty", "")),
            "topics": topics,
        })
    return entries


def build_codeforces_entries(df):
    """Codeforces problems: statement is the problem, tags are the solution approach."""
    entries = []
    for _, row in df.iterrows():
        stmt = clean(row.get("problem_statement", ""))
        tags = row.get("tags", "")
        if isinstance(tags, str):
            tags = tags.strip("[]").replace("'", "").split(", ")
        tags = [t.strip() for t in tags if t.strip()]
        if not stmt:
            continue
        problem = stmt[:500]
        solution = f"Algorithm: {', '.join(tags)}. " + "Solve with the standard technique for this problem type."
        entries.append({
            "problem": problem,
            "solution": solution,
            "source": "codeforces",
            "tags": tags,
        })
    return entries


def build_scp_entries(df):
    """SCP-116K science problems: problem statement + worked solution."""
    entries = []
    for _, row in df.iterrows():
        prob = clean(row.get("problem", ""))
        sol = clean(row.get("extract_solution", "")) or clean(row.get("r1_response", ""))
        dom = clean(row.get("domain", ""))
        if not prob or not sol:
            continue
        entries.append({
            "problem": prob[:600],
            "solution": sol[:600],
            "source": "scp",
            "domain": dom,
        })
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triz", default="scratch/triz_corpus.json")
    ap.add_argument("--asknature", default="/tmp/asknature.json")
    ap.add_argument("--leetcode", default="/tmp/leetcode.json")
    ap.add_argument("--codeforces", default="/tmp/codeforces.parquet")
    ap.add_argument("--scp", default="/tmp/scp_0000.parquet")
    ap.add_argument("--out", default="scratch/large_corpus.json")
    args = ap.parse_args()

    all_entries = []

    # TRIZ.
    if os.path.exists(args.triz):
        triz = load_triz(args.triz)
        for e in triz:
            all_entries.append({
                "problem": e["problem"],
                "solution": e["solution"],
                "source": "triz",
                "improving": e.get("improving", ""),
                "worsening": e.get("worsening", ""),
            })
        print(f"TRIZ: {len(triz)} entries")

    # AskNature.
    if os.path.exists(args.asknature):
        an = load_asknature(args.asknature)
        an_entries = build_asknature_entries(an)
        all_entries.extend(an_entries)
        print(f"AskNature: {len(an_entries)} entries")

    # LeetCode.
    if os.path.exists(args.leetcode):
        lc = load_asknature(args.leetcode)
        lc_entries = build_leetcode_entries(lc)
        all_entries.extend(lc_entries)
        print(f"LeetCode: {len(lc_entries)} entries")

    # Codeforces.
    if os.path.exists(args.codeforces):
        import pandas as pd
        cf = pd.read_parquet(args.codeforces)
        cf_entries = build_codeforces_entries(cf)
        all_entries.extend(cf_entries)
        print(f"Codeforces: {len(cf_entries)} entries")

    # SCP-116K.
    if os.path.exists(args.scp):
        import pandas as pd
        scp = pd.read_parquet(args.scp)
        scp_entries = build_scp_entries(scp)
        all_entries.extend(scp_entries)
        print(f"SCP-116K: {len(scp_entries)} entries")

    # Dedupe by (problem, solution).
    seen = set()
    uniq = []
    for e in all_entries:
        key = (e["problem"], e["solution"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    print(f"Total unique: {len(uniq)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(uniq, f, indent=2)
    print(f"Wrote {args.out}")

    # Sample.
    for e in uniq[:3]:
        print(f"  [{e['source']}] {e['problem'][:60]} -> {e['solution'][:60]}")


if __name__ == "__main__":
    main()
