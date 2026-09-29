"""T3.3 — Coverage map (stdlib, no extra deps).

Runs the test suite under ``trace`` to count which super_solver modules have at
least one line executed, and writes assets/coverage_report.json listing covered
vs never-executed modules. This is a coarse map (module-level, not line-level):
its job is to flag modules with ZERO test exercise.
"""
from __future__ import annotations

import json
import os
import sys
import trace

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "src")
TESTS = os.path.join(ROOT, "tests")


def main():
    # Run pytest under trace, collecting executed files.
    tracer = trace.Trace(count=1, trace=0, ignoremods=(), ignoredirs=[sys.prefix])

    import pytest

    def runtests():
        pytest.main(["-q", TESTS, "-p", "no:cacheprovider", "-x", "--no-header"])

    tracer.runfunc(runtests)
    results = tracer.results()
    counts = {}
    for (fname, _lineno), _n in results.counts.items():
        if "super_solver" in fname and fname.endswith(".py"):
            key = os.path.relpath(fname, SRC)
            counts[key] = counts.get(key, 0) + _n

    # enumerate all modules
    all_mods = []
    for dirpath, _dirs, files in os.walk(os.path.join(SRC, "super_solver")):
        for fn in files:
            if fn.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, fn), SRC)
                all_mods.append(rel)

    covered = sorted(m for m in all_mods if counts.get(m, 0) > 0)
    uncovered = sorted(m for m in all_mods if counts.get(m, 0) == 0)
    report = {
        "total_modules": len(all_mods),
        "covered_modules": len(covered),
        "uncovered_modules": uncovered,
        "module_line_hits": {m: counts[m] for m in covered},
    }
    path = os.path.join(ROOT, "assets", "coverage_report.json")
    json.dump(report, open(path, "w"), indent=2)
    print(f"covered {len(covered)}/{len(all_mods)} modules")
    print(f"uncovered ({len(uncovered)}): {uncovered}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
