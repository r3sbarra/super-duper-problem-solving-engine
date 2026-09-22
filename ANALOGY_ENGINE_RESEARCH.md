# Super-Solver Analogy Engine — Falsifiable Improvement Research

## 1. What `analogize()` retrieves, and by what similarity

There are **three distinct analogize surfaces**, not one:

| Surface | Entry point | What it does | Status |
|---|---|---|---|
| MCP `super_solver_analogize` | `mcp/server.py:165` | calls `engine.gentner.retrieve_analogs()` | **Broken** |
| lab-ass `analogize_problem` | `lab_ass_bridge/hooks.py:351` | `svc.analogize(vec)` → `VectorIndex.analogize` (`vectorize/index.py:165`) → `search(kind="problem")` | Noisy |
| `primitive_analogize` | `vectorize/__init__.py:149` | `match_by_primitives` over `kind="solution"` (`primitives.py:200`) | Works cross-domain |

**Analogical vector path** (`analogize`): encodes the query with `ProblemVectorizer`, then does an O(n) cosine scan over **only `kind="problem"` rows** via `VectorIndex.search()` (`index.py:104-131`), `min_similarity` defaults to `0.0` so no low-sim results are ever rejected. It returns the top problem rows and their linked solution/path via `_link()` (`index.py:253`), which resolves `solution_id`/`path_id` metadata links.

**Broken MCP surface (verified)**: `engine.gentner.retrieve_analogs` is called at `mcp/server.py:165`, but `StructureMappingEngine.gentner_sme.py` defines only `align_systems`. `retrieve_analogs` does not exist → confirmed `AttributeError` at runtime. The documented "VSA/SME cross-domain" tool is dead code.

## 2. Concrete failure mode + code-level lever

**Failure** (verified empirically against the live 738-row DB — 368 problems / 367 solutions, dominated by coding/DB/material tasks): a number-theory query "characterize the distribution of prime gaps" via `analogize` returned a **"reduce latency in a distributed system"** problem at cosine `0.312`; "prove infinitely many primes" returned only `0.347`. The problem-kind-only cosine has no cross-domain structure and no similarity floor, so in a sparse/foreign-coverage domain the top hits are pure surface noise.

**The lever is not "seed more"** — a structural fix already exists on the *solution* side: `primitive_analogize` strips to action+object-type atoms with IDF weighting (`primitives.py:200-283`) and matched the same two math queries at overlap **1.0** (Sieve-of-Eratosthenes, GCD/Euclid) using only entities already in the index. So the failure is a **routing/wiring defect**: the weakest path (`svc.analogize`, problem-only) is the one the lab-ass bridge calls, while the working structural path is unexposed and unconnected.

Concrete code levers, in order of impact:
- Point `analogize_problem` (`hooks.py:351`) at the structural path, OR lower `solution_weight` route via `hybrid_analogize`.
- Add a similarity floor (e.g. drop hits below a tuned threshold) so `analogize` doesn't return guaranteed-noise rows.
- Expose and wire `primitive_analogize` into the MCP tool as the real cross-domain surface.

## 3. Proposed improvement + falsifiable test

**Improvement:** Route lab-ass/MCP analogical transfer through the structural `primitive_analogize` path (falling back to vector search only when primitive overlap is zero), and delete/replace the dead `retrieve_analogs` call. This reuses existing, proven machinery — no new model, no re-seeding.

**Falsifiable test (must pass before and after):**
For each of N=10 math queries (primality, prime gaps, GCD, factoring) run both `svc.analogize` and `svc.primitive_analogize` on the current DB and assert:
1. **H1 (mechanism):** `primitive_analogize` returns ≥3 hits with `primitive_overlap ≥ 0.5` for math queries, and returns **no** non-math (DB/pipeline) hits in the top-3. Current `analogize` fails this (returns DB-latency at 0.312).
2. **H0 (regression, null):** swapping the bridge to `primitive_analogize` does not reduce top-k overlap for the existing coding/DB queries below their current `analogize` scores (guard against over-correction).
3. **Fix the dead surface:** `hasattr(engine.gentner, "retrieve_analogs")` becomes `True` (or the MCP tool is rewired), and calling it does not raise `AttributeError`.

Success = all three pass on the untouched 738-row index. This is falsifiable: keeping the current wiring fails test 1; a naive re-route that tanks coding recall fails test 2.

*Blocker reported honestly:* I did not modify code. The three surfaces being inconsistent (MCP→SME-dead, bridge→vector, CLI→primitives) means "the fix" is a wiring decision, not a single-line swap; the report scopes the exact target call sites.
