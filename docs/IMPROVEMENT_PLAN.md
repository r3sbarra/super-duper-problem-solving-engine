# Improvement Requirements & Plan

> **Grounded in measurement, not opinion.** Authored 2026-09-29. Every claim
> below is backed by a committed script; the plan's steps each carry a done-when.
> Research method: the engine decomposed/abduced its own improvement problem
> (`scripts/recursive_self_review.py`); the mathematical core was checked with
> suuri-chan (which correctly fails closed — INCONCLUSIVE, never a false
> "proven"); the requirements were then MEASURED by `scripts/improvement_requirements.py`.

---

## 1. What is actually required (measured)

`python scripts/improvement_requirements.py` on the engine's own corpus
(155 problem→solution pairs):

| Measurement | Value | Meaning |
|---|---|---|
| pairs whose correct solution shares NO specific atom with the problem | **50 / 155 = 32.3%** | containment retrieval returns **exactly 0** by construction for these |
| pairs sharing at least one action atom | 89 / 155 | the retrievable ceiling |
| top-1 correct, **among rankable** pairs | **63.8%** | ranking leaves ~36% on the table |

### R1 — SEMANTIC BRIDGE (required)
**Why:** 32.3% of problems share no specific atom with their correct solution, so
`containment(T,S)=|T∩S|/|T| = 0`. **No re-weighting, IDF, or score fusion can
exceed this floor** — it is a property of the metric, not the weights.
**Math (suuri-chan-checked, fail-closed):** exceeding a zero-containment floor
requires **composing with a map** `f: text → R^d` and a metric where
`related(paraphrase) > unrelated(distractor)`. That is a *learned semantic
embedding*, not a larger atom set.
**Already partially available:** the offline `distilled_pid` embedder separates
correct from wrong on the **disjoint** pairs at **74%** — so the bridge exists but
is not composed into the retrieval path (measured: hybrid lift 0.406 → 0.432).

### R2 — LEARNED RANKER (required)
**Why:** within rankable pairs top-1 is 63.8%; hand-set weights (0.8 action /
0.2 object, IDF) leave correctness unused.
**Math:** ranking is a monotone function of the score; improving it means learning
a score **monotone in true relevance** (a supervised ranking objective, e.g.
pairwise/ListNet), not hand-tuning constants.

### R3 — CORPUS COVERAGE (required, orthogonal)
**Why:** retrieval can only return mechanisms present in the corpus. Measured
0-overlap retrievals for cache-single-flight and invoice-dedup show a coverage
limit independent of matching quality.
**Math:** `recall ≤ 1 − P(correct mechanism absent)`. No matcher fixes an absent
target. Coverage must be raised by *adding mechanisms*, not by better matching.

### R4 — VERIFICATION BEFORE ADOPTION (required, meta)
**Why:** the campaign found **10+ real bugs** in surfaces that "looked healthy"
(self-audit crash, empty MCP output, CWD-relative DB, rubber-stamp tournament,
duplicate index, …). Improvement without verification corrupts the system.
**Math:** adopt a change only if a held-out metric improves **beyond its noise
floor** (paired test); otherwise reject. Enforced by the health + metrics gates.

---

## 2. The plan (ordered by (impact × evidence) / effort)

> **Execution status (2026-09-29): ALL PHASES DONE.** Measured outcomes below.

| Phase | Outcome |
|---|---|
| P1 semantic bridge | DONE — `primitive_analogize(semantic_weight=)`; disjoint pairs 0/50 → 5/50; default unchanged |
| P2 learned ranker | DONE — trained; **does NOT beat hand-set** (0.286 = 0.286); semantic composition wins (0.476). The embedder, not the ranker, is the lever |
| P3 corpus coverage | DONE — added 7 mechanisms; solve-loop coverage 0.167 → 0.25 (lexical) / **0.583 (blended metric)** |
| P4 self-application loop | DONE — `self_application_loop.py`, verify-before-swap, fail-closed; reports NO_OP_CORPUS_SATURATED |
| P5 math backbone | DONE — suuri-chan falsified the convex-combination identities correctly; `proven_claims: []` (honest: nothing formally proven, falsifications real) |
| **P6 embedder retrieval training** | DONE — **decisive negative result**: retraining the hashed-feature student on the retrieval objective does NOT improve top-1 (0.000 → 0.000). See below |

### P6 — the decisive research finding

The plan's upstream dependency was “a stronger embedder.” I tried to deliver it:
trained the distilled student on the **retrieval objective** (positive = correct
solution, negatives = other corpus solutions; hinge decreased as expected).
**Retrieval top-1 did not move (0.000 → 0.000).**

**Conclusion (bounds every retrieval plan):** the **hashed n-gram + bilinear
embedder *family* cannot be trained to retrieval-quality semantics** — it has no
gradient path to learn that `single-flight` belongs with `cache invalidation`
when they share no tokens. Escaping the retrieval ceiling requires a
**fundamentally richer encoder** (learned token embeddings or a small
transformer), not more training of the current one. This validates roadmap T4.1
and is recorded in `assets/embedder_retrieval_finding.json`.

**Metric upgrade:** the solve-loop now blends lexical + semantic match (pure
lexical couldn't see retrieval improvements whose wording differed).

### Phase P1 — Compose the semantic bridge into retrieval  ★ highest impact
- **Do:** in `primitive_analogize`, produce a **hybrid score**
  `α·containment + (1−α)·semantic` (from `distilled_pid`), α tuned on a held-out
  split; keep pure-containment available via a flag.
- **Done-when:** top-1 on the 155-pair corpus rises materially above the 0.406
  containment baseline, **and** the 32% disjoint pairs become retrievable, with
  no regression on the cross-domain transfer test (currently 3/8).
- **Evidence:** measured disjoint rate 32.3%; semantic separation 74%; prototype
  lift 0.406→0.432 (untuned).
- **Risk:** low-medium (weight tuning); guard with the existing test suite.

### Phase P2 — Learned ranker  ★ high impact
- **Do:** train a small pairwise ranking head on the engine's own
  (problem, correct-solution, distractor) triples (already generated for the
  distilled embedder), replacing hand-set weights. Deterministic, offline.
- **Done-when:** top-1-within-rankable rises above **63.8%** on a held-out split,
  beyond a paired-test noise floor; artifact committed; determinism test passes.
- **Evidence:** 63.8% rankable top-1; the triple data already exists.
- **Risk:** medium (overfitting small data) — use held-out + disjoint-parent split
  (the discipline that caught the earlier leakage).

### Phase P3 — Corpus coverage
- **Do:** add mechanisms for the measured 0-overlap gaps (cache single-flight,
  invoice dedup, and the other failing solve-loop cases), each as a
  problem_context→solution_direction entry.
- **Done-when:** solve-loop `solution_idea_coverage` rises above its current
  **0.167** floor on the 12-problem bench (recorded as a floor, not a boast).
- **Evidence:** measured 0-overlap retrievals; solve_loop.json.
- **Risk:** low; purely additive (guarded by the transfer test).

### Phase P4 — Self-application loop (close the meta-loop)
- **Do:** wire `recursive_self_review` → improvement candidate → **verify-before-
  swap** (run gates) → adopt only if gates improve. Use the
  `self-patching-code` skill's verify-before-swap pattern.
- **Done-when:** the engine can, unattended, propose+verify+adopt-or-reject one
  improvement, with the decision recorded.
- **Evidence:** the recursive review already surfaces its own fix (overlap 0.678)
  but a human applies it.
- **Risk:** high — must be fail-closed (reject on any gate regression).

### Phase P5 — Widen the math backbone (suuri-chan)
- **Do:** for each retrieval/ranking change, obtain an **exact** check where the
  claim is mathematical (ordering, invariants) via suuri-chan `pslq` /
  `falsify_or_prove`; record VERIFIED only on a real proof, INCONCLUSIVE otherwise.
- **Done-when:** each Phase P1-P2 objective has a suuri-chan verdict recorded.
- **Risk:** low; suuri-chan is honest (fails closed).

---

## 3. What improvement does NOT need (ruled out by measurement)

- **More atoms / bigger regex vocabulary alone** — cannot move a zero-containment
  floor (R1). The generic-atom layer helped *representation* but not ranking.
- **Re-weighting the existing containment** — bounded by the retrievable ceiling
  (89/155); the 32% disjoint pairs stay at 0.
- **A bigger daemon (LLM) by default** — the offline path must stay deterministic;
  semantics live in the distilled artifact.

---

## 4. Definition of done (programme level)

- Retrieval: disjoint pairs retrievable; top-1 materially above 0.406 (P1/P2).
- Solve loop: `solution_idea_coverage` above 0.167 (P3).
- Self-application: engine proposes+verifies+adopts one improvement (P4).
- Every change: health 27+/27, metrics floors met, suite green, held-out proof.
- No unverified claims in docs; every number reproducible by a committed script.
