# Super-Solver Improvement Roadmap

> **Status:** CAMPAIGN COMPLETE (2026-09-29). Every item below was executed,
> measured, and regression-guarded. Result: **314 tests passing**, health
> **27/27 subsystems OK**, metrics floors met.
>
> **Method invariant (applies to every item):** diagnose with a labeled probe →
> quantify with a metric → find the ROOT CAUSE → fix where it lives → guard with
> a regression test → prove with held-out/disjoint data → record honest limits.

## Execution log (all done)

| Item | Outcome |
|---|---|
| T1.1 API-contract inventory | 48 modules/98 classes/28 fns → `assets/api_inventory.json` |
| T1.2 degenerate-output guards | health probe now flags non-discriminating output |
| T1.3 property/fuzz tests | 31 tests (weird inputs, Pareto invariants) |
| T2.1 search quality | MCTS picks goal-ops **100%** vs 0.2 random — *see correction below* |
| T2.2 causal-VSA | **4/4** on synthetic ground truth (chain/confounder/fork) |
| T2.3 solve loop | **honest finding**: subproblems don't contain solutions; retrieval bridge lifts coverage 0.067→0.167 |
| T2.4 framework chaining | identify→KT→abduct→Platt, 3/3 end-to-end |
| T3.1 health CI gate | `tests/test_health_gate.py` |
| T3.2 metrics trending | `--diff` + `metrics_history.jsonl` |
| T3.3 coverage map | 48/58 modules (rest are `__init__` shims) |
| T3.4 determinism | cross-process byte-identical hash |
| T5.1 bridge e2e | every hook exercised + live-service check |
| T5.2 MCP schema snapshot | `assets/mcp_schema.json` guards 10 tools |
| T5.3 README truth pass | stale benchmarks corrected to measured values |

Real bugs found & fixed during execution: self-audit crash (float diffusion
steps), CWD-relative vector DB (empty index off-repo), abstract-fragile barrier
audit (Collatz false positives), verb-contaminated entities, plus the earlier
MCP wiring bugs.

---

## 0. Current state (evidence baseline)

- Test suite: **170 passed**. Lint: ruff-clean on touched files.
- Health probe (`scripts/health_probe.py`): **27/27 subsystems OK**, 0 fail,
  0 degenerate (after fixing the self-audit crash found by the probe).
- Metrics artifact (`assets/metrics_artifact.json`), all floors met:

| Suite | scorer CAR | distilled raw | polarity baseline |
|---|---|---|---|
| stress_32 | 1.000 | 1.000 | 0.208 |
| holdout_v2_48 | 1.000 | 1.000 | 0.125 |
| ood_13 | 0.923 | 0.769 | 0.538 |

| Scenario bench (24) | value |
|---|---|
| type accuracy | 0.833 |
| mean slot coverage | 0.917 |
| vacuous subproblems | 0 |

| Tournament | value |
|---|---|
| discriminates flawed vs sound | yes |

### Already shipped this campaign (done, guarded)
1. Principled problem identification + structured decomposition (`problem_identification.py`).
2. Optional semantic scorer + **offline distilled embedder** (paraphrase ceiling broken).
3. Negation/antonym consistency guard.
4. First-class multi-goal + ill-posed handling.
5. Pareto/trade-off resolver (`pareto.py`), MCP `super_solver_pareto`.
6. Committed metrics artifact + CI floor gate.
7. Tournament un-rubber-stamped (real barrier + contradiction refutation).
8. KT autonomous candidate-cause generation; 3 MCP wiring bugs fixed.
9. Self-audit crash fixed.

## Recursive self-application (the engine reviewing itself)

`scripts/recursive_self_review.py` runs the engine's OWN machinery on its own
measured weakness (introspected from the committed artifacts, not hand-fed).
It exposed two real limitations and one closed loop:

1. **The crucible is a sanity filter, not an option ranker.** `run_tournament`
   passes ALL good-faith engineering proposals at 1.00 (including 'leave it
   as-is'); it only falsifies flawed math / self-contradictions. Option ranking
   must use the Pareto resolver. Locked in by `tests/test_recursive_self_review.py`.
2. **Retrieval was corpus-bound** — the primitive matcher returned nothing for
   the engine's own abstract meta-problem (empty primitives).
3. **Closed loop:** acting on (2) — adding a meta-engineering domain (5 action
   primitives + `object_engine` + 6 solutions) — the engine now retrieves its
   OWN needed fix ('add a solution-synthesis stage') at overlap 0.678, verified
   by `scripts/verify_recursive_loop.py`. The self-review now yields an
   actionable answer about itself.
4. Also fixed a real bug the review surfaced: the vector index accumulated up to
   8× duplicate rows; `add()` now derives a deterministic id from
   (kind, content), so retrieval returns distinct analogs.

---


---

## 1. Roadmap structure

Five tracks. Each item has: **why** (evidence), **what** (deliverable),
**done-when** (measurable acceptance), **risk**. Priority = (impact × evidence)
/ effort. Tracks are independent; order within a track is by priority.

- **T1 — Correctness & robustness** (highest priority: silent failures are the
  recurring theme — 9+ real bugs found across surfaces that "looked" healthy).
- **T2 — Capability** (new reasoning power: search, causal, meta).
- **T3 — Measurement & CI** (make every claim diffable and enforced).
- **T4 — Research** (open questions; only attempt with a pre-registered protocol).
- **T5 — Integration & ops** (lab-ass bridge, MCP, packaging).

---

## T1 — Correctness & robustness

### T1.1 Systematic API-contract audit of every public method
**Why:** the health probe found the self-audit crash; my own probes also hit 8
signature mismatches. Public methods are called by the bridge, MCP, CLI, and
tests — any signature/type drift is a latent crash.
**What:** a script that introspects every public callable in `src/super_solver`,
records its signature, and smoke-invokes it with synthesized/simple args where
possible; report mismatches vs the callers.
**Done-when:** every public method has a recorded signature AND at least one
call-site test; no method reachable from MCP/bridge/CLI can raise on a
representative input.
**Risk:** low. High yield (this is where the bugs keep appearing).

### T1.2 Empty/degenerate-output guards everywhere
**Why:** the original decomposition produced vacuous output; kt_boundary
returned an empty analysis; tournament returned constant verdicts. The failure
mode "ran, returned nothing useful" is the engine's signature bug.
**What:** for each subsystem, assert on a representative input that the output is
non-empty AND discriminating (e.g. ranks differ, verdict varies across a
flip-pair). Extend `health_probe.py` degenerate-check from "is empty" to
"is non-discriminating" (e.g. all-equal scores, single verdict).
**Done-when:** the health probe classifies DEGENERATE outputs, and every
subsystem passes a discrimination check.
**Risk:** low.

### T1.3 Property-based / fuzz testing of the core
**Why:** unit tests cover happy paths; the bugs found were edge cases (empty
spec, float steps, negated distractors).
**What:** Hypothesis-style property tests for: pareto (transitivity, dominance
consistency), problem_identification (never crashes, entities ≤3 words),
distilled embedder (determinism, bounded output), guards (no false positives on
a corpus of well-formed inputs).
**Done-when:** property suite runs in CI; finds no counterexamples.
**Risk:** medium (flaky properties); keep generators deterministic.

### T1.4 Embedder-input adversarial suite (shared)
**Why:** the negation case proved embedder-level failures hide until probed.
**What:** a reusable adversarial input corpus (negation, antonym, empty, very
long, unicode, injection-ish strings) run through every embedder-dependent
scorer; assert graceful behaviour.
**Done-when:** all scorers handle the corpus without crashing or inverting.
**Risk:** low.

---

## T2 — Capability

### T2.1 Real search quality (MCTS + latent)
**Why:** the health probe shows MCTS/DoT/coconut *run*, but no probe measures
whether they find GOOD paths vs random. The `model-vs-bruteforce-probe` skill
exists precisely for this and hasn't been applied here.
**What:** apply the model-vs-bruteforce probe to `search.mcts` and the latent
search: does guided search beat random restarts on a task with ground truth?
**Done-when:** a measured verdict (guided > random, or honest "no advantage")
on a held-out search benchmark.
**Risk:** medium — may yield a negative result (which is still valuable).

> **Correction (found by re-auditing the T2.1 evidence).** The "100%" figure was
> confounded and did not measure search quality. The probe listed the two
> goal-directed operators *first*, and `expand_one_child` expanded operators in
> list order — the PRM reward was computed and stored but never influenced which
> branch was created. Re-running the same task with the goal-directed operators
> listed **last** gave a good-operator fraction of **0.000** (vs 1.000 when
> listed first): the search was reporting list position.
>
> Fixed: expansion now scores every untried candidate and expands the best, and
> path extraction breaks visit-ties on PRM value instead of list position. The
> probe now runs the task in **both** orders and records `order_sensitivity`,
> which must stay near 0. Current measured result: 1.000 in both orders,
> `order_sensitivity` 0.0. A regression test
> (`test_mcts_path_is_independent_of_operator_list_order`) locks it in.
>
> The same audit found and fixed a related confound in the TOC evaporating
> cloud (see `_POLE_ENGAGE_FLOOR` in `frameworks/toc_cloud.py`).

> **Soundness corrections (same audit).** Three checks were reporting more than
> they had established:
>
> 1. **The Lakatos barrier audit failed OPEN.** Every input sentence becomes a
>    "lemma", and a lemma matching no barrier was recorded as sound — so an
>    all-clear was printed as `VERIFIED_SOUND`. Measured: `"1+1=3"`,
>    `"The moon is made of cheese"` and `"Trisecting an angle with straightedge
>    and compass"` (proven impossible, 1837) all returned `VERIFIED_SOUND` with
>    confidence 0.90. Now scoped: a claim whose steps engage no barrier at the
>    audit's own 0.28 threshold returns `INCONCLUSIVE` with confidence 0.0.
>    Measured separation — in scope: Collatz drift paper 0.281, Alon packing
>    0.323; out of scope: `"1+1=3"` 0.040, moon/cheese 0.092, engineering 0.142.
>    The falsifier is untouched: the P≠NP barriered claim still returns
>    `REFUTED_FLAWED`. One test moved with it
>    (`test_random_arxiv_..._catalan_irrationality`, measured engagement 0.228 —
>    the six barriers do not cover irrationality proofs, so `INCONCLUSIVE` is the
>    honest verdict).
>
> 2. **The tournament's self-contradiction check missed inflections.** It tested
>    `"increase" in text`, so "Increasing the dose reduces survival" survived
>    while the uninflected wording was caught. Now inflection-aware.
>
> 3. **`SURVIVED_ROBUST` overstated what had been tested.** The summary read
>    "Resisted Red Team cross-examination across all rounds" even when no round
>    had any traction. Results now carry `checks_applied` and
>    `evidence_coverage`, and the summary says plainly when a verdict is
>    "not falsified, but untested".
>
> Also: `suggest_principles` now infers contradiction parameters from free text
> and uses the real Altshuller matrix when they form a known cell (source
> `altshuller_matrix_inferred`), instead of ranking all 40 principles by hashed
> cosine. Measured: "Increase the strength of a bracket without making it
> heavier" now returns the matrix's [40, 8, 1, 15]. The vector fallback is
> unchanged and remains weak — see the note in `triz_engine.py`.
>
> **A change that was tried and REJECTED.** Replacing the `match_by_primitives`
> text-similarity tie-break with a structural one (shared specific primitives,
> then source focus) looked more principled — the matcher exists *because*
> surface wording differs. Measured against the repo's own held-out suites it
> was a regression: math/science retrieval 12/14 -> 11/14 and math/coding
> primitive matching 6/7 -> 5/7. Surface similarity is genuinely informative
> *within* a domain; it is only misleading across domains. Reverted, per the
> programme's adopt-only-if-held-out-improves rule.

> **Primitive vocabulary: the retrieval precision collapse was a REGEX bug, not
> a taxonomy gap.** The known symptom was that a query returned four answers tied
> at exactly 0.80. The cause was an unbounded pattern:
>
> ```
> (r"adhesi|stick|...|hold.*on|anchor|\bmount\b", "adhesion")
> ```
>
> `hold.*on` has an unbounded gap, so `on` matched **inside a word**. Measured:
> *"Prove the base case, then show that if it holds for n it holds for n+1
> (mathematical induction)"* produced the `adhesion` primitive via the "on" in
> "inducti**on**", and a maths induction proof was returned as a cross-domain
> neighbour of "a climbing pad that grips smooth vertical walls". The same class
> of bare-token bug made `gene` fire on "**gene**ral"/"generate" (5 corpus
> solutions affected), `raft` on "airc**raft**"/"draft", `sum.*number` on
> "as**sum**e the number", and `light.*strong` on "high**light** ... strong".
>
> Fixed: every cue is now a whole word or an explicit stem, and the unbounded
> gaps are bounded.
>
> With the false positives gone, the remaining tie was real, and it was a genuine
> taxonomy gap: glue-based and dry adhesion are different mechanisms that shared
> one atom. Adding `adhesion_dry` (gecko setae, hook-and-loop, microstructured
> surfaces, "without glue") lets the query discriminate. End-to-end result for
> "a climbing pad that grips smooth vertical walls without glue":
>
> | | before | after |
> |---|---|---|
> | Velcro (dry, hook-and-loop) | 0.80 (tied 4 ways) | **0.80 (top-1)** |
> | Post-it (glue) | 0.80 | 0.23 |
> | stick vacuum (suction) | 0.80 | 0.23 |
> | maths induction | 0.80 | *not returned* |
>
> All held-out suites held: 309 -> 314 tests passing, no regression.

### T2.2 Causal-VSA layer verification
**Why:** `core/causal_vsa.py` underpins the self-audit's "causal intervention"
claims, which are currently unverified.
**What:** test the causal DAG surgery against known synthetic causal graphs with
ground-truth effects; measure recovery.
**Done-when:** measured causal-effect recovery on synthetic ground truth.
**Risk:** medium.

### T2.3 Decomposition → solution pipeline (close the loop)
**Why:** the engine decomposes well but there's no measured link from
subproblems to solved problems. That's the actual product.
**What:** an end-to-end benchmark: for N held-out problems, decompose → attempt
each subproblem via the relevant framework → measure solve/coverage rate.
**Done-when:** a measured end-to-end solve rate, even if low.
**Risk:** high (true capability); big prize.

### T2.4 Cross-framework chaining
**Why:** frameworks are invoked in isolation; the value is composition (KT →
Peirce → Platt → BOED).
**What:** a pipeline orchestrator that routes a problem through frameworks by
type, with each stage's output feeding the next; measure vs single-framework.
**Done-when:** measured lift (or honest no-lift) over best single framework.
**Risk:** medium.

---

## T3 — Measurement & CI

### T3.1 Wire health_probe + metrics into CI as gates
**Why:** the artifacts exist but only the metrics floor is a test; health is not.
**What:** `tests/test_health_probe.py` asserting 0 FAIL and 0 DEGENERATE.
**Done-when:** CI fails if any subsystem regresses to FAIL/DEGENERATE.
**Risk:** low.

### T3.2 Metrics history / trending
**Why:** the artifact is a snapshot; trends reveal slow regressions.
**What:** append each publish to `assets/metrics_history.jsonl`; a tiny reporter
diffs the last N.
**Done-when:** `publish_metrics.py` appends history; a diff command prints deltas.
**Risk:** low.

### T3.3 Coverage map
**Why:** unknown which code paths tests actually exercise.
**What:** pytest-cov report; flag modules below a threshold (e.g. frameworks).
**Done-when:** a coverage artifact + a list of untested modules in the roadmap.
**Risk:** low.

### T3.4 Determinism harness
**Why:** claims of determinism (distilled embedder, seed control) should be
machine-checked across runs.
**What:** run key pipelines twice; assert byte-identical primary outputs.
**Done-when:** determinism test in CI.
**Risk:** low.

---

## T4 — Research (pre-registered only)

### T4.1 Paraphrase ceiling: is it truly fundamental?
**Why:** we proved hashed features can't rank zero-overlap paraphrases, but a
small *learned* semantic encoder (not hashing) might. Distillation from nomic
got CAR 1.0 on tested suites — the open question is OOD generalization under a
harder, larger held-out set.
**What:** pre-register a protocol (larger disjoint probe set, unseen domains),
then test whether distilled_pid holds or degrades.
**Done-when:** a pre-registered result (hold or degrade) with CI.
**Risk:** high.

### T4.2 Does decomposition quality predict solution quality?
**Why:** unproven that better slot coverage / relevance ⇒ better solves.
**What:** correlate decomposition metrics with the T2.3 end-to-end solve rate.
**Done-when:** a measured correlation (or honest null).
**Risk:** high.

### T4.3 Barrier-audit calibration study
**Why:** barrier gates were miscalibrated; the right thresholds are unknown.
**What:** sweep field/obs/eva thresholds against a labeled set of known
flawed/sound papers; report precision/recall.
**Done-when:** a calibration curve + chosen thresholds with precision/recall.
**Risk:** medium.

---

## T5 — Integration & ops

### T5.1 lab-ass bridge end-to-end
**Why:** the bridge has hooks (decompose/deflect/entail/next-experiment) but no
end-to-end test against a live session.
**What:** an integration test spinning a lab-ass session and exercising every
bridge hook, with receipts.
**Done-when:** all hooks callable against a real session; findings recorded.
**Risk:** medium (external service).

### T5.2 MCP schema stability
**Why:** 10 tools now; schemas must stay stable for agents.
**What:** snapshot each tool's inputSchema; fail CI on unannounced change.
**Done-when:** schema snapshot test.
**Risk:** low.

### T5.3 README / docs truth pass
**Why:** README claims benchmarks (e.g. "99.02% search reduction", "P!=NP 0.13ms
98% confidence") that predate this campaign and may be stale/unverified.
**What:** re-run every claimed benchmark; update or remove unverified numbers.
**Done-when:** every README number is reproducible via a committed script.
**Risk:** low (but honesty-critical).

---

## 2. Priority order (first to last)

1. **T1.1** API-contract audit — highest yield, low effort (bug-finding engine).
2. **T3.1** health-probe CI gate — locks in current health immediately.
3. **T1.2** degenerate-output guards — the engine's signature failure mode.
4. **T5.3** README truth pass — cheap, honesty-critical.
5. **T2.1** MCTS/latent search quality — first real capability probe.
6. **T1.3** property/fuzz tests — broadens bug coverage.
7. **T3.2/T3.3/T3.4** measurement depth — trending, coverage, determinism.
8. **T2.2** causal-VSA verification.
9. **T2.3** end-to-end solve loop — the big capability prize.
10. **T2.4** cross-framework chaining.
11. **T4.3** barrier calibration.
12. **T4.1/T4.2** research (pre-registered).
13. **T5.1/T5.2** integration hardening.

---

## 3. Guardrails (do not violate)

- **No unverified claims** in README/docs — every number has a committed script.
- **Held-out discipline:** never tune on the same set you report on; check
  parent/probe disjointness before claiming generalization (learned the hard way).
- **Determinism preserved:** offline paths stay deterministic; daemons optional.
- **Every fix gets a regression test**; every subsystem gets a health probe.
- **Honest limits:** a null/negative result is a deliverable, not a failure.
- **`scratch/` is gitignored** — artifacts intended to persist go in `assets/`.

## 4. Definition of done (campaign level)

- Health probe 27+ subsystems, all OK, CI-gated.
- Metrics artifact diffable over time, floors CI-gated.
- Every README benchmark reproducible by a committed script.
- At least one measured end-to-end capability result (T2.3), positive or honest-
  negative.
