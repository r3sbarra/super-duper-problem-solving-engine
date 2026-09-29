# Super-Solver Improvement Roadmap

> **Status:** living document. Authored 2026-09-29 after a multi-hour hardening
> campaign. Replaces step-by-step improvisation with a mapped program.
> **Method invariant (applies to every item):** diagnose with a labeled probe →
> quantify with a metric → find the ROOT CAUSE → fix where it lives → guard with
> a regression test → prove with held-out/disjoint data → record honest limits.

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
