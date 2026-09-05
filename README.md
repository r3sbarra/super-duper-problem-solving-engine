# Super-Duper-Problem-Solving-Engine

> **A Neurosymbolic Reasoning & Scientific Inquiry Engine**  
> Unifying 8 historically proven scientific and cognitive problem-solving paradigms with continuous vector representations (latent guidance, VSA/HDC, potential-field manifolds, DPLL SAT, and SMT theories).

[![Tests](https://img.shields.io/badge/tests-80%20passed-brightgreen.svg)]()

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-purple.svg)](LICENSE)

---

## 1. Overview

`super-duper-problem-solving-engine` is designed to overcome the core failure modes of standard autoregressive Large Language Models (LLMs) when applied to scientific research:
- **Premature Commitment:** Linear word-by-word generation traps models in flawed reasoning paths without backtracking.
- **Confirmation Bias:** Models seek supporting evidence rather than designing crucial experiments to falsify hypotheses.
- **Dead-End Loops:** Re-exploring known failed approaches due to lack of geometric repulsion memory.
- **Superficial Analogies:** Matching lexical keywords rather than deep relational causal graphs.

The engine serves as an intelligent, zero-daemon reasoning core for discovery platforms such as **lab-ass**.

```
+----------------------------------------------------------------------------------------------------+
|                                 PROVEN PROBLEM-SOLVING TAXONOMY                                    |
+------------------------------------+-----------------------------------+---------------------------+
| SCIENTIFIC INQUIRY                 | INVENTIVE & SYSTEMS ENGINEERING   | COGNITIVE & HEURISTIC     |
+------------------------------------+-----------------------------------+---------------------------+
| 1. Platt's Strong Inference (1964) | 4. Altshuller's TRIZ (40 Ops)     | 6. Polya's Heuristics     |
|    - Alternative Hypotheses Tree   |    - 39x39 Contradiction Matrix   |    - Working Backwards    |
|    - Crucial Exclusory Experiments |    - Trade-off Resolution Tensors |    - Auxiliary Reduction  |
|                                    |                                   |                           |
| 2. Peircean Discovery Cycle        | 5. Goldratt's Theory of           | 7. Gentner's Structure-   |
|    - Abduction -> Deduction ->     |    Constraints (TOC)              |    Mapping Theory (SME)   |
|      Induction                     |    - Bottleneck Identification    |    - Relational Analogy   |
|                                    |    - Evaporating Cloud Conflicts  |    - VSA Tensor Bindings  |
| 3. Kepner-Tregoe IS / IS NOT       |                                   |                           |
|    - 4D Boundary Separation        |                                   |                           |
|    - Failure Envelope Hyperplane   |                                   |                           |
+------------------------------------+-----------------------------------+---------------------------+
                                     |
           +-------------------------v-------------------------+
           |       CONTINUOUS VECTOR REASONING SUBSTRATE       |
           |  - Coconut: Latent Continuous Thoughts (BFS)      |
           |  - DoT: Thought Diffusion with CFG Guidance       |
           |  - JEPA: Representation-Space Latent World Model  |
           |  - VSA / HDC: Algebraic Binding & Cleanup Memory  |
           |  - Negative Manifolds: Geometric Dead-End Evasion |
           |  - DPLL: Pure-Python Refutation Logic Solver      |
           |  - BOED: Bayesian Optimal Experimental Design     |
           |  - Synaptic Consolidation: Clustered Memory Prune |
           +---------------------------------------------------+
```

---

## 2. 👶 ELI5: How Does It Actually Work? (Explain Like I'm 5)

Imagine you are trying to find a treasure buried in a giant, foggy maze. 

A standard AI is like someone who **sprints forward talking non-stop**. If it hits a brick wall, it doesn't back up—it just invents a fairy tale about how the wall is actually a door, gets stuck, and repeats itself over and over.

Here is how **this engine** solves hard problems like a genius detective:

### 1. 🔍 Step 1: Drawing the Crime Scene (IS vs. IS-NOT)
Before jumping to wild conclusions, the engine draws a chalk line:
- *"The light switch only sparks in the bedroom (IS), not in the kitchen (IS-NOT)."*
- *"It only happens on Tuesdays (IS), not on Saturdays (IS-NOT)."*
By separating what **IS** happening from what **IS NOT**, it immediately throws away 90% of useless guesses. *(Kepner-Tregoe)*

### 2. 🧲 Step 2: Quicksand Repulsors (Negative Manifold)
When an experiment fails or an idea is disproven, the engine plants a **glowing red magnetic beacon** right at that failure spot. 
Whenever future ideas float through its brain, that beacon acts like reverse gravity, **physically pushing thoughts away from the quicksand** so it never makes the same mistake twice.

### 3. 🎯 Step 3: The 20-Questions Knife (Platt's Strong Inference & BOED)
Most people try to prove their favorite pet idea right. Real scientists try to **eliminate** wrong ideas as fast as possible.
The engine looks at all competing ideas and asks:
> *"What single test will slice this suspect list in half no matter what the answer is?"*
It picks the most diagnostic test with maximum information gain, cutting down 1,000 possibilities to 1 in just 10 steps.

### 4. 💡 Step 4: Smashing "Impossible" Trade-Offs (TRIZ & Null-Space)
Ever faced a dilemma like:
- *"I want my rocket to be super strong, but also super light!"*
- *"I want fast medical testing, but 100% accuracy!"*
Most people compromise and settle for mediocre. The engine uses **40 Universal Inventive Principles** and math tricks (Null-Space Projection) to reveal hidden assumptions and dissolve the contradiction entirely.

### 5. 🧠 Step 5: Thinking in Shapes, Not Words (Coconut & Thought Diffusion)
Instead of guessing one word at a time, the engine shapes ideas like sculpting clay in continuous 384-dimensional space. It pulls thoughts closer to the goal like a magnet while steering clear of dead ends.

### 6. ⚖️ Step 6: The Formal Consistency Checker (DPLL SAT & SMT Solvers)
Before accepting a deduction trajectory, the engine feeds its logic into a formal propositional SAT solver (DPLL) and theory solver (SMT EUF / QF-LRA). If the premises do not strictly entail the target claim under resolution refutation, a countermodel is returned.

### 7. 😴 Step 7: Sleeping & Cleaning Up (Synaptic Memory Consolidation)
Just like your brain organizes memories while you sleep, when the engine accumulates lots of similar dead ends, it merges them into a single clean summary rule so its memory stays razor-sharp and never slows down.

### 8. 🪞 Step 8: Self-Auditing & Optimization
The engine can point its diagnostic tools at its own internal parameters. It audits parameter sensitivity via causal DAG surgery, verifies invariant constraints, and tunes itself to operate cleanly.

---

## 3. The 8 Proven Frameworks Vectorized

### 1. John R. Platt’s "Strong Inference" (*Science*, 1964)
- **Principle:** Progress is exponential when maintaining competing hypotheses and designing crucial experiments to **exclude/falsify** hypotheses.
- **Vector Formulation:** Hypotheses $\mathcal{H} \subset \mathbb{S}^{d-1}$ are partitioned; crucial experiments are ranked by expected Shannon information gain $\Delta I(\mathbf{E})$, pruning MCTS branches. Supports both strict Popperian falsification and noisy-channel Bayesian updates.

### 2. Charles Sanders Peirce’s Triadic Inquiry
- **Principle:** The 3-stage scientific discovery loop: *Abduction* (creative hypothesis from surprising anomaly) $\to$ *Deduction* (necessary testable predictions) $\to$ *Induction* (empirical measurement).
- **Vector Formulation:** Anomaly vector $\mathbf{a} = \mathbf{x}_{\text{obs}} - \hat{\mathbf{x}}_{\text{expected}} \xrightarrow{\text{abduct}} \mathbf{h} \xrightarrow{\mathbf{M}_{\text{deduct}}} \hat{\mathbf{y}} \xrightarrow{\text{induct}} \mathcal{L}_{\text{loss}}$.

### 3. Kepner-Tregoe (KT) 4D Boundary Analysis
- **Principle:** Isolates root causes by explicitly contrasting what the problem IS vs what it IS NOT across 4 dimensions: **Identity, Location, Timing, and Extent**.
- **Vector Formulation:** Dual centroids $(\mathbf{c}_{\text{IS}}, \mathbf{c}_{\text{IS\_NOT}})$ with a high-margin separating hyperplane and per-dimension diagnostic breakdown.

### 4. Altshuller’s TRIZ (40 Inventive Principles & Contradiction Matrix)
- **Principle:** Eliminates engineering and scientific contradictions without compromise using 40 universal transformation operators.
- **Vector Formulation:** Includes Altshuller's canonical 39x39 Contradiction Matrix (`CONTRADICTION_MATRIX`) and anti-symmetric contradiction tensors steering state vectors in latent space.

### 5. Goldratt’s Theory of Constraints (TOC) & "Evaporating Cloud"
- **Principle:** Focuses on the primary bottleneck constraint and resolves conflicts between requirements by surfacing and invalidating hidden implicit assumptions.
- **Vector Formulation:** Evaluates candidate injections using harmonic requirement balance, avoiding polarized collapse into either extreme.

### 6. George Pólya’s Heuristics (*How to Solve It*, 1945)
- **Principle:** Divide & conquer via auxiliary problem decomposition and teleological planning (working backwards from goal $\mathcal{G} \to \mathbf{s}_t$).

### 7. Gentner’s Structure-Mapping Engine (SME)
- **Principle:** Relational structural alignment across distant scientific domains based on causal systems rather than surface attributes.
- **Vector Formulation:** Vector Symbolic Architecture (VSA) hypervector bindings $\mathbf{v}_R \otimes (\mathbf{v}_X \oplus \mathbf{v}_Y)$ using Plate's Holographic Reduced Representations (HRR circular convolution) with associative `CleanupMemory`.

### 8. Imre Lakatos's Epistemology of Mathematics (*Proofs and Refutations*, 1976)
- **Principle:** Mathematical heuristic auditing via lemma deconstruction, barrier profile screening, and local/global counterexample matching.
- **Vector Formulation:** Screens claimed proofs against semantic profiles of canonical mathematical impossibility barriers (Tardos Monotone Barrier, Baker-Gill-Solovay Relativization, Razborov-Rudich Natural Proofs, Selberg Sieve Parity, Conway Collatz Undecidability, Scholze-Stix Gap). Generates adversarial boundary instances to highlight unaddressed obstacles or verify invariant drift.

---

## 4. Continuous Latent Reasoning Substrate

- **Chain of Continuous Thought (Coconut-inspired):** Vector transitions and latent Breadth-First Search (BFS) in continuous representation space, avoiding premature token commitment.
- **Diffusion of Thoughts (DoT-inspired):** Artificial potential-field trajectory optimization in vector space with goal attractors ($w_{\text{goal}} = 1.4$) and inverse-distance dead-end repulsors ($w_{\text{dead}} = 1.0$).
- **Negative Manifold Repulsor (NMR):** Mathematical potential field deflecting search vectors away from known falsified hypotheses.
- **Latent MCTS with Process Reward Model (PRM):** UCB1 tree search scored by step-wise task progress, exclusory gain, and hazard penalties.
- **DPLL & Resolution Refutation Solver:** Pure-Python formal satisfiability checker providing deterministic consistency proofs under propositional parsing.
- **SMT Theory Solvers:** Congruence Closure for Equality with Uninterpreted Functions (EUF) and Fourier-Motzkin elimination for Quantifier-Free Linear Real Arithmetic (QF-LRA).
- **Bayesian Optimal Experimental Design (BOED):** Maximizes Shannon Expected Information Gain (EIG) to pick the single most diagnostic test.
- **Synaptic Memory Consolidation:** Compresses redundant dead ends into normalized cluster centroids.

---

## 5. Quickstart & Usage Examples

### Installation

```bash
git clone https://github.com/r3sbarra/super-duper-problem-solving-engine.git
cd super-duper-problem-solving-engine
pip install -e .
```

### 1. Formulating a Problem with Kepner-Tregoe 4D Boundary

```python
from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.core.types import KTBoundary

engine = SuperDuperProblemSolvingEngine()

boundary = KTBoundary(
    identity_is="Resistivity drop occurs sharply at 104 C (377 K)",
    identity_is_not="Does not reach true zero electrical resistance",
    location_is="Multiphase sintered pellets containing Cu2S secondary phases",
    location_is_not="Pure transparent stoichiometric single crystals",
    timing_is="Occurs strictly during 104 C heating/cooling thermal cycle",
    timing_is_not="Continuously variable with copper substitution concentration",
    extent_is="Only certain inhomogeneous fragments show half-levitation",
    extent_is_not="Bulk homogeneous superconductivity throughout entire ingot",
)

problem = engine.formulate_problem(
    title="LK-99 Ambient-Temperature Superconductivity Anomaly",
    specification="Reported room-temperature superconductivity in modified lead apatite Pb10-x Cux (PO4)6O.",
    boundary=boundary,
    goal_criteria=["Identify true physical mechanism of 104 C transition"],
)
```

### 2. Resolving Contradictions with TRIZ Matrix

```python
# Direct lookup in Altshuller's Contradiction Matrix
results = engine.triz.resolve_contradiction_matrix(
    improving_parameter="speed", worsening_parameter="accuracy"
)
for r in results:
    print(f"Principle {r['principle_id']}: {r['name']} - {r['description']}")
```

### 3. Autonomous Discovery Path Deduction

```python
from super_solver.core.types import CrucialExperiment

crucial_experiments = [
    CrucialExperiment(
        id="exp_pure_crystal",
        name="Max Planck Pure Single-Crystal Synthesis",
        description="Synthesize pristine Pb9Cu(PO4)6O without Cu2S impurities.",
        target_hypotheses=["hyp_abduct_1", "hyp_abduct_2"],
        exclusory_predictions={
            "hyp_abduct_1": "superconducts",
            "hyp_abduct_2": "transparent_insulator",
        },
    )
]

path = engine.deduce_discovery_path(
    problem=problem,
    candidate_hypotheses=[
        "Genuine ambient-pressure room-temperature superconductor",
        "Artifact caused by Cu2S first-order structural phase transition at 104 C",
    ],
    crucial_experiments=crucial_experiments,
    ground_truth_outcomes={"exp_pure_crystal": "transparent_insulator"},
)

print(path.final_breakthrough)
# Output: CONFIRMED: Artifact caused by Cu2S first-order structural phase transition at 104 C

# Export visual Mermaid flowchart
print(engine.export_mermaid_diagram(path))
```

### 5. Auditing & Verifying Unverified Mathematical Papers (Lakatos Engine)

```python
# Verify an unverified mathematical preprint claiming P != NP
result = engine.verify_math_paper(
    paper_id="arXiv:1708.03486",
    paper_title="A Solution to the P versus NP Problem",
    abstract_text="We apply the Berg-Ulfberg monotone approximation method to general Boolean circuits computing Clique to establish an exponential lower bound...",
    target_conjecture="P versus NP",
)

print(result.verdict)
# Output: VerificationStatus.REFUTED_FLAWED (Confidence: 98%)
print(result.barrier_violations)
# Output: ['Tardos Monotone Circuit Approximation Barrier (Éva Tardos, 1988)']
print(result.counterexamples)
# Output: ["Tardos's Boolean Function f_T (1988): Contradicts Theorem 6 since f_T in P..."]
```

### 6. Autonomously Suggesting Alternative Research Paths & Barrier Evasion

When a researcher is stuck, when an engineering trade-off occurs, or when an unverified paper is refuted:

```python
paths = engine.suggest_paths(
    problem_specification="Lithium metal battery: Rapid charging causes dendrite growth and thermal runaway.",
    top_k=3,
)

for p in paths:
    print(f"[{p.strategy_type}] {p.title}")
    print(f"  Action: {p.recommended_next_action}")
    print(
        f"  Feasibility: {p.feasibility_score} | Novelty: {p.novelty_score} | Dead-End Margin: {p.dead_end_safety_margin}"
    )
```

### 7. Self-Improvement, Self-Audit, & Synaptic Memory Consolidation

The engine can run directly against itself to audit its invariants, perform causal interventions on its internal parameters, and prune redundant dead-end memories:

```python
# 1. Run full self-audit (DPLL proof, Causal VSA graph surgery, BOED EIG design)
audit = engine.run_self_audit_and_optimization()
print(audit["dpll_invariant_verification"]["verdict"])  # LOGICALLY_SOUND_PROOF
print(audit["causal_intervention_audit"]["average_causal_effect"])  # +0.37

# 2. Consolidate memory (synaptic clustering of redundant dead ends)
pruned = engine.consolidate_memory(similarity_threshold=0.75)
print(f"Pruned {pruned} redundant hazard vectors into cluster centroids.")
```

Or run via the CLI:

```bash
# Run the autonomous engine self-improvement loop
python -m super_solver.cli self-improve

# Prune and consolidate redundant hazard memories
python -m super_solver.cli consolidate-memory --threshold 0.75
```

### 8. Cross-Domain Analogical Transfer (Structural Primitives)

Surface vector retrieval fails when a problem in one domain is structurally
similar to a problem in another (e.g. a climbing pad that grips walls and
Velcro's hook-and-loop mimicry). The engine strips problems and solutions into
structural primitives — atomic action + object-type atoms (e.g. `reduce_density`,
`mimic_structure`, `adhesion`) — and matches on those, which is robust to
surface entity differences.

```python
from super_solver.vectorize import VectorizationService

svc = VectorizationService()
# ... store source solutions ...

# Cross-domain retrieval: climbing pad -> Velcro/gecko mimicry
hits = svc.primitive_analogize(
    "How to make a climbing pad that grips smooth vertical walls without glue",
    top_k=3,
)
for h in hits:
    print(f"{h['primitive_overlap']:.2f}: {h['content']}")
```

Or via the CLI:

```bash
python -m super_solver.cli vectorize primitive-analogize \
  "How to make a climbing pad that grips smooth vertical walls without glue"
```

---

## 6. Optional Small LLM Integration (Zero-Daemon Fallback)

By default, the engine is **100% offline, deterministic, and daemon-free**. Optionally, a local small language model (via Ollama or any OpenAI-compatible endpoint) can be enabled to assist in:
1. **Semantic Clause Extraction**: Translating informal, natural language premises into formal propositional Horn/CNF clauses for the DPLL solver.
2. **Open-Vocabulary Primitive Extraction**: Extracting structural action-object primitives for cross-domain transfer when problems contain vocabulary outside the curated regex dictionary.

### Configuration
```bash
# Enable optional small LLM assistance (default: false)
export SUPER_SOLVER_LLM_ENABLED=true

# Option A: Local Ollama (default: http://localhost:11434, model: qwen2.5:3b)
export SUPER_SOLVER_LLM_URL="http://localhost:11434"
export SUPER_SOLVER_LLM_MODEL="qwen2.5:3b"

# Option B: Generic OpenAI-compatible endpoint
export SUPER_SOLVER_LLM_URL="http://localhost:8000/v1"
export OPENAI_BASE_URL="http://localhost:8000/v1"
```

If the endpoint is disabled, unreachable, or times out, the engine **instantly falls back** to the zero-daemon deterministic regex parser and feature hashing with zero latency overhead.

---

## 7. Capabilities, Scope & Limitations Disclaimer

To maintain scientific rigor and clear engineering expectations:

1. **Heuristic Screening vs. Formal Theorem Proving**:
   - The **Lakatos Engine** performs semantic barrier screening and counterexample template mapping based on vector proximity of abstract prose to canonical mathematical obstructions (e.g. Tardos Monotone Barrier, BGS Relativization).
   - It is **not** an interactive theorem prover (like Lean 4, Isabelle, or Coq). A verdict of `VERIFIED_SOUND` indicates that the paper's techniques avoid known canonical barriers in its stated domain under semantic screening; it does not replace a step-by-step formal derivation check.
2. **SMT Solver Scalability (EUF + QF-LRA)**:
   - The engine includes a pure-Python SMT solver supporting Congruence Closure (EUF) and Fourier-Motzkin variable elimination (QF-LRA).
   - Fourier-Motzkin elimination exhibits $O((N/2)^2)$ pairwise constraint growth per eliminated variable, making it ideal for compact systems ($\le 12$ variables). For industrial-scale constraint systems, an optional bridge to `z3-solver` is provided.
3. **Continuous Latent Guidance vs. Generative Transformers**:
   - The **Coconut** and **DoT** modules implement deterministic vector transitions, spherical normalization, and artificial potential fields (attractors and inverse-distance dead-end repulsors). They provide fast geometric trajectory steering without requiring billion-parameter GPU weights.
4. **Natural Language Logic Ingestion**:
   - In zero-daemon mode, DPLL propositional ingestion uses exact atom regex pattern matching (`->`, `OR`, `NOT`). When processing unstructured prose without logical connectives, enable the optional small LLM parser or provide pre-structured premises.

---

## 8. Benchmark Performance

Running `python tests/benchmark_solver.py`:

| Dimension / Metric | Measured Result | Impact |
| :--- | :--- | :--- |
| **Search Space Reduction (N=1024)** | **99.02%** reduction (10 Platt assays vs 1,024 brute-force) | Exponential hypothesis tree pruning via Shannon information gain. |
| **Dead-End Hazard Precision** | **100.0%** precision (0% false positive hazard alerts) | Deflects reasoning vectors cleanly away from known failure modes. |
| **Vector Step & Proximity Latency**| **0.313 ms** / step on CPU | Local NumPy vector operations with zero external daemon overhead. |
| **Thought Diffusion Convergence** | **+95.5%** goal proximity gain in 10 steps (**0.45 ms**) | Non-autoregressive CFG potential field denoising toward goal manifold. |
| **P != NP Barrier Detection (Blum)**| **0.13 ms** (98% confidence) | Instant barrier detection (Tardos Monotone Approximation Barrier). |
| **Collatz Tao Verification** | **0.05 ms** (90% confidence) | Evaluates 2-adic logarithmic drift and Conway undecidability evasion. |
| **Gaia MOND Falsification Path** | **7.02 ms** (100% confidence) | Autonomous abduction, assay design, and Bayesian hypothesis elimination. |
| **Multi-Paradigm Path Suggestion** | **3.40 - 5.15 ms** across domains | Ranks TRIZ, Polya, Gentner, and Lakatos paths with curiosity scoring. |
| **Full 6-Stage Suite Execution** | **0.10 seconds** total execution time | Instantaneous execution suitable for high-throughput autonomous swarms. |

---

## 9. Integration with `lab-ass`

The engine provides direct plug-and-play integration with the `lab-ass` research platform:
- **`LabAssClient`:** Asynchronous REST client connecting to `lab-ass` sessions on `:8000`.
- **`LabAssSuperSolverBridge`:** Autonomous hooks for decomposing research angles via Polya/TRIZ (`/decompose`), deflecting proposed findings away from toxic dead ends (`/deflect`), checking logical consequence (`/entail`), and selecting experiments via BOED (`/next-experiment`).

---

## 10. Running Tests

```bash
.venv/bin/pytest -v tests/
```

All **80+ unit and integration tests** pass in ~13 seconds with 100% test coverage across mathematical, scientific, engineering, dialectical debate, and meta-optimization domains.

---

## 11. License

This project is licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for details.




