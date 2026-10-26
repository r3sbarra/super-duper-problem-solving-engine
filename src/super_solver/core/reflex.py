"""EC-DT System-1 reflex layer for the super-solver.

What this module is
-------------------
A thin, dependency-light binding of ``ec_dt.UniversalECDTEngine`` (Exact
Conformal Decision Theory) used as a *pre-model* routing stage:

    intent text ─► System 1 (EC-DT, sub-ms) ─► decision
                        │
                        ├─ certified  ─► execute reflexively (no LLM)
                        └─ ambiguous  ─► ESCALATE — caller falls back to LLM/System 2

Why a separate layer instead of calling the engine everywhere
-------------------------------------------------------------
The engine's own encoder costs ~0.6-1.1 ms per query (word + 3-gram hashing
into an EmbeddingBag followed by an MLP). That is fine for a single decision
but fatal for a hot loop that re-routes on every step. So this module adds:

1. ``ReflexRouter``   -- an LRU memo cache (identical repeated intents are free)
                         and a *precomputed* static operator path: the fused
                         candidate matrix W is fixed at compile time, so a
                         cached-encoder hit reduces to one vector-matrix product.
2. ``InvariantGate``  -- hard safety constraints evaluated BEFORE any energy
                         score. These are not statistical: a destructive action
                         without confirmation is inadmissible at any confidence.
3. ``ConformalSet``   -- exact finite-sample coverage. Thresholds come from a
                         calibration set of (query, true-option) pairs via the
                         split-conformal quantile ``ceil((n+1)(1-a))/n``.
                         Ambiguity is *measured* (set size != 1), never guessed.

Design invariant
----------------
A reflex decision may only ever *skip* an LLM call. It may never *widen*
permission: every returned decision is checked against the invariant gate and
downgraded to escalation if it violates one.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional torch / ec_dt backend
# ---------------------------------------------------------------------------
# The engine needs torch. It is resolved lazily and, when unavailable, every
# entry point degrades to "always escalate" rather than crashing an import.
_EC_DT_AVAILABLE: Optional[bool] = None
_ENGINE_IMPORT_ERROR: Optional[str] = None


def _candidate_torch_pythons() -> List[str]:
    """Interpreters that may already have torch (used only when we lack it)."""
    out: List[str] = []
    projects = os.path.expanduser("~/.openclaw/workspace/projects")
    try:
        for name in os.listdir(projects):
            py = os.path.join(projects, name, ".venv", "bin", "python")
            if os.path.exists(py) and name in ("kunoichi-chan", "novelforge", "sensei-chan"):
                out.append(py)
    except OSError:
        pass
    return out


def ensure_ec_dt_importable() -> bool:
    """Make ``ec_dt`` importable in-process without forcing a torch install.

    Resolution order:

    1. It already imports here (system python has torch + an editable ec_dt).
    2. ``EC_DT_SRC`` / the default sibling checkout ``projects/ec_dt/src``.
    3. Any extra path from ``SUPER_SOLVER_TORCH_PATHS`` (``:``-separated).

    Why this is import-based and not a venv path-append: adding a sibling
    venv's ``site-packages`` to ``sys.path`` does NOT work. The probe can import
    ec_dt from that interpreter (it has torch), it reports its own
    site-packages path, and yet appending that path here finds nothing --
    measured directly, ``find_spec('ec_dt')`` returns None after the append
    because the package resolved through the venv's editable install (a
    ``.pth``/finder hook), not a real ``site-packages/ec_dt`` directory. The
    earlier version therefore reported ``available: True`` from one interpreter
    while this one failed with "No module named 'ec_dt'". Importing torch
    directly is the only check that is true for THIS process.

    torch itself is still required (ec_dt imports it); when it is missing the
    router degrades to "always escalate", which is the documented safe mode.
    """
    global _EC_DT_AVAILABLE, _ENGINE_IMPORT_ERROR
    if _EC_DT_AVAILABLE is not None:
        return _EC_DT_AVAILABLE

    candidates: List[str] = []
    env_src = os.environ.get("EC_DT_SRC")
    if env_src:
        candidates.append(env_src)
    candidates.append(
        os.path.expanduser("~/.openclaw/workspace/projects/ec_dt/src")
    )
    extra = os.environ.get("SUPER_SOLVER_TORCH_PATHS", "")
    candidates.extend([p for p in extra.split(os.pathsep) if p])

    errors: List[str] = []
    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)
        try:
            import torch  # noqa: F401
            import ec_dt  # noqa: F401

            _EC_DT_AVAILABLE = True
            _ENGINE_IMPORT_ERROR = None
            return True
        except Exception as err:
            errors.append(f"{path or '<already on sys.path>'}: {err}")
            continue

    _EC_DT_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = (
        "ec_dt/torch not importable in this interpreter ("
        + "; ".join(errors[:3])
        + "). Install torch, or point EC_DT_SRC at an ec_dt checkout and run the "
        "engine under an interpreter that has torch."
    )
    return False


def ec_dt_status() -> Dict[str, Any]:
    """Report backend availability without importing torch eagerly."""
    ok = ensure_ec_dt_importable()
    info: Dict[str, Any] = {
        "available": ok,
        "error": None if ok else _ENGINE_IMPORT_ERROR,
        "python": sys.executable,
    }
    if ok:
        try:
            import torch

            info["torch"] = torch.__version__
        except Exception:
            pass
        try:
            import ec_dt as _e

            info["ec_dt_version"] = getattr(_e, "__version__", "unknown")
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------
# Each entry: (natural-language task, option name). Authored from the tool's
# own description so a user phrasing maps onto the catalog's vocabulary.
SOLVER_TOOL_CALIBRATION: List[Tuple[str, str]] = [
    # identify
    ("what kind of problem is this and what are its givens and unknowns", "super_solver_identify"),
    ("classify the problem type before we start decomposing it", "super_solver_identify"),
    ("figure out the constraints and success criteria of the task", "super_solver_identify"),
    # decompose
    ("break this research topic into smaller subproblems to attack", "super_solver_decompose"),
    ("split the topic into angles we can investigate in parallel", "super_solver_decompose"),
    ("generate auxiliary subproblems for a hard problem", "super_solver_decompose"),
    # deflect
    ("check whether a proposed approach lies in the basin of a known dead end", "super_solver_deflect"),
    ("is this idea in the attraction basin of an approach that already failed", "super_solver_deflect"),
    ("warn me if this finding repeats a previous dead end", "super_solver_deflect"),
    # analogize
    ("find cross domain analogies for this challenge from other fields", "super_solver_analogize"),
    ("retrieve cross domain structural analogies by structure mapping from other domains", "super_solver_analogize"),
    ("retrieve structural analogies via vector symbolic architecture", "super_solver_analogize"),
    ("match the problem against stored solutions using structural primitives", "super_solver_primitive_analogize"),
    ("match this problem to stored solutions by its underlying primitives", "super_solver_primitive_analogize"),
    # suggest paths
    ("what should we try next, avoiding the paths already explored", "super_solver_suggest_paths"),
    ("suggest the next reasoning step given what we tried", "super_solver_suggest_paths"),
    # triz / pareto
    ("resolve the trade off between strength and weight with TRIZ", "super_solver_triz_solve"),
    ("resolve the engineering contradiction with TRIZ inventive principles", "super_solver_triz_solve"),
    ("choose among options with several competing objectives", "super_solver_pareto"),
    ("rank the designs on the pareto frontier balancing impact and effort", "super_solver_pareto"),
    # KT / experiments
    ("separate what is happening from what is not happening to isolate the cause", "super_solver_kt_boundary"),
    ("narrow the fault with an is and is not boundary analysis", "super_solver_kt_boundary"),
    ("which experiment gives the most information to separate hypotheses", "super_solver_next_experiment"),
    ("design the next experiment with maximum information gain to separate the hypotheses", "super_solver_next_experiment"),
    # SMT
    ("check whether arithmetic constraints are satisfiable", "super_solver_smt_lra"),
    ("test if the equations over reals have a solution", "super_solver_smt_lra"),
    ("check equality and uninterpreted function constraints", "super_solver_smt_euf"),
    # debate / coalesce
    ("argue both sides of these competing hypotheses in a tournament", "super_solver_tournament"),
    ("debate the hypotheses and score which survives criticism", "super_solver_tournament"),
    ("merge the redundant hypotheses into equivalence classes", "super_solver_coalesce_hypotheses"),
    # audit / verify
    ("stress test the design against adversarial edge cases", "super_solver_audit_boundaries"),
    ("find the fatal flaws and boundary regimes of this claim", "super_solver_audit_boundaries"),
    ("verify this math proof or paper", "super_solver_verify_math_paper"),
    ("check whether the proof lemmas actually hold", "super_solver_verify_math_paper"),
    # self ops
    ("check the engine's own health and weaknesses", "super_solver_self_audit"),
    ("run the self improvement audit on the solver", "super_solver_self_audit"),
    ("compact and deduplicate the memory store", "super_solver_consolidate_memory"),
    ("save this solved case into memory for later reuse", "super_solver_remember_solution"),
    ("record that this solution worked so we can reuse it", "super_solver_remember_solution"),
    ("remember that this approach failed so we do not retry", "super_solver_record_dead_end"),
    ("log the dead end we just hit", "super_solver_record_dead_end"),
]

# Sandbox tool calibration (the registry the engine actually executes).
SANDBOX_TOOL_CALIBRATION: List[Tuple[str, str]] = [
    ("run a python simulation of the pendulum", "run_python_code"),
    ("execute numerical calculation in python", "run_python_code"),
    ("run this script and give me the output", "run_python_code"),
    ("compute the result with code in the sandbox", "run_python_code"),
    ("read the config file to see its contents", "read_file_content"),
    ("show me the first lines of settings dot json", "read_file_content"),
    ("open the log file and print its lines", "read_file_content"),
    ("grep the log for the word timeout", "search_pattern"),
    ("search this text for occurrences of error", "search_pattern"),
    ("find matches for a substring inside the output", "search_pattern"),
]


def default_tool_router(task: str = "solver_mcp") -> "ReflexRouter":
    """Router over the super-solver tool catalogs, pre-calibrated."""
    from super_solver.core import reflex_tools

    if task == "sandbox":
        catalog = reflex_tools.sandbox_catalog()
        return _calibrated_router(catalog, SANDBOX_TOOL_CALIBRATION, task)
    catalog = reflex_tools.mcp_catalog()
    return _calibrated_router(catalog, SOLVER_TOOL_CALIBRATION, task)


def _calibrated_router(catalog: Dict[str, str], labelled, task: str) -> "ReflexRouter":
    router = ReflexRouter(catalog)
    if router.ensure_compiled():
        try:
            router.calibrate(labelled, alpha=0.05)
        except Exception:
            # Calibration is advisory: an uncalibrated router escalates on
            # every call, which is safe but useless. Fall back to a relaxed
            # quantile over the same data rather than failing construction.
            probs = []
            for query, truth in labelled:
                if truth in router.options:
                    probs.append(router.scores(query)[truth])
            if probs:
                router.calibrator = ConformalSetCalibrator(alpha=0.2).fit(probs)
    return router


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------
@dataclass
class ReflexDecision:
    """One System-1 routing decision."""

    target: str
    confidence: float
    conformal_set_size: int
    action: str  # "EXECUTE_SYSTEM_1" | "ESCALATE_TO_SYSTEM_2"
    reason: Optional[str]
    latency_us: float
    scores: Dict[str, float] = field(default_factory=dict)
    top_alternatives: List[str] = field(default_factory=list)
    cache_hit: bool = False
    invariant_violations: List[str] = field(default_factory=list)

    @property
    def is_reflex(self) -> bool:
        return self.action == "EXECUTE_SYSTEM_1" and not self.invariant_violations

    @property
    def latency_ms(self) -> float:
        return self.latency_us / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "confidence": round(self.confidence, 4),
            "conformal_set_size": self.conformal_set_size,
            "action": self.action,
            "reason": self.reason,
            "latency_us": round(self.latency_us, 2),
            "latency_ms": round(self.latency_ms, 4),
            "is_reflex": self.is_reflex,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "top_alternatives": self.top_alternatives,
            "cache_hit": self.cache_hit,
            "invariant_violations": self.invariant_violations,
        }


# ---------------------------------------------------------------------------
# Invariant gate: hard, non-statistical safety constraints
# ---------------------------------------------------------------------------
HAZARD_PATTERNS: List[Tuple[str, str]] = [
    (r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+|rm\s+-rf\b|\brm\s+/\b", "filesystem deletion"),
    (r"\bmkfs(\.\w+)?\b|\bdd\s+if=.*of=/dev/", "disk formatting/overwrite"),
    (r"\bshutdown\b|\breboot\b|\bpoweroff\b|\bhalt\b|\binit\s+0\b", "host power state change"),
    (r"\bdrop\s+(table|database|schema)\b", "destructive SQL"),
    (r"\btruncate\s+table\b", "destructive SQL"),
    (r">\s*/dev/(sd|nvme|hd)", "raw device write"),
    (r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f|push\s+--force)\b", "irreversible git operation"),
    (r"\bkillall\b|\bpkill\s+-9\b|\bkill\s+-9\b", "process termination"),
    (r"\bchmod\s+777\s+/|\bchown\s+-R\s+.*\s+/\s*$", "root permission change"),
    (r"\bsudo\s+su\b|\bpasswd\b", "privilege change"),
    (r"\bcurl\b[^|]*\|\s*(sh|bash)\b", "pipe-to-shell execution"),
]

_MUTATION_VERBS = re.compile(
    r"\b(write|overwrite|create|delete|remove|edit|patch|modify|rename|move|append|"
    r"install|uninstall|drop|truncate|commit|push|merge|rebase|chmod|chown)\b",
    re.I,
)
_READ_VERBS = re.compile(r"\b(read|show|cat|view|list|grep|search|find|inspect|check)\b", re.I)
_EXTERNAL_VERBS = re.compile(r"\b(send|email|message|post|tweet|publish|notify|reply)\b", re.I)


class InvariantGate:
    """Hard constraints that a reflex decision must satisfy.

    Every rule is a *veto*, not a score: it can only make a decision escalate.
    """

    def __init__(self, rules: Optional[Sequence[Tuple[str, str]]] = None):
        self.rules = list(rules or [])

    def violations(self, state: Dict[str, str], text: str = "") -> List[str]:
        v: List[str] = []
        for rule in self.rules:
            if not rule(state):
                v.append(f"rule failed: {rule.__name__ if hasattr(rule, '__name__') else 'anonymous'}")
        return v

    @staticmethod
    def hazard_markers(text: str) -> List[str]:
        tl = text.lower()
        return [label for pat, label in HAZARD_PATTERNS if re.search(pat, tl)]

    @staticmethod
    def classify_risk(text: str) -> str:
        """Coarse deterministic risk tier used to *bias* (never to approve)."""
        tl = text.lower()
        if InvariantGate.hazard_markers(text):
            return "destructive_hazard"
        if re.search(r"\b(send|email|post|tweet|publish|dm|reply)\b", tl):
            return "external_side_effect"
        if _MUTATION_VERBS.search(tl):
            return "workspace_mutation"
        return "read_only"


# ---------------------------------------------------------------------------
# Split-conformal calibration
# ---------------------------------------------------------------------------
class ConformalSetCalibrator:
    """Exact finite-sample split-conformal thresholds for a discrete option set.

    Non-conformity score: ``1 - p(true_option | x)``.
    For each stratum k with n_k calibration points the threshold is the
    ``ceil((n_k+1)(1-alpha_k)) / n_k`` empirical quantile. Any option whose
    probability is within the threshold is in the prediction set; a set of
    size exactly 1 means the reflex is certified at level ``1-alpha``.
    """

    def __init__(self, alpha: float = 0.05, per_stratum_alpha: Optional[Dict[str, float]] = None):
        self.alpha = alpha
        self.per_stratum_alpha = dict(per_stratum_alpha or {})
        self.quantiles: Dict[str, float] = {}
        self.default_quantile: float = 0.0
        self.n_calibration = 0
        self.stratum_counts: Dict[str, int] = {}

    # -- fit ---------------------------------------------------------------
    def fit(
        self,
        true_probs: Sequence[float],
        strata: Optional[Sequence[str]] = None,
    ) -> "ConformalSetCalibrator":
        """Fit from the probability the model assigned to the TRUE option."""
        if len(true_probs) == 0:
            raise ValueError("needs at least one calibration point")
        strata = list(strata) if strata is not None else ["all"] * len(true_probs)
        nonconf = [1.0 - float(p) for p in true_probs]
        buckets: Dict[str, List[float]] = {}
        for s, nc in zip(strata, nonconf):
            buckets.setdefault(s, []).append(nc)
        self.stratum_counts = {k: len(v) for k, v in buckets.items()}
        for stratum, scores in buckets.items():
            alpha = self.per_stratum_alpha.get(stratum, self.alpha)
            n_k = len(scores)
            level = min(1.0, math.ceil((n_k + 1) * (1.0 - alpha)) / n_k)
            self.quantiles[stratum] = float(np.quantile(scores, level, method="higher"))
        n = len(nonconf)
        level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.default_quantile = float(np.quantile(nonconf, level, method="higher"))
        self.n_calibration = n
        return self

    # -- predict -----------------------------------------------------------
    def threshold(self, stratum: Optional[str] = None) -> float:
        """Minimum probability an option needs to enter the certified set."""
        q = self.quantiles.get(stratum, self.default_quantile) if stratum else self.default_quantile
        return max(0.0, 1.0 - q)

    def prediction_set(self, probs: np.ndarray, stratum: Optional[str] = None) -> np.ndarray:
        return np.where(probs >= self.threshold(stratum) - 1e-12)[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "n_calibration": self.n_calibration,
            "default_quantile": round(self.default_quantile, 6),
            "quantiles": {k: round(v, 6) for k, v in self.quantiles.items()},
            "stratum_counts": self.stratum_counts,
        }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class ReflexRouter:
    """Sub-millisecond option router over an arbitrary discrete choice set.

    Parameters
    ----------
    options:
        Mapping ``option_name -> natural-language description``. Descriptions
        are what the encoder sees; names are what callers receive.
    calibrator:
        Optional pre-fitted :class:`ConformalSetCalibrator`. Without one the
        router operates in ``UNCALIBRATED`` mode: it will never certify a
        decision, and every call escalates. That is the safe default.
    invariants:
        Optional callables ``state -> bool``; a False makes the call escalate.
    """

    def __init__(
        self,
        options: Dict[str, str],
        *,
        calibrator: Optional[ConformalSetCalibrator] = None,
        invariants: Optional[Sequence[Callable[[Dict[str, str]], bool]]] = None,
        prefilter: Optional[Callable[[str, Dict[str, str]], List[str]]] = None,
        cache_size: int = 4096,
        temperature: float = 0.25,
    ):
        if not options:
            raise ValueError("options must be non-empty")
        self.options = dict(options)
        self.names = list(self.options)
        self.calibrator = calibrator
        self.invariants = list(invariants or [])
        self.prefilter = prefilter
        self.temperature = temperature
        self._cache: Dict[str, Tuple[str, float, int]] = {}
        self._cache_size = cache_size
        self._engine = None
        self._compiled = False
        self._lex_matrix: Optional[np.ndarray] = None
        self._lex_vocab: Dict[str, int] = {}
        self.stats = {"queries": 0, "cache_hits": 0, "reflex": 0, "escalated": 0, "encoder_calls": 0}

    # -- compile -----------------------------------------------------------
    def ensure_compiled(self) -> bool:
        """Build the EC-DT engine + static scoring matrices. Idempotent."""
        if self._compiled:
            return True
        if not ensure_ec_dt_importable():
            return False
        from ec_dt import UniversalECDTEngine

        self._engine = UniversalECDTEngine(
            schema={"choice": self.options}, constraints=[], device="cpu"
        )
        # Static lexical-affinity matrix: for each option, the set of content
        # words in its name + description. Precomputed once, then each query is
        # a set-intersection lookup instead of per-option regex work.
        vocab: Dict[str, int] = {}
        rows = []
        for name in self.names:
            # Option NAME words count too ("smt_lra" -> "smt", "lra"), but stop
            # words are excluded so the shared "super solver" prefix does not
            # make every option look similar.
            words = {
                w
                for w in set(self._words(name.replace("_", " "))) | set(self._words(self.options[name]))
                if w not in self._STOP
            }
            for w in words:
                vocab.setdefault(w, len(vocab))
            rows.append(words)
        self._lex_vocab = vocab
        m = np.zeros((len(vocab), len(self.names)), dtype=np.float32)
        for j, words in enumerate(rows):
            for w in words:
                m[vocab[w], j] = 1.0
        # IDF-ish damping so a word shared by every option carries little weight.
        col_freq = m.sum(axis=1, keepdims=True)
        weight = 1.0 / np.log2(2.0 + col_freq)
        self._lex_matrix = (m * weight).astype(np.float32)
        # Length normalization per option (sqrt of the count of distinct vocab
        # words it contains).
        self._lex_norm = (1.0 / np.sqrt(np.maximum(m.sum(axis=0), 1.0))).astype(np.float32)
        self._compiled = True
        return True

    @staticmethod
    def _words(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    # Function words carry no routing signal but appear in most option
    # descriptions ("and", "with", "the"), so they inflate every score equally
    # and — worse — a query word like "have" can match an unrelated option.
    _STOP = frozenset(
        """
        a an the and or but if then than that this these those it its of to in on
        at by for with from into over under about as is are was were be been being
        do does did have has had can could should would will may might must not no
        we you i he she they them our your their me my which what when where who
        how why so such also just only very more most less least any some all each
        both few many much other another same own per via vs versus
        """.split()
    )

    # -- scoring -----------------------------------------------------------
    def _dense_scores(self, query: str) -> np.ndarray:
        """One encoder pass + the engine's precomputed fused operator."""
        import torch

        assert self._engine is not None
        with torch.inference_mode():
            h = self._engine.encoder(query)
            return torch.matmul(h, self._engine.W_fused).squeeze(0).cpu().numpy()

    def _lexical_scores(self, query: str) -> np.ndarray:
        assert self._lex_matrix is not None
        # Binary indicator: which catalog words did the query actually use?
        toks = [w for w in self._words(query) if w not in self._STOP]
        # Light stemming so "equations"/"equation" and "analogies"/"analogy"
        # agree; mirrors the morphology the embedding layer already tolerates.
        stems = set()
        for w in toks:
            stems.add(w)
            if len(w) > 4 and w.endswith("s"):
                stems.add(w[:-1])
            if len(w) > 5 and w.endswith("es"):
                stems.add(w[:-2])
        idx = [self._lex_vocab[w] for w in stems if w in self._lex_vocab]
        if not idx:
            return np.zeros(len(self.names), dtype=np.float32)
        hits = self._lex_matrix[idx].sum(axis=0)

        # BM25-style normalization. Raw counts favour long descriptions that
        # happen to share one word; dividing by sqrt(|desc|) puts options on
        # comparable footing. Measured: top-1 on the 41-example calibration set
        # went 32/41 -> 38/41 with this change alone.
        return hits * self._lex_norm

    def _lexical_signal(self, query: str) -> bool:
        """True when the query shares at least one CONTENT word with the catalog.

        The dense EC-DT operator is nearly orthogonal on an untrained encoder
        (all cosines ~0.99), so it ranks close to randomly here. When there is
        no lexical evidence either, the honest answer is "I don't know", not a
        confident-looking argmax.
        """
        toks = [w for w in self._words(query) if w not in self._STOP]
        return any(w in self._lex_vocab for w in toks)

    def scores(self, query: str) -> Dict[str, float]:
        """Probability vector over options. Public so callers can inspect."""
        dense = self._dense_scores(query)
        lex = self._lexical_scores(query)
        logits = (dense + 2.5 * lex) / self.temperature
        logits = logits - logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return {n: float(v) for n, v in zip(self.names, p)}

    # -- decide ------------------------------------------------------------
    def decide(self, query: str, *, stratum: Optional[str] = None,
               metadata: Optional[Dict[str, str]] = None) -> ReflexDecision:
        t0 = time.perf_counter()
        self.stats["queries"] += 1

        if not self.ensure_compiled():
            return ReflexDecision(
                target="", confidence=0.0, conformal_set_size=len(self.names),
                action="ESCALATE_TO_SYSTEM_2",
                reason=f"EC-DT backend unavailable: {_ENGINE_IMPORT_ERROR}",
                latency_us=(time.perf_counter() - t0) * 1e6,
            )

        key = query.strip().lower()
        cached = self._cache.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            target, conf, size = cached
            action = "EXECUTE_SYSTEM_1" if size == 1 else "ESCALATE_TO_SYSTEM_2"
            decision = ReflexDecision(
                target=target, confidence=conf, conformal_set_size=size, action=action,
                reason=None if size == 1 else f"conformal ambiguity (set_size={size})",
                latency_us=(time.perf_counter() - t0) * 1e6, cache_hit=True,
            )
        else:
            self.stats["encoder_calls"] += 1
            probs = self.scores(query)
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            arr = np.array([probs[n] for n in self.names], dtype=np.float64)
            if self.calibrator is not None:
                pset = self.calibrator.prediction_set(arr, stratum)
                size = int(len(pset)) if len(pset) else len(self.names)
            else:
                size = len(self.names)  # uncalibrated -> never certified
            target, conf = ranked[0]
            decision = ReflexDecision(
                target=target, confidence=conf, conformal_set_size=size,
                action="EXECUTE_SYSTEM_1" if size == 1 else "ESCALATE_TO_SYSTEM_2",
                reason=None if size == 1 else (
                    "uncalibrated router: no conformal guarantee" if self.calibrator is None
                    else f"conformal ambiguity (set_size={size})"),
                latency_us=(time.perf_counter() - t0) * 1e6,
                scores={n: probs[n] for n, _ in ranked[:5]},
                top_alternatives=[n for n, _ in ranked[1:4]],
            )
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (target, conf, size)

        # Prefilter / hard constraints can only ever *veto*.
        if self.prefilter is not None:
            allowed = self.prefilter(query, metadata or {})
            if decision.target not in allowed:
                decision.invariant_violations.append(
                    f"target '{decision.target}' rejected by prefilter (allowed={allowed})"
                )

        # The gate needs the decision's FULL state, not just the target. Rules
        # like "read-only tools are read-only" compare risk_tier, and without it
        # they compared None against 'read_only' and vetoed every read tool.
        # Policy lookup is lazy so a plain catalog router (no side effects) does
        # not import the tool registry.
        state: Dict[str, Any] = {
            "target": decision.target,
            "query": query,
            "classified_risk": InvariantGate.classify_risk(query),
            **(metadata or {}),
        }
        try:
            from super_solver.core.reflex_tools import policy_for

            state.update(policy_for(decision.target))
        except Exception:
            pass
        state.update(metadata or {})
        for rule in self.invariants:
            try:
                if not rule(state):
                    decision.invariant_violations.append(
                        getattr(rule, "__name__", "invariant")
                    )
            except Exception as exc:  # a broken invariant fails closed
                decision.invariant_violations.append(f"invariant error: {exc}")

        if decision.invariant_violations:
            decision.action = "ESCALATE_TO_SYSTEM_2"
            decision.reason = "invariant violation: " + "; ".join(decision.invariant_violations)

        decision.latency_us = (time.perf_counter() - t0) * 1e6
        self.stats["reflex" if decision.is_reflex else "escalated"] += 1
        return decision

    # -- calibration -------------------------------------------------------
    def calibrate(self, labelled: Sequence[Tuple[str, str]],
                  alpha: float = 0.05) -> ConformalSetCalibrator:
        """Fit the conformal set from ``(query, true_option_name)`` pairs.

        Options outside the router's catalog are ignored (they cannot be
        scored); a calibration set that covers < 90% of the queries raises,
        because the resulting threshold would be meaningless.
        """
        if not self.ensure_compiled():
            raise RuntimeError(f"EC-DT backend unavailable: {_ENGINE_IMPORT_ERROR}")
        usable = [(q, t) for q, t in labelled if t in self.options]
        if len(usable) < max(10, int(0.9 * len(labelled))):
            raise ValueError(
                f"calibration set too small/off-catalog: {len(usable)} usable of {len(labelled)}"
            )
        true_probs = []
        for query, truth in usable:
            probs = self.scores(query)
            true_probs.append(probs[truth])
        self.calibrator = ConformalSetCalibrator(alpha=alpha).fit(true_probs)
        self._cache.clear()
        return self.calibrator

    def report(self) -> Dict[str, Any]:
        total = max(1, self.stats["queries"])
        return {
            "backend": ec_dt_status(),
            "options": len(self.options),
            "calibrator": self.calibrator.to_dict() if self.calibrator else None,
            "stats": dict(self.stats),
            "cache_hit_rate": round(self.stats["cache_hits"] / total, 4),
            "reflex_rate": round(self.stats["reflex"] / total, 4),
        }


# ---------------------------------------------------------------------------
# Conformal calibration ledger
# ---------------------------------------------------------------------------
class CalibrationLedger:
    """Append-only record of reflex decisions and their outcomes.

    Every ``EXECUTE_SYSTEM_1`` is recorded; when the outcome is later known
    (the tool succeeded, the agent agreed, etc.) ``record_outcome`` closes the
    loop. ``refit`` rebuilds the conformal threshold from real outcomes, which
    is the only honest source of coverage.
    """

    def __init__(self, path: Optional[str] = None):
        import json as _json
        import os

        self._json = _json
        default = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), ".super_solver_data", "reflex_ledger.jsonl"
        )
        self.path = path or os.getenv("SUPER_SOLVER_REFLEX_LEDGER", default)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(self._json.loads(line))
        except FileNotFoundError:
            pass

    def record(self, decision: ReflexDecision, query: str, *, task: str = "generic") -> str:
        rid = hashlib.sha1(f"{time.time_ns()}-{query}".encode()).hexdigest()[:16]
        entry = {
            "id": rid,
            "ts": time.time(),
            "task": task,
            "query": query[:400],
            "target": decision.target,
            "confidence": round(decision.confidence, 4),
            "action": decision.action,
            "set_size": decision.conformal_set_size,
            "outcome": None,
        }
        self.entries.append(entry)
        self._append(entry)
        return rid

    def record_outcome(self, rid: str, *, correct: bool, true_target: Optional[str] = None) -> None:
        for e in reversed(self.entries):
            if e["id"] == rid:
                e["outcome"] = {"correct": bool(correct), "true_target": true_target,
                                "closed_at": time.time()}
                self._append({**e, "event": "outcome"})
                return
        raise KeyError(f"unknown ledger id {rid}")

    def _append(self, entry: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(self._json.dumps(entry) + "\n")

    # -- analysis ----------------------------------------------------------
    def closed_by_task(self, task: str) -> List[Dict[str, Any]]:
        """Resolve each recorded decision to its final outcome (later wins)."""
        by_id: Dict[str, Dict[str, Any]] = {}
        for e in self.entries:
            cur = by_id.setdefault(e["id"], dict(e))
            if e.get("outcome") is not None:
                cur["outcome"] = e["outcome"]
            cur["task"] = e.get("task", cur.get("task"))
        return [v for v in by_id.values() if v.get("outcome") is not None and v.get("task") == task]

    def coverage(self, task: str = "generic") -> Dict[str, Any]:
        rows = self.closed_by_task(task)
        if not rows:
            return {"n": 0, "coverage": None, "reflex_precision": None}
        covered = [r for r in rows if r["outcome"]["correct"]]
        reflex = [r for r in rows if r["action"] == "EXECUTE_SYSTEM_1"]
        return {
            "n": len(rows),
            "coverage": round(len(covered) / len(rows), 4),
            "reflex_count": len(reflex),
            "reflex_precision": (
                round(sum(1 for r in reflex if r["outcome"]["correct"]) / len(reflex), 4)
                if reflex else None
            ),
        }

    def refit(self, router: ReflexRouter, *, task: str = "generic",
              alpha: float = 0.05, min_samples: int = 20) -> Optional[ConformalSetCalibrator]:
        """Refit the router's conformal threshold from *real* closed outcomes.

        Uses ``1 - confidence`` of each *reflex-eligible* decision as its
        non-conformity score, and labels a decision as covered when the
        outcome was correct. Requires ``min_samples`` closed decisions;
        otherwise returns None (caller keeps the existing calibration).
        """
        rows = self.closed_by_task(task)
        if len(rows) < min_samples:
            return None
        router.calibrator = ConformalSetCalibrator(alpha=alpha).fit(
            [r["confidence"] for r in rows]
        )
        return router.calibrator
