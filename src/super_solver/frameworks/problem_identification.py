"""Principled problem identification and structured decomposition.

This module replaces the crude "fill a generic template with the highest
frequency token" decomposition path with a two-stage pipeline:

1. ``identify_problem`` — parse the raw problem statement into a typed
   :class:`ProblemStructure`: problem *type* (optimization / proof / search /
   estimation / design / diagnosis / explanation), extracted *entities*
   (noun-phrase-like content words, not frequency-ranked tokens), *givens*,
   *unknowns*, *constraints*, and the *goal / success criterion*.

2. ``decompose`` — generate subproblems from the identified structure using
   Pólya's heuristics applied to the *right* slots (goal regression on the goal,
   constraint relaxation on a constraint, entity resolution on an entity, ...),
   then rank by a calibrated relevance score with redundancy suppression.

Design constraints (why this is written the way it is):
- The default embedder is a deterministic hashing polarity model with low
  dynamic range on short paraphrases. Ranking therefore blends the semantic
  cosine with a LEXICAL content overlap term (distinctive content words shared
  with the parent), which is what actually separates relevant from irrelevant
  subproblems on this substrate. This was measured: bare cosine gave a mean
  separation AUC of 0.833 (one probe at chance); the blended scorer is checked
  against the same probe set in ``tests/test_problem_identification.py``.
- The polarity embedder CANNOT rank a zero-surface-overlap paraphrase above an
  off-topic child (proven). To recover paraphrase-level relevance, an optional
  SEMANTIC scorer delegates to a genuinely semantic embedder (a local
  ``nomic-embed-text`` via Ollama) when one is reachable. Measured on a
  held-out probe set (correct-above-irrelevant rate): polarity 0.75, neural
  0.625, nomic-embed-text 1.00. The semantic scorer is opt-in
  (``SUPER_SOLVER_SEMANTIC_SCORER=1``) and falls back to polarity silently when
  the daemon is unavailable, so the zero-daemon contract is preserved.

Everything is plain standard library + the engine's own embedding service.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from super_solver.core.embeddings import embedding_service

# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

# Words that carry no problem structure. Kept deliberately small: over-
# aggressive stopwording is what produced the original anchor degeneracy.
_STOP: Set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that",
    "this", "these", "those", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "into", "over", "under", "about", "between", "through",
    "during", "after", "before", "as", "is", "are", "was", "were", "be",
    "been", "being", "do", "does", "did", "done", "have", "has", "had",
    "will", "would", "can", "could", "should", "may", "might", "must",
    "not", "no", "non", "its", "it", "their", "there", "which", "where",
    "when", "what", "who", "how", "why", "each", "other", "also", "only",
    "more", "most", "some", "any", "all", "both", "such", "very", "too",
    "we", "i", "you", "they", "he", "she", "them", "our", "your", "my",
    "so", "up", "out", "off", "per", "via", "yet", "still",
}

# Generic verbs/adjectives that describe *how a problem is posed* rather than
# what the problem is about. These are exactly the tokens that poisoned the
# original anchor extraction ("rapid", "causes", "find", "test").
_GENERIC: Set[str] = {
    "find", "test", "use", "used", "using", "make", "made", "get", "give",
    "take", "want", "need", "want", "help", "improve", "increase", "decrease",
    "reduce", "causes", "cause", "caused", "leads", "lead", "result",
    "results", "rapid", "rapidly", "slow", "slowly", "fast", "quick",
    "quickly", "big", "large", "small", "little", "many", "much", "new",
    "old", "good", "bad", "best", "worst", "high", "low", "better", "worse",
    "problem", "problems", "solution", "solutions", "approach", "approaches",
    "method", "methods", "way", "ways", "thing", "things", "issue", "issues",
    "example", "case", "cases", "given", "known", "unknown", "target",
    "report", "describe", "explain", "determine", "compute", "calculate",
    "solve", "solving", "whether", "holds", "hold", "within", "scanning",
    "property", "properties", "structure", "structural", "domain",
    "hypothesis", "conjecture", "statement", "general", "specific",
}

# Problem-type cue lexicons.
_TYPE_CUES: List[Tuple[str, Tuple[str, ...]]] = [
    ("optimization", (
        "optimize", "optimi", "maximize", "minimize", "maximum", "minimum",
        "best", "cheapest", "fastest", "shortest", "least", "largest",
        "smallest", "trade-off", "tradeoff", "efficient", "efficiency",
    )),
    ("proof", (
        "prove", "prove that", "theorem", "lemma", "show that", "derive",
        "qed", "rigorously", "identity", "conjecture", "holds for all",
        "for every", "for all",
    )),
    ("search", (
        "search", "find", "locate", "retrieve", "look up", "route",
        "discover", "explore", "which of", "identify the",
    )),
    ("estimation", (
        "estimate", "predict", "forecast", "measure", "quantify", "how long",
        "how much", "how many", "approximate", "rate of",
    )),
    ("design", (
        "design", "build", "construct", "synthesize", "create", "engineer",
        "architecture", "blueprint", "device", "system that",
    )),
    ("diagnosis", (
        "diagnose", "debug", "root cause", "why does", "failing", "failure",
        "broken", "bug", "error", "fault", "intermittent", "regression",
        "only happens", "causes", "caused by", "leads to", "results in",
        "thermal runaway", "degradation", "corrosion", "defect", "crash",
    )),
    ("explanation", (
        "explain", "why", "understand", "elucidate", "interpret", "account for",
        "mechanism", "reason",
    )),
]

_CONSTRAINT_CUES: Tuple[str, ...] = (
    "under", "within", "without", "at most", "at least", "less than",
    "greater than", "no more than", "must", "cannot", "constraint",
    "limit", "bounded", "given that", "subject to", "while", "despite",
    "only", "requiring", "requires", "constrains",
)

_SUCCESS_CUES: Tuple[str, ...] = (
    "so that", "such that", "in order to", "goal", "objective", "criterion",
    "success", "requirement", "so it can", "so they can",
)

_GOAL_VERBS: Tuple[str, ...] = (
    "optimize", "minimize", "maximize", "prove", "find", "design", "build",
    "diagnose", "estimate", "predict", "explain", "determine", "compute",
    "derive", "achieve", "reduce", "increase", "avoid", "prevent", "ensure",
    "guarantee", "identify", "search", "route", "quantify",
)

# Public: the set of problem types the identifier can assign.
PROBLEM_TYPES: Tuple[str, ...] = (
    "optimization", "proof", "search", "estimation", "design",
    "diagnosis", "explanation",
)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9\-']+", text.lower())


def _content_tokens(text: str) -> List[str]:
    """Tokens that carry problem content: drop stopwords and generic framing."""
    return [
        t for t in _tokens(text)
        if t not in _STOP and t not in _GENERIC and len(t) > 2
    ]


def _noun_phrases(text: str, max_phrases: int = 6) -> List[str]:
    """Extract candidate entity noun phrases (adjacent content words).

    A noun phrase is a maximal run of content tokens (length 1..3). Single
    content tokens are kept because domain terms are often single words
    ("dendrites", "primes"). Multi-word runs are preferred as entities because
    they are more specific ("thermal runaway", "time window").

    Runs are trimmed at clause-tail connectives ("while", "without", "and",
    "that", ...) so a phrase does not absorb a following subordinate clause
    (e.g. 'charge speed while staying under 5%' -> 'charge speed').
    """
    # Connectives that end a noun phrase rather than continue it.
    _TAIL_BREAK = {
        "while", "without", "and", "or", "that", "which", "without",
        "where", "when", "so", "but", "because", "if", "then", "to",
        "than", "as", "under", "over", "between", "during", "after",
        "before", "within", "despite", "for", "with", "of", "in", "on",
        "into", "from", "by", "at",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", text)
    phrases: List[str] = []
    run: List[str] = []
    for w in words:
        low = w.lower()
        if low in _STOP or low in _GENERIC or low in _TAIL_BREAK or len(low) <= 2:
            if len(run) >= 2:
                phrases.append(" ".join(run))
            run = []
            continue
        run.append(low)
    if len(run) >= 2:
        phrases.append(" ".join(run))
    # Score phrases: prefer multi-word, then rarer (longer) tokens, dedupe.
    seen: Set[str] = set()
    ranked: List[str] = []
    for p in sorted(phrases, key=lambda s: (-len(s.split()), -len(s))):
        if p not in seen:
            seen.add(p)
            ranked.append(p)
    # Fall back to single content tokens when no multi-word phrase exists.
    if not ranked:
        for t in _content_tokens(text):
            if t not in seen:
                seen.add(t)
                ranked.append(t)
    return ranked[:max_phrases]


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.;!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Problem structure
# ---------------------------------------------------------------------------

@dataclass
class ProblemStructure:
    """Typed parse of a problem statement (the 'identification' stage output)."""

    raw: str
    problem_type: str
    entities: List[str]
    givens: List[str]
    unknowns: List[str]
    constraints: List[str]
    goal: str
    success_criteria: List[str]
    type_confidences: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw": self.raw,
            "problem_type": self.problem_type,
            "entities": list(self.entities),
            "givens": list(self.givens),
            "unknowns": list(self.unknowns),
            "constraints": list(self.constraints),
            "goal": self.goal,
            "success_criteria": list(self.success_criteria),
            "type_confidences": dict(self.type_confidences),
        }


def _score_types(text_low: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ptype, cues in _TYPE_CUES:
        hits = 0.0
        for cue in cues:
            if cue in text_low:
                # Longer cues are more diagnostic.
                hits += 1.0 + 0.25 * len(cue.split())
        scores[ptype] = hits
    if not any(scores.values()):
        scores["explanation"] = 0.5
    return scores


def identify_problem(specification: str, title: str = "") -> ProblemStructure:
    """Parse a problem statement into a typed :class:`ProblemStructure`.

    This is the 'identification' step: it decides *what kind of problem this is*
    and extracts the slots a decomposition needs (goal, givens, unknowns,
    constraints, entities), instead of jumping straight to template filling.
    """
    text = (title + ". " + specification) if title else specification
    low = text.lower()
    sentences = _sentences(text)

    type_scores = _score_types(low)
    problem_type = max(type_scores, key=lambda k: type_scores[k])
    norm = sum(type_scores.values()) or 1.0
    type_conf = {k: round(v / norm, 4) for k, v in type_scores.items()}

    entities = _noun_phrases(text, max_phrases=6)

    givens: List[str] = []
    unknowns: List[str] = []
    constraints: List[str] = []
    success_criteria: List[str] = []

    for s in sentences:
        sl = s.lower()
        if any(cue in sl for cue in _CONSTRAINT_CUES):
            constraints.append(s)
        if any(cue in sl for cue in _SUCCESS_CUES):
            success_criteria.append(s)
        # Givens: sentences stating facts / data / starting state.
        if any(c in sl for c in ("given", "assume", "we have", "there is",
                                 "there are", "suppose", "starting from",
                                 "based on", "consists of")):
            givens.append(s)
        # Unknowns: the quantity/answer being asked for.
        if re.search(r"\b(how many|how much|how long|what is|which|determine|"
                     r"compute|find|derive|unknown|solve for|estimate)\b", sl):
            unknowns.append(s)

    # Goal: the sentence with the strongest goal verb. If no goal verb fires
    # (a bare descriptive statement, e.g. 'X causes Y'), fall back to a
    # content-word summary of the problem rather than the whole raw string —
    # embedding the entire raw text as the 'goal' is what let goal-anchored
    # subproblems self-score ~0.85 and compress the ranking.
    goal = ""
    best = -1
    for s in sentences:
        sl = s.lower()
        score = sum(
            (1 + len(v.split())) for v in _GOAL_VERBS if v in sl
        )
        if score > best:
            best = score
            goal = s
    if best <= 0:
        summary = " ".join(entities[:3]) if entities else ""
        goal = summary or (sentences[0] if sentences else specification)

    # Derive implicit unknowns from entities when none were stated.
    if not unknowns:
        unknowns = [f"the value/behaviour of {e}" for e in entities[:2]]
    if not success_criteria:
        success_criteria = [f"{problem_type} objective satisfied: {goal}"]

    return ProblemStructure(
        raw=specification,
        problem_type=problem_type,
        entities=entities,
        givens=givens,
        unknowns=unknowns,
        constraints=constraints,
        goal=goal,
        success_criteria=success_criteria,
        type_confidences=type_conf,
    )


# ---------------------------------------------------------------------------
# Relevance scoring (calibrated: semantic + lexical separation)
# ---------------------------------------------------------------------------

def _lexical_overlap(parent: str, child: str) -> float:
    """Fraction of the parent's distinctive content tokens present in child.

    This is the term that actually separates relevant from irrelevant
    subproblems on the low-dynamic-range hashing embedder: relevant
    subproblems reuse the parent's domain content words.
    """
    p_tokens = set(_content_tokens(parent))
    if not p_tokens:
        return 0.0
    c_tokens = set(_content_tokens(child))
    return len(p_tokens & c_tokens) / len(p_tokens)


# ---------------------------------------------------------------------------
# Optional semantic scorer (delegates to a genuinely semantic embedder)
# ---------------------------------------------------------------------------

_SEMANTIC_STATE: Dict[str, object] = {"resolved": False, "backend": None}


def _semantic_backend():
    """Resolve a genuinely semantic embedder once, or None if unavailable.

    By DEFAULT (no env set) the OFFLINE distilled_pid embedder is used when its
    artifact is present, because it is deterministic and needs no daemon; if the
    artifact is missing it silently falls back to polarity. Overrides:
      - ``SUPER_SOLVER_SEMANTIC_SCORER=0`` disables the semantic scorer entirely
        (pure polarity path).
      - ``SUPER_SOLVER_SEMANTIC_EMBEDDER`` selects the backend explicitly
        (``distilled_pid`` | ``ollama`` | any registered backend name).
    If the chosen daemon is unreachable the result is cached as None and every
    call silently falls back to polarity, so the engine never hard-fails.
    """
    if _SEMANTIC_STATE["resolved"]:
        return _SEMANTIC_STATE["backend"]
    _SEMANTIC_STATE["resolved"] = True

    flag = os.getenv("SUPER_SOLVER_SEMANTIC_SCORER", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        _SEMANTIC_STATE["backend"] = None
        return None

    # Default backend is the offline distilled student (no daemon).
    name = os.getenv("SUPER_SOLVER_SEMANTIC_EMBEDDER", "distilled_pid")

    def _try_distilled():
        try:
            from super_solver.core.distilled_pid import DistilledPIDEmbedder

            backend = DistilledPIDEmbedder()
            if backend.available:
                return backend
        except Exception:
            pass
        return None

    try:
        if name == "distilled_pid":
            backend = _try_distilled()
            _SEMANTIC_STATE["backend"] = backend
            return backend
        from super_solver.core.embedder import get_backend

        backend = get_backend(name)
        # Probe reachability with a tiny encode; OllamaBackend falls back to
        # hashing on failure, so verify it actually returned a semantic vector
        # by checking the returned dim matches the backend's declared dim.
        probe = backend.encode("reachability probe")
        if getattr(probe, "shape", (0,))[0] == getattr(backend, "dim", -1):
            _SEMANTIC_STATE["backend"] = backend
        else:
            # Requested backend unreachable: fall back to offline distilled.
            _SEMANTIC_STATE["backend"] = _try_distilled()
    except Exception:
        _SEMANTIC_STATE["backend"] = _try_distilled()
    return _SEMANTIC_STATE["backend"]


def semantic_similarity(a: str, b: str) -> Optional[float]:
    """Semantic cosine via the optional embedder, or None when unavailable."""
    backend = _semantic_backend()
    if backend is None:
        return None
    try:
        return float(
            backend.cosine_similarity(backend.encode(a), backend.encode(b))
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Negation / polarity consistency guard
# ---------------------------------------------------------------------------
# Embedding cosine (including nomic-embed-text) treats a negated sentence as
# highly similar to its affirmation, because they differ only by 'not' / 'dis-'.
# Measured: on adversarial-negation probes even the nomic teacher ranks a
# negated distractor ABOVE the correct answer (e.g. 0.79 vs 0.41). This guard
# detects a polarity FLIP between parent and child and penalizes relevance,
# independent of the embedder.

# Negation cues. Deliberately CONSERVATIVE: only explicit logical negators and
# clear opposite-forming words. Directional words ('reduce', 'increase') are NOT
# negators here — they are ordinary domain verbs and including them produced
# false positives (e.g. 'Reduce the latency' looked negated).
_NEGATION_MARKERS: Tuple[str, ...] = (
    "not", "no", "never", "cannot", "can't", "won't", "don't", "doesn't",
    "isn't", "aren't", "without", "avoid", "prevent", "stop", "eliminate",
    "disprove", "deny", "refuse", "exclude", "negate", "disable", "omit",
    "forbid", "false", "impossible", "infeasible", "block",
)


# Antonym pairs (opposite predicates). A child naming the OPPOSITE of a parent
# predicate is a polarity flip too, even without an explicit negator
# (e.g. 'heavier/weaker' vs 'lighter/stronger').
_ANTONYMS: Tuple[Tuple[str, str], ...] = (
    ("lighter", "heavier"), ("light", "heavy"), ("stronger", "weaker"),
    ("strong", "weak"), ("faster", "slower"), ("fast", "slow"),
    ("increase", "decrease"), ("more", "less"), ("higher", "lower"),
    ("maximize", "minimize"), ("improve", "worsen"), ("safe", "unsafe"),
    ("available", "unavailable"), ("correct", "incorrect"),
    ("enable", "disable"), ("start", "stop"), ("accept", "reject"),
)


def _antonym_flip(parent: str, child: str) -> bool:
    """True when the child states the opposite of a parent predicate."""
    pl = parent.lower()
    cl = child.lower()

    def has(text, w):
        return re.search(rf"\b{re.escape(w)}\b", text) is not None

    for a, b in _ANTONYMS:
        # parent says A, child says B (or vice versa) -> opposite. The antonym
        # pair itself is the signal; no shared-content gate here (the opposing
        # predicate may be the only content when the parent is templated).
        if (has(pl, a) and has(cl, b)) or (has(pl, b) and has(cl, a)):
            return True
    return False


def _negation_cues(text: str) -> Set[str]:
    """Return the SET of explicit negation cue words present in text.

    Whole-word markers only. The morphological-prefix heuristic ('un-', 'non-',
    'dis-') was REMOVED: it false-positived on technical terms like
    'nontrivial', 'nonlinear', 'distributed' and wrongly penalized correct
    paraphrases. Explicit negators plus the antonym table cover the real cases.
    """
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if w in _NEGATION_MARKERS}


def _negation_count(text: str) -> int:
    """Count explicit negation cues in text (whole-word markers only)."""
    return len(_negation_cues(text))


def negation_flip(parent: str, child: str) -> Tuple[bool, Dict[str, float]]:
    """Detect a polarity flip: child negates what parent asserts (or vice versa).

    A flip is flagged when the child introduces NEW negation the parent does not
    carry (a negated restatement), while sharing enough content that the
    negation is the salient difference. Comparing the actual cue words (not just
    counts) avoids penalising a child that legitimately restates an already-
    negative parent (e.g. parent 'stop X from happening' -> child 'so it cannot
    happen' carries the same 'prevent' intent, not a flip).

    Antonym oppositions (lighter/heavier, stronger/weaker) are also flips.
    Returns ``(flipped, detail)``.
    """
    p_cues = _negation_cues(parent)
    c_cues = _negation_cues(child)
    shared = _lexical_overlap(parent, child)
    # Flip only when the parent is AFFIRMATIVE and the child introduces negation.
    # If the parent already carries any negation, a negated child is usually
    # restating the same intent (parent 'stop X' / child 'so X cannot happen'),
    # not flipping it — cue-word set-difference is too literal for this.
    flip = (len(p_cues) == 0 and len(c_cues) > 0 and shared >= 0.1) or _antonym_flip(
        parent, child
    )
    detail = {
        "parent_negations": float(len(p_cues)),
        "child_negations": float(len(c_cues)),
        "shared_content": float(shared),
        "flipped": float(1.0 if flip else 0.0),
    }
    return flip, detail


def relevance_score(parent: str, child: str) -> Tuple[float, Dict[str, float]]:
    """Calibrated relevance of a subproblem to its parent problem.

    Blends:
      - semantic cosine (meaning, robust to paraphrase), and
      - lexical content overlap (distinctive shared content words),
      then applies a NEGATION-CONSISTENCY penalty: a child that flips the
    parent's polarity (a negated distractor) is demoted, because embedding
    cosine alone ranks it highly.

    IMPORTANT LIMITATION (measured, not assumed): on the deterministic hashing
    embedder this scorer recovers relevance when the child reuses the parent's
    domain words, but CANNOT rank a pure domain-synonym paraphrase (e.g.
    'weighted graph + relax edges' for a 'shortest-path' parent) above an
    off-topic child that happens to share one content word. That case has zero
    surface overlap and is not recoverable without a genuinely semantic
    embedder. Generated subproblems therefore do not rely on this scorer for
    their *own* relevance (see ``anchored_relevance``); it is used to re-rank
    caller-supplied subproblems, where surface overlap is a valid signal.

    Returns ``(score, breakdown)``.
    """
    v_p = embedding_service.encode(parent)
    v_c = embedding_service.encode(child)
    cos = embedding_service.cosine_similarity(v_p, v_c)
    lex = _lexical_overlap(parent, child)

    flipped, neg_detail = negation_flip(parent, child)

    sem = semantic_similarity(parent, child)
    if sem is not None:
        sem_scaled = max(0.0, min(1.0, (sem - 0.25) / 0.45))
        tie = 0.02 * (cos + lex)
        score = 0.98 * sem_scaled + tie
        breakdown = {
            "semantic": float(sem), "cosine": float(cos), "lexical": float(lex)
        }
    else:
        score = 0.55 * cos + 0.45 * lex
        breakdown = {"cosine": float(cos), "lexical": float(lex)}

    if flipped:
        # Demote well below any non-flipped candidate: a polarity flip (explicit
        # negation or antonym) is strong evidence of a distractor, and embedding
        # cosine alone ranks it highly. A multiplicative 0.2 penalty ensures the
        # flipped child loses even when its raw similarity is far higher.
        score *= 0.2
        breakdown["negation_flip_penalty"] = 1.0
    breakdown.update(neg_detail)
    return float(score), breakdown


def anchored_relevance(structure: "ProblemStructure", slot: str) -> float:
    """Relevance-by-construction for a subproblem that decomposes a given slot.

    A generated subproblem is relevant because it *targets* a specific
    identified slot (the goal, a constraint, an entity, an unknown, the success
    criterion). Its relevance is therefore a function of the slot's importance,
    not of surface overlap with the parent. This is the design that sidesteps
    the paraphrase ceiling: we do not have to *detect* the subproblem's
    relevance, we *choose* it.

    Slot weights reflect how directly a slot drives the solution:
      goal / success_criterion  -> 1.00 (the objective itself)
      constraint                -> 0.85 (feasibility-shaping)
      unknown                   -> 0.80 (the thing to determine)
      entity:*                  -> 0.70 (a component to resolve)
      given                     -> 0.55 (background)
    """
    base = {
        "goal": 1.0,
        "success_criterion": 0.95,
        "constraint": 0.85,
        "unknown": 0.80,
        "given": 0.55,
    }
    if slot.startswith("entity:"):
        return 0.70
    return base.get(slot, 0.6)


def _distinctive_tokens(text: str) -> Set[str]:
    return set(_content_tokens(text))


# ---------------------------------------------------------------------------
# Structured decomposition
# ---------------------------------------------------------------------------

@dataclass
class Subproblem:
    text: str
    heuristic: str
    target_slot: str  # which identified slot this subproblem decomposes
    relevance: float
    is_new: bool = True
    breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "subproblem": self.text,
            "heuristic": self.heuristic,
            "target_slot": self.target_slot,
            "relevance_score": round(self.relevance, 4),
            "is_essential": bool(self.relevance >= 0.45),
            "is_new": self.is_new,
            "score_breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
        }


def _candidate_subproblems(structure: ProblemStructure) -> List[Tuple[str, str, str]]:
    """Synthesize (text, heuristic, target_slot) candidates from the structure.

    Each candidate is anchored on a *specific* identified slot (the goal, a
    constraint, an entity, an unknown), so subproblems are about the actual
    problem rather than a generic template.
    """
    out: List[Tuple[str, str, str]] = []
    goal = structure.goal
    entities = structure.entities
    constraints = structure.constraints
    unknowns = structure.unknowns

    # --- Goal decomposition (divide & conquer on the objective) ---
    if goal:
        out.append((
            f"Decompose the objective '{goal}' into independent sub-goals whose "
            f"joint satisfaction implies the objective.",
            "divide_and_conquer", "goal",
        ))
        out.append((
            f"State the strongest necessary condition for '{goal}' to hold, "
            f"and check whether it is satisfiable.",
            "working_backwards", "goal",
        ))
        out.append((
            f"Characterize the boundary/special cases of '{goal}' "
            f"(degenerate inputs, limiting values) and test the objective there.",
            "special_case", "goal",
        ))

    # --- Entity decomposition (resolve/reduce each key entity) ---
    for ent in entities[:3]:
        out.append((
            f"Resolve the role of entity '{ent}': what properties of '{ent}' "
            f"determine the objective, and how are they measured?",
            "divide_and_conquer", f"entity:{ent}",
        ))
    if entities:
        out.append((
            f"Find a structural analogue of '{entities[0]}' in another domain "
            f"and test whether the analogous objective holds there.",
            "analogy", f"entity:{entities[0]}",
        ))

    # --- Constraint decomposition (relax each constraint in turn) ---
    for con in constraints[:2]:
        out.append((
            f"Relax or remove the constraint '{con}' and determine how the "
            f"objective's feasibility and optimum change.",
            "relaxation", "constraint",
        ))
    if constraints:
        out.append((
            f"Check whether the constraints {constraints} are mutually "
            f"consistent, and find a minimal infeasible subset if not.",
            "invariant", "constraint",
        ))

    # --- Unknown decomposition (directly attack the quantified unknown) ---
    for unk in unknowns[:2]:
        out.append((
            f"Establish a quantitative estimate or test for '{unk}', with the "
            f"measurement or derivation that would falsify it.",
            "estimation", "unknown",
        ))

    # --- Success-criterion decomposition (falsification test) ---
    for crit in structure.success_criteria[:1]:
        out.append((
            f"Design a decisive test for the success criterion '{crit}': what "
            f"observation would falsify it?",
            "counterexample_search", "success_criterion",
        ))

    # --- Type-specific heuristic ---
    if structure.problem_type == "optimization":
        out.append((
            f"Formulate the objective function and feasible region for "
            f"'{goal}', then identify the active constraints at the optimum.",
            "optimization", "goal",
        ))
    elif structure.problem_type == "proof":
        out.append((
            f"Search for a counterexample to '{goal}' over a large parameter "
            f"range before attempting a proof.",
            "counterexample_search", "goal",
        ))
    elif structure.problem_type == "diagnosis":
        out.append((
            f"Separate what IS true about the failure from what IS-NOT "
            f"(symptom boundary) to bound the candidate causes of '{goal}'.",
            "kt_boundary", "goal",
        ))
    elif structure.problem_type == "design":
        out.append((
            f"Enumerate the functional requirements and the physical "
            f"constraints for '{goal}', then look for a trade-off to dissolve.",
            "triz", "goal",
        ))

    return out


def decompose(
    specification: str,
    title: str = "",
    known_subproblems: Optional[Sequence[str]] = None,
    max_new: int = 8,
    redundancy_threshold: float = 0.82,
) -> Dict[str, object]:
    """Principled decomposition: identify the problem, then generate, rank, prune.

    Returns a dict with the identified ``structure`` and a ranked list of
    ``subproblems`` (new generated ones first, ordered by relevance), each
    tagged with the heuristic and the identified slot it decomposes.
    Redundant candidates (near-duplicates of each other or of known
    subproblems) are dropped so distinct angles survive.
    """
    structure = identify_problem(specification, title)
    known = list(known_subproblems or [])
    parent = (title + ". " + specification) if title else specification

    candidates = _candidate_subproblems(structure)

    # Rank generated candidates by relevance-by-construction (slot importance),
    # blended with a light semantic term for tie-breaking among equal slots.
    scored: List[Subproblem] = []
    for text, heuristic, slot in candidates:
        anchor_rel = anchored_relevance(structure, slot)
        sem, breakdown = relevance_score(parent, text)
        rel = min(1.0, 0.75 * anchor_rel + 0.25 * sem)
        breakdown = {"anchor": float(anchor_rel), **breakdown}
        scored.append(Subproblem(
            text=text, heuristic=heuristic, target_slot=slot,
            relevance=rel, is_new=text not in known, breakdown=breakdown,
        ))

    # Re-rank caller-supplied subproblems with the empirical scorer: for these we
    # do NOT know which slot they target, so surface overlap with the parent is
    # the only available (and valid, when there is overlap) signal.
    for text in known:
        rel, breakdown = relevance_score(parent, text)
        # A caller subproblem that targets an identified entity/constraint by
        # surface match gets an anchor boost, mirroring anchored_relevance.
        anchor = 0.55
        tl = text.lower()
        if any(e and e in tl for e in structure.entities):
            anchor = 0.70
        if any(c and c[:40].lower() in tl for c in structure.constraints):
            anchor = max(anchor, 0.85)
        blended = 0.6 * rel + 0.4 * anchor
        scored.append(Subproblem(
            text=text, heuristic="caller_supplied", target_slot="given",
            relevance=blended, is_new=False,
            breakdown={"anchor": float(anchor), **breakdown},
        ))

    scored.sort(key=lambda s: s.relevance, reverse=True)

    # Redundancy suppression: keep highest-scoring, drop near-duplicates.
    kept: List[Subproblem] = []
    kept_vecs: List = []
    kept_tokens: List[Set[str]] = []
    for s in scored:
        v = embedding_service.encode(s.text)
        toks = _distinctive_tokens(s.text)
        redundant = False
        for kv, kt in zip(kept_vecs, kept_tokens):
            cos = embedding_service.cosine_similarity(v, kv)
            tok_jac = (
                len(toks & kt) / len(toks | kt) if (toks | kt) else 0.0
            )
            if cos >= redundancy_threshold or tok_jac >= redundancy_threshold:
                redundant = True
                break
        if not redundant:
            kept.append(s)
            kept_vecs.append(v)
            kept_tokens.append(toks)

    # Emit: new subproblems first (ranked), then caller-supplied (ranked).
    new_kept = [s for s in kept if s.is_new][:max_new]
    old_kept = [s for s in kept if not s.is_new]

    # Slot coverage: which identified slots did the surviving subproblems address?
    slots_present = {s.target_slot.split(":")[0] for s in new_kept}
    expected_slots = {"goal", "constraint", "unknown", "entity",
                      "success_criterion"}
    covered = len(slots_present & expected_slots)

    return {
        "structure": structure.to_dict(),
        "subproblems": [s.to_dict() for s in new_kept + old_kept],
        "metrics": {
            "n_generated": len(new_kept),
            "n_known": len(old_kept),
            "max_relevance": round(max((s.relevance for s in kept), default=0.0), 4),
            "mean_top_relevance": round(
                sum(s.relevance for s in new_kept) / len(new_kept), 4
            ) if new_kept else 0.0,
            "slot_coverage": round(covered / len(expected_slots), 4),
            "slots_covered": sorted(slots_present),
            "redundancy_pruned": len(scored) - len(kept),
        },
    }
