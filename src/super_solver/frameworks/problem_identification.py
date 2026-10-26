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
    "take", "want", "need", "help", "improve", "increase", "decrease",
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

# Problem-type cue lexicons. Cues are matched as substrings against the
# lowercased problem text; longer/multi-word cues score higher (see
# _score_types). Lexicons are deliberately broad on the COMMON directive verbs
# ('reduce', 'improve', 'keep', 'faster') because those were the main source of
# type-identification misses (directive problems defaulting to 'explanation').
_TYPE_CUES: List[Tuple[str, Tuple[str, ...]]] = [
    ("optimization", (
        "optimize", "optimi", "maximize", "minimize", "maximum", "minimum",
        "best", "cheapest", "fastest", "shortest", "least", "largest",
        "smallest", "trade-off", "tradeoff", "efficient", "efficiency",
        "reduce", "increase", "minimize", "faster", "slower", "cheaper",
        "cut ", "lower ", "higher ", "save ", "speed up", "scale",
        "without increasing", "without degrading", "at the lowest",
        "maximize", "balance", "trade off", "improve", "lose", "gain",
        "shorten", "boost",
    )),
    ("proof", (
        "prove", "theorem", "lemma", "show that", "derive", "qed",
        "rigorously", "identity", "conjecture", "holds for all",
        "for every", "for all", "irrational", "diverges", "converges",
        "n-1", "n squared", "is prime", "infinitely many",
    )),
    ("search", (
        "search", "find", "locate", "retrieve", "look up", "route",
        "discover", "explore", "which of", "identify the", "where is",
        "shortest path", "detect",
    )),
    ("estimation", (
        "estimate", "predict", "forecast", "measure", "quantify", "how long",
        "how much", "how many", "approximate", "rate of", "projection",
        "project ", "extrapolate",
    )),
    ("design", (
        "design", "build", "construct", "synthesize", "create", "engineer",
        "architecture", "blueprint", "device", "system that", "implement",
        "keep ", "ensure ", "guarantee ", "prevent ", "maintain ",
        "make a", "make the", "scheme", "protocol", "mechanism to",
        "survive", "withstand", "resilient", "fault-tolerant",
    )),
    ("diagnosis", (
        "diagnose", "debug", "root cause", "why does", "failing", "failure",
        "broken", "bug", "error", "fault", "intermittent", "regression",
        "only happens", "causes", "caused by", "leads to", "results in",
        "thermal runaway", "degradation", "corrosion", "defect", "crash",
        "stampede", "leak", "outage", "spike", "degrade",
    )),
    ("explanation", (
        "explain", "why", "understand", "elucidate", "interpret", "account for",
        "mechanism", "reason", "how does", "how do", "what causes",
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
        "while", "without", "and", "or", "that", "which", "where", "when", "so", "but", "because", "if", "then", "to",
        "than", "as", "under", "over", "between", "during", "after",
        "before", "within", "despite", "for", "with", "of", "in", "on",
        "into", "from", "by", "at",
    }
    # Common directive verbs. A noun phrase must not run ACROSS a verb: breaking
    # runs at verbs stops merges like 'preserve patient records coordinate staff
    # across wards' becoming one bogus entity.
    _VERB_BREAK = {
        "preserve", "coordinate", "keep", "reduce", "increase", "make",
        "build", "design", "ensure", "guarantee", "prevent", "maintain",
        "avoid", "achieve", "improve", "optimize", "detect", "explain",
        "prove", "find", "estimate", "predict", "route", "scale",
        "speed", "lose", "writes", "write", "storing", "store", "handle",
        # symptom / analysis verbs that otherwise contaminate entities
        "slows", "slowed", "slowing", "overheats", "overheated", "overheating",
        "flooding", "flooded", "isolate", "diagnose", "diagnosing", "fails",
        "failing", "failed", "causes", "caused", "causing", "grows", "growing",
        "drops", "dropping", "rises", "rising", "leaks", "leaking", "runs",
        "running", "shows", "showing", "occurs", "occurring", "happens",
        "spikes", "spiking", "degrades", "degrading", "recovers", "recovering",
    }
    _MAX_RUN = 3  # a noun phrase is at most 3 words
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", text)
    phrases: List[str] = []
    run: List[str] = []
    for w in words:
        low = w.lower()
        if (
            low in _STOP or low in _GENERIC or low in _TAIL_BREAK
            or low in _VERB_BREAK or len(low) <= 2
        ):
            if len(run) >= 2:
                phrases.append(" ".join(run))
            run = []
            continue
        run.append(low)
        if len(run) >= _MAX_RUN:
            phrases.append(" ".join(run))
            run = []
    if len(run) >= 2:
        phrases.append(" ".join(run))
    # Score phrases: prefer multi-word, then rarer (longer) tokens, dedupe.
    seen: Set[str] = set()
    ranked: List[str] = []
    for p in sorted(phrases, key=lambda s: (-len(s.split()), -len(s))):
        if p not in seen:
            seen.add(p)
            ranked.append(p)
    # Fall back to single content tokens when no multi-word phrase exists,
    # EXCLUDING break-verbs (a symptom verb like 'overheats' is not an entity).
    if not ranked:
        for t in _content_tokens(text):
            if t not in seen and t not in _VERB_BREAK:
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
    objectives: List[str] = field(default_factory=list)
    is_vague: bool = False

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
            "objectives": list(self.objectives),
            "is_vague": bool(self.is_vague),
            "multi_goal": len(self.objectives) >= 2,
        }


def _score_types(text_low: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ptype, cues in _TYPE_CUES:
        hits = 0.0
        for cue in cues:
            # Word-boundary match for whole-word cues, to stop substring false
            # positives like 'prove' matching inside 'improve'. Cues that are
            # deliberate stems (end with a fragment such as 'optimi') or multi-
            # word phrases keep plain substring matching.
            if cue.endswith(" ") or " " in cue.strip() or cue.endswith(
                ("imi", "iz", "at", "ur")
            ):
                matched = cue in text_low
            else:
                matched = re.search(rf"\b{re.escape(cue)}\b", text_low) is not None
            if matched:
                # Longer cues are more diagnostic.
                hits += 1.0 + 0.25 * len(cue.split())
        scores[ptype] = hits
    if not any(scores.values()):
        # No cue fired: a bare directive ('make it better') is most often an
        # optimization/design ask; a bare question is an explanation. Prefer a
        # directive default over 'explanation' so vague asks stay actionable.
        stripped = text_low.strip()
        if stripped.endswith("?") or re.match(r"\s*(why|how|what)\b", stripped):
            scores["explanation"] = 0.5
        else:
            scores["optimization"] = 0.4
    return scores


def _split_objectives(text: str) -> List[str]:
    """Split a multi-objective goal into its component objectives.

    Detects conjunction-joined directives such as 'Minimize latency AND cost AND
    carbon footprint' or 'Detect fraud, reduce false positives, and keep the
    model explainable' and returns each objective as a standalone clause. This
    is what makes a multi-goal problem yield ONE decomposition per objective
    instead of a single mushy goal.

    Returns [] when the problem has a single objective (nothing to split).
    """
    # Split on coordinating conjunctions that join clauses/objectives.
    # Protect the common 'X and Y' noun pairs by requiring a directive verb or a
    # comma to justify a split.
    _DIRECTIVE = (
        "minimize", "maximize", "reduce", "increase", "improve", "detect",
        "keep", "ensure", "lower", "raise", "cut", "speed", "shorten",
        "boost", "eliminate", "prevent", "avoid", "maintain", "preserve",
        "optimize", "balance", "raise", "grow", "shrink", "make", "build",
        "design", "achieve", "guarantee", "support", "handle",
    )
    low = text.lower()
    # 'X and Y' inside an explicit trade-off / comparison frame is ONE objective,
    # not two: 'optimize the trade-off between energy density and charge speed'.
    if re.search(r"\b(trade[- ]?off|tradeoff|balance|between|versus|vs|compare)\b", low):
        return []
    # Normalize separators: ', and ' / ' and ' / '; ' / ' plus '
    parts = re.split(r"\s*(?:,\s*and\s+|\s+and\s+|;\s*|\s+plus\s+)\s*", low)
    parts = [p.strip(" .") for p in parts if p.strip(" .")]
    if len(parts) < 2:
        return []
    # Count parts that contain a directive verb.
    directive_parts = [p for p in parts if any(v in p for v in _DIRECTIVE)]
    if len(directive_parts) >= 2:
        return directive_parts
    # Single directive + >=2 bare conjuncts (e.g. 'Minimize latency and cost and
    # carbon footprint'): the bare conjuncts are still objectives of a
    # multi-objective problem, so return the directive plus the bare measures
    # that look like quantities (short noun phrases, not clauses).
    if len(directive_parts) == 1 and len(parts) >= 2:
        bare = [p for p in parts if p not in directive_parts]
        # Accept the directive plus the remaining conjuncts as objectives when
        # at least one bare measure exists. A trailing qualifier ('... for a
        # global API') is fine; we keep it as part of the last objective.
        if bare:
            return directive_parts + bare
    return []


def _is_vague(text: str, entities: List[str]) -> bool:
    """Heuristic: a problem is ill-posed when it names few/no concrete entities
    and is very short, or uses only placeholder words.

    Vague problems should be flagged so the pipeline emits CLARIFICATION angles
    instead of pretending to decompose a phantom objective.
    """
    tokens = _content_tokens(text)
    # Placeholder / non-referential words that carry no problem content.
    _PLACEHOLDER = {
        "stuff", "things", "everything", "something", "anything", "it",
        "better", "worse", "good", "bad", "nice", "improve", "fix",
    }
    substantive = [t for t in tokens if t not in _PLACEHOLDER]
    concrete_entities = [e for e in entities if len(e.split()) >= 1]
    if len(substantive) <= 2:
        return True
    if not concrete_entities and len(substantive) <= 4:
        return True
    # Only-placeholder content (e.g. 'make it better', 'fix everything').
    if substantive and all(t in _PLACEHOLDER for t in substantive):
        return True
    return False


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
        objectives=_split_objectives(goal or specification),
        is_vague=_is_vague(specification or title, entities),
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


def _antonym_variant_pattern(word: str) -> str:
    """Regex for an antonym word plus its common inflections.

    A plain ``\\bincrease\\b`` misses "increases"/"increasing", so a hypothesis
    that asserts a predicate and its opposite in inflected form ("Increasing the
    dose reduces survival") slipped past the self-contradiction check while the
    uninflected wording was caught. Both stems are allowed because English drops
    a trailing "e" before -ing/-ed ("increase" -> "increas").
    """
    escaped = re.escape(word)
    stems = [escaped]
    if word.endswith("e"):
        stems.append(re.escape(word[:-1]))
    return rf"\b(?:{'|'.join(stems)})(?:s|es|d|ed|ing)?\b"


def _antonym_flip(parent: str, child: str) -> bool:
    """True when the child states the opposite of a parent predicate."""
    pl = parent.lower()
    cl = child.lower()

    def has(text, w):
        return re.search(_antonym_variant_pattern(w), text) is not None

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
        # Clarification angles gate a vague problem: everything downstream is
        # meaningless until the objective is pinned down, so they rank first.
        "clarification": 1.0,
    }
    if slot.startswith("entity:"):
        return 0.70
    if slot.startswith("objective:"):
        # Each objective of a multi-goal problem is as important as the whole
        # goal: decomposing them separately is the point.
        return 1.0
    return base.get(slot, 0.6)


def _distinctive_tokens(text: str) -> Set[str]:
    return set(_content_tokens(text))# ---------------------------------------------------------------------------
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

    # --- Ill-posed handling: emit CLARIFICATION angles instead of pretending
    #     to decompose a phantom objective. These come first (they gate
    #     everything else), and are relevance-anchored to the goal slot.
    if structure.is_vague:
        out.append((
            "Clarify the objective: what observable outcome defines success, "
            "and against what baseline is 'better' measured?",
            "clarification", "goal",
        ))
        out.append((
            "Identify the concrete system, artifact, or quantity the problem "
            "is actually about; name the specific components involved.",
            "clarification", "entity",
        ))
        out.append((
            "Surface the hidden constraints: budget, deadline, and the "
            "trade-off that must not be sacrificed.",
            "clarification", "constraint",
        ))
        out.append((
            "Define a measurable baseline and target so progress can be "
            "evaluated and falsified.",
            "clarification", "success_criterion",
        ))

    # --- Multi-goal handling: one objective-level decomposition per objective,
    #     so a multi-objective problem yields one angle per objective instead
    #     of a single mushy goal. Skipped for vague problems, where the
    #     'objectives' are just placeholder noise.
    if not structure.is_vague:
        for obj in structure.objectives:
            out.append((
                f"Optimize objective '{obj}' on its own: characterize its own "
                f"optimum ignoring the other objectives, as a baseline.",
                "multi_objective", f"objective:{obj}",
            ))
        if len(structure.objectives) >= 2:
            out.append((
                "Build the Pareto frontier across the objectives "
                f"({', '.join(structure.objectives[:4])}): enumerate non-dominated "
                "candidates and identify the knee/compromise point and the "
                "objective pairs that genuinely trade off.",
                "pareto", "objective:pareto_frontier",
            ))

    # --- Goal decomposition (divide & conquer on the objective) ---
    # Skipped for vague problems: echoing an undefined objective ('Make it
    # better.') produces spurious subproblems. Clarification angles above do the
    # real work until the objective is pinned down.
    if goal and not structure.is_vague:
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

    # --- Type-specific heuristic (skipped when vague: a type-specific angle
    #     on an undefined objective just echoes the vagueness) ---
    if structure.is_vague:
        return out
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
    # Clarification angles (vague problems) and per-objective angles (multi-goal
    # problems) get a small PRIORITY boost so they sort above the generic goal
    # angles they are meant to gate/expand — otherwise the goal angles, which
    # embed the goal text verbatim, win the semantic tie-break and crowd them
    # out of the top-k.
    _PRIORITY_SLOTS = {"clarification", "objective"}
    scored: List[Subproblem] = []
    for text, heuristic, slot in candidates:
        anchor_rel = anchored_relevance(structure, slot)
        sem, breakdown = relevance_score(parent, text)
        kind = slot.split(":")[0]
        boost = 0.06 if kind in _PRIORITY_SLOTS else 0.0
        rel = min(1.0, 0.75 * anchor_rel + 0.25 * sem + boost)
        breakdown = {"anchor": float(anchor_rel), "priority_boost": boost, **breakdown}
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
        for kv, kt in zip(kept_vecs, kept_tokens, strict=False):
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

    # Slot-balanced selection: reserve at least one subproblem per identified
    # non-empty slot BEFORE filling the remaining budget by relevance, so a
    # long multi-goal problem cannot crowd out the entity/unknown/constraint
    # angles. This directly raises slot coverage.
    new_all = [s for s in kept if s.is_new]
    old_kept = [s for s in kept if not s.is_new]

    present_slot_kinds = []
    for kind in ("goal", "constraint", "unknown", "entity", "success_criterion",
                 "objective"):
        if any(s.target_slot.split(":")[0] == kind for s in new_all):
            present_slot_kinds.append(kind)

    selected: List[Subproblem] = []
    selected_ids: Set[int] = set()
    for kind in present_slot_kinds:
        best = next(
            (s for s in new_all
             if s.target_slot.split(":")[0] == kind and id(s) not in selected_ids),
            None,
        )
        if best is not None:
            selected.append(best)
            selected_ids.add(id(best))
    # Fill remaining budget by relevance.
    for s in new_all:
        if len(selected) >= max_new:
            break
        if id(s) not in selected_ids:
            selected.append(s)
            selected_ids.add(id(s))
    # Re-rank the balanced selection by relevance for presentation.
    selected.sort(key=lambda s: s.relevance, reverse=True)
    new_kept = selected[:max_new]

    # Slot coverage: which identified slots did the surviving subproblems address?
    slots_present = {s.target_slot.split(":")[0] for s in new_kept}
    expected_slots = {"goal", "constraint", "unknown", "entity",
                      "success_criterion"}
    # Vague problems legitimately cover different slots than well-posed ones
    # (clarification instead of entity/unknown); don't penalise that.
    if structure.is_vague:
        expected_slots = {"goal", "constraint", "success_criterion"}
    covered = len(slots_present & expected_slots)

    # --- Retrieval bridge: attach candidate SOLUTION ideas, not just subproblems
    # Subproblems are meta-level ("decompose the objective"), so on their own
    # they never contain the problem-specific solution idea (measured: solution
    # coverage ~0.07). The cross-domain primitive analogizer DOES retrieve real
    # solution mechanisms, so surface them here to close the loop.
    solution_ideas: List[Dict[str, object]] = []
    try:
        from super_solver.vectorize import vectorization_service

        for h in vectorization_service.primitive_analogize(
            specification, top_k=max(2, min(3, max_new))
        ):
            if h.get("empty_match"):
                continue
            solution_ideas.append({
                "idea": h.get("content", ""),
                "title": h.get("title", ""),
                "mechanism": "structural_primitives",
                "overlap": float(h.get("primitive_overlap", 0.0)),
            })
    except Exception:
        pass

    return {
        "structure": structure.to_dict(),
        "subproblems": [s.to_dict() for s in new_kept + old_kept],
        "solution_ideas": solution_ideas,
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
