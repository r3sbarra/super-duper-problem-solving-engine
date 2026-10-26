"""Adaptive, pretraining-free option scoring for the EC-DT reflex layer.

Why this module exists
----------------------
The first version of the reflex layer leaned on ``ec_dt``'s
``UniversalSemanticEncoder``. Measured on the real tool catalogs, that encoder
is untrained and near-orthogonal: every query<->option cosine landed at ~0.99,
score spread ~0.006. It therefore ranks close to arbitrarily, and a conformal
threshold fitted over it can only certify a coin flip.

The alternative that does not require pretraining
-------------------------------------------------
The option catalog IS the training corpus. Every tool ships a description at
construction time. From those descriptions alone we can derive:

1. **IDF over the catalog.** A word present in every description ("solver")
   carries zero signal; a word present in one ("quickselect") carries almost
   all of it. This is *derived*, not learned -- no labels, no weights, no
   gradient steps.
2. **Coverage normalization.** Score(query, option) = (IDF mass of query terms
   found in the option) / (IDF mass of all query terms). This is bounded in
   [0, 1], so it is directly comparable to a conformal threshold and cannot be
   inflated by long descriptions the way raw counts can.
3. **An explicit OOV signal.** If the query shares no catalog vocabulary, the
   honest answer is "no evidence" rather than an argmax. That is reported as
   ``evidence=0`` and the router escalates.

What is deliberately NOT here
-----------------------------
No synonym expansion, no embeddings, no query rewriting. Each of those would
either reintroduce a pretrained dependency or silently fit the evaluation set.

Relation to EC-DT
-----------------
EC-DT's contribution is kept where it is sound: the quotient-manifold
constraint check (admissible states) and the split-conformal calibration. Its
learned ranker is demoted to an optional tie-breaker. The decision *procedure*
is unchanged -- score, take a conformal set, certify only if the set is a
singleton and every invariant passes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

STOPWORDS: Set[str] = set(
    """
    a an the and or but if then than that this these those it its of to in on
    at by for with from into over under about as is are was were be been being
    do does did have has had can could should would will may might must not no
    we you i he she they them our your their me my which what when where who
    how why so such also just only very more most less least any some all each
    both few many much other another same own per via versus
    """.split()
)


def tokens(text: str) -> List[str]:
    """Content tokens, lowercased, with a light plural fold.

    Plurals are folded because a catalog description says "constraints" while a
    user says "constraint"; without the fold every such pair costs the full
    IDF mass of the term. Only a trailing ``s``/``es`` is removed, so the fold
    cannot conflate genuinely different words.
    """
    out: List[str] = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOPWORDS:
            continue
        out.append(w)
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            out.append(w[:-1])
        elif len(w) > 5 and w.endswith("es"):
            out.append(w[:-2])
    return out


def term_set(text: str) -> Set[str]:
    return set(tokens(text))


@dataclass
class CatalogIndex:
    """IDF statistics derived from a set of option descriptions.

    Built once per catalog; everything after that is a set intersection, so a
    query never touches a model.
    """

    options: Dict[str, str]
    docs: Dict[str, Set[str]] = field(init=False)
    df: Dict[str, int] = field(init=False)
    n_docs: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.docs = {}
        df: Dict[str, int] = {}
        for name, desc in self.options.items():
            terms = term_set(name.replace("_", " ") + " " + desc)
            self.docs[name] = terms
            for t in terms:
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.n_docs = max(len(self.options), 1)

    def idf(self, term: str) -> float:
        """Smoothed IDF; exactly 0.0 for a term present in EVERY option.

        A term in every description ("solver") cannot discriminate, and giving
        it any weight would make every option look equally relevant.
        """
        d = self.df.get(term, 0)
        if d == 0:
            return 0.0
        if d >= self.n_docs:
            return 0.0
        return float(math.log((self.n_docs - d + 0.5) / (d + 0.5)))

    def scores(self, query: str) -> Tuple[Dict[str, float], float]:
        """Evidence per option, plus the query's total IDF mass.

        Two quantities are needed to rank honestly:

        * ``total`` -- how much evidence the query carries at all. Zero means
          the query shares nothing with the catalog and NO ranking is
          meaningful; the caller must escalate rather than pick an argmax.
        * ``scores[option]`` -- the IDF mass of query terms this option
          explains, normalized by ``sqrt(total)`` plus a small prior.

        The sqrt normalization is the important part. Plain coverage
        (``hit/total``) saturates at 1.0 on a SINGLE shared term, so "chop this
        investigation into pieces ... at the same time" scored 1.0 for the
        tool whose description merely contains "time" (measured). Dividing by
        sqrt(total) makes one lucky term worth much less than a matched set of
        several distinctive terms, without abandoning the bounded [0,1] range
        the conformal threshold needs.
        """
        q_terms = term_set(query)
        masses = {t: self.idf(t) for t in q_terms if t in self.df}
        total = sum(masses.values())
        if total <= 0.0:
            return {name: 0.0 for name in self.options}, 0.0
        denom = math.sqrt(total) + 1.0
        out = {}
        for name, terms in self.docs.items():
            hit = sum(m for t, m in masses.items() if t in terms)
            out[name] = hit / denom
        return out, total

    def coverage(self, query: str, option: str) -> float:
        """Fraction of the query's IDF mass explained by one option (diagnostic)."""
        q_terms = term_set(query)
        masses = {t: self.idf(t) for t in q_terms if t in self.df}
        total = sum(masses.values())
        if total <= 0.0:
            return 0.0
        return sum(m for t, m in masses.items() if t in self.docs[option]) / total


def rank(index: CatalogIndex, query: str) -> List[Tuple[str, float]]:
    """Options ordered by IDF coverage (descending)."""
    scores, _ = index.scores(query)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def discriminating_terms(index: CatalogIndex, limit: int = 10) -> List[Tuple[float, str]]:
    """Highest-IDF terms in the catalog (useful for auditing a catalog)."""
    return sorted(((index.idf(t), t) for t in index.df), reverse=True)[:limit]
