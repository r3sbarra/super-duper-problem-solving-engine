"""Multi-objective Pareto resolution for the super-solver.

When a problem has several objectives ("minimize latency AND cost AND carbon
footprint"), returning one decomposition per objective is not enough: the
solution must reason about the TRADE-OFF — which candidates are non-dominated,
where the knee is, and how to pick when objectives conflict.

This module is deterministic and offline. It treats objectives as dimensions
with a direction (minimize/maximize), takes candidate solutions scored on each
objective, and computes:

  - the Pareto-optimal (non-dominated) set,
  - a knee point (best compromise: closest to the ideal in normalized space),
  - a scalarized ranking (weighted-sum / Tchebycheff) for when a single pick is
    required,
  - the conflict structure (which objective pairs actually trade off).

Nothing here calls a network or an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Objective:
    """One objective dimension of a multi-goal problem."""

    name: str
    direction: str = "minimize"  # "minimize" | "maximize"

    def normalized(self, value: float, lo: float, hi: float) -> float:
        """Map a raw value to a 'lower is better' score in [0, 1].

        Minimize: (v - lo) / (hi - lo). Maximize: inverted.
        """
        span = hi - lo
        if span <= 0:
            return 0.0
        n = (value - lo) / span
        if self.direction == "maximize":
            n = 1.0 - n
        return max(0.0, min(1.0, n))


@dataclass
class Candidate:
    """A candidate solution scored on each objective."""

    name: str
    values: Dict[str, float]  # objective name -> raw value
    payload: Optional[dict] = None


@dataclass
class ParetoResult:
    pareto_front: List[str] = field(default_factory=list)     # non-dominated names
    dominated: List[str] = field(default_factory=list)
    knee: Optional[str] = None
    scalarized_ranking: List[Tuple[str, float]] = field(default_factory=list)
    conflicts: List[Tuple[str, str, float]] = field(default_factory=list)
    ideal: Dict[str, float] = field(default_factory=dict)
    nadir: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pareto_front": list(self.pareto_front),
            "dominated": list(self.dominated),
            "knee": self.knee,
            "scalarized_ranking": [
                {"candidate": n, "score": round(s, 4)} for n, s in self.scalarized_ranking
            ],
            "conflicts": [
                {"a": a, "b": b, "correlation": round(c, 4)} for a, b, c in self.conflicts
            ],
            "ideal": self.ideal,
            "nadir": self.nadir,
        }


def _dominates(a: Dict[str, float], b: Dict[str, float], objs: Sequence[Objective]) -> bool:
    """True if candidate a weakly dominates b on all objectives (normalized-lower).

    Uses normalized 'lower is better' values; requires the caller to have
    normalized already.
    """
    strictly_better = False
    for o in objs:
        va, vb = a[o.name], b[o.name]
        if va > vb:  # a is worse on this objective
            return False
        if va < vb:
            strictly_better = True
    return strictly_better


def resolve(
    objectives: Sequence[Objective],
    candidates: Sequence[Candidate],
    weights: Optional[Dict[str, float]] = None,
) -> ParetoResult:
    """Compute the Pareto frontier, knee, scalarized ranking, and conflicts.

    Steps:
      1. Normalize each objective to a 'lower is better' score in [0, 1] using
         the observed min/max across candidates (ideal = 0, nadir = 1).
      2. Non-dominated set: a candidate is Pareto-optimal if no other candidate
         weakly dominates it on every objective (and is strictly better on >=1).
      3. Knee: on the frontier, the point minimizing distance to the ideal
         (Tchebycheff/L2 compromise).
      4. Scalarization: weighted sum over normalized values (default equal
         weights); ranking for a single pick.
      5. Conflicts: Pearson correlation between objective pairs across
         candidates; strongly negative correlation = a real trade-off.
    """
    if not objectives or not candidates:
        return ParetoResult()

    names = [o.name for o in objectives]
    # 1. normalize
    lo = {n: min(c.values.get(n, 0.0) for c in candidates) for n in names}
    hi = {n: max(c.values.get(n, 0.0) for c in candidates) for n in names}
    norm: Dict[str, Dict[str, float]] = {}
    for c in candidates:
        norm[c.name] = {
            o.name: o.normalized(c.values.get(o.name, 0.0), lo[o.name], hi[o.name])
            for o in objectives
        }

    # 2. non-dominated set
    front: List[str] = []
    dominated: List[str] = []
    for c in candidates:
        is_dominated = any(
            d.name != c.name and _dominates(norm[d.name], norm[c.name], objectives)
            for d in candidates
        )
        (dominated if is_dominated else front).append(c.name)

    # 3. knee (closest to ideal=zero vector, Tchebycheff distance)
    knee = None
    if front:
        def dist(nm: str) -> float:
            v = norm[nm]
            # Tchebycheff (worst normalized objective) + small L2 tie-break
            return max(v.values()) + 0.01 * sum(x * x for x in v.values())
        knee = min(front, key=dist)

    # 4. scalarized ranking (weighted sum; lower is better)
    w = weights or {n: 1.0 / len(names) for n in names}
    wsum = {n: w.get(n, 0.0) for n in names}
    tot = sum(wsum.values()) or 1.0
    wsum = {n: v / tot for n, v in wsum.items()}
    ranking = sorted(
        ((c.name, sum(wsum[n] * norm[c.name][n] for n in names)) for c in candidates),
        key=lambda t: t[1],
    )

    # 5. conflicts: correlation between objective pairs
    conflicts: List[Tuple[str, str, float]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            xs = [norm[c.name][names[i]] for c in candidates]
            ys = [norm[c.name][names[j]] for c in candidates]
            corr = _pearson(xs, ys)
            if corr <= -0.5:
                conflicts.append((names[i], names[j], corr))

    return ParetoResult(
        pareto_front=front,
        dominated=dominated,
        knee=knee,
        scalarized_ranking=ranking,
        conflicts=conflicts,
        ideal={n: lo[n] for n in names},
        nadir={n: hi[n] for n in names},
    )


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    return cov / denom if denom > 0 else 0.0
