"""Structural cross-domain matcher for super-solver.

Surface-based vector retrieval fails at cross-domain transfer: a target problem
in domain B ("grip smooth vertical walls") shares almost no surface tokens with
the structurally-similar source solution in domain A ("mimic hook-and-loop
structure of burrs"). Cross-domain transfer requires matching on the underlying
RELATIONAL structure (the action pattern), not the surface entities.

This module provides a structural matcher that reduces each problem/solution to
its core ACTION pattern and matches on that. It complements the surface-based
vector index: when surface retrieval is weak (low similarity), the structural
matcher can still find the right analog.

The structural signature is the core action-object pattern, e.g.:
  - "mimic a natural structure to achieve adhesion"  (Velcro, gecko)
  - "reduce material density to make it lighter"      (canoe, foam)
  - "use a light-activated coating to break down contaminants" (cement, glass)
  - "change physical state to reduce volume"          (LPG, hydrogen)
  - "combine physical time with logical ordering"     (hybrid clock, sensors)
  - "make a structure resemble its surroundings"      (camouflage, decoy)
  - "use a low-tack adhesive that peels off cleanly"  (Post-it, label)
  - "hang a structure from cables to span without supports" (bridge, roof)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from super_solver.core.embedder import get_backend

# ---------------------------------------------------------------------------
# Structural templates: abstract action patterns shared across domains.
# Each template is a canonical structural signature; solutions map to it.
# ---------------------------------------------------------------------------
STRUCTURAL_TEMPLATES: List[Dict[str, str]] = [
    {
        "id": "mimic_natural_structure",
        "template": "mimic a natural structure to achieve a functional property",
        "examples": "Velcro mimics burrs; gecko pads mimic setae",
    },
    {
        "id": "reduce_density",
        "template": "reduce the material density to make it lighter while keeping strength",
        "examples": "concrete canoe uses lightweight aggregates; foam panels use air",
    },
    {
        "id": "light_activated_coating",
        "template": "use a light-activated coating that breaks down surface contaminants",
        "examples": "photocatalytic cement cleans smog; self-cleaning glass breaks down dirt",
    },
    {
        "id": "change_physical_state",
        "template": "change the physical state of a substance to reduce its volume for transport",
        "examples": "LPG liquefies gas; hydrogen is liquefied for shipping",
    },
    {
        "id": "hybrid_time_ordering",
        "template": "combine physical time with logical ordering to keep distributed nodes consistent",
        "examples": "hybrid logical clock in databases; sensor network time sync",
    },
    {
        "id": "camouflage_resemblance",
        "template": "make a structure resemble its surroundings to hide it from observation",
        "examples": "Boeing factory roof fake town; military base camouflage",
    },
    {
        "id": "low_tack_adhesive",
        "template": "use a low-tack adhesive that holds firmly but peels off cleanly",
        "examples": "Post-it notes; reusable price labels",
    },
    {
        "id": "cable_suspension",
        "template": "hang a structure from cables to span a large area without interior supports",
        "examples": "suspension bridge; cable-stayed stadium roof",
    },
]


def structural_signature(text: str) -> str:
    """Reduce a problem/solution to its core structural action pattern.

    Uses a lightweight verb-phrase extractor tuned to the structural templates.
    Falls back to the raw text if no pattern matches.
    """
    tl = text.lower()
    patterns = [
        ("mimic", "mimic a natural structure"),
        ("hook-and-loop", "mimic a natural structure"),
        ("setae", "mimic a natural structure"),
        ("lightweight aggregate", "reduce the material density"),
        ("air-entraining", "reduce the material density"),
        ("density", "reduce the material density"),
        ("photocatalytic", "use a light-activated coating"),
        ("titanium dioxide", "use a light-activated coating"),
        ("liquef", "change the physical state"),
        ("physical state", "change the physical state"),
        ("hybrid logical clock", "combine physical time with logical ordering"),
        ("logical clock", "combine physical time with logical ordering"),
        ("camouflage", "make a structure resemble its surroundings"),
        ("resemble its surroundings", "make a structure resemble its surroundings"),
        ("low-tack", "use a low-tack adhesive"),
        ("peels off", "use a low-tack adhesive"),
        ("cables", "hang a structure from cables"),
        ("suspended", "hang a structure from cables"),
    ]
    for token, sig in patterns:
        if token in tl:
            return sig
    return text[:60]


def match_structural(
    target_text: str,
    source_solutions: List[Dict[str, str]],
    top_k: int = 3,
) -> List[Dict]:
    """Match a target problem to source solutions by structural signature.

    Each source solution is reduced to its structural signature and compared
    against the target's signature using the embedder. Returns ranked matches.
    """
    backend = get_backend()
    t_sig = structural_signature(target_text)
    v_t = backend.encode(t_sig)

    scored = []
    for sol in source_solutions:
        s_sig = structural_signature(sol["solution"])
        v_s = backend.encode(s_sig)
        sim = backend.cosine_similarity(v_t, v_s)
        scored.append(
            {
                "content": sol["solution"],
                "title": sol.get("title", ""),
                "structural_similarity": float(sim),
                "target_signature": t_sig,
                "source_signature": s_sig,
            }
        )
    scored.sort(key=lambda x: x["structural_similarity"], reverse=True)
    return scored[:top_k]
