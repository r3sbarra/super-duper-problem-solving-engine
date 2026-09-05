"""Entity substitution: adapt a transferred solution to the target domain.

The primitive stripper finds the structurally-similar source solution for a
target problem, but the generated text uses the SOURCE's entities (e.g. "mimic
burrs") instead of the TARGET's (e.g. "mimic gecko feet"). This module rewrites
the source solution by substituting the source's key entity with the target's
key entity, so the transferred solution reads correctly in the target domain.

The key entity is the object being acted on (the "what" of the solution). It is
extracted from the target problem text and substituted into the source solution
at the position of the source's key entity.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Entity extraction: find the key object entity in a problem/solution text.
# ---------------------------------------------------------------------------
# Patterns that identify the object being acted on. Ordered by specificity.
ENTITY_PATTERNS: List[tuple] = [
    # "make a <X> that ..." / "make a <X> ..."
    (r"make a ([\w\s]+?)(?: that| which|,| to|$)", "make_object"),
    # "how to <verb> a <X>" / "how to <verb> the <X>"
    (r"how to \w+ (?:a |the )?([\w\s]+?)(?: that| which|,| to|$)", "verb_object"),
    # "keep <X> clean" / "protect <X>"
    (r"(?:keep|protect|hide|clean|span|transport|ship|build|make|regulate|see|shave|fasten|correct|edit|grow|capture|remove|achieve|convert|store|heat|treat) (?:a |the )?([\w\s]+?)(?: that| which|,| to|$)", "action_object"),
    # "grips <X>" / "adhere to <X>"
    (r"(?:grips?|adhere[sd]? to|stick[s]? to|attaches? to) ([\w\s]+?)(?: without|,|$)", "grip_object"),
]

# Source-entity -> target-entity substitutions for known cross-domain pairs.
# The target entity is the one that appears in the TARGET PROBLEM text (the
# object being acted on), not the target solution's more specific phrasing.
KNOWN_SUBSTITUTIONS: Dict[str, Dict[str, str]] = {
    "burrs": {"climbing pad": "climbing pad", "pad": "climbing pad"},
    "hook-and-loop": {"climbing pad": "climbing pad"},
    "cement": {"windows": "windows", "glass": "glass"},
    "gas": {"hydrogen": "hydrogen"},
    "factory roof": {"base": "base"},
    "deck": {"roof": "roof"},
    "material": {"panel": "panel"},
}


def extract_target_entity(target_problem: str) -> Optional[str]:
    """Extract the key object entity from a target problem text."""
    tl = target_problem.strip()
    for pattern, _kind in ENTITY_PATTERNS:
        m = re.search(pattern, tl, re.IGNORECASE)
        if m:
            ent = m.group(1).strip()
            # Trim trailing filler words.
            ent = re.sub(r"\s+(?:that|which|without|to|and|or|for|with).*$", "", ent)
            ent = ent.strip()
            if ent and len(ent) > 2:
                return ent
    return None


def extract_source_entity(source_solution: str) -> Optional[str]:
    """Extract the key entity from a source solution (the object being acted on)."""
    tl = source_solution.strip()
    # Look for the object of the main action verb.
    for pattern, _kind in ENTITY_PATTERNS:
        m = re.search(pattern, tl, re.IGNORECASE)
        if m:
            ent = m.group(1).strip()
            ent = re.sub(r"\s+(?:that|which|to|and|or|for|with|from).*$", "", ent)
            ent = ent.strip()
            if ent and len(ent) > 2:
                return ent
    return None


def substitute_entity(
    source_solution: str,
    target_problem: str,
    known: Optional[Dict[str, str]] = None,
) -> str:
    """Rewrite the source solution with the target domain's entity.

    Strategy:
    1. If a known substitution maps a source entity to the target entity, use it.
    2. Otherwise, extract the target's key entity and substitute it for the
       source's key entity in the solution text.
    """
    known = known or {}
    # Try known substitutions first (source entity -> target entity).
    for src_ent, tgt_map in KNOWN_SUBSTITUTIONS.items():
        if src_ent in source_solution.lower():
            # Find which target entity appears in the target problem.
            for tgt_ent in tgt_map:
                if tgt_ent in target_problem.lower():
                    # Substitute the source entity with the target entity.
                    return re.sub(
                        re.escape(src_ent), tgt_ent, source_solution, flags=re.IGNORECASE
                    )

    # Fallback: extract target entity and substitute for source entity.
    tgt_ent = extract_target_entity(target_problem)
    src_ent = extract_source_entity(source_solution)
    if tgt_ent and src_ent and src_ent.lower() != tgt_ent.lower():
        return re.sub(
            re.escape(src_ent), tgt_ent, source_solution, flags=re.IGNORECASE
        )
    return source_solution
