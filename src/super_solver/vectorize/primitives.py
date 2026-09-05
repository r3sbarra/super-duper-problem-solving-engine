"""Primitive stripper for cross-domain analogical transfer.

Richard's idea: instead of extracting relations (fragile, needs an LLM), strip
problems and solutions into STRUCTURAL PRIMITIVES — atomic action + object-type
atoms that capture the underlying structure while discarding surface entities.

Two problems in different domains share a structure when they share primitives:
  - "make a canoe hull float despite dense concrete"  -> [make_float, reduce_density, material]
  - "make a building panel light enough to lift"       -> [make_light, reduce_density, material]
Both share `reduce_density` + `material`, so they match even though the surface
entities (canoe vs panel, concrete vs foam) differ.

The stripper maps text to a primitive set via pattern matching. Matching is
Jaccard-style overlap on the primitive set, which is robust to surface
differences because primitives are domain-agnostic atoms.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Primitive vocabulary: (pattern, primitive) pairs.
# Patterns are matched against lowercased text; primitives are the structural
# atoms. Order matters: more specific patterns first.
# ---------------------------------------------------------------------------
ACTION_PRIMITIVES: List[tuple] = [
    # density / weight
    (r"float", "make_float"),
    (r"dense|density|heavy|lightweight|air-entraining|light enough|reduce.*weight", "reduce_density"),
    (r"strong but light|light.*strong|strength.*weight", "reduce_density"),
    # mimic / natural structure
    (r"mimic|hook-and-loop|burr|setae|gecko|natural structure|imitate|biomimic", "mimic_structure"),
    # state change
    (r"liquef|physical state|change.*state|compress.*gas|phase", "change_state"),
    (r"transport.*gas|ship.*gas|compact tank|volume.*transport", "reduce_volume"),
    # coating / surface
    (r"photocatalytic|titanium dioxide|coating|break down.*pollut|break down.*dirt|self-clean", "surface_coating"),
    (r"clean|pollut|smog|contaminant|break down.*dirt", "clean_surface"),
    # time / ordering
    (r"hybrid logical clock|logical clock|time-synchron|time sync|ordering|consisten", "time_ordering"),
    (r"latency|distributed|replicated|nodes|sensor network", "distributed_system"),
    # camouflage / hide
    (r"camouflage|hide|conceal|resemble its surroundings|invisible|deceive|reconnaissance|surveillance", "hide_structure"),
    # adhesion
    (r"adhesi|stick|peel|glue|suction|grip|fasten|attach", "adhesion"),
    (r"low-tack|reusable.*label|price tag", "low_tack_adhesive"),
    # suspension / span
    (r"cable|suspend|hang.*cable|span|tower|bridge|roof.*column", "cable_suspension"),
    (r"span.*river|span.*area|without.*support|no.*column", "span_without_support"),
    # energy / power
    (r"battery|energy density|solid-state|store energy", "energy_storage"),
    (r"solar|photovoltaic|perovskite|sunlight.*electricity", "solar_energy"),
    (r"fusion|plasma|ignition|net energy", "fusion_energy"),
    (r"microwave|magnetron|heat food|excite water", "microwave_heating"),
    # bio / chem
    (r"antibiotic|penicillin|bacterial|infection", "antibiotic"),
    (r"crispr|dna|genome|gene|edit.*dna", "gene_editing"),
    (r"lab-grown|cell culture|bioreactor|muscle cell|meat", "cell_culture"),
    (r"vertical farm|hydroponic|led lighting|crops.*stacked", "vertical_farming"),
    (r"plastic.*ocean|marine.*plastic|floating.*boom|cleanup", "ocean_cleanup"),
    (r"carbon.*capture|direct air|co2.*sorb|greenhouse", "carbon_capture"),
    (r"quantum.*error|qubit|decoherence|surface code", "quantum_error_correction"),
    # mechanisms
    (r"fastener|open and closed|wearing out|zipper|teeth.*mesh", "reusable_fastener"),
    (r"pre-drill|flat-pack|assemble.*quick|no special tools", "preliminary_action"),
    (r"vacuum|handheld|detach|portable", "segmentation"),
    (r"tank.*fail|storage tank|inspection|safety analysis|molasses", "safety_inspection"),
    (r"pacemaker|heartbeat|electrical pulse|implant", "electrical_implant"),
    (r"x-ray|radiation.*tissue|see inside|bone.*absorb", "penetrating_radiation"),
    (r"shave|blade|razor|cut.*skin", "blade_guard"),
    (r"bifocal|near.*far vision|lens power|glasses", "combined_lens"),
    (r"note.*stick|paper note|remove.*residue", "temporary_adhesive"),
]

# Object-type primitives: what kind of thing the problem is about.
OBJECT_PRIMITIVES: List[tuple] = [
    (r"canoe|boat|hull|ship|marine|vessel", "object_watercraft"),
    (r"panel|building|construction|roof|structure|factory|base|bridge|stadium", "object_structure"),
    (r"glass|window|surface", "object_surface"),
    (r"gas|hydrogen|lpg|fuel|tank", "object_gas"),
    (r"database|data store|distributed|sensor|network|nodes|clock", "object_distributed"),
    (r"fastener|zipper|label|note|adhesive|tag", "object_fastener"),
    (r"climbing|grip|gripping", "object_grip"),
    (r"material|concrete|foam|polymer|silicone|cement|coating", "object_material"),
    (r"food|meat|cell|tissue", "object_biological"),
    (r"energy|battery|solar|fusion|power", "object_energy"),
]


def strip_to_primitives(text: str) -> Set[str]:
    """Reduce text to its structural primitive set (action + object-type atoms)."""
    tl = text.lower()
    prims: Set[str] = set()
    for pattern, prim in ACTION_PRIMITIVES:
        if re.search(pattern, tl):
            prims.add(prim)
    for pattern, prim in OBJECT_PRIMITIVES:
        if re.search(pattern, tl):
            prims.add(prim)
    return prims


def primitive_overlap(a: Set[str], b: Set[str]) -> float:
    """Containment score: how much of the target's primitives are in the source.

    Uses |target ∩ source| / |target| (not symmetric Jaccard) so a source
    solution with EXTRA structural primitives (e.g. 'mimic_structure') is not
    penalized for being more specific. This matters because problems and
    solutions use different primitive vocabularies: a target problem says
    'grip walls' (adhesion) while its solution says 'mimic burrs'
    (mimic_structure + adhesion).
    """
    if not a:
        return 0.0
    inter = a & b
    return len(inter) / len(a)


def match_by_primitives(
    target_text: str,
    source_solutions: List[Dict[str, str]],
    top_k: int = 3,
) -> List[Dict]:
    """Match a target problem to source solutions by primitive overlap.

    Each source solution is stripped to primitives and compared against the
    target's primitives. Returns ranked matches with the overlap score.
    """
    t_prims = strip_to_primitives(target_text)
    scored = []
    for sol in source_solutions:
        s_prims = strip_to_primitives(sol["solution"])
        overlap = primitive_overlap(t_prims, s_prims)
        scored.append(
            {
                "content": sol["solution"],
                "title": sol.get("title", ""),
                "primitive_overlap": float(overlap),
                "target_primitives": sorted(t_prims),
                "source_primitives": sorted(s_prims),
            }
        )
    scored.sort(key=lambda x: x["primitive_overlap"], reverse=True)
    return scored[:top_k]
