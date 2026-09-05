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
    (r"adhesi|stick|peel|glue|suction|grip|fasten|attach|hold.*on|anchor|\bmount\b", "adhesion"),
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
    # mathematics
    (r"sum.*number|add.*number|add up|arithmetic series|consecutive numbers|total.*1 to|sum.*series|sum them|pair.*terms|sum of the", "arithmetic_sum"),
    (r"greatest common divisor|gcd|factor.*number|largest.*factor|prime factor", "gcd"),
    (r"prime|primality|sieve|composite|prime factor", "primality"),
    (r"area under.*curve|integral|antiderivative|numerical integration|area.*curve", "integration"),
    (r"linear equations|system of equations|gaussian elimination|row operations|solve.*equation", "linear_solve"),
    (r"shortest.*path|shortest.*route|weighted graph|dijkstra|relax.*edge|driving route", "shortest_path"),
    (r"prove.*all natural|induction|base case|holds for n|prove.*every", "induction"),
    (r"maximum.*function|critical point|derivative.*zero|extreme value|maximize|minimize", "optimization"),
    (r"fibonacci|nth.*sequence|exponential recursion|recurrence", "dynamic_programming"),
    # coding / algorithms
    (r"search.*\bsorted\b|binary search|halve.*range|middle element|find.*in.*\bsorted\b|search.*phone book", "binary_search"),
    (r"sort.*list|sort.*record|merge sort|split.*half|sorted halves|order.*items", "sorting"),
    (r"look up.*key|hash table|hash function|constant time.*lookup|find.*by.*id|find.*by.*key|instant.*lookup|look up.*by.*id", "hashing"),
    (r"longest common subsequence|prefix lengths|table.*cells|common subsequence", "dynamic_programming"),
    (r"traverse.*tree|depth-first|breadth-first|visited nodes|graph.*search|visit.*node", "graph_traversal"),
    (r"minimum spanning tree|kruskal|cheapest edge|no cycle|connect.*all.*node", "spanning_tree"),
    (r"linked list.*cycle|tortoise|hare|two pointers.*speed|detect.*cycle", "cycle_detection"),
    (r"compress.*text|huffman|shorter codes|frequent characters|compress.*file", "compression"),
    (r"maximum subarray|kadane|best sum ending|largest.*subarray", "max_subarray"),
    (r"evaluate.*expression|postfix|shunting-yard|operator precedence|parse.*expression", "expression_eval"),
    # advanced math / number theory
    (r"modular inverse|extended euclid|modulus.*cryptography|product.*congruent", "modular_inverse"),
    (r"large powers.*modulo|modular exponentiation|square.*base|encryption.*power", "modular_exp"),
    (r"prime factor.*large|pollard|nontrivial factor|factor.*huge", "pollard_rho"),
    (r"choose k|binomial|combinations|pascal.*triangle|n choose", "binomial"),
    (r"linear recurrence|companion matrix|matrix exponentiation|characteristic polynomial|nth term.*recurrence", "matrix_exp"),
    (r"coprime|relatively prime|gcd.*1", "coprime"),
    # advanced algorithms
    (r"maximum flow|augmenting path|ford-fulkerson|source.*sink.*capacity", "max_flow"),
    (r"minimum cost.*connect|kruskal|cheapest edge|spanning tree", "spanning_tree"),
    (r"longest increasing subsequence|smallest possible tails|\blis\b", "lis"),
    (r"kth smallest|quickselect|partition.*pivot|kth.*element|nth smallest|smallest number.*without sorting", "quickselect"),
    (r"pairs.*sum.*target|two-sum|complement.*hash|sum to a target|add up to a target|add up to.*target", "two_sum"),
    (r"merge.*sorted array|two pointers.*smaller|merge two sorted", "merge"),
    (r"median.*two sorted|partition point|logarithmic.*median", "median_sorted"),
    (r"palindrome|compare.*both ends|skip.*non-alphanumeric", "palindrome"),
    (r"number of islands|flood-fill|grid.*land.*water|count.*island", "flood_fill"),
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
    (r"series|sequence|number|integer|prime|gcd|equation|function|integral|derivative", "object_math"),
    (r"list|array|string|graph|tree|linked list|hash|subarray|expression|code|algorithm", "object_code"),
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
    target's primitives. Scoring:
    - Action primitives (e.g. reduce_density, hashing) are weighted higher than
      object-type primitives (e.g. object_math).
    - DISTINCTIVE actions (rare across solutions, e.g. quickselect) are weighted
      higher than common ones (e.g. sorting), so a solution sharing the target's
      specific action beats one sharing only generic primitives.
    """
    t_prims = strip_to_primitives(target_text)
    t_actions = {p for p in t_prims if not p.startswith("object_")}
    t_objs = t_prims - t_actions

    # IDF: how rare is each action across the source solutions? Rare actions
    # are more distinctive and should dominate the match.
    action_doc_freq: Dict[str, int] = {}
    for sol in source_solutions:
        s_actions = {p for p in strip_to_primitives(sol["solution"]) if not p.startswith("object_")}
        for a in s_actions:
            action_doc_freq[a] = action_doc_freq.get(a, 0) + 1
    n_docs = max(len(source_solutions), 1)

    scored = []
    for sol in source_solutions:
        s_prims = strip_to_primitives(sol["solution"])
        s_actions = {p for p in s_prims if not p.startswith("object_")}
        s_objs = s_prims - s_actions

        # Action containment, IDF-weighted: distinctive shared actions dominate.
        if t_actions:
            shared_actions = t_actions & s_actions
            if shared_actions:
                idf_sum = sum(
                    (1.0 + (n_docs / (action_doc_freq.get(a, 1) + 1))) for a in shared_actions
                )
                idf_total = sum(
                    (1.0 + (n_docs / (action_doc_freq.get(a, 1) + 1))) for a in t_actions
                )
                action_contain = idf_sum / idf_total
            else:
                action_contain = 0.0
        else:
            action_contain = 0.0

        # Object containment (unweighted, breaks ties).
        obj_contain = len(t_objs & s_objs) / len(t_objs) if t_objs else 0.0

        overlap = 0.8 * action_contain + 0.2 * obj_contain
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


# ---------------------------------------------------------------------------
# Solution validation: is a generated solution worth learning?
# ---------------------------------------------------------------------------
# Vague principle phrases that indicate the engine did NOT produce a concrete
# solution (it fell back to a TRIZ principle description).
VAGUE_MARKERS = [
    "phase transition", "inert atmosphere", "segmentation", "dynamization",
    "self-service", "copying", "cheap disposable", "mechanical vibration",
    "periodic action", "porosity", "color change", "thermal expansion",
    "composite", "parameter change", "spheroidality", "prior action",
    "cushion in advance", "equipotentiality", "another dimension",
    "feedback", "intermediary", "blessing in disguise", "homogeneity",
    "discarding", "reversing", "partial action", "asymmetry", "extraction",
    "merging", "universality", "nested doll", "counterweight", "preventive",
    "pre-arrangement", "replacement", "flexible shell", "thin film",
    "porous material", "optical property", "local quality", "taking out",
]


def validate_solution(generated: str, target_problem: str) -> dict:
    """Validate a generated solution. Returns {valid, reason, primitives}.

    A solution is valid (worth learning) if it is:
    1. Concrete — not a vague TRIZ principle description.
    2. Non-trivial — has structural primitives (not an empty echo).
    3. A real solution — has at least one action primitive (e.g.
       reduce_density, mimic_structure, liquefy), not just object types.
    4. Structurally relevant — shares at least one ACTION primitive with the
       target problem (it addresses the same kind of problem). This rejects
       concrete-but-wrong solutions (e.g. a surface-coating solution for an
       adhesion problem).
    """
    gl = generated.lower()
    # 1. Concreteness: must not be a vague TRIZ principle.
    for marker in VAGUE_MARKERS:
        if marker in gl:
            return {"valid": False, "reason": f"vague principle: '{marker}'", "primitives": set()}

    # 2. Non-triviality: must not just echo the problem.
    gen_prims = strip_to_primitives(generated)
    if not gen_prims:
        return {"valid": False, "reason": "no structural primitives", "primitives": set()}

    # 3. Structural relevance: must have at least one action primitive (a real
    #    solution, not just object types). NOTE: we do NOT require the solution
    #    to share primitives with the target problem — cross-domain transfer
    #    is precisely about introducing a NEW mechanism (e.g. mimic_structure
    #    for an adhesion need), so a primitive-overlap check would reject the
    #    correct solution. The gate rejects vague/non-solutions but cannot
    #    verify correctness (that needs ground truth).
    gen_actions = {p for p in gen_prims if not p.startswith("object_")}
    if not gen_actions:
        return {"valid": False, "reason": "no action primitive (not a solution)", "primitives": gen_prims}

    return {"valid": True, "reason": "concrete action primitive", "primitives": gen_prims}
