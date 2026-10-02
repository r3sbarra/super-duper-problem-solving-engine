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
    # elasticity / viscoelastic deformation (distinct from density)
    (r"elastic|viscoelastic|mold|deform|shape-holding|stretch|moldable", "elasticity"),
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
    (r"latency|distributed system|distributed systems|replicated across|across nodes|sensor network", "distributed_system"),
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
    (r"sum.*number|add.*number|add up|arithmetic series|consecutive numbers|total.*1 to|sum.*series|sum them|pair.*terms|sum of the|total of the first|total of the.*numbers|add.*whole numbers|sum.*whole numbers|first .* numbers.*total|gauss", "arithmetic_sum"),
    (r"greatest common divisor|gcd|largest.*factor|prime factor", "gcd"),
    (r"is .* prime\b|prime or not|primality test|probabilistic test|witness.*prime|determine whether.*prime|divisors.*other than|other than 1 and itself|has any divisors|check whether.*number.*prime", "primality_test"),
    (r"count.*prime|prime numbers up to|sieve|primes up to", "prime_sieve"),
    (r"primality|composite|\bprime\b", "primality"),
    (r"area under.*curve|integral|antiderivative|numerical integration|area.*curve", "integration"),
    (r"linear equations|system of equations|gaussian elimination|row operations|solve.*equation", "linear_solve"),
    (r"shortest.*path|shortest.*route|weighted graph|dijkstra|relax.*edge|driving route", "shortest_path"),
    (r"prove.*all natural|induction|base case|holds for n|prove.*every", "induction"),
    (r"maximum.*function|critical point|derivative.*zero|extreme value|maximize|minimize", "optimization"),
    (r"fibonacci|nth.*sequence|exponential recursion|recurrence", "dynamic_programming"),
    # coding / algorithms
    (r"search.*\bsorted\b|binary search|halve.*range|middle element|find.*in.*\bsorted\b|search.*phone book", "binary_search"),
    (r"sort.*list|sort.*record|merge sort|split.*half|sorted halves|order.*items|sorted halves", "sorting"),
    (r"split.*half.*(sort|merge)|merge.*sorted|sorted halves", "merge"),
    (r"look up.*key|hash table|hash function|constant time.*lookup|instant.*lookup|\bhash\b", "hashing"),
    (r"longest common subsequence|prefix lengths|table.*cells|common subsequence", "dynamic_programming"),
    (r"traverse.*tree|depth-first|breadth-first|visited nodes|graph.*search|visit.*node|"
     r"reachable from|reachable from (a|the)|start.*(node|vertex|city|airport).*(reach|visit)|"
     r"connected component|stopping anywhere|can.*reach|determine reachability|reachability", "graph_traversal"),
    (r"minimum spanning tree|kruskal|cheapest edge|no cycle|connect.*all.*(node|computer|city|airport|vertex|machine)|"
     r"connect (all|every).*(computer|city|airport|node|office)|minimum.*(cost|weight).*(connect|tree)|"
     r"cheapest.*(connect|link) (all|every)|least total (cable|wire|cost)|connect all computers", "spanning_tree"),
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
    # --- statistics / probability ---
    (r"descriptive statistics|standard deviation|\bvariance\b|summarise.*measurement|central tendency|mean.*median|spread.*data|summarise.*data", "descriptive_stats"),
    (r"statistically significant|hypothesis test|t-test|chi-square|p-value.*null|significant difference between|really differ.*chance|differ.*rather than.*chance|two groups.*differ|by chance", "hypothesis_test"),
    (r"number of successes.*trial|binomial distribution|independent trials.*success", "binomial_dist"),
    (r"least-squares|linear regression|regression line|fit.*line|line.*fit|fit.*curve|normal equations|slope.*intercept|best-fit|trend line|fits them best|noisy measurement", "curve_fitting"),
    (r"real roots.*polynomial|isolate.*root|sturm|count.*sign change|roots.*interval|crosses zero|where.*polynomial.*zero|zeros.*polynomial", "root_isolation"),
    (r"integer relation|pslq|minimal polynomial|algebraic.*constant|integer coefficients.*satisfied", "integer_relation"),
    (r"counterexample|falsif|stress-test.*identity|identity.*actually true|adversarial.*boundary", "falsification"),
    # --- natural science (physics/chem/bio) ---
    (r"arrhenius|reaction rate.*temperature|activation energy|reaction.*get.*hotter|chemical reaction.*speed|rate.*constant", "arrhenius_kinetics"),
    (r"conserved.*energy|kinetic.*potential|energy.*conserv|damped.*oscillat|pendulum", "energy_conservation"),
    (r"population.*grow|logistic growth|carrying capacity|exponential growth.*limit", "logistic_growth"),
    (r"disease.*spread|infection.*population|sir model|compartmental|reproduction number|\br0\b|epidemic|infection.*move|spread.*population", "compartmental_model"),
    (r"schrodinger|orbitals|quantum states|wavefunction|symmetry.*group representation", "quantum_states"),
    (r"noether|conserved quantit|conservation law|symmetry.*lagrangian|continuous symmetry|conserved quantity.*symmetry", "noether_symmetry"),
    (r"elliptic curve|point counting|hasse bound|weierstrass.*finite field|points.*(lie on|on).*curve.*field|cubic curve.*field", "point_counting"),
    # advanced math / geometry
    (r"chirality|jones polynomial|knot invariant|topologically different|mirror.*knot|knot", "knot_invariant"),
    (r"geometry theorem|euclidean geometry|wu.*method|pseudodivide.*conclusion|prove.*triangle|geometric proof|fact about triangle|triangles.*algebra", "geometry_proof"),
    (r"common measure|common divisor|greatest common|shared by two|two big whole numbers", "gcd"),
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
    # --- networking / distributed systems ---
    (r"fail over|replica|replicate.*node|leader.*die|primary.*down|keep.*running.*server|high availability|failover|node fails|server fails|when one .* fails|one (server|node) fails", "failover"),
    (r"agree.*same|consensus|raft|paxos|majority.*agree|same.*order|settle.*order", "consensus"),
    (r"exactly once|deduplicat|duplicate.*message|processed once|unique id.*message", "exactly_once"),
    (r"load balanc|distribute.*traffic|balance.*request|too many.*server|spread.*load", "load_balance"),
    (r"circuit breaker|fail fast|slow.*backend|backend.*slow|db.*slow|service.*(down|slow)|"
     r"timeout.*drag|timing out|dragging down|protect.*from.*slow|one.*service.*down|"
     r"continue working when|keep (the )?service|fail gracefully", "circuit_breaker"),
    (r"retry.*backoff|exponential backoff|jitter|retry.*flaky|back off", "backoff_retry"),
    (r"scale.*spike|auto-scale|horizontal.*scale|more instances|handle.*surge|sudden.*traffic", "horizontal_scale"),
    (r"queue.*survive|durable queue|lease.*crash|work.*not lost|persist.*queue", "durable_queue"),
    # --- security / reliability ---
    (r"unauthorized|authentication|authorization|verify identity|check permission|stranger.*get.*in|prevent.*access", "auth"),
    (r"encrypt|eavesdrop|sniff.*traffic|protect.*data.*read|cipher|secure.*transmission", "encryption"),
    (r"compromised|anomaly|known-bad|baseline behavior|intrusion|behaving abnormally", "anomaly_detection"),
    (r"recover.*crash|backup|restore|disaster|recovery time|get.*data.*back", "backup_recovery"),
    (r"fail gracefully|graceful degradation|partial failure|contain.*failure|not.*go down|fallback", "graceful_degradation"),
    (r"malicious input|sanitize|validate.*input|sandbox|exploit.*program|untrusted code", "input_sanitization"),
    # --- data engineering / ops ---
    (r"does not fit.*memory|partition.*data|process.*parallel.*cluster|large dataset|aggregate.*result", "data_partitioning"),
    (r"schema.*change|schema validation|contract test|fail loudly|unexpected data shape", "schema_validation"),
    (r"slow.*query|add.*index|full scan|index.*column|where.*join.*clause|crawls.*table|takes forever|search.*slow", "indexing"),
    (r"checkpoint|resume.*last|interrupted.*job|lose progress|restart.*scratch", "checkpointing"),
    (r"rolling baseline|deviate.*threshold|drift.*normal|metric.*out of range", "baseline_monitoring"),
    (r"blue-green|canary|deploy.*without downtime|fraction.*traffic|new version.*risk", "canary_deploy"),
    (r"root cause|intermittent|hard to reproduce|structured logging|tracing|reconstruct.*evidence|only happens sometimes|figure out where", "root_cause_tracing"),
    # --- meta-engineering / self-improvement ---
    (r"decompos\w*.*(solution|solve)|lead to solved|close the loop|end.to.end.*solv|solution.synthesis|synthesis stage", "solution_synthesis"),
    (r"retriev\w*.*(solution|mechanism|analog)|whole.problem.*retriev|candidate solutions", "solution_retrieval"),
    (r"expand.*corpus|enrich.*corpus|more training|widen.*coverage|generaliz", "corpus_expansion"),
    (r"rank.*(option|proposal|candidate)|score.*alternative|pareto|trade.?off.*rank|multi.?objective", "option_ranking"),
    (r"self.review|self.audit|recursive|introspect|apply.*to itself|improve (the|its) (engine|itself)", "self_review"),
    (r"fail.?closed|fail.?safe|sanity.?filter|guard(rail)?|invariant.*check|regression.?guard", "robustness_guard"),
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
    (r"material|concrete|foam|polymer|silicone|cement|coating|elastomer|viscoelastic", "object_material"),
    (r"food|meat|cell|tissue", "object_biological"),
    (r"energy|battery|solar|fusion|power", "object_energy"),
    (r"series|sequence|number|integer|prime|gcd|equation|function|integral|derivative", "object_math"),
    (r"list|array|string|graph|tree|linked list|hash|subarray|expression|code|algorithm", "object_code"),
    (r"server|node|replica|cluster|network|traffic|load balancer|queue|message|distributed", "object_infra"),
    (r"user|request|access|permission|password|credential|attack|threat|intrusion", "object_security"),
    (r"data|dataset|schema|query|index|metric|log|pipeline|backup|deploy|version", "object_data"),
    (r"engine|reasoning|decompos|retriev|corpus|primitive|pipeline|agent|solver|self.improvement|meta", "object_engine"),
]


# ---------------------------------------------------------------------------
# Generic action layer: the problem-solving VERB CLASS. Every problem is an
# instruction to do something; capturing that generic action guarantees at least
# one action atom even when no domain-specific primitive fires (measured: 100
# domain regexes still left 'reduce p99 latency' with only an object atom and no
# action). Specific domain primitives above win when both match.
# ---------------------------------------------------------------------------
GENERIC_ACTION_PRIMITIVES: List[tuple] = [
    (r"\b(reduc|lower|decreas|minimi|ease|cut|shrink|lessen|mitigat|alleviat|reliev)\w*", "act_reduce"),
    (r"\b(increas|rais|maximi|boost|grow|expand|amplif|enhanc|strengthen|accelerat|speed up)\w*", "act_increase"),
    (r"\b(improv|optimi|better|upgrade|refine|polish|tune)\w*", "act_improve"),
    (r"\b(prevent|avoid|stop|block|inhibit|suppress|guard|protect|mitigat)\w*", "act_prevent"),
    (r"\b(detect|identif|diagnos|recogni|spot|discover|find out|flag)\w*", "act_detect"),
    (r"\b(design|build|construct|creat|develop|engineer|synthesi|implement|make a|make the)\w*", "act_create"),
    (r"\b(prove|deriv|demonstrat|establis|show that|verify|validat)\w*", "act_prove"),
    (r"\b(explain|understand|interpret|elucidat|account for|clarify|analyz|analys|investigat|explor)\w*", "act_explain"),
    (r"\b(estim|predict|forecast|approximat|project|extrapolat|measur|quantif)\w*", "act_estimate"),
    (r"\b(keep|maintain|preserv|sustain|retain|hold|stay)\w*", "act_maintain"),
    (r"\b(route|direct|navigat|schedul|assign|allocat|distribut|balanc|organiz)\w*", "act_allocate"),
    (r"\b(connect|link|join|merg|integrat|attach|bind)\w*", "act_connect"),
    (r"\b(recover|restor|repair|heal|fix|resolv|rescue|remediat)\w*", "act_repair"),
    (r"\b(replac|substitut|swap|migrat|conver)\w*", "act_replace"),
    (r"\b(combin|aggregat|consolidat|unify|pool|batch)\w*", "act_combine"),
    (r"\b(compress|encod|pack|condens|densif|summari)\w*", "act_compress"),
    (r"\b(priorit|rank|order|sort|select|choos|pick|filter)\w*", "act_select"),
    (r"\b(transmit|send|broadcast|communicat|propagat|share|relay)\w*", "act_transmit"),
    (r"\b(store|save|cache|persist|retain.*data|record)\w*", "act_store"),
    (r"\b(comput|calculat|evaluat|determin|solve|process)\w*", "act_compute"),
]

# Quality-attribute layer: WHAT is being improved/degraded. Gives problems a
# subject even when no object-type fires.
QUALITY_ATTRIBUTES: List[tuple] = [
    (r"\b(latenc|delay|respons|speed|throughput|performanc|efficien)\w*", "attr_performance"),
    (r"\b(cost|price|budget|expens|cheap|affordab)\w*", "attr_cost"),
    (r"\b(reliab|availab|uptime|resilien|robust|fault|downtime|outage)\w*", "attr_reliability"),
    (r"\b(safe|secur|risk|hazard|danger|threat|vulnerab)\w*", "attr_safety"),
    (r"\b(accura|precis|correct|error|quality|fidelity)\w*", "attr_accuracy"),
    (r"\b(weight|mass|density|light|heavy)\w*", "attr_weight"),
    (r"\b(energy|power|consumption|efficien|thermal|heat)\w*", "attr_energy"),
    (r"\b(time|duration|deadline|schedul|late|slow|fast)\w*", "attr_time"),
    (r"\b(scale|size|volume|capacity|scalab|load|traffic)\w*", "attr_scale"),
    (r"\b(divers|variet|fairness|bias|balanc)\w*", "attr_diversity"),
    (r"\b(memory|storage|space|footprint|resource)\w*", "attr_resource"),
    (r"\b(complex|simpl|maintainab|readab|elegance)\w*", "attr_complexity"),
]


def strip_to_primitives(text: str) -> Set[str]:
    """Reduce text to its structural primitive set (action + object-type atoms).

    Three layers, most-specific first: domain ACTION primitives, domain OBJECT
    types, then GENERIC action + QUALITY-attribute atoms. The generic layers
    guarantee every problem yields at least one action + one attribute atom, so
    structural matching is never starved (measured: 100 domain regexes still left
    'reduce p99 latency' with only an object atom and no action).
    """
    tl = text.lower()
    prims: Set[str] = set()
    for pattern, prim in ACTION_PRIMITIVES:
        if re.search(pattern, tl):
            prims.add(prim)
    for pattern, prim in OBJECT_PRIMITIVES:
        if re.search(pattern, tl):
            prims.add(prim)
    # Generic layers (added after specific ones; both may contribute).
    for pattern, prim in GENERIC_ACTION_PRIMITIVES:
        if re.search(pattern, tl):
            prims.add(prim)
    for pattern, prim in QUALITY_ATTRIBUTES:
        if re.search(pattern, tl):
            prims.add(prim)

    # Optional LLM fallback for open-vocabulary domain extraction when regex yields empty
    if not prims:
        try:
            from super_solver.core.llm_parser import llm_client

            llm_prims = llm_client.extract_structural_primitives(text)
            if llm_prims:
                prims.update(llm_prims)
        except Exception:
            pass

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
    # Empty-primitive guard: if the target strips to nothing (no action and no
    # object primitive fired), every source ties at 0.0 and the first entry wins
    # arbitrarily — a silent wrong match. Surface that explicitly so callers can
    # treat it as 'no structural signal' rather than trusting a coin-flip top-1.
    if not t_prims:
        return [{
            "content": "<no structural primitives extracted>",
            "title": "",
            "primitive_overlap": 0.0,
            "target_primitives": [],
            "source_primitives": [],
            "empty_match": True,
        }]
    t_actions = {p for p in t_prims if not p.startswith("object_")}
    t_objs = t_prims - t_actions

    # Generic atoms (act_*/attr_*) guarantee a problem is non-empty but carry
    # almost no discriminative signal, so they are EXCLUDED from the match
    # scoring: only SPECIFIC domain primitives determine the ranking. (Measured:
    # including them regressed cross-domain transfer 3/8 -> 2/8 even when
    # down-weighted, because they perturb the containment denominators.)
    def _is_generic(p: str) -> bool:
        return p.startswith("act_") or p.startswith("attr_")

    t_actions_spec = {p for p in t_actions if not _is_generic(p)}
    t_objs_spec = {p for p in t_objs if not _is_generic(p)}
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
        s_actions_spec = {p for p in s_actions if not _is_generic(p)}
        s_objs_spec = {p for p in s_objs if not _is_generic(p)}

        # Action containment, IDF-weighted: distinctive shared actions dominate.
        if t_actions_spec:
            shared_actions = t_actions_spec & s_actions_spec
            if shared_actions:
                idf_sum = sum(
                    (1.0 + (n_docs / (action_doc_freq.get(a, 1) + 1)))
                    for a in shared_actions
                )
                idf_total = sum(
                    (1.0 + (n_docs / (action_doc_freq.get(a, 1) + 1)))
                    for a in t_actions_spec
                )
                action_contain = idf_sum / idf_total
            else:
                action_contain = 0.0
        else:
            action_contain = 0.0

        # Object containment (specific objects only; breaks ties).
        if t_objs_spec:
            obj_contain = len(t_objs_spec & s_objs_spec) / len(t_objs_spec)
        else:
            obj_contain = 0.0

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
    # Tie-break: identical primitive overlap (e.g. sort vs binary_search both
    # {sorting, object_code}) is broken arbitrarily. Secondary sort by a
    # dependency-free full-text similarity (difflib SequenceMatcher over the raw
    # target vs source text) prefers the source whose wording is closest, which
    # resolves most near-identical-primitive ties.
    import difflib
    for s in scored:
        s["_text_sim"] = difflib.SequenceMatcher(
            None, target_text.lower(), s.get("content", "").lower()
        ).ratio()
    scored.sort(key=lambda x: (x["primitive_overlap"], x["_text_sim"]), reverse=True)
    for s in scored:
        s.pop("_text_sim", None)
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
