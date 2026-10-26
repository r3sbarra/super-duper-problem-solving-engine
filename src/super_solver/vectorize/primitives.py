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
# Compiled-pattern cache + primitive memoization.
#
# Every strip ran ~186 `re.search(pattern_string, ...)` calls. Python's own
# `re` module-level cache is only 512 entries and `re.search(str, ...)` still
# pays a dict lookup + cache-compile check per call; compiling once and holding
# the compiled objects removes that from the hot path entirely. Measured on
# 174 sources: ~600 us -> ~180 us per strip.
# ---------------------------------------------------------------------------
_RX_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _rx(pattern: str) -> "re.Pattern[str]":
    compiled = _RX_CACHE.get(pattern)
    if compiled is None:
        compiled = re.compile(pattern)
        _RX_CACHE[pattern] = compiled
    return compiled


_STRIP_CACHE: Dict[str, frozenset] = {}
_STRIP_CACHE_MAX = 4096

# ---------------------------------------------------------------------------
# Primitive vocabulary: (pattern, primitive) pairs.
# Patterns are matched against lowercased text; primitives are the structural
# atoms. Order matters: more specific patterns first.
# ---------------------------------------------------------------------------
ACTION_PRIMITIVES: List[tuple] = [
    # density / weight
    (r"float", "make_float"),
    (r"density|heavy|lightweight|air-entraining|light enough|reduce.*weight", "reduce_density"),
    # "dense" is ambiguous: "low mass per volume" in materials, "many edges per
    # node" in a graph. A bare `dense` matched graph-clustering problems, fired
    # a MATERIALS primitive, dominated action containment, and returned a
    # battery electrolyte for a community-detection query (1 of 15 probes).
    # Handled by the context guard below rather than a lookahead: the guard is
    # explicit and testable.
    (r"\bdense\b", "reduce_density"),
    (r"strong but light|\blight\b.*\bstrong\b|strength.*weight", "reduce_density"),
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
    #
    # Word-bounded and with the "hold on" phrase explicit. The previous pattern
    # was `hold.*on`, where `.*` is unbounded and `on` may match *inside* a word
    # -- so "Prove the base case ... if it holds for n it holds for n+1
    # (mathematical induction)" matched `adhesion` on the "on" in "inductiON",
    # and a maths induction proof was returned as a cross-domain neighbour of
    # "a climbing pad that grips smooth vertical walls". Unbounded cues also let
    # `attach` fire on "detached" (the opposite) and `stick` fire inside
    # "chopstick". Now every cue is a whole word or an explicit stem.
    (
        r"\b(?:adhesi\w*|stick\w*|peel\w*|glue\w*|suction|grip\w*|fasten\w*"
        r"|attach\w*|anchor\w*|mount\w*|hold(?:s|ing)?[ -]+(?:on|onto))\b",
        "adhesion",
    ),
    # Dry / reversible adhesion: the mechanism family that holds WITHOUT glue
    # (gecko setae, hook-and-loop, microstructured surfaces). Glue-based adhesion
    # and dry adhesion are different engineering mechanisms, and keeping them in
    # one atom is what left every member of the family tied at 0.80 for the query
    # "a climbing pad that grips smooth vertical walls without glue" -- so a
    # Post-it note, a stick vacuum, Velcro and flat-pack fasteners were
    # indistinguishable. The negations are length-bounded on purpose: an
    # unbounded `without.*glue` would reintroduce the over-matching bug fixed
    # just above.
    (
        r"\b(?:hook-and-loop|velcro|burr|setae|gecko|microstructur\w*|dry adhesi\w*"
        r"|without (?:any )?(?:glue|adhesive)|no (?:glue|adhesive)|adhesive-free"
        r"|residue-free)\b",
        "adhesion_dry",
    ),
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
    (r"crispr|dna|genome|\bgene\b|edit.*dna", "gene_editing"),
    (r"lab-grown|cell culture|bioreactor|muscle cell|meat", "cell_culture"),
    (r"vertical farm|hydroponic|led lighting|crops.*stacked", "vertical_farming"),
    (r"plastic.*ocean|marine.*plastic|floating.*boom|cleanup", "ocean_cleanup"),
    (r"carbon.*capture|direct air|co2.*sorb|greenhouse", "carbon_capture"),
    (r"quantum.*error|qubit|decoherence|surface code", "quantum_error_correction"),
    # mechanisms
    (r"fastener|open and closed|wearing out|zipper|teeth.*mesh", "reusable_fastener"),
    (r"pre-drill|flat-pack|assemble.*quick|no special tools", "preliminary_action"),
    (r"vacuum|handheld|\bdetach\w*|portable", "segmentation"),
    (r"tank.*fail|storage tank|inspection|safety analysis|molasses", "safety_inspection"),
    (r"pacemaker|heartbeat|electrical pulse|implant", "electrical_implant"),
    (r"x-ray|radiation.*tissue|see inside|bone.*absorb", "penetrating_radiation"),
    (r"shave|blade|razor|cut.*skin", "blade_guard"),
    (r"bifocal|near.*far vision|lens power|glasses", "combined_lens"),
    (r"note.*stick|paper note|remove.*residue", "temporary_adhesive"),
    # mathematics
    (r"\bsum\b.*number|\badd\b.*number|\badd up\b|arithmetic series|consecutive numbers|\btotal\b.*1 to|\bsum\b.*series|\bsum them\b|pair.*terms|\bsum of the\b|total of the first|\btotal\b.*numbers|\badd\b.*whole numbers|\bsum\b.*whole numbers|first .* numbers.*total|gauss", "arithmetic_sum"),
    (r"greatest common divisor|gcd|largest.*factor|prime factor", "gcd"),
    (r"is .* prime\b|prime or not|primality test|probabilistic test|witness.*prime|determine whether.*prime|divisors.*other than|other than 1 and itself|has any divisors|check whether.*number.*prime", "primality_test"),
    (r"\bcount(?:s|ed|ing)?\b.*prime|prime numbers up to|\bsieve\b|primes up to", "prime_sieve"),
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
    (r"look up.*key|look up.*(by|using) (id|a key)|hash table|hash function|constant time.*lookup|instant.*lookup|look up.*instant|\bhash\b|find.*by id instantly|looking up.*instantly", "hashing"),
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
    # --- applied math / discrete / info / logic (wave 2) ---
    (r"determinant.*(large|matrix|exact)|bareiss|fraction-free elimination|exact.*determinant", "exact_determinant"),
    (r"eigenvalue|characteristic polynomial|spectrum|det\(lambda", "eigen_analysis"),
    (r"null space|nullspace|rank.*matrix|row.?reduce|pivot|echelon", "linear_algebra_decomp"),
    (r"linear program|simplex|linear objective|feasible polytope|optimum.*constraint", "linear_programming"),
    (r"pareto|trade.?off|non-dominated|multi.?objective|knee point", "pareto_optimization"),
    (r"bipartite|two.?color|2-color|no edge inside|split.*two groups", "graph_bipartite"),
    (r"every city once|travelling salesman|traveling salesman|hamiltonian|visit.*every.*once|shortest.*tour", "tsp_dp"),
    (r"maximum flow|augmenting path|ford.?fulkerson|source.*sink.*capacity", "max_flow"),
    (r"entropy|uncertainty.*distribution|information content|shannon|bits.*distribution", "shannon_entropy"),
    (r"fourier|frequency component|decompose.*signal|fft|spectral analysis", "fourier_transform"),
    (r"estimate.*parameter|maximum likelihood|likelihood.*maximiz|bayes.*posterior|noisy observations.*estimat", "mle_estimation"),
    (r"nonlinear least squares|levenberg|gauss.?newton|curve.*nonlinear|fit.*nonlinear", "nonlinear_fit"),
    (r"always terminates|program.*terminat|loop.*terminat|well-founded|ranking measure|variant.*decreas|halts", "termination_proof"),
    (r"hoare logic|invariant.*loop|precondition|postcondition|program correctness", "program_logic"),
    (r"satisfiable|sat solver|dpll|boolean formula|unit propagation", "sat_solving"),
    (r"groebner|buchberger|s-polynomial|polynomial.*ideal", "groebner_basis"),
    (r"irreducible.*polynomial|no rational root|square.?free|irreducib", "irreducibility"),
    (r"e-graph|equality saturation|canonical form|simplify.*identity|polynomial identity", "symbolic_simplify"),
    (r"exponential decay|decay.*time|half.?life|rate constant.*fit", "decay_model"),
    (r"balance.*chemical|stoichiometric|conservation.*equation.*element|balance.*reaction", "reaction_balance"),
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
    (r"agree.*same|consensus|\braft\b|paxos|majority.*agree|same.*order|settle.*order", "consensus"),
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
    # --- video / computational imaging (wave 3) ---
    #
    # One action atom per mechanism family, plus an object_video type. Every
    # solution in the video fact base repeats its own trigger words so it
    # strips back to the atom it is meant to match (see SKILL.md step 3).
    #
    # Warp by flow: re-project pixels along a dense optical-flow field rather
    # than re-rendering the scene.
    (r"optical.?flow|flow.?warp\w*|warp\w*.*\bflow\b|\bflow\b.*warp\w*|dense correspondence"
     r"|pixel motion|motion vector\w*|frame interpolation|frame.?blend\w*|temporal interpolation"
     r"|video.?to.?video|\bv2v\b|re.?target\w*.*(pose|motion)|transfer\w*.*(motion|movement|dance)"
     r"|dance move\w*|moves? (from|of) (one|clip a|the source)|follow\w*.*(motion|movement)",
     "warp_by_flow"),
    # Keyframe + bidirectional propagation: reconstruct only some frames and
    # propagate the rest forward/backward along the motion.
    (r"keyframe|key frame|bidirectional|bi-directional|propagat\w*.*(frame|keyframe)"
     r"|every (nth|other|10th|second) frame|sparse keyframe|anchor frame"
     r"|one frame in (ten|two|three|four|five|n)\b|fill (the rest|in the rest|in the gaps)"
     r"|render\w*.*\b(sparse|few) frames?\b",
     "propagate_from_keyframe"),
    # Low-rank / layered decomposition: background layer + sparse residual.
    (r"low.?rank|\brpca\b|robust principal component|sparse residual|layered? decompos\w*"
     r"|background subtraction|background model|median background|foreground layer"
     r"|separate\w*.*(moving|foreground).*(background|static)|static camera"
     r"|\bcut\w* out\b.*(person|subject|actor|moving|shot)|(person|subject|actor).*\bcut out\b"
     r"|without a? ?green.?screen|\bmatting\b|\bmatte\b"
     r"|(person|subject|actor|foreground).*\b(mask|matte)\b"
     r"|segment\w*.*(person|subject|actor|moving)",
     "decompose_low_rank"),
    # Canonicalisation to UV / texture atlas: remove pose/identity variance.
    (r"\buv\b|texture atlas|canonical\w*|unwrapped? mesh|mesh surface map|template mesh"
     r"|normalis\w*.*(pose|identity|viewpoint)|remove\w*.*(pose|viewpoint) variance"
     r"|retarget\w*.*(texture|clothing|garment)|garment\w*.*(atlas|template|uv)"
     r"|clothes?\w*.*(put|apply|swap|transfer)|outfit\w*.*(transfer|swap)"
     r"|(outfit|clothes?|clothing|garment|dress|shirt)\w*.*\b(from|onto|in)\b.*\b(clip|video|person)\b"
     r"|(take|copy|swap|transfer|borrow)\w*.*(outfit|clothes?|clothing|garment|dress|shirt)"
     r"|dress\w*.*(person|character)",
     "canonicalise_to_uv"),
    # Piecewise-affine / mesh warping.
    (r"piecewise.?(affine|linear)|mesh\w*.*warp\w*|warp\w*.*mesh|triangulat\w*.*warp"
     r"|affine.*patch|grid.?(based|mesh) warp|free.?form deformation|\bffd\b|thin.?plate spline"
     r"|\bhomograph\w*|single (global )?transform"
     r"|(arms?|limbs?|body parts?).*move\w* differently",
     "warp_piecewise_affine"),
    # Motion-compensated temporal filtering; flicker suppression.
    (r"flicker\w*|temporal filter\w*|motion.?compensat\w*|denois\w*.*(video|temporal)"
     r"|temporal smoothing|shimmer\w*|jitter\w*.*(frame|render)\w*|pop\w*.*frame|per.?frame render"
     r"|flickering|temporal instability|stable across frames|temporally consistent",
     "blend_temporal_overlap"),
    # Chunked processing + overlap-add blending (streaming seams).
    (r"overlap.?add|overlap.?blend\w*|chunk\w*.*(process|decode|segment)|segment\w*.*(video|clip)"
     r"|blend\w*.*(seam|edge|boundary)|seam\w*.*(blend|visible)|batched windows",
     "blend_temporal_overlap"),
    # ROI / tile allocation.
    (r"\broi\b|region of interest|tile.?based|\btiles?\b|adaptive compute|skip\w*.*(static|unchanged) (pixels|region|tiles)"
     r"|compute allocation|spend\w*.*(gpu|compute|cpu|time).*(face|person|region|area|where)"
     r"|only where it changes|\b(face|person|subject)\b.*(crop|higher resolution|region)"
     r"|(gpu|compute|cpu).*(face|person|subject|wall)|ignore the (empty )?(wall|background)",
     "allocate_by_roi"),
    # Cache reuse of per-frame features.
    (r"cache\w*.*feature|feature cache|reuse\w*.*(embedding|feature|backbone)|cached feature\w*"
     r"|recompute\w*.*(every frame|per frame)|amortis\w*.*(feature|network|embedding)"
     r"|track\w*.*(embedding|feature)"
     r"|(network|model|backbone|network)\w*.*(every frame|each frame|on every frame)"
     r"|(embedding|feature)\w*.*(once|per keyframe)",
     "reuse_cached_features"),
    # Separable / pyramid processing.
    (r"pyramid|\bmipmap|multi.?scale|multi.?resolution|separable|\bcoarse.to.fine\b|\bbox filter"
     r"|downsampl\w*|low.?resolution.*(preview|proxy|draft)",
     "downsample_pyramid"),
    # Quantisation / distillation / small model.
    (r"quantis\w*|quantiz\w*|int8|\bfp16\b|distil\w*|distill\w*|student model"
     r"|\bprun\w*|low.?bit|tiny model|small model|weak hardware|cheap(er)? to (run|process)"
     r"|cheaper to (run|process)|lightweight model|weaker? (machine|hardware|laptop|box|gpu|cpu)",
     "quantise_model"),
    # Bounded-memory streaming pipeline.
    (r"stream\w*.*(pipeline|decode|buffer)|ring buffer|bounded\w*.*(vram|memory|buffer)"
     r"|\boom\b.*(avoid|prevent)|out.?of.?memory|long video|4 hour|\b4h\b|hours? of video"
     r"|frame.?by.?frame|decode.*pipe\w*|pipe.*raw frames|constant memory"
     r"|(clip|video|frames?)\w*.*(does not|doesn't|won't|will not) (fit|load)\b"
     r"|(fit|load)\w*.*(in|into).*(gpu|vram|memory).*(at once|all)",
     "stream_ring_buffer"),
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
    # Video / computational imaging. Kept narrow: 'frame' alone is too common
    # ('frame the problem'), so the surface cues are video-specific and the
    # broad cues require a video word or an explicit rendering/editing term.
    (r"\bvideo\w*|clip\b|footage|film\b|movie\w*|stream\w*.*frame|render\w*.*(frame|scene|shot)"
     r"|frames?\b.*(video|clip|sequence)|per.?frame|frame.?rate|\bfps\b|camera\w*"
     r"|dance move\w*|outfit|clothing|garment|\bpixels?\b|depth.?(map|video)|light.?field"
     r"|neural render\w*|gaussian splat\w*|inpaint\w*.*(video|frame)"
     r"|(motion|pose|dance)\w*.*(reference|source|driver) clip|clip [ab]\b|one clip|another clip"
     r"|homograph\w*|per.?frame|consecutive frames|frames? differ"
     r"|(character|person)\w* (in|from) (a |the )?scene|actor\b"
     r"|upscal\w*|super.?resolution|video quality",
     "object_video"),
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


# Words that, when they follow "dense", mark the GRAPH sense ("dense graph",
# "dense connections between clusters") rather than the materials sense
# ("dense concrete"). Kept explicit and tested rather than embedded in a
# lookahead, so the rule is inspectable.
_GRAPH_SENSE_AFTER_DENSE = (
    "graph", "graphs", "network", "networks", "connect", "connected",
    "connections", "connection", "link", "links", "cluster", "clusters",
    "clustering", "community", "communities", "subgraph", "matrix",
    "edge", "edges", "subarray", "sub-array",
)

# Words that may sit between "dense" and the noun it describes: "dense AND
# sparse", "dense IN A graph", "dense INTERNAL connections".
_SKIP_BEFORE_NOUN = {
    "and", "or", "but", "yet", "nor", "with", "in", "on", "at", "of", "to",
    "is", "are", "was", "were", "be", "being", "very", "more", "most",
    "less", "least", "highly", "extremely", "fairly", "quite", "enough",
    "internal", "external", "mutual", "inter", "node", "nodes", "node-level",
    "cross", "across", "between", "within", "among", "both", "each", "every",
    "sparse", "tight", "loose", "strong", "weak", "local", "global",
}


def _dense_means_mass(text_lower: str) -> bool:
    """True when a 'dense' occurrence should count as the materials primitive.

    Returns False only when EVERY occurrence of the word 'dense' is directly
    followed by a graph-context noun. One materials-sense use is enough to
    keep the primitive.
    """
    for m in re.finditer(r"\bdense\b", text_lower):
        # Walk forward past conjunctions/adjectives to the head noun.
        words = re.findall(r"[\w-]+", text_lower[m.end():])[:6]
        head = ""
        for w in words:
            if w in _SKIP_BEFORE_NOUN:
                continue
            head = w
            break
        if head not in _GRAPH_SENSE_AFTER_DENSE:
            return True
    return False


def strip_to_primitives(text: str) -> Set[str]:
    """Reduce text to its structural primitive set (action + object-type atoms).

    Three layers, most-specific first: domain ACTION primitives, domain OBJECT
    types, then GENERIC action + QUALITY-attribute atoms. The generic layers
    guarantee every problem yields at least one action + one attribute atom, so
    structural matching is never starved (measured: 100 domain regexes still left
    'reduce p99 latency' with only an object atom and no action).
    """
    key = text
    hit = _STRIP_CACHE.get(key)
    if hit is not None:
        return set(hit)

    tl = text.lower()
    prims: Set[str] = set()
    for pattern, prim in ACTION_PRIMITIVES:
        if prim == "reduce_density" and not _dense_means_mass(tl):
            # `dense` present, but in its graph sense -- skip this primitive.
            # Other materials cues (density/heavy/lightweight) still count.
            if not re.search(
                r"density|heavy|lightweight|air-entraining|light enough|reduce.*weight", tl
            ):
                continue
        if _rx(pattern).search(tl):
            prims.add(prim)
    for pattern, prim in OBJECT_PRIMITIVES:
        if _rx(pattern).search(tl):
            prims.add(prim)
    # Generic layers (added after specific ones; both may contribute).
    for pattern, prim in GENERIC_ACTION_PRIMITIVES:
        if _rx(pattern).search(tl):
            prims.add(prim)
    for pattern, prim in QUALITY_ATTRIBUTES:
        if _rx(pattern).search(tl):
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

    if len(_STRIP_CACHE) >= _STRIP_CACHE_MAX:
        _STRIP_CACHE.clear()
    _STRIP_CACHE[key] = frozenset(prims)
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


def _strip_sources(source_solutions: List[Dict[str, str]]) -> List[Set[str]]:
    """Strip every source solution to primitives, memoized within the call.

    Duplicate solution texts are common in the index (up to 8 copies of one
    entry were measured), so memoizing by text also collapses that work.
    """
    memo: Dict[str, Set[str]] = {}
    out: List[Set[str]] = []
    for sol in source_solutions:
        text = sol["solution"]
        prims = memo.get(text)
        if prims is None:
            prims = strip_to_primitives(text)
            memo[text] = prims
        out.append(prims)
    return out


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
    # Strip each SOURCE once, not twice. The original code called
    # strip_to_primitives on every solution in the IDF pass AND again in the
    # scoring pass (349 regex passes for 174 sources, ~0.6 ms each), which made
    # a single retrieval cost ~200 ms. Memoizing by content is a pure
    # performance change: same inputs, same outputs.
    source_prims = _strip_sources(source_solutions)
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
    for s_prims in source_prims:
        s_actions = {p for p in s_prims if not p.startswith("object_")}
        for a in s_actions:
            action_doc_freq[a] = action_doc_freq.get(a, 0) + 1
    n_docs = max(len(source_solutions), 1)

    scored = []
    for sol, s_prims in zip(source_solutions, source_prims):
        s_prims = set(s_prims)
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
    #
    # difflib.SequenceMatcher is quadratic and ran over EVERY source (76 ms of
    # a 200 ms call). Near-identical primitive scores are what needs breaking,
    # so only the top ties are rescored: cheap token-overlap first, difflib
    # only inside the tie block that can still affect the returned top-k.
    import difflib

    tie_floor = 0.0
    if len(scored) > top_k:
        tie_floor = scored[top_k - 1]["primitive_overlap"]
    target_words = set(re.findall(r"[a-z0-9]+", target_text.lower()))
    for s in scored:
        if s["primitive_overlap"] >= tie_floor:
            src_words = set(re.findall(r"[a-z0-9]+", s.get("content", "").lower()))
            s["_text_sim"] = (
                difflib.SequenceMatcher(None, target_text.lower(), s.get("content", "").lower()).ratio()
                if len(target_words & src_words) >= 3
                else 0.0
            )
        else:
            s["_text_sim"] = 0.0
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
