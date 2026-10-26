"""Regression tests for reasoning/representation defects found by auditing.

Each test here pins a behaviour that was *measured wrong* before the fix:

* SME scored role-inverted relations at 1.0 and silently overwrote entity
  bindings (no one-to-one constraint).
* Entity substitution rewrote substrings, turning "gasket" into "hydrogenket".
* The boundary/null-space kernels hardcoded a 384-d vector, so every embedder
  with a different width raised a broadcast error.
* TOC evaporation ranked on hashed-cosine noise, so an unrelated sentence could
  win the cloud.
"""

from __future__ import annotations

import numpy as np

from super_solver.core.vector_operators import (
    ContinuousBoundaryDiagnosticKernel,
    NullSpaceAssumptionProjector,
)
from super_solver.frameworks.gentner_sme import RelationalStatement, StructureMappingEngine
from super_solver.frameworks.toc_cloud import EvaporatingCloud, TOCEngine
from super_solver.vectorize.entity_substitution import substitute_entity

# ---------------------------------------------------------------------------
# Gentner SME: one-to-one constraint
# ---------------------------------------------------------------------------


def test_sme_does_not_overwrite_entity_bindings():
    """A base entity must map to exactly one target entity.

    Before the fix, ``sun`` was bound to ``virus`` by the first statement and
    then silently rebound to ``magnet`` by the second, so the reported mapping
    was internally inconsistent and depended on statement order.
    """
    sme = StructureMappingEngine()
    base = [
        RelationalStatement(relation="causes", entity_a="sun", entity_b="heat", domain="solar"),
        RelationalStatement(relation="orbits", entity_a="sun", entity_b="planet", domain="solar"),
    ]
    target = [
        RelationalStatement(relation="causes", entity_a="virus", entity_b="fever", domain="bio"),
        RelationalStatement(relation="orbits", entity_a="magnet", entity_b="iron", domain="phys"),
    ]
    res = sme.align_systems(base, target)
    mapping = res["entity_mappings"]
    # sun was committed to virus; the second statement cannot rebind it.
    assert mapping["sun"] == "virus", mapping
    # The map must be injective.
    assert len(set(mapping.values())) == len(mapping), mapping
    # The second statement could not be aligned consistently, so it is unmatched.
    assert res["systematic_match_count"] == 1


def test_sme_alignment_is_order_independent():
    sme = StructureMappingEngine()
    base = [
        RelationalStatement(relation="causes", entity_a="sun", entity_b="heat", domain="d"),
        RelationalStatement(relation="orbits", entity_a="sun", entity_b="planet", domain="d"),
    ]
    target = [
        RelationalStatement(relation="causes", entity_a="virus", entity_b="fever", domain="t"),
        RelationalStatement(relation="orbits", entity_a="magnet", entity_b="iron", domain="t"),
    ]
    forward = sme.align_systems(base, target)["entity_mappings"]
    backward = sme.align_systems(list(reversed(base)), list(reversed(target)))["entity_mappings"]
    assert len(forward) == len(backward)
    assert len(set(forward.values())) == len(forward)


def test_sme_still_aligns_the_classic_analogy():
    """The solar-system -> atom mapping must survive the constraint."""
    sme = StructureMappingEngine()
    base = [
        RelationalStatement(relation="attracts", entity_a="sun", entity_b="planet", domain="solar"),
        RelationalStatement(
            relation="revolves_around", entity_a="planet", entity_b="sun", domain="solar"
        ),
    ]
    target = [
        RelationalStatement(
            relation="attracts", entity_a="nucleus", entity_b="electron", domain="atom"
        ),
        RelationalStatement(
            relation="revolves_around", entity_a="electron", entity_b="nucleus", domain="atom"
        ),
    ]
    res = sme.align_systems(base, target)
    assert res["systematic_match_count"] == 2
    assert res["entity_mappings"]["sun"] == "nucleus"
    assert res["entity_mappings"]["planet"] == "electron"
    assert res["isomorphic_transfer_viable"] is True


# ---------------------------------------------------------------------------
# Entity substitution: word boundaries + single application
# ---------------------------------------------------------------------------


def test_entity_substitution_respects_word_boundaries():
    """"gas" must not be found inside "gasket"."""
    out = substitute_entity(
        "Use a gasket to seal the gas line and keep the gas from leaking",
        "how to ship hydrogen safely",
    )
    assert "gasket" in out, out
    assert "hydrogenket" not in out, out
    assert "hydrogen line" in out


def test_entity_substitution_applies_only_one_mapping():
    """The adapter used to fire every known mapping in a single pass."""
    out = substitute_entity(
        "The deck uses gas to reduce material use",
        "build a roof that uses hydrogen and panel",
    )
    # Only the gas -> hydrogen mapping should have been applied.
    assert "deck" in out, out
    assert "material" in out, out
    assert "hydrogen" in out, out


# ---------------------------------------------------------------------------
# Vector operators: dimension must follow the caller's embedder
# ---------------------------------------------------------------------------


def test_boundary_kernel_accepts_any_embedding_dimension():
    """Hardcoded 384 crashed for rich (512), neural (256), ollama (768), hybrid (1152)."""
    rng = np.random.default_rng(0)
    for dim in (256, 384, 512, 768, 1152):
        kernel = ContinuousBoundaryDiagnosticKernel(
            [rng.normal(size=dim)], [rng.normal(size=dim)]
        )
        out = kernel.evaluate_candidate(rng.normal(size=dim))
        assert isinstance(out, dict)


def test_boundary_kernel_dim_defaults_sensibly_when_empty():
    """With nothing to infer from it must still build a usable (zero) field."""
    kernel = ContinuousBoundaryDiagnosticKernel([], [])
    assert kernel.dim == 384  # documented legacy default
    assert ContinuousBoundaryDiagnosticKernel([], [], dim=512).dim == 512


def test_null_space_projector_follows_vector_dimension():
    rng = np.random.default_rng(0)
    for dim in (256, 512, 768):
        proj = NullSpaceAssumptionProjector([rng.normal(size=dim)])
        v = rng.normal(size=dim)
        projected = proj.project_to_null_space(v)
        assert projected.shape == (dim,)
        assert 0.0 <= proj.assumption_overlap(v) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# TOC evaporating cloud: screen on the conflicting means
# ---------------------------------------------------------------------------


def _cloud():
    return EvaporatingCloud(
        objective="deliver software fast",
        requirement_a="innovate",
        requirement_b="stability",
        prerequisite_d="ship frequently",
        prerequisite_d_prime="freeze changes",
    )


def test_evaporate_rejects_an_injection_unrelated_to_the_conflict():
    """Measured before the fix: a cooking recipe won the cloud."""
    res = TOCEngine().evaporate(
        _cloud(),
        [
            "CI/CD pipeline with automated tests",
            "freeze development",
            "Use a completely unrelated cooking recipe with basil and tomatoes",
        ],
    )
    assert "cooking recipe" not in (res["winning_injection"] or "")
    by_inj = {r["injection"]: r for r in res["rankings"]}
    assert by_inj[
        "Use a completely unrelated cooking recipe with basil and tomatoes"
    ]["eligible"] is False


def test_evaporate_flags_a_pole_restatement_as_a_collapse():
    """"freeze development" restates prerequisite D'; it is a collapse, not a resolution."""
    res = TOCEngine().evaporate(_cloud(), ["freeze development", "CI/CD pipeline with tests"])
    by_inj = {r["injection"]: r for r in res["rankings"]}
    assert by_inj["freeze development"]["restates_pole"] is True
    assert by_inj["freeze development"]["eligible"] is False


def test_evaporate_prefers_the_balanced_injection_over_the_poles():
    res = TOCEngine().evaporate(
        _cloud(),
        [
            "ignore testing entirely",
            "freeze development",
            "CI/CD pipeline with automated tests",
        ],
    )
    assert res["resolved"] is True
    assert res["winning_injection"] == "CI/CD pipeline with automated tests"


# ---------------------------------------------------------------------------
# Lakatos barrier audit: must fail CLOSED
# ---------------------------------------------------------------------------


def test_lakatos_audit_fails_closed_on_claims_outside_its_scope():
    """A claim that engages no known barrier must be INCONCLUSIVE, not SOUND.

    Every input sentence becomes a "lemma", and a lemma matching no barrier was
    recorded as sound -- so an all-clear was reported as VERIFIED_SOUND. That is
    fail-open, and it is how "1+1=3", "The moon is made of cheese" and
    "Trisecting an angle with straightedge and compass" (proven impossible in
    1837) all came back as VERIFIED_SOUND with confidence 0.90.
    """
    from super_solver.core.types import VerificationStatus
    from super_solver.frameworks.math_verification import LakatosProofVerificationEngine

    engine = LakatosProofVerificationEngine()
    for claim in (
        "1+1=3",
        "The moon is made of cheese",
        "Trisecting an angle with straightedge and compass",
    ):
        res = engine.verify_mathematical_paper(
            paper_id="x", paper_title=claim, abstract_text=claim, target_conjecture=claim
        )
        assert res.verdict == VerificationStatus.INCONCLUSIVE, (claim, res.verdict)
        assert res.confidence == 0.0, claim


def test_lakatos_audit_still_refutes_a_real_barrier_violation():
    """Failing closed must not blunt the falsifier."""
    from super_solver.core.types import VerificationStatus
    from super_solver.frameworks.math_verification import LakatosProofVerificationEngine

    claim = "A proof that P != NP via monotone circuit approximations"
    res = LakatosProofVerificationEngine().verify_mathematical_paper(
        paper_id="x", paper_title=claim, abstract_text=claim, target_conjecture=claim
    )
    assert res.verdict == VerificationStatus.REFUTED_FLAWED
    assert res.barrier_violations


# ---------------------------------------------------------------------------
# Tournament: inflected self-contradiction + honest evidence reporting
# ---------------------------------------------------------------------------


def test_tournament_flags_inflected_self_contradiction():
    """The check matched "increase" but not "increasing", so inflected
    self-contradictions survived while uninflected ones were caught."""
    from super_solver.frameworks.tournament import (
        DialecticalTournamentEngine,
        TournamentVerdict,
    )

    res = DialecticalTournamentEngine().run_match(
        "Increasing the antibiotic dose reduces the bacterial survival rate"
    )
    assert res.survived is False
    assert res.verdict == TournamentVerdict.WEAKENED_INCONSISTENT


def test_tournament_reports_evidence_coverage_and_does_not_overclaim():
    """A SURVIVED_ROBUST verdict must state what was actually tested.

    The summary used to read "Resisted Red Team cross-examination across all
    rounds" even when no round had any traction on the claim.
    """
    from super_solver.frameworks.tournament import DialecticalTournamentEngine

    res = DialecticalTournamentEngine().run_match("The moon is made of cheese")
    assert 0.0 <= res.evidence_coverage <= 1.0
    assert res.evidence_coverage == len(res.checks_applied) / len(res.rounds)
    if not res.checks_applied:
        assert "UNTESTED" in res.summary
        assert "Resisted" not in res.summary


def test_tournament_coverage_reports_which_checks_applied():
    from super_solver.frameworks.tournament import DialecticalTournamentEngine

    res = DialecticalTournamentEngine().run_match(
        "Proving P != NP by monotone circuit approximation bounds",
        boundary_is_not="a polynomial-time decision procedure for SAT",
    )
    assert "barrier_audit" in res.checks_applied
    assert "boundary_consistency" in res.checks_applied
    assert res.evidence_coverage > 0.0


# ---------------------------------------------------------------------------
# TRIZ: use the real matrix when the text names its parameters
# ---------------------------------------------------------------------------


def test_triz_infers_matrix_parameters_from_free_text():
    """When the text names a known contradiction pair, return the matrix cell."""
    from super_solver.frameworks.triz_engine import TRIZEngine

    engine = TRIZEngine()
    params = engine.infer_contradiction_parameters(
        "Increase the strength of a bracket without making it heavier"
    )
    assert params == ["strength", "weight"], params

    res = engine.suggest_principles("Increase the strength of a bracket without making it heavier")
    assert res[0]["source"] == "altshuller_matrix_inferred"
    assert res[0]["matched_parameters"] == ["strength", "weight"]
    # Real matrix data, not vector noise.
    assert {r["principle_id"] for r in res} <= {40, 8, 1, 15}


def test_triz_parameter_order_follows_the_text():
    """The earlier-mentioned parameter is the one being improved."""
    from super_solver.frameworks.triz_engine import TRIZEngine

    engine = TRIZEngine()
    fwd = engine.suggest_principles("Increase strength without adding weight")
    rev = engine.suggest_principles("Make it lighter without losing strength")
    assert fwd[0]["matched_parameters"] == ["strength", "weight"]
    assert rev[0]["matched_parameters"] == ["weight", "strength"]


def test_triz_falls_back_to_vector_similarity_when_no_parameters_are_named():
    from super_solver.frameworks.triz_engine import TRIZEngine

    res = TRIZEngine().suggest_principles(
        "How to make a canoe hull that floats despite being made of dense concrete"
    )
    assert res and res[0]["source"] == "vector_similarity"


# ---------------------------------------------------------------------------
# Primitive vocabulary: regex hazards + dry-adhesion discrimination
# ---------------------------------------------------------------------------


def test_unbounded_adhesion_pattern_no_longer_matches_induction():
    """`hold.*on` let `on` match inside a word.

    Measured before the fix: "Prove the base case, then show that if it holds
    for n it holds for n+1 (mathematical induction)" produced the `adhesion`
    primitive, because `hold.*on` matched the "on" in "inductiON". A maths
    induction proof was then returned as a cross-domain neighbour of "a climbing
    pad that grips smooth vertical walls".
    """
    from super_solver.vectorize.primitives import strip_to_primitives

    prims = strip_to_primitives(
        "Prove the base case, then show that if it holds for n it holds for n+1 "
        "(mathematical induction)"
    )
    assert "adhesion" not in prims, prims
    assert "induction" in prims


def test_adhesion_still_fires_on_real_adhesion_phrases():
    from super_solver.vectorize.primitives import strip_to_primitives

    for text in (
        "hold on to the surface",
        "attach the bracket to the wall",
        "stick the label to the box",
        "grip the smooth wall",
    ):
        assert "adhesion" in strip_to_primitives(text), text


def test_bare_primitive_tokens_do_not_match_inside_longer_words():
    """`gene` fired on "general"/"generate"; `raft` fired on "aircraft"/"draft"."""
    from super_solver.vectorize.primitives import strip_to_primitives

    assert "gene_editing" not in strip_to_primitives("In general the system generates reports")
    assert "consensus" not in strip_to_primitives("Draft a plan and graft it onto the aircraft")
    # ...while the real cues still work.
    assert "gene_editing" in strip_to_primitives("Edit the DNA of a living organism")
    assert "consensus" in strip_to_primitives("Raft consensus settles the log order")


def test_dry_adhesion_is_distinct_from_glue_adhesion():
    from super_solver.vectorize.primitives import strip_to_primitives

    dry = strip_to_primitives("Mimic the hook-and-loop structure of burrs to stick to fabric")
    assert "adhesion_dry" in dry and "adhesion" in dry

    glue = strip_to_primitives("Use a low-tack adhesive that was originally a failed glue")
    assert "adhesion" in glue
    assert "adhesion_dry" not in glue, glue


def test_cross_domain_retrieval_picks_the_dry_adhesion_analog():
    """The end-to-end effect of the two fixes above.

    Before: a Post-it note, a stick vacuum, Velcro and flat-pack fasteners all
    scored exactly 0.80 for "a climbing pad that grips smooth vertical walls
    without glue", and a maths induction proof scored 0.80 too. The dry-adhesion
    analog must now win outright and the unrelated hits must be separated out.
    """
    from super_solver.vectorize import vectorization_service
    from super_solver.vectorize.domain_knowledge import build_domain_knowledge_index

    # Under pytest the vector index is an empty :memory: database (the conftest
    # fixture clears it around every test), so seed the committed fact base here.
    build_domain_knowledge_index(vectorization_service)

    hits = vectorization_service.primitive_analogize(
        "How to make a climbing pad that grips smooth vertical walls without glue",
        top_k=5,
    )
    assert hits, "no analogs returned"
    assert "hook-and-loop" in hits[0]["content"], hits[0]["content"]
    # The glue-based and unrelated analogs must be clearly separated from the top.
    assert hits[1]["primitive_overlap"] < hits[0]["primitive_overlap"] - 0.2
    assert not any("induction" in h["content"] for h in hits), "induction leak"
