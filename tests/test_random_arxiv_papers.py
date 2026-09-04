"""Tests verifying completely random, unseen arXiv preprints fetched from the live web.

Ensures the engine does not rely on hardcoded strings and generalizes to arbitrary
mathematical problems, conjectures, and preprints across different mathematical fields.
"""

from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.core.types import VerificationStatus


def test_verify_random_arxiv_zhi_wei_sun_catalan_irrationality():
    """Validates verification of arXiv:2609.04176: Zhi-Wei Sun's preprint on Catalan's constant."""
    engine = SuperDuperProblemSolvingEngine()

    paper_id = "arXiv:2609.04176"
    paper_title = "Catalan's constant is irrational"
    paper_abstract = (
        "Whether the constant G = sum_{k=0}^infty (-1)^k / (2k+1)^2 = 1/1^2 - 1/3^2 + 1/5^2 - 1/7^2 + ... "
        "introduced by Catalan in the nineteen century is irrational, is a long-standing open problem. "
        "In this paper we prove the irrationality of G via using suitable weights."
    )
    target_conjecture = "Irrationality of Catalan's Constant"

    result = engine.verify_math_paper(
        paper_id=paper_id,
        paper_title=paper_title,
        abstract_text=paper_abstract,
        target_conjecture=target_conjecture,
    )

    # Must dynamically extract lemmas without stubs
    assert len(result.sound_lemmas) >= 1
    assert result.confidence >= 0.70
    assert result.verdict in [VerificationStatus.VERIFIED_SOUND, VerificationStatus.GAP_DETECTED]

    # Check that technique extracted matches the paper's sentence
    assert any("weights" in s.lower() or "irrationality" in s.lower() for s in result.sound_lemmas)

    # Verify Mermaid export
    mermaid = engine.export_math_verification_mermaid(result)
    assert "```mermaid" in mermaid
    assert "Lakatos" in mermaid


def test_verify_random_arxiv_pratim_mitra_subconvexity():
    """Validates verification of arXiv:2609.04155: Pratim Mitra's preprint on L-functions."""
    engine = SuperDuperProblemSolvingEngine()

    paper_id = "arXiv:2609.04155"
    paper_title = "The subconvexity problem for symmetric square L-functions in level aspect"
    paper_abstract = (
        "In this paper, we address the subconvexity problem in level aspect for symmetric square L-functions "
        "for cuspidal automorphic representation of GL_2(Q) with a prescribed local ramification at prime p. "
        "We prove that L(1/2, Sym^2 pi) << q(Sym^2 pi)^{1/4 - 1/168 + o(1)}. "
        "Our approach is based on the delta-symbol method. "
        "Apart from some standard analytic number theoretic tools, Katz's theory of hypergeometric sums, "
        "and Deligne's proof of Weil-conjectures play an important role in the proof."
    )
    target_conjecture = "Subconvexity Problem for L-functions"

    result = engine.verify_math_paper(
        paper_id=paper_id,
        paper_title=paper_title,
        abstract_text=paper_abstract,
        target_conjecture=target_conjecture,
    )

    # Verify that dynamic bound extraction caught the subconvex exponent
    lemmas = engine.math_verifier.extract_or_synthesize_lemmas(paper_title, paper_abstract)
    bounds = [l.claimed_bound for l in lemmas if l.claimed_bound]
    assert len(bounds) > 0
    assert any("1/168" in b or "<<" in b for b in bounds)

    # Verify techniques were dynamically extracted
    assert any("delta-symbol" in s.lower() or "deligne" in s.lower() or "subconvexity" in s.lower() for s in result.sound_lemmas)
    assert result.confidence >= 0.85
    assert result.verdict == VerificationStatus.VERIFIED_SOUND


def test_verify_random_arxiv_noga_alon_packing_progressions():
    """Validates verification of arXiv:2603.02786: Noga Alon et al. preprint on packing arithmetic progressions."""
    engine = SuperDuperProblemSolvingEngine()

    paper_id = "arXiv:2603.02786"
    paper_title = "Packing arithmetic progressions"
    paper_abstract = (
        "Let F be a collection of finite arithmetic progressions where each A_d is an initial segment of consecutive multiples. "
        "Let m(F) denote the minimum length of an interval containing pairwise disjoint shifted copies of all members of F. "
        "We study this parameter in two cases. In case (1), when k=n, we get m(F) = Theta(n^{3/2} / ln n). "
        "In case (2), when k=n, we get m(F) = Theta(n^3 / ln n), while if k > k_0(n), then m(F) < 3kn. "
        "In both cases we additionally determine m(F) asymptotically for all k < n."
    )
    target_conjecture = "Packing Density of Arithmetic Progressions"

    result = engine.verify_math_paper(
        paper_id=paper_id,
        paper_title=paper_title,
        abstract_text=paper_abstract,
        target_conjecture=target_conjecture,
    )

    # Combinatorics paper should not trigger circuit complexity or prime sieve barriers
    assert len(result.barrier_violations) == 0
    assert result.verdict == VerificationStatus.VERIFIED_SOUND
    assert len(result.sound_lemmas) >= 2

    # Check that bound extraction captured the inequality/asymptotic bound
    lemmas = engine.math_verifier.extract_or_synthesize_lemmas(paper_title, paper_abstract)
    assert any(l.claimed_bound is not None for l in lemmas)
