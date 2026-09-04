"""Tests mathematical proof verification on unverified/controversial arXiv preprints.

Tests:
1. Refuting an unverified flawed paper: Norbert Blum's claimed P != NP proof (arXiv:1708.03486)
   via Lakatos counterexamples and the Tardos Monotone Circuit Barrier.
2. Verifying a breakthrough paper: Terence Tao's Collatz map paper (arXiv:1909.03562)
   via logarithmic drift invariance and Conway undecidability barrier auditing.
"""

from super_solver.core.types import VerificationStatus
from super_solver.engine import SuperDuperProblemSolvingEngine


def test_verify_unverified_p_vs_np_paper_norbert_blum():
    """Tests auditing and refuting Norbert Blum's unverified P != NP preprint (arXiv:1708.03486)."""
    engine = SuperDuperProblemSolvingEngine()

    paper_id = "arXiv:1708.03486"
    paper_title = "A Solution to the P versus NP Problem"
    paper_abstract = (
        "We consider the Clique problem and standard Boolean circuits with AND, OR, and NOT gates. "
        "Using the approximation method of Berg and Ulfberg developed for monotone networks, "
        "we bound the difference between the target function and its inductive approximations. "
        "Theorem 6 establishes an exponential lower bound of 2^{Omega(n^{1/6})} for the size "
        "of any Boolean circuit computing the clique function, thereby deducing that P != NP."
    )
    target_conjecture = "P versus NP"

    result = engine.verify_math_paper(
        paper_id=paper_id,
        paper_title=paper_title,
        abstract_text=paper_abstract,
        target_conjecture=target_conjecture,
    )

    # 1. Verify that the verdict correctly refutes the flawed paper
    assert result.verdict == VerificationStatus.REFUTED_FLAWED
    assert result.confidence >= 0.95

    # 2. Verify that the exact barrier violation was identified (Tardos Monotone Barrier)
    assert any("Tardos" in b for b in result.barrier_violations)

    # 3. Verify that the concrete counterexample was synthesized
    assert any("Tardos" in c for c in result.counterexamples)
    assert any("Theorem 6" in f for f in result.flawed_lemmas)

    # 4. Verify Mermaid visual export
    mermaid = engine.export_math_verification_mermaid(result)
    assert "```mermaid" in mermaid
    assert "Falsified Step" in mermaid
    assert "Tardos" in mermaid
    assert "REFUTED FLAWED" in mermaid


def test_verify_terence_tao_collatz_breakthrough_paper():
    """Tests verifying Terence Tao's breakthrough Collatz preprint (arXiv:1909.03562)."""
    engine = SuperDuperProblemSolvingEngine()

    paper_id = "arXiv:1909.03562"
    paper_title = "Almost all orbits of the Collatz map attain almost bounded values"
    paper_abstract = (
        "Let Collatz denote the Syracuse mapping on odd integers. For any function R(N) -> infinity, "
        "we prove that almost all integers n in [1, N] have min_{k} Collatz^k(n) < R(n) in the sense of "
        "logarithmic density. We establish uniform geometric decay by modeling 2-adic valuations "
        "as a skew random walk with strictly negative logarithmic drift log2(3) - 2 < 0 and "
        "proving non-concentration of the associated renewal measures."
    )
    target_conjecture = "Collatz 3x+1 Conjecture"

    result = engine.verify_math_paper(
        paper_id=paper_id,
        paper_title=paper_title,
        abstract_text=paper_abstract,
        target_conjecture=target_conjecture,
    )

    # 1. Verify that the verdict correctly verifies the sound breakthrough paper
    assert result.verdict == VerificationStatus.VERIFIED_SOUND
    assert result.confidence >= 0.90

    # 2. Verify that no barrier violations were triggered because 'almost all' avoids Conway's undecidability
    assert len(result.barrier_violations) == 0
    assert len(result.counterexamples) == 0

    # 3. Verify that the negative logarithmic drift was confirmed
    assert any("drift" in s.lower() for s in result.sound_lemmas)

    # 4. Verify Mermaid visual export
    mermaid = engine.export_math_verification_mermaid(result)
    assert "```mermaid" in mermaid
    assert "Sound Step" in mermaid
    assert "VERIFIED SOUND" in mermaid
