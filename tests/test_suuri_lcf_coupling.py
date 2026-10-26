"""Test Suuri-chan Milner LCF formal micro-kernel coupling with math_verification."""
from __future__ import annotations

import pytest

from super_solver.frameworks.math_verification import LakatosProofVerificationEngine as MathematicalProofVerifier
from super_solver.core.types import VerificationStatus


def test_verify_with_lcf_microkernel_refl():
    verifier = MathematicalProofVerifier()
    res = verifier.verify_with_lcf_microkernel("a + b = a + b")
    assert res is not None
    assert res["is_valid"] is True
    assert res["rule"] == "refl"
    assert "steps_count" in res
    assert res["steps_count"] >= 1


def test_verify_with_lcf_microkernel_symbolic():
    verifier = MathematicalProofVerifier()
    res = verifier.verify_with_lcf_microkernel("1/3 + 5/7")
    assert res is not None
    assert res["is_valid"] is True
    assert res["value"] == "22/21"


def test_verify_mathematical_paper_with_formal_cert():
    verifier = MathematicalProofVerifier()
    # Paper claiming an identity that can be verified symbolically or mechanically
    result = verifier.verify_mathematical_paper(
        paper_id="test.ident.01",
        paper_title="On the exact value of 1/3 + 5/7",
        abstract_text="We establish that 1/3 + 5/7 reduces unconditionally to 22/21.",
        target_conjecture="1/3 + 5/7",
    )
    assert result.paper_id == "test.ident.01"
    assert result.formal_certificate is not None
    assert result.lcf_verified is True
    assert any("FORMAL LCF CERTIFICATE" in t for t in result.reasoning_trace)
