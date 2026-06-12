"""Tests for the English->verified-program requirements pipeline.

The LLM proposer is mocked (a fixed IR) so the suite needs no API key; the
nsynth backend is real (tests skip if the release binary is absent). Covers:
IR validation/serialization, the train/holdout split + holdout verification,
the reference cross-check, honest refusal / unsupported / no-examples paths,
and the deterministic proposer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ncpu.requirements.ir import IoExample, ParamSpec, RequirementsIR
from ncpu.requirements.pipeline import resolve
from ncpu.requirements.proposer import (
    DeterministicProposer,
    Proposer,
    ProposerError,
)

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "nsynth" / "target" / "release" / "mog_synth"
backend_required = pytest.mark.skipif(
    not BACKEND.is_file(), reason="mog_synth release binary not built"
)


class MockProposer:
    """Returns a preset IR regardless of input — stands in for the LLM."""

    def __init__(self, ir: RequirementsIR):
        self.ir = ir

    def propose(self, english: str) -> RequirementsIR:
        return self.ir


def _abs_diff_ir(reference: str | None = None) -> RequirementsIR:
    return RequirementsIR(
        entry_point="abs_diff",
        description="Absolute difference of two integers.",
        params=[ParamSpec("a", "i64"), ParamSpec("b", "i64")],
        return_type="i64",
        io_examples=[
            IoExample([10, 3], 7),
            IoExample([3, 10], 7),
            IoExample([5, 5], 0),
            IoExample([-4, 2], 6),
            IoExample([0, 9], 9),
            IoExample([8, 8], 0),
        ],
        invariants=["result is non-negative", "abs_diff(a,b) == abs_diff(b,a)"],
        edge_cases=["equal inputs -> 0", "negative inputs"],
        reference_impl=reference,
    )


# ---------------------------------------------------------------------------
# IR unit tests
# ---------------------------------------------------------------------------

def test_ir_roundtrip_and_signature():
    ir = _abs_diff_ir()
    ir2 = RequirementsIR.from_dict(ir.to_dict())
    assert ir2.entry_point == "abs_diff"
    assert ir2.signature_str() == "fn abs_diff(a: i64, b: i64) -> i64"
    assert ir2.validate() == []
    assert ir2.synth_supported()


def test_ir_unsupported_when_output_is_list():
    ir = RequirementsIR(
        entry_point="rolling_max",
        description="rolling maximum",
        params=[ParamSpec("xs", "[i64]")],
        return_type="[i64]",
        io_examples=[IoExample([[1, 3, 2]], [1, 3, 3])],
    )
    assert ir.validate() == []  # structurally fine
    assert not ir.synth_supported()  # but synth can't target list output


def test_ir_validate_flags_missing_examples():
    ir = RequirementsIR(entry_point="f", description="x")
    assert any("no io_examples" in p for p in ir.validate())


def test_proposer_protocol_isinstance():
    assert isinstance(MockProposer(_abs_diff_ir()), Proposer)
    assert isinstance(DeterministicProposer(), Proposer)


# ---------------------------------------------------------------------------
# pipeline — needs the real synthesizer
# ---------------------------------------------------------------------------

@backend_required
def test_resolve_synthesizes_and_generalizes():
    result = resolve("the absolute difference", proposer=MockProposer(_abs_diff_ir()))
    assert result.status == "synthesized", result.notes
    assert result.holdout_count >= 1
    # synthesized program must reproduce EVERY held-out example
    assert result.holdout_passed == result.holdout_count
    assert result.transpiled.get("python")
    assert result.confidence in ("medium", "high")


@backend_required
def test_resolve_high_confidence_with_agreeing_reference():
    ref = "def abs_diff(a, b):\n    return abs(a - b)\n"
    result = resolve("abs diff", proposer=MockProposer(_abs_diff_ir(reference=ref)))
    assert result.status == "synthesized"
    assert result.synth_vs_reference_agree is True
    assert result.reference_holdout_passed == result.holdout_count
    assert result.confidence == "high"


@backend_required
def test_resolve_refuses_on_inconsistent_examples():
    # an impossible mapping: no function reproduces all of these
    ir = RequirementsIR(
        entry_point="impossible",
        description="no function satisfies these",
        params=[ParamSpec("a", "i64")],
        return_type="i64",
        io_examples=[
            IoExample([1], 5), IoExample([1], 6), IoExample([1], 7),
            IoExample([2], 99), IoExample([2], -99),
        ],
    )
    result = resolve("junk", proposer=MockProposer(ir))
    assert result.status == "refused"
    assert "honest refusal" in " ".join(result.notes)


def test_resolve_unsupported_output_type_no_backend_needed():
    ir = RequirementsIR(
        entry_point="rolling_max",
        description="rolling maximum",
        params=[ParamSpec("xs", "[i64]")],
        return_type="[i64]",
        io_examples=[IoExample([[1, 3, 2]], [1, 3, 3]), IoExample([[5]], [5])],
    )
    result = resolve("rolling max", proposer=MockProposer(ir))
    assert result.status == "unsupported"
    assert result.ir is not None  # contract captured even though synth can't target it


def test_resolve_no_ir_when_proposer_fails():
    class Failing:
        def propose(self, english: str) -> RequirementsIR:
            raise ProposerError("no key")

    result = resolve("anything", proposer=Failing())
    assert result.status == "no_ir"
    assert "no key" in " ".join(result.notes)


# ---------------------------------------------------------------------------
# deterministic proposer
# ---------------------------------------------------------------------------

def test_deterministic_proposer_parses_arrows():
    ir = DeterministicProposer().propose(
        "def add(a, b):\n  add(1, 2) -> 3\n  add(5, 7) -> 12\n  add(0, 0) -> 0"
    )
    assert ir.entry_point == "add"
    assert len(ir.io_examples) == 3
    assert ir.synth_supported()


def test_deterministic_proposer_refuses_pure_prose():
    with pytest.raises(ProposerError):
        DeterministicProposer().propose("please write something that sorts a list nicely")
