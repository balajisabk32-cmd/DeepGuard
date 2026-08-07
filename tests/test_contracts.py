"""Contract validation tests.

Covers the two invariants plan.md calls load-bearing and that the original
scaffold did not test at all:
  1. Score polarity (§3.2) — enforced reflectively over the model fields.
  2. Explanation/verdict agreement (§3.5) — v1's own sample payload failed this.
"""

import re

import pytest
from pydantic import ValidationError

from src.common.contracts import (
    Verdict,
    PreprocessResult,
    RPPGResult,
    LipSyncResult,
    FusionResult,
)

ALL_MODELS = [PreprocessResult, RPPGResult, LipSyncResult, FusionResult]


def _valid_fusion(**overrides):
    payload = dict(
        session_id="test-session-123",
        verdict=Verdict.LIKELY_AUTHENTIC,
        manipulation_probability=0.15,
        evidence_weight=0.85,
        modality={
            "rppg": {"score": 0.12, "quality": 0.90, "weight": 0.5},
            "lipsync": {"score": 0.18, "quality": 0.80, "weight": 0.5},
        },
        explanation="Signals are consistent with authentic video.",
        warnings=[],
        total_processing_time_ms=1250,
    )
    payload.update(overrides)
    return FusionResult(**payload)


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------

def test_verdict_enum_has_exactly_four_values():
    assert {v.value for v in Verdict} == {
        "LIKELY_AUTHENTIC",
        "LIKELY_MANIPULATED",
        "UNCERTAIN",
        "INSUFFICIENT_EVIDENCE",
    }


def test_fusion_result_schema():
    result = _valid_fusion()
    assert result.session_id == "test-session-123"
    assert result.verdict == Verdict.LIKELY_AUTHENTIC


@pytest.mark.parametrize("model", ALL_MODELS)
def test_unknown_fields_are_rejected(model):
    """extra='forbid': a typo'd field name must be an error, not a silent drop."""
    with pytest.raises(ValidationError):
        model(definitely_not_a_real_field=1)


def test_probability_bounds_enforced():
    with pytest.raises(ValidationError):
        _valid_fusion(manipulation_probability=1.4)
    with pytest.raises(ValidationError):
        _valid_fusion(evidence_weight=-0.1)


def test_av_start_offset_is_required():
    """A default of 0.0 would silently reproduce the bug the field prevents (§6.3)."""
    assert PreprocessResult.model_fields["av_start_offset_sec"].is_required()


def test_duration_fields_allow_long_sources():
    """A 3-minute upload is truncated, not rejected (§6.2)."""
    field = PreprocessResult.model_fields["source_duration_sec"]
    assert all(getattr(m, "le", None) is None for m in field.metadata)


# --------------------------------------------------------------------------
# Polarity rule (§3.2)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model", ALL_MODELS)
def test_no_ambiguously_named_score_fields(model):
    """No field may be named bare `score` or `confidence`.

    v1 shipped `rppg_score: 0.34` with no stated polarity and a sample payload
    that contradicted itself. Polarity must live in the field name.
    """
    banned = {"score", "confidence", "confidence_score", "final_score"}
    assert not (banned & set(model.model_fields)), f"{model.__name__} has an ambiguous field"


@pytest.mark.parametrize("model", [RPPGResult, LipSyncResult])
def test_score_and_quality_fields_follow_naming_convention(model):
    fields = set(model.model_fields)
    assert any(f.endswith("_manipulation_score") for f in fields)
    assert any(f.endswith("_quality") for f in fields)


# --------------------------------------------------------------------------
# Explanation / verdict agreement (§3.5)
# --------------------------------------------------------------------------

# Patterns that must NOT appear for a given verdict — an explanation that reassures
# while the verdict accuses (or vice versa) is the exact v1 defect.
#
# The negative lookbehind matters: "inconsistent with authentic" CONTAINS
# "consistent with authentic" as a substring, so a plain `in` check reports a
# contradiction on a perfectly correct sentence.
REASSURING = r"(?<!in)consistent with authentic"
ACCUSING = r"inconsistent with authentic"

CONTRADICTORY = {
    Verdict.LIKELY_AUTHENTIC: (ACCUSING,),
    Verdict.LIKELY_MANIPULATED: (REASSURING,),
    Verdict.UNCERTAIN: (REASSURING, ACCUSING),
    Verdict.INSUFFICIENT_EVIDENCE: (REASSURING, ACCUSING),
}


@pytest.mark.parametrize("verdict", list(Verdict))
def test_generated_explanation_never_contradicts_verdict(verdict):
    from src.fusion import explain

    text = explain(
        verdict=verdict,
        reason="test",
        p=0.5,
        evidence_weight=0.5,
        modality_quality={"rppg": 0.9, "lipsync": 0.9},
    ).lower()

    for pattern in CONTRADICTORY[verdict]:
        assert not re.search(pattern, text), (
            f"{verdict.value} explanation contradicts itself: {text!r}"
        )
    assert text.strip(), "explanation must not be empty"
