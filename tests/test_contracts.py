"""Contract validation tests."""

import pytest
from src.common.contracts import (
    Verdict,
    PreprocessResult,
    RPPGResult,
    LipSyncResult,
    FusionResult
)

def test_verdict_enum():
    assert Verdict.LIKELY_AUTHENTIC == "LIKELY_AUTHENTIC"
    assert Verdict.LIKELY_MANIPULATED == "LIKELY_MANIPULATED"
    assert Verdict.UNCERTAIN == "UNCERTAIN"
    assert Verdict.INSUFFICIENT_EVIDENCE == "INSUFFICIENT_EVIDENCE"

def test_fusion_result_schema():
    result = FusionResult(
        session_id="test-session-123",
        verdict=Verdict.LIKELY_AUTHENTIC,
        manipulation_probability=0.15,
        evidence_weight=0.85,
        modality={
            "rppg": {"score": 0.12, "quality": 0.90, "weight": 0.5},
            "lipsync": {"score": 0.18, "quality": 0.80, "weight": 0.5}
        },
        explanation="Pulse and lip alignment are consistent with authentic video.",
        warnings=[],
        total_processing_time_ms=1250
    )
    assert result.session_id == "test-session-123"
    assert result.verdict == Verdict.LIKELY_AUTHENTIC
