"""Fusion scorer tests — plan.md §3.5.

The CP2 gate requires all four verdicts reachable BY UNIT TEST over synthetic
inputs, not by corpus clips: the clips that trigger INSUFFICIENT_EVIDENCE may not
exist yet at T+12.
"""

import pytest

from src.common.contracts import Verdict
from src.fusion import load_thresholds, score

CFG = load_thresholds()


def _score(rppg_s, rppg_q, lip_s, lip_q):
    return score(
        modality_scores={"rppg": rppg_s, "lipsync": lip_s},
        modality_quality={"rppg": rppg_q, "lipsync": lip_q},
        session_id="unit-test",
        thresholds=CFG,
    )


def test_all_four_verdicts_are_reachable():
    """CP2 gate criterion."""
    reached = {
        _score(0.05, 0.9, 0.05, 0.9).verdict,   # clean, agreeing -> authentic
        _score(0.95, 0.9, 0.95, 0.9).verdict,   # both accuse     -> manipulated
        _score(0.50, 0.9, 0.50, 0.9).verdict,   # mid band        -> uncertain
        _score(0.90, 0.05, 0.90, 0.05).verdict, # no evidence     -> insufficient
    }
    assert reached == set(Verdict)


def test_zero_quality_modality_cannot_drag_the_verdict():
    """A silent video must not be called manipulated because lip-sync is absent."""
    result = _score(rppg_s=0.05, rppg_q=0.9, lip_s=1.0, lip_q=0.0)
    assert result.verdict is Verdict.LIKELY_AUTHENTIC
    assert result.modality["lipsync"]["quality"] == 0.0


def test_single_modality_evidence_weight_is_capped_at_half():
    """Documented ceiling (§3.5): raising min_evidence_weight above 0.5 would
    silently break every silent clip."""
    result = _score(rppg_s=0.1, rppg_q=1.0, lip_s=0.1, lip_q=0.0)
    assert result.evidence_weight == pytest.approx(0.5)


def test_insufficient_evidence_takes_priority_over_probability():
    """Even a maximally suspicious score must abstain when evidence is thin."""
    result = _score(rppg_s=1.0, rppg_q=0.05, lip_s=1.0, lip_q=0.05)
    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_thresholds_come_from_config_not_code():
    for key in ("authentic_max_p", "manipulated_min_p", "min_evidence_weight"):
        assert key in CFG["decision"]
    assert "min_quality_for_scoring" in CFG["rppg"]
