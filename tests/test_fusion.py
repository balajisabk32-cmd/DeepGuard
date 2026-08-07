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


def test_single_modality_evidence_weight_is_capped_at_its_prior():
    """A lone modality can never exceed its own prior share of the total.

    Was pinned at 0.5 when there were two equally-weighted modalities. With
    rppg/lipsync/visual it is rppg's prior over the supplied total — the ceiling
    is a property of the weights, not a constant, and min_evidence_weight must
    stay below the SMALLEST prior or single-modality clips silently all abstain.
    """
    result = _score(rppg_s=0.1, rppg_q=1.0, lip_s=0.1, lip_q=0.0)
    priors = CFG["fusion"]["prior_weights"]
    expected = priors["rppg"] / (priors["rppg"] + priors["lipsync"])
    assert result.evidence_weight == pytest.approx(expected)
    # Compare against the prior SHARE, not the raw prior. These coincide only
    # while the priors happen to sum to 1.0; the moment they do not, asserting on
    # raw priors silently stops testing anything. evidence_weight is always
    # sum(w*q)/sum(w), so the share is what a lone modality can actually reach.
    total = sum(priors.values())
    smallest_share = min(v / total for v in priors.values())
    assert CFG["decision"]["min_evidence_weight"] < smallest_share, (
        "a channel that is fully confident and alone must still clear the "
        "evidence floor; adding a channel dilutes every share, so this floor "
        "must be re-derived whenever a modality is added"
    )


def test_absent_modality_does_not_crash_the_verdict():
    """Adding a modality to config must not break a two-modality caller."""
    r = _score(0.2, 0.9, 0.2, 0.9)
    assert 0.0 <= r.manipulation_probability <= 1.0
    assert any("modalities_absent" in w for w in r.warnings)


def test_every_configured_modality_fuses():
    """Supplying every configured channel must raise no 'absent' warning.

    Derived from the config rather than hard-coded to three names: this test used
    to list rppg/lipsync/visual literally, so when the 4th (pixel) channel was
    added it failed for the RIGHT reason but with a misleading message. Reading
    the channel set from config means adding a 5th channel extends the test
    instead of breaking it.
    """
    names = list(CFG["fusion"]["prior_weights"])
    r = score(
        modality_scores={m: 0.2 for m in names},
        modality_quality={m: 0.8 for m in names},
        session_id="unit-test", thresholds=CFG,
    )
    assert set(r.modality) == set(names)
    assert not any("modalities_absent" in w for w in r.warnings)


def test_insufficient_evidence_takes_priority_over_probability():
    """Even a maximally suspicious score must abstain when evidence is thin."""
    result = _score(rppg_s=1.0, rppg_q=0.05, lip_s=1.0, lip_q=0.05)
    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_thresholds_come_from_config_not_code():
    for key in ("authentic_max_p", "manipulated_min_p", "min_evidence_weight"):
        assert key in CFG["decision"]
    assert "min_quality_for_scoring" in CFG["rppg"]
