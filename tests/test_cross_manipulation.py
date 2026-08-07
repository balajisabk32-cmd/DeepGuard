"""Cross-manipulation evaluation protocol tests.

Validated against synthetic score tables, because the real corpus does not exist
yet. The protocol has to be correct BEFORE data arrives — otherwise the first
number the team reports is wrong and nobody notices.
"""

import numpy as np
import pytest

from eval.cross_manipulation import auc, bootstrap_auc_ci, evaluate
from src.pipeline.aggregate import _weighted_median, plan_windows


# ------------------------------------------------------------------ AUC

def test_auc_perfect_and_inverted():
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert auc(labels, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == 1.0
    assert auc(labels, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) == 0.0


def test_auc_chance_on_tied_scores():
    """A detector that outputs the same number for everything must score 0.5,
    not 1.0 — which is what a naive rank implementation does with ties."""
    labels = np.array([0, 0, 1, 1])
    assert auc(labels, np.array([0.5, 0.5, 0.5, 0.5]) ) == pytest.approx(0.5)


def test_auc_is_nan_for_single_class():
    assert np.isnan(auc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])))


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(30), np.ones(30)].astype(int)
    scores = np.r_[rng.normal(0.3, 0.1, 30), rng.normal(0.7, 0.1, 30)]
    lo, hi = bootstrap_auc_ci(labels, scores, n=500)
    assert lo <= auc(labels, scores) <= hi


def test_ci_is_wider_at_small_n():
    """The CI must actually reflect sample size — this is the claim the report
    leans on when it says a small corpus cannot support a strong conclusion."""
    rng = np.random.default_rng(1)

    def ci_width(n):
        labels = np.r_[np.zeros(n), np.ones(n)].astype(int)
        scores = np.r_[rng.normal(0.4, 0.2, n), rng.normal(0.6, 0.2, n)]
        lo, hi = bootstrap_auc_ci(labels, scores, n=400)
        return hi - lo

    assert ci_width(5) > ci_width(60)


# ------------------------------------------------------- leave-one-method-out

def _rows(methods, per_method=4, n_auth=6, fake_score=0.8, real_score=0.2):
    rows = [{"clip_id": f"a{i}", "class_label": "A", "attack_type": "none",
             "score": str(real_score)} for i in range(n_auth)]
    for m in methods:
        rows += [{"clip_id": f"{m}{i}", "class_label": "C", "attack_type": m,
                  "score": str(fake_score)} for i in range(per_method)]
    return rows


def test_each_manipulation_is_held_out_in_turn():
    res = evaluate(_rows(["deepfakes", "face2face", "neuraltextures"]))
    assert {f["held_out"] for f in res["folds"]} == {
        "deepfakes", "face2face", "neuraltextures"}
    for f in res["folds"]:
        assert f["n_fake"] == 4 and f["n_authentic"] == 6


def test_requires_at_least_two_methods():
    """A single manipulation type cannot support a cross-manipulation claim."""
    res = evaluate(_rows(["deepfakes"]))
    assert "error" in res


def test_abstentions_excluded_not_scored_as_half():
    """Scoring an abstention as 0.5 would reward a detector for refusing to
    answer. They must be excluded and reported."""
    rows = _rows(["deepfakes", "faceswap"])
    rows[0]["score"] = "INSUFFICIENT_EVIDENCE"
    rows[1]["score"] = "UNCERTAIN"
    res = evaluate(rows)
    fold = res["folds"][0]
    assert fold["abstained"] == 2
    assert fold["n_authentic"] == 4          # 6 authentic minus 2 abstentions


def test_perfect_separation_gives_auc_one():
    res = evaluate(_rows(["deepfakes", "faceswap"], fake_score=0.9, real_score=0.1))
    assert all(f["auc"] == 1.0 for f in res["folds"])


# ------------------------------------------------------ multi-window aggregation

def test_plan_windows_overlap_and_cap():
    starts = plan_windows(60.0, window_sec=12.0, hop_sec=6.0, max_windows=5)
    assert len(starts) == 5
    assert starts[0] == 0.0
    assert all(s + 12.0 <= 60.0 + 1e-6 for s in starts)


def test_short_clip_falls_back_to_single_window():
    assert plan_windows(10.0, window_sec=12.0) == [0.0]


def test_weighted_median_ignores_a_zero_quality_outlier():
    """One window landing on a scene cut must not move the verdict."""
    values = np.array([0.20, 0.22, 0.21, 0.95])
    weights = np.array([1.0, 1.0, 1.0, 0.0])
    assert _weighted_median(values, weights) < 0.3


def test_weighted_median_falls_back_when_all_weights_zero():
    v = np.array([0.1, 0.5, 0.9])
    assert _weighted_median(v, np.zeros(3)) == pytest.approx(0.5)
