"""Tests for HighAccuracyLipSyncDetector and ForensicMultiTaskLoss."""

import pytest
import torch

from src.lipsync.detector import (
    BehavioralBranch,
    CrossModalMismatchFusion,
    ForensicMultiTaskLoss,
    HighAccuracyLipSyncDetector,
    TimeDistributedCNN,
    VisualTemporalBranch,
    evaluate_video,
)


def test_time_distributed_cnn_shapes():
    cnn = TimeDistributedCNN(out_dim=256, pretrained=False)
    # [B=2, T=4, C=3, H=112, W=112]
    x = torch.randn(2, 4, 3, 112, 112)
    out = cnn(x)
    assert out.shape == (2, 4, 256)


def test_visual_branch_shapes():
    branch = VisualTemporalBranch(embed_dim=256, num_layers=1, num_heads=2, pretrained=False)
    x = torch.randn(2, 4, 3, 112, 112)
    out = branch(x)
    assert out.shape == (2, 4, 256)


def test_behavioral_branch_shapes():
    branch = BehavioralBranch(in_dim=69, embed_dim=128, num_layers=1, num_heads=2)
    au = torch.randn(2, 4, 69)
    conf = torch.ones(2, 4)
    out = branch(au, frame_confidence=conf)
    assert out.shape == (2, 4, 128)


def test_fusion_shapes():
    fusion = CrossModalMismatchFusion(v_dim=256, b_dim=128, common_dim=128, num_heads=2)
    v_seq = torch.randn(2, 4, 256)
    b_seq = torch.randn(2, 4, 128)
    fused, v_pooled, b_pooled, mismatch = fusion(v_seq, b_seq)
    assert fused.shape == (2, 257)  # 128 + 128 + 1
    assert v_pooled.shape == (2, 128)
    assert b_pooled.shape == (2, 128)
    assert mismatch.shape == (2, 4)


def test_high_accuracy_detector_forward():
    model = HighAccuracyLipSyncDetector(au_dim=69, v_embed=256, b_embed=128, common_dim=128, pretrained=False)
    frames = torch.randn(2, 4, 3, 112, 112)
    au = torch.randn(2, 4, 69)
    conf = torch.ones(2, 4)

    outputs = model(frames, au, frame_confidence=conf)
    assert outputs["logits"].shape == (2, 1)
    assert outputs["v_pooled"].shape == (2, 128)
    assert outputs["b_pooled"].shape == (2, 128)
    assert outputs["v_seq"].shape == (2, 4, 256)
    assert outputs["mismatch_score"].shape == (2, 4)


def test_forensic_multitask_loss():
    model = HighAccuracyLipSyncDetector(au_dim=69, v_embed=256, b_embed=128, common_dim=128, pretrained=False)
    criterion = ForensicMultiTaskLoss()

    frames = torch.randn(2, 4, 3, 112, 112)
    au = torch.randn(2, 4, 69)
    targets = torch.tensor([0.0, 1.0])

    outputs = model(frames, au)
    total_loss, metrics = criterion(outputs, targets)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() > 0
    assert "bce" in metrics
    assert "coherence" in metrics
    assert "contrastive" in metrics


def test_evaluate_video_real():
    """Reference clips live in TEST_VIDEOS/ and are gitignored media, so the path
    is resolved from the repo root and the test skips when they are absent rather
    than failing on a machine that simply does not have them."""
    from pathlib import Path

    import pytest

    clip = Path(__file__).resolve().parents[1] / "TEST_VIDEOS" / "REAL.mp4"
    if not clip.exists():
        pytest.skip("TEST_VIDEOS/REAL.mp4 not present (gitignored media)")

    model = HighAccuracyLipSyncDetector(pretrained=False)
    res = evaluate_video(model, str(clip), device="cpu", max_sec=2.0)
    assert res["video"] == str(clip)
    assert "prob_manipulated" in res
    assert 0.0 <= res["prob_manipulated"] <= 1.0
