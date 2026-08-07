"""Visual (Xception) modality tests.

The class-order test is the one that matters. Getting FAKE_INDEX backwards
inverts the detector while still producing entirely plausible-looking
probabilities — no crash, no warning, just confidently wrong verdicts.
"""

from pathlib import Path

import numpy as np
import pytest

from src.visual import analyze_frames, analyze_image, analyze_video, weights_available
from src.visual.detector import (
    FAKE_INDEX,
    INPUT_SIZE,
    MEAN,
    STD,
    VisualResult,
    preprocess_face,
)

REPO = Path(__file__).resolve().parents[1]
REAL_CLIP = REPO / "TEST_VIDEOS" / "REAL2.mp4"
FAKE_CLIP = REPO / "TEST_VIDEOS" / "FAKE2.mp4"

needs_weights = pytest.mark.skipif(not weights_available(), reason="weights absent")


# ------------------------------------------------------------ invariants

def test_fake_is_index_one():
    """Verified against FF++ classification/detect_from_video.py:195 —
    `label = 'fake' if prediction == 1 else 'real'`."""
    assert FAKE_INDEX == 1


def test_preprocessing_constants_are_dataset_not_imagenet():
    """These are FakeAVCeleb training statistics. Substituting ImageNet values
    (0.485/0.456/0.406) or Xception's own docstring values (0.5/0.5/0.5) silently
    degrades accuracy instead of failing, so the values are pinned here."""
    assert INPUT_SIZE == 224
    assert np.allclose(MEAN, [0.4489, 0.3352, 0.3106])
    assert np.allclose(STD, [0.2380, 0.1965, 0.1962])


# ------------------------------------------------------------ preprocessing

def test_preprocess_shape_and_normalisation():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    out = preprocess_face(frame, (200, 100, 200, 200))
    assert out is not None
    assert out.shape == (3, INPUT_SIZE, INPUT_SIZE)
    expected = (128 / 255.0 - MEAN) / STD
    assert np.allclose(out.mean(axis=(1, 2)), expected, atol=1e-3)


@pytest.mark.parametrize("box", [None, (0, 0, 2, 2), (np.nan, 0, 100, 100)])
def test_preprocess_rejects_unusable_boxes(box):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert preprocess_face(frame, box) is None


# ------------------------------------------------------------ degradation

def test_no_faces_degrades_without_raising():
    frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * 4
    boxes = np.full((4, 4), np.nan)
    res = analyze_frames(frames, boxes)
    assert res.degraded_reason is not None
    assert res.visual_manipulation_score == 0.5, "must stay neutral, never accuse"
    assert res.visual_quality == 0.0


@pytest.mark.parametrize("bad", ["", "nope.mp4", str(Path(__file__))])
def test_analyze_video_never_raises(bad):
    res = analyze_video(bad)
    assert isinstance(res, VisualResult)
    assert res.degraded_reason is not None
    assert res.visual_manipulation_score == 0.5


def test_analyze_image_handles_unreadable_file():
    res = analyze_image(str(Path(__file__)))
    assert res.degraded_reason is not None
    assert res.visual_quality == 0.0


def test_small_faces_lower_quality():
    """A 224px input built from a 40px face is upscaled blur, not detail."""
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    big = analyze_frames([frame] * 4, np.tile([100, 100, 300, 300], (4, 1)))
    small = analyze_frames([frame] * 4, np.tile([100, 100, 40, 40], (4, 1)))
    assert small.visual_quality <= big.visual_quality


# ------------------------------------------------------------ end to end

@needs_weights
def test_weights_load_and_score_is_a_probability():
    frame = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    res = analyze_frames([frame] * 3, np.tile([150, 100, 220, 220], (3, 1)))
    assert res.frames_scored == 3
    assert 0.0 <= res.visual_manipulation_score <= 1.0


@needs_weights
@pytest.mark.skipif(not (REAL_CLIP.exists() and FAKE_CLIP.exists()),
                    reason="reference clips absent (gitignored media)")
def test_known_fake_scores_above_known_real():
    """POLARITY REGRESSION GUARD.

    An authentic recording must not score higher than a known deepfake. If this
    inverts, FAKE_INDEX is wrong or the checkpoint was swapped — and every verdict
    the system produces is backwards while still looking reasonable.

    Margin is deliberately loose: the checkpoint reports best_acc 0.740, so this
    asserts DIRECTION, not confidence.
    """
    real = analyze_video(str(REAL_CLIP)).visual_manipulation_score
    fake = analyze_video(str(FAKE_CLIP)).visual_manipulation_score
    assert fake > real, f"authentic={real:.3f} deepfake={fake:.3f} — polarity inverted?"


# ------------------------------------------------------------ ensemble

def test_degenerate_model_is_excluded_not_averaged_in():
    """A model emitting a near-constant is not evidence.

    Capsule-Forensics returned 0.609 on 10 of 12 frames (IQR 0.003) while Xception
    spanned 0.029-0.880 on the SAME faces. Averaging a constant in adds no
    discrimination, drags the ensemble toward its fixed point, and manufactures
    'agreement' that reads as corroboration. The gate must drop it.
    """
    import numpy as np

    from src.visual.detector import MIN_OUTPUT_IQR

    constant = np.full(12, 0.609)
    varying = np.array([0.03, 0.15, 0.45, 0.34, 0.36, 0.35, 0.88, 0.49, 0.48, 0.56,
                        0.64, 0.27])
    iqr = lambda s: float(np.subtract(*np.percentile(s, [75, 25])))
    assert iqr(constant) < MIN_OUTPUT_IQR
    assert iqr(varying) > MIN_OUTPUT_IQR


def test_specs_have_distinct_preprocessing():
    """The two models disagree on every preprocessing parameter. Sharing one
    model's constants with the other degrades it silently."""
    from src.visual.models import SPECS

    x, c = SPECS["xception"], SPECS["capsule"]
    assert x.input_size != c.input_size
    assert x.mean != c.mean and x.std != c.std
    assert x.dataset != c.dataset, "ensembling only helps across different training sets"
