"""Frame-level visual forgery detection — third modality, two-model ensemble.

    visual.analyze_video(path) -> VisualResult      # frame-by-frame, aggregated
    visual.analyze_image(path) -> VisualResult      # single image, same weights

DUAL PURPOSE BY CONSTRUCTION: both entry points run the identical models over the
identical preprocessing. The website's image detector and the video pipeline's
per-frame branch cannot drift apart, because there is one code path.

ENSEMBLE
  xception  FakeAVCeleb        checkpoint best_acc 0.740
  capsule   FaceForensics++    Capsule-Forensics-v2

Trained on DIFFERENT datasets, so their agreement carries real information.
DISAGREEMENT REDUCES QUALITY rather than being averaged away: when two
independently-trained detectors reach opposite conclusions about the same face,
the honest output is low confidence, not the midpoint dressed up as a decision.

Per-model scores are always reported so a judge can see which model said what.

CLASS ORDER and PREPROCESSING are per-model and verified — see models.py.
LICENCE: FakeAVCeleb ships no licence and is request-gated; FF++ is research /
non-commercial. Weights stay local, gitignored, never in a public image.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.visual.models import FAKE_INDEX, LOADERS, SPECS, available, preprocess

MAX_FRAMES = 32
MIN_FACE_PX = 64
CROP_MARGIN = 1.3
# Minimum interquartile range of a model's per-frame outputs for it to count as
# discriminating at all. Below this it is emitting a constant. See the gate in
# analyze_frames for the measurement that motivated it.
MIN_OUTPUT_IQR = 0.02

# Back-compat aliases (tests and older callers reference these).
INPUT_SIZE = SPECS["xception"].input_size
MEAN = np.array(SPECS["xception"].mean, dtype=np.float32)
STD = np.array(SPECS["xception"].std, dtype=np.float32)


@dataclass
class VisualResult:
    visual_manipulation_score: float = 0.5
    visual_quality: float = 0.0
    frames_scored: int = 0
    frame_scores: list = field(default_factory=list)
    score_spread: float = 0.0
    mean_face_px: float = 0.0
    per_model: dict = field(default_factory=dict)
    model_agreement: float = 1.0      # 1.0 = identical, 0.0 = opposite
    models_used: list = field(default_factory=list)
    degenerate_models: list = field(default_factory=list)
    processing_time_ms: int = 0
    degraded_reason: str | None = None


def weights_available() -> bool:
    return len(available()) > 0


def preprocess_face(frame_bgr: np.ndarray, box, spec=None) -> np.ndarray | None:
    """Kept for the xception spec by default; models.preprocess is the general form."""
    return preprocess(frame_bgr, box, spec or SPECS["xception"], CROP_MARGIN)


def _run_model(name: str, frames_bgr, boxes, idx) -> np.ndarray:
    """P(fake) per sampled frame for one detector. Batched."""
    import torch

    net = LOADERS[name]()
    if net is None:
        return np.zeros(0)
    spec = SPECS[name]

    tensors = [t for t in (preprocess(frames_bgr[i], boxes[i], spec, CROP_MARGIN)
                           for i in idx) if t is not None]
    if not tensors:
        return np.zeros(0)

    batch = torch.from_numpy(np.stack(tensors))
    with torch.no_grad():
        logits = net(batch)
        if spec.output == "sigmoid1":
            # Single logit. sigmoid IS P(fake) because labels are [Real, Fake].
            # Sending this through softmax(dim=1)[:, 1] raises IndexError on a
            # width-1 output — which is the good outcome. A 2-logit model read as
            # sigmoid would NOT raise; it would return a plausible wrong number.
            probs = torch.sigmoid(logits).reshape(-1)
        else:
            probs = torch.softmax(logits, dim=1)[:, FAKE_INDEX]
    return probs.numpy().astype(np.float64)


def analyze_frames(frames_bgr, boxes, max_frames: int = MAX_FRAMES) -> VisualResult:
    """Core path shared by the video and image entry points."""
    t0 = time.time()
    names = available()
    if not names:
        r = VisualResult(degraded_reason="weights_unavailable")
        r.processing_time_ms = int((time.time() - t0) * 1000)
        return r

    n = len(frames_bgr)
    if n == 0:
        return VisualResult(degraded_reason="no_frames")

    idx = np.linspace(0, n - 1, min(max_frames, n)).astype(int)
    usable = [i for i in idx
              if boxes is not None and not np.isnan(np.asarray(boxes[i], float)).any()]
    if not usable:
        r = VisualResult(degraded_reason="no_usable_faces")
        r.processing_time_ms = int((time.time() - t0) * 1000)
        return r

    per_model, all_scores, degenerate, low_spread = {}, [], [], {}
    for name in names:
        s = _run_model(name, frames_bgr, boxes, usable)
        if not s.size:
            continue
        per_model[name] = float(np.median(s))

        # Within-video output spread is REPORTED, not used to exclude.
        #
        # An earlier version dropped any model whose per-frame IQR fell below a
        # threshold, reasoning that a constant is not evidence. That test is
        # unsound: low spread also means "confidently consistent". Measured —
        # effb7 on REAL3 had IQR 0.001 while ranking the set perfectly, so the
        # gate would have discarded the best detector for being sure.
        #
        # Non-discriminating models are excluded explicitly via SPECS[...].enabled,
        # on cross-video evidence, rather than guessed at from one clip.
        iqr = float(np.subtract(*np.percentile(s, [75, 25]))) if s.size >= 4 else 1.0
        low_spread[name] = iqr
        all_scores.append(s)

    if not all_scores:
        r = VisualResult(
            degraded_reason=("all_models_degenerate:" + ",".join(sorted(degenerate)))
            if degenerate else "no_usable_faces")
        r.per_model = {k: round(v, 4) for k, v in per_model.items()}
        r.processing_time_ms = int((time.time() - t0) * 1000)
        return r

    res = VisualResult()
    res.models_used = sorted(set(per_model) - set(degenerate))
    res.degenerate_models = sorted(degenerate)
    res.per_model = {k: round(v, 4) for k, v in per_model.items()}
    res.frames_scored = int(min(s.size for s in all_scores))

    # Aggregate per-frame scores, then calibrate. Both come from config so the
    # operating point can be refit without touching code.
    from src.fusion import load_thresholds

    vcfg = load_thresholds().get("visual", {})
    agg = vcfg.get("aggregation", "p90")
    per = [float(np.percentile(s, 90)) if agg == "p90" else float(np.median(s))
           for s in all_scores]
    combined = float(np.mean(per))

    cal = vcfg.get("calibration")
    if cal:
        # Monotone logit-space rescale: the raw sigmoid is compressed near zero
        # and never reaches a decision threshold on its native scale.
        q = float(np.clip(combined, 1e-6, 1 - 1e-6))
        z = np.log(q / (1 - q))
        combined = float(1.0 / (1.0 + np.exp(-(cal["a"] * z + cal["b"]))))

    res.visual_manipulation_score = float(np.clip(combined, 0.0, 1.0))
    res.score_spread = float(np.mean([s.std() for s in all_scores]))
    res.frame_scores = [round(float(x), 4) for x in all_scores[0][:100]]

    sizes = [float(max(boxes[i][2], boxes[i][3])) for i in usable]
    res.mean_face_px = float(np.mean(sizes)) if sizes else 0.0

    # Agreement between independently-trained models.
    # Agreement only counts between models that actually discriminate.
    vals = [per_model[m] for m in res.models_used]
    res.model_agreement = 1.0 if len(vals) < 2 else float(1.0 - abs(vals[0] - vals[1]))

    coverage = float(np.clip(len(usable) / max(len(idx), 1), 0.0, 1.0))
    resolution = float(np.clip((res.mean_face_px - MIN_FACE_PX) / (160.0 - MIN_FACE_PX),
                               0.0, 1.0))
    res.visual_quality = float(np.clip(coverage * resolution * res.model_agreement,
                                       0.0, 1.0))
    if res.visual_quality < 0.05:
        res.degraded_reason = "faces_too_small_few_or_models_disagree"

    res.processing_time_ms = int((time.time() - t0) * 1000)
    return res


def analyze_video(video_path: str, max_sec: float = 20.0,
                  clip=None, max_frames: int = MAX_FRAMES) -> VisualResult:
    """Frame-by-frame analysis. Never raises."""
    t0 = time.time()
    try:
        if clip is not None:
            frames, boxes = clip.frames, clip.boxes
        else:
            from src.rppg import backends
            from src.rppg.analyze import read_frames

            frames, _t, _fps, _w = read_frames(video_path, max_sec=max_sec)
            if not frames:
                return VisualResult(degraded_reason="decode_failed")
            boxes, _c = backends.OpenCVBackend().detect_boxes(frames)
            boxes = backends.smooth_boxes(boxes)
        return analyze_frames(frames, boxes, max_frames=max_frames)
    except Exception as exc:  # noqa: BLE001
        r = VisualResult(degraded_reason=f"unhandled:{type(exc).__name__}")
        r.processing_time_ms = int((time.time() - t0) * 1000)
        return r


_CASCADE_CHAIN = (
    ("haarcascade_frontalface_default.xml", 1.1, 5, False),
    ("haarcascade_frontalface_alt2.xml", 1.1, 4, False),
    ("haarcascade_frontalface_default.xml", 1.05, 3, False),
    ("haarcascade_profileface.xml", 1.1, 4, False),
    ("haarcascade_profileface.xml", 1.1, 4, True),
)


@lru_cache(maxsize=8)
def _cascade(name: str):
    c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    return None if c.empty() else c


def detect_face_robust(img_bgr: np.ndarray):
    """Best-effort single-image face box, escalating through detectors.

    The video path tolerates detector misses because the box trajectory is
    interpolated and median-filtered across frames. A single image has no such
    fallback, so one frontal-cascade miss means no result at all — which is
    exactly what happened on a frame where the subject was turned away.
    """
    gray_full = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    H, W = gray_full.shape[:2]
    for name, scale_factor, neighbours, mirror in _CASCADE_CHAIN:
        casc = _cascade(name)
        if casc is None:
            continue
        gray = cv2.flip(gray_full, 1) if mirror else gray_full
        found = casc.detectMultiScale(gray, scale_factor, neighbours,
                                      minSize=(max(W // 20, 24), max(H // 20, 24)))
        if len(found) == 0:
            continue
        x, y, w, h = max(found, key=lambda b: b[2] * b[3])
        if mirror:
            x = W - x - w
        return (float(x), float(y), float(w), float(h))
    return None


def analyze_image(image_path: str) -> VisualResult:
    """Single-image entry point for the website's image detector."""
    t0 = time.time()
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return VisualResult(degraded_reason="could_not_read_image")
        box = detect_face_robust(img)
        if box is None:
            r = VisualResult(degraded_reason="no_face_detected")
            r.processing_time_ms = int((time.time() - t0) * 1000)
            return r
        res = analyze_frames([img], np.array([box]), max_frames=1)
        res.processing_time_ms = int((time.time() - t0) * 1000)
        return res
    except Exception as exc:  # noqa: BLE001
        r = VisualResult(degraded_reason=f"unhandled:{type(exc).__name__}")
        r.processing_time_ms = int((time.time() - t0) * 1000)
        return r
