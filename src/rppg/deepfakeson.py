"""DeepFakesON-Phys — physiological deepfake DETECTOR (not a BVP extractor).

    deepfakeson.analyze(frames, boxes) -> DfonResult    # P(manipulated)

WHY THIS IS DIFFERENT FROM EVERYTHING ELSE IN src/rppg/
CHROM, POS and PhysNet all try to RECOVER a pulse waveform, then reason about it.
On this corpus that has failed uniformly — near-zero band SNR on compressed
social media, an 8 Mbps capture, and a controlled recording alike.

DeepFakesON-Phys (Hernandez-Ortega et al., arXiv 2010.00400) skips extraction
entirely. It is a Convolutional Attention Network trained end-to-end to classify
fake vs genuine from the same two-stream input DeepPhys uses for heart-rate
estimation. It can therefore discriminate on physiological *texture* that is too
weak to yield a clean waveform — which is exactly the regime this footage sits in.

Two checkpoints ship with it, trained on different datasets:
    dfon_phys_celebdf.h5   Celeb-DF v2
    dfon_phys_dfdc.h5      DFDC Preview
Both are ~1.8 MB. Independent training sets means their agreement carries real
information, and it makes them a genuine second opinion against effb7 — which
matters now that effb7 measured AUC 0.550 on the 13-clip corpus.

INPUT CONTRACT — transcribed from the reference implementation, not inferred:
    two streams, each (N, 3, 36, 36) float32, CHANNELS FIRST
    values in 0..255 range — the reference does NOT divide by 255
    stream 1 (motion)     : normalised frame difference, per-pixel z-scored over time
    stream 2 (appearance) : raw frames, per-pixel z-scored over time
    face crop             : Haar box, resized to 36x36

THE uint8 CAST IS DELIBERATE AND LOAD-BEARING
The reference writes both streams to PNG via `np.uint8(imagen)` before inference.
That truncates and wraps the z-scored values — it looks like a bug, and it
quantises hard. But the model was TRAINED on exactly that, so reproducing it is
required for the input to be in-distribution. Removing the cast would produce a
plausible-looking score from an out-of-distribution input, the same silent
failure that cost this project a full LIPINC integration and a MediaPipe
reimplementation of dlib alignment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Keras 2 checkpoints; must precede the first tensorflow import.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
L = 36                       # reference: L = 36
EPS = 0.1                    # reference: desviaciones + 000.1

CHECKPOINTS = {
    "dfon_celebdf": REPO / "models" / "dfon_phys_celebdf.h5",
    "dfon_dfdc": REPO / "models" / "dfon_phys_dfdc.h5",
}


@dataclass
class DfonResult:
    dfon_manipulation_score: float = 0.5
    dfon_quality: float = 0.0
    per_model: dict = field(default_factory=dict)
    frames_scored: int = 0
    model_agreement: float = 1.0
    degraded_reason: str | None = None


@lru_cache(maxsize=4)
def _model(name: str):
    path = CHECKPOINTS.get(name)
    if path is None or not path.exists():
        return None
    try:
        import tensorflow as tf

        return tf.keras.models.load_model(str(path), compile=False)
    except Exception:
        return None


def available() -> list[str]:
    return [n for n in CHECKPOINTS if _model(n) is not None]


def _crops(frames_bgr, boxes) -> np.ndarray:
    """36x36 face crops, one per usable frame. (T, 36, 36, 3) float64."""
    out = []
    for frame, box in zip(frames_bgr, boxes):
        if box is None or np.isnan(np.asarray(box, dtype=float)).any():
            continue
        x, y, w, h = [int(v) for v in box]
        H, W = frame.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        face = frame[y0:y1, x0:x1]
        if face.size == 0:
            continue
        out.append(cv2.resize(face, (L, L), interpolation=cv2.INTER_AREA))
    return np.asarray(out, dtype=np.float64)


def build_streams(crops: np.ndarray):
    """(motion, appearance), each (T, 3, 36, 36) float32 — reference formulas."""
    T = len(crops)
    if T < 3:
        return None, None

    C = crops                                   # (T, L, L, 3)

    # motion: normalised successive difference, then per-pixel z-score over time
    D = np.zeros_like(C)
    num = C[1:] - C[:-1]
    den = C[1:] + C[:-1]
    D[:-1] = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-9)
    D = (D - D.mean(axis=0, keepdims=True)) / (D.std(axis=0, keepdims=True) + EPS)

    # appearance: raw frames, per-pixel z-score over time
    A = (C - C.mean(axis=0, keepdims=True)) / (C.std(axis=0, keepdims=True) + EPS)

    # The reference round-trips both through uint8 PNGs. Reproduced exactly:
    # the model was trained on these wrapped/truncated values.
    D = np.uint8(D).astype(np.float32)
    A = np.uint8(A).astype(np.float32)

    return D.transpose(0, 3, 1, 2), A.transpose(0, 3, 1, 2)


def analyze(frames_bgr, boxes, models: tuple[str, ...] | None = None) -> DfonResult:
    """Physiological deepfake score for a clip. Never raises."""
    try:
        names = list(models) if models else available()
        if not names:
            return DfonResult(degraded_reason="weights_unavailable")

        crops = _crops(frames_bgr, boxes)
        if len(crops) < 16:
            return DfonResult(degraded_reason=f"too_few_face_frames({len(crops)})")

        motion, appearance = build_streams(crops)
        if motion is None:
            return DfonResult(degraded_reason="stream_build_failed")

        per = {}
        for n in names:
            net = _model(n)
            if net is None:
                continue
            p = net.predict([motion, appearance], batch_size=128, verbose=0)
            per[n] = float(np.median(np.asarray(p).reshape(-1)))

        if not per:
            return DfonResult(degraded_reason="no_model_produced_output")

        vals = list(per.values())
        res = DfonResult()
        res.per_model = {k: round(v, 4) for k, v in per.items()}
        res.frames_scored = int(len(crops))
        res.dfon_manipulation_score = float(np.clip(np.mean(vals), 0.0, 1.0))
        res.model_agreement = 1.0 if len(vals) < 2 else float(1.0 - abs(vals[0] - vals[1]))
        res.dfon_quality = float(np.clip(res.model_agreement, 0.0, 1.0))
        return res
    except Exception as exc:  # noqa: BLE001
        return DfonResult(degraded_reason=f"unhandled:{type(exc).__name__}")
