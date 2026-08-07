"""Traditional pixel forensics — classical, analytical, no trained weights.

    pixel_forensics.analyze(frames, boxes) -> PixelForensicsResult

THE FOURTH MANDATED CHANNEL
rPPG (physiology), lip-sync (biomechanics) and the frame-by-frame CNN (learned
artifacts) are all in place. This is classical image forensics: hand-derived
measurements of the pixels themselves, independent of any trained model. Because
it shares no mechanism with the CNN, it is the channel most likely to have
DECORRELATED errors — which our measurements showed is the precondition for
fusion to help at all (effb7 + LIPINC had error correlation +0.596 and fusing
them made things worse).

FEATURE DESIGN IS EVIDENCE-DRIVEN, NOT ASSUMED
Derived from 4,347 human annotations of real deepfakes (ExDDV, CC BY-NC-SA).
What annotators actually report seeing:

    warping / distortion   54.1%     <- dominant
    mouth / lips           39.7%
    eyes                   38.6%
    blur / pixelation      12.7%
    shadow / lighting      11.9%
    temporal flicker       11.3%
    edges / blending        4.4%     <- much rarer than expected

That ordering overturned the obvious design. Generic whole-frame forensics (ELA,
global DCT residual, seam detection) targets the 4.4% case. The signal humans
actually key on is LOCAL GEOMETRIC DISTORTION and REGION-SPECIFIC TEXTURE FAILURE
concentrated in the mouth and eyes. Every feature below targets that.

NOT YET CALIBRATED — DOES NOT VOTE
Features are computed and reported, but the channel is disabled in fusion until
calibrated on a labelled corpus. Shipping an uncalibrated score would repeat the
exact failure that disabled xception (AUC 0.222) and capsule (constant output):
a plausible number with nothing behind it. Admission by measurement.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# Region boxes as fractions of the face bounding box (x0, y0, x1, y1).
# Mouth and eyes are weighted heaviest because that is where annotators look.
REGIONS = {
    "mouth":    (0.28, 0.62, 0.72, 0.92),
    "eye_l":    (0.12, 0.32, 0.44, 0.50),
    "eye_r":    (0.56, 0.32, 0.88, 0.50),
    "nose":     (0.38, 0.44, 0.62, 0.66),
    "forehead": (0.25, 0.08, 0.75, 0.28),
    "cheek_l":  (0.10, 0.48, 0.32, 0.70),
    "cheek_r":  (0.68, 0.48, 0.90, 0.70),
}
FOCUS = ("mouth", "eye_l", "eye_r")     # the 40%/39% regions
MIN_REGION_PX = 12


@dataclass
class PixelForensicsResult:
    pixel_manipulation_score: float = 0.5
    pixel_quality: float = 0.0
    features: dict = field(default_factory=dict)
    frames_used: int = 0
    processing_time_ms: int = 0
    degraded_reason: str | None = None


def _region(gray: np.ndarray, box, frac) -> np.ndarray | None:
    x, y, w, h = [float(v) for v in box]
    H, W = gray.shape[:2]
    fx0, fy0, fx1, fy1 = frac
    x0, x1 = int(np.clip(x + fx0 * w, 0, W - 1)), int(np.clip(x + fx1 * w, 1, W))
    y0, y1 = int(np.clip(y + fy0 * h, 0, H - 1)), int(np.clip(y + fy1 * h, 1, H))
    if x1 - x0 < MIN_REGION_PX or y1 - y0 < MIN_REGION_PX:
        return None
    patch = gray[y0:y1, x0:x1]
    return patch if patch.size else None


def _sharpness(patch: np.ndarray) -> float:
    """Variance of Laplacian — standard focus/texture measure."""
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def _hf_residual(patch: np.ndarray) -> float:
    """Mean |image - blurred|. Generative upsampling suppresses fine detail, so a
    manipulated region carries less high-frequency energy than authentic skin."""
    blur = cv2.GaussianBlur(patch, (0, 0), 1.5)
    return float(np.mean(np.abs(patch.astype(np.float64) - blur.astype(np.float64))))


def _warp_energy(prev: np.ndarray, cur: np.ndarray) -> float:
    """Non-rigid deformation between consecutive frames of the SAME region.

    Targets the 54% case. Dense optical flow within an already face-aligned
    region should be near-uniform for a real face — genuine motion is mostly
    rigid once the box tracks the head. Generated frames deform locally and
    inconsistently, so the SPREAD of the flow field (not its magnitude) is the
    signal. Magnitude alone would just measure head movement.
    """
    if prev.shape != cur.shape or min(prev.shape) < 8:
        return float("nan")
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None,
                                        0.5, 2, 9, 2, 5, 1.1, 0)
    mag = np.linalg.norm(flow, axis=2)
    return float(mag.std())


def _stats(series: list[float]) -> tuple[float, float]:
    """(median, coefficient of variation) ignoring NaNs. CV captures flicker."""
    a = np.asarray([v for v in series if np.isfinite(v)], dtype=np.float64)
    if a.size < 2:
        return float("nan"), float("nan")
    med = float(np.median(a))
    return med, float(a.std() / (abs(med) + 1e-9))


def extract_features(frames_bgr, boxes, max_frames: int = 40) -> dict:
    """Per-region classical forensics aggregated over the clip."""
    n = len(frames_bgr)
    idx = [i for i in np.linspace(0, n - 1, min(max_frames, n)).astype(int)
           if boxes is not None and not np.isnan(np.asarray(boxes[i], float)).any()]
    if len(idx) < 4:
        return {}

    sharp = {r: [] for r in REGIONS}
    hf = {r: [] for r in REGIONS}
    lum = {r: [] for r in REGIONS}
    warp = {r: [] for r in FOCUS}
    prev_patch: dict[str, np.ndarray] = {}
    used = 0

    for i in idx:
        gray = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
        ok = False
        for name, frac in REGIONS.items():
            p = _region(gray, boxes[i], frac)
            if p is None:
                continue
            ok = True
            sharp[name].append(_sharpness(p))
            hf[name].append(_hf_residual(p))
            lum[name].append(float(p.mean()))
            if name in FOCUS:
                # resize to a common size so flow is comparable across frames
                pr = cv2.resize(p, (48, 48))
                if name in prev_patch:
                    warp[name].append(_warp_energy(prev_patch[name], pr))
                prev_patch[name] = pr
        used += int(ok)

    f: dict[str, float] = {}
    all_sharp = [v for r in REGIONS for v in sharp[r] if np.isfinite(v)]
    face_sharp = float(np.median(all_sharp)) if all_sharp else float("nan")

    for r in REGIONS:
        s_med, s_cv = _stats(sharp[r])
        h_med, _ = _stats(hf[r])
        f[f"sharp_{r}"] = s_med
        f[f"flicker_{r}"] = s_cv                      # temporal texture instability
        f[f"hf_{r}"] = h_med
        # region sharpness relative to the whole face: a manipulated region is
        # typically SMOOTHER than the authentic skin surrounding it
        f[f"sharp_ratio_{r}"] = (s_med / face_sharp) if np.isfinite(face_sharp) and face_sharp > 1e-9 else float("nan")

    for r in FOCUS:
        w_med, w_cv = _stats(warp[r])
        f[f"warp_{r}"] = w_med
        f[f"warp_cv_{r}"] = w_cv

    # lighting symmetry: left vs right cheek luminance (11.9% of annotations)
    ml, _ = _stats(lum["cheek_l"])
    mr, _ = _stats(lum["cheek_r"])
    f["lighting_asymmetry"] = (abs(ml - mr) / (0.5 * (ml + mr) + 1e-9)) \
        if np.isfinite(ml) and np.isfinite(mr) else float("nan")

    f["frames_used"] = float(used)
    return f


CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "results" / "pixel_calibration.json"


@lru_cache(maxsize=1)
def _calibration() -> dict | None:
    """Exported logistic coefficients, or None if the channel is uncalibrated."""
    try:
        with open(CALIBRATION_PATH, encoding="utf-8") as fh:
            cal = json.load(fh)
        if not all(k in cal for k in ("features", "mean", "scale", "coef", "intercept")):
            return None
        return cal
    except Exception:  # noqa: BLE001 - an unreadable calibration must not break the run
        return None


def _calibrated_score(feats: dict, cal: dict) -> float | None:
    """p = sigmoid(coef . ((x - mean) / scale) + intercept)."""
    z = float(cal["intercept"])
    for name, mu, sd, w in zip(cal["features"], cal["mean"], cal["scale"], cal["coef"]):
        v = feats.get(name)
        if v is None or not np.isfinite(v):
            return None                 # a missing feature invalidates the whole vector
        z += w * ((float(v) - mu) / (sd if abs(sd) > 1e-12 else 1.0))
    return 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))


def analyze(frames_bgr, boxes, max_frames: int = 40) -> PixelForensicsResult:
    """Classical forensic features for a clip. Never raises.

    CALIBRATED, BUT IT DOES NOT VOTE.

    Fitted on SDFVD2.0 (876 clips, 52 subjects) with subject-grouped CV, so the
    8 augmented copies of a clip and the real/fake pair of one subject never span
    a split. Measured honestly:

        all features,    grouped CV   AUC 0.711
        scale-free only, grouped CV   AUC 0.609   <- what ships
        scale-free only, RANDOM split AUC 0.614   (leaky, for reference only)

    Only scale-free features ship. The 0.10 AUC the absolute sharpness/HF terms
    add is largely capture resolution: the manipulated clips in this corpus are
    re-encoded, so a model using them learns "blurrier => fake" and would not
    transfer.

    quality stays 0 because the channel FAILED its admission test, not because it
    is unmeasured. Against effb7 on the 281 clips both scored (subject-grouped):
        effb7 alone   0.670
        effb7 + pixel 0.639   (-0.031)
    Error correlation is only -0.079, so the two are genuinely decorrelated - the
    channel is simply too weak to pay for itself. It is reported in full, with a
    real score, and contributes exactly 0 to the verdict.
    """
    t0 = time.time()
    try:
        feats = extract_features(frames_bgr, boxes, max_frames=max_frames)
        res = PixelForensicsResult()
        res.processing_time_ms = int((time.time() - t0) * 1000)
        if not feats:
            res.degraded_reason = "too_few_usable_frames"
            return res
        res.features = {k: (None if not np.isfinite(v) else round(float(v), 5))
                        for k, v in feats.items()}
        res.frames_used = int(feats.get("frames_used", 0))

        cal = _calibration()
        if cal is None:
            res.degraded_reason = "uncalibrated_reporting_only"
            return res
        p = _calibrated_score(feats, cal)
        if p is None:
            res.degraded_reason = "calibration_features_incomplete"
            return res
        res.pixel_manipulation_score = round(float(p), 4)
        # Admission is decided by the measurement stored alongside the weights,
        # never hardcoded here - recalibrating on better data flips this on
        # without a code change.
        if cal.get("admitted"):
            res.pixel_quality = 1.0
            res.degraded_reason = None
        else:
            res.pixel_quality = 0.0
            res.degraded_reason = (
                f"calibrated_auc_{cal.get('auc_grouped', 0):.3f}_below_admission_gate")
        return res
    except Exception as exc:  # noqa: BLE001
        return PixelForensicsResult(
            degraded_reason=f"unhandled:{type(exc).__name__}",
            processing_time_ms=int((time.time() - t0) * 1000))
