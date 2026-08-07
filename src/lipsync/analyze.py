"""Speech-to-lip alignment — Role 2's deliverable (plan §2, §3.4).

    lipsync.analyze(video_path) -> LipSyncResult

Never raises. Silent video, no speech, or no face all return a contract-valid
result with lipsync_quality=0 and a degraded_reason, so fusion simply runs on
rPPG alone (plan §6.2).

THE DISCRIMINATOR IS LAG CONSISTENCY, NOT LAG
A genuine recording can carry a large constant A/V offset from the muxer or the
encoder — this repo's own clips range from +33 ms to -168 ms. What a real
recording does NOT do is drift window to window. Wav2Lip-style generation
produces mouth motion frame by frame with no global clock, so the best-matching
lag wanders. Hence `lag_iqr_ms` carries the most weight (plan §3.4), and the
constant part is removed up front via `av_start_offset_sec`.

VISUAL SIGNAL
True Mouth Aspect Ratio needs inner-lip landmarks, which need FaceLandmarker
weights that are not vendored yet (mediapipe 1.0.0 removed `mp.solutions`, so the
legacy FaceMesh path is gone too). Until those land, the visual channel is mouth
region motion energy — mean |frame difference| inside the mouth ROI. This is a
proxy for the MAR *derivative*, which is what the design correlates against the
speech envelope anyway, so the downstream maths is unchanged. Swap in real MAR
when the weights arrive; nothing else needs to move.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import signal as sps

from src.common.contracts import LipSyncResult
from src.fusion import load_thresholds
from src.lipsync.audio_io import load_audio, voiced_mask
from src.rppg import backends
from src.rppg.analyze import read_frames
from src.rppg.signal_core import detrend_linear, resample_to_uniform

MOUTH_ROI = (0.28, 0.60, 0.72, 0.95)     # x0, y0, x1, y1 as fractions of the face box
ENV_FS = 50.0                            # Hz, common grid for both channels

NEURAL_CKPT = Path(__file__).resolve().parents[2] / "models" / "lipsync_detector.pth"


@lru_cache(maxsize=1)
def _neural_model():
    """Load the trained detector once, or return None if there is no checkpoint.

    Cached because the model was previously rebuilt on every call — including once
    per window during aggregation, where it re-scored the SAME first 10s of video
    each time (evaluate_video takes a path, not a window). That was ~15s of the
    runtime producing a value that could not vary by window.
    """
    if not NEURAL_CKPT.exists():
        return None
    try:
        import torch

        from src.lipsync.detector import HighAccuracyLipSyncDetector

        model = HighAccuracyLipSyncDetector(pretrained=False)
        state = torch.load(NEURAL_CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("state_dict", state))
        model.eval()
        return model
    except Exception:
        return None


@lru_cache(maxsize=32)
def _neural_score(video_path: str):
    """P(manipulated) from the neural branch, or None when untrained."""
    model = _neural_model()
    if model is None:
        return None
    try:
        from src.lipsync.detector import evaluate_video

        return float(evaluate_video(model, video_path, device="cpu", max_sec=10.0)
                     ["prob_manipulated"])
    except Exception:
        return None


def _degraded(session_id: str, reason: str, t0: float, fps: float = 25.0) -> LipSyncResult:
    return LipSyncResult(
        session_id=session_id,
        lipsync_manipulation_score=0.5,     # neutral: never accuse on absent evidence
        lipsync_quality=0.0,
        median_lag_ms=None,
        lag_iqr_ms=None,
        mean_peak_ncc=None,
        speech_windows_used=0,
        lag_resolution_ms=1000.0 / max(fps, 1.0),
        mar_decimated=[],
        envelope_decimated=[],
        processing_time_ms=int((time.time() - t0) * 1000),
        degraded_reason=reason,
    )


def mouth_motion_series(frames_bgr, boxes: np.ndarray) -> np.ndarray:
    """Per-frame articulation energy inside the mouth ROI. NaN where unusable."""
    n = len(frames_bgr)
    out = np.full(n, np.nan)
    prev = None
    for i, frame in enumerate(frames_bgr):
        if np.isnan(boxes[i]).any():
            prev = None
            continue
        x, y, w, h = boxes[i]
        fx0, fy0, fx1, fy1 = MOUTH_ROI
        H, W = frame.shape[:2]
        x0, x1 = int(np.clip(x + fx0 * w, 0, W - 1)), int(np.clip(x + fx1 * w, 1, W))
        y0, y1 = int(np.clip(y + fy0 * h, 0, H - 1)), int(np.clip(y + fy1 * h, 1, H))
        if x1 - x0 < 8 or y1 - y0 < 8:
            prev = None
            continue
        roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi = cv2.resize(roi, (64, 48))
        if prev is not None:
            # Normalised so it does not scale with face size or exposure.
            out[i] = float(np.mean(np.abs(roi - prev)) / (roi.mean() + 1e-6))
        prev = roi
    return out


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    hi = min(hi, 0.95 * 0.5 * fs)
    if lo >= hi or len(x) < 27:
        return np.zeros_like(x)
    b, a = sps.butter(3, [lo / (0.5 * fs), hi / (0.5 * fs)], btype="bandpass")
    if len(x) <= 3 * max(len(a), len(b)):
        return np.zeros_like(x)
    # detrend_linear, not scipy.signal.detrend: the latter calls LAPACK, which
    # aborts the process under OMP Error #15 once torch is loaded. See signal_core.
    return sps.filtfilt(b, a, detrend_linear(x))


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def _ncc_lag(a: np.ndarray, b: np.ndarray, fs: float, max_lag_ms: float):
    """Normalised cross-correlation peak with parabolic sub-sample interpolation.

    Plain `np.correlate` on unnormalised signals is dominated by DC and carries a
    triangular length bias toward zero lag — it would report "perfect sync" for
    almost anything. Both series are z-scored and the result divided by N.
    """
    n = min(len(a), len(b))
    if n < 16:
        return None, 0.0
    a, b = _z(a[:n]), _z(b[:n])
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None, 0.0

    full = sps.correlate(a, b, mode="full", method="fft") / n
    lags = sps.correlation_lags(n, n, mode="full")
    lim = int(max_lag_ms / 1000.0 * fs)
    keep = np.abs(lags) <= max(lim, 1)
    full, lags = full[keep], lags[keep]
    if full.size < 3:
        return None, 0.0

    k = int(np.argmax(full))
    peak = float(full[k])
    if 0 < k < len(full) - 1:
        y0, y1, y2 = full[k - 1], full[k], full[k + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    else:
        delta = 0.0
    return (lags[k] + delta) / fs * 1000.0, peak


def analyze(video_path: str, session_id: str = "local",
            thresholds: Optional[dict] = None,
            max_sec: float = 20.0,
            start_sec: float = 0.0,
            clip=None) -> LipSyncResult:
    """Never raises. `clip` accepts a pre-decoded DecodedClip so aggregation can
    reuse one decode and one detection pass across every window."""
    t0 = time.time()
    try:
        cfg = thresholds or load_thresholds()
        lc = cfg["lipsync"]

        if clip is not None:
            audio = clip.audio
            if audio is None or not audio.ok:
                return _degraded(session_id,
                                 (audio.reason if audio else None) or "no_audio", t0)
            frames, t, nominal = clip.frames, clip.t, clip.nominal_fps
            boxes = clip.boxes
        else:
            if not Path(video_path).exists():
                return _degraded(session_id, "file_not_found", t0)

            audio = load_audio(video_path, max_sec=start_sec + max_sec)
            if not audio.ok:
                return _degraded(session_id, audio.reason or "no_audio", t0)
            if start_sec > 0:
                skip = int(start_sec * audio.sr)
                audio.samples = audio.samples[skip:]

            frames, t, nominal, _w = read_frames(video_path, max_sec=max_sec,
                                                 start_sec=start_sec)
            boxes = None

        if len(frames) < 60:
            return _degraded(session_id, "too_few_frames", t0, nominal)

        if boxes is None:
            boxes, _counts = backends.OpenCVBackend().detect_boxes(frames)
            boxes = backends.smooth_boxes(boxes)
        if np.isnan(boxes[:, 0]).mean() > 0.9:
            return _degraded(session_id, "no_face_detected", t0, nominal)

        motion = mouth_motion_series(frames, boxes)
        if np.isfinite(motion).mean() < 0.5:
            return _degraded(session_id, "mouth_roi_unusable", t0, nominal)

        # --- put both channels on one clock ------------------------------
        # The video timeline is shifted by the muxer's own A/V offset so it is
        # not mistaken for a lip-sync error.
        t_aligned = t - audio.av_start_offset_sec
        vis = resample_to_uniform(t_aligned, motion, ENV_FS)
        if vis.size < 64:
            return _degraded(session_id, "visual_series_too_short", t0, nominal)

        env_raw = np.abs(audio.samples)
        hop = max(int(audio.sr / ENV_FS), 1)
        n_env = len(env_raw) // hop
        env = np.array([env_raw[i * hop:(i + 1) * hop].mean() for i in range(n_env)])

        n = min(len(vis), len(env))
        if n < 64:
            return _degraded(session_id, "series_too_short", t0, nominal)
        vis, env = vis[:n], env[:n]

        lo, hi = lc["envelope_band_hz"]
        vis_f = _bandpass(vis, ENV_FS, lo, hi)
        env_f = _bandpass(env, ENV_FS, lo, hi)

        # --- VAD gate ----------------------------------------------------
        vmask = voiced_mask(audio.samples, audio.sr)
        if vmask.size:
            vt = np.arange(vmask.size) * 0.03
            voiced = np.interp(np.arange(n) / ENV_FS, vt, vmask.astype(float)) > 0.5
        else:
            voiced = np.ones(n, dtype=bool)

        # --- windowed alignment ------------------------------------------
        win = int(lc["window_sec"] * ENV_FS)
        hop_w = int(lc["hop_sec"] * ENV_FS)
        lags, peaks = [], []
        for s in range(0, n - win + 1, max(hop_w, 1)):
            if voiced[s:s + win].mean() < 0.5:
                continue
            lag, peak = _ncc_lag(vis_f[s:s + win], env_f[s:s + win],
                                 ENV_FS, lc["max_lag_ms"])
            if lag is not None:
                lags.append(lag)
                peaks.append(peak)

        used = len(lags)
        # Coverage: did we get enough voiced windows to measure at all?
        coverage = float(np.clip(used / max(lc["min_voiced_windows"], 1), 0.0, 1.0))
        # Reliability: was the alignment measurement itself trustworthy?
        #
        # Coverage alone was the whole quality score, which meant "I ran" was
        # being reported as "I know something". A lag read off a correlation that
        # never peaked is noise regardless of how many windows produced it, and
        # every lag-derived feature (median_lag, lag_iqr) inherits that noise.
        # Reported quality must reflect whether the measurement DISCRIMINATED,
        # not merely whether it completed.
        peak_ref = float(lc.get("reliable_peak_ncc", 0.30))
        reliability = float(np.clip(np.mean(peaks) / peak_ref, 0.0, 1.0)) if peaks else 0.0
        quality = float(np.clip(coverage * reliability, 0.0, 1.0))

        if used < 3:
            r = _degraded(session_id, "insufficient_speech_windows", t0, nominal)
            return r

        median_lag = float(np.median(lags))
        lag_iqr = float(np.subtract(*np.percentile(lags, [75, 25])))
        mean_ncc = float(np.mean(peaks))

        a = np.clip((lag_iqr - lc["lag_iqr_floor_ms"])
                    / max(lc["lag_iqr_ceil_ms"] - lc["lag_iqr_floor_ms"], 1e-6), 0, 1)
        b = 1.0 - np.clip(mean_ncc / 0.4, 0, 1)
        c = np.clip((abs(median_lag) - 60.0) / 200.0, 0, 1)
        w = lc["weights"]
        classical = w["iqr"] * a + w["ncc"] * b + w["offset"] * c

        # Neural branch — fused ONLY when trained weights exist.
        #
        # It was previously constructed as HighAccuracyLipSyncDetector(pretrained=False)
        # with no checkpoint load, so every call built a freshly RANDOM network.
        # Measured on one unchanged video: 0.5193, 0.5109, 0.4898 across three runs.
        # That noise was being mixed in at weight 0.2. An untrained model does not
        # contribute weak evidence, it contributes none — so it is now excluded
        # entirely rather than diluted, and the classical weights stay normalised.
        neural_prob = _neural_score(video_path)
        if neural_prob is None:
            raw = float(np.clip(classical, 0, 1))
        else:
            nw = float(lc.get("neural_weight", 0.2))
            raw = float(np.clip((1.0 - nw) * classical + nw * neural_prob, 0, 1))

        # Same shrinkage discipline as rPPG: contribute in proportion to evidence.
        score = 0.5 + (raw - 0.5) * quality
        reason = None if quality >= 1.0 else f"partial_evidence_shrink_{quality:.2f}"


        step_v = max(1, len(vis_f) // 500)
        step_e = max(1, len(env_f) // 500)

        return LipSyncResult(
            session_id=session_id,
            lipsync_manipulation_score=round(score, 4),
            lipsync_quality=round(quality, 4),
            median_lag_ms=round(median_lag, 2),
            lag_iqr_ms=round(lag_iqr, 2),
            mean_peak_ncc=round(mean_ncc, 4),
            speech_windows_used=used,
            lag_resolution_ms=round(1000.0 / ENV_FS, 2),
            mar_decimated=[float(v) for v in vis_f[::step_v][:500]],
            envelope_decimated=[float(v) for v in env_f[::step_e][:500]],
            processing_time_ms=int((time.time() - t0) * 1000),
            degraded_reason=reason,
        )

    except Exception as exc:  # noqa: BLE001
        return _degraded(session_id, f"unhandled:{type(exc).__name__}", t0)
