"""rPPG orchestrator — Role 1's deliverable (plan §2, §3.4).

    rppg.analyze(video_path) -> RPPGResult

CONTRACT: this function never raises. Any failure path returns a contract-valid
RPPGResult with rppg_quality=0, a neutral manipulation score, and a
degraded_reason. A module that throws produces a 500 in front of a judge
(plan §2.4 rule).

SCORING (plan §3.4): the score is cross-region DISAGREEMENT, not signal strength.
"No clean pulse => fake" is wrong for face swaps: the composite keeps authentic
skin outside the blend mask and synthesised skin inside it, so what breaks is
agreement BETWEEN regions, not the presence of a pulse.
"""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.common.contracts import RPPGResult
from src.fusion import load_thresholds
from src.rppg import backends
from src.rppg.signal_core import (
    best_extraction,
    hr_with_ci,
    peak_frequency,
    resample_to_uniform,
)

MAX_ANALYSIS_SEC = 20.0
MIN_ANALYSIS_SEC = 4.0


def _degraded(session_id: str, reason: str, t0: float, warnings=None) -> RPPGResult:
    return RPPGResult(
        session_id=session_id,
        heart_rate_bpm=None,
        heart_rate_ci_bpm=None,
        rppg_manipulation_score=0.5,     # neutral — never accuse on absent evidence
        rppg_quality=0.0,
        band_snr_db=-99.0,
        cross_region_corr_min=0.0,
        cross_region_hr_spread_bpm=0.0,
        phase_dispersion=0.0,
        waveform_decimated=[],
        processing_time_ms=int((time.time() - t0) * 1000),
        degraded_reason=reason,
    )


def read_frames(video_path: str, max_sec: float = MAX_ANALYSIS_SEC,
                start_sec: float = 0.0):
    """Decode frames with REAL presentation timestamps.

    cv2.CAP_PROP_POS_MSEC reports the container timestamp of each frame. Using it
    instead of index/CAP_PROP_FPS is what keeps variable-frame-rate input honest
    (plan §6.3). Falls back to the nominal rate only when timestamps are unusable,
    and says so in the warnings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], np.array([]), 0.0, ["could_not_open_video"]

    nominal = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(nominal) or nominal <= 0:
        nominal = 30.0

    # Seek by timestamp for multi-window aggregation. CAP_PROP_POS_MSEC seeking
    # lands on the nearest keyframe, so the true start is re-read from the stream
    # rather than assumed — the time axis stays honest.
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    frames, stamps, warnings = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Read POS_MSEC *after* the decode: before it, the property reports the
        # position of the frame about to be read, so the first two samples are
        # both 0.0 and the monotonicity check below rejects a perfectly good
        # timestamp track.
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        frames.append(frame)
        stamps.append(ts / 1000.0)
        if stamps[-1] - stamps[0] >= max_sec:
            break
    cap.release()

    t = np.array(stamps, dtype=np.float64)
    # Container timestamps are sometimes all-zero or non-monotonic.
    if t.size < 2 or not np.all(np.diff(t) > 0) or t[-1] <= 0:
        t = np.arange(len(frames), dtype=np.float64) / nominal
        warnings.append("pts_unusable_fell_back_to_nominal_fps")

    return frames, t, float(nominal), warnings


def _circular_std(angles: np.ndarray) -> float:
    if angles.size == 0:
        return 0.0
    r = np.abs(np.mean(np.exp(1j * angles)))
    return float(np.sqrt(max(-2.0 * np.log(max(r, 1e-12)), 0.0)))


def _cross_region_features(sigs: dict[str, np.ndarray], fs: float):
    """Pairwise correlation, HR spread across ROIs, and phase dispersion at f0."""
    names = [k for k, v in sigs.items() if v.size > 32]
    if len(names) < 2:
        return 0.0, 0.0, 0.0, {}

    n = min(len(sigs[k]) for k in names)
    trimmed = {k: sigs[k][:n] for k in names}

    freqs = {}
    for k in names:
        f0, _ = peak_frequency(trimmed[k], fs)
        if f0 is not None:
            freqs[k] = f0

    corrs, phases = [], []
    for a, b in combinations(names, 2):
        xa, xb = trimmed[a], trimmed[b]
        if xa.std() < 1e-12 or xb.std() < 1e-12:
            corrs.append(0.0)
            continue
        corrs.append(float(np.corrcoef(xa, xb)[0, 1]))

        # Cross-spectral phase at the shared dominant frequency.
        f_ref = np.median(list(freqs.values())) if freqs else None
        if f_ref:
            k = int(round(f_ref * n / fs))
            if 0 < k < n // 2:
                fa, fb = np.fft.rfft(xa), np.fft.rfft(xb)
                phases.append(float(np.angle(fa[k] * np.conj(fb[k]))))

    spread = (max(freqs.values()) - min(freqs.values())) * 60 if len(freqs) >= 2 else 0.0
    return (
        float(min(corrs)) if corrs else 0.0,
        float(spread),
        _circular_std(np.array(phases)),
        freqs,
    )


def analyze(
    video_path: str,
    session_id: str = "local",
    thresholds: Optional[dict] = None,
    prefer_mediapipe: bool = True,
    start_sec: float = 0.0,
    max_sec: float = MAX_ANALYSIS_SEC,
    clip=None,
) -> RPPGResult:
    """Full rPPG analysis. Never raises.

    `clip` accepts a pre-decoded DecodedClip so multi-window aggregation can
    decode and detect once instead of once per (window x modality).
    """
    t0 = time.time()
    try:
        cfg = thresholds or load_thresholds()
        rcfg = cfg["rppg"]

        if clip is not None:
            frames, t, nominal, warns = clip.frames, clip.t, clip.nominal_fps, list(clip.warnings)
            boxes, counts = clip.boxes, clip.counts
        else:
            if not Path(video_path).exists():
                return _degraded(session_id, "file_not_found", t0)
            frames, t, nominal, warns = read_frames(video_path, max_sec=max_sec,
                                                    start_sec=start_sec)
            boxes = counts = None

        if len(frames) < 60:
            return _degraded(session_id, "too_few_frames", t0, warns)

        duration = float(t[-1] - t[0])
        if duration < MIN_ANALYSIS_SEC:
            return _degraded(session_id, "clip_shorter_than_4s", t0, warns)

        if boxes is None:
            backend = backends.get_backend(prefer_mediapipe)
            boxes, counts = backend.detect_boxes(frames)
            boxes = backends.smooth_boxes(boxes)

        detection_rate = float(np.mean(~np.isnan(boxes[:, 0])))
        if detection_rate < 0.10:
            return _degraded(session_id, "no_face_detected", t0, warns)

        # Reuse the hoisted per-frame work when aggregation supplied it.
        if clip is not None and getattr(clip, "roi_series", None) is not None:
            series = clip.roi_series
        else:
            series, _ = backends.extract_roi_series(frames, boxes, counts)

        # Effective sampling rate from REAL elapsed time, not the nominal rate.
        fs = (len(frames) - 1) / duration

        sigs, best_name, best_snr, best_sig = {}, "none", -np.inf, np.array([])
        for j, roi in enumerate(backends.ROI_NAMES):
            rgb_uniform = np.column_stack([
                resample_to_uniform(t, series[:, j, c], fs) for c in range(3)
            ]) if np.isfinite(series[:, j, :]).any() else np.array([])

            if rgb_uniform.size == 0 or rgb_uniform.shape[0] < 32:
                continue

            name, sig, _f0, snr = best_extraction(rgb_uniform, fs)
            if sig.size:
                sigs[roi] = sig
            if snr > best_snr:
                best_name, best_snr, best_sig = name, snr, sig

        if not sigs or best_sig.size == 0:
            return _degraded(session_id, "no_usable_roi_signal", t0, warns)

        # ---- PPG spatial-temporal map (preferred when enabled) ----
        # 30 patches -> up to 435 pairwise comparisons, versus 3 pairs from the
        # 3-ROI path where `min` of 3 noisy estimates decided the verdict.
        map_res = None
        mcfg = rcfg.get("map", {})
        if mcfg.get("enabled"):
            try:
                from src.rppg.ppgmap import analyze_map, build_stmap, map_manipulation_score
                gr, gc = mcfg.get("grid", [6, 5])
                stmap = (clip.stmap if clip is not None
                         and getattr(clip, "stmap", None) is not None
                         else build_stmap(frames, boxes, rows=gr, cols=gc))
                map_res = analyze_map(stmap, t, fs)
                if map_res.degraded_reason:
                    map_res = None
            except Exception:
                map_res = None  # map is an enhancement; never let it break the run

        bpm, ci, snr = hr_with_ci(best_sig, fs)
        quality = float(np.clip(
            (snr - rcfg["quality_snr_floor_db"])
            / (rcfg["quality_snr_ceil_db"] - rcfg["quality_snr_floor_db"]),
            0.0, 1.0,
        ))

        corr_min, spread_bpm, phase_disp, _ = _cross_region_features(sigs, fs)

        # The map's patch-averaged SNR is a better quality estimate than a single
        # ROI's, because it is an average over ~20 independent measurements.
        if map_res is not None and np.isfinite(map_res.mean_patch_snr_db):
            quality = float(np.clip(
                (map_res.mean_patch_snr_db - rcfg["quality_snr_floor_db"])
                / (rcfg["quality_snr_ceil_db"] - rcfg["quality_snr_floor_db"]),
                0.0, 1.0,
            ))
            corr_min = map_res.corr_p25
            spread_bpm = map_res.hr_spread_bpm
            phase_disp = map_res.phase_dispersion

        # ---- graded contribution, not a hard gate ----
        #
        # A hard cutoff makes rPPG all-or-nothing: above it the score is trusted
        # fully, below it the modality contributes literally nothing. On real
        # compressed footage almost everything lands below, so the modality
        # silently drops out of every verdict.
        #
        # Instead, shrink the score toward neutral in proportion to evidence:
        #
        #     score = 0.5 + (raw - 0.5) * shrink(quality)
        #
        # Full abstention now happens only below `min_quality_for_evidence`, where
        # the signal is pure noise. Between there and `quality_trust_full` the
        # modality still moves the verdict, just proportionally to what it knows.
        # This is shrinkage toward the prior, not a confidence fudge — a weak
        # measurement is still a measurement.
        floor = rcfg.get("min_quality_for_evidence", 0.05)
        trust = rcfg.get("quality_trust_full", rcfg["min_quality_for_scoring"])

        if len(sigs) < 2:
            score, reason, shrink = 0.5, "too_few_usable_rois", 0.0
        elif quality <= floor:
            score, reason, shrink = 0.5, "insufficient_pulse_snr", 0.0
        else:
            w = rcfg["weights"]
            d1 = 1.0 - np.clip(corr_min, 0.0, 1.0)
            d2 = np.clip(spread_bpm / rcfg["hr_spread_norm_bpm"], 0.0, 1.0)
            d3 = np.clip(phase_disp / (np.pi / 2), 0.0, 1.0)
            raw = float(np.clip(w["corr"] * d1 + w["spread"] * d2 + w["phase"] * d3, 0, 1))

            shrink = float(np.clip((quality - floor) / max(trust - floor, 1e-6), 0.0, 1.0))
            score = 0.5 + (raw - 0.5) * shrink
            reason = None if shrink >= 1.0 else f"partial_evidence_shrink_{shrink:.2f}"

        if detection_rate < cfg["ingest"]["min_face_detection_rate"]:
            warns.append(f"low_face_detection_rate_{detection_rate:.2f}")

        step = max(1, len(best_sig) // 500)
        waveform = [float(v) for v in best_sig[::step][:500]]

        return RPPGResult(
            session_id=session_id,
            heart_rate_bpm=None if bpm is None else round(bpm, 1),
            heart_rate_ci_bpm=None if ci is None else (round(ci[0], 1), round(ci[1], 1)),
            rppg_manipulation_score=round(score, 4),
            rppg_quality=round(quality, 4),
            band_snr_db=round(float(snr), 2) if np.isfinite(snr) else -99.0,
            cross_region_corr_min=round(corr_min, 4),
            cross_region_hr_spread_bpm=round(spread_bpm, 2),
            phase_dispersion=round(phase_disp, 4),
            waveform_decimated=waveform,
            processing_time_ms=int((time.time() - t0) * 1000),
            degraded_reason=reason,
            map_patches_used=None if map_res is None else map_res.n_patches_used,
            map_corr_p25=None if map_res is None else round(map_res.corr_p25, 4),
            map_mean_patch_snr_db=(
                None if map_res is None else round(float(map_res.mean_patch_snr_db), 2)
            ),
            map_hr_temporal_jump_bpm=(
                None if map_res is None else round(map_res.hr_temporal_jump_bpm, 2)
            ),
            map_coherence=(
                None if map_res is None or map_res.coherence_map.size == 0
                else [[None if np.isnan(v) else round(float(v), 3) for v in row]
                      for row in map_res.coherence_map]
            ),
        )

    except Exception as exc:  # noqa: BLE001 - the whole point is to never propagate
        return _degraded(session_id, f"unhandled:{type(exc).__name__}", t0)
