"""Multi-clip test-time aggregation.

    python -m src.pipeline.aggregate "<video>"

Single-window predictions are noisy. This repo has measured that directly: the
same deepfake clip gave band SNR -2.80 dB over a 20s window and -0.60 dB over
13s. A 2.2 dB swing from the window length alone is not physiology, it is
estimator variance — and a verdict read off one window inherits all of it.

So: score several OVERLAPPING windows per video and aggregate.

WHY MEDIAN, NOT MEAN
The failure mode is a single window landing on a scene cut, an occlusion, or a
burst of head motion, which produces one wild score. A mean carries that
straight into the verdict; a median ignores it. The spread across windows is
kept and reported, because a detector whose own windows disagree should say so
rather than average the disagreement away.

The per-window spread is also usable evidence in its own right: authentic clips
tend to score consistently, and window-to-window instability is itself weak
evidence of trouble. It is REPORTED here, not folded into the score, because
nothing in this repo has been measured against a labelled corpus yet.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.common.contracts import FusionResult
from src.fusion import load_thresholds, score
from src.lipsync import analyze as lipsync_analyze
from src.pipeline.decode import decode_clip
from src.rppg import analyze as rppg_analyze

WINDOW_SEC = 12.0      # >= rppg min_window_sec; below this HR cannot be resolved
HOP_SEC = 6.0          # 50% overlap
MAX_WINDOWS = 5        # latency guard


@dataclass
class WindowScore:
    start_sec: float
    rppg_score: float
    rppg_quality: float
    lipsync_score: float
    lipsync_quality: float
    notes: list[str] = field(default_factory=list)


def video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(n / fps) if fps and fps > 0 and n > 0 else 0.0


def plan_windows(duration: float, window_sec: float = WINDOW_SEC,
                 hop_sec: float = HOP_SEC, max_windows: int = MAX_WINDOWS) -> list[float]:
    """Overlapping window starts. Short clips fall back to a single window."""
    if duration <= window_sec * 1.2:
        return [0.0]
    starts = list(np.arange(0.0, max(duration - window_sec, 0.0) + 1e-6, hop_sec))
    if len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in idx]
    return [float(s) for s in starts]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median that respects per-window evidence weight."""
    if values.size == 0:
        return 0.5
    if weights.sum() <= 1e-9:
        return float(np.median(values))
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w) / w.sum()
    return float(v[int(np.searchsorted(c, 0.5))])


def aggregate(video_path: str, session_id: str = "cli",
              thresholds: dict | None = None,
              window_sec: float = WINDOW_SEC,
              hop_sec: float = HOP_SEC,
              max_windows: int = MAX_WINDOWS):
    cfg = thresholds or load_thresholds()
    t0 = time.time()

    duration = video_duration(video_path)
    starts = plan_windows(duration, window_sec, hop_sec, max_windows)

    # Decode and detect ONCE for the whole clip; every window is a cheap view.
    clip = decode_clip(video_path, max_sec=max(duration, window_sec))

    windows: list[WindowScore] = []
    for s in starts:
        view = clip.window(s, window_sec)
        r = rppg_analyze(video_path, session_id=session_id, thresholds=cfg,
                         prefer_mediapipe=False, clip=view)
        l = lipsync_analyze(video_path, session_id=session_id, thresholds=cfg,
                            clip=view)
        notes = [n for n in (r.degraded_reason, l.degraded_reason) if n]
        windows.append(WindowScore(
            start_sec=s,
            rppg_score=r.rppg_manipulation_score, rppg_quality=r.rppg_quality,
            lipsync_score=l.lipsync_manipulation_score, lipsync_quality=l.lipsync_quality,
            notes=notes,
        ))

    r_s = np.array([w.rppg_score for w in windows])
    r_q = np.array([w.rppg_quality for w in windows])
    l_s = np.array([w.lipsync_score for w in windows])
    l_q = np.array([w.lipsync_quality for w in windows])

    agg_rppg = _weighted_median(r_s, r_q)
    agg_lip = _weighted_median(l_s, l_q)

    warnings = []
    if len(windows) > 1:
        # Spread is reported, never silently folded into the score.
        if r_s.std() > 0.15:
            warnings.append(f"rppg_window_spread_{r_s.std():.2f}")
        if l_s.std() > 0.15:
            warnings.append(f"lipsync_window_spread_{l_s.std():.2f}")
    for n in sorted({n for w in windows for n in w.notes}):
        warnings.append(n)

    fused = score(
        modality_scores={"rppg": agg_rppg, "lipsync": agg_lip},
        modality_quality={"rppg": float(r_q.mean()), "lipsync": float(l_q.mean())},
        session_id=session_id,
        total_processing_time_ms=int((time.time() - t0) * 1000),
        warnings=warnings,
        thresholds=cfg,
    )
    return fused, windows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-window aggregated detection")
    ap.add_argument("video")
    ap.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    ap.add_argument("--hop-sec", type=float, default=HOP_SEC)
    ap.add_argument("--max-windows", type=int, default=MAX_WINDOWS)
    args = ap.parse_args(argv)

    fused, windows = aggregate(args.video, window_sec=args.window_sec,
                               hop_sec=args.hop_sec, max_windows=args.max_windows)

    print("=" * 72)
    print(f"MULTI-WINDOW AGGREGATION  -  {args.video}")
    print("=" * 72)
    print(f"{'start':>7} | {'rPPG score':>11} {'qual':>6} | {'lip score':>10} {'qual':>6} | notes")
    print("-" * 72)
    for w in windows:
        print(f"{w.start_sec:>6.1f}s | {w.rppg_score:>11.3f} {w.rppg_quality:>6.3f} | "
              f"{w.lipsync_score:>10.3f} {w.lipsync_quality:>6.3f} | {','.join(w.notes)[:22]}")
    print("-" * 72)

    r_s = np.array([w.rppg_score for w in windows])
    l_s = np.array([w.lipsync_score for w in windows])
    print(f"  windows            {len(windows)}")
    print(f"  rPPG    spread     {r_s.std():.3f}   (std across windows)")
    print(f"  lipsync spread     {l_s.std():.3f}")
    print(f"  VERDICT            {fused.verdict.value}")
    print(f"  P(manipulated)     {fused.manipulation_probability:.3f}")
    print(f"  evidence weight    {fused.evidence_weight:.3f}")
    if fused.warnings:
        print(f"  warnings           {', '.join(fused.warnings)}")
    print(f"  total              {fused.total_processing_time_ms} ms")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
