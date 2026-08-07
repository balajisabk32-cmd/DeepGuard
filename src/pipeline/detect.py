"""End-to-end detection: rPPG + lip-sync -> quality-weighted fusion.

    python -m src.pipeline.detect "<video>"

This is the join the architecture was built for (plan §3.1). Each modality reports
a manipulation score AND its own evidence quality; fusion weights by quality, so a
modality that cannot see anything cannot drag the verdict.

Neither analyzer raises, so this does not either.
"""

from __future__ import annotations

import argparse
import sys
import time

from src.common.contracts import FusionResult
from src.fusion import load_thresholds, score
from src.lipsync import analyze as lipsync_analyze
from src.pipeline.decode import decode_clip
from src.rppg import analyze as rppg_analyze
from src.visual import analyze_video as visual_analyze_video
from src.visual.aigen import analyze as aigen_analyze
from src.visual.pixel_forensics import analyze as pixel_analyze


def detect(video_path: str, session_id: str = "cli",
           thresholds: dict | None = None, clip: object | None = None) -> tuple[FusionResult, object, object, object, object, object]:
    cfg = thresholds or load_thresholds()
    t0 = time.time()

    # One decode + one detection pass shared by all three modalities.
    if clip is None:
        clip = decode_clip(video_path, max_sec=20.0)

    r = rppg_analyze(video_path, session_id=session_id, thresholds=cfg,
                     prefer_mediapipe=False, clip=clip)
    l = lipsync_analyze(video_path, session_id=session_id, thresholds=cfg, clip=clip)
    v = visual_analyze_video(video_path, clip=clip)
    # Fourth mandated channel. Currently reports features only — it is not
    # calibrated, so it returns a neutral score and zero quality, which means
    # log-odds fusion gives it exactly zero influence.
    px = pixel_analyze(clip.frames, clip.boxes)
    # Fifth channel: fully synthetic imagery. Full frames, no face crop. Held at
    # a neutral score with quality 0 — see aigen.analyze() for the measurement.
    ag = aigen_analyze(clip.frames)

    warnings = []
    if r.degraded_reason:
        warnings.append(f"rppg:{r.degraded_reason}")
    if l.degraded_reason:
        warnings.append(f"lipsync:{l.degraded_reason}")
    if v.degraded_reason:
        warnings.append(f"visual:{v.degraded_reason}")
    if px.degraded_reason:
        warnings.append(f"pixel:{px.degraded_reason}")
    if ag.degraded_reason:
        warnings.append(f"aigen:{ag.degraded_reason}")

    fused = score(
        modality_scores={"rppg": r.rppg_manipulation_score,
                         "lipsync": l.lipsync_manipulation_score,
                         "visual": v.visual_manipulation_score,
                         "pixel": px.pixel_manipulation_score,
                         "aigen": ag.aigen_manipulation_score},
        modality_quality={"rppg": r.rppg_quality,
                          "lipsync": l.lipsync_quality,
                          "visual": v.visual_quality,
                          "pixel": px.pixel_quality,
                          "aigen": ag.aigen_quality},
        session_id=session_id,
        total_processing_time_ms=int((time.time() - t0) * 1000),
        warnings=warnings,
        thresholds=cfg,
    )
    return fused, r, l, v, px, ag


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepGuard end-to-end detection")
    ap.add_argument("video")
    ap.add_argument("--session-id", default="cli")
    args = ap.parse_args(argv)

    fused, r, l, v, px, ag = detect(args.video, session_id=args.session_id)

    hr = "n/a" if r.heart_rate_bpm is None else f"{r.heart_rate_bpm:.0f} bpm"
    lag = "n/a" if l.median_lag_ms is None else f"{l.median_lag_ms:+.0f} ms"
    iqr = "n/a" if l.lag_iqr_ms is None else f"{l.lag_iqr_ms:.0f} ms"

    print("=" * 66)
    print(f"DEEPGUARD  -  {args.video}")
    print("=" * 66)
    print(f"  rPPG      score {r.rppg_manipulation_score:.3f}  quality {r.rppg_quality:.3f}"
          f"   HR {hr}  SNR {r.band_snr_db:+.1f} dB")
    if r.map_patches_used:
        print(f"            map: {r.map_patches_used} patches, corr_p25 {r.map_corr_p25:+.3f},"
              f" mean SNR {r.map_mean_patch_snr_db:+.2f} dB")
    print(f"  lip-sync  score {l.lipsync_manipulation_score:.3f}  quality {l.lipsync_quality:.3f}"
          f"   lag {lag}  IQR {iqr}  windows {l.speech_windows_used}")
    print(f"  visual    score {v.visual_manipulation_score:.3f}  quality {v.visual_quality:.3f}"
          f"   frames {v.frames_scored}  spread {v.score_spread:.3f}  face {v.mean_face_px:.0f}px")
    print(f"  pixel     score {px.pixel_manipulation_score:.3f}  quality {px.pixel_quality:.3f}"
          f"   frames {px.frames_used}  ({px.degraded_reason or 'calibrated'})")
    print(f"  ai-gen    score {ag.aigen_manipulation_score:.3f}  quality {ag.aigen_quality:.3f}"
          f"   frames {ag.frames_scored}  raw {ag.raw_score}  ({ag.degraded_reason or 'admitted'})")
    p_fake = fused.manipulation_probability
    p_real = 1.0 - p_fake
    print("-" * 66)
    print(f"  VERDICT               {fused.verdict.value}")
    print(f"  CONFIDENCE            {p_real * 100:.1f}% Real  |  {p_fake * 100:.1f}% Fake")
    print(f"  P(manipulated)        {fused.manipulation_probability:.3f}")
    print(f"  evidence weight       {fused.evidence_weight:.3f}")
    print(f"  {fused.explanation}")
    if fused.warnings:
        print(f"  warnings: {', '.join(fused.warnings)}")
    print(f"  total {fused.total_processing_time_ms} ms")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
