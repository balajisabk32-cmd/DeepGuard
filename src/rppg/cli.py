"""Report runner:  python -m src.rppg.cli "<video>"

Prints the full RPPGResult plus the reasoning behind it, so the numbers can be
checked by hand rather than trusted.
"""

from __future__ import annotations

import argparse
import sys

from src.rppg.analyze import analyze


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="rPPG analysis report")
    ap.add_argument("video")
    ap.add_argument("--session-id", default="cli")
    ap.add_argument("--no-mediapipe", action="store_true",
                    help="force the OpenCV backend even if weights are present")
    args = ap.parse_args(argv)

    r = analyze(args.video, session_id=args.session_id,
                prefer_mediapipe=not args.no_mediapipe)

    hr = "n/a" if r.heart_rate_bpm is None else (
        f"{r.heart_rate_bpm:.1f} bpm"
        + (f"  (CI {r.heart_rate_ci_bpm[0]:.0f}-{r.heart_rate_ci_bpm[1]:.0f})"
           if r.heart_rate_ci_bpm else "")
    )

    print("=" * 62)
    print(f"rPPG REPORT  -  {args.video}")
    print("=" * 62)
    print(f"  heart rate            {hr}")
    print(f"  band SNR              {r.band_snr_db:.2f} dB")
    print(f"  rppg_quality          {r.rppg_quality:.3f}   (evidence strength)")
    print("  --- cross-region consistency ---")
    print(f"  min pairwise corr     {r.cross_region_corr_min:+.3f}")
    print(f"  HR spread across ROIs {r.cross_region_hr_spread_bpm:.2f} bpm")
    print(f"  phase dispersion      {r.phase_dispersion:.3f} rad")
    print("  --- output ---")
    print(f"  manipulation score    {r.rppg_manipulation_score:.3f}   "
          "(0=authentic, 1=manipulated, 0.5=neutral)")
    print(f"  degraded_reason       {r.degraded_reason}")
    print(f"  processing time       {r.processing_time_ms} ms")
    print("=" * 62)

    if r.rppg_quality < 0.30:
        print("\nQuality gate FIRED: score forced neutral (0.5).")
        print("Pulse SNR is too low for cross-region disagreement to mean anything.")
        print("This is the designed behaviour, not a failure - see plan §3.4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
