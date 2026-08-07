"""Run full DeepGuard multi-modal detection pipeline on all videos in TEST_VIDEOS."""

import os as _os
import sys as _sys

# Scripts live in scripts/ but resolve paths and imports against the
# REPO ROOT, so they behave identically no matter where they are invoked.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import glob
import os
import sys
import time

from src.pipeline.detect import detect


def main():
    video_files = sorted(glob.glob("TEST_VIDEOS/*.mp4"))
    if not video_files:
        print("No .mp4 files found in TEST_VIDEOS/")
        return

    print("=" * 125)
    print(f"DEEPGUARD MULTI-MODAL PIPELINE // BATCH RUN ON {len(video_files)} TEST VIDEOS")
    print("=" * 125)
    header = (
        f"{'VIDEO FILE':<14} | {'VERDICT':<18} | {'CONFIDENCE':<18} | {'% REAL':<9} | {'% FAKE':<9} | "
        f"{'rPPG (Q)':<12} | {'LipSync (Q)':<12} | {'Visual (Q)':<12} | {'TIME':<7}"
    )
    print(header)
    print("-" * 125)

    summary_stats = {"LIKELY_MANIPULATED": 0, "LIKELY_AUTHENTIC": 0, "UNCERTAIN": 0, "ERROR": 0}

    for v in video_files:
        t0 = time.time()
        v_name = os.path.basename(v)
        try:
            fused, r, l, vis = detect(v, session_id="batch_eval")
            elapsed = time.time() - t0

            p_fake = fused.manipulation_probability
            p_real = 1.0 - p_fake
            pct_real = f"{p_real * 100:.1f}%"
            pct_fake = f"{p_fake * 100:.1f}%"

            if p_real >= p_fake:
                conf_str = f"{pct_real} Real"
            else:
                conf_str = f"{pct_fake} Fake"

            r_str = f"{r.rppg_manipulation_score:.2f} ({r.rppg_quality:.2f})"
            l_str = f"{l.lipsync_manipulation_score:.2f} ({l.lipsync_quality:.2f})"
            vis_str = f"{vis.visual_manipulation_score:.2f} ({vis.visual_quality:.2f})"

            verdict_val = fused.verdict.value
            summary_stats[verdict_val] = summary_stats.get(verdict_val, 0) + 1

            row = (
                f"{v_name:<14} | {verdict_val:<18} | {conf_str:<18} | {pct_real:<9} | {pct_fake:<9} | "
                f"{r_str:<12} | {l_str:<12} | {vis_str:<12} | {elapsed:<5.2f}s"
            )
            print(row)
        except Exception as e:
            elapsed = time.time() - t0
            summary_stats["ERROR"] += 1
            print(f"{v_name:<14} | {'ERROR':<18} | {'N/A':<18} | {'N/A':<9} | {'N/A':<9} | {str(e):<38} | {elapsed:<5.2f}s")

    print("=" * 125)
    print("SUMMARY BY VERDICT:")
    for k, v in summary_stats.items():
        print(f"  {k:<20}: {v}")
    print("=" * 125)


if __name__ == "__main__":
    main()
