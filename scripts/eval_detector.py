"""Evaluation script for HighAccuracyLipSyncDetector on repository video files."""

import os as _os
import sys as _sys

# Scripts live in scripts/ but resolve paths and imports against the
# REPO ROOT, so they behave identically no matter where they are invoked.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import os
from src.lipsync.detector import HighAccuracyLipSyncDetector, evaluate_video


def main():
    model = HighAccuracyLipSyncDetector(pretrained=False)
    videos = [
        "REAL.mp4",
        "coh_REAL.mp4",
        "coh_Deep.mp4",
        "Deepfake tom cruise magic trick #shorts #deepfakes #tomcruise.mp4",
        "WIN_20260807_14_02_13_Pro.mp4",
    ]

    print("=" * 80)
    print("HIGH-ACCURACY DUAL-BRANCH LIP-SYNC DETECTOR EVALUATION RESULTS")
    print("=" * 80)

    for v in videos:
        if not os.path.exists(v):
            continue
        res = evaluate_video(model, v, device="cpu", max_sec=5.0)
        print(f"Video: {res['video']}")
        print(f"  Logit                     : {res['logit']:+.4f}")
        print(f"  P(Manipulated)            : {res['prob_manipulated']:.4f}")
        print(f"  Cross-Modal Mismatch Mean : {res['mismatch_mean']:.4f}")
        print(f"  Cross-Modal Mismatch Max  : {res['mismatch_max']:.4f}")
        print(f"  Frames Scored             : {res['frames_scored']}")
        print(f"  Processing Time           : {res['elapsed_sec']:.2f}s")
        print("-" * 80)


if __name__ == "__main__":
    main()
