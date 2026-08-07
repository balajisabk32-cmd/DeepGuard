"""
verdict.py
----------
Converts the raw Visual Output (0.0-1.0 scores) into a human-readable
"percent chance real" verdict with a breakdown by sub-signal.

This is presentation only — it does not change the underlying scoring
logic in visual_analyzer.py, spatial_detector.py, blink_monitor.py, or
motion_jitter.py.

⚠️ Calibration reminder: the percentages below are only as trustworthy as
the underlying sub-scores. Until the spatial CNN is fine-tuned and the
blink-rate / jitter thresholds are calibrated against your own labeled
real/fake videos (see README.md), treat "percent real" as a rough, noisy
signal — not a certified result.
"""

from typing import Dict


# (score_threshold, label) — first threshold the score meets, top to bottom, wins.
VERDICT_THRESHOLDS = [
    (0.70, "Likely REAL"),
    (0.40, "Uncertain — needs review"),
    (0.00, "Likely SYNTHETIC"),
]


def to_percent_verdict(visual_output: Dict) -> Dict:
    """
    visual_output: the dict returned by VisualConsistencyAnalyzer.analyze().
    Returns a dict with percent-scale fields ready to print or log.
    """
    score = visual_output["visual_score"]
    diagnostics = visual_output.get("diagnostics", {})

    label = VERDICT_THRESHOLDS[-1][1]
    for threshold, name in VERDICT_THRESHOLDS:
        if score >= threshold:
            label = name
            break

    return {
        "percent_real": round(score * 100, 1),
        "verdict": label,
        "breakdown": {
            "spatial_pct": round(visual_output["spatial_cnn_score"] * 100, 1),
            "behavioral_pct": round(diagnostics.get("behavioral_score", 0.0) * 100, 1),
            "jitter_pct": round(diagnostics.get("jitter_score", 0.0) * 100, 1),
        },
        "spatial_method": diagnostics.get("spatial_method", "unknown"),
        "frames_analyzed": visual_output.get("frames_analyzed", 0),
        "face_detection_rate_pct": round(diagnostics.get("face_detection_rate", 0.0) * 100, 1),
    }


def print_verdict(visual_output: Dict) -> None:
    v = to_percent_verdict(visual_output)
    print(f"\n{'='*44}")
    print(f"  {v['percent_real']}% chance REAL   —   {v['verdict']}")
    print(f"{'='*44}")
    print(f"  Spatial (pixel artifacts):   {v['breakdown']['spatial_pct']}%  "
          f"[{v['spatial_method']}]")
    print(f"  Behavioral (blinks/eyes):    {v['breakdown']['behavioral_pct']}%")
    print(f"  Motion (jitter/warping):     {v['breakdown']['jitter_pct']}%")
    print(f"  Frames analyzed: {v['frames_analyzed']}  "
          f"(face detected in {v['face_detection_rate_pct']}%)")
    print(f"{'='*44}\n")
