# Feature 1 — Visual Consistency Analyzer

Spatial & Behavioral Artifact Detection for deepfake/synthetic-video screening.

## What it does

Three independent sub-detectors run over every analyzed frame, then get fused
into one score:

| Module | File | Signal |
|---|---|---|
| Spatial Pixel Inspection | `spatial_detector.py` | Per-frame CNN (or heuristic fallback) score for GAN/diffusion pixel artifacts and jawline blending seams |
| Blink & Behavioral Monitoring | `blink_monitor.py` | Eye Aspect Ratio (EAR) over time → blink count, blink rate, frozen-eye detection |
| Motion Jitter & Warping Check | `motion_jitter.py` | Frame-to-frame jerk of nose-tip / eye-corner landmarks → face-swap warping detection |

`visual_analyzer.py` orchestrates all three against a video file and returns:

```json
{
  "visual_score": 0.83,
  "spatial_cnn_score": 0.79,
  "blinks_detected": 14,
  "frames_analyzed": 450,
  "diagnostics": {
    "spatial_method": "heuristic",
    "face_detection_rate": 0.97,
    "frames_with_face": 437,
    "blink_rate_per_min": 18.6,
    "frozen_eyes": false,
    "ear_mean": 0.29,
    "ear_std": 0.041,
    "jitter_score": 0.88,
    "mean_jerk": 0.0031,
    "jerk_std": 0.0012,
    "fusion_weights": {"spatial": 0.5, "behavioral": 0.25, "jitter": 0.25}
  }
}
```

`visual_score` (and `spatial_cnn_score` / `blinks_detected` / `frames_analyzed`)
match the schema in the spec — that's what you hand to the Fusion Engine
alongside the Lip-Sync and rPPG scores.

## Install

```bash
pip install -r requirements.txt
```

torch/torchvision are optional — omit them if you just want the heuristic
spatial scorer (see below).

## Run

```bash
python demo.py path/to/video.mp4
python demo.py path/to/video.mp4 --stride 2 --w-spatial 0.6 --w-behavioral 0.2 --w-jitter 0.2
```

## Using it as a library

```python
from visual_analyzer import VisualConsistencyAnalyzer, FusionWeights

with VisualConsistencyAnalyzer(
    spatial_weights_path="finetuned_head.pt",   # optional
    fusion_weights=FusionWeights(spatial=0.5, behavioral=0.25, jitter=0.25),
) as analyzer:
    visual_output = analyzer.analyze("video.mp4")

# hand off to your fusion engine alongside Feature 2 / Feature 3 scores
final_verdict = fusion_engine.combine(
    visual=visual_output["visual_score"],
    lip_sync=lip_sync_score,
    rppg=rppg_score,
)
```

## ⚠️ Things you need to calibrate before trusting this in production

This module is a working, tested pipeline — but three pieces are placeholders
that need real data behind them:

1. **Spatial CNN head is untrained.** `spatial_detector.py` builds an
   ImageNet-pretrained EfficientNet-B0 with a fresh classification head. It
   will not detect deepfakes until you fine-tune it on a labeled dataset
   (FaceForensics++, Celeb-DF, DFDC) and load the weights via
   `SpatialArtifactDetector.load_weights(path)`. Until then, the pipeline
   automatically uses a heuristic fallback (FFT spectral-artifact score +
   jawline blur-consistency score) — real signal, but weaker than a properly
   fine-tuned CNN.
2. **Blink-rate "natural range" (8–30/min) and jitter thresholds
   (`natural_ceiling = 0.012`)** in `blink_monitor.py` / `motion_jitter.py`
   are reasonable literature-backed starting points, not values fitted to
   your data. Run this over a batch of known-real and known-fake videos and
   adjust these constants (or replace the hand-written scoring functions
   with a small trained classifier over the raw features) before relying on
   the outputs.
3. **Fusion weights** (`spatial=0.5, behavioral=0.25, jitter=0.25`) are a
   sensible default, not a tuned result. Once you have labeled videos
   flowing through Features 1–3, consider learning the fusion weights
   (e.g. logistic regression over the three sub-scores) rather than hand-setting them.

## MediaPipe version note

Uses the modern Tasks API (`mp.tasks.vision.FaceLandmarker`), required
because current mediapipe releases (which is what you'll get on Python
3.13+ — the legacy line only ships wheels up through Python 3.12) dropped
the old `mp.solutions.face_mesh` API entirely. The Tasks API needs a small
`.task` model bundle that `visual_analyzer.py` downloads automatically on
first run and caches under `~/.cache/visual_consistency_analyzer/`. If your
network blocks `storage.googleapis.com`, download
`face_landmarker.task` manually from the URL in the error message and pass
its path via `VisualConsistencyAnalyzer(face_landmarker_model_path="...")`.

## Files

```
spatial_detector.py   # Sub-component 1: CNN + heuristic pixel-artifact scorer
blink_monitor.py       # Sub-component 2: EAR-based blink/behavioral scorer
motion_jitter.py        # Sub-component 3: landmark jerk/warping scorer
visual_analyzer.py     # Orchestrator — this is Feature 1's public entry point
demo.py                 # CLI runner
requirements.txt
```
