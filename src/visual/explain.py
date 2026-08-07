"""Region attribution — why the frame-by-frame CNN reached its score.

    explain.region_attribution(frames, boxes) -> RegionAttribution

WHY OCCLUSION AND NOT GRAD-CAM
Grad-CAM needs gradients w.r.t. an intermediate feature map. Our EfficientNet is
a **TorchScript** artefact — the module graph is frozen and there are no named
layers to hook, so Grad-CAM is not available without re-authoring the model.

Occlusion sensitivity needs only forward passes: mask a region, measure how much
P(manipulated) moves. It is model-agnostic, works on any black box, and the
number it produces is a *measured* score delta rather than a gradient
approximation. Cost is one extra forward per region — trivial next to the 16-32
frames already being scored.

THE DESIGN CHOICE THAT MAKES THIS USEFUL
Attribution uses the SAME face regions as `pixel_forensics` (mouth, eyes, nose,
forehead, cheeks) and the same face box as the rPPG coherence grid. So all three
channels answer "where?" on a shared coordinate system:

    CNN            -> which region, when masked, changes the verdict
    pixel forensics-> which region shows texture / warp anomalies
    rPPG           -> which patches fail to beat in phase with the face

That cross-channel agreement (or disagreement) on *location* is the explanation,
and it is grounded in three independent mechanisms rather than one saliency map.

NOT A CONFIDENCE MEASURE. A large delta means the region mattered to THIS score;
it does not mean the region is manipulated. Reported as attribution, never as
evidence of tampering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.visual.models import LOADERS, SPECS, preprocess
from src.visual.pixel_forensics import REGIONS

MAX_FRAMES = 4          # 4 baseline + 7x4 occluded = 32 forwards (latency budget)
OCCLUDE_VALUE = 0.0     # post-normalisation mean -> a neutral grey patch


@dataclass
class RegionAttribution:
    baseline: float = 0.5
    regions: dict = field(default_factory=dict)   # name -> score delta
    top_region: str | None = None
    frames_used: int = 0
    processing_time_ms: int = 0
    degraded_reason: str | None = None


def _score_batch(net, spec, tensors) -> float:
    """p90 of P(manipulated) across the batch — matches the scoring aggregation."""
    import torch

    with torch.no_grad():
        logits = net(torch.from_numpy(np.stack(tensors)))
        if spec.output == "sigmoid1":
            p = torch.sigmoid(logits).reshape(-1)
        else:
            p = torch.softmax(logits, dim=1)[:, 1]
    return float(np.percentile(p.numpy().astype(np.float64), 90))


def region_attribution(frames_bgr, boxes, model: str = "effb7",
                       max_frames: int = MAX_FRAMES) -> RegionAttribution:
    """Occlusion-based attribution over face regions. Never raises."""
    t0 = time.time()
    ms = lambda: int((time.time() - t0) * 1000)  # noqa: E731
    try:
        net = LOADERS[model]()
        spec = SPECS[model]
        if net is None:
            return RegionAttribution(degraded_reason="weights_unavailable",
                                     processing_time_ms=ms())

        n = len(frames_bgr)
        idx = [i for i in np.linspace(0, n - 1, min(max_frames, n)).astype(int)
               if boxes is not None and not np.isnan(np.asarray(boxes[i], float)).any()]
        base_tensors, keep = [], []
        for i in idx:
            t = preprocess(frames_bgr[i], boxes[i], spec, 1.3)
            if t is not None:
                base_tensors.append(t)
                keep.append(i)
        if len(base_tensors) < 2:
            return RegionAttribution(degraded_reason="too_few_usable_frames",
                                     processing_time_ms=ms())

        baseline = _score_batch(net, spec, base_tensors)

        # Map each region from FACE-BOX fractions into the resized tensor.
        # This must mirror `preprocess` exactly. Two details bite:
        #   1. preprocess CLAMPS the crop to the frame, so the crop is not a full
        #      `side` square when the face sits near an edge — the origin is
        #      max(cx-side/2, 0), not cx-side/2.
        #   2. the clamped crop is then resized to a square, so x and y are
        #      rescaled by DIFFERENT factors. One shared factor mislocates every
        #      region on any edge-cropped face.
        S = spec.input_size
        H, W = frames_bgr[0].shape[:2]
        rects: dict[str, list] = {}
        for i in keep:
            x, y, w, h = [float(v) for v in boxes[i]]
            side = max(w, h) * 1.3
            cx, cy = x + w / 2.0, y + h / 2.0
            x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
            x1, y1 = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
            sx, sy = S / max(x1 - x0, 1), S / max(y1 - y0, 1)
            for name, (fx0, fy0, fx1, fy1) in REGIONS.items():
                rx0 = int(np.clip(((x + fx0 * w) - x0) * sx, 0, S - 1))
                rx1 = int(np.clip(((x + fx1 * w) - x0) * sx, 1, S))
                ry0 = int(np.clip(((y + fy0 * h) - y0) * sy, 0, S - 1))
                ry1 = int(np.clip(((y + fy1 * h) - y0) * sy, 1, S))
                rects.setdefault(name, []).append((rx0, ry0, rx1, ry1))

        deltas: dict[str, float] = {}
        for name, boxes_px in rects.items():
            occluded = []
            for t, (rx0, ry0, rx1, ry1) in zip(base_tensors, boxes_px):
                if rx1 - rx0 < 4 or ry1 - ry0 < 4:
                    continue
                m = t.copy()
                m[:, ry0:ry1, rx0:rx1] = OCCLUDE_VALUE
                occluded.append(m)
            if len(occluded) < 2:
                continue
            deltas[name] = round(baseline - _score_batch(net, spec, occluded), 4)

        if not deltas:
            return RegionAttribution(baseline=round(baseline, 4),
                                     degraded_reason="no_region_measurable",
                                     processing_time_ms=ms())

        res = RegionAttribution()
        res.baseline = round(baseline, 4)
        res.regions = deltas
        # Largest positive delta: masking it DROPPED the manipulation score most,
        # i.e. that region contributed most of the evidence.
        res.top_region = max(deltas, key=lambda k: deltas[k])
        res.frames_used = len(base_tensors)
        res.processing_time_ms = ms()
        return res
    except Exception as exc:  # noqa: BLE001
        return RegionAttribution(degraded_reason=f"unhandled:{type(exc).__name__}",
                                 processing_time_ms=ms())
