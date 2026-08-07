"""Fifth channel — fully synthetic (AI-generated) imagery.

    from src.visual.aigen import analyze
    analyze(clip.frames) -> AIGenResult

WHY THIS IS A SEPARATE CHANNEL AND NOT "ANOTHER DEEPFAKE MODEL"
`effb7` is a FACE-MANIPULATION detector: it is trained on DFDC, it consumes a
face crop, and it asks "was this face altered?". It has nothing to say about a
frame that was generated whole-cloth by a diffusion model, because there is no
authentic face underneath to have been altered.

This channel asks the different question — "was this IMAGE synthesised?" — and so
it differs in two deliberate ways:

  * it consumes the FULL FRAME, not the face crop. Generation artefacts live in
    background texture, lighting coherence and high-frequency statistics across
    the whole image, and cropping to a face throws most of that away.
  * it does not need a face at all. Frames where face detection failed are still
    scored, which is exactly the footage the other four channels abstain on.

MODEL
`Organika/sdxl-detector` — SwinV2 (86.7M params), fine-tuned to separate
diffusion-generated images from photographs. Labels are {0: artificial,
1: human}, so P(synthetic) is softmax[..., 0]. Chosen over CLIP-based
UniversalFakeDetect because ViT-L/14 costs ~1.5 s/frame on this CPU and would
have blown the interactive latency budget on its own.

AGGREGATION
Median, not p90. p90 is right for face manipulation, where only SOME frames carry
a visible splice. Synthesis is a property of EVERY frame of a generated clip, so
the median is the robust estimator and p90 would just amplify per-frame noise
into a false positive on hard authentic footage.

READ THE ADMISSION NOTE IN `analyze` BEFORE GIVING THIS CHANNEL A VOTE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

MODEL_ID = "Organika/sdxl-detector"
MAX_FRAMES = 8
SYNTHETIC_INDEX = 0          # VERIFIED against config.id2label: {0: artificial, 1: human}


@dataclass
class AIGenResult:
    aigen_manipulation_score: float = 0.5   # held neutral: see analyze() docstring
    aigen_quality: float = 0.0
    raw_score: float | None = None          # unmapped model output, diagnostics only
    frames_scored: int = 0
    frame_scores: list = field(default_factory=list)
    score_spread: float = 0.0
    model_id: str = MODEL_ID
    processing_time_ms: int = 0
    degraded_reason: str | None = None


@lru_cache(maxsize=1)
def _load():
    """Processor + model, or (None, None) if unavailable. Never raises."""
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        net = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        net.eval()
        # Verify the label mapping rather than trusting the constant. A silently
        # flipped index would invert every score while still looking plausible.
        id2label = {int(k): str(v).lower() for k, v in net.config.id2label.items()}
        if "artificial" not in id2label.get(SYNTHETIC_INDEX, ""):
            return None, None
        return proc, net
    except Exception:  # noqa: BLE001
        return None, None


def available() -> bool:
    return _load()[0] is not None


def analyze(frames_bgr, max_frames: int = MAX_FRAMES) -> AIGenResult:
    """P(frame was synthesised), aggregated over the clip. Never raises.

    ADMISSION — REJECTED ON MEASUREMENT. This channel does not vote, and its
    raw output is deliberately NOT surfaced as a probability.

    Measured on TEST_VIDEOS (3 fake / 3 real face-swap clips):

        AUC 0.000  -- perfectly inverted
        every REAL clip scored P(synthetic) 0.998 - 1.000

    A model that calls authentic video "synthetic" with near-certainty is not
    suffering a corpus mismatch, it is failing outright. The cause was isolated
    by degrading a KNOWN REAL PHOTOGRAPH toward video quality:

        original 780x438                       P(synthetic) 0.0000
        resized  854x480   (no compression)    P(synthetic) 0.0000
        resized  854x480 + JPEG q60            P(synthetic) 0.6355
        resized 1282x720 + JPEG q30            P(synthetic) 0.3826

    Resizing alone changes nothing; COMPRESSION alone drives the score. The model
    was trained on pristine Wikimedia photographs against SDXL renders, so it has
    learned "compression artefacts => synthetic". Every H.264 frame we decode
    therefore saturates it.

    This is the same species of confound as the resolution artefact in
    `pixel_forensics` (all-features AUC 0.711 vs scale-free 0.609): a feature
    that separates the classes in the training corpus for a reason that has
    nothing to do with the thing we claim to detect.

    CONSEQUENCE: `aigen_manipulation_score` stays at a neutral 0.5. The raw model
    output is preserved in `raw_score` for diagnostics, but it is not presented
    as P(synthetic), because on compressed video that number means "this frame
    was compressed", and showing it next to the other channels would invite
    exactly the wrong reading.

    To make this channel votable it needs a detector trained WITH compression
    augmentation, validated on generated video (Sora/Runway/Pika) at realistic
    bitrates. Until then it is scaffolding, honestly labelled.
    """
    t0 = time.time()
    ms = lambda: int((time.time() - t0) * 1000)  # noqa: E731
    try:
        import torch
        from PIL import Image

        proc, net = _load()
        if net is None:
            return AIGenResult(degraded_reason="model_unavailable",
                               processing_time_ms=ms())

        n = len(frames_bgr)
        if n == 0:
            return AIGenResult(degraded_reason="no_frames", processing_time_ms=ms())

        idx = np.linspace(0, n - 1, min(max_frames, n)).astype(int)
        # Full frames, BGR -> RGB. No face crop: see module docstring.
        imgs = [Image.fromarray(frames_bgr[i][:, :, ::-1]) for i in idx]

        with torch.no_grad():
            out = net(**proc(images=imgs, return_tensors="pt"))
            p = torch.softmax(out.logits, dim=1)[:, SYNTHETIC_INDEX].numpy()

        p = np.asarray(p, dtype=np.float64)
        res = AIGenResult()
        res.frame_scores = [round(float(x), 4) for x in p]
        res.frames_scored = int(len(p))
        res.raw_score = round(float(np.median(p)), 4)
        # NOT surfaced as P(synthetic). On compressed video the raw output tracks
        # codec artefacts, not synthesis - measured, see docstring.
        res.aigen_manipulation_score = 0.5
        res.score_spread = round(float(p.max() - p.min()), 4)
        res.aigen_quality = 0.0
        res.degraded_reason = "compression_confound_auc_0.000_on_video"
        res.processing_time_ms = ms()
        return res
    except Exception as exc:  # noqa: BLE001
        return AIGenResult(degraded_reason=f"unhandled:{type(exc).__name__}",
                           processing_time_ms=ms())
