"""Detector registry — one entry per trained visual model.

Each detector owns its OWN preprocessing. This is not a style choice: the two
models disagree on every preprocessing parameter, and using one model's constants
with the other silently degrades it instead of failing.

    Xception (FakeAVCeleb)   224x224   mean (0.4489,0.3352,0.3106) std (0.2380,0.1965,0.1962)
    Capsule  (FF++)          300x300   mean (0.485,0.456,0.406)    std (0.229,0.224,0.225)

WHY TWO MODELS
They are trained on DIFFERENT datasets — FakeAVCeleb and FaceForensics++. Two
independently-trained detectors agreeing is worth much more than either alone at
its own accuracy, and where they DISAGREE is a genuine uncertainty signal rather
than something to average away. Agreement is reported, not hidden.

CLASS ORDER: index 1 = fake for both. Verified per model, not assumed:
  Xception -> FF++ classification/detect_from_video.py:195
              `label = 'fake' if prediction == 1 else 'real'`
  Capsule  -> Eval_Capsule-Forensics-v2.py:106,120 scores pred_prob[:,1] with
              pos_label=1; training collapses all manipulation classes to 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO / "models"

FAKE_INDEX = 1


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    weights: Path
    input_size: int
    mean: tuple
    std: tuple
    dataset: str          # what it was trained on — the basis for ensembling
    note: str = ""
    # "softmax2": 2 logits, P(fake) = softmax(...)[:, 1]
    # "sigmoid1": 1 logit,  P(fake) = sigmoid(...)
    # Getting this wrong does not crash — it silently produces a plausible but
    # meaningless number, so it is declared per model rather than inferred.
    output: str = "softmax2"
    # Set False to keep a model loadable and inspectable but OUT of the fused
    # score. Used when a model is measurably not discriminating.
    enabled: bool = True


SPECS = {
    "xception": DetectorSpec(
        name="xception",
        weights=MODELS_DIR / "xception_fakeavceleb_video.pt",
        input_size=224,
        mean=(0.4489, 0.3352, 0.3106),
        std=(0.2380, 0.1965, 0.1962),
        dataset="FakeAVCeleb",
        note="DISABLED for fusion: measured AUC 0.222 on the 9-clip TEST_VIDEOS set "
             "(3 fake / 6 real) versus effb7's 0.833 on the SAME clips — below chance, "
             "i.e. anti-correlated with truth here. Averaging the two gave 0.222: the "
             "weaker member destroyed the stronger one outright. Equal-weight ensembling "
             "is only safe between comparably good models. Still loaded and reported.",
        enabled=False,
    ),
    "capsule": DetectorSpec(
        name="capsule",
        weights=MODELS_DIR / "capsule_ffpp.pt",
        input_size=300,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        dataset="FaceForensics++",
        note="DISABLED: emits a near-constant. Measured medians 0.608/0.609/0.609/0.609 "
             "across four different clips, and 0.562 on random noise. Loads cleanly "
             "(0 missing keys) so the failure is silent — most likely the checkpoint "
             "expects a different vgg19 feature slice than torchvision 0.26 produces. "
             "Still loaded and REPORTED for transparency; excluded from the fused score.",
        enabled=False,
    ),
    "effb7": DetectorSpec(
        name="effb7",
        weights=MODELS_DIR / "deepfake_efficientnet_b7.pt",
        input_size=380,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        dataset="DFDC",
        note="Seferbekov DFDC 1st-place solution via facetorch; Apache-2.0; "
             "TorchScript, single logit -> sigmoid = P(fake); labels [Real, Fake]",
        output="sigmoid1",
    ),
}


def _tune_threads() -> None:
    """Use every core for inference.

    torch defaults to cpu_count-2 (10 of 12 here), which leaves measurable time
    on the table for a latency-bound interactive path. Measured on effb7 @380px,
    batch of 8, while another job was competing for CPU:
        10 threads -> 0.99 s/img
        12 threads -> 0.66 s/img
    Set once at load; harmless if the process already set it.
    """
    import os

    import torch

    try:
        torch.set_num_threads(os.cpu_count() or torch.get_num_threads())
    except Exception:  # noqa: BLE001 - a perf hint must never break inference
        pass


def preprocess(frame_bgr: np.ndarray, box, spec: DetectorSpec,
               crop_margin: float = 1.3) -> np.ndarray | None:
    """Square margin-expanded face crop -> normalised CHW float32 for `spec`."""
    if box is None or np.isnan(np.asarray(box, dtype=float)).any():
        return None
    x, y, w, h = [float(v) for v in box]
    H, W = frame_bgr.shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * crop_margin
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x1, y1 = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (spec.input_size, spec.input_size), interpolation=cv2.INTER_AREA)
    arr = (rgb.astype(np.float32) / 255.0 - np.array(spec.mean, dtype=np.float32)) \
        / np.array(spec.std, dtype=np.float32)
    return arr.transpose(2, 0, 1)


@lru_cache(maxsize=4)
def load_xception():
    _tune_threads()
    spec = SPECS["xception"]
    if not spec.weights.exists():
        return None
    try:
        import torch

        from src.visual.xception import xception

        net = xception(num_classes=2, pretrained=False)
        ckpt = torch.load(spec.weights, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt.get("state_dict", ckpt))
        net.eval()
        return net
    except Exception:
        return None


@lru_cache(maxsize=4)
def load_capsule():
    """Capsule-Forensics-v2. Returns a callable(batch)->logits, or None.

    The checkpoint holds ONLY the capsule head; its VggExtractor is built from
    torchvision's ImageNet vgg19, which must already be cached. Without that the
    detector is unavailable — and unavailable is the correct outcome, because a
    randomly-initialised extractor would produce confident nonsense.
    """
    _tune_threads()
    spec = SPECS["capsule"]
    if not spec.weights.exists():
        return None
    try:
        import torch

        from src.visual.capsule_net import CapsuleNet, VggExtractor

        vgg = VggExtractor()
        caps = CapsuleNet(num_class=2, gpu_id=-1)
        caps.load_state_dict(torch.load(spec.weights, map_location="cpu",
                                        weights_only=False))
        caps.eval()
        vgg.eval()

        def run(batch):
            with torch.no_grad():
                _classes, class_ = caps(vgg(batch), random=False)
            return class_

        return run
    except Exception:
        return None


@lru_cache(maxsize=4)
def load_effb7():
    """TorchScript — self-contained, no architecture code required."""
    _tune_threads()
    spec = SPECS["effb7"]
    if not spec.weights.exists():
        return None
    try:
        import torch

        net = torch.jit.load(str(spec.weights), map_location="cpu")
        net.eval()
        return net
    except Exception:
        return None


LOADERS: dict[str, Callable] = {
    "xception": load_xception,
    "capsule": load_capsule,
    "effb7": load_effb7,
}


def available(include_disabled: bool = False) -> list[str]:
    """Detectors whose weights load AND which are enabled for fusion."""
    return [n for n, fn in LOADERS.items()
            if fn() is not None and (include_disabled or SPECS[n].enabled)]
