"""
spatial_detector.py
--------------------
Sub-component 1 of Feature 1 (Visual Consistency Analyzer): Spatial Pixel Inspection.

Scans individual face crops for AI-generation artifacts:
  - GAN/diffusion pixel patterns (checkerboard upsampling artifacts, spectral irregularities)
  - Boundary blurring along jawlines / skin-to-background seams (common face-swap tell)

Primary path: a fine-tuned CNN (EfficientNet-B0 by default; swap in a ViT via
`timm` if you prefer) that outputs a single authenticity logit per face crop.

IMPORTANT — about the CNN weights:
This ships with an ImageNet-pretrained EfficientNet backbone and a *freshly
initialized* classification head. That head has NOT been fine-tuned on any
deepfake dataset, so out of the box its predictions are not meaningful.
To make the CNN path actually detect deepfakes you must fine-tune it on a
labeled dataset such as FaceForensics++, Celeb-DF, or DFDC, then load the
resulting weights with `SpatialArtifactDetector.load_weights(path)`.

Until fine-tuned weights are supplied, this module automatically falls back
to a lightweight, dependency-free heuristic scorer (frequency-spectrum +
jawline-blur analysis) so the pipeline is runnable end-to-end without
requiring you to train anything first. This heuristic is a reasonable
placeholder, not a production-grade detector — treat its scores as weak
signal until the CNN is properly fine-tuned.

torch/torchvision are optional. If they aren't installed, the CNN path is
skipped automatically and a warning is logged once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

logger = logging.getLogger("visual_consistency_analyzer.spatial")

try:
    import torch
    import torch.nn as nn
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    _TORCH_AVAILABLE = False


@dataclass
class SpatialResult:
    score: float                 # 0.0 (looks synthetic) .. 1.0 (looks authentic)
    method: str                  # "cnn" or "heuristic"
    detail: dict


class SpatialArtifactDetector:
    """
    Wraps either a fine-tuned CNN or a heuristic fallback behind one interface.

    Usage:
        detector = SpatialArtifactDetector()
        detector.load_weights("finetuned_deepfake_head.pt")  # optional
        result = detector.predict(face_crop_bgr)
    """

    def __init__(self, device: str = "cpu", input_size: int = 224):
        self.input_size = input_size
        self.device = device
        self.has_finetuned_weights = False
        self._model = None

        if _TORCH_AVAILABLE:
            self._model = self._build_model()
        else:
            logger.warning(
                "torch/torchvision not installed — SpatialArtifactDetector will "
                "use the heuristic fallback scorer only. Install torch and "
                "torchvision, then call load_weights(), to enable the CNN path."
            )

    def _build_model(self):
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[1].in_features
        # Single-logit head: sigmoid(logit) = P(authentic)
        backbone.classifier[1] = nn.Linear(in_features, 1)
        backbone.eval()
        backbone.to(self.device)
        return backbone

    def load_weights(self, checkpoint_path: str) -> None:
        """Load a fine-tuned state_dict (trained on FF++ / DFDC / Celeb-DF etc.)."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is not installed; cannot load CNN weights.")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self.has_finetuned_weights = True
        logger.info("Loaded fine-tuned spatial detector weights from %s", checkpoint_path)

    # ------------------------------------------------------------------ #
    # CNN path
    # ------------------------------------------------------------------ #
    def _predict_cnn(self, face_bgr: np.ndarray) -> SpatialResult:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self._model(tensor)
            prob_authentic = torch.sigmoid(logit).item()

        return SpatialResult(
            score=float(prob_authentic),
            method="cnn",
            detail={"finetuned": self.has_finetuned_weights},
        )

    # ------------------------------------------------------------------ #
    # Heuristic fallback path
    # ------------------------------------------------------------------ #
    @staticmethod
    def _jawline_blur_score(face_bgr: np.ndarray) -> float:
        """
        Face-swap boundaries often show a soft blending seam along the
        jaw/cheek perimeter that the interior of the face doesn't have.
        We compare edge sharpness (Laplacian variance) in an outer ring
        vs. the face interior — a real face has fairly consistent sharpness;
        a blended seam shows a sharpness *drop* at the boundary.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 20 or w < 20:
            return 0.5

        interior = gray[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        outer_mask = np.ones_like(gray, dtype=bool)
        outer_mask[int(h * 0.12):int(h * 0.88), int(w * 0.12):int(w * 0.88)] = False
        outer_ring = gray[outer_mask]

        interior_sharpness = cv2.Laplacian(interior, cv2.CV_64F).var()
        outer_sharpness = cv2.Laplacian(
            outer_ring.reshape(-1, 1).astype(np.uint8), cv2.CV_64F
        ).var()

        if interior_sharpness < 1e-6:
            return 0.5

        ratio = outer_sharpness / interior_sharpness
        # ratio near/above 1 -> consistent sharpness -> authentic-leaning
        # ratio much below 1 -> blurred seam -> synthetic-leaning
        return float(np.clip(ratio, 0.0, 1.5) / 1.5)

    @staticmethod
    def _spectral_artifact_score(face_bgr: np.ndarray) -> float:
        """
        GAN/diffusion upsampling frequently leaves periodic high-frequency
        spectral energy (checkerboard-style artifacts) that isn't present
        in natural camera sensor noise. We take the FFT magnitude spectrum
        and measure how much energy sits in the high-frequency band
        relative to total energy. Real photos have a fairly smooth
        1/f falloff; synthetic images often show anomalous high-freq bumps.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (128, 128))
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        cy, cx = 64, 64
        y, x = np.ogrid[:128, :128]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        low_band = magnitude[dist <= 20].sum()
        high_band = magnitude[dist > 45].sum()
        total = magnitude.sum() + 1e-8

        high_ratio = high_band / total
        # empirically natural photos: ~0.02-0.08 high-freq ratio at this scale.
        # Score decays as ratio departs from that natural range.
        natural_center = 0.05
        deviation = abs(high_ratio - natural_center)
        score = float(np.clip(1.0 - deviation * 6.0, 0.0, 1.0))
        return score

    def _predict_heuristic(self, face_bgr: np.ndarray) -> SpatialResult:
        blur_score = self._jawline_blur_score(face_bgr)
        spectral_score = self._spectral_artifact_score(face_bgr)
        combined = 0.5 * blur_score + 0.5 * spectral_score
        return SpatialResult(
            score=combined,
            method="heuristic",
            detail={"jawline_blur_score": blur_score, "spectral_score": spectral_score},
        )

    # ------------------------------------------------------------------ #
    def predict(self, face_bgr: np.ndarray) -> SpatialResult:
        """
        face_bgr: cropped face region, BGR (OpenCV convention), any size >= ~40x40.
        Returns a SpatialResult with score in [0, 1], 1 = looks authentic.
        """
        if face_bgr is None or face_bgr.size == 0:
            return SpatialResult(score=0.5, method="none", detail={"reason": "empty_crop"})

        if _TORCH_AVAILABLE and self.has_finetuned_weights:
            return self._predict_cnn(face_bgr)

        return self._predict_heuristic(face_bgr)
