"""
visual_analyzer.py
-------------------
Feature 1: Visual Consistency Analyzer — Spatial & Behavioral Artifact Detection.

Orchestrates the three sub-components against a video file:
  1. SpatialArtifactDetector  — per-frame CNN/heuristic pixel-artifact scan
  2. BlinkBehavioralMonitor   — EAR-based blink & frozen-eye analysis
  3. MotionJitterDetector     — landmark jerk / warping analysis

and fuses them into the single Visual Authenticity Score that gets handed
to the system's Fusion Engine alongside the Lip-Sync (Feature 2) and rPPG
(Feature 3) scores.

Output schema matches the spec:
    {
        "visual_score": float,          # 0.0 - 1.0
        "spatial_cnn_score": float,     # 0.0 - 1.0
        "blinks_detected": int,
        "frames_analyzed": int,
        ... plus extra diagnostic metadata for debugging/audit trails
    }
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from spatial_detector import SpatialArtifactDetector
from blink_monitor import BlinkBehavioralMonitor
from motion_jitter import MotionJitterDetector

logger = logging.getLogger("visual_consistency_analyzer")
logging.basicConfig(level=logging.INFO)

# Official Google-hosted Face Landmarker model bundle (Tasks API). This is a
# one-time ~4 MB download, cached locally after the first run. If your
# environment can't reach this host, download the file manually from the
# same URL and pass its local path via `face_landmarker_model_path=`.
_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "visual_consistency_analyzer" / "face_landmarker.task"


def _ensure_face_landmarker_model(model_path: Optional[str]) -> str:
    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"face_landmarker model not found at: {model_path}")
        return model_path

    if _DEFAULT_MODEL_CACHE.is_file():
        return str(_DEFAULT_MODEL_CACHE)

    _DEFAULT_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading face_landmarker model bundle to %s ...", _DEFAULT_MODEL_CACHE)
    try:
        urllib.request.urlretrieve(_DEFAULT_MODEL_URL, _DEFAULT_MODEL_CACHE)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the MediaPipe face_landmarker model bundle "
            f"from {_DEFAULT_MODEL_URL} ({exc}). If your network blocks "
            "storage.googleapis.com, download the file manually from that "
            "URL and pass its path via VisualConsistencyAnalyzer("
            "face_landmarker_model_path='...')."
        ) from exc

    return str(_DEFAULT_MODEL_CACHE)


@dataclass
class FusionWeights:
    spatial: float = 0.5
    behavioral: float = 0.25
    jitter: float = 0.25

    def normalized(self) -> "FusionWeights":
        total = self.spatial + self.behavioral + self.jitter
        return FusionWeights(self.spatial / total, self.behavioral / total, self.jitter / total)


class VisualConsistencyAnalyzer:
    def __init__(
        self,
        spatial_weights_path: Optional[str] = None,
        fusion_weights: Optional[FusionWeights] = None,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        face_landmarker_model_path: Optional[str] = None,
    ):
        """
        spatial_weights_path: optional path to fine-tuned CNN weights (see
            spatial_detector.py). If omitted, the heuristic scorer is used.
        frame_stride: process every Nth frame (>1 speeds up long videos).
        max_frames: cap on number of frames analyzed (None = whole video).
        face_landmarker_model_path: optional local path to a pre-downloaded
            face_landmarker.task bundle. If omitted, it's downloaded once
            and cached under ~/.cache/visual_consistency_analyzer/.
        """
        self.fusion_weights = (fusion_weights or FusionWeights()).normalized()
        self.frame_stride = max(1, frame_stride)
        self.max_frames = max_frames

        self.spatial_detector = SpatialArtifactDetector()
        if spatial_weights_path:
            self.spatial_detector.load_weights(spatial_weights_path)

        model_path = _ensure_face_landmarker_model(face_landmarker_model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _landmarks_to_pixel_xy(face_landmarks, frame_w: int, frame_h: int) -> np.ndarray:
        """face_landmarks: a plain list of NormalizedLandmark (Tasks API face_landmarks[i])."""
        pts = np.array(
            [[lm.x * frame_w, lm.y * frame_h] for lm in face_landmarks],
            dtype=np.float32,
        )
        return pts

    @staticmethod
    def _bbox_from_landmarks(landmarks_xy: np.ndarray, frame_w: int, frame_h: int, pad: float = 0.25):
        x_min, y_min = landmarks_xy.min(axis=0)
        x_max, y_max = landmarks_xy.max(axis=0)
        w, h = x_max - x_min, y_max - y_min
        x_min -= w * pad
        x_max += w * pad
        y_min -= h * pad
        y_max += h * pad
        x_min = int(max(0, x_min))
        y_min = int(max(0, y_min))
        x_max = int(min(frame_w, x_max))
        y_max = int(min(frame_h, y_max))
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _face_scale_px(landmarks_xy: np.ndarray) -> float:
        """Inter-ocular distance, used to normalize jitter thresholds."""
        left_eye_outer = landmarks_xy[362]
        right_eye_outer = landmarks_xy[33]
        return float(np.linalg.norm(left_eye_outer - right_eye_outer))

    # ------------------------------------------------------------------ #
    def analyze(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        blink_monitor = BlinkBehavioralMonitor()
        jitter_detector = MotionJitterDetector()

        spatial_scores = []
        frames_analyzed = 0
        frames_with_face = 0
        face_scales = []
        frame_idx = 0
        last_timestamp_ms = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % self.frame_stride != 0:
                frame_idx += 1
                continue
            if self.max_frames is not None and frames_analyzed >= self.max_frames:
                break

            # Tasks API VIDEO mode requires strictly increasing timestamps.
            timestamp_ms = int(frame_idx * (1000.0 / fps))
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            frame_idx += 1
            frames_analyzed += 1
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.face_landmarks:
                blink_monitor.update(None)
                jitter_detector.update(None)
                continue

            frames_with_face += 1
            landmarks_xy = self._landmarks_to_pixel_xy(result.face_landmarks[0], w, h)
            face_scales.append(self._face_scale_px(landmarks_xy))

            # 1. Spatial CNN/heuristic pass on the cropped face
            x1, y1, x2, y2 = self._bbox_from_landmarks(landmarks_xy, w, h)
            face_crop = frame[y1:y2, x1:x2]
            spatial_result = self.spatial_detector.predict(face_crop)
            spatial_scores.append(spatial_result.score)

            # 2. Blink / behavioral tracking
            blink_monitor.update(landmarks_xy)

            # 3. Motion jitter tracking
            jitter_detector.update(landmarks_xy)

        cap.release()

        avg_face_scale = float(np.mean(face_scales)) if face_scales else 150.0
        blink_result = blink_monitor.finalize(fps=fps, total_frames_analyzed=max(frames_analyzed, 1))
        jitter_result = jitter_detector.finalize(face_scale_px=avg_face_scale)

        spatial_cnn_score = float(np.mean(spatial_scores)) if spatial_scores else 0.5
        face_detection_rate = frames_with_face / frames_analyzed if frames_analyzed else 0.0

        visual_score = self._fuse(spatial_cnn_score, blink_result.behavioral_score, jitter_result.jitter_score)

        return {
            # --- core spec schema ---
            "visual_score": round(visual_score, 4),
            "spatial_cnn_score": round(spatial_cnn_score, 4),
            "blinks_detected": blink_result.blinks_detected,
            "frames_analyzed": frames_analyzed,
            # --- diagnostic metadata ---
            "diagnostics": {
                "spatial_method": self.spatial_detector.has_finetuned_weights and "cnn" or "heuristic",
                "face_detection_rate": round(face_detection_rate, 4),
                "frames_with_face": frames_with_face,
                "blink_rate_per_min": blink_result.blink_rate_per_min,
                "frozen_eyes": blink_result.frozen_eyes,
                "behavioral_score": round(blink_result.behavioral_score, 4),
                "ear_mean": blink_result.ear_mean,
                "ear_std": blink_result.ear_std,
                "jitter_score": round(jitter_result.jitter_score, 4),
                "mean_jerk": jitter_result.mean_jerk,
                "jerk_std": jitter_result.jerk_std,
                "fusion_weights": {
                    "spatial": round(self.fusion_weights.spatial, 3),
                    "behavioral": round(self.fusion_weights.behavioral, 3),
                    "jitter": round(self.fusion_weights.jitter, 3),
                },
            },
        }

    def _fuse(self, spatial: float, behavioral: float, jitter: float) -> float:
        w = self.fusion_weights
        return w.spatial * spatial + w.behavioral * behavioral + w.jitter * jitter

    @staticmethod
    def verdict_label(visual_score: float) -> str:
        """
        Coarse, human-readable label from the fused score. Thresholds below
        are reasonable starting points, NOT calibrated against labeled data.
        Until you calibrate against real/fake videos (see README), treat
        this as a readability aid, not a certified probability.
        """
        if visual_score >= 0.70:
            return "Likely Real"
        if visual_score >= 0.45:
            return "Uncertain"
        return "Likely Synthetic"

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
