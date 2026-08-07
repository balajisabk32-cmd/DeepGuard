"""
blink_monitor.py
-----------------
Sub-component 2 of Feature 1: Blink & Behavioral Monitoring (MediaPipe Face Mesh).

Tracks the Eye Aspect Ratio (EAR) per frame and derives blink events over the
course of a video. Flags two classic synthetic-face tells:
  - Frozen eyes: EAR barely changes across the whole clip (near-zero variance).
  - Unnatural blink frequency: too few blinks for the video length (early
    GAN-based face generators notoriously under-produce blinks), or an
    implausibly high/metronomic rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Standard 6-point EAR landmark sets for MediaPipe Face Mesh (468-landmark model).
# Order per eye: [outer_corner, top_1, top_2, inner_corner, bottom_1, bottom_2]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Natural adult blink rate range, blinks per minute (broad literature range).
NATURAL_BLINK_RATE_MIN = 8.0
NATURAL_BLINK_RATE_MAX = 30.0


def _euclidean(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def eye_aspect_ratio(landmarks_xy: np.ndarray, eye_idx: List[int]) -> float:
    """landmarks_xy: (468, 2) pixel coordinates. Returns EAR for one eye."""
    p1, p2, p3, p4, p5, p6 = [landmarks_xy[i] for i in eye_idx]
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)
    if horizontal < 1e-6:
        return 0.3  # neutral fallback
    return vertical / (2.0 * horizontal)


@dataclass
class BlinkMonitorResult:
    blinks_detected: int
    blink_rate_per_min: float
    frozen_eyes: bool
    ear_mean: float
    ear_std: float
    behavioral_score: float          # 0.0 (synthetic-leaning) .. 1.0 (authentic-leaning)
    ear_series: List[float] = field(default_factory=list)
    frames_with_face: int = 0


class BlinkBehavioralMonitor:
    """
    Stateful, per-video accumulator. Call `update(landmarks_xy)` once per
    frame with pixel-space (x, y) landmark coordinates from MediaPipe Face
    Mesh, then call `finalize(fps)` once at the end of the video.
    """

    def __init__(self, ear_blink_threshold: float = 0.21, min_consecutive_frames: int = 2):
        self.ear_blink_threshold = ear_blink_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self._ear_series: List[float] = []
        self._below_threshold_run = 0
        self._blink_count = 0
        self._frames_with_face = 0

    def update(self, landmarks_xy: Optional[np.ndarray]) -> Optional[float]:
        """Feed one frame's landmarks (or None if no face detected). Returns EAR or None."""
        if landmarks_xy is None:
            return None

        left_ear = eye_aspect_ratio(landmarks_xy, LEFT_EYE_IDX)
        right_ear = eye_aspect_ratio(landmarks_xy, RIGHT_EYE_IDX)
        ear = (left_ear + right_ear) / 2.0

        self._ear_series.append(ear)
        self._frames_with_face += 1

        if ear < self.ear_blink_threshold:
            self._below_threshold_run += 1
        else:
            if self._below_threshold_run >= self.min_consecutive_frames:
                self._blink_count += 1
            self._below_threshold_run = 0

        return ear

    def finalize(self, fps: float, total_frames_analyzed: int) -> BlinkMonitorResult:
        # catch a blink that was still in progress at the last frame
        if self._below_threshold_run >= self.min_consecutive_frames:
            self._blink_count += 1

        if not self._ear_series:
            return BlinkMonitorResult(
                blinks_detected=0,
                blink_rate_per_min=0.0,
                frozen_eyes=True,
                ear_mean=0.0,
                ear_std=0.0,
                behavioral_score=0.3,  # no face tracked -> low-confidence, penalize slightly
                ear_series=[],
                frames_with_face=0,
            )

        ear_arr = np.array(self._ear_series)
        ear_mean = float(ear_arr.mean())
        ear_std = float(ear_arr.std())

        duration_minutes = max(total_frames_analyzed / fps / 60.0, 1e-6)
        blink_rate = self._blink_count / duration_minutes

        # Frozen eyes: eyelids essentially don't move at all across the clip.
        frozen_eyes = ear_std < 0.015

        behavioral_score = self._score_behavior(blink_rate, frozen_eyes, duration_minutes)

        return BlinkMonitorResult(
            blinks_detected=self._blink_count,
            blink_rate_per_min=round(blink_rate, 2),
            frozen_eyes=frozen_eyes,
            ear_mean=round(ear_mean, 4),
            ear_std=round(ear_std, 4),
            behavioral_score=behavioral_score,
            ear_series=self._ear_series,
            frames_with_face=self._frames_with_face,
        )

    @staticmethod
    def _score_behavior(blink_rate: float, frozen_eyes: bool, duration_minutes: float) -> float:
        if frozen_eyes:
            return 0.05

        # Very short clips can legitimately show 0 blinks — don't over-penalize.
        if duration_minutes < 0.15:
            return 0.6

        if NATURAL_BLINK_RATE_MIN <= blink_rate <= NATURAL_BLINK_RATE_MAX:
            return 1.0

        # Graceful falloff outside the natural range rather than a hard cliff.
        if blink_rate < NATURAL_BLINK_RATE_MIN:
            deficit = NATURAL_BLINK_RATE_MIN - blink_rate
            return float(np.clip(1.0 - deficit / NATURAL_BLINK_RATE_MIN, 0.05, 1.0))
        else:
            excess = blink_rate - NATURAL_BLINK_RATE_MAX
            return float(np.clip(1.0 - excess / (NATURAL_BLINK_RATE_MAX * 2), 0.05, 1.0))
