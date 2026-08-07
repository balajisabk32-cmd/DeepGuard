"""
motion_jitter.py
-----------------
Sub-component 3 of Feature 1: Motion Jitter & Warping Check.

Face-swap / reenactment algorithms composite a generated face onto each
frame independently (or with imperfect temporal smoothing), which shows up
as small high-frequency position jitter in tracked landmarks — motion a real,
rigid piece of tissue like the nose tip cannot physically produce frame to
frame under normal video capture.

We track a small set of anatomically stable landmarks (nose tip, plus the
two inner eye corners as a secondary check) and analyze the *jerk*
(second derivative of position) of their trajectories. Smooth natural head
motion has low jerk; face-swap warping introduces high-frequency spikes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

NOSE_TIP_IDX = 1
LEFT_EYE_INNER_IDX = 362
RIGHT_EYE_INNER_IDX = 133

TRACKED_IDX = [NOSE_TIP_IDX, LEFT_EYE_INNER_IDX, RIGHT_EYE_INNER_IDX]


@dataclass
class JitterResult:
    jitter_score: float              # 0.0 (heavily jittery/warped) .. 1.0 (smooth/authentic)
    mean_jerk: float
    jerk_std: float
    frames_tracked: int
    per_landmark_jerk: dict = field(default_factory=dict)


class MotionJitterDetector:
    """
    Stateful, per-video accumulator. Call `update(landmarks_xy)` once per
    frame with pixel-space (x, y) landmark coordinates, then `finalize()`.
    """

    def __init__(self, jerk_clip_percentile: float = 95.0):
        self.jerk_clip_percentile = jerk_clip_percentile
        self._tracks = {idx: [] for idx in TRACKED_IDX}
        self._frames_tracked = 0

    def update(self, landmarks_xy: Optional[np.ndarray]) -> None:
        if landmarks_xy is None:
            return
        self._frames_tracked += 1
        for idx in TRACKED_IDX:
            self._tracks[idx].append(landmarks_xy[idx].copy())

    @staticmethod
    def _jerk_series(positions: List[np.ndarray]) -> np.ndarray:
        """Second derivative (acceleration-of-position magnitude) of a 2D trajectory."""
        if len(positions) < 3:
            return np.array([])
        pos = np.stack(positions)  # (T, 2)
        velocity = np.diff(pos, axis=0)          # (T-1, 2)
        acceleration = np.diff(velocity, axis=0)  # (T-2, 2)
        jerk_mag = np.linalg.norm(acceleration, axis=1)
        return jerk_mag

    def finalize(self, face_scale_px: float = 150.0) -> JitterResult:
        """
        face_scale_px: an approximate reference face size (e.g. inter-ocular
        distance in pixels) used to normalize jerk so the score is roughly
        resolution/face-size independent instead of being in raw pixels.
        """
        if self._frames_tracked < 3:
            return JitterResult(
                jitter_score=0.5,
                mean_jerk=0.0,
                jerk_std=0.0,
                frames_tracked=self._frames_tracked,
                per_landmark_jerk={},
            )

        per_landmark_jerk = {}
        all_jerks = []
        for idx in TRACKED_IDX:
            jerks = self._jerk_series(self._tracks[idx])
            if jerks.size == 0:
                continue
            normalized = jerks / max(face_scale_px, 1.0)
            per_landmark_jerk[idx] = float(normalized.mean())
            all_jerks.append(normalized)

        if not all_jerks:
            return JitterResult(
                jitter_score=0.5,
                mean_jerk=0.0,
                jerk_std=0.0,
                frames_tracked=self._frames_tracked,
                per_landmark_jerk={},
            )

        combined = np.concatenate(all_jerks)
        mean_jerk = float(combined.mean())
        jerk_std = float(combined.std())

        # Empirical thresholds (normalized by face scale): natural handheld/
        # webcam footage typically sits well under ~0.01 mean normalized jerk.
        # These should be recalibrated against your own labeled footage.
        natural_ceiling = 0.012
        score = float(np.clip(1.0 - mean_jerk / natural_ceiling, 0.0, 1.0))

        return JitterResult(
            jitter_score=score,
            mean_jerk=round(mean_jerk, 6),
            jerk_std=round(jerk_std, 6),
            frames_tracked=self._frames_tracked,
            per_landmark_jerk={str(k): round(v, 6) for k, v in per_landmark_jerk.items()},
        )
