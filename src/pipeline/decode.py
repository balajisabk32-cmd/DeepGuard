"""Decode-once clip cache for multi-window analysis.

Aggregation was re-decoding the video and re-running Haar detection for every
(window x modality) pair: 2 windows x 2 modalities = 4 full passes, ~58s on a 20s
clip. Decode and detection are identical work every time — only the slice differs.

This decodes once, detects once, and hands out cheap views. Cost goes from
O(windows x modalities) to O(1) for the expensive part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.lipsync.audio_io import AudioTrack, load_audio
from src.rppg import backends
from src.rppg.analyze import read_frames


@dataclass
class DecodedClip:
    frames: list
    t: np.ndarray               # real PTS, seconds
    nominal_fps: float
    boxes: np.ndarray           # (N,4), NaN where undetected, already smoothed
    counts: np.ndarray
    audio: AudioTrack | None
    warnings: list
    # Per-frame pixel work, hoisted out of the window loop. These are the real
    # cost: every window otherwise re-walks every frame to recompute the same
    # ROI means and patch grid. Computed once, sliced per window.
    roi_series: np.ndarray | None = None    # (N, 3 rois, 3 ch)
    stmap: np.ndarray | None = None         # (N, rows, cols, 3)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def fs(self) -> float:
        return (len(self.t) - 1) / self.duration if self.duration > 0 else 0.0

    def window(self, start_sec: float, length_sec: float) -> "DecodedClip":
        """View covering [start, start+length). Slices boxes with the frames, so
        detection is never repeated and the two stay index-aligned."""
        t0 = self.t[0] + start_sec
        m = (self.t >= t0) & (self.t < t0 + length_sec)
        idx = np.flatnonzero(m)
        if idx.size == 0:
            idx = np.arange(len(self.t))

        audio = self.audio
        if audio is not None and audio.ok and audio.sr:
            a0 = int(max(start_sec, 0.0) * audio.sr)
            a1 = int((max(start_sec, 0.0) + length_sec) * audio.sr)
            audio = AudioTrack(
                samples=audio.samples[a0:a1],
                sr=audio.sr,
                av_start_offset_sec=audio.av_start_offset_sec,
                ok=audio.ok,
                reason=audio.reason,
            )

        return DecodedClip(
            frames=[self.frames[i] for i in idx],
            t=self.t[idx],
            nominal_fps=self.nominal_fps,
            boxes=self.boxes[idx],
            counts=self.counts[idx],
            audio=audio,
            warnings=self.warnings,
            roi_series=None if self.roi_series is None else self.roi_series[idx],
            stmap=None if self.stmap is None else self.stmap[idx],
        )


def decode_clip(video_path: str, max_sec: float = 60.0,
                with_audio: bool = True) -> DecodedClip:
    """One decode, one detection pass. Never raises."""
    frames, t, nominal, warns = read_frames(video_path, max_sec=max_sec)
    if not frames:
        return DecodedClip([], np.zeros(0), 30.0, np.zeros((0, 4)),
                           np.zeros(0, dtype=int), None, warns or ["decode_failed"])

    boxes, counts = backends.OpenCVBackend().detect_boxes(frames)
    boxes = backends.smooth_boxes(boxes)

    audio = load_audio(video_path, max_sec=max_sec) if with_audio else None

    roi_series = stmap = None
    try:
        roi_series, _ = backends.extract_roi_series(frames, boxes, counts)
    except Exception:
        pass
    try:
        from src.fusion import load_thresholds
        from src.rppg.ppgmap import build_stmap

        mcfg = load_thresholds()["rppg"].get("map", {})
        if mcfg.get("enabled"):
            gr, gc = mcfg.get("grid", [6, 5])
            stmap = build_stmap(frames, boxes, rows=gr, cols=gc)
    except Exception:
        stmap = None   # the map is an enhancement; never let it break decoding

    return DecodedClip(frames, t, nominal, boxes, counts, audio, list(warns),
                       roi_series=roi_series, stmap=stmap)
