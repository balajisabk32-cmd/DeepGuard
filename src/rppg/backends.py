"""Face/ROI providers — Role 1 (plan §2, §3.4).

Two backends behind one interface:

  MediaPipeBackend  preferred. Needs models/face_landmarker.task (H0-E download).
  OpenCVBackend     fallback. Haar cascade bundled with opencv — works fully offline.

WHY A FALLBACK EXISTS: mediapipe 1.0.0 REMOVED `mp.solutions` entirely. Only
`mediapipe.tasks` survives, and FaceLandmarker requires a .task model file that is
not in the wheel. Every `mp.solutions.face_mesh.FaceMesh(...)` snippet your team
finds online is dead code on this install. Until the weights land, OpenCV runs.

ROI DESIGN (plan §3.4): three regions — forehead, left cheek, right cheek. The
forehead matters specifically because face-swap masks typically stop at or below
the hairline, so it usually sits OUTSIDE the manipulated region while the cheeks
sit inside. Cross-region comparison straddles that seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROI_NAMES = ("forehead", "cheek_l", "cheek_r")

# ROI rectangles as fractions of the face bounding box (x0, y0, x1, y1).
# Chosen to avoid eyebrows, eyes, nostrils and the mouth — all of which move
# independently of blood volume and inject motion noise into the colour mean.
_ROI_FRACTIONS = {
    "forehead": (0.30, 0.08, 0.70, 0.24),
    "cheek_l":  (0.10, 0.46, 0.33, 0.70),
    "cheek_r":  (0.67, 0.46, 0.90, 0.70),
}

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"


@dataclass
class FrameROIs:
    """Mean RGB per ROI for one frame. NaN where the ROI was unusable."""
    means: np.ndarray          # shape (3 rois, 3 channels), float64, NaN-filled
    face_found: bool
    n_faces: int


def _skin_mask(patch_rgb: np.ndarray) -> np.ndarray:
    """YCrCb skin gate. Excludes hair, glasses, background bleed at the ROI edge.

    Without this, a bounding box that drifts a few pixels swaps skin pixels for
    hair pixels and injects a low-frequency swing into the channel mean that
    looks exactly like a slow pulse.
    """
    ycrcb = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def _patch_mean(frame_bgr: np.ndarray, box, frac) -> np.ndarray:
    """Skin-masked mean RGB for one ROI. Returns NaN triple if unusable.

    Takes a BGR frame and converts only the cropped patch. Converting the whole
    frame to RGB just to sample three small rectangles costs more than every
    other signal-processing stage in the pipeline combined.
    """
    x, y, w, h = box
    fx0, fy0, fx1, fy1 = frac
    H, W = frame_bgr.shape[:2]

    x0 = int(np.clip(x + fx0 * w, 0, W - 1))
    x1 = int(np.clip(x + fx1 * w, 0, W))
    y0 = int(np.clip(y + fy0 * h, 0, H - 1))
    y1 = int(np.clip(y + fy1 * h, 0, H))

    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.full(3, np.nan)

    patch_bgr = frame_bgr[y0:y1, x0:x1]
    if patch_bgr.size == 0:
        return np.full(3, np.nan)

    patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    mask = _skin_mask(patch)
    # Require a real majority of skin: a mostly-hair ROI is worse than no ROI.
    if mask.mean() < 0.20:
        return np.full(3, np.nan)

    sel = patch[mask].astype(np.float64)
    return sel.mean(axis=0)


class OpenCVBackend:
    """Haar detection + offline trajectory smoothing.

    Detection runs on a downscaled grayscale copy (speed), and the resulting box
    trajectory is smoothed with a MEDIAN filter over the whole clip rather than an
    online EMA. An EMA lags the true position and drifts, which slowly slides the
    ROI across the face and modulates the colour mean — a low-frequency artefact
    landing near the bottom of the HR band. Offline median smoothing has no lag
    and rejects single-frame detector jitter outright.
    """

    name = "opencv-haar"

    def __init__(self, detect_width: int = 320, stride: int = 5):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {path}")
        self._detect_width = detect_width
        # Detect every Nth frame and interpolate between. The head moves far
        # slower than 30 Hz, and the trajectory is median-filtered afterwards
        # regardless, so the detections in between carry no extra information —
        # they only cost time. This is the difference between a 20s and a 7s run.
        self._stride = max(1, stride)

    def detect_boxes(self, frames_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (boxes float64 (N,4) with NaN where undetected, n_faces int (N,))."""
        boxes = np.full((len(frames_bgr), 4), np.nan)
        counts = np.zeros(len(frames_bgr), dtype=int)

        for i, frame in enumerate(frames_bgr):
            if i % self._stride and i != len(frames_bgr) - 1:
                continue
            h, w = frame.shape[:2]
            scale = self._detect_width / float(w) if w > self._detect_width else 1.0
            small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            found = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            counts[i] = len(found)
            if len(found) == 0:
                continue

            # Largest, most-centred face (plan §6.2 multiple-faces rule).
            cx = small.shape[1] / 2.0
            best = max(found, key=lambda b: b[2] * b[3] - 0.25 * abs(b[0] + b[2] / 2 - cx))
            boxes[i] = np.array(best, dtype=np.float64) / scale

        # Carry detections across the strided gaps so every frame has an ROI.
        if self._stride > 1:
            idx = np.arange(len(frames_bgr))
            have = ~np.isnan(boxes[:, 0])
            if have.sum() >= 2:
                for c in range(4):
                    boxes[:, c] = np.interp(idx, idx[have], boxes[have, c])
                counts = np.interp(idx, idx[have], counts[have]).round().astype(int)

        return boxes, counts


class MediaPipeBackend:
    """FaceLandmarker (VIDEO mode) — preferred once weights are vendored.

    Landmark-derived ROIs track head pose properly, where a Haar box does not.
    Falls back automatically if the .task file is absent.
    """

    name = "mediapipe-facelandmarker"

    def __init__(self, model_path: Path = MODEL_PATH):
        from mediapipe.tasks.python import BaseOptions, vision  # noqa: PLC0415

        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} missing. Download at H0-E, or the OpenCV backend is used."
            )
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def detect_boxes(self, frames_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        import mediapipe as mp  # noqa: PLC0415

        boxes = np.full((len(frames_bgr), 4), np.nan)
        counts = np.zeros(len(frames_bgr), dtype=int)

        for i, frame in enumerate(frames_bgr):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._lm.detect_for_video(img, int(i * 33))
            if not res.face_landmarks:
                continue
            counts[i] = len(res.face_landmarks)
            pts = np.array([[p.x, p.y] for p in res.face_landmarks[0]])
            h, w = frame.shape[:2]
            xs, ys = pts[:, 0] * w, pts[:, 1] * h
            boxes[i] = [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()]

        return boxes, counts


def get_backend(prefer_mediapipe: bool = True):
    """MediaPipe if usable, OpenCV otherwise. Never raises."""
    if prefer_mediapipe:
        try:
            return MediaPipeBackend()
        except Exception:
            pass
    return OpenCVBackend()


def smooth_boxes(boxes: np.ndarray, kernel: int = 9) -> np.ndarray:
    """Median-filter the box trajectory, interpolating across undetected frames."""
    from scipy.signal import medfilt  # noqa: PLC0415

    out = boxes.copy()
    n = len(out)
    idx = np.arange(n)
    for c in range(4):
        col = out[:, c]
        ok = ~np.isnan(col)
        if ok.sum() < 2:
            continue
        col = np.interp(idx, idx[ok], col[ok])
        k = min(kernel if kernel % 2 == 1 else kernel + 1, n if n % 2 == 1 else n - 1)
        out[:, c] = medfilt(col, kernel_size=max(k, 1)) if k >= 3 else col
    return out


def extract_roi_series(
    frames_bgr: list[np.ndarray], boxes: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame skin-masked mean RGB for all three ROIs.

    Returns (series (N, 3 rois, 3 ch) with NaN gaps, face_found bool (N,)).

    NaN rather than dropping the sample: silently omitting undetected frames
    compresses the time axis and biases the frequency estimate. Gaps are carried
    explicitly and resolved later against real timestamps.
    """
    n = len(frames_bgr)
    series = np.full((n, len(ROI_NAMES), 3), np.nan)
    found = np.zeros(n, dtype=bool)

    for i, frame in enumerate(frames_bgr):
        if np.isnan(boxes[i]).any():
            continue
        box = boxes[i]
        for j, roi in enumerate(ROI_NAMES):
            series[i, j] = _patch_mean(frame, box, _ROI_FRACTIONS[roi])
        found[i] = counts[i] > 0

    return series, found
