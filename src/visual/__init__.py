"""Frame-level visual forgery detection (Xception) — third modality."""

from src.visual.detector import (
    VisualResult,
    analyze_frames,
    analyze_image,
    analyze_video,
    weights_available,
)

__all__ = ["VisualResult", "analyze_video", "analyze_image", "analyze_frames",
           "weights_available"]
