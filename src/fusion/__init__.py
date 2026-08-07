"""Quality-weighted fusion. Role 1 owns from T+9; Role 3 is backup."""

from src.fusion.scorer import explain, load_thresholds, score

__all__ = ["score", "explain", "load_thresholds"]
