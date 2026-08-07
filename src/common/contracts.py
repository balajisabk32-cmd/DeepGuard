"""Executable Pydantic contracts for Multi-Modal Deepfake & Manipulation Detection (v2.1).

SCORE POLARITY RULE (plan.md §3.2) — the one rule that prevents the 3am bug:

    *_manipulation_score  ->  P(manipulated) in [0, 1].  HIGHER = MORE SUSPICIOUS.
    *_quality             ->  evidence strength in [0, 1].  HIGHER = MORE TRUSTWORTHY.

No field is ever named just `score` or `confidence`. Polarity lives in the field
name so it cannot be misread at 3am. `tests/test_contracts.py` enforces this
reflectively — a field that breaks the convention fails the suite.

All models set extra="forbid": a mistyped field name is an error, not a silently
dropped value. With four people building against these shapes in parallel, a typo
that validates is a bug you find at integration time instead of immediately.
"""

from enum import Enum
from typing import Literal, Optional, List, Tuple, Dict

from pydantic import BaseModel, ConfigDict, Field, conlist

SCHEMA_VERSION = "2.0"


class _Strict(BaseModel):
    """Base: reject unknown fields, validate on assignment."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Verdict(str, Enum):
    LIKELY_AUTHENTIC      = "LIKELY_AUTHENTIC"
    LIKELY_MANIPULATED    = "LIKELY_MANIPULATED"
    UNCERTAIN             = "UNCERTAIN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class PreprocessResult(_Strict):
    session_id: str
    schema_version: Literal["2.0"] = "2.0"
    video_path: str
    audio_path: Optional[str] = None    # None == silent video. A valid state, not an error.

    # REQUIRED — deliberately has no default.
    # This field exists only because a forgotten A/V offset silently corrupts
    # median_lag_ms, where it is indistinguishable from the manipulation we are
    # trying to detect. A default of 0.0 would reproduce exactly the bug the field
    # was added to prevent, silently. Compute it or fail loudly. See plan Appendix B.
    av_start_offset_sec: float

    nominal_fps: float                  # metadata/display ONLY — never a time axis.
                                        # cv2.CAP_PROP_FPS on VFR input rebuilds the
                                        # uniform grid the plan forbids. See §6.3.
    frame_timestamps_sec: List[float]   # real PTS — the ONLY valid time axis

    source_duration_sec: float = Field(ge=4.0)                # unbounded above
    analyzed_duration_sec: float = Field(ge=4.0, le=20.0)     # what we actually processed

    resolution: Tuple[int, int]
    landmarks: List[Optional[List[Tuple[float, float, float]]]]  # None per undetected frame
    face_detection_rate: float = Field(ge=0.0, le=1.0)
    faces_detected_max: int
    warnings: List[str] = []


class RPPGResult(_Strict):
    session_id: str
    heart_rate_bpm: Optional[float] = None
    heart_rate_ci_bpm: Optional[Tuple[float, float]] = None   # interval, not a point estimate
    rppg_manipulation_score: float = Field(ge=0.0, le=1.0)
    rppg_quality: float = Field(ge=0.0, le=1.0)
    band_snr_db: float
    cross_region_corr_min: float          # min pairwise Pearson across the 3 ROIs
    cross_region_hr_spread_bpm: float
    phase_dispersion: float
    waveform_decimated: conlist(float, max_length=500)
    processing_time_ms: int
    degraded_reason: Optional[str] = None  # set to "insufficient_pulse_snr" when the
                                           # §3.4 quality gate forces a neutral score

    # --- PPG spatial-temporal map (FakeCatcher / DeepRhythm style) ---
    # Optional so the 3-ROI path stays valid when the map is disabled.
    map_patches_used: Optional[int] = None
    map_corr_p25: Optional[float] = None          # robust stand-in for r_min
    map_mean_patch_snr_db: Optional[float] = None
    map_hr_temporal_jump_bpm: Optional[float] = None   # paper §3.3 HRV dimension
    # Heatmap for the dashboard. Inner values are Optional: a patch that never
    # held enough skin pixels has no coherence value, and None is the honest
    # representation of that — not 0.0, which would read as "disagrees strongly".
    map_coherence: Optional[List[List[Optional[float]]]] = None


class LipSyncResult(_Strict):
    session_id: str
    lipsync_manipulation_score: float = Field(ge=0.0, le=1.0)
    lipsync_quality: float = Field(ge=0.0, le=1.0)
    median_lag_ms: Optional[float] = None
    lag_iqr_ms: Optional[float] = None     # primary discriminator — a real recording can
                                           # carry a constant offset, but not a wandering one
    mean_peak_ncc: Optional[float] = None
    speech_windows_used: int
    lag_resolution_ms: float               # honest quantization floor (1/fps)
    mar_decimated: conlist(float, max_length=500)
    envelope_decimated: conlist(float, max_length=500)
    processing_time_ms: int
    degraded_reason: Optional[str] = None


class FusionResult(_Strict):
    session_id: str
    schema_version: Literal["2.0"] = "2.0"
    verdict: Verdict
    manipulation_probability: float = Field(ge=0.0, le=1.0)
    evidence_weight: float = Field(ge=0.0, le=1.0)   # capped at 0.5 single-modality (§3.5)
    modality: Dict[str, Dict[str, float]]            # {"rppg": {"score":…, "quality":…, "weight":…}}
    explanation: str                                 # generated FROM the decision path
    warnings: List[str] = []
    total_processing_time_ms: int
