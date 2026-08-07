"""Executable Pydantic contracts for Multi-Modal Deepfake & Manipulation Detection (v2.1)."""

from enum import Enum
from typing import Literal, Optional, List, Tuple, Dict
from pydantic import BaseModel, Field, conlist

SCHEMA_VERSION = "2.0"

class Verdict(str, Enum):
    LIKELY_AUTHENTIC      = "LIKELY_AUTHENTIC"
    LIKELY_MANIPULATED    = "LIKELY_MANIPULATED"
    UNCERTAIN             = "UNCERTAIN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

class PreprocessResult(BaseModel):
    session_id: str
    schema_version: Literal["2.0"] = "2.0"
    video_path: str
    audio_path: Optional[str] = None          # None == silent video, valid state
    av_start_offset_sec: float = 0.0          # video.start_time - audio.start_time
    nominal_fps: float                        # metadata only — NEVER use as time axis
    frame_timestamps_sec: List[float]         # real PTS — the ONLY valid time axis
    source_duration_sec: float = Field(ge=4.0)
    analyzed_duration_sec: float = Field(ge=4.0, le=20.0)
    resolution: Tuple[int, int]
    landmarks: List[Optional[List[Tuple[float, float, float]]]]  # None per undetected frame
    face_detection_rate: float = Field(ge=0.0, le=1.0)
    faces_detected_max: int
    warnings: List[str] = []

class RPPGResult(BaseModel):
    session_id: str
    heart_rate_bpm: Optional[float] = None
    heart_rate_ci_bpm: Optional[Tuple[float, float]] = None
    rppg_manipulation_score: float = Field(ge=0.0, le=1.0)
    rppg_quality: float = Field(ge=0.0, le=1.0)
    band_snr_db: float
    cross_region_corr_min: float     # min pairwise Pearson across 3 ROIs
    cross_region_hr_spread_bpm: float
    phase_dispersion: float
    waveform_decimated: conlist(float, max_length=500)
    processing_time_ms: int
    degraded_reason: Optional[str] = None

class LipSyncResult(BaseModel):
    session_id: str
    lipsync_manipulation_score: float = Field(ge=0.0, le=1.0)
    lipsync_quality: float = Field(ge=0.0, le=1.0)
    median_lag_ms: Optional[float] = None
    lag_iqr_ms: Optional[float] = None        # primary discriminator
    mean_peak_ncc: Optional[float] = None
    speech_windows_used: int
    lag_resolution_ms: float
    mar_decimated: conlist(float, max_length=500)
    envelope_decimated: conlist(float, max_length=500)
    processing_time_ms: int
    degraded_reason: Optional[str] = None

class FusionResult(BaseModel):
    session_id: str
    schema_version: Literal["2.0"] = "2.0"
    verdict: Verdict
    manipulation_probability: float = Field(ge=0.0, le=1.0)
    evidence_weight: float = Field(ge=0.0, le=1.0)
    modality: Dict[str, Dict[str, float]]   # {"rppg": {"score":..., "quality":..., "weight":...}}
    explanation: str
    warnings: List[str] = []
    total_processing_time_ms: int
