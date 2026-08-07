"""Input-robustness tests — plan.md §6.2.

Every row of the §6.2 table gets a test here, and CP3 does not pass with any row
untested. Under upload-only, arbitrary-input handling is a DEMO REQUIREMENT, not
Phase-3 polish: a judge asking "can I try my own video?" is the highest-risk live
moment (§6.4).

These are skipped until the ingestion pipeline lands in Phase 1. Convert each to a
real assertion as the path becomes available — do not delete them to make the suite
green.
"""

import pytest

pytestmark = pytest.mark.edge_case

CASES = [
    ("no_face_in_any_frame", "INSUFFICIENT_EVIDENCE + 'no face detected' message"),
    ("face_in_under_60pct_of_frames", "analyze detected span, warn, reduce confidence"),
    ("multiple_faces", "track largest/most-centred, warn explicitly"),
    ("side_profile_extreme_yaw", "rppg_quality -> 0, lip-sync continues if mouth visible"),
    ("no_audio_track", "audio_path=None, lipsync_quality=0, rPPG alone. NOT an error"),
    ("audio_present_no_speech", "speech_windows_used < 8, warn 'no speech detected'"),
    ("clip_under_4s", "reject at upload with a duration message"),
    ("clip_over_20s", "accept, truncate to first 20s, set analyzed_duration_sec, warn"),
    ("4k_or_very_large", "downscale to 640px LONG EDGE"),
    ("portrait_rotated", "apply ffmpeg rotation metadata before landmarking"),
    ("exotic_codec_hevc_av1_prores", "decode or typed error naming the codec"),
    ("corrupted_truncated_file", "'file could not be read as video', no traceback"),
    ("zero_byte_or_renamed_non_video", "reject at upload via magic-byte + ffprobe"),
    ("over_100mb", "reject before touching disk"),
]


@pytest.mark.parametrize("case,expected", CASES, ids=[c for c, _ in CASES])
@pytest.mark.skip(reason="Ingestion pipeline lands in Phase 1 (plan §5)")
def test_edge_case_returns_typed_response(case, expected):
    """Must return a typed, plain-English response. Never a stack trace."""
    raise AssertionError(f"unimplemented: {case} -> {expected}")


@pytest.mark.skip(reason="Ingestion pipeline lands in Phase 1 (plan §5)")
def test_class_f_clips_never_return_likely_manipulated():
    """P0 bug if violated (§4.2): Class F clips are AUTHENTIC, shot in adverse
    conditions. They must degrade to UNCERTAIN, never accuse a real person."""
    raise AssertionError("unimplemented")


@pytest.mark.skip(reason="Ingestion pipeline lands in Phase 1 (plan §5)")
def test_vfr_input_uses_pts_not_nominal_fps():
    """cv2.CAP_PROP_FPS on a VFR file rebuilds the uniform grid the plan forbids,
    fabricating periodicity in the HR band (§6.1 Critical, §6.3)."""
    raise AssertionError("unimplemented")


@pytest.mark.skip(reason="Ingestion pipeline lands in Phase 1 (plan §5)")
def test_av_start_offset_is_subtracted_before_lag_estimation():
    """Otherwise the container's own A/V offset lands in median_lag_ms, where it is
    indistinguishable from manipulation (§6.3, Appendix B)."""
    raise AssertionError("unimplemented")
