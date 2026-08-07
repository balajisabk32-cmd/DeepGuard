"""Audio demux and A/V alignment — Role 2 (plan §2, Appendix B).

Uses PyAV, which links FFmpeg's libraries in-process. No `ffmpeg` binary on PATH
is required, which matters here because there isn't one — and it also gives exact
per-stream `start_time`, which a subprocess pipeline would have to re-probe.

WHY THE OFFSET IS NOT OPTIONAL
MP4 routinely stores the video and audio streams with different start times.
Measured on this repo's clips:

    REAL.mp4                       video +0.0334s  audio 0.0000s  ->  +33.4 ms
    Deepfake tom cruise            video +0.0333s  audio 0.0000s  ->  +33.3 ms
    WIN_20260807_14_02_13_Pro.mp4  video  0.0000s  audio 0.1677s  -> -167.7 ms

That last one is four times the plan's 40 ms "suspicious offset" threshold. Ignore
it and an authentic recording reads as a dubbed fake — you would be measuring the
muxer, not the speaker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGET_SR = 16000          # webrtcvad accepts 8/16/32/48 kHz only


@dataclass
class AudioTrack:
    samples: np.ndarray     # float32 mono in [-1, 1]
    sr: int
    av_start_offset_sec: float   # video.start_time - audio.start_time
    ok: bool
    reason: str | None = None

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr if self.sr else 0.0


def _stream_start(stream) -> float:
    if stream is None or stream.start_time is None:
        return 0.0
    try:
        return float(stream.start_time * stream.time_base)
    except Exception:
        return 0.0


def load_audio(path: str, target_sr: int = TARGET_SR,
               max_sec: float | None = None) -> AudioTrack:
    """Decode the first audio stream to mono float32. Never raises.

    A silent video is a valid state, not an error (plan §6.2): it returns
    ok=False with a reason, and the caller degrades to rPPG alone.
    """
    empty = np.zeros(0, dtype=np.float32)
    try:
        import av
    except Exception:
        return AudioTrack(empty, target_sr, 0.0, False, "pyav_unavailable")

    try:
        container = av.open(path)
    except Exception as exc:
        return AudioTrack(empty, target_sr, 0.0, False, f"open_failed:{type(exc).__name__}")

    try:
        astreams = [s for s in container.streams if s.type == "audio"]
        vstreams = [s for s in container.streams if s.type == "video"]
        if not astreams:
            return AudioTrack(empty, target_sr, 0.0, False, "no_audio_stream")

        offset = _stream_start(vstreams[0] if vstreams else None) - _stream_start(astreams[0])

        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=target_sr
        )

        chunks: list[np.ndarray] = []
        total = 0
        limit = int(max_sec * target_sr) if max_sec else None
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                arr = out.to_ndarray().reshape(-1)
                chunks.append(arr)
                total += arr.size
            if limit and total >= limit:
                break

        if not chunks:
            return AudioTrack(empty, target_sr, offset, False, "audio_decode_empty")

        pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
        if limit:
            pcm = pcm[:limit]
        return AudioTrack(pcm, target_sr, float(offset), True)

    except Exception as exc:
        return AudioTrack(empty, target_sr, 0.0, False, f"decode_failed:{type(exc).__name__}")
    finally:
        try:
            container.close()
        except Exception:
            pass


def voiced_mask(pcm: np.ndarray, sr: int, frame_ms: int = 30,
                aggressiveness: int = 2) -> np.ndarray:
    """Per-frame speech mask via WebRTC VAD, with an energy fallback.

    Gating on speech matters because silence produces spurious correlation peaks:
    two flat signals correlate at whatever lag the noise happens to favour.
    """
    n_per = int(sr * frame_ms / 1000)
    n_frames = max(len(pcm) // n_per, 0)
    if n_frames == 0:
        return np.zeros(0, dtype=bool)

    try:
        import webrtcvad
        vad = webrtcvad.Vad(aggressiveness)
        pcm16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
        out = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            block = pcm16[i * n_per:(i + 1) * n_per].tobytes()
            try:
                out[i] = vad.is_speech(block, sr)
            except Exception:
                out[i] = False
        if out.any():
            return out
    except Exception:
        pass

    # Fallback: relative energy gate.
    energy = np.array([
        float(np.sqrt(np.mean(pcm[i * n_per:(i + 1) * n_per] ** 2) + 1e-12))
        for i in range(n_frames)
    ])
    return energy > max(np.median(energy) * 1.2, 1e-4)
