"""PhysNet BVP extraction — supervised alternative to CHROM/POS.

    physnet.extract(frames, boxes) -> (bvp_signal, fs) or (None, reason)

WHY THIS EXISTS
CHROM and POS are hand-designed chrominance projections. They found no
recoverable pulse on any clip in this corpus — compressed social media, an 8 Mbps
camera capture, and a controlled recording all measured near-zero band SNR. That
pointed at the footage rather than the extractor, but it had never been tested
against a *learned* extractor.

RESULT: NOT USED FOR FUSION — MEASURED WORSE THAN CHROM
Band SNR on identical clips, PhysNet vs CHROM on the same face boxes:

    clip        PhysNet          CHROM
    REAL2.mp4   -6.78 dB         -3.26 dB
    REAL5.mp4   -5.51 dB         -2.32 dB
    REAL3.mp4   -6.92 dB         -1.74 dB
    FAKE2.mp4   -6.98 dB         -4.03 dB

Checked whether chunk-boundary discontinuities explained it — they do not. Scored
per 128-frame chunk with no concatenation, REAL2 gave -6.40/-5.73/-3.43 dB and
REAL5 gave -3.03/-6.42/-7.65/-4.18 dB, with heart rate swinging 96 to 178 BPM
across 17 seconds of the same seated subject. A real pulse does not move 80 BPM
in 17 seconds; that instability is itself the evidence that nothing is being
locked onto.

Most likely cause is domain gap: PhysNet was trained on UBFC-rPPG, which is
uncompressed lab capture with a still, well-lit subject. This corpus is
compressed, handheld, and re-encoded.

WHAT THIS BUYS US ANYWAY
The negative result is worth more than a marginal positive one. Both a classical
projection AND a supervised extractor trained on ground-truth pulse oximetry find
near-zero pulse SNR on this footage. That makes "the compression destroyed the
signal, and we abstain rather than guess" a measured conclusion rather than an
excuse — which is exactly what the problem statement's rPPG requirement needs
answering with.

Kept loadable and reportable, excluded from the fused score, per the same
admission-by-measurement rule that disabled xception and capsule.

CONTRACT — taken from rPPG-Toolbox's own inference config, not guessed:
    input       NCDHW, i.e. (batch, 3, T, 72, 72)
    T           128 frames per chunk (CHUNK_LENGTH)
    crop        Haar box scaled by LARGE_BOX_COEF = 1.5
    resize      72 x 72
    data type   DiffNormalized  (NOT standardized, NOT raw)
    output      rPPG waveform (BVP), one value per frame

DiffNormalized is transcribed from BaseLoader.diff_normalize_data:
    d[j] = (x[j+1] - x[j]) / (x[j+1] + x[j] + 1e-7)
    d   /= std(d)
    d    = append(d, zeros(1))        # pad back to T
    NaNs -> 0
Substituting z-scoring here would not fail; it would quietly return a plausible
waveform from an out-of-distribution input. That failure mode has already cost
this project twice, so the formula is reproduced exactly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "models" / "physnet_ubfc.pth"

FRAME_SIZE = 72          # RESIZE.H / RESIZE.W
CHUNK = 128              # PHYSNET.FRAME_NUM / CHUNK_LENGTH
LARGE_BOX_COEF = 1.5     # CROP_FACE.LARGE_BOX_COEF


@lru_cache(maxsize=1)
def _model():
    """Load PhysNet once. None when weights or torch are unavailable."""
    if not WEIGHTS.exists():
        return None
    try:
        import torch

        from src.rppg.physnet_model import PhysNet_padding_Encoder_Decoder_MAX

        net = PhysNet_padding_Encoder_Decoder_MAX(frames=CHUNK)
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Checkpoints trained under DataParallel carry a "module." prefix.
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        net.load_state_dict(state)
        net.eval()
        return net
    except Exception:
        return None


def weights_available() -> bool:
    return _model() is not None


def _crop(frame_bgr: np.ndarray, box) -> np.ndarray | None:
    """Haar box scaled by 1.5 and resized to 72x72, matching the training config."""
    if box is None or np.isnan(np.asarray(box, dtype=float)).any():
        return None
    x, y, w, h = [float(v) for v in box]
    H, W = frame_bgr.shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * LARGE_BOX_COEF
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x1, y1 = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def diff_normalize(data: np.ndarray) -> np.ndarray:
    """Exact transcription of BaseLoader.diff_normalize_data. (T,H,W,C) -> (T,H,W,C)."""
    n, h, w, c = data.shape
    out = np.zeros((n - 1, h, w, c), dtype=np.float32)
    for j in range(n - 1):
        out[j] = (data[j + 1] - data[j]) / (data[j + 1] + data[j] + 1e-7)
    sd = np.std(out)
    out = out / sd if sd > 1e-12 else out
    out = np.append(out, np.zeros((1, h, w, c), dtype=np.float32), axis=0)
    out[np.isnan(out)] = 0
    return out


def extract(frames_bgr, boxes, fs: float):
    """BVP waveform for a clip. Returns (signal, reason). signal is None on failure.

    Frames are consumed in non-overlapping 128-frame chunks — the length the model
    was trained on — and the per-chunk outputs are z-scored before concatenation,
    because PhysNet's output scale is arbitrary per chunk.
    """
    net = _model()
    if net is None:
        return None, "physnet_unavailable"

    import torch

    crops, kept = [], 0
    for frame, box in zip(frames_bgr, boxes):
        c = _crop(frame, box)
        if c is None:
            continue
        crops.append(c)
        kept += 1
    if kept < CHUNK:
        return None, f"too_few_face_frames({kept}<{CHUNK})"

    arr = np.asarray(crops, dtype=np.float32)
    n_chunks = len(arr) // CHUNK
    pieces = []
    with torch.no_grad():
        for k in range(n_chunks):
            chunk = arr[k * CHUNK:(k + 1) * CHUNK]
            dn = diff_normalize(chunk)                      # (T,H,W,C)
            x = torch.from_numpy(dn.transpose(3, 0, 1, 2))  # (C,T,H,W)
            x = x.unsqueeze(0)                              # NCDHW
            rppg = net(x)[0].reshape(-1).cpu().numpy().astype(np.float64)
            sd = rppg.std()
            pieces.append((rppg - rppg.mean()) / sd if sd > 1e-12 else rppg * 0.0)

    if not pieces:
        return None, "no_full_chunk"
    return np.concatenate(pieces), None
