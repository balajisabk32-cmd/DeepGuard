"""PPG spatial-temporal maps — FakeCatcher / DeepRhythm style, classical.

Implements the STMap structure from Kammari et al. 2024 §2 and the four
discriminative dimensions from §3, without a CNN:

  1. Cross-region spatial consistency   -> pairwise corr + HR spread across patches
  2. Spectral power distribution / SNR  -> per-patch band SNR, peak sharpness
  3. Temporal HRV                       -> HR jump between consecutive windows
  4. Phase coherence                    -> weighted circular dispersion at f0

WHY THIS REPLACES 3 ROIs
The 3-ROI version reduced the whole face to `min` of 3 pairwise correlations.
`min` of 3 noisy estimates is the highest-variance statistic available — one bad
patch decides the verdict. A 5x6 grid gives ~30 patches and up to 435 pairs, so
the statistic becomes a DISTRIBUTION whose error falls as sqrt(N). Weak per-patch
SNR stops being fatal.

DeepRhythm's contribution (§4) is that ROIs are not equally informative: it
weights regions with stronger rPPG. Done here classically by weighting every
patch by its own band SNR, so noise-dominated patches cannot outvote clean ones.

The paper's authentic-face reference points (§3): pairwise r > 0.70,
HR spread < 12 BPM. Both are encoded in config/thresholds.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np

from src.rppg.signal_core import HR_BAND, best_extraction, peak_frequency, psd

GRID_ROWS, GRID_COLS = 6, 5
BLOCK = 8                      # pixels per patch edge after resize
MIN_VALID_FRAMES = 0.60        # a patch must survive this fraction of frames


def min_skin_fraction(cfg: dict | None = None) -> float:
    """Single source of truth, shared with the live viewer (config/thresholds.yaml).

    Hardcoding this per module let the offline detector and the heatmap drift to
    0.35 and 0.30, so the viewer showed patches the detector had thrown away.
    """
    if cfg is None:
        from src.fusion import load_thresholds  # noqa: PLC0415
        cfg = load_thresholds()
    return float(cfg["rppg"].get("map", {}).get("min_skin_fraction", 0.35))


@dataclass
class PPGMapResult:
    n_patches_used: int = 0
    corr_p25: float = 0.0              # robust stand-in for the paper's r_min
    corr_median: float = 0.0
    hr_spread_bpm: float = 0.0         # SNR-weighted, trimmed
    phase_dispersion: float = 0.0
    hr_temporal_jump_bpm: float = 0.0  # paper §3.3 — the dimension 3 ROIs never had
    mean_patch_snr_db: float = -np.inf
    best_snr_db: float = -np.inf
    global_hr_bpm: float | None = None
    coherence_map: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    reference_signal: np.ndarray = field(default_factory=lambda: np.zeros(0))
    degraded_reason: str | None = None


# ------------------------------------------------------------------ STMap


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def build_stmap(frames_bgr, boxes: np.ndarray,
                rows: int = GRID_ROWS, cols: int = GRID_COLS,
                skin_fraction: float | None = None) -> np.ndarray:
    """Spatial-temporal map: (n_frames, rows, cols, 3) of skin-masked mean RGB.

    The face crop is resized ONCE per frame to rows*BLOCK x cols*BLOCK and the
    patch means fall out of a single reshape. Doing 30 independent crop+convert+
    mask operations per frame instead would cost more than the entire rest of the
    pipeline; this way 30 patches cost about the same as 3.

    Downsizing with INTER_AREA is also a spatial low-pass, which raises per-patch
    SNR rather than lowering it — averaging pixels is what rPPG wants anyway.
    """
    n = len(frames_bgr)
    out = np.full((n, rows, cols, 3), np.nan)
    H, W = rows * BLOCK, cols * BLOCK
    skin_fraction = min_skin_fraction() if skin_fraction is None else skin_fraction

    for i, frame in enumerate(frames_bgr):
        if np.isnan(boxes[i]).any():
            continue
        x, y, w, h = boxes[i]
        fh, fw = frame.shape[:2]
        x0, x1 = int(np.clip(x, 0, fw - 2)), int(np.clip(x + w, 1, fw))
        y0, y1 = int(np.clip(y, 0, fh - 2)), int(np.clip(y + h, 1, fh))
        if x1 - x0 < cols or y1 - y0 < rows:
            continue

        crop = cv2.resize(frame[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_AREA)
        rgb_u8 = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mask = _skin_mask(rgb_u8)
        rgb = rgb_u8.astype(np.float64)

        blocks = rgb.reshape(rows, BLOCK, cols, BLOCK, 3)
        mblocks = mask.reshape(rows, BLOCK, cols, BLOCK)

        counts = mblocks.sum(axis=(1, 3))
        sums = (blocks * mblocks[..., None]).sum(axis=(1, 3))

        ok = counts >= (BLOCK * BLOCK * skin_fraction)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = sums / counts[..., None]
        means[~ok] = np.nan
        out[i] = means

    return out


# ------------------------------------------------------------------ features


def snr_weights(snrs: np.ndarray) -> np.ndarray:
    """DeepRhythm-style ROI attention: weight each patch by its own band SNR."""
    return np.clip((np.asarray(snrs, dtype=float) + 6.0) / 12.0, 0.02, 1.0)


def patch_coherence(signals: np.ndarray, snrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch correlation against the SNR-weighted global pulse reference.

    THIS is the coherence quantity. It answers "does this patch beat in time with
    the rest of the face?" — not "is there a strong pulse here?", which is a
    per-patch SNR map and a completely different (and non-discriminative) picture.

    A face-swap seam shows up as a band of LOW coherence where synthesized skin
    fails to stay phase-locked to the authentic skin outside the mask, even when
    both regions individually show plenty of signal.

    Returns (coherence per patch in [-1, 1], z-scored signals).
    """
    S = np.atleast_2d(np.asarray(signals, dtype=float))
    if S.shape[0] < 2 or S.shape[1] < 8:
        return np.zeros(S.shape[0]), S

    z = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-12)
    w = snr_weights(snrs)
    ref = np.average(z, axis=0, weights=w)

    if ref.std() < 1e-12:
        return np.zeros(S.shape[0]), z

    coh = np.array([float(np.corrcoef(zi, ref)[0, 1]) if zi.std() > 1e-12 else 0.0
                    for zi in z])
    return np.nan_to_num(coh), z


def _weighted_circular_std(angles: np.ndarray, weights: np.ndarray) -> float:
    if angles.size == 0:
        return 0.0
    w = weights / max(weights.sum(), 1e-12)
    r = np.abs(np.sum(w * np.exp(1j * angles)))
    return float(np.sqrt(max(-2.0 * np.log(max(r, 1e-12)), 0.0)))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    w = weights / max(weights.sum(), 1e-12)
    mu = float(np.sum(w * values))
    return float(np.sqrt(max(np.sum(w * (values - mu) ** 2), 0.0)))


def _temporal_hr_jump(sig: np.ndarray, fs: float,
                      win_sec: float = 6.0, hop_sec: float = 1.5) -> float:
    """Paper §3.3: real hearts drift smoothly, generators jump.

    Median absolute HR change between consecutive sliding windows. A physiological
    rate wanders by a few BPM; a spectral peak riding on noise hops arbitrarily.
    Deliberately median-based so one bad window cannot dominate.
    """
    L, hop = int(win_sec * fs), int(hop_sec * fs)
    if len(sig) < L + hop:
        return 0.0
    hrs = []
    for s in range(0, len(sig) - L + 1, hop):
        f0, _ = peak_frequency(sig[s:s + L], fs)
        if f0 is not None:
            hrs.append(f0 * 60)
    if len(hrs) < 3:
        return 0.0
    return float(np.median(np.abs(np.diff(hrs))))


def analyze_map(stmap: np.ndarray, t: np.ndarray, fs: float,
                method: str = "pos_overlap") -> PPGMapResult:
    """Per-patch extraction, SNR-weighted aggregation, four feature dimensions.

    ONE extractor for every patch — never per-patch `best_extraction`.

    CHROM computes S = Xf - alpha*Yf while POS computes h = S0 + beta*S1; for the
    same physical pulse these can come out with opposite sign. Choosing the
    best-SNR method independently per patch therefore mixes polarity conventions
    across the grid and MANUFACTURES anti-correlation between patches that are
    physically in phase. Measured: injecting a perfectly coherent pulse drove
    corr_p25 to -0.96 (it should approach +1), and more amplitude made it worse.

    Sign differences must be physical — a real seam — not algorithmic. So the
    method is fixed across the grid. POS is the default because the review
    (Kammari et al. §2) reports it most stable across skin tones and head motion.
    """
    from src.rppg.signal_core import chrom_full, chrom_overlap, pos_overlap, resample_to_uniform

    extractors = {
        "pos_overlap": pos_overlap,
        "chrom_full": chrom_full,
        "chrom_overlap": chrom_overlap,
    }
    extract = extractors.get(method, pos_overlap)

    n_frames, rows, cols, _ = stmap.shape
    res = PPGMapResult()

    sigs, snrs, freqs, coords = [], [], [], []
    for r in range(rows):
        for c in range(cols):
            chan = stmap[:, r, c, :]
            if np.isfinite(chan[:, 0]).mean() < MIN_VALID_FRAMES:
                continue
            rgb = np.column_stack([resample_to_uniform(t, chan[:, k], fs) for k in range(3)])
            if rgb.size == 0 or rgb.shape[0] < 64:
                continue
            sig = extract(rgb, fs)
            if sig.size == 0 or not np.isfinite(sig).all():
                continue
            f0, snr = peak_frequency(sig, fs)
            if f0 is None or not np.isfinite(snr):
                continue
            sigs.append(sig)
            snrs.append(snr)
            freqs.append(f0 * 60.0)
            coords.append((r, c))

    if len(sigs) < 4:
        res.degraded_reason = "too_few_valid_patches"
        return res

    L = min(len(s) for s in sigs)
    S = np.array([s[:L] for s in sigs])
    snrs = np.array(snrs)
    freqs = np.array(freqs)

    # DeepRhythm-style attention: weight each patch by its own SNR so
    # noise-dominated patches cannot outvote informative ones.
    w = np.clip((snrs + 6.0) / 12.0, 0.02, 1.0)

    res.n_patches_used = len(sigs)
    res.mean_patch_snr_db = float(np.average(snrs, weights=w))
    res.best_snr_db = float(snrs.max())

    # --- 1. spatial consistency -------------------------------------------
    corrs, pair_w, phases = [], [], []
    f_ref = float(np.average(freqs, weights=w)) / 60.0
    k_ref = int(round(f_ref * L / fs))

    F = np.fft.rfft(S, axis=1) if 0 < k_ref < L // 2 else None

    for a, b in combinations(range(len(sigs)), 2):
        sa, sb = S[a], S[b]
        if sa.std() < 1e-12 or sb.std() < 1e-12:
            continue
        corrs.append(float(np.corrcoef(sa, sb)[0, 1]))
        pair_w.append(float(w[a] * w[b]))
        if F is not None:
            phases.append(float(np.angle(F[a, k_ref] * np.conj(F[b, k_ref]))))

    if len(corrs) < 3:
        res.degraded_reason = "too_few_patch_pairs"
        return res

    corrs = np.array(corrs)
    pair_w = np.array(pair_w)

    # p25 rather than min: with hundreds of pairs, `min` is pure worst-case noise.
    res.corr_p25 = float(np.percentile(corrs, 25))
    res.corr_median = float(np.median(corrs))

    # --- 2/4. HR spread and phase dispersion ------------------------------
    res.hr_spread_bpm = _weighted_std(freqs, w) * 2.0   # ~ +/-1 sigma spread
    res.phase_dispersion = _weighted_circular_std(np.array(phases), pair_w) if phases else 0.0

    # --- reference signal + per-patch coherence (shared with the live viewer) ---
    coh, Z = patch_coherence(S, snrs)
    ref = np.average(Z, axis=0, weights=w)
    res.reference_signal = ref

    f0_ref, _ = peak_frequency(ref, fs)
    res.global_hr_bpm = None if f0_ref is None else float(f0_ref * 60)

    # --- 3. temporal HRV --------------------------------------------------
    res.hr_temporal_jump_bpm = _temporal_hr_jump(ref, fs)

    # --- coherence heatmap: each patch vs the global reference ------------
    cmap = np.full((rows, cols), np.nan)
    for (r, c), v in zip(coords, coh):
        cmap[r, c] = float(v)
    res.coherence_map = cmap

    return res


def map_manipulation_score(res: PPGMapResult, cfg: dict) -> float:
    """Combine the four dimensions into P(manipulated). Paper §3 thresholds."""
    m = cfg["rppg"].get("map", {})
    r_auth = m.get("authentic_corr", 0.70)          # paper: r > 0.70 for real faces
    spread_norm = cfg["rppg"]["hr_spread_norm_bpm"]  # paper: dHR > 12 BPM suspicious
    jump_norm = m.get("hr_jump_norm_bpm", 12.0)
    w = m.get("weights", {"corr": 0.35, "spread": 0.25, "phase": 0.20, "hrv": 0.20})

    d_corr = 1.0 - np.clip(res.corr_p25 / r_auth, 0.0, 1.0)
    d_spread = np.clip(res.hr_spread_bpm / spread_norm, 0.0, 1.0)
    d_phase = np.clip(res.phase_dispersion / (np.pi / 2), 0.0, 1.0)
    d_hrv = np.clip(res.hr_temporal_jump_bpm / jump_norm, 0.0, 1.0)

    return float(np.clip(
        w["corr"] * d_corr + w["spread"] * d_spread
        + w["phase"] * d_phase + w["hrv"] * d_hrv,
        0.0, 1.0,
    ))
