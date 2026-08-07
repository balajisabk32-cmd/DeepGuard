"""CHROM / POS extraction and spectral estimation — plan.md §3.4, Appendix A.

Named signal_core (not signal) to avoid shadowing the stdlib `signal` module,
which scipy imports internally.

PRECISION NOTE — the defect that makes naive implementations look correct:
Welch with nperseg = fs*10 at fs=30 gives bin spacing fs/nperseg = 0.1 Hz = 6 BPM.
A test targeting 72 BPM (1.2 Hz) lands EXACTLY on bin 12 and reports ~0 error.
Retarget to 75 BPM (1.25 Hz) and the same code is off by up to 3 BPM. The apparent
accuracy is an artefact of bin alignment, not of the estimator.

Fixed here by zero-padding (nfft >> nperseg) plus parabolic interpolation of the
spectral peak, which recovers sub-bin frequency regardless of alignment.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal as sps

HR_BAND = (0.7, 4.0)          # 42-240 BPM
_MIN_SAMPLES = 32


# ---------------------------------------------------------------- resampling


def resample_to_uniform(t: np.ndarray, x: np.ndarray, fs: float) -> np.ndarray:
    """Interpolate a possibly non-uniform, NaN-gapped series onto a uniform grid.

    This is where variable-frame-rate video is handled honestly. Indexing frames
    as i/fps on VFR input rebuilds a uniform grid that never existed and fabricates
    periodicity inside the HR band (plan §6.1, Critical).
    """
    ok = np.isfinite(x) & np.isfinite(t)
    if ok.sum() < _MIN_SAMPLES:
        return np.array([])
    t_ok, x_ok = t[ok], x[ok]
    grid = np.arange(t_ok[0], t_ok[-1], 1.0 / fs)
    return np.interp(grid, t_ok, x_ok)


def detrend_linear(x: np.ndarray) -> np.ndarray:
    """Closed-form least-squares line removal. Numerically identical to
    `scipy.signal.detrend(x, type="linear")`, but does NOT call LAPACK.

    scipy's linear detrend goes through `scipy.linalg.lstsq`. On this machine
    (Anaconda scipy + pip torch) both link their own copy of libiomp5md.dll, and
    calling LAPACK once torch is loaded raises OMP Error #15 and ABORTS THE
    PROCESS — not an exception, a fatal abort no `except` can catch.

    That matters here because lip-sync is moving to SyncNet (torch) while rPPG
    stays on scipy: the two would share a process and take the whole pipeline
    down mid-demo. Two lines of algebra removes the failure mode outright, and
    it is faster than lstsq besides.
    """
    n = len(x)
    if n < 2:
        return np.asarray(x, dtype=np.float64)
    t = np.arange(n, dtype=np.float64)
    dt = t - t.mean()
    denom = float((dt * dt).sum())
    slope = float((dt * (x - x.mean())).sum() / denom) if denom > 0 else 0.0
    return np.asarray(x, dtype=np.float64) - (x.mean() + slope * dt)


@lru_cache(maxsize=64)
def _butter_bandpass(order: int, lo: float, hi: float, fs: float):
    """Filter DESIGN is deterministic in its arguments, so it is cached.

    The overlap-add extractors call _bandpass once per 1.6s window, per channel,
    per patch. On a 12s clip with a 30-patch map that is several hundred identical
    `sps.butter` designs — and the design, not the filtering, dominated runtime.
    Rounding the key keeps float jitter in `fs` from defeating the cache.
    """
    nyq = 0.5 * fs
    return sps.butter(order, [lo / nyq, hi / nyq], btype="bandpass")


def _bandpass(x: np.ndarray, fs: float, lo: float = HR_BAND[0], hi: float = HR_BAND[1]):
    nyq = 0.5 * fs
    hi = min(hi, nyq * 0.95)
    if lo >= hi or len(x) < 27:
        return np.zeros_like(x)
    b, a = _butter_bandpass(3, round(lo, 4), round(hi, 4), round(fs, 3))
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return np.zeros_like(x)
    return sps.filtfilt(b, a, detrend_linear(x))


# ---------------------------------------------------------------- extractors


def chrom_full(rgb: np.ndarray, fs: float) -> np.ndarray:
    """CHROM over the whole signal (De Haan & Jeanne 2013). rgb: (N, 3)."""
    if len(rgb) < _MIN_SAMPLES:
        return np.array([])
    mean = rgb.mean(axis=0)
    if np.any(mean <= 0):
        return np.array([])
    cn = rgb / mean
    xs = 3 * cn[:, 0] - 2 * cn[:, 1]
    ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    xf, yf = _bandpass(xs, fs), _bandpass(ys, fs)
    sy = yf.std()
    alpha = xf.std() / sy if sy > 1e-12 else 0.0
    return xf - alpha * yf


def _overlap_add(rgb: np.ndarray, fs: float, win_sec: float, project) -> np.ndarray:
    """Hann-windowed overlap-add over short segments, 50% hop."""
    n = len(rgb)
    L = max(int(round(win_sec * fs)), _MIN_SAMPLES)
    if n < L:
        return np.array([])
    hop = L // 2
    out = np.zeros(n)
    wsum = np.zeros(n)
    win = np.hanning(L)

    for start in range(0, n - L + 1, hop):
        seg = rgb[start:start + L]
        mean = seg.mean(axis=0)
        if np.any(mean <= 0):
            continue
        h = project(seg / mean, fs)
        if h.size != L:
            continue
        out[start:start + L] += win * h
        wsum[start:start + L] += win

    good = wsum > 1e-9
    out[good] /= wsum[good]
    return out


def chrom_overlap(rgb: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    def project(cn: np.ndarray, fs_: float) -> np.ndarray:
        xs = 3 * cn[:, 0] - 2 * cn[:, 1]
        ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
        xf, yf = _bandpass(xs, fs_), _bandpass(ys, fs_)
        sy = yf.std()
        alpha = xf.std() / sy if sy > 1e-12 else 0.0
        return xf - alpha * yf

    return _overlap_add(rgb, fs, win_sec, project)


def pos_overlap(rgb: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    """POS (Wang et al. 2017)."""
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])

    def project(cn: np.ndarray, _fs: float) -> np.ndarray:
        s = proj @ cn.T
        s1 = s[1].std()
        h = s[0] + (s[0].std() / s1 if s1 > 1e-12 else 0.0) * s[1]
        return h - h.mean()

    return _overlap_add(rgb, fs, win_sec, project)


# ---------------------------------------------------------------- spectral


def psd(x: np.ndarray, fs: float, seg_sec: float = 8.0):
    """Welch PSD, zero-padded 8x for sub-bin peak resolution."""
    n = len(x)
    if n < _MIN_SAMPLES:
        return np.array([]), np.array([])
    nperseg = int(min(n, max(round(seg_sec * fs), _MIN_SAMPLES)))
    noverlap = nperseg // 2 if n > nperseg else 0
    nfft = int(2 ** np.ceil(np.log2(nperseg * 8)))
    f, p = sps.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                     detrend="constant")
    return f, p


def _parabolic_peak(f: np.ndarray, p: np.ndarray, k: int) -> float:
    """Sub-bin peak location by parabolic fit on log-power."""
    if k <= 0 or k >= len(p) - 1:
        return float(f[k])
    y0, y1, y2 = np.log(p[k - 1] + 1e-30), np.log(p[k] + 1e-30), np.log(p[k + 1] + 1e-30)
    denom = y0 - 2 * y1 + y2
    delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(f[k] + delta * (f[1] - f[0]))


def peak_frequency(x: np.ndarray, fs: float) -> tuple[float | None, float]:
    """Dominant in-band frequency (Hz) and its band SNR in dB."""
    f, p = psd(x, fs)
    if f.size == 0:
        return None, -np.inf
    band = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    if not band.any() or not np.isfinite(p[band]).any() or p[band].max() <= 0:
        return None, -np.inf

    k = int(np.flatnonzero(band)[np.argmax(p[band])])
    f0 = _parabolic_peak(f, p, k)
    return f0, band_snr_db(f, p, f0)


def band_snr_db(f: np.ndarray, p: np.ndarray, f0: float) -> float:
    """Pulse SNR: energy at f0 and its first harmonic vs the rest of the band.

    Definition transcribed from plan §3.4 — signal bins are within +/-0.1 Hz of f0
    and +/-0.2 Hz of 2*f0; everything else in 0.7-4.0 Hz is noise.
    """
    band = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    sig = band & ((np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.2))
    noise = band & ~sig
    ps, pn = p[sig].sum(), p[noise].sum()
    if pn <= 0 or ps <= 0:
        return -np.inf
    return float(10 * np.log10(ps / pn))


def hr_with_ci(x: np.ndarray, fs: float, win_sec: float = 8.0, hop_sec: float = 2.0):
    """Heart rate in BPM with an honest interval.

    A point estimate like `78.4 bpm` from a short window claims 0.1 BPM precision
    that the frequency resolution cannot support. The interval comes from the
    spread of per-window estimates when the clip is long enough, and from the
    peak's half-power width when it is not.
    """
    f0, snr = peak_frequency(x, fs)
    if f0 is None:
        return None, None, -np.inf

    n, L, hop = len(x), int(win_sec * fs), int(hop_sec * fs)
    ests = []
    if n >= L + hop:
        for s in range(0, n - L + 1, hop):
            fw, _ = peak_frequency(x[s:s + L], fs)
            if fw is not None:
                ests.append(fw * 60)

    bpm = float(f0 * 60)
    if len(ests) >= 3:
        lo, hi = np.percentile(ests, [10, 90])
        return bpm, (float(min(lo, bpm)), float(max(hi, bpm))), snr

    f, p = psd(x, fs)
    band = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    half = p[band].max() / 2.0
    above = f[band][p[band] >= half]
    width = (above.max() - above.min()) * 60 if above.size else 6.0
    return bpm, (bpm - width / 2, bpm + width / 2), snr


def best_extraction(rgb: np.ndarray, fs: float) -> tuple[str, np.ndarray, float | None, float]:
    """Run CHROM (full + overlap-add) and POS; keep whichever gives the best SNR.

    plan §Appendix A: "run both, keep whichever gives higher band_snr_db".
    """
    best = ("none", np.array([]), None, -np.inf)
    for name, fn in (
        ("chrom_full", chrom_full),
        ("chrom_overlap", chrom_overlap),
        ("pos_overlap", pos_overlap),
    ):
        try:
            sig = fn(rgb, fs)
        except Exception:
            continue
        if sig.size < _MIN_SAMPLES or not np.isfinite(sig).all():
            continue
        f0, snr = peak_frequency(sig, fs)
        if snr > best[3]:
            best = (name, sig, f0, snr)
    return best
