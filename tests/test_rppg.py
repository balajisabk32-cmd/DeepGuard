"""rPPG unit tests — Role 1 (plan §2 T+0 task).

The T+0 task is explicit: prove the extractor recovers a KNOWN frequency from a
synthetic signal before touching a real face. These are that proof.

The bin-alignment test is the important one. It is the defect that made the
original rppg_test/test_chrom_synthetic.py report PASS for the wrong reason.
"""

import numpy as np
import pytest

from src.rppg.analyze import analyze
from src.rppg.signal_core import (
    band_snr_db,
    chrom_full,
    hr_with_ci,
    peak_frequency,
    pos_overlap,
    psd,
    resample_to_uniform,
)

FPS = 30.0
DUR = 20.0


def synth_rgb(bpm: float, noise: float = 0.5, seed: int = 42, fps: float = FPS,
              dur: float = DUR) -> np.ndarray:
    """Skin-like RGB with a pulse at `bpm`, sensor noise and slow lighting drift."""
    t = np.arange(0, dur, 1.0 / fps)
    p = np.sin(2 * np.pi * (bpm / 60.0) * t)
    rng = np.random.default_rng(seed)
    drift = 10 * np.sin(2 * np.pi * 0.05 * t)
    return np.column_stack([
        150 + 0.5 * p + rng.normal(0, noise, t.size) + drift,   # R reacts least
        100 - 5.0 * p + rng.normal(0, noise, t.size) + drift,   # G reacts most
        100 - 1.5 * p + rng.normal(0, noise, t.size) + drift,
    ])


# --------------------------------------------------------------- precision

@pytest.mark.parametrize("bpm", [66, 72, 75, 81, 93, 108])
def test_recovers_known_frequency_regardless_of_bin_alignment(bpm):
    """The estimator must not depend on the target landing on a Welch bin.

    At fps=30 with nperseg=fps*10, bin spacing is 0.1 Hz = 6 BPM. A naive
    argmax reports ~0 error at 72 BPM (exactly bin 12) and 3 BPM error at 75 —
    half a bin — for reasons that have nothing to do with the signal. Zero-padding
    plus parabolic interpolation removes the dependence entirely.
    """
    sig = chrom_full(synth_rgb(bpm), FPS)
    f0, _ = peak_frequency(sig, FPS)
    assert f0 is not None
    assert abs(f0 * 60 - bpm) < 1.0, f"{bpm} BPM recovered as {f0 * 60:.2f}"


def test_naive_argmax_would_fail_this(  ):
    """Guard against a future 'simplification' back to a bare argmax."""
    from scipy.signal import welch

    sig = chrom_full(synth_rgb(75), FPS)
    f, p = welch(sig, fs=FPS, nperseg=int(FPS * 10))
    band = (f >= 0.7) & (f <= 4.0)
    naive = f[band][np.argmax(p[band])] * 60
    assert abs(naive - 75) > 2.0, "bin spacing changed; revisit this guard"

    f0, _ = peak_frequency(sig, FPS)
    assert abs(f0 * 60 - 75) < 1.0


def test_pos_also_recovers_the_frequency():
    f0, _ = peak_frequency(pos_overlap(synth_rgb(78), FPS), FPS)
    assert f0 is not None and abs(f0 * 60 - 78) < 2.0


def test_hr_interval_brackets_the_truth():
    bpm, ci, _ = hr_with_ci(chrom_full(synth_rgb(72), FPS), FPS)
    assert ci is not None and ci[0] <= bpm <= ci[1]
    assert ci[0] <= 72 <= ci[1], f"true rate outside reported interval {ci}"


# --------------------------------------------------------------- quality

def test_snr_separates_pulse_from_pure_noise():
    clean = chrom_full(synth_rgb(72, noise=0.2), FPS)
    _, snr_clean = peak_frequency(clean, FPS)

    rng = np.random.default_rng(0)
    noise_only = rng.normal(0, 1, int(FPS * DUR))
    _, snr_noise = peak_frequency(noise_only, FPS)

    assert snr_clean > snr_noise + 3.0


def test_band_snr_is_finite_and_ordered():
    sig = chrom_full(synth_rgb(72, noise=0.2), FPS)
    f, p = psd(sig, FPS)
    assert band_snr_db(f, p, 1.2) > band_snr_db(f, p, 3.5)


# --------------------------------------------------------------- robustness

def test_resample_handles_nan_gaps_without_shifting_frequency():
    """Dropped frames must not compress the time axis.

    Silently omitting undetected frames (rather than carrying them as gaps) makes
    20 s of video look like 18 s and biases every frequency estimate upward.
    """
    t = np.arange(0, DUR, 1 / FPS)
    x = np.sin(2 * np.pi * 1.2 * t)
    gapped = x.copy()
    gapped[100:160] = np.nan                      # 2 s of lost detection

    out = resample_to_uniform(t, gapped, FPS)
    f0, _ = peak_frequency(out, FPS)
    assert abs(f0 * 60 - 72) < 2.0


def test_resample_returns_empty_when_too_little_data():
    t = np.arange(10) / FPS
    assert resample_to_uniform(t, np.full(10, np.nan), FPS).size == 0


@pytest.mark.parametrize("bad", ["", "does_not_exist.mp4", __file__])
def test_analyze_never_raises_on_bad_input(bad):
    """Contract: any failure is a degraded result, never an exception (§2.4)."""
    r = analyze(bad, session_id="t")
    assert r.degraded_reason is not None
    assert r.rppg_quality == 0.0
    assert r.rppg_manipulation_score == 0.5, "must stay neutral, never accuse"


def test_short_signal_does_not_crash_extractors():
    tiny = synth_rgb(72, dur=0.5)
    assert chrom_full(tiny, FPS).size == 0
    assert pos_overlap(tiny, FPS).size == 0


def test_constant_input_is_handled():
    flat = np.full((int(FPS * DUR), 3), 100.0)
    sig = chrom_full(flat, FPS)
    f0, snr = peak_frequency(sig, FPS) if sig.size else (None, -np.inf)
    assert f0 is None or np.isfinite(snr)
