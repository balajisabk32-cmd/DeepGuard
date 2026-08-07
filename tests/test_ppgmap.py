"""PPG spatial-temporal map tests — FakeCatcher/DeepRhythm features.

The polarity test is the important one. It guards a bug that produced a
plausible-looking but exactly-backwards result, and that no shape or smoke test
would have caught.
"""

import numpy as np
import pytest

from src.fusion import load_thresholds
from src.rppg.ppgmap import analyze_map, map_manipulation_score

CFG = load_thresholds()
ROWS, COLS, FPS, DUR = 6, 5, 30.0, 14.0


def make_stmap(phase_map=None, freq_map=None, gain=6.0, noise=0.4, seed=0):
    """Synthetic STMap: (frames, rows, cols, 3) with a controllable pulse per patch."""
    rng = np.random.default_rng(seed)
    n = int(FPS * DUR)
    t = np.arange(n) / FPS
    phase_map = np.zeros((ROWS, COLS)) if phase_map is None else phase_map
    freq_map = np.full((ROWS, COLS), 1.2) if freq_map is None else freq_map

    st = np.empty((n, ROWS, COLS, 3))
    amp = np.array([0.5, -5.0, -1.5])
    base = np.array([150.0, 100.0, 100.0])
    for r in range(ROWS):
        for c in range(COLS):
            p = np.sin(2 * np.pi * freq_map[r, c] * t + phase_map[r, c])
            drift = 8 * np.sin(2 * np.pi * 0.05 * t)
            st[:, r, c, :] = (base + gain * amp * p[:, None]
                              + rng.normal(0, noise, (n, 3)) + drift[:, None])
    return st, t


def test_coherent_pulse_gives_positive_patch_correlation():
    """POLARITY REGRESSION GUARD.

    A pulse identical in every patch must make patches POSITIVELY correlated.

    This failed before: analyze_map used per-patch `best_extraction`, so some
    patches came back via CHROM (S = Xf - alpha*Yf) and others via POS
    (h = S0 + beta*S1). Those conventions can carry opposite sign for the same
    physical pulse, so the grid mixed polarities and manufactured anti-correlation
    between patches that were perfectly in phase. Symptom: corr_p25 went to -0.96
    and got WORSE as the pulse got stronger.

    If this test fails, someone reintroduced per-patch method selection.
    """
    st, t = make_stmap(gain=6.0)
    res = analyze_map(st, t, FPS)
    assert res.n_patches_used >= 10
    assert res.corr_p25 > 0.5, (
        f"coherent pulse gave corr_p25={res.corr_p25:+.3f}; "
        "patch extractors are mixing polarity conventions"
    )


def test_stronger_pulse_increases_agreement():
    """Correlation must move toward +1 with SNR, never away from it."""
    lo = analyze_map(*make_stmap(gain=3.0), FPS).corr_p25
    hi = analyze_map(*make_stmap(gain=12.0), FPS).corr_p25
    assert hi > lo


def test_authentic_scores_lower_than_seam_discontinuity():
    """The core claim: a phase+rate seam across the grid scores as manipulated."""
    auth = analyze_map(*make_stmap(gain=6.0), FPS)

    ph = np.zeros((ROWS, COLS)); ph[2:, :] = np.pi * 0.8
    fr = np.full((ROWS, COLS), 1.2); fr[2:, :] = 1.1
    swap = analyze_map(*make_stmap(phase_map=ph, freq_map=fr, gain=6.0), FPS)

    s_auth = map_manipulation_score(auth, CFG)
    s_swap = map_manipulation_score(swap, CFG)
    assert s_swap > s_auth + 0.25, f"authentic={s_auth:.3f} swap={s_swap:.3f}"
    assert s_auth < 0.35, "authentic clip must not read as manipulated"


def test_temporal_hr_jump_is_low_for_a_steady_pulse():
    """Paper §3.3: real hearts drift smoothly. A steady synthetic pulse must not
    look like an unnatural HR jump."""
    res = analyze_map(*make_stmap(gain=6.0), FPS)
    assert res.hr_temporal_jump_bpm < 6.0


def test_map_reports_more_patches_than_the_three_roi_path():
    res = analyze_map(*make_stmap(gain=6.0), FPS)
    assert res.n_patches_used > 3
    assert res.coherence_map.shape == (ROWS, COLS)


def test_degrades_cleanly_on_empty_input():
    st = np.full((100, ROWS, COLS, 3), np.nan)
    t = np.arange(100) / FPS
    res = analyze_map(st, t, FPS)
    assert res.degraded_reason is not None
    assert res.n_patches_used == 0


def _margin(gain: float, noise: float) -> tuple[float, float]:
    """Returns (mean patch SNR of the authentic case, authentic->swap score margin)."""
    auth = analyze_map(*make_stmap(gain=gain, noise=noise), FPS)
    ph = np.zeros((ROWS, COLS)); ph[2:, :] = np.pi * 0.8
    swap = analyze_map(*make_stmap(phase_map=ph, gain=gain, noise=noise), FPS)
    return (float(auth.mean_patch_snr_db),
            map_manipulation_score(swap, CFG) - map_manipulation_score(auth, CFG))


def test_separation_degrades_with_snr():
    """Documents the operating requirement rather than asserting a guessed cutoff.

    Separation is a function of SNR, so the honest test is that it MOVES the right
    way — clean input separates, buried input does not. The measured threshold on
    real footage (mean patch SNR around 4 dB) belongs in the report, not hard-coded
    here where a synthetic noise floor decides it.
    """
    snr_clean, margin_clean = _margin(gain=6.0, noise=0.4)
    snr_buried, margin_buried = _margin(gain=0.3, noise=12.0)

    assert snr_clean > snr_buried, "SNR estimate must track actual signal strength"
    assert margin_clean > 0.25, f"clean input failed to separate (margin {margin_clean:.3f})"
    assert margin_clean > margin_buried, (
        "separation must degrade as the pulse is buried; "
        f"clean={margin_clean:.3f} buried={margin_buried:.3f}"
    )
