# DeepGuard // Technical Hints & Developer Guide

This document contains technical hints, developer notes, signal processing rules, and troubleshooting guidance for **DeepGuard** — specifically focused on the rPPG spatial coherence visualizer (`src/rppg/webcam_heatmap.py`), OpenCV camera captures, and biosignal fusion modules.

---

## 1. Spatial Coherence Heatmap Diagnostic Tool (`webcam_heatmap.py`)

The spatial pulse-coherence viewer renders live biosignal evidence across a 6×5 facial grid.

### CLI Execution Modes

```bash
# 1. Live camera capture (probing default webcams)
python -m src.rppg.webcam_heatmap

# 2. Analyze a video file interactively
python -m src.rppg.webcam_heatmap --video REAL.mp4

# 3. Headless annotation (saves output video without GUI window)
python -m src.rppg.webcam_heatmap --video REAL.mp4 --out annotated.mp4 --headless
```

### Keyboard Shortcuts
- `q` : Exit the visualization loop.
- `s` : Save a PNG snapshot of the current annotated frame to disk.

---

## 2. Hardware & Camera Technical Hints

### Windows OpenCV Backend Selection
When opening webcams on Windows, OpenCV can fail or hang with Media Foundation (`MSMF`) error `-1072875772`.
- **Hint**: Always probe `cv2.CAP_DSHOW` (DirectShow) first before falling back to `cv2.CAP_MSMF` or `cv2.CAP_ANY`:
  ```python
  cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
  ```

### Camera Auto-Exposure & White Balance Locking
Auto-exposure continuously renormalizes pixel intensity, injecting low-frequency noise into the human heart rate frequency band (**0.7–4.0 Hz / 42–240 BPM** — `HR_BAND` in `signal_core.py`, `band_hz` in `config/thresholds.yaml`).
- **Hint**: Lock exposure and white balance at capture time:
  ```python
  cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
  cap.set(cv2.CAP_PROP_AUTO_WB, 0)
  ```

---

## 3. Signal Core & Mathematical Technical Hints

### Why Spatial Coherence over Per-Patch SNR?
- **Per-patch SNR** fails because dark hair, shadows, or high compression produce low SNR on authentic faces, incorrectly flagging authentic regions as fakes.
- **Cross-Region Spatial Coherence** measures whether facial patches beat *in phase with each other*. A face-swap blend boundary disrupts phase consistency across regions (forehead vs. cheeks), providing a robust detection seam.

### Zero-Phase Filtering Requirement
- Standard causal filters (`scipy.signal.sosfilt`, `scipy.signal.lfilter`) introduce frequency-dependent phase delays. Because rPPG coherence measures phase agreement across spatial patches, causal filtering manufactures artificial phase dispersion.
- **Rule**: Always use zero-phase forward-backward filtering.
- **What the code actually does** (`signal_core._bandpass`): `butter(3, …, btype="bandpass")` returning `b, a`, then `scipy.signal.filtfilt`. Both `filtfilt` (b/a form) and `sosfiltfilt` (SOS form) are zero-phase — either satisfies the rule. There is no SOS anywhere in `src/`; if you raise the filter order, switch to `sosfiltfilt`, which is more numerically stable at higher orders.

### Single Extractor Consistency
- Do not mix different rPPG extraction algorithms (e.g., CHROM in patch A and POS in patch B). CHROM and POS have different inherent phase offsets. Mixing them manufactures anti-correlation between physically in-phase patches.

### Non-Uniform Timestamp Axis
- Variable Frame Rate (VFR) inputs or dropped frames corrupt pulse frequency estimation if treated as uniform frames.
- **Rule**: Store exact Presentation Timestamps (PTS) per frame and resample onto a uniform grid with `resample_to_uniform(t, x, fs)`.
- **The grid is NOT a fixed 30 Hz.** `fs` is the *effective* rate measured from the real timestamps — `fs = (len(frames) - 1) / (t[-1] - t[0])`. Hardcoding 30 Hz would reintroduce exactly the VFR bug this section warns about.
- `nominal_fps` in `PreprocessResult` is metadata for display only. Never use `cv2.CAP_PROP_FPS` as a time axis: on a VFR file it reports a single made-up figure and indexing frames as `i/fps` rebuilds a uniform grid that never existed.

---

## 4. Quality Gating & Polarity Rules

### Score Polarity Invariant
To prevent integration errors, field naming strictly follows polarity conventions:
- `*_manipulation_score` $\in [0, 1]$ : Higher value indicates **higher suspicion of manipulation**.
- `*_quality` $\in [0, 1]$ : Higher value indicates **higher signal trustworthiness**.

### Evidence-Graded Contribution (replaced the old hard gate)

There is **no single SNR cutoff** that flips the score to neutral. A hard cutoff made rPPG all-or-nothing, and because almost all real footage lands below it, the modality silently dropped out of every verdict. It now contributes in proportion to its evidence:

```
score = 0.5 + (raw_score - 0.5) * shrink(quality)
```

| `rppg_quality` | Behaviour | `degraded_reason` |
|---|---|---|
| `<= min_quality_for_evidence` (0.05) | full abstention, score forced `0.50` | `insufficient_pulse_snr` |
| between 0.05 and 0.35 | shrunk toward `0.50` proportionally | `partial_evidence_shrink_<f>` |
| `>= quality_trust_full` (0.35) | used at face value | `None` |

Fewer than 2 usable ROIs also forces `0.50`, with `too_few_usable_rois`.

**`quality_snr_floor_db: -3.0` is not a gate.** It is the lower anchor of the SNR→quality mapping:

```
quality = clip((snr_db - quality_snr_floor_db) / (quality_snr_ceil_db - quality_snr_floor_db), 0, 1)
```

so −3 dB maps to quality 0 and +9 dB to quality 1. `min_quality_for_scoring: 0.30` is retained only as the "trusted" label consumers read; it no longer gates the score.

---

## 5. Diagnostic Troubleshooting Guide

| Symptom / Error | Root Cause | Technical Fix / Action |
|---|---|---|
| `patches=0` & `snr=-99.00dB` | Face not detected, OR per-patch skin fraction is below `rppg.map.min_skin_fraction` (0.35). | Ensure the face is well-lit and frontal. Tune the `YCrCb` mask, or lower `min_skin_fraction` in `config/thresholds.yaml` — **not** in the modules. Both `ppgmap.build_stmap` and `webcam_heatmap` read it from config via `ppgmap.min_skin_fraction()`; hardcoding it per module previously let them drift to 0.35 and 0.30, so the heatmap rendered patches the detector had discarded. |
| Camera fails to open (`MSMF error`) | OpenCV Windows Media Foundation driver conflict. | Force DirectShow backend: `cv2.VideoCapture(0, cv2.CAP_DSHOW)`. |
| Docker container cannot access webcam | Docker Desktop on Windows/macOS does not support host USB camera passthrough natively. | DeepGuard architecture is **upload-only** in Docker. For live camera testing, run natively in Python environment. |
| Coherence heatmap is uniformly **amber** (≈ 0.0) | **The usual case.** No recoverable pulse — patches are uncorrelated noise, so coherence sits near zero. Compressed social-media video reliably does this: both Tom Cruise test clips gave mean coherence +0.31 and +0.39 at roughly −2 dB SNR. | Not a bug. Record at 1080p, CRF ≤ 23, lamp on the face, subject still for 12s, then re-check `map_mean_patch_snr_db`. Separation needs roughly **≥ 4 dB**. |
| Coherence heatmap shows all **red** (≈ −1.0) | Genuinely anti-phase patches. Rare and worth investigating — random noise gives amber, not red. Most likely a polarity fault (mixed CHROM/POS extractors across the grid) rather than anything physiological. | Confirm one extractor is used for every patch. `tests/test_ppgmap.py::test_coherent_pulse_gives_positive_patch_correlation` guards this. |
| Heatmap green but verdict says otherwise | Rapid head motion or changing illumination inside the 12s window. | Keep the subject still for the full window; avoid flickering LEDs. |
| High lag jitter in lip-sync | Audio stream missing or misaligned start offset. | Pass `av_start_offset_sec` from `PreprocessResult` contract to prevent stream offset corruption. |

---

## 6. Verification & Test Commands

```bash
# Run unit test suite
pytest

# Test contract invariants specifically
pytest tests/test_contracts.py

# Test rPPG spatial patch map core
pytest tests/test_ppgmap.py

# Test fusion scoring engine
pytest tests/test_fusion.py
```
