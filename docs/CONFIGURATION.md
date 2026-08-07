# Configuration

Everything tunable lives in **`config/thresholds.yaml`**. No magic numbers in
code — a threshold in a source file is a threshold nobody can audit.

Load it with `src.fusion.load_thresholds()`.

---

## `decision`

| Key | Value | Meaning |
|---|---|---|
| `authentic_max_p` | 0.35 | p ≤ this → `LIKELY_AUTHENTIC` |
| `manipulated_min_p` | 0.65 | p ≥ this → `LIKELY_MANIPULATED` |
| `min_evidence_weight` | 0.11 | below this → `INSUFFICIENT_EVIDENCE` |

The band between the two probability thresholds is `UNCERTAIN`. Widening it makes
the system abstain more and err less.

### ⚠ `min_evidence_weight` is derived, not chosen

For a lone usable channel, `evidence_weight` equals its prior **share**:

```
min_evidence_weight  <  min(prior) / Σ(priors)
```

**Re-derive it whenever a channel is added** — a new prior dilutes every existing
share:

| channels | Σ priors | smallest share | floor |
|---|---|---|---|
| 3 | 1.00 | 0.300 | 0.20 |
| 4 | 1.00 | 0.150 | 0.12 |
| 5 | 1.15 | 0.130 | **0.11** |

Adding the 4th channel left the floor at 0.20 while the smallest share fell to
0.15 — which would have made the system abstain *more* because a non-voting
channel was added. `tests/test_fusion.py` asserts the relationship and catches it.

---

## `fusion.prior_weights`

```yaml
prior_weights: {rppg: 0.25, lipsync: 0.25, visual: 0.35, pixel: 0.15, aigen: 0.15}
```

Relative importance **when evidence is equal**. `visual` leads because it is the
only channel with trained weights and a measured baseline.

Priors do **not** determine influence — `quality` does. `pixel` and `aigen` carry
priors so they appear in the output, but both report quality 0 and therefore
contribute exactly zero.

---

## `rppg`

| Key | Value | Notes |
|---|---|---|
| `band_hz` | `[0.7, 4.0]` | 42–240 BPM |
| `min_window_sec` | 12.0 | below this, frequency resolution is too coarse |
| `quality_snr_floor_db` / `ceil_db` | −3.0 / 9.0 | maps SNR → quality |
| `min_quality_for_evidence` | 0.05 | at or below → full abstention, score forced to 0.5 |
| `quality_trust_full` | 0.35 | at or above → score used at face value |

Between the two quality bounds the score is **shrunk toward 0.5**
proportionally. A hard cutoff made rPPG contribute nothing on real compressed
footage — which is almost all footage.

### `rppg.map`

| Key | Value | Notes |
|---|---|---|
| `grid` | `[6, 5]` | 30 patches → up to 435 pairwise comparisons |
| `min_skin_fraction` | 0.35 | **shared** by the offline detector and the live viewer |
| `authentic_corr` | 0.70 | reference point for a genuine face |
| `hr_jump_norm_bpm` | 12.0 | temporal HRV dimension |

`min_skin_fraction` lives here because two consumers must agree. They previously
hardcoded 0.35 and 0.30 independently, so the heatmap rendered patches green that
`analyze()` was silently discarding — the picture disagreed with the verdict.

---

## `lipsync`

| Key | Value | Notes |
|---|---|---|
| `envelope_band_hz` | `[1.0, 8.0]` | syllable rate |
| `window_sec` / `hop_sec` | 1.0 / 0.5 | correlation windows |
| `max_lag_ms` | 300 | search range |
| `reliable_peak_ncc` | 0.30 | NCC below this means the lag is noise |
| `lag_iqr_floor_ms` / `ceil_ms` | 40 / 160 | maps drift → score |
| `weights` | `iqr 0.50, ncc 0.30, offset 0.20` | IQR dominates |

`lag_iqr` carries the most weight because a *constant* offset is an encoding
property, while manipulation shows up as **wandering** alignment.

---

## `visual`

| Key | Value | Notes |
|---|---|---|
| `input_size` | 224 | xception spec; effb7 overrides to 380 |
| `fake_index` | 1 | **verified** against upstream inference code |
| `crop_margin` | 1.3 | square crop around the face box |
| `max_frames` | 32 | see warning below |
| `min_face_px` | 64 | below this the input is upscaled blur |
| `aggregation` | `p90` | median 0.833, p90 **0.889**, max 0.778 |
| `calibration.a` / `.b` | 0.3613 / 0.5313 | `sigmoid(a·logit(p90) + b)` |

### ⚠ `max_frames` and `calibration` are coupled

The calibration constants were fitted **at 32 frames with p90 aggregation**.
Changing the frame count changes the p90 distribution and invalidates them.
Cutting frames for latency requires refitting — it is a measurement task, not a
config tweak.

`max` aggregation looked perfect at 24 frames (AUC 1.000) and collapsed to 0.778
at 32. It was reading sampling noise, so it is not used.

**Honest performance of this calibration:** fitted on all 9 clips → 6/9 confident,
0 wrong, 3 abstain. **Leave-one-out → 5/9 confident, 1 wrong, 4 abstain.**
Believe the LOO number.

---

## `ingest`

| Key | Value |
|---|---|
| `min_duration_sec` | 4.0 |
| `max_decode_sec` | 300.0 |
| `analyze_first_sec` | 20.0 |
| `max_upload_mb` | 100 |
| `min_face_detection_rate` | 0.6 |

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `DEEPGUARD_REAL_MODULES` | `1` | `0` serves simulated payloads (frontend work without weights) |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | frontend → backend base URL |
| `TF_USE_LEGACY_KERAS` | `1` | set internally; TF-based models need Keras 2 |

---

## Calibration artefacts

| File | Produced by | Consumed by |
|---|---|---|
| `results/pixel_calibration.json` | `calibrate_pixel.py` | `src/visual/pixel_forensics.py` |
| `results/pixel_admission.json` | `pixel_admission.py` | documentation only |

The pixel channel reads `admitted` from its calibration file rather than a
hardcoded flag, so re-calibrating on better data can enable voting **without a
code change**.
