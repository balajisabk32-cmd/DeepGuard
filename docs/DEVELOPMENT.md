# Development

## Setup

```bash
python -m pip install -r requirements.txt
```

```bash
cd deepguard-x && npm install
```

Model weights live in `models/` and are **gitignored** — GitHub rejects blobs
over 100 MB and `deepfake_efficientnet_b7.pt` is 268 MB. Obtain them separately
and drop them in. The synthetic-imagery model downloads itself from the Hugging
Face hub on first use.

The pipeline degrades rather than crashing when a weight file is missing: the
affected channel returns `degraded_reason="weights_unavailable"` and abstains.

---

## Running

```bash
python -m uvicorn src.pipeline.api:app --host 127.0.0.1 --port 8000
```

```bash
cd deepguard-x && npm run dev
```

Single-clip CLI, no server needed:

```bash
python -m src.pipeline.detect TEST_VIDEOS/FAKE2.mp4
```

---

## Tests

```bash
python -m pytest -q
```

The suite is small but load-bearing. Notable invariants it protects:

| Test | Guards |
|---|---|
| `test_fusion.py::test_single_modality_evidence_ceiling` | `min_evidence_weight < min(prior share)` — asserts the **relationship**, not a number |
| `test_fusion.py::test_every_configured_modality_fuses` | reads the channel set from config, so adding a channel extends it rather than breaking it |
| `test_fusion.py::test_insufficient_evidence_takes_priority` | a maximally suspicious score built on nothing must abstain, not accuse |
| `test_rppg.py` | zero-phase filtering, HR estimation accuracy across a BPM sweep |

> Tests that assert relationships survive refactors; tests that assert constants
> silently stop testing anything. Both fusion invariants above were caught by
> relationship assertions after a config change made the constants meaningless.

---

## Evaluation scripts

Run from the repo root. All write JSON into `results/`.

| Script | Purpose |
|---|---|
| `scripts/full_pipeline_test.py` | All channels over `TEST_VIDEOS`, with per-stage latency |
| `scripts/calibrate_pixel.py` | Fit the pixel-forensics calibration (subject-grouped CV) |
| `scripts/pixel_admission.py` | Does pixel forensics improve on effb7? (admission test) |
| `scripts/eval_test_videos.py` | Per-model scores over the corpus |
| `scripts/sdfvd_calibrate.py` | Feature extraction over SDFVD2.0 |
| `scripts/sdfvd_effb7.py` | effb7 scores over SDFVD2.0 |
| `scripts/visualize_detection.py` | Render an annotated output video |

### Evaluation rules

1. **Group your splits.** SDFVD2.0 ships 8 augmented copies per clip and pairs
   `v{N}` (real) with `vs{N}` (fake). A random split leaks both ways and inflated
   AUC from 0.711 to 0.790.
2. **Report leave-one-out**, not the fitted number. Learned meta-fusion measured
   0.700 fitted and **0.300** LOO on 11 clips.
3. **In-domain AUC is not performance.** effb7 scores 1.000 on DFDC crops and
   0.550 in the wild.

---

## Troubleshooting

**`OMP Error #15` / process abort**
Torch and Anaconda's SciPy both link `libiomp5md.dll`. Any SciPy call reaching
LAPACK — notably `scipy.signal.detrend(type="linear")` — aborts the process.
`src/rppg/signal_core.py` uses a closed-form `detrend_linear()` for this reason.
Do not reintroduce SciPy detrending.

**Report renders for some clips, crashes on others**
`map_coherence` is `null` when no pulse map could be built (measured: 1 clip in
7). Guard nullable payload fields — never call `.map()` unguarded.

**Backend answers `/health` but serves stale results**
A uvicorn from an earlier session may still hold the port. It responds perfectly
while running old code. Confirm the PID, or run on a different port and set
`NEXT_PUBLIC_API_BASE`.

**First analysis takes ~90 s, later ones ~35 s**
Cold TorchScript deserialisation. The FastAPI `startup` hook warms both models;
if you bypass the server, the first call pays it.

**LIPINC returns ~0.000 for everything**
Face alignment is out of distribution. It needs the upstream dlib preprocessing;
a MediaPipe substitute silently produces garbage. Input **width** matters —
400px vs 800px inverted one clip from 0.974 to 0.029.

---

## Conventions

- **No magic numbers in code.** Thresholds belong in `config/thresholds.yaml`.
- **Score polarity is uniform**: `*_manipulation_score` is always P(manipulated).
- **`analyze()` never raises.** Return a degraded result with `degraded_reason`.
- **Abstain with `score=0.5, quality=0.0`.** Never `score=0.0` — that is a
  confident vote for "authentic".
- **Record measurements where the decision lives.** Disabled models carry their
  measured AUC in a `note` field so the reason cannot drift from the code.
- **Never ship a number you did not measure.** Placeholder waveforms and default
  vitals all shipped once and all had to be removed.
