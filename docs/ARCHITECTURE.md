# Architecture

How the pieces fit, and why they are shaped this way.

---

## 1. Data flow

```
POST /upload ──► data/sessions/<id>/video.mp4
                          │
                 WS /ws/analyze/<id>
                          │
              ┌───────────▼────────────┐
              │ decode_clip()          │  src/pipeline/decode.py
              │  • PyAV, true PTS      │  ONE decode + ONE face pass,
              │  • ≤20 s analysed      │  reused by every channel
              │  • per-frame face box  │
              └───────────┬────────────┘
                          │  Clip(frames, boxes, t)
        ┌─────────┬───────┼────────┬──────────┐
        ▼         ▼       ▼        ▼          ▼
     rppg     lipsync   pixel   visual      aigen
   analyze()  analyze() analyze() analyze()  analyze()
        │         │       │        │          │
        └────── (score, quality) per channel ─┘
                          │
              ┌───────────▼────────────┐
              │ fusion.score()         │  src/fusion/scorer.py
              │  log-odds + gate       │
              └───────────┬────────────┘
                          ▼
              region_attribution()       src/visual/explain.py  (XAI)
                          ▼
                   result payload → UI
```

**The single decode is the central optimisation.** Five channels each doing
their own decode and face detection would multiply the most expensive
non-inference step by five.

---

## 2. Module map

| Module | Responsibility |
|---|---|
| `src/pipeline/decode.py` | PyAV decode, true PTS timestamps, per-frame face boxes |
| `src/pipeline/detect.py` | Orchestrates all five channels → fusion. CLI entry point |
| `src/pipeline/api.py` | FastAPI app: upload, WebSocket streaming, result payload, warmup |
| `src/rppg/signal_core.py` | CHROM/POS projection, zero-padded FFT, `detrend_linear` |
| `src/rppg/ppgmap.py` | 6×5 spatio-temporal map, patch coherence |
| `src/rppg/backends.py` | Face-tracking backends (OpenCV Haar, MediaPipe) |
| `src/rppg/webcam_heatmap.py` | `CoherenceTracker` — rolling coherence, shared by live stream and viewer |
| `src/lipsync/audio_io.py` | PyAV audio demux, container start-offset correction |
| `src/lipsync/analyze.py` | Envelope × mouth-motion cross-correlation, lag IQR |
| `src/visual/models.py` | `DetectorSpec` registry, preprocessing, thread tuning |
| `src/visual/detector.py` | Frame-by-frame scoring, p90 aggregation, calibration |
| `src/visual/pixel_forensics.py` | Classical region features + calibrated score |
| `src/visual/aigen.py` | Synthetic-imagery channel (full frames) |
| `src/visual/explain.py` | Occlusion-based region attribution (XAI) |
| `src/fusion/scorer.py` | Log-odds fusion, evidence weight, decision gate |
| `src/common/contracts.py` | Pydantic result contracts (strict) |

---

## 3. The channel contract

Every channel obeys the same three rules. They are what make fusion sound.

**1. Return two numbers.**

```python
<channel>_manipulation_score  # P(manipulated), 0..1. 0.5 == "no opinion"
<channel>_quality             # evidence strength, 0..1. 0 == "I saw nothing"
```

Score polarity is uniform: **higher always means more likely manipulated.**

**2. Never raise.** Every `analyze()` wraps its body and returns a degraded
result with `degraded_reason` set. A channel failing must never take down the
verdict — the other four still have something to say.

**3. Report abstention explicitly.** A channel with nothing to contribute
returns score `0.5` **and** quality `0.0`. Returning `0.0` as a score would be a
confident vote for "authentic".

---

## 4. Fusion

```python
z = Σ (wᵢ / w_max) · qualityᵢ · logit(scoreᵢ)
p = sigmoid(z)
evidence_weight = Σ (wᵢ · qualityᵢ) / Σ wᵢ
```

### Why log-odds, not a weighted mean

Two independent channels each reporting 0.7 should produce **more** confidence
than 0.7. A weighted mean of probabilities cannot express accumulation of
independent evidence; adding log-odds is the naive-Bayes form that can.

### Why quality multiplies the logit

It scales the *strength* of a channel's claim rather than blending its number
toward 0.5. A channel with quality 0 contributes a term of exactly zero — it is
as if it never spoke, which is precisely the intent.

### The decision gate

Evaluated in this order:

```
evidence_weight < min_evidence_weight  →  INSUFFICIENT_EVIDENCE
p ≤ authentic_max_p   (0.35)           →  LIKELY_AUTHENTIC
p ≥ manipulated_min_p (0.65)           →  LIKELY_MANIPULATED
otherwise                              →  UNCERTAIN
```

The evidence check comes **first** deliberately: a maximally suspicious score
built on nothing must abstain, not accuse.

`UNCERTAIN` and `INSUFFICIENT_EVIDENCE` answer different questions —
*"the channels disagreed"* vs *"the channels could not see"* — and the UI renders
them as separate panels.

---

## 5. The evidence-floor invariant

For a **lone** usable channel, `evidence_weight` equals that channel's prior
**share** of the total. So the floor must sit below the smallest share, or a clip
carrying one perfect channel abstains forever.

```
min_evidence_weight  <  min(prior) / Σ(priors)
```

**This must be re-derived every time a channel is added**, because a new prior
dilutes every existing share:

| channels | Σ priors | smallest share | floor |
|---|---|---|---|
| 3 | 1.00 | 0.300 | 0.20 |
| 4 | 1.00 | 0.150 | 0.12 |
| 5 | 1.15 | 0.130 | **0.11** |

Adding the 4th channel silently broke this: the floor stayed at 0.20 while the
smallest share fell to 0.15, which would have made the system abstain *more*
because a non-voting channel was added. `tests/test_fusion.py` asserts the
**relationship**, not the number, which is the only reason it was caught.

---

## 6. Explainability

Occlusion sensitivity over seven face regions. Grad-CAM is unavailable because
EfficientNet-B7 ships as a frozen **TorchScript** graph with no hookable named
layers.

```
baseline = p90(model(frames))
Δ(region) = baseline − p90(model(frames with region masked))
```

The regions are shared with `pixel_forensics` and overlap the rPPG grid, so every
channel answers "where?" on one coordinate system. Cost is 4 baseline + 28
occluded forwards.

A large Δ means the region drove **this score** — it is not evidence that the
region was manipulated.

---

## 7. Frontend

Next.js App Router, two routes:

| Route | Purpose |
|---|---|
| `/` | Landing. **Zero network calls**, no analysis widget |
| `/detect` | The product: upload, live stream, report |

| Component | Role |
|---|---|
| `InferenceSandbox` | Upload, WebSocket lifecycle, canvas overlay, live cards |
| `FeatureMap` | Live per-patch RGB / pulse-residual grid + region map |
| `ForensicReport` | Full report overlay |
| `ChannelPanel` | Five channels, abstention ledger, XAI attribution |
| `lib/api.ts` | Typed payloads + WebSocket client |

The canvas overlay scales face boxes from **source pixel space** to the rendered
element, which is why every `frame_data` message carries `width`/`height`.
Without them the overlay silently used a 320×240 default and drew off-screen.

---

## 8. Threading and latency

| Stage | Cost |
|---|---|
| decode | 4–9 s |
| rppg + lipsync + pixel | < 1 s combined |
| **effb7** (32 frames @ 380px) | **~21 s** |
| aigen (8 full frames) | ~3 s |
| **XAI attribution** (32 forwards) | **~16 s** |
| **total** | **median 35.5 s** |

Torch defaults to `cpu_count − 2` threads; `_tune_threads()` raises it to all
cores. Measured on effb7, batch of 8: **0.99 s/img → 0.66 s/img**.

Model loading (~30 s) happens once in the FastAPI `startup` hook. Cutting
`visual.max_frames` would be the obvious latency win but is **not** a free
change: the p90 aggregation and the fitted `a`/`b` calibration constants are
tied to the frame count and would need refitting.
