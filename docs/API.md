# API Reference

Base URL defaults to `http://localhost:8000`. The frontend reads
`NEXT_PUBLIC_API_BASE` to override it.

---

## REST

### `GET /health`

```json
{ "status": "ok", "real_modules": true }
```

`real_modules` reflects `DEEPGUARD_REAL_MODULES` (default `1`). When `0`, the
server returns simulated payloads — useful for frontend work without model
weights.

### `POST /upload`

Multipart upload, field name `file`.

```json
{ "session_id": "uuid", "video_path": "data/sessions/<uuid>/video.mp4" }
```

### `GET /result/{session_id}`

Returns the cached result payload for a completed session, or 404.

---

## WebSocket — `/ws/analyze/{session_id}`

Connect after `POST /upload`. The server pushes messages until it sends
`result` (or `error`), then closes. All four message types carry `type`.

### `progress`

```json
{ "type": "progress", "progress": 40, "phase": "rPPG Extraction" }
```

### `frame_data`

Emitted for every 4th decoded frame.

```json
{
  "type": "frame_data",
  "frame_idx": 12,
  "timestamp": 0.4,
  "box": [x, y, w, h],
  "width": 480,
  "height": 854,
  "mar": 0.0143,
  "audio_envelope": 0.6210,
  "coherence": [[0.31, null, ...], ...],
  "rgb_grid": [[[r,g,b], ...], ...]
}
```

**Nullable fields carry meaning — do not coerce to 0.**

| Field | `null` means |
|---|---|
| `box` | no face detected in this frame |
| `mar` | no face box, or no previous frame to difference against |
| `audio_envelope` | clip has no audio track |
| `coherence` | tracker's ~12 s window has not filled yet |
| `rgb_grid` | no face box to sample |

`box` is in **source pixel space**; scale by `width`/`height` before drawing.

### `result`

```json
{
  "type": "result",
  "verdict": "LIKELY_MANIPULATED",
  "confidence_real": 29.4,
  "confidence_fake": 70.6,
  "explanation": "…",
  "evidence_weight": 0.55,
  "min_evidence_weight": 0.11,
  "warnings": ["rppg:partial_evidence_shrink_0.29"],
  "metrics": { "rppg": {...}, "lipsync": {...}, "pixel": {...},
               "visual": {...}, "aigen": {...} },
  "attribution": { "baseline": 0.0075, "regions": {...},
                   "top_region": "nose", "frames_used": 4 },
  "waveform_decimated": [...],
  "mar_decimated": [...],
  "envelope_decimated": [...],
  "map_coherence": [[...]] 
}
```

`verdict` is one of `LIKELY_AUTHENTIC`, `LIKELY_MANIPULATED`, `UNCERTAIN`,
`INSUFFICIENT_EVIDENCE`.

**`map_coherence` is nullable.** It is `null` when no pulse map could be built.
Calling `.map()` on it without a guard crashed the report for exactly the clips
where rPPG had nothing to say.

### Metric object

Common to every channel:

| Field | Meaning |
|---|---|
| `score` | P(manipulated), 0..1. `0.5` = no opinion |
| `quality` | evidence strength, 0..1. `0` = abstained |
| `prior` | configured weight from `thresholds.yaml` |
| `contribution` | share of fused log-odds actually supplied (0..1) |
| `degraded_reason` | why the channel is degraded or non-voting |

Channel-specific: `hr`, `snr` (rppg) · `lag`, `iqr` (lipsync) · `models_used`,
`frames_scored`, `score_spread` (visual) · `frames_used` (pixel) · `raw_score`,
`model_id` (aigen).

> **`contribution`, not `prior`, is what moved the verdict.** A channel with a
> large prior and zero quality contributes exactly 0.

### `error`

```json
{ "type": "error", "message": "Failed to decode video frames." }
```
