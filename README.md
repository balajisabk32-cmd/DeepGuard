# DeepGuard

Multi-modal deepfake detection. Five independent channels, quality-weighted
fusion, and an explicit right to say **"I don't know."**

Upload-only. No live capture. No GPU required.

```
rPPG  ·  lip-sync  ·  pixel forensics  ·  frame-by-frame CNN  ·  synthetic imagery
                              ↓
                  quality-weighted log-odds fusion
                              ↓
     AUTHENTIC  ·  MANIPULATED  ·  UNCERTAIN  ·  INSUFFICIENT EVIDENCE
```

---

## Quick start

Two processes: a FastAPI backend and a Next.js frontend.

**Backend** (from the repo root):

```bash
python -m uvicorn src.pipeline.api:app --host 127.0.0.1 --port 8000
```

**Frontend** (from `deepguard-x/`):

```bash
npm install && npm run dev
```

Open **http://localhost:3000**, then click *Launch DeepGuard*.

If the backend runs on a non-default port, point the UI at it:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8001 npm run dev
```

> The backend warms EfficientNet-B7 and the SwinV2 synthetic detector at startup.
> That takes ~30 s. It is deliberate — without it the *first* upload pays the
> cold-load cost and lands around 60 s instead of ~35 s.

---

## Documentation

| Document | What's in it |
|---|---|
| **[PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md)** | End-to-end narrative, datasets, measured results, Q&A. **Start here.** |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module map, data flow, contracts, fusion maths |
| [docs/API.md](docs/API.md) | REST endpoints and the WebSocket message contract |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | `thresholds.yaml` reference and the invariants that govern it |
| [docs/MODELS.md](docs/MODELS.md) | Every model, its weights, licence, and admission status |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Setup, tests, evaluation scripts, troubleshooting |

---

## What it actually does

A video is decoded **once**; the frames and face boxes are shared by every
channel. Each channel returns two numbers — a manipulation score **and how much
evidence it had**. Fusion combines them in log-odds weighted by that evidence, so
a channel that cannot see anything cannot drag the verdict.

| # | Channel | Looks for | Votes? |
|---|---|---|---|
| 01 | **rPPG** | pulse coherence across face regions | yes |
| 02 | **Lip-sync** | speech envelope vs mouth motion, lag drift | yes |
| 03 | **Pixel forensics** | texture / warp / flicker anomalies | **no** — AUC 0.609, below gate |
| 04 | **Frame-by-frame** | per-frame splice artefacts (EfficientNet-B7) | yes |
| 05 | **Synthetic imagery** | whole-image generation (SwinV2) | **no** — compression confound |

Two channels do not vote. That is a measured decision, not an oversight — see
[Admission by measurement](#admission-by-measurement).

---

## Honest performance

| Metric | Value |
|---|---|
| TEST_VIDEOS (7 clips) | **6/7 correct**, 1 false negative |
| effb7 in-domain (DFDC) | AUC 1.000 |
| effb7 **in the wild** | AUC **0.550** ← quote this one |
| Pixel forensics (subject-grouped CV) | AUC 0.609 |
| Latency, idle 12-core CPU | median **35.5 s**, max 51.0 s |

`n=7` supports no AUC claim. **The binding constraint on this project is corpus
size, not model availability.**

---

## Admission by measurement

> No model enters the verdict without a *measured* improvement over what we
> already have.

Scoring above chance is not the bar — improving the **system** is. Things that
failed that bar and are therefore disabled or non-voting:

| Model | Measured | Outcome |
|---|---|---|
| xception | AUC 0.222 vs effb7's 0.833 | disabled |
| capsule | near-constant output across four clips | disabled |
| PhysNet | −6.78 dB SNR vs CHROM's −3.26 | excluded |
| DeepFakesON-Phys | identical score for all 13 clips | excluded |
| Pixel forensics | fusing it moved AUC **0.670 → 0.639** | reports, does not vote |
| Synthetic imagery | AUC **0.000** on video (compression confound) | reports, does not vote |

Full detail and the isolating experiments are in
[PROJECT_EXPLAINED.md §6](PROJECT_EXPLAINED.md).

---

## Design rules

These are enforced in code and tests, not just aspirations:

1. **Absent evidence ≠ zero evidence.** A channel that did not run must not
   render as a `0.000` score. The UI shows non-voting channels greyed with the
   measured reason.
2. **No fabricated data reaches the user.** Placeholder sine waves, checkerboard
   heatmaps, and hardcoded vitals have all been removed; a nullable field renders
   as "unavailable", never as a plausible number.
3. **Quality gates influence, not the prior.** Contribution = prior × evidence
   quality, and it is the only number that moved the verdict.
4. **Abstention is a correct output.** `UNCERTAIN` and `INSUFFICIENT_EVIDENCE`
   are distinct verdicts with distinct causes and must never be merged.

---

## Repository layout

```
├── config/thresholds.yaml     every threshold, one auditable file
├── src/
│   ├── common/contracts.py    pydantic result contracts
│   ├── pipeline/              decode → detect → api (FastAPI + WebSocket)
│   ├── rppg/                  pulse extraction, PPG map, coherence tracker
│   ├── lipsync/               audio demux, envelope/MAR correlation
│   ├── visual/                CNN registry, pixel forensics, XAI, aigen
│   └── fusion/scorer.py       log-odds fusion + decision gate
├── deepguard-x/               Next.js frontend
├── tests/                     pytest suite
├── models/                    vendored weights (gitignored)
└── results/                   measurement artefacts (JSON)
```

---

## Licence note

Model weights carry their own licences and are **not** uniformly permissive.
`Organika/sdxl-detector` (Channel 05) is **CC-BY-NC-3.0 — non-commercial only**.
See [docs/MODELS.md](docs/MODELS.md) before any commercial use.
