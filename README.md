# DeepGuard // Multi-Modal Deepfake & Manipulation Detection (v2.1)

**DeepGuard** is a quality-weighted multi-modal video manipulation detector fusing **cross-region blood flow consistency (rPPG)** with **speech-to-lip biomechanical alignment (VAD-gated cross-correlation)**.

Upload-only. No live capture. It abstains when the evidence is weak, and says so.

---

## 0. Quick Start (Offline Docker)

```bash
docker compose up
```

Open [http://localhost:8501](http://localhost:8501).

Local development without Docker:

```bash
python -m pip install -r requirements.txt
```

```bash
pytest
```

---

## 1. Project Structure

```
├── config/
│   └── thresholds.yaml         # Every threshold, one auditable file. No magic numbers in code.
├── data/
│   ├── corpus/                 # Evaluation clips + MANIFEST.csv (media is gitignored)
│   └── sessions/               # Per-session scratch, purged on startup
├── eval/
│   └── run.py                  # CP2/CP3 evaluation harness
├── models/                     # Vendored weights, baked into the image (gitignored)
├── src/
│   ├── common/contracts.py     # Executable Pydantic v2 schemas — the shared contract
│   ├── rppg/                   # Biosignal extraction (CHROM / POS), 3-ROI consistency
│   ├── lipsync/                # Speech-to-lip alignment
│   ├── fusion/                 # Quality-weighted scorer + explanation generator
│   ├── pipeline/               # ffmpeg normalize, single-pass landmarking, FastAPI
│   └── ui/                     # Streamlit explainability dashboard
├── tests/                      # Contract, fusion, and §6.2 edge-case tests
├── plan.md                     # 24-Hour Execution Plan (v2.1)
├── Dockerfile                  # Single image, two entrypoints
└── docker-compose.yml          # api + ui services
```

---

## 2. Threat Model Coverage

**This table is a design claim, not a measurement.** It states which modality is *expected* to catch which attack class and why. Measured results live in `results/cp3.md` once the CP3 gate has run — until that file exists, nothing here has been evaluated.

| Attack class | Example tools | rPPG consistency | Lip-sync alignment | Status |
|---|---|---|---|---|
| **Face swap** | SimSwap, FaceFusion | Expected to catch — mask boundary breaks cross-region pulse phase | Weak (lips still match audio) | In corpus (Class C, 6 clips) |
| **Lip-sync / dub** | Wav2Lip | Weak (pulse largely intact) | Expected to catch — lag IQR drifts window to window | In corpus (Class D, 4 clips) |
| **Reenactment** | Face2Face-style | Partial | Partial | **Not evaluated** (Class E, best-effort, ≤2 clips) |
| **Synthetic video** | Text-to-video models | Expected to catch — no coherent pulse | Varies | **Not evaluated** (Class E, best-effort, ≤2 clips) |
| **Voice clone only** | RVC-style | No | No — lips genuinely match the cloned audio | **Out of scope, declared** |

Why fuse at all: no single modality covers both of the first two rows.

---

## 3. Disclosed Limitations & Roadmap

- **Demonstration corpus, not a benchmark.** ~28 curated clips across a handful of subjects. Results carry wide confidence intervals and are reported with them.
- **Compression sensitivity.** rPPG micro-signal degrades under heavy re-encoding. Normalization does not recover it — transcoding is normalization, not a fix. Measured and disclosed, not solved.
- **rPPG demographic variance.** Accuracy is known to vary with skin tone due to melanin absorption. **This is disclosed, not mitigated.** Fusing a second independent modality reduces reliance on rPPG alone, but does not correct the bias, and our corpus is far too small to quantify it. Roadmap item #2.
- **Full-face swaps weaken the core signal.** Cross-region comparison works because swap masks typically exclude the forehead. A mask covering all three ROIs removes the contrast.
- **Audio-only deepfakes.** Out of scope. Roadmap item #1.

---

## 4. Submission

> **H0-A deliverable — Role 4, due T+0:15.** Fill this in from the actual event rules before any code is written.

| Item | Value |
|---|---|
| Event / track | _TBD_ |
| Submission deadline (local time) | _TBD_ |
| Required artifacts | _TBD_ |
| Demo time limit | _TBD_ |
| Pre-recorded demo permitted? | _TBD_ |
| Judging rubric weights | _TBD_ |
| Submission portal URL | _TBD_ |

**Submit a working version at T+20**, not in the final hour. Portal outages and slow video uploads are routine.

---

## 5. Environment

> **H0-B deliverable — Role 3, due T+0:30.** Nominate the demo laptop before anything else.

| Machine | Owner | Cores / RAM | GPU | Python | ffmpeg on PATH | Docker | Demo laptop? |
|---|---|---|---|---|---|---|---|
| 1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 3 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 4 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Notes**

- The entire MVP path is CPU-only by design. A GPU is a stretch-path convenience, never a requirement.
- **Windows:** ffmpeg must be on `PATH`; Docker Desktop must use the WSL2 backend.
- **Setup is owned, not shared.** Role 3 is the designated environment fixer. If your environment breaks, hand it over and keep working — do not let four people debug four installs.
- **Lock dependencies at T+0:45.** On the first machine that installs cleanly, run `make lock` and commit `requirements.lock.txt`. `requirements.txt` states intent; the lock file is the contract.

---

## 6. Privacy & Consent

The corpus contains team members' faces and derived heart-rate waveforms — biometric and health-adjacent personal data.

- `data/corpus/*` and `data/sessions/*` are gitignored. **Media is never committed.**
- Written consent is recorded for every person appearing in the corpus.
- `MANIFEST.csv` **is** tracked — keep real names out of `provenance_notes`; use subject IDs.
- Dataset EULA terms are checked before any frame appears in the deck or README. Most academic deepfake datasets forbid redistribution and public display of frames.
- Confirm this repository's visibility before pushing anything derived from the corpus.
