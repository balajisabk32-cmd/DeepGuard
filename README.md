# DeepGuard // Multi-Modal Deepfake & Manipulation Detection (v2.1)

**DeepGuard** is a quality-weighted multi-modal video manipulation detector fusing **cross-region blood flow consistency (rPPG)** with **speech-to-lip biomechanical alignment (VAD-gated cross-correlation)**.

---

## 0. Quick Start (Offline Docker)

```bash
docker compose up
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 1. Project Structure

```
├── config/
│   └── thresholds.yaml         # Decision thresholds & quality gates
├── data/
│   ├── corpus/                 # Evaluation dataset & MANIFEST.csv
│   └── sessions/               # Temporary video/audio processing outputs
├── models/                     # Vendored model weights (e.g. face_landmarker.task)
├── src/
│   ├── common/
│   │   └── contracts.py        # Executable Pydantic v2 schemas
│   ├── rppg/                   # Biosignal extraction (CHROM / POS)
│   ├── lipsync/                # Biomechanical speech-to-lip alignment
│   ├── pipeline/               # ffmpeg transcode, single-pass landmarking, FastAPI
│   └── ui/                     # Streamlit explainability dashboard
├── tests/                      # Unit & contract verification tests
├── plan.md                     # 24-Hour Execution Plan (v2.1)
├── requirements.txt            # Python dependencies
└── docker-compose.yml          # Container configuration
```

---

## 2. Threat Model Coverage Matrix

| Attack Class | Example Tools | rPPG Consistency | Lip-Sync Alignment | System Verdict |
|---|---|---|---|---|
| **Face Swap** | SimSwap, FaceFusion | **Catches It** (Phase dispersion spike) | Weak (Lips match audio) | `LIKELY_MANIPULATED` |
| **Lip-Sync / Dub** | Wav2Lip | Weak (Pulse intact) | **Catches It** (Lag IQR drift) | `LIKELY_MANIPULATED` |
| **Reenactment** | Face2Face | Partial | Partial | `UNCERTAIN / MANIPULATED` |
| **Synthetic Video** | Sora, Gen-2 | **Catches It** (No coherent pulse) | Varies | `LIKELY_MANIPULATED` |
| **Voice Clone Only** | RVC, Bark | Out of Scope | Out of Scope | *Declared Out of Scope* |

---

## 3. Disclosed Limitations & Roadmap

- **Demonstration Corpus**: Evaluated on ~28 curated clips, not benchmark scale.
- **Compression Sensitivity**: Disclosed degradation under heavy re-encoding (CRF > 28).
- **rPPG Demographic Variance**: Melanin absorption variations in green channel accounted for via multi-modality fusion.
- **Audio-Only Deepfakes**: Out of scope; planned as Roadmap Item #1.
