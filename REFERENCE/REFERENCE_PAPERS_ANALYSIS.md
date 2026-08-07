# DeepGuard // Reference Literature Analysis & Architectural Mapping

This document provides a technical analysis of the two primary research papers in the `REFERENCE/` directory and details how their theoretical findings directly map to the **DeepGuard** multi-modal deepfake detection codebase.

---

## Executive Summary & Core Thesis

No single biosignal or visual modality can detect all classes of video manipulation. Modern generative deepfakes require a **multi-modal fusion strategy**:

$$\text{Verdict} = \mathcal{F}\Big(\text{rPPG Spatial Coherence (Paper 1)}, \text{ Lip-Sync Biomechanical Alignment (Paper 2)}\Big)$$

| Modality / Paper | Primary Biological Mechanism | Primary Target Attack Vector | Failure Mode / Vulnerability | DeepGuard Implementation |
|---|---|---|---|---|
| **Paper 1: rPPG Review** (*Kammari et al., 2025*) | Micro-vascular blood volume pulse (BVP) spatial phase agreement. | **Face Swaps** (SimSwap, FaceFusion) — blend boundaries break cross-region phase. | High video compression (H.264/CRF > 28) suppresses sub-pixel color modulation. | `src/rppg/` (`signal_core.py`, `ppgmap.py`) |
| **Paper 2: TrueSync** (*El-Taj et al., 2025*) | Speech-to-lip temporal alignment & biomechanical lag consistency. | **Lip-Sync / Dubbing** (Wav2Lip, DeepFaceLive) — mouth motion drifts across audio windows. | Genuine face swaps (pulse remains intact; lips match audio). | `src/lipsync/` & `src/common/contracts.py` |

---

## 1. Deep Analysis of Paper 1: rPPG Spatial Pulse Coherence

**Reference:** *K.S. Kammari et al., "A Comprehensive Review of Deepfake Detection Techniques Utilizing Remote Photoplethysmography", IEEE Access, 2025.* (`REFERENCE/A_Comprehensive_Review_of_Deepfake_Detection_Techn.pdf`)

### 1.1 Physical & Biological Mechanism
- Remote Photoplethysmography (rPPG) extracts the periodic absorption of green light ($\sim 520 - 580 \text{ nm}$) caused by arterial blood pulsation in facial capillary beds.
- **Deepfake Artifact**: Generative face-swap models composite synthetic facial skin (cheeks/nose) onto an authentic head (forehead). This introduces a blend boundary that destroys **cross-region phase synchronization** ($\sigma_\phi$) and creates artificial **heart rate spread** ($\Delta \text{HR} > 12 \text{ BPM}$).

### 1.2 Algorithmic Mapping in DeepGuard

```
RGB Video ──► FaceMesh Landmarking ──► 6x5 Grid Crop (30 Patches)
                     │
                     ▼
       Zero-Phase Bandpass (0.7-3.5 Hz)
                     │
                     ▼
         POS / CHROM Signal Extraction  ──►  Pairwise Coherence Matrix
                     │
                     ▼
          SNR Quality Gating (-3 dB)    ──►  rppg_manipulation_score
```

1. **POS & CHROM Projection** (`src/rppg/signal_core.py`):
   - Projects temporal RGB channels onto a plane orthogonal to the skin-tone vector to cancel specular reflection and motion artifacts.
2. **Spatial-Temporal Patch Grid** (`src/rppg/ppgmap.py`):
   - Rather than 3 static ROIs, DeepGuard uses a $6 \times 5$ spatial patch grid ($30$ patches, up to $435$ pairs). The 25th percentile pairwise correlation ($r_{p25}$) provides a robust statistic against local noise.
3. **Diagnostic Coherence Heatmap** (`src/rppg/webcam_heatmap.py`):
   - Renders live patch-wise phase coherence: **Green** (in phase with face), **Amber** (uncorrelated), **Red** (anti-phase seam).

---

## 2. Deep Analysis of Paper 2: Visual Lip-Sync Matching

**Reference:** *H. El-Taj et al., "Deepfake Detection Based on Visual Lip-sync Match and Blink Rate", IJCESEN, 2025.* (`REFERENCE/Deepfake_Detection_Based_on_Visual_Lip-sync_Match_.pdf`)

### 2.1 Physical & Biomechanical Mechanism
- Natural human speech exhibits strict biomechanical synchronization between vocal tract acoustic energy (syllabic envelope, $1 - 8 \text{ Hz}$) and visual mouth opening (Mouth Aspect Ratio / MAR).
- **Deepfake Artifact**: Dubbing and Wav2Lip models generate mouth movement frame-by-frame. While peak correlation (NCC) may appear high locally, the temporal lag between audio and video **wanders across temporal windows**, causing high **Lag Interquartile Range** ($\text{Lag}_{\text{IQR}}$).

### 2.2 Algorithmic Mapping in DeepGuard

```
Audio Track ──► VAD Speech Gating ──► Acoustic Envelope (1-8 Hz)
                                              │
                                              ▼
Video Track ──► Lip Landmarking   ──► MAR Derivative Series
                                              │
                                              ▼
                          Windowed Cross-Correlation (NCC)
                                              │
                                              ▼
                    Median Lag & Lag IQR  ──►  lipsync_manipulation_score
```

1. **Voice Activity Detection (VAD) Gating**:
   - Silences and non-speech gaps are excluded to prevent false correlation peaks.
2. **Lag IQR Discriminator**:
   - Authentic recordings maintain constant latency ($\text{Lag}_{\text{IQR}} \approx 0 \text{ ms}$). Audio dubbing fakes display wandering lag ($\text{Lag}_{\text{IQR}} > 60 \text{ ms}$).

---

## 3. Modality Fusion & Threat Coverage Matrix

### 3.1 Quality-Weighted Fusion Scoring (`src/fusion/scorer.py`)
DeepGuard implements dynamic evidence weighting:

$$\text{Evidence Weight } w_m = \text{Quality}_m \times \text{Modality Trust Factor}$$

- **Single-Modality Cap**: A single modality's weight is capped at $0.50$ to prevent unconfirmed single-channel decisions.
- **Abstention Mechanics**: If overall evidence quality drops below the threshold, the system abstains honestly with `UNCERTAIN` or `INSUFFICIENT_EVIDENCE`.

### 3.2 Threat Matrix Summary

| Attack Vector | Example Generative Tools | rPPG Signal Output | Lip-Sync Signal Output | DeepGuard Verdict |
|---|---|---|---|---|
| **Face Swap** | SimSwap, FaceFusion, Roop | Phase broken ($r_{p25} < 0.30$) | Normal lag consistency | `LIKELY_MANIPULATED` |
| **Lip-Sync / Dub** | Wav2Lip, SadTalker | Intact cardiac pulse | Wandering lag ($\text{Lag}_{\text{IQR}} > 60\text{ms}$) | `LIKELY_MANIPULATED` |
| **Authentic Video** | Camera recording | Coherent ($r_{p25} > 0.70$) | Constant lag ($\text{Lag}_{\text{IQR}} < 20\text{ms}$) | `LIKELY_AUTHENTIC` |
| **Heavy Compression**| Re-encoded social media | Degraded SNR ($<-3\text{ dB}$) | Low resolution | `INSUFFICIENT_EVIDENCE` |

---

## 4. Key Reference Documents in Repository

- `REFERENCE/A_Comprehensive_Review_of_Deepfake_Detection_Techn.pdf`
- `REFERENCE/rppg_deepfake_paper_summary.md`
- `REFERENCE/Deepfake_Detection_Based_on_Visual_Lip-sync_Match_.pdf`
- `REFERENCE/lipsync_deepfake_paper_summary.md`
- `TECHNICAL_HINTS.md`
