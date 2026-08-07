# Deepfake Detection Based on Visual Lip-sync Match and Blink Rate
*Paper Analysis & Architectural Mapping for DeepGuard*

**Reference Paper:** *Homam El-Taj, Fatima Alammari, Joud Alkhowaiter, Layal Bogari, Renad Essa, "Deepfake Detection Based on Visual Lip-sync Match and Blink Rate" (TrueSync), International Journal of Computational and Experimental Science and Engineering (IJCESEN), Vol. 11, No. 1, pp. 886-898, 2025.*  
**Document Path:** `REFERENCE/Deepfake_Detection_Based_on_Visual_Lip-sync_Match_.pdf`

---

## Executive Summary & Core Insights

This paper introduces **TrueSync**, an audio-visual deepfake detection framework that targets two fundamental weakness in generative facial tools (Wav2Lip, DeepFaceLive, GAN-swaps):
1. **Visual Lip-Sync Mismatch**: Discrepancies between speech audio frequencies/phonemes and mouth movement (Mouth Aspect Ratio / lip dynamics).
2. **Blink Rate Anomalies**: Involuntary physiological blinking patterns (17–22 blinks/min in authentic humans) vs. static/absent or algorithmically inconsistent eye blinks in fakes.

### Key Takeaway & Validation for DeepGuard (v2.1)
The paper confirms that **relying solely on visual pixels is insufficient** against modern deepfakes. Combining temporal audio-visual alignment (lip-sync offset & lag variation) with a second biological/physiological indicator creates a resilient defense against single-modality forgery attacks.

---

## 1. TrueSync System Pipeline

```
Video Input (mp4/avi) ──► 1. Preprocessing & Face Tracking (MediaPipe / OpenCV)
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼                                             ▼
2. Eye Blink Analysis (CNN-LSTM)              3. Lip-Sync Alignment (SyncNet)
   - Spatial feature extraction (CNN)            - Audio onset envelope (1-8 Hz)
   - Temporal blink tracking (LSTM)              - MAR derivative series
   - Normal range: 17-22 blinks/min              - Windowed NCC & Lag IQR (ms)
         │                                             │
         └──────────────────────┬──────────────────────┘
                                ▼
                   4. Quality-Weighted Fusion Score
                                ▼
                 Final Verdict & Confidence Score
```

---

## 2. Audio-Visual Lip-Sync Matching Methodology

- **Mechanic**: Correlates the temporal speech acoustic envelope (syllabic energy, 1–8 Hz band) with visual mouth dynamics (Mouth Aspect Ratio / MAR).
- **Primary Metrics**:
  - **Sync Offset ($\text{Lag}_{\text{median}}$)**: Constant time lag (ms) between audio onset and mouth opening.
  - **Lag Inconsistency ($\text{Lag}_{\text{IQR}}$)**: Interquartile range of lag across multiple speech windows. Genuine speech retains a constant lag (encoder delay), whereas Wav2Lip/dubbing fakes exhibit **wandering lag** across windows.
  - **Peak Cross-Correlation (NCC)**: Normalized cross-correlation peak height between MAR derivative and audio envelope.
- **Phoneme-to-Viseme Matching**: Map bilabial consonants (/m/, /b/, /p/) which require mouth closure during high acoustic energy against raw RMS amplitude.

---

## 3. Eye Blink Rate & Pattern Analysis

- **Human Baseline**: Normal adult resting blink rate is **17–22 blinks per minute**, with each blink lasting **100–400 ms**.
- **Deepfake Anomalies**:
  - **Absent Blinking**: Many early and low-tier GAN models train on open-eye web datasets, producing zero blinks over 15+ seconds.
  - **Rapid/Abnormal Blinking**: Unnatural bursts of blinking caused by frame-stitching artifacts or loss of temporal tracking.
  - **Incomplete Blinks**: Eyelids close partially without full anatomical closure.
- **Model Architecture**: Hybrid **CNN-LSTM** (CNN extracts spatial eye landmarks/eye aspect ratio (EAR); LSTM tracks temporal blink duration & interval distributions).

---

## 4. Fusion Strategy & Decision Policy

TrueSync fuses the two independent channels into a unified manipulation confidence score:

$$\text{Manipulation Score } S = w_1 \cdot (1 - \text{SyncQuality}) + w_2 \cdot \text{BlinkAnomalyScore}$$

- If audio is absent or unvoiced, the system down-weights the lip-sync branch and evaluates eye-blink patterns.
- If face pose/yaw is extreme, blink quality is down-weighted and lip-sync alignment dominates.

---

## 5. Datasets & Experimental Results

Evaluated across benchmark deepfake datasets:
- **FaceForensics++ (FF++)**: 94.2% Detection Accuracy
- **Deepfake Detection Challenge (DFDC)**: 91.8% Accuracy
- **Celeb-DF (v2)**: 93.5% Accuracy

---

## 6. Documented Limitations & Failure Modes

1. **Silent / Low-Speech Clips**: Lip-sync matching requires voiced speech segments ($\ge 8$ voiced windows). Silent clips return reduced confidence.
2. **Audio Noise / Compression**: Background noise or low sample rate audio distorts speech onset envelopes.
3. **Extreme Head Pose & Yaw**: Side profiles obscure mouth corners and eye landmarks, reducing landmark precision.
4. **Adverse Lighting**: Dark environments degrade eye aspect ratio tracking, causing false blink detection.

---

## 7. Architectural Mapping to DeepGuard (`v2.1`)

| TrueSync Paper Concept | DeepGuard Implementation (`src/lipsync/` & `contracts.py`) |
|---|---|
| **Mouth Aspect Ratio (MAR)** | Inner lip distance over outer mouth width: $\text{MAR} = \frac{\|LM_{13} - LM_{14}\|}{\|LM_{78} - LM_{308}\|}$ |
| **Speech Audio Envelope** | `librosa.onset.onset_strength()` filtered to 1–8 Hz syllabic frequency band. |
| **Windowed Alignment** | VAD-gated normalized cross-correlation across 1.0s speech windows (0.5s hop). |
| **Lag IQR Discriminator** | `lag_iqr_ms` tracks offset variation. Wandering lag indicates Wav2Lip dubbing. |
| **A/V Start Offset Correction** | `av_start_offset_sec` subtracted prior to alignment to eliminate container muxer bias. |
| **Multi-Modal Fusion** | Combined with rPPG cross-region consistency in `src/fusion/scorer.py`. |
