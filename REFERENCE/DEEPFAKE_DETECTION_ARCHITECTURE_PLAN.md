# DeepGuard NextGen // Multi-Modal Deepfake Detection Architecture Plan

**Author:** Senior ML Architect & Media Forensics Lead  
**Reference Papers Synthesized:**
1. *Utilizing rPPG Signal Synchronization and Deep Learning Techniques for Deepfake Video Detection* (Susi et al., IEEE Access 2025)
2. *A Research on Deepfake Face Detection Techniques Based on Multimodal Biometric Cross-Verification* (Haozhe Wu, ITM Web of Conferences 2025)
3. *Deepfake Detection Based on Visual Lip-sync Match and Blink Rate* (El-Taj et al., IJCESEN 2025)

---

## 1. Executive Summary & Architectural Vision

Generative deepfake technologies (SimSwap, DeepFaceLab, Wav2Lip, SadTalker, Sora, LivePortrait) optimize pixel textures and local smoothness. However, **they fail to respect physiological laws and cross-modal physical constraints**:
1. **Biological Incoherence**: Generative faces cannot replicate synchronous blood volume pulse (BVP) across bilateral facial regions.
2. **Biomechanical Desync**: Synthetic lip motion drifts out-of-phase with spoken acoustic energy across consecutive speech windows.
3. **Physical & Ocular Anomalies**: Synthetic videos exhibit missing/irregular eye blink dynamics and mismatched corneal light reflections.

This document outlines a **Tri-Modal Cross-Verification Architecture** fusing physiological rPPG signal synchronization, audio-visual biomechanical lip alignment, and ocular physical micro-anomalies into an attention-guided dynamic fusion engine.

---

## 2. Multi-Modal Pipeline Architecture

```
                                    ┌─────────────────────────────────────────────────────────┐
                                    │                     INPUT VIDEO                         │
                                    └────────────────────────────┬────────────────────────────┘
                                                                 │
                                ┌────────────────────────────────┼────────────────────────────────┐
                                ▼                                ▼                                ▼
                   ┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
                   │    MODULE 1: rPPG        │    │    MODULE 2: LIP-SYNC    │    │    MODULE 3: OCULAR      │
                   │  Bilateral Signal Sync   │    │  Biomechanical Alignment │    │  Physical Micro-Anomalies│
                   └────────────┬─────────────┘    └────────────┬─────────────┘    └────────────┬─────────────┘
                                │                               │                               │
                                ▼                               ▼                               ▼
                   - DWT Wavelet Filter (db4)      - Acoustic Mel-Spectrum        - EAR Blink Tracking (LSTM)
                   - Cross-Cheek PCC & PSD         - MAR Derivative Series        - Corneal Specular Highlights
                   - SNR Quality Gating            - Windowed SyncNet Lag IQR     - Micro-expression Jitter
                                │                               │                               │
                                └───────────────────────┬───────┴───────────────────────┘
                                                        │
                                                        ▼
                                    ┌─────────────────────────────────────────────────────────┐
                                    │      ATTENTION-GUIDED DYNAMIC FUSION ENGINE             │
                                    │    - Quality-Weighted Attention Gating                  │
                                    │    - Cross-Modal Consistency Verification               │
                                    │    - Honest Abstention Thresholding                     │
                                    └───────────────────────────┬─────────────────────────────┘
                                                                │
                                                                ▼
                                    ┌─────────────────────────────────────────────────────────┐
                                    │                    FINAL VERDICT                        │
                                    │    LIKELY_AUTHENTIC / LIKELY_MANIPULATED / UNCERTAIN   │
                                    └─────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Subsystem Specifications

### Subsystem 1: Bilateral rPPG Signal Synchronization (`Sync_rPPG`)
*Inspired by Susi et al. (2025)*

- **Bilateral ROI Extraction**: Extract raw temporal RGB signals from 4 symmetric facial sub-regions: Left Cheek ($ROI_{LC}$), Right Cheek ($ROI_{RC}$), Forehead ($ROI_{FH}$), and Nose ($ROI_{NS}$).
- **Discrete Wavelet Transform (DWT) Filtering**:
  - Apply 4-level DWT decomposition using `db4` Daubechies wavelets to isolate cardiac frequency bands ($0.7 - 3.5 \text{ Hz}$) from high-frequency sensor noise and low-frequency motion artifacts.
- **Statistical Synchronization Feature Vector**:
  $$\vec{F}_{\text{rPPG}} = \Big[ \text{PCC}(S_{\text{LC}}, S_{\text{RC}}), \text{PSD}_{\text{peak}}, \text{SNR}_{\text{band}}, \text{MAD}(S), \text{SD}(S) \Big]$$
  - Genuine faces maintain high cross-cheek Pearson correlation ($PCC > 0.75$) and sharp spectral peaks. Face-swaps break cross-cheek phase agreement, creating severe inter-cheek signal divergence.

### Subsystem 2: Audio-Visual Biomechanical Lip Alignment (`TrueSync`)
*Inspired by El-Taj et al. (2025)*

- **Acoustic-Visual Feature Pair**:
  - Extract syllabic acoustic energy envelope ($1 - 8 \text{ Hz}$) from the audio stream via 80-band Mel-spectrograms.
  - Calculate 3D Mouth Aspect Ratio ($\text{MAR} = \frac{\|p_2 - p_8\| + \|p_3 - p_7\| + \|p_4 - p_6\|}{2 \|p_1 - p_5\|}$) from MediaPipe/FaceMesh landmarks.
- **Wandering Lag Discriminator**:
  - Compute windowed Normalized Cross-Correlation (NCC) across 5-frame sliding windows.
  - Extract **Median Lag ($\text{Lag}_{\text{med}}$)** and **Lag Interquartile Range ($\text{Lag}_{\text{IQR}}$)**.
  - Dubbed videos (Wav2Lip, SadTalker) exhibit wandering lag ($\text{Lag}_{\text{IQR}} > 60\text{ ms}$) across windows, whereas genuine recordings maintain constant encoder latency.

### Subsystem 3: Ocular Micro-Anomalies & Optical Physics
*Inspired by Haozhe Wu (2025) & El-Taj et al. (2025)*

- **Temporal Eye Blink Rate Analysis**:
  - Compute Eye Aspect Ratio ($\text{EAR}$) over time using a hybrid **CNN-LSTM** network.
  - Flag physiological anomalies: missing blinks over 15+ seconds, abnormal frequency ($>30$ or $<10$ blinks/min), or incomplete eyelid closure.
- **Corneal Specular Highlight Verification**:
  - Extract left and right corneal pupil reflections.
  - Calculate environmental light source direction consistency across eyes. GANs synthesize eyes independently, introducing sub-pixel light vector discrepancies.

---

## 4. Attention-Guided Dynamic Fusion Engine

*Inspired by Haozhe Wu (2025)*

Traditional simple average or unweighted voting models fail when one modality is corrupted (e.g. video compression degrading rPPG). We employ an **Attention-Guided Quality Gating Fusion**:

### 4.1 Feature Embedding & Quality Assessment
For each modality $m \in \{\text{rPPG}, \text{LipSync}, \text{Ocular}\}$:
1. Compute Modality Feature Vector $h_m$ and Signal Quality Metric $Q_m \in [0, 1]$.
2. Compute Dynamic Attention Weight $\alpha_m$:
   $$\alpha_m = \frac{\exp(W_a h_m + \gamma Q_m)}{\sum_{k} \exp(W_a h_k + \gamma Q_k)}$$

### 4.2 Fusion Verdict Classifier & Honest Abstention
- **Combined Manipulation Probability**:
  $$P(\text{Manipulated}) = \sigma \left( \sum_{m} \alpha_m \cdot W_m h_m \right)$$
- **Abstention Logic**:
  - If $\sum \alpha_m Q_m < \tau_{\text{evidence}}$: Return **`INSUFFICIENT_EVIDENCE`**.
  - If $0.45 \le P(\text{Manipulated}) \le 0.55$: Return **`UNCERTAIN`**.
  - If $P(\text{Manipulated}) > 0.65$: Return **`LIKELY_MANIPULATED`**.
  - If $P(\text{Manipulated}) < 0.35$: Return **`LIKELY_AUTHENTIC`**.

---

## 5. Expected Performance & Benchmark Metrics

| Dataset | Modality Focus | Target AUC | Target Accuracy |
|---|---|---|---|
| **Celeb-DF v2** | Bilateral rPPG + Ocular | $\ge 0.985$ | $\ge 97.5\%$ |
| **FaceForensics++ (c20)** | rPPG + LipSync Fusion | $\ge 0.978$ | $\ge 96.2\%$ |
| **Wav2Lip / Dubbed Sets** | TrueSync Lip-Sync Lag IQR | $\ge 0.990$ | $\ge 98.8\%$ |
| **Wild Deepfakes (Shorts/TikTok)** | Tri-Modal Attention Fusion | $\ge 0.945$ | $\ge 93.0\%$ |
