# DeepGuard // Analysis of Reference Repositories

This document synthesizes key takeaways, architectural patterns, and integration opportunities from two prominent open-source deepfake detection repositories:
1. `Daisy-Zhang/Awesome-Deepfakes-Detection` (Curated Benchmark & Datasets Hub)
2. `abhijithjadhav/Deepfake_detection_using_deep_learning` (ResNeXt + LSTM Hybrid Model)

---

## 1. Repository 1: `Daisy-Zhang/Awesome-Deepfakes-Detection`

### What It Contains
- **Comprehensive Benchmark Hub**: Curated collection of academic papers (CVPR, ICCV, ECCV, TPAMI), SOTA detection algorithms, and benchmark datasets.
- **Dataset Registry**:
  - **Video Datasets**: Celeb-DF v2, DFDC Preview, FaceForensics++ (c20/c40), DeeperForensics-1.0, ForgeryNet, Deepfake-TIMIT, UADFV.
  - **Image Datasets**: DFFD (Digital Face Manipulation Database), WildDeepfake.
- **Sister Repository**: `Awesome-AIGC-Detection` (focusing on Diffusion models, Midjourney, Sora, and generative AIGC content).

### What We Can Extract & Adopt for DeepGuard
1. **Evaluation Protocols**: Standardized evaluation split methodologies (subject-disjoint splits, cross-dataset generalization benchmarks).
2. **Academic Baseline Benchmarks**: Performance targets (AUC, EER, Log-loss) for Xception, EfficientNet-B4, and frequency-domain models.
3. **Dataset Links & Preprocessing Guidelines**: Reference pipelines for handling compressed social media clips (H.264/CRF).

---

## 2. Repository 2: `abhijithjadhav/Deepfake_detection_using_deep_learning`

### What It Contains
- **Hybrid CNN-LSTM Architecture**: PyTorch implementation pairing a pre-trained **ResNeXt** CNN spatial feature extractor with a sequence-learning **LSTM** temporal head.
- **Video Preprocessing Pipeline**: Frame sampling, face bounding box cropping via `face_recognition`/OpenCV, and tensor normalization.

### Architecture Comparison vs. DeepGuard

| Component | `abhijithjadhav` Repo | DeepGuard NextGen Architecture |
|---|---|---|
| **Spatial Backbone** | ResNeXt-50 (2048-d features per frame) | ResNet-18 / ResNeXt-50 perioral crop visual branch |
| **Temporal Head** | Single bidirectional LSTM (`nn.LSTM`) | Multi-head Transformer Encoder + Positional Embeddings |
| **Modality Scope** | Single visual frame sequence | **Tri-Modal Fusion** (Visual perioral + Full-face AU + rPPG pulse) |
| **Cross-Modal Attention**| None (Visual-only) | Bidirectional Cross-Attention + Per-timestep Mismatch Scoring |
| **Loss Function** | Standard Binary Cross Entropy | `ForensicMultiTaskLoss` (BCE + Temporal Coherence + Contrastive) |

---

## 3. Integration Blueprint for DeepGuard

1. **Backbone Upgrade**: Upgrade our `TimeDistributedCNN` frontend in [`src/lipsync/detector.py`](file:///c:/Users/Balaji/OneDrive/Desktop/Innogenesis/src/lipsync/detector.py#L25-L45) to accept `ResNeXt-50_32x4d` for 2048-d spatial representations.
2. **Hybrid LSTM-Transformer Block**: Combine ResNeXt spatial feature extraction with our temporal Transformer cross-attention fusion.
3. **Benchmark Manifest Alignment**: Use `Daisy-Zhang` dataset structures to populate [`data/corpus/MANIFEST.csv`](file:///c:/Users/Balaji/OneDrive/Desktop/Innogenesis/data/corpus/MANIFEST.csv).
