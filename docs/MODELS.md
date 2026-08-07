# Models

Every model that has been through this project, what it is, and whether it is
allowed to affect the verdict.

---

## Active

| Model | Role | Weights | Licence |
|---|---|---|---|
| **EfficientNet-B7** (Seferbekov, DFDC 1st place) | Channel 04, frame-by-frame | `models/deepfake_efficientnet_b7.pt` (268 MB, TorchScript) | Apache-2.0 |
| **Organika/sdxl-detector** (SwinV2, 86.7M) | Channel 05, synthetic imagery | HF hub, auto-downloaded | **CC-BY-NC-3.0 — non-commercial only** |
| **MediaPipe FaceLandmarker** | face tracking | `models/face_landmarker.task` | Apache-2.0 |
| CHROM / POS | Channel 01, classical rPPG | none (algorithmic) | — |
| Classical pixel forensics | Channel 03 | `results/pixel_calibration.json` | — |

> **Licence warning.** Channel 05's weights are non-commercial. Its ancestor
> (`umm-maybe/AI-image-detector`) was trained on Reddit-scraped images of unclear
> provenance. Fine for a hackathon or research; remove it before any commercial
> deployment.

---

## Rejected — and why

No model enters the verdict without a measured improvement. Each of these was
integrated, measured, and then excluded. The measurements live in the module
docstrings so the reason cannot drift from the code.

| Model | Measurement | Outcome |
|---|---|---|
| **Xception** (FaceForensics++) | AUC **0.222** on TEST_VIDEOS vs effb7's 0.833 — worse than chance | `enabled=False` |
| **Capsule-Forensics** | outputs 0.608 / 0.609 / 0.609 / 0.609 on four different clips — degenerate | `enabled=False` |
| **PhysNet** (UBFC) | band SNR **−6.78 dB** vs CHROM's −3.26 | excluded from fusion |
| **DeepFakesON-Phys** | identical 0.134 for all 13 clips | excluded |
| **LIPINC-V2** | AUC 0.717 (best single on our corpus) but **150 s/clip**, and error-correlated with effb7 at **+0.596** so fusing it *lowered* accuracy | not in the live path |

### The LIPINC lesson

LIPINC was the single best model we measured (0.717 vs effb7's 0.670). Fusing
them still produced **0.700** — worse than LIPINC alone. Error correlation
+0.596 meant they failed on the same clips, so the combination added cost
without adding independence.

**A better model does not imply a better system.**

---

## Non-voting channels

Both are fully implemented and reported in the UI. Neither affects the verdict.

### Channel 03 — Pixel forensics

Calibrated on SDFVD2.0, 876 clips, 52 subjects, **subject-grouped** CV:

| features | random split | grouped |
|---|---|---|
| all | 0.790 | 0.711 |
| scale-free | 0.614 | **0.609** ← ships |

Only scale-free features ship. The 0.10 AUC that absolute sharpness adds is
largely **capture resolution** — the manipulated clips are re-encoded, so a model
using those terms learns "blurrier ⇒ fake" and would not transfer.

Admission test on the 281 clips both channels scored:

| | AUC |
|---|---|
| pixel alone | 0.586 |
| effb7 alone | 0.670 |
| effb7 + pixel | **0.639** (−0.031) |

Error correlation was only −0.079 — the channels *are* decorrelated. It still
fails: too weak to pay for 18 features on 281 clips.

**To enable it:** re-calibrate on better data and flip `admitted: true` in
`results/pixel_calibration.json`. No code change required.

### Channel 05 — Synthetic imagery

AUC **0.000** on TEST_VIDEOS — inverted, with every real clip scoring 0.998–1.000
"synthetic". Polarity was verified against known real photographs first
(0.067 / 0.409 / 0.000), so the mapping is correct and the model genuinely fails.

Cause isolated by degrading a known real photograph:

| condition | P(synthetic) |
|---|---|
| original 780×438 | 0.0000 |
| resized 854×480, no compression | 0.0000 |
| resized 854×480 **+ JPEG q60** | **0.6355** |
| resized 1282×720 + JPEG q30 | 0.3826 |

**Resizing changes nothing; compression is the entire effect.** Trained on
pristine Wikimedia photos vs SDXL renders, it learned "compression artefacts ⇒
synthetic", so every H.264 frame saturates it.

Its score is therefore pinned at a neutral `0.5`; the raw output is kept in
`raw_score` for diagnostics only, because on compressed video that number means
"this frame was compressed."

**To enable it:** a detector trained *with* compression augmentation, validated
on generated video at realistic bitrates.

---

## Adding a model

1. Register a `DetectorSpec` in `src/visual/models.py` (input size, normalisation,
   `output` = `softmax2` or `sigmoid1`).
2. **Verify the class index empirically.** Do not trust `config.id2label` — score
   known-real inputs and confirm the direction. Channel 05's polarity was checked
   this way before its failure could be attributed correctly.
3. Measure it against the current system, not against chance. Report AUC with a
   **grouped** split if the dataset has augmentations or paired real/fake subjects.
4. If it does not improve the system, set `enabled=False` and record the number
   in the `note` field.
