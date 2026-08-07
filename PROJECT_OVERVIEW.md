# DeepGuard — Multi-Modal Deepfake & Manipulation Detection

**A physiologically-grounded detector that reports what it can measure, and abstains when it cannot.**

> Every figure in this document is a measurement we ran, with its sample size stated. Where we have not measured something, it is marked *not evaluated*. Nothing here is projected, illustrative, or inferred.

---

## 1. The problem

Modern generative models produce faces that survive frame-by-frame inspection. Most detectors respond by hunting pixel artifacts — and they excel on the manipulation methods they were trained on. The well-documented failure is generalization: strong in-domain, far weaker on unseen methods and unseen capture conditions.

We attack it from four independent directions, so that no single failure mode is decisive:

| Channel | Physical basis | Targets |
|---|---|---|
| **rPPG** | Blood-volume pulse, measured across facial regions | Face swaps — a blend boundary breaks cross-region pulse phase |
| **Lip-sync** | Speech-to-lip biomechanics | Lip-sync deepfakes, dubbing |
| **Traditional pixel forensics** | Classical image forensics | Local warping, region-specific texture failure |
| **Frame-by-frame CNN** | Learned spatial artifacts | Broad coverage across manipulation families |

---

## 2. Architecture

```
                 ┌─────────────────────────────────────┐
   video ──────► │  Ingest: normalise, true PTS,       │
                 │  single face-tracking pass          │
                 └──────────────┬──────────────────────┘
                                │
     ┌──────────────┬───────────┴───────────┬──────────────┐
     ▼              ▼                       ▼              ▼
 ┌────────┐   ┌───────────┐         ┌──────────────┐  ┌──────────┐
 │  rPPG  │   │ Lip-Sync  │         │  Traditional │  │  Frame   │
 │30-patch│   │ VAD-gated │         │  pixel       │  │  CNN     │
 │coherence│  │ alignment │         │  forensics   │  │          │
 └───┬────┘   └─────┬─────┘         └──────┬───────┘  └────┬─────┘
     │ score+quality│                      │               │
     └──────────────┴──────────┬───────────┴───────────────┘
                               ▼
              ┌──────────────────────────────────┐
              │ Quality-weighted log-odds fusion │
              │ → verdict + evidence weight      │
              │ → per-channel contribution       │
              └──────────────────────────────────┘
```

### 2.1 rPPG — spatial pulse coherence

The face is tiled into a **6×5 grid of 30 patches**, each yielding a pulse signal, giving up to **435 pairwise phase comparisons**.

The design decision that matters: we do **not** ask *"is there a pulse?"* A face-swap composites synthesised skin inside a mask while leaving authentic skin outside it, so a pulse often survives. What cannot survive is **agreement across the seam**. We score cross-region *disagreement* — phase dispersion, heart-rate spread, pairwise correlation.

Engineering that materially affects correctness:
- Heart rate reported as a **confidence interval**, never a false-precision point estimate
- **Zero-phase filtering** throughout — a causal filter would manufacture the very phase dispersion being measured
- **True presentation timestamps**, never nominal frame rate, so variable-frame-rate video cannot fabricate periodicity

### 2.2 Lip-sync

Mouth motion against the 1–8 Hz syllabic speech envelope, VAD-gated, measured in sliding windows.

**The discriminator is lag *consistency*, not lag.** A genuine recording can carry a large constant A/V offset from its encoder — we measured **−167 ms** in one test clip. What it does not do is drift window to window. Stream offsets are therefore corrected before measurement.

An open-source pre-trained vision temporal transformer specialised for lip-sync manipulation is available as an opt-in deep-scan pass.

### 2.3 Traditional pixel forensics — *in development*

Feature design is derived from **4,347 human annotations of real deepfakes** (ExDDV, CC BY-NC-SA), not from assumption. What annotators actually report:

| artifact | share of annotations |
|---|---|
| warping / distortion | 54.1% |
| mouth / lips | 39.7% |
| eyes | 38.6% |
| blur / pixelation | 12.7% |
| shadow / lighting | 11.9% |
| temporal flicker | 11.3% |

The channel therefore targets **local geometric distortion and region-specific texture failure around the mouth and eyes**, rather than generic whole-frame compression forensics.

**Status: not yet implemented. No performance is claimed.**

### 2.4 Frame-by-frame CNN

An open-source pre-trained EfficientNet derived from the winning solution of the Deepfake Detection Challenge, applied to aligned face crops.

**Dual purpose by construction:** the same weights and identical preprocessing serve both the standalone image detector and the video pipeline's per-frame branch — one code path, so the two cannot disagree about the same face.

Per-frame scores aggregate at the **90th percentile**, not the median: a manipulated clip has *some* clearly-compromised frames and the median washes them out.

### 2.5 Fusion

Channels combine in **log-odds**, not as a weighted average of probabilities.

Under a weighted mean, a channel reporting 0.5 drags the verdict toward 0.5 — so a branch saying *"I have no information"* destroys evidence from branches that do. In log-odds, `logit(0.5) = 0`: an uninformative channel contributes **exactly nothing**, agreeing channels reinforce, disagreeing ones cancel.

Every channel reports **two numbers** — a manipulation score *and* an evidence quality reflecting whether the measurement was informative, not merely whether it completed.

### 2.6 Abstention

Four verdicts: `LIKELY_AUTHENTIC` · `LIKELY_MANIPULATED` · `UNCERTAIN` · `INSUFFICIENT_EVIDENCE`

**The two error types are not symmetric.** Labelling a real person's video as manipulated is defamatory; missing a fake is a miss. The uncertainty band is deliberately wide.

---

## 3. Measured results

### 3.1 Detection — benchmark data

**EfficientNet channel, DFDC benchmark face crops (n = 22: 11 manipulated, 11 authentic):**

| metric | value |
|---|---|
| AUC | **1.000** |
| Accuracy @ threshold 0.5 | 0.818 |
| Authentic score range | 0.008 – 0.014 |
| Manipulated score range | 0.110 – 0.995 |

Perfect rank separation: every manipulated crop scores above every authentic one. Accuracy is below AUC only because the default 0.5 threshold is mis-set for this domain — the optimal operating point here is ≈0.05, at which accuracy is 100%.

**This is an in-domain result** — the model was trained on DFDC, and n = 22 is small. It demonstrates the channel is correctly integrated. It is not a generalization claim.

### 3.2 Fusion strategy — validated at scale

Measured on a **20,000-sample, 7-detector benchmark** (DeepSafe, MIT licence), 5-fold cross-validated, held-out AUC:

| strategy | AUC |
|---|---|
| naive mean of all 7 (log-odds) | 0.667 ± 0.008 |
| best single detector | 0.773 ± 0.001 |
| learned meta-learner (logistic) | 0.800 ± 0.002 |
| learned meta-learner (gradient boosting) | **0.811 ± 0.001** |

Selective fusion by detector quality, same data:

```
top-1  0.773      top-4  0.703
top-2  0.754      top-5  0.679
top-3  0.762      top-7  0.667
```

**Two findings drive our design:**
1. **Naive multi-model fusion is actively harmful** — 11 AUC points lost across seven published detectors. Three of the seven scored at or below chance.
2. **Learned weights fix it** (+3.8 over best-single) by assigning *negative* weight to anti-correlated detectors — which equal weighting cannot do.

### 3.3 Why we ship fixed priors rather than learned weights

We tested learned fusion on our own channels. Fitted on all available data it reports 0.700; **leave-one-out it collapses to 0.300** — worse than chance. Eleven labelled examples cannot support learned weights.

We therefore ship principled priors with quality gating and abstention, and report the cross-validated number rather than the fitted one.

### 3.4 Models evaluated and rejected

Six pre-trained detectors were integrated and measured. Five were excluded, each on evidence recorded in the code:

| model | measurement | outcome |
|---|---|---|
| Xception (FakeAVCeleb) | below chance on our evaluation set; averaging it with the strongest model dragged the ensemble down to its own level | excluded |
| Capsule-Forensics | emitted a near-constant 0.609 across four different clips, 0.562 on random noise | excluded |
| PhysNet (UBFC) | band SNR worse than classical CHROM on every clip: −6.78 vs −3.26 dB, −5.51 vs −2.32, −6.92 vs −1.74, −6.98 vs −4.03 | excluded |
| DeepFakesON-Phys | one checkpoint returned a constant 0.134 on all inputs (AUC exactly 0.500); six clips produced bit-identical scores | excluded |
| Two further repositories | no distributable weights | unusable |

All remain loadable and reported for transparency; none contribute to the fused score.

### 3.5 Test suite

**89 automated tests passing**, including regression guards on score polarity, explanation/verdict consistency, and preprocessing constants.

---

## 4. Engineering discipline

**Admission by measurement.** No channel or model enters fusion without a measured result showing it improves the outcome. This rule was written after measuring a weak model degrade a strong ensemble, and it is enforced in the detector registry.

**Every model's polarity is verified, never assumed.** Of the models integrated, most use index 1 = fake and at least one uses the opposite. Getting this backwards inverts a detector while still producing entirely plausible probabilities — no crash, no warning. Each convention is verified against its source implementation and locked behind a test.

**Preprocessing is never "improved".** Reimplementing or optimising a trained model's input pipeline is a silent accuracy leak. We measured a preprocessing change flip one clip's score from 0.974 to 0.029 — no error raised, just a different answer. Optimisations are validated bit-identical against the reference.

**Failures degrade, they never crash.** Silent video, no face, corrupted file, portrait phone footage, unusual codecs — every path returns a typed, plain-English result.

**Thresholds live in one auditable file.** No magic numbers in code.

---

## 5. Disclosed limitations

- **No validated generalization benchmark yet.** Our in-the-wild clip set is ad-hoc internet footage with unverified provenance and labels. We therefore quote **no** generalization figure from it. Establishing a proper benchmark (DeepfakeTIMIT + VidTIMIT, subject-disjoint splits) is the next milestone.
- **The traditional pixel channel is not implemented.** Its design is evidence-based; its performance is unmeasured.
- **rPPG abstains on compressed footage.** We measured near-zero recoverable pulse SNR using classical CHROM/POS *and* two supervised extractors. The limiting factor is the footage, not the extractor — and the system reports insufficient evidence rather than guessing.
- **rPPG demographic variance.** Accuracy is known to vary with skin tone due to melanin absorption. This is **disclosed, not mitigated**; our data is far too limited to quantify it.
- **Audio-only deepfakes** (voice cloning with authentic video) are out of scope.
- **Lip-sync deep-scan latency** is ~150 s per clip; it is opt-in, not in the interactive path.

---

## 6. Roadmap

1. **Validated benchmark** — DeepfakeTIMIT + VidTIMIT with subject-disjoint splits; this is the precondition for every other number
2. **Traditional pixel channel** — implement and calibrate against that benchmark
3. **Cross-manipulation evaluation** — leave-one-method-out harness is already built and awaiting a multi-method corpus
4. **Grad-CAM on the CNN channel** — spatial explanation to pair with the pulse-coherence map
5. **Learned fusion** — once the corpus supports it

---

## 7. Open-source components

Built on open-source pre-trained models and research implementations under their respective licences: an EfficientNet deepfake classifier (Apache-2.0), a vision temporal transformer for lip-sync detection, classical rPPG extraction (CHROM/POS), MediaPipe face landmarking, and the ExDDV annotation set (CC BY-NC-SA, used for feature design only). Model weights are not redistributed.

Our contribution is the **multi-channel fusion architecture, the evidence-quality framework, the calibrated abstention design, and the measurement discipline** that makes the whole reproducible.

---

## 8. One-sentence summary

> **DeepGuard fuses independent physiological and forensic signals into a single calibrated verdict, weights each by how much it actually knows, and abstains honestly when the evidence cannot support a decision.**
