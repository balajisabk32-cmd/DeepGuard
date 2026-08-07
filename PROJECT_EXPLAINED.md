# DeepGuard — End-to-End Explanation

Multi-modal deepfake detection. Five independent channels, quality-weighted
fusion, and an explicit right to say "I don't know."

Every number in this document is measured. Where a number is bad, it is printed
anyway — that is the point of the document.

---

## 1. The one-paragraph version

A video comes in. We decode it once and share those frames across five detectors
that look for completely different physical evidence: a pulse in the skin, speech
that matches the mouth, classical compression/warp forensics, a per-frame CNN,
and a whole-image synthesis detector. Each returns **two** numbers — a score and
*how much evidence it actually had*. Fusion combines them in log-odds weighted by
that evidence. If the total evidence is too thin, or the channels land in the
undecided band, the system **abstains** instead of guessing.

Two of the five channels currently contribute exactly zero to the verdict. That
is not a bug — they failed a measured admission test and are shown as
non-voting. Section 6 explains why that is the most important design decision in
the project.

---

## 2. Workflow — what happens to an uploaded video

```
                          upload (.mp4)
                                │
                    ┌───────────▼───────────┐
                    │  decode_clip()        │   ONE decode, ONE face-detection
                    │  • PyAV, true PTS     │   pass, shared by all channels
                    │  • ≤20 s analysed     │   (~4–9 s)
                    │  • face box / frame   │
                    └───────────┬───────────┘
                                │ frames + boxes + timestamps
        ┌───────────┬───────────┼───────────┬─────────────┐
        ▼           ▼           ▼           ▼             ▼
   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐   ┌──────────┐
   │  rPPG  │  │lip-sync│  │ pixel  │  │frame-  │   │ synthetic│
   │        │  │        │  │forensic│  │by-frame│   │ imagery  │
   │ 0.3 s  │  │ 0.2 s  │  │ 0.4 s  │  │ ~21 s  │   │  ~3 s    │
   └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘   └────┬─────┘
       │ score+q   │ score+q   │ score+q   │ score+q     │ score+q
       └───────────┴─────┬─────┴───────────┴─────────────┘
                         ▼
                 ┌───────────────┐
                 │  log-odds     │   z = Σ (wᵢ/w_max)·qualityᵢ·logit(scoreᵢ)
                 │  fusion       │   p = sigmoid(z)
                 └───────┬───────┘
                         ▼
                 ┌───────────────┐
                 │ decision gate │   evidence < floor → INSUFFICIENT_EVIDENCE
                 │               │   p ≤ 0.35 → AUTHENTIC
                 │               │   p ≥ 0.65 → MANIPULATED
                 │               │   else     → UNCERTAIN
                 └───────┬───────┘
                         ▼
              occlusion attribution (XAI, ~16 s)
                         ▼
                    report + UI
```

**Streaming.** While the heavy channels run, the WebSocket streams live frame
data to the browser: face box, rolling pulse-coherence grid, mouth-motion energy,
speech envelope, and per-patch mean RGB. All of it is measured — see §8.

**Latency.** Median **35.5 s**, max **51.0 s** on an idle 12-core CPU, no GPU.
The first request used to cost 91 s; that was entirely TorchScript
deserialisation, now paid at server startup by a warmup hook.

---

## 3. The five channels

### Channel 01 — rPPG (blood-volume pulse)

**Question:** does this face have a heartbeat, consistently, across all of it?

Not "is there a pulse" — a compressed video of a real person often has a weak
one. The discriminative signal is **cross-region coherence**: on a real face,
every patch of skin beats in phase. A grafted region does not share the donor's
circulation, so it drifts.

- CHROM/POS chrominance projection over a **6×5 patch grid** (30 patches, up to
  435 pairwise comparisons)
- Zero-padded FFT with parabolic peak interpolation for heart-rate estimation
- Zero-phase filtering (`filtfilt`) — a causal filter manufactures phase
  dispersion, which is the very thing we measure
- Features: pairwise correlation p25, HR spread, phase dispersion, HRV jump

**One extractor across the whole grid.** Mixing CHROM on some patches and POS on
others produced a corr_p25 of −0.96 that got *worse* with more signal — the
"anti-correlation" was a polarity artefact of mixing projections, not evidence.

### Channel 02 — Lip-sync

**Question:** does the mouth move when the audio says it should?

- Audio demuxed with PyAV; **true container start offset** applied (measured
  −167.7 ms on one recording — assume zero and every lag is wrong by that much)
- Visual signal is mouth-ROI motion energy, a proxy for the MAR derivative
- Both resampled to a common 50 Hz grid, band-passed 1–8 Hz
- Windowed cross-correlation → per-window lag

**`lag_iqr` carries the most weight, not the lag itself.** A constant offset is
an encoding property. Manipulation shows up as *wandering* alignment.

Quality = coverage × (mean peak NCC / reliability threshold). Coverage alone
reported "I ran" as "I know something".

### Channel 03 — Pixel forensics (classical)

**Question:** do the texture statistics of the face regions hang together?

Seven regions (mouth, both eyes, nose, forehead, both cheeks) weighted toward
mouth/eyes per the ExDDV annotation study. Features: Laplacian sharpness ratios,
high-frequency residual, optical-flow warp *spread*, temporal flicker CV,
lighting asymmetry.

Calibrated on SDFVD2.0 (see §5). **Does not vote** — see §6.

### Channel 04 — Frame-by-frame CNN

**Question:** does any individual frame look spliced?

- **EfficientNet-B7**, Seferbekov's DFDC 1st-place solution (TorchScript, Apache-2.0)
- 32 face crops per clip, margin 1.3, 380×380
- Aggregated with **p90**, not the median: a manipulated clip has *some* clearly
  bad frames and the median washes them out
- Calibrated: `sigmoid(a·logit(p90) + b)`, a=0.3613, b=0.5313 — the raw sigmoid
  compresses into 0.007–0.03 on this content and never reaches a useful threshold

This is the only channel with trained weights and a measured baseline, so it
carries the largest prior (0.35) and does most of the work.

### Channel 05 — Synthetic imagery (AI-generated)

**Question:** was this frame generated whole-cloth, rather than altered?

Different question, different input: **full frames, no face crop, no face
required**. Generation artefacts live in background texture and global lighting
coherence, which a face crop discards.

- `Organika/sdxl-detector`, SwinV2, 86.7M params
- Median aggregation (synthesis is a property of *every* frame, unlike splicing)

**Does not vote.** It failed hard — see §6.

---

## 4. Fusion, and the right to abstain

```
z = Σ (wᵢ / w_max) · qualityᵢ · logit(scoreᵢ)
p = sigmoid(z)
evidence_weight = Σ (wᵢ · qualityᵢ) / Σ wᵢ
```

Log-odds, not a weighted mean of probabilities. Two independent channels each
saying 0.7 should be *more* than 0.7 — averaging cannot express that.

**Quality, not the prior, gates influence.** The prior says what a channel is
worth when evidence is equal; quality says how much evidence there actually was.
Contribution is the product, and it is the only number that moved the verdict.

### Four verdicts

| Verdict | Condition | Meaning |
|---|---|---|
| `LIKELY_AUTHENTIC` | p ≤ 0.35 | evidence points real |
| `LIKELY_MANIPULATED` | p ≥ 0.65 | evidence points fake |
| `UNCERTAIN` | 0.35 < p < 0.65 | channels were heard; they disagreed or landed mid-band |
| `INSUFFICIENT_EVIDENCE` | evidence_weight < 0.11 | too little was measurable to ask |

These last two are **different failures** and the UI must never merge them.
"I heard the evidence and it's ambiguous" ≠ "I couldn't see anything."

### The invariant that keeps biting

`evidence_weight` for a lone channel equals its **prior share**. So every time a
channel is added, every existing share shrinks and the floor must be re-derived:

| channels | Σ priors | smallest share | floor |
|---|---|---|---|
| 3 | 1.00 | 0.300 | 0.20 |
| 4 | 1.00 | 0.150 | 0.12 |
| 5 | 1.15 | 0.130 | **0.11** |

Adding the 4th channel silently broke this — the floor stayed at 0.20 while the
smallest share fell to 0.15, which would have made the system abstain *more*
because we added a channel that never votes. A test asserting the **relationship**
(not the number) caught it. That test compared raw priors, which only coincides
with shares while priors sum to 1.0; it now compares shares.

---

## 5. Datasets

| Dataset | Size used | Role | What it actually gave us |
|---|---|---|---|
| **TEST_VIDEOS** | 7 clips (4 fake / 3 real) | Smoke + demo | End-to-end verdicts, latency. Too small for any AUC claim. |
| **DFDC** (via crops) | 22 crops | effb7 in-domain check | AUC **1.000** — and this number is misleading, see below |
| **SDFVD2.0** | 876 clips, 52 subjects | Pixel-forensics calibration | Grouped-CV AUC 0.609 (scale-free) / 0.711 (all features) |
| **DeepSafe benchmark** | 20k | Fusion strategy study | naive fusion 0.667, best single 0.773, GBM meta 0.811 |
| **DeepfakeTIMIT** | sample | Cross-check | — |
| **FaceForensics++** | via pretrained weights | Source of xception/capsule checkpoints | Both disabled after measurement |

### The two dataset traps we hit

**1. SDFVD2.0 leaks twice.** It ships **8 augmented copies** of every clip
(`real_v10_aug_0..7`) and names the manipulated version of subject *N* as `vs{N}`
against real `v{N}`. A random split therefore puts near-duplicate augmentations
of one clip on both sides, *and* the same subject as real-in-train,
fake-in-test. Measured cost of that illusion:

| features | random split | subject-grouped |
|---|---|---|
| all | 0.790 | **0.711** |
| scale-free | 0.614 | **0.609** |

We report the grouped numbers.

**2. In-domain AUC means almost nothing.** effb7 scores **1.000** on DFDC crops
and **0.550** in the wild. Quote the second number.

---

## 6. Admission by measurement — the core principle

> No model enters the verdict without a *measured* improvement over what we
> already have.

Scoring above chance is not the bar. The bar is improving the system.

### What got rejected, and why

| Model | Measured | Verdict |
|---|---|---|
| **xception** | AUC **0.222** on TEST_VIDEOS vs effb7's 0.833 | disabled |
| **capsule** | near-constant output: 0.608 / 0.609 / 0.609 / 0.609 across four different clips | disabled |
| **PhysNet** | −6.78 dB band SNR vs CHROM's −3.26 | excluded from fusion |
| **DeepFakesON-Phys** | 0.134 for all 13 clips — degenerate | excluded |
| **Pixel forensics** | grouped AUC 0.609; **fusing it with effb7 moved AUC 0.670 → 0.639** | reports, does not vote |
| **Synthetic imagery** | AUC **0.000** on video — perfectly inverted | reports, does not vote |

### The pixel-forensics result in detail

On the 281 clips where both channels scored, subject-grouped:

| | AUC |
|---|---|
| pixel alone | 0.586 |
| effb7 alone | 0.670 |
| effb7 + pixel | **0.639** |
| delta | **−0.031** |

Error correlation was only −0.079 — the channels *are* decorrelated, which is
normally the good case for fusion. It still fails: the channel is too weak to pay
for 18 features on 281 clips. Different failure mode from the effb7×LIPINC case
(+0.596 correlated errors), same conclusion.

### The synthetic-imagery result in detail

AUC 0.000 is not "slightly wrong", it is inverted — every *real* clip scored
0.998–1.000 "synthetic". We isolated the cause by degrading a **known real
photograph** toward video quality:

| condition | P(synthetic) |
|---|---|
| original 780×438 | 0.0000 |
| resized 854×480, no compression | 0.0000 |
| resized 854×480 **+ JPEG q60** | **0.6355** |
| resized 1282×720 + JPEG q30 | 0.3826 |

**Resizing changes nothing. Compression is the entire effect.** The model was
trained on pristine Wikimedia photos against SDXL renders, so it learned
"compression artefacts ⇒ synthetic". Every H.264 frame saturates it.

This is the same species of confound as the resolution artefact in pixel
forensics — a feature that separates classes in the training corpus for a reason
unrelated to the thing we claim to detect. Consequently the channel's score is
pinned at a neutral 0.5 and the raw output is kept only as a diagnostic, because
on compressed video that number means "this frame was compressed."

---

## 7. Explainability (XAI)

**Occlusion sensitivity, not Grad-CAM.** effb7 ships as a frozen TorchScript
graph — there are no named layers to hook for gradients. Occlusion needs only
forward passes: mask a region, measure how far P(manipulated) moves. The result
is a *measured* score delta rather than a gradient approximation.

The regions are the **same seven** that pixel forensics samples and the rPPG grid
covers, so all channels answer "where?" on one coordinate system. Cost: 4
baseline + 28 occluded forwards.

A large delta means the region drove *this score*. It is **not** evidence the
region was manipulated, and the UI says so.

---

## 8. Honesty log — fabrications found and removed

This project accumulated placeholder data that looked like measurement. Each of
these shipped at some point and was removed:

| Fabrication | Why it mattered |
|---|---|
| `mar = 0.25 + 0.05·sin(t·6)` streamed live | a decorative sine animated under the user's own video, captioned as their mouth movement |
| `audio_envelope = 0.1 + 0.1·sin(t·4)` | same, captioned as their audio |
| Checkerboard `coherence_map` fallback | a fabricated heatmap rendered as measured pulse coherence |
| Green/amber grid pulse before coherence existed | used the *exact* palette of the real overlay, so the first 12 s read as findings |
| Hardcoded `xception: 0.15`, `capsule: 0.12` | both models are disabled; the numbers were invented |
| `?? "0.000"` fallback in the report | rendered "measured, found nothing" for models that never ran |
| `hr = 74.0`, `lag = 12.0`, `iqr = 18.0` defaults | plausible-looking vitals for clips where nothing was measured |
| ThreatMatrix "DETECTED" on 4 attack classes | including Face2Face, whose stated vector described a channel that does not exist, and Sora/Flux, never tested |

The rule that emerged: **absent evidence and zero evidence are different claims,
and the UI must render them differently.**

---

## 9. Current honest performance

**TEST_VIDEOS, 7 clips, full pipeline:**

| clip | truth | verdict | p(fake) |
|---|---|---|---|
| FAKE | FAKE | LIKELY_MANIPULATED | 0.870 |
| FAKE2 | FAKE | LIKELY_MANIPULATED | 0.706 |
| FAKE4 | FAKE | **LIKELY_AUTHENTIC** ✗ | 0.323 |
| FAKE5 | FAKE | LIKELY_MANIPULATED | 0.661 |
| REAL | REAL | LIKELY_AUTHENTIC | 0.315 |
| REAL3 | REAL | LIKELY_AUTHENTIC | 0.291 |
| REAL5 | REAL | LIKELY_AUTHENTIC | 0.184 |

**6/7 correct.** All 3 real clips correct. One false negative.

**n=7 supports no AUC claim.** The defensible numbers are: effb7 in-the-wild
0.550, pixel forensics grouped 0.609, LIPINC 0.717 on our corpus.

**The binding constraint is the corpus, not the models.**

---

## 10. Q&A

**Q: Your rPPG channel scored 0.500 on five of seven clips. Is it working?**
0.500 is the abstention value — it means the channel found insufficient pulse SNR
and declined to vote, not that it found nothing suspicious. On compressed
in-the-wild video that is the honest and common outcome. Where it did have
signal (FAKE2, quality 0.136) it contributed 5.4% of the verdict. We would rather
show three abstentions than three invented scores.

**Q: So effb7 is doing all the work. Why bother with five channels?**
Largely yes — on this corpus effb7 supplies ~55% of the contribution and lip-sync
~39%. The multi-channel design earns its keep in two ways: it degrades instead of
failing when a channel is blind (silent clip, no face, heavy compression), and it
makes the failure *visible* rather than silently folding a guess into the score.

**Q: Two of five channels don't vote. Isn't that just dead weight?**
They are honest scaffolding with measured reasons. Pixel forensics is calibrated
and one better dataset away from voting — flipping `admitted: true` in the stored
calibration turns it on with **no code change**. The synthetic channel needs a
compression-augmented model. Shipping them silently voting would have made the
system worse; we measured that directly (0.670 → 0.639).

**Q: Why is your AUC 1.000 on DFDC but 0.550 in the wild?**
Because in-domain evaluation measures memorisation of a distribution, not
detection. We quote 0.550.

**Q: What stops you from claiming a detection you can't support?**
Four things: quality-weighted fusion (a blind channel cannot drag the verdict),
the evidence floor (below 0.11 we return INSUFFICIENT_EVIDENCE), the UNCERTAIN
band, and admission by measurement (§6).

**Q: What's the single biggest weakness?**
Corpus size. Every operating point is fitted on tens of clips, and the leave-one-out
number is always worse than the fitted one (learned meta: fitted 0.700, LOO 0.300).
We report LOO.

**Q: Can it detect a Sora or Midjourney video?**
Unknown, and we say so. `effb7` detects *face manipulation*, not synthesis. The
dedicated synthetic channel exists but failed its compression test on video. The
ThreatMatrix marks that row **NOT EVALUATED** rather than claiming coverage.

**Q: How fast, and does it need a GPU?**
No GPU. Median 35.5 s, max 51.0 s per clip on 12 CPU cores. The heavy costs are
effb7 (~21 s for 32 frames) and occlusion attribution (~16 s).

**Q: Why abstain at all? Judges want an answer.**
A detector that always answers is a detector that produces confident errors on the
hardest cases — which are exactly the cases that matter. An abstention is a
correct, actionable output: it says "escalate this to a human."

---

## 11. What we would do next

1. **A real corpus.** Every limitation above traces back to n. Cross-manipulation
   (leave-one-method-out) AUC is the number to chase.
2. **Retrain the synthetic channel with compression augmentation** and validate on
   generated video at realistic bitrates.
3. **Refit pixel forensics** on a corpus without the augmentation/subject leak, and
   re-run the admission test.
4. **Cut effb7 latency** — 32 frames at 380px dominates the budget. A frame-count
   reduction requires refitting the p90 calibration, so it is a measurement task.
