# DeepGuard — Revised Delivery Plan

**Problem statement mandates four channels:** rPPG · lip-sync · traditional pixel comparison · frame-by-frame detection. Everything below is organised around delivering all four, visibly, with honest numbers.

---

## 0. Where we actually stand (read before planning)

### Measured, current

| channel | state | measured |
|---|---|---|
| Frame-by-frame CNN (effb7) | **working** | AUC **1.000** in-domain (DFDC crops) · **0.550** in-the-wild (13 clips) |
| Lip-sync (LIPINC-V2) | working, slow | AUC **0.717**, ~150 s/clip, abstains on 2 of 13 |
| Lip-sync (classical MAR×envelope) | runs, non-discriminative | `lag_iqr` saturates ~205 ms on every clip |
| rPPG (CHROM/POS + 30-patch coherence) | runs, abstains | near-zero band SNR on every clip tested |
| **Traditional pixel comparison** | **DOES NOT EXIST** | — see §Phase 4 gap |
| Fusion (quality-weighted log-odds) | working | 0.700 on 11 clips |

### The headline finding — this is the pitch

> **AUC 1.000 on the DFDC benchmark, 0.550 on in-the-wild footage.**
> That gap is exactly why a single-model detector is not enough, and it is the justification for the four-channel architecture.

### Hard-won constraints — do not re-litigate these

- **Corpus is 13 clips.** It cannot distinguish 0.667 from 0.717, cannot support learned fusion weights, and turned an apparent 0.889 into a measured 0.550 when 4 clips were added.
- **Learned fusion is off the table.** Validated on a 20,000-sample benchmark: naive averaging costs 11 AUC points, learned weights gain 3.8. On our 11 clips the same method scores 0.700 fitted and **0.300 leave-one-out**. Hand-set priors stay.
- **Six pretrained models failed** on this footage (xception 0.222, capsule constant, PhysNet worse than CHROM, DeepFakesON-Phys degenerate, plus two with no weights). Stop adding models.
- **Never "improve" a trained model's preprocessing.** Measured: a preprocessing change flipped one clip from 0.974 → 0.029, and a MediaPipe reimplementation of dlib alignment returned ~0.000 on everything.

---

## Phase 1 — Split the landing page from the product

**Goal:** the landing page sells; a separate page does the work. "Launch DeepGuard" navigates, it does not analyse in place.

| # | Task | Detail |
|---|---|---|
| 1.1 | Create `/app/detect/page.tsx` | The real product page. Owns upload, progress, visuals, report. |
| 1.2 | Strip `InferenceSandbox` from `app/page.tsx` | Landing keeps Hero, BentoGrid, ThreatMatrix, Footer only. |
| 1.3 | Wire "Launch DeepGuard" → `/detect` | Next `<Link>`, no modal, no in-page analysis. |
| 1.4 | Move `ForensicReport` under `/detect` | It is a result view, not a landing overlay. |
| 1.5 | Keep `data-lenis-prevent` on the report | Already fixed; Lenis otherwise swallows its scroll. |

**Acceptance:** landing page makes zero network calls. `/detect` runs a full analysis end to end. Browser back returns to landing with no lost state.

---

## Phase 2 — De-slop the landing page

**Goal:** every number on screen is real or clearly labelled as illustrative. No invented metrics.

### Known fabrications to remove

| location | current | replace with |
|---|---|---|
| `Hero.tsx` | *(fixed)* random-jitter metrics via `Math.random()` every 1.4 s | already reads real last-analysis values; verify it shows `—` / `AWAITING UPLOAD` pre-run |
| `BentoGrid.tsx` | `EAR 0.31`, `BLINK 19/min`, `REFLECT OK` — hardcoded | either real values from a bundled sample analysis, or drop the tiles |
| `BentoGrid.tsx` | "Ocular Physics / Corneal Reflection Detector" | **we do not have this.** Remove or move to Roadmap |
| `BentoGrid.tsx` | fusion-weight slider (`rPPG 65%`, `SYNC 52%`, `OCULAR 40%`) | show the **real** priors from `config/thresholds.yaml` |
| `ThreatMatrix.tsx` | attack-coverage "Status" column implying measured results | mark unmeasured rows **"not evaluated"** — Class E was never tested |
| `Hero.tsx` | "un-hackable" | remove. Indefensible and invites attack |

### Content that should replace it

Real, and stronger than the mockups:

- **AUC 1.000 benchmark / 0.550 in-the-wild** — the generalization gap, stated plainly
- **Six detectors evaluated, five rejected on measurement** — with the numbers
- **Fusion validated on 20,000 labelled samples**: naive averaging −11 AUC, learned +3.8
- **Abstention as design**: four verdicts, and why calling a real person fake is the worse error

**Acceptance:** every figure on the landing page traces to a measurement in `results/` or `config/`. Grep for hardcoded percentages returns nothing unexplained.

---

## Phase 3 — `/detect`: import, live visuals, fast result

**Goal:** upload → immediate real visual feedback → result. The visuals cover latency *and* are genuine evidence, not a loading animation.

### 3.1 Latency budget (current ~15–20 s)

| stage | now | target | how |
|---|---|---|---|
| decode + face track | ~3 s + ~7 s | ~5 s | reuse `decode_clip` boxes everywhere; stride tuning |
| effb7 frame scoring | ~6–10 s | ~4 s | 32 → 16 frames (p90 aggregation tolerates it) |
| rPPG coherence | ~2 s | ~2 s | already streamed live |
| classical lip-sync | ~2 s | ~2 s | — |
| **total** | **15–20 s** | **≤ 10 s** | |

LIPINC (150 s) stays **out of the interactive path** — opt-in "deep scan" only.

### 3.2 Visuals during processing — all real, none decorative

Phase 2 removes fake data; the loading screen must obey the same rule.

| visual | source | when |
|---|---|---|
| Face tracking box | `frame_data.box` (already streamed) | immediate, < 2 s |
| **RGB channel means per ROI** | `clip.roi_series` — the actual rPPG input | immediate, streaming |
| 6×5 pulse-coherence grid | `CoherenceTracker`, already wired | fills ~⅔ through (needs 12 s buffer — say so in the UI) |
| Per-frame CNN scores | effb7 outputs as computed | streaming |
| Audio envelope + mouth motion | PyAV + MAR series | streaming |

**Replace the synthetic `mar`/`audio_envelope` sine waves in `api.py` with the real series.** They are currently `math.sin()` placeholders.

### 3.3 Honest UX

- Coherence grid shows **"buffering — needs 12 s of signal"**, not a fake animation. Physiology needs a time window; saying so is more impressive than faking it.
- Progress phases name what is actually running.

**Acceptance:** first real visual < 2 s. Full result ≤ 10 s. No `Math.random()` or `math.sin()` in any displayed value.

---

## Phase 4 — Show all four channel weights in the output

**Goal:** the output makes the four-channel novelty visible and inspectable.

### 4.1 The gap — this is new work, not wiring

The problem statement lists **four** channels. We have **three**:

```
rPPG                     ✅ exists (abstains on this footage)
lip-sync                 ✅ exists (classical + LIPINC)
frame-by-frame CNN       ✅ exists (effb7)
traditional pixel        ❌ DOES NOT EXIST
```

`visual` is currently the CNN — that *is* the frame-by-frame channel. "Traditional pixel comparison" must be a separate, classical-forensics channel.

**Build it as classical CV — no training, no weights, no new licence risk:**

| technique | detects |
|---|---|
| Error Level Analysis (ELA) | recompression inconsistency between face and background |
| DCT / frequency-domain residual | GAN upsampling fingerprints |
| Noise-residual consistency | face region vs surrounding scene |
| Blending-seam edge analysis | discontinuity at the swap boundary |

This is genuinely traditional forensics, it is fast, it is independent of the CNN, and it is the channel most likely to have **decorrelated errors** — which is exactly the condition our measurements showed is required for fusion to help.

### 4.2 Output panel

Every channel reports **score + quality + weight + contribution**:

| channel | score | quality | prior | contribution | status |
|---|---|---|---|---|---|
| rPPG | 0.500 | 0.03 | 0.25 | ~0 | abstained — no recoverable pulse |
| Lip-sync | 0.546 | 1.00 | 0.25 | +0.13 | contributing |
| Traditional pixel | — | — | 0.25 | — | *(Phase 4 build)* |
| Frame-by-frame | 0.650 | 1.00 | 0.25 | +0.62 | dominant |

Plus: fused P(manipulated), evidence weight, verdict, plain-English explanation.

**Show abstention explicitly.** A channel reporting "no usable evidence" is a feature — it is why the system does not guess.

**Acceptance:** all four channels visible with non-fabricated values. `config/thresholds.yaml` priors updated to four channels, `min_evidence_weight` kept **below** the smallest prior.

---

## Demo set (curated, and labelled as such)

Clips where the full pipeline currently produces correct verdicts:

| clip | verdict | P(fake) |
|---|---|---|
| **FAKE2** | LIKELY_MANIPULATED | 70.9% |
| **REAL5** | LIKELY_AUTHENTIC | 18.1% |
| REAL3 | LIKELY_AUTHENTIC | 29.4% |
| REAL | LIKELY_AUTHENTIC | 31.5% |

FAKE2 → REAL5 is the verdict flip. Curation is fine; **state the 0.550 alongside it.** A curated demo plus the honest generalization number is evidence. A curated demo presented as a hit rate collapses under one question.

---

## Sequencing and risk

| phase | effort | risk | blocking? |
|---|---|---|---|
| 1 — page split | low | low | no |
| 2 — de-slop | low–med | low | no |
| 3 — visuals + latency | **medium** | medium — latency target may need frame cuts | no |
| 4 — traditional pixel channel | **medium–high** | **highest — new detector, needs calibration** | **yes, for the problem statement** |

**Do Phase 4 first if time is short.** Phases 1–3 are presentation; Phase 4 is the missing mandated capability. A polished three-channel demo does not satisfy a four-channel problem statement.

## Must-fix regardless

`PROJECT_OVERVIEW.md` still claims **AUC 0.889** and **"0 incorrect confident calls"** from the 9-clip set. On the current 13 clips the honest figures are **0.550** (effb7) and **0.700** (fused). A judge can run this. Correct it before any review.
