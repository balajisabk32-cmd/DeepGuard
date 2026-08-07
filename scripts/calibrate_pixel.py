"""Calibrate the traditional pixel-forensics channel on SDFVD2.0.

    python calibrate_pixel.py

WHY THE SPLIT IS GROUPED
SDFVD2.0 ships 8 augmented copies of every source clip (`real_v10_aug_0..7`) and
names the manipulated version of subject N as `vs{N}` against real `v{N}`. A
random split therefore leaks twice over: near-duplicate augmentations of one clip
land on both sides, and the same subject appears as real in train and fake in
test. Both inflate AUC without improving anything.

Groups are the SUBJECT id shared by both classes, so a subject is wholly in train
or wholly in test. The script reports the random-split number too, purely to show
how large the illusion is — that number is never used for a decision.

THE RESOLUTION CONFOUND
Absolute sharpness/high-frequency energy partly encodes capture resolution, and
the manipulated clips in this corpus may be upscaled. A detector that learns
"blurrier => fake" would score well here and fail everywhere else. So two feature
sets are fitted and reported separately:
    all    - every feature, confounded
    robust - scale-free only (sharpness RATIOS, warp CVs, flicker, asymmetry)
The robust set is what ships, even though it scores lower.

ADMISSION BY MEASUREMENT
The channel only earns a vote if the honest grouped AUC clears ADMIT_AUC. Below
that it stays reporting-only with quality 0, exactly as it is today.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Scripts live in scripts/ but resolve paths and imports against the
# REPO ROOT, so they behave identically no matter where they are invoked.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import json
import re

import numpy as np

FEATURES_JSON = "results/sdfvd_features.json"
OUT_JSON = "results/pixel_calibration.json"
ADMIT_AUC = 0.65      # below this the channel is not worth a vote
N_SPLITS = 5
SEED = 0


def subject(fname: str) -> str | None:
    """Subject id shared between the real and manipulated version of a clip."""
    for pat in (r"^fake_vs(\d+)_", r"^vs(\d+)\.", r"^real_v(\d+)_", r"^v(\d+)\."):
        m = re.match(pat, fname)
        if m:
            return m.group(1)
    return None


def auc(y, s) -> float:
    """Rank-based AUC; ties averaged. Avoids a sklearn dependency here."""
    y = np.asarray(y)
    order = np.argsort(np.asarray(s, dtype=float))
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within tied score groups
    s_sorted = np.asarray(s, dtype=float)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = json.load(open(FEATURES_JSON, encoding="utf-8"))
    names = [k for k in rows[0] if not k.startswith("__") and k != "frames_used"]
    robust = [n for n in names
              if n.startswith(("sharp_ratio_", "warp_cv_", "flicker_"))
              or n == "lighting_asymmetry"]

    keep = [r for r in rows if subject(r["__file"]) is not None]
    X_all = np.array([[float(r.get(n, 0.0) or 0.0) for n in names] for r in keep])
    y = np.array([int(r["__label"]) for r in keep])
    g = np.array([subject(r["__file"]) for r in keep])
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"{len(keep)} clips  |  {int((y==1).sum())} fake / {int((y==0).sum())} real"
          f"  |  {len(set(g))} subjects")
    print(f"features: {len(names)} all, {len(robust)} scale-free\n")

    def cv_auc(X, splitter, groups=None) -> tuple[float, float, np.ndarray]:
        oof = np.zeros(len(y))
        for tr, te in splitter.split(X, y, groups):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=1.0))
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        # per-fold spread says whether one lucky fold is carrying the mean
        folds = [auc(y[te], oof[te]) for _, te in splitter.split(X, y, groups)]
        return auc(y, oof), float(np.std(folds)), oof

    idx_robust = [names.index(n) for n in robust]
    results = {}
    print(f"{'feature set':<10}{'split':<10}{'AUC':>8}{'fold sd':>10}")
    print("-" * 38)
    for tag, cols in (("all", list(range(len(names)))), ("robust", idx_robust)):
        X = X_all[:, cols]
        a_rand, sd_rand, _ = cv_auc(X, KFold(N_SPLITS, shuffle=True, random_state=SEED))
        a_grp, sd_grp, oof = cv_auc(X, GroupKFold(N_SPLITS), groups=g)
        print(f"{tag:<10}{'random':<10}{a_rand:>8.3f}{sd_rand:>10.3f}   <- LEAKY, not used")
        print(f"{tag:<10}{'grouped':<10}{a_grp:>8.3f}{sd_grp:>10.3f}")
        results[tag] = {"auc_grouped": round(a_grp, 4), "fold_sd": round(sd_grp, 4),
                        "auc_random_leaky": round(a_rand, 4), "oof": oof}

    chosen = "robust"
    a = results[chosen]["auc_grouped"]
    admitted = a >= ADMIT_AUC
    print(f"\nshipping feature set: {chosen}  (AUC {a:.3f}, gate {ADMIT_AUC})")
    print(f"ADMITTED TO FUSION: {admitted}")

    # Final model on all data, exported as plain coefficients so inference needs
    # no sklearn at runtime: p = sigmoid(w . ((x - mean) / scale) + b)
    Xr = X_all[:, idx_robust]
    scaler = StandardScaler().fit(Xr)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(Xr), y)

    payload = {
        "version": 1,
        "dataset": "SDFVD2.0",
        "n_clips": len(keep), "n_subjects": len(set(g)),
        "feature_set": chosen,
        "features": robust,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "auc_grouped": results[chosen]["auc_grouped"],
        "auc_grouped_all_features": results["all"]["auc_grouped"],
        "auc_random_leaky": results[chosen]["auc_random_leaky"],
        "fold_sd": results[chosen]["fold_sd"],
        "admit_gate": ADMIT_AUC,
        "admitted": bool(admitted),
        "notes": ("Grouped by subject id so augmentations and the real/fake pair of "
                  "the same subject never span the split. Scale-free features only, "
                  "to avoid learning capture resolution as a forgery cue."),
    }
    for k in ("all", "robust"):
        results[k].pop("oof", None)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {OUT_JSON}")

    top = sorted(zip(robust, lr.coef_[0]), key=lambda kv: -abs(kv[1]))[:8]
    print("\nstrongest coefficients (standardised):")
    for n, c in top:
        print(f"   {n:<24}{c:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
