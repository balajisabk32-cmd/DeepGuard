"""Does the pixel channel ADD anything on top of the frame-by-frame CNN?

    python pixel_admission.py

Scoring above chance alone is not the bar. The bar is improving the system we
already have: if pixel forensics and effb7 make correlated errors, fusing them
lowers AUC even though pixel looks informative in isolation. That already
happened once on this project (effb7 x LIPINC error correlation +0.596 made the
fusion worse than LIPINC alone), so it gets measured before anything is admitted.

Everything is evaluated on the SUBJECT-grouped out-of-fold predictions, on the
clips where BOTH channels have a score.
"""

from __future__ import annotations

import json

import numpy as np

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # scripts/ on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibrate_pixel import auc, subject


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = json.load(open("results/sdfvd_features.json", encoding="utf-8"))
    cnn = json.load(open("results/sdfvd_effb7.json", encoding="utf-8"))
    cal = json.load(open("results/pixel_calibration.json", encoding="utf-8"))

    keep = [r for r in rows
            if subject(r["__file"]) is not None and r["__file"] in cnn]
    if len(keep) < 40:
        print(f"only {len(keep)} overlapping clips - too few to decide")
        return 1

    feats = cal["features"]
    X = np.nan_to_num(np.array([[float(r.get(f, 0.0) or 0.0) for f in feats]
                                for r in keep]), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([int(r["__label"]) for r in keep])
    g = np.array([subject(r["__file"]) for r in keep])
    c = np.array([float(cnn[r["__file"]]["score"]) for r in keep])

    for r in keep:
        assert int(r["__label"]) == int(cnn[r["__file"]]["label"]), \
            f"label disagreement on {r['__file']}"

    print(f"{len(keep)} overlapping clips  |  {int((y==1).sum())} fake / "
          f"{int((y==0).sum())} real  |  {len(set(g))} subjects\n")

    gkf = GroupKFold(min(5, len(set(g))))

    def oof(mat):
        p = np.zeros(len(y))
        for tr, te in gkf.split(mat, y, g):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000))
            m.fit(mat[tr], y[tr])
            p[te] = m.predict_proba(mat[te])[:, 1]
        return p

    p_pix = oof(X)
    p_cnn = c                                   # already a model output
    p_both = oof(np.column_stack([X, c]))

    a_pix, a_cnn, a_both = auc(y, p_pix), auc(y, p_cnn), auc(y, p_both)

    print(f"{'channel':<28}{'AUC':>8}")
    print("-" * 36)
    print(f"{'pixel forensics alone':<28}{a_pix:>8.3f}")
    print(f"{'effb7 alone':<28}{a_cnn:>8.3f}")
    print(f"{'effb7 + pixel':<28}{a_both:>8.3f}")
    print(f"{'delta vs effb7 alone':<28}{a_both - a_cnn:>+8.3f}")

    # Error correlation: the mechanism that decides whether fusing can help.
    e_pix = np.abs(y - p_pix)
    e_cnn = np.abs(y - (p_cnn - p_cnn.min()) / max(float(np.ptp(p_cnn)), 1e-9))
    rho = float(np.corrcoef(e_pix, e_cnn)[0, 1])
    print(f"\nerror correlation pixel vs effb7: {rho:+.3f}")
    print("  (high positive => same clips fail => little to gain from fusing)")

    improves = a_both > a_cnn + 0.01
    print(f"\nADMIT pixel as a voting channel: {improves}")
    if not improves:
        print("  -> stays reporting-only (quality 0). It is shown in the output,\n"
              "     with its calibrated score, but contributes 0 to the verdict.")

    json.dump({"n": len(keep), "auc_pixel": round(a_pix, 4),
               "auc_effb7": round(a_cnn, 4), "auc_fused": round(a_both, 4),
               "delta": round(a_both - a_cnn, 4),
               "error_correlation": round(rho, 4), "admitted": bool(improves)},
              open("results/pixel_admission.json", "w", encoding="utf-8"), indent=2)
    print("\nwrote results/pixel_admission.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
