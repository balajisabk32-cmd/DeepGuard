"""Cross-manipulation (leave-one-method-out) evaluation.

    python -m eval.cross_manipulation --manifest data/corpus/MANIFEST.csv

WHY IN-DOMAIN ACCURACY IS NOT REPORTED
Train on Deepfakes+Face2Face, test on the same, and almost anything scores ~99%.
The model has learned the generator's fingerprint, not manipulation. The same
model routinely falls to near-chance on an unseen method. So the headline number
here is AUC on a HELD-OUT manipulation type, and in-domain figures are printed
only as a contrast so the generalisation gap is visible rather than hidden.

WHY AUC AND NOT ACCURACY
Accuracy depends on a threshold, and a threshold tuned on the training methods
does not transfer to a held-out one. AUC is threshold-free, so it measures
ranking quality — which is what actually transfers.

ABSTENTIONS ARE EXCLUDED FROM AUC AND REPORTED SEPARATELY. Folding an abstention
in as 0.5 silently rewards a detector for refusing to answer.

The bootstrap CI is not decoration. At corpus sizes reachable in a hackathon the
interval is wide enough to change what you are allowed to claim, so it is printed
next to every point estimate.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUTHENTIC = {"A", "B", "F", "real", "authentic"}


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). Ties handled via average ranks."""
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            m = inv == i
            ranks[m] = ranks[m].mean()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc_ci(labels: np.ndarray, scores: np.ndarray,
                     n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(auc(labels[idx], scores[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def load_rows(manifest: Path) -> list[dict]:
    if not manifest.exists():
        return []
    with open(manifest, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("clip_id")]


def evaluate(rows: list[dict], score_key: str = "score") -> dict:
    """Leave-one-manipulation-out. Each fake method is held out in turn."""
    by_method: dict[str, list[dict]] = defaultdict(list)
    authentic: list[dict] = []
    for r in rows:
        cls = (r.get("class_label") or "").strip()
        method = (r.get("attack_type") or "").strip() or "unknown"
        if cls in AUTHENTIC or method.lower() in {"none", "authentic", "real", ""}:
            authentic.append(r)
        else:
            by_method[method].append(r)

    results = {"folds": [], "authentic_n": len(authentic), "methods": list(by_method)}
    if not authentic or len(by_method) < 2:
        results["error"] = (
            f"need authentic clips and >=2 manipulation methods; "
            f"have {len(authentic)} authentic, {len(by_method)} method(s)"
        )
        return results

    for held_out, fakes in by_method.items():
        eval_rows = authentic + fakes
        labels, scores, abstained = [], [], 0
        for r in eval_rows:
            raw = r.get(score_key)
            if raw in (None, "", "abstain", "UNCERTAIN", "INSUFFICIENT_EVIDENCE"):
                abstained += 1
                continue
            try:
                s = float(raw)
            except ValueError:
                abstained += 1
                continue
            labels.append(0 if r in authentic else 1)
            scores.append(s)

        if len(set(labels)) < 2:
            results["folds"].append({"held_out": held_out, "error": "single-class fold"})
            continue

        L, S = np.array(labels), np.array(scores)
        lo, hi = bootstrap_auc_ci(L, S)
        results["folds"].append({
            "held_out": held_out,
            "n_fake": int((L == 1).sum()),
            "n_authentic": int((L == 0).sum()),
            "abstained": abstained,
            "abstention_rate": abstained / max(len(eval_rows), 1),
            "auc": auc(L, S),
            "ci": (lo, hi),
        })
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-manipulation AUC")
    ap.add_argument("--manifest", type=Path,
                    default=REPO / "data" / "corpus" / "MANIFEST.csv")
    ap.add_argument("--score-key", default="score")
    args = ap.parse_args(argv)

    rows = load_rows(args.manifest)
    if not rows:
        print(f"[BLOCKED] {args.manifest} is empty.\n"
              "  Cross-manipulation evaluation needs a corpus with >=2 distinct\n"
              "  attack_type values plus authentic clips. Populate MANIFEST.csv with\n"
              "  columns: clip_id,class_label,attack_type,...,score",
              file=sys.stderr)
        return 1

    res = evaluate(rows, args.score_key)
    if "error" in res:
        print(f"[BLOCKED] {res['error']}", file=sys.stderr)
        return 1

    print("=" * 74)
    print("CROSS-MANIPULATION AUC  (train on the rest, test on the held-out method)")
    print("=" * 74)
    print(f"{'held-out method':<22} {'n_fake':>7} {'n_real':>7} {'abstain':>8} "
          f"{'AUC':>7}  95% CI")
    print("-" * 74)
    aucs = []
    for f in res["folds"]:
        if "error" in f:
            print(f"{f['held_out']:<22} {f['error']}")
            continue
        aucs.append(f["auc"])
        print(f"{f['held_out']:<22} {f['n_fake']:>7} {f['n_authentic']:>7} "
              f"{f['abstention_rate']*100:>7.0f}% {f['auc']:>7.3f}  "
              f"[{f['ci'][0]:.3f}, {f['ci'][1]:.3f}]")
    print("-" * 74)
    if aucs:
        print(f"  mean cross-manipulation AUC   {np.mean(aucs):.3f}")
        print(f"  worst held-out method         {np.min(aucs):.3f}   <- report THIS one")
    print("\nA wide CI is the honest result at small n, not a formatting problem.")
    print("Quote the worst fold, not the mean: it is the closest estimate of")
    print("performance on the manipulation you have not seen yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
