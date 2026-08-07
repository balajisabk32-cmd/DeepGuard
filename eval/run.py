"""Evaluation harness — the CP2/CP3 verification command (plan §4.3, §5).

    python -m eval.run --split dev     --out results/cp2.md
    python -m eval.run --split holdout --out results/cp3.md

REPORTING RULES BAKED IN (plan §4.2):
  - AUC is reported with a bootstrap CI and is NOT a pass/fail gate. At n=10 the
    CI spans roughly 0.4–1.0, so gating on it passes or fails near-randomly.
  - Abstention rate is a headline metric, not a hidden failure.
  - Any Class-F clip returning LIKELY_MANIPULATED is a P0 bug: those are authentic
    clips shot in adverse conditions and must degrade to UNCERTAIN.

HOLDOUT DISCIPLINE: open it exactly twice. The CP2 look is diagnostic-only and
contaminates everything after it; the CP3 number is the reportable one and carries
a footnote saying so.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "corpus" / "MANIFEST.csv"


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepGuard evaluation harness")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = load_manifest()
    if not rows:
        print(
            "MANIFEST.csv is empty - the corpus is the #1 Hour-0 blocker (plan H0-C).\n"
            "Nothing to evaluate. Populate data/corpus/ and MANIFEST.csv first.",
            file=sys.stderr,
        )
        return 1

    # Scoring loop lands with the real pipeline in Phase 2.
    raise NotImplementedError(
        f"{len(rows)} clips in manifest; scoring loop is wired at T+9 (plan §5)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
