"""Quality-weighted fusion scorer — plan.md §3.5.

OWNERSHIP: Role 1 owns this module from T+9 (plan §2). Role 3 is the designated
backup and must have read it before anyone sleeps (§5). What is here is the
formula transcribed verbatim from the plan so the contract tests have something
real to assert against — tune the thresholds in config/thresholds.yaml, not here.

Deliberately NOT a learned model. With ~28 compression-confounded clips a logistic
regression memorises camera and codec rather than manipulation (plan §0.1, C5).
A hand-weighted mean is low-variance by construction and implements the graceful
degradation rule directly instead of bolting it on afterwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import yaml

from src.common.contracts import FusionResult, Verdict

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "thresholds.yaml"


def load_thresholds(path: Optional[Path] = None) -> dict:
    """Single source of truth. No magic numbers in code (plan §3.5)."""
    with open(path or _DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def score(
    modality_scores: Mapping[str, float],
    modality_quality: Mapping[str, float],
    session_id: str,
    total_processing_time_ms: int = 0,
    warnings: Optional[list[str]] = None,
    thresholds: Optional[dict] = None,
) -> FusionResult:
    """Fuse per-modality manipulation scores into one verdict.

    Args:
        modality_scores:  {"rppg": P(manipulated), "lipsync": P(manipulated)}
                          Higher = more suspicious. See the polarity rule in contracts.py.
        modality_quality: {"rppg": evidence strength, "lipsync": evidence strength}
                          A modality with quality 0 contributes nothing and cannot
                          drag the verdict — that is the whole point of the design.
    """
    cfg = thresholds or load_thresholds()
    decision = cfg["decision"]
    weights: Dict[str, float] = cfg["fusion"]["prior_weights"]
    warnings = list(warnings or [])

    num = sum(weights[m] * modality_quality[m] * modality_scores[m] for m in weights)
    den = sum(weights[m] * modality_quality[m] for m in weights)

    evidence_weight = den / sum(weights.values())
    p = num / den if den > 1e-6 else 0.5

    # --- decision path: the branch taken is what the explanation is built from ---
    if evidence_weight < decision["min_evidence_weight"]:
        verdict = Verdict.INSUFFICIENT_EVIDENCE
        reason = "insufficient_evidence"
    elif p <= decision["authentic_max_p"]:
        verdict = Verdict.LIKELY_AUTHENTIC
        reason = "below_authentic_threshold"
    elif p >= decision["manipulated_min_p"]:
        verdict = Verdict.LIKELY_MANIPULATED
        reason = "above_manipulated_threshold"
    else:
        verdict = Verdict.UNCERTAIN
        reason = "between_thresholds"

    usable = {m: q for m, q in modality_quality.items() if q > 0.0}
    if not usable:
        warnings.append("no_usable_modality")

    return FusionResult(
        session_id=session_id,
        verdict=verdict,
        manipulation_probability=round(p, 4),
        evidence_weight=round(evidence_weight, 4),
        modality={
            m: {
                "score": float(modality_scores[m]),
                "quality": float(modality_quality[m]),
                "weight": float(weights[m]),
            }
            for m in weights
        },
        explanation=explain(verdict, reason, p, evidence_weight, modality_quality),
        warnings=warnings,
        total_processing_time_ms=total_processing_time_ms,
    )


def explain(
    verdict: Verdict,
    reason: str,
    p: float,
    evidence_weight: float,
    modality_quality: Mapping[str, float],
) -> str:
    """Build the explanation FROM the decision path — never from a free-form template.

    v1's sample payload claimed a clean pulse next to a score that said otherwise.
    A test asserts the directional claim here always agrees with `verdict`.
    """
    dropped = [m for m, q in modality_quality.items() if q < 0.3]
    caveat = ""
    if dropped:
        caveat = (
            f" {', '.join(sorted(dropped))} contributed little usable evidence and was"
            " down-weighted accordingly."
        )

    if verdict is Verdict.INSUFFICIENT_EVIDENCE:
        return (
            "Not enough usable signal to judge this clip. Neither pulse nor lip "
            f"alignment cleared the evidence threshold (evidence weight {evidence_weight:.2f})."
            " No verdict is offered rather than guessing." + caveat
        )
    if verdict is Verdict.LIKELY_AUTHENTIC:
        return (
            "Signals are consistent with authentic video: blood-flow phase agrees "
            "across facial regions and lip motion tracks speech steadily "
            f"(manipulation probability {p:.2f})." + caveat
        )
    if verdict is Verdict.LIKELY_MANIPULATED:
        return (
            "Signals are inconsistent with authentic video: cross-region pulse "
            "agreement and/or speech-to-lip alignment broke down "
            f"(manipulation probability {p:.2f})." + caveat
        )
    return (
        "Evidence is mixed. The clip sits between the authentic and manipulated "
        f"thresholds (manipulation probability {p:.2f}), so no confident call is made."
        + caveat
    )
