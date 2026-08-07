"""Full four-channel pipeline over TEST_VIDEOS, with per-stage latency.

    python full_pipeline_test.py

Records, per clip: every channel's score/quality, the fused verdict, the
attribution top region, and the wall time of each stage. The latency columns
exist because the interactive path has a 60 s budget — a correctness run that
does not measure time cannot tell us whether we can ship it.

Ground truth is taken from the filename (FAKE* / REAL*), which is how the rest
of the eval scripts in this repo label this corpus.
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

import glob
import json
import os
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

OUT = "results/full_pipeline_test.json"


def main() -> int:
    from src.fusion import load_thresholds
    from src.lipsync import analyze as lipsync_analyze
    from src.pipeline.decode import decode_clip
    from src.pipeline.detect import detect
    from src.rppg import analyze as rppg_analyze
    from src.visual.explain import region_attribution

    cfg = load_thresholds()
    vids = sorted(glob.glob("TEST_VIDEOS/*.mp4"))
    print(f"{len(vids)} clips\n")
    hdr = (f"{'clip':<14}{'truth':<6}{'verdict':<22}{'p(fake)':>8}{'ew':>6}"
           f"{'rppg':>7}{'lip':>7}{'pix':>7}{'cnn':>7}{'top region':>12}{'sec':>7}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for v in vids:
        name = os.path.basename(v).replace(".mp4", "")
        truth = "FAKE" if name.upper().startswith("FAKE") else "REAL"
        t0 = time.time()
        try:
            clip = decode_clip(v, max_sec=20.0)
            t_dec = time.time() - t0

            t = time.time()
            fused, r, l, vv, px = detect(v, session_id=name, thresholds=cfg, clip=clip)
            t_det = time.time() - t

            t = time.time()
            att = region_attribution(clip.frames, clip.boxes)
            t_att = time.time() - t

            wall = time.time() - t0
            row = {
                "clip": name, "truth": truth,
                "verdict": fused.verdict.value,
                "p_fake": round(fused.manipulation_probability, 4),
                "evidence_weight": round(fused.evidence_weight, 4),
                "rppg": [round(r.rppg_manipulation_score, 3), round(r.rppg_quality, 3)],
                "lipsync": [round(l.lipsync_manipulation_score, 3), round(l.lipsync_quality, 3)],
                "pixel": [round(px.pixel_manipulation_score, 3), round(px.pixel_quality, 3)],
                "visual": [round(vv.visual_manipulation_score, 3), round(vv.visual_quality, 3)],
                "top_region": att.top_region,
                "attribution": att.regions,
                "sec": {"decode": round(t_dec, 1), "detect": round(t_det, 1),
                        "attribution": round(t_att, 1), "wall": round(wall, 1)},
                "warnings": list(fused.warnings or []),
            }
            print(f"{name:<14}{truth:<6}{fused.verdict.value:<22}"
                  f"{fused.manipulation_probability:>8.3f}{fused.evidence_weight:>6.2f}"
                  f"{r.rppg_manipulation_score:>7.3f}{l.lipsync_manipulation_score:>7.3f}"
                  f"{px.pixel_manipulation_score:>7.3f}{vv.visual_manipulation_score:>7.3f}"
                  f"{str(att.top_region):>12}{wall:>7.1f}")
        except Exception as exc:  # noqa: BLE001
            row = {"clip": name, "truth": truth, "error": f"{type(exc).__name__}: {exc}"}
            print(f"{name:<14}{truth:<6}ERROR {type(exc).__name__}: {exc}")
        rows.append(row)

    ok = [r for r in rows if "error" not in r]
    # Confident calls only. Abstentions are a correct output, so they are counted
    # separately rather than scored as wrong.
    conf = [r for r in ok if r["verdict"] in ("LIKELY_AUTHENTIC", "LIKELY_MANIPULATED")]
    correct = [r for r in conf
               if (r["verdict"] == "LIKELY_MANIPULATED") == (r["truth"] == "FAKE")]
    walls = [r["sec"]["wall"] for r in ok]

    print("\n" + "=" * 72)
    print(f"clips           {len(ok)}/{len(rows)} completed")
    print(f"confident       {len(conf)}  ({len(correct)} correct, {len(conf)-len(correct)} wrong)")
    print(f"abstained       {len(ok)-len(conf)}  (UNCERTAIN / INSUFFICIENT_EVIDENCE)")
    if walls:
        walls_sorted = sorted(walls)
        print(f"latency         median {walls_sorted[len(walls)//2]:.1f}s   "
              f"max {max(walls):.1f}s   over-60s {sum(w > 60 for w in walls)}/{len(walls)}")
    print("=" * 72)

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
