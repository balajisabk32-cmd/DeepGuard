"""Compression sensitivity probe — plan.md Phase 3 (13:00-16:00) and the Class B control.

    python -m eval.compression_probe REAL.mp4 fake.mp4

Answers one question: when two clips differ in BOTH manipulation and encoding
quality, is the rPPG difference caused by the manipulation or by the encoding?

Method: hold detection fixed (boxes are found once on the original and reused),
then degrade the pixels and re-measure. That isolates SIGNAL degradation from
DETECTION degradation — otherwise a dropped face detection is indistinguishable
from a destroyed pulse.

ffmpeg is not required. Downscale + per-frame JPEG round-trip stands in for
bitrate reduction, which is what actually destroys the rPPG micro-signal.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from src.rppg import backends
from src.rppg.analyze import read_frames
from src.rppg.signal_core import best_extraction, resample_to_uniform

# (label, long-edge cap or None, JPEG quality or None)
LEVELS = [
    ("original",        None, None),
    ("854px",            854, None),
    ("854px q70",        854,   70),
    ("854px q40",        854,   40),
    ("640px q40",        640,   40),
    ("480px q25",        480,   25),
]


def quality_from_snr(snr: float) -> float:
    return float(np.clip((snr + 3.0) / 12.0, 0.0, 1.0))


def degrade(frame: np.ndarray, long_edge: int | None, jpeg_q: int | None) -> np.ndarray:
    out = frame
    if long_edge is not None:
        h, w = out.shape[:2]
        s = long_edge / float(max(h, w))
        if s < 1.0:
            out = cv2.resize(out, (max(int(w * s), 2), max(int(h * s), 2)),
                             interpolation=cv2.INTER_AREA)
    if jpeg_q is not None:
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def probe(video: str, max_sec: float, label: str) -> list[tuple[str, float, float]]:
    frames, t, _nom, _w = read_frames(video, max_sec=max_sec)
    if len(frames) < 60:
        print(f"  {label}: too few frames", file=sys.stderr)
        return []

    fs = (len(frames) - 1) / (t[-1] - t[0])
    h0, w0 = frames[0].shape[:2]

    # Detect once on the originals; reuse for every degradation level.
    boxes, counts = backends.OpenCVBackend().detect_boxes(frames)
    boxes = backends.smooth_boxes(boxes)

    rows = []
    for name, long_edge, jpeg_q in LEVELS:
        deg = [degrade(f, long_edge, jpeg_q) for f in frames]
        sh, sw = deg[0].shape[:2]
        scaled = boxes * np.array([sw / w0, sh / h0, sw / w0, sh / h0])

        series, _ = backends.extract_roi_series(deg, scaled, counts)
        best = -np.inf
        for j in range(len(backends.ROI_NAMES)):
            rgb = np.column_stack(
                [resample_to_uniform(t, series[:, j, c], fs) for c in range(3)]
            )
            if rgb.size == 0 or rgb.shape[0] < 32:
                continue
            _n, _s, _f0, snr = best_extraction(rgb, fs)
            best = max(best, snr)
        rows.append((name, best, quality_from_snr(best)))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="rPPG compression sensitivity probe")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--max-sec", type=float, default=None,
                    help="truncate all clips to this length so duration is not a confound")
    args = ap.parse_args(argv)

    if args.max_sec is None:
        durations = []
        for v in args.videos:
            cap = cv2.VideoCapture(v)
            n, f = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if f > 0:
                durations.append(n / f)
        args.max_sec = min(durations) if durations else 20.0

    print(f"Matched analysis window: {args.max_sec:.1f}s "
          "(duration equalised so it cannot explain the difference)\n")

    results = {v: probe(v, args.max_sec, v) for v in args.videos}

    width = max(len(v) for v in args.videos)
    print(f"{'level':<12} " + " ".join(f"{v[:26]:>28}" for v in args.videos))
    print("-" * (13 + 29 * len(args.videos)))
    for i, (name, _, _) in enumerate(LEVELS):
        cells = []
        for v in args.videos:
            rows = results[v]
            cells.append(f"{rows[i][1]:>9.2f} dB  q={rows[i][2]:<5.3f}" if i < len(rows)
                         else " " * 28)
        print(f"{name:<12} " + " ".join(f"{c:>28}" for c in cells))

    print("\nRead it this way: compare each clip DOWN its own column to see how fast")
    print("its pulse signal dies, and ACROSS a row to compare clips at equal encoding.")
    print("A gap that vanishes on the matched rows was never about manipulation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
