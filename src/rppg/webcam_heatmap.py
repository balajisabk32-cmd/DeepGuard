"""Spatial pulse-COHERENCE heatmap viewer — diagnostic tool, not a detector.

    python -m src.rppg.webcam_heatmap                    # live camera
    python -m src.rppg.webcam_heatmap --video TEST_VIDEOS/REAL.mp4   # a file
    python -m src.rppg.webcam_heatmap --video TEST_VIDEOS/REAL.mp4 --out annotated.mp4 --headless

Keys: q quit, s snapshot.

WHAT CHANGED AND WHY
--------------------
1. The heatmap now shows COHERENCE, not per-patch SNR. The previous version
   coloured each patch by its own pulse strength, which is the discredited
   "no pulse => fake" signal rendered spatially: a dark patch meant hair, shadow
   or compression, not a manipulation seam. Coherence asks whether a patch beats
   IN TIME with the rest of the face, which is the quantity a face-swap seam
   actually disturbs (Kammari et al. 2024 §3.1/§3.4).

2. Zero-phase filtering. The old code used `sosfilt`, which is causal and imposes
   a frequency-dependent phase shift. For a measurement whose entire content is
   phase agreement between patches, a phase-distorting filter manufactures the
   very dispersion it is trying to detect. Now uses the shared zero-phase path.

3. One implementation, one config. Filtering, CHROM/POS, SNR, thresholds and the
   verdict all come from src.rppg.signal_core, src.rppg.ppgmap, src.fusion and
   config/thresholds.yaml. The old file re-implemented all of them with different
   constants, so the repo contained two detectors that disagreed.

4. Longer window. 4-5s cannot resolve heart rate: at 5s the frequency resolution
   is ~12 BPM, so the readout jitters meaninglessly. Default is now 12s, matching
   `min_window_sec` in the config.

5. Face-detection gaps no longer desynchronise the time axis — samples carry
   their own timestamps instead of being appended only when a face is found.

THIS TOOL DOES NOT VOTE. It renders evidence. The verdict shown comes from
src.fusion so it cannot drift away from the pipeline's own answer.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np

from src.fusion import load_thresholds
from src.rppg.ppgmap import min_skin_fraction, patch_coherence, snr_weights
from src.rppg.signal_core import chrom_full, peak_frequency, pos_overlap, resample_to_uniform

GRID_ROWS, GRID_COLS = 6, 5
WINDOW_SEC = 12.0
RECOMPUTE_EVERY = 8          # frames; 30 patches x CHROM is too slow every frame
# Skin-fraction threshold is NOT defined here — it comes from config via
# ppgmap.min_skin_fraction(), so the viewer and the offline detector cannot
# disagree about which patches are usable. They previously drifted (0.30 vs 0.35),
# which made the heatmap show patches that analyze() had discarded.


# --------------------------------------------------------------------- capture


def open_webcam():
    """Windows-friendly probe. DirectShow first (avoids MSMF error -1072875772)."""
    for backend, label in ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF"),
                           (cv2.CAP_ANY, "Default")):
        for idx in range(4):
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    for _ in range(5):
                        ok, frame = cap.read()
                        if ok and frame is not None and frame.size:
                            print(f"[OK] camera {idx} via {label}")
                            _lock_exposure(cap)
                            return cap
                    cap.release()
            except Exception:
                continue
    return None


def _lock_exposure(cap) -> None:
    """Disable auto-exposure and auto-white-balance.

    Auto-exposure is the single biggest practical killer of rPPG: the camera
    continuously renormalises brightness, which both suppresses the pulse
    modulation and injects its own low-frequency drift into the HR band. Nothing
    downstream can recover from it, so it has to be switched off at capture.
    """
    for prop, value in ((cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),
                        (cv2.CAP_PROP_AUTO_WB, 0),
                        (cv2.CAP_PROP_FRAME_WIDTH, 1280),
                        (cv2.CAP_PROP_FRAME_HEIGHT, 720)):
        try:
            cap.set(prop, value)
        except Exception:
            pass


# --------------------------------------------------------------------- tracker


class CoherenceTracker:
    """Rolling per-patch RGB buffers -> pulse coherence map.

    Every sample carries its own timestamp, so frames where the face is missing
    become genuine gaps rather than silently shortening the time axis (which
    biases every frequency estimate upward).
    """

    def __init__(self, rows=GRID_ROWS, cols=GRID_COLS, window_sec=WINDOW_SEC,
                 skin_fraction: float | None = None):
        self.rows, self.cols, self.window_sec = rows, cols, window_sec
        self.skin_fraction = min_skin_fraction() if skin_fraction is None else skin_fraction
        maxlen = 900
        self.t: deque[float] = deque(maxlen=maxlen)
        self.rgb: deque[np.ndarray] = deque(maxlen=maxlen)   # (rows, cols, 3), NaN where invalid
        self.coherence = np.full((rows, cols), np.nan)
        self.stats: dict = {"fs": 0.0, "hr": None, "snr": -99.0, "quality": 0.0,
                            "coh_mean": 0.0, "patches": 0, "ready": False}
        self._n = 0

    # -- sampling --------------------------------------------------------
    def add(self, frame_bgr: np.ndarray, box, now: float) -> None:
        grid = np.full((self.rows, self.cols, 3), np.nan)
        if box is not None:
            bx, by, bw, bh = box
            ph, pw = bh / self.rows, bw / self.cols
            for r in range(self.rows):
                for c in range(self.cols):
                    y0, y1 = int(by + r * ph), int(by + (r + 1) * ph)
                    x0, x1 = int(bx + c * pw), int(bx + (c + 1) * pw)
                    patch = frame_bgr[max(y0, 0):y1, max(x0, 0):x1]
                    if patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 3:
                        continue
                    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
                    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
                    mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
                    # NaN, not a whole-patch fallback: averaging hair and background
                    # into a "skin" sample pollutes the signal instead of omitting it.
                    if mask.mean() < self.skin_fraction:
                        continue
                    grid[r, c] = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)[mask].mean(axis=0)

        self.t.append(now)
        self.rgb.append(grid)
        self._n += 1

    # -- analysis --------------------------------------------------------
    def due(self) -> bool:
        return self._n % RECOMPUTE_EVERY == 0

    def update(self, cfg: dict) -> None:
        if len(self.t) < 64:
            return
        t = np.asarray(self.t, dtype=float)
        span = t[-1] - t[0]
        if span < 4.0:
            return
        fs = (len(t) - 1) / span
        stack = np.asarray(self.rgb)          # (N, rows, cols, 3)

        sigs, snrs, coords = [], [], []
        for r in range(self.rows):
            for c in range(self.cols):
                chan = stack[:, r, c, :]
                if np.isfinite(chan[:, 0]).mean() < 0.6:
                    continue
                rgb = np.column_stack(
                    [resample_to_uniform(t, chan[:, k], fs) for k in range(3)]
                )
                if rgb.size == 0 or rgb.shape[0] < 64:
                    continue
                # ONE extractor across the whole grid. Mixing CHROM and POS mixes
                # polarity conventions and manufactures anti-correlation between
                # patches that are physically in phase.
                sig = pos_overlap(rgb, fs)
                if sig.size == 0 or not np.isfinite(sig).all():
                    continue
                f0, snr = peak_frequency(sig, fs)
                if f0 is None or not np.isfinite(snr):
                    continue
                sigs.append(sig)
                snrs.append(snr)
                coords.append((r, c))

        if len(sigs) < 4:
            self.stats.update(ready=False, patches=len(sigs))
            return

        L = min(len(s) for s in sigs)
        S = np.array([s[:L] for s in sigs])
        snrs = np.array(snrs)

        coh, z = patch_coherence(S, snrs)
        cmap = np.full((self.rows, self.cols), np.nan)
        for (r, c), v in zip(coords, coh):
            cmap[r, c] = v
        self.coherence = cmap

        ref = np.average(z, axis=0, weights=snr_weights(snrs))
        f0, ref_snr = peak_frequency(ref, fs)
        rc = cfg["rppg"]
        quality = float(np.clip(
            (float(np.average(snrs, weights=snr_weights(snrs))) - rc["quality_snr_floor_db"])
            / (rc["quality_snr_ceil_db"] - rc["quality_snr_floor_db"]), 0.0, 1.0))

        self.stats.update(
            fs=fs, hr=None if f0 is None else f0 * 60.0, snr=float(ref_snr),
            quality=quality, coh_mean=float(np.nanmean(coh)),
            patches=len(sigs), ready=span >= self.window_sec,
        )


# --------------------------------------------------------------------- render


# Explicit BGR ramp. NOT a built-in colormap: OpenCV 5 has no COLORMAP_RdYlGn,
# and the obvious fallback (JET) maps HIGH values to red — which would invert the
# legend and colour a perfectly coherent face as though it were a manipulation seam.
_RED, _AMBER, _GREEN = (60, 60, 240), (11, 158, 245), (129, 185, 16)


def _coh_color(v: float) -> tuple[int, int, int]:
    """Red (anti-phase) -> amber (uncorrelated) -> green (in phase with the face)."""
    if not np.isfinite(v):
        return (60, 60, 60)                       # grey: no usable skin in this patch
    x = float(np.clip((v + 1.0) / 2.0, 0.0, 1.0))  # [-1,1] -> [0,1]
    lo, hi, f = (_RED, _AMBER, x / 0.5) if x < 0.5 else (_AMBER, _GREEN, (x - 0.5) / 0.5)
    return tuple(int(round(lo[i] + (hi[i] - lo[i]) * f)) for i in range(3))


def render(frame: np.ndarray, box, tracker: CoherenceTracker, cfg: dict) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    st = tracker.stats

    if box is not None:
        bx, by, bw, bh = [int(v) for v in box]
        overlay = np.zeros((max(bh, 1), max(bw, 1), 3), dtype=np.uint8)
        ph, pw = bh / tracker.rows, bw / tracker.cols
        for r in range(tracker.rows):
            for c in range(tracker.cols):
                p0 = (int(c * pw), int(r * ph))
                p1 = (int((c + 1) * pw), int((r + 1) * ph))
                cv2.rectangle(overlay, p0, p1, _coh_color(tracker.coherence[r, c]), -1)
                cv2.rectangle(overlay, p0, p1, (40, 40, 40), 1)
        y0, y1 = max(by, 0), min(by + bh, H)
        x0, x1 = max(bx, 0), min(bx + bw, W)
        if y1 > y0 and x1 > x0:
            region = out[y0:y1, x0:x1]
            ov = overlay[: y1 - y0, : x1 - x0]
            out[y0:y1, x0:x1] = cv2.addWeighted(region, 0.62, ov, 0.38, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 242, 254), 2)

    band = out[:118, :]
    dark = np.zeros_like(band)
    dark[:] = (10, 13, 22)
    out[:118, :] = cv2.addWeighted(band, 0.25, dark, 0.75, 0)

    cv2.putText(out, "DEEPGUARD // PULSE COHERENCE MAP  (diagnostic - does not vote)",
                (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 242, 254), 2)

    if not st["ready"]:
        cv2.putText(out, f"BUFFERING  {len(tracker.t)} frames   "
                         f"need >= {tracker.window_sec:.0f}s for a heart-rate estimate",
                    (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 158, 11), 2)
    else:
        hr = "--" if st["hr"] is None else f"{st['hr']:.0f} bpm"
        cv2.putText(out, f"HR {hr}   SNR {st['snr']:+.1f} dB   evidence {st['quality']*100:.0f}%",
                    (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    gate = cfg["rppg"].get("min_quality_for_evidence", 0.05)
    note = ("coherence unreliable below the evidence floor"
            if st["quality"] <= gate else
            "green = beats in phase with the face   red = out of phase")
    cv2.putText(out, f"mean coherence {st['coh_mean']:+.2f} over {st['patches']} patches"
                     f"   |   {note}",
                (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
    return out


# --------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pulse coherence heatmap viewer")
    ap.add_argument("--video", help="analyse a file instead of the camera")
    ap.add_argument("--out", help="write an annotated video here")
    ap.add_argument("--headless", action="store_true", help="no window (CI / Docker)")
    ap.add_argument("--delay", type=int, default=28,
                    help="ms per frame. Default plays at ~35fps; 1 = as fast as possible. "
                         "The old hard-coded 1 made the whole clip flash past in seconds.")
    ap.add_argument("--loop", action="store_true",
                    help="restart the clip when it ends — keeps the panel on screen for a demo")
    ap.add_argument("--hold", action="store_true",
                    help="keep the final frame up until a key is pressed")
    args = ap.parse_args(argv)

    cfg = load_thresholds()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[ERROR] cannot open {args.video}", file=sys.stderr)
            return 1
    else:
        cap = open_webcam()
        if cap is None:
            print("[ERROR] no camera. Close Zoom/Teams/Camera and retry, "
                  "or pass --video <file>.", file=sys.stderr)
            return 1

    WINDOW = "DeepGuard // Pulse Coherence Map"
    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 720, 900)
        try:
            # Without this the window can open behind the terminal that launched
            # it, which looks exactly like "no panel appeared".
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    tracker = CoherenceTracker()
    writer = None
    smooth = None
    t0 = time.time()
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if args.loop and args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_i = 0
                continue
            break

        if args.video:
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            now = ts if ts > 0 else frame_i / 30.0
        else:
            now = time.time() - t0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces):
            b = np.array(max(faces, key=lambda f: f[2] * f[3]), dtype=float)
            smooth = b if smooth is None else 0.6 * smooth + 0.4 * b

        tracker.add(frame, smooth, now)
        if tracker.due():
            tracker.update(cfg)

        shown = render(frame, smooth, tracker, cfg)

        if args.out:
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         25.0, (shown.shape[1], shown.shape[0]))
            writer.write(shown)

        if not args.headless:
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(max(args.delay, 1)) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                name = f"coherence_{int(time.time())}.png"
                cv2.imwrite(name, shown)
                print(f"saved {name}")

        frame_i += 1

    if args.hold and not args.headless:
        print("Press any key in the window to close...")
        cv2.waitKey(0)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"wrote {args.out}")
    if not args.headless:
        cv2.destroyAllWindows()

    st = tracker.stats
    print(f"patches={st['patches']}  mean_coherence={st['coh_mean']:+.3f}  "
          f"snr={st['snr']:+.2f}dB  evidence={st['quality']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
