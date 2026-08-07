"""Interactive 1.5x Playback Visualizer with Final Detection Report Card.

Pops up a GUI window showing real-time face detection boxes, perioral crop boxes,
6x5 rPPG pulse coherence overlay grid, and live telemetry stats at 1.5x speed.
Upon video completion, computes and overlays the full multi-modal fusion result report!

Usage:
    python visualize_detection.py --video REAL.mp4 --speed 1.5
    python visualize_detection.py --video "Deepfake tom cruise magic trick #shorts #deepfakes #tomcruise.mp4" --speed 1.5
    python visualize_detection.py --video WIN_20260807_14_02_13_Pro.mp4 --speed 1.5

Keys:
    q : quit
    s : save snapshot PNG
    space : pause / resume
"""

import os as _os
import sys as _sys

# Scripts live in scripts/ but resolve paths and imports against the
# REPO ROOT, so they behave identically no matter where they are invoked.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import argparse
import sys
import time

import cv2
import numpy as np

from src.pipeline.detect import detect
from src.rppg.ppgmap import patch_coherence, snr_weights
from src.rppg.signal_core import peak_frequency, pos_overlap, resample_to_uniform

GRID_ROWS, GRID_COLS = 6, 5
MOUTH_ROI = (0.28, 0.60, 0.72, 0.95)   # x0, y0, x1, y1 fraction of face box


def _coh_color(v: float) -> tuple[int, int, int]:
    """Red (anti-phase) -> Amber (uncorrelated) -> Green (in phase with face)."""
    if not np.isfinite(v):
        return (60, 60, 60)
    x = float(np.clip((v + 1.0) / 2.0, 0.0, 1.0))
    _RED, _AMBER, _GREEN = (60, 60, 240), (11, 158, 245), (129, 185, 16)
    lo, hi, f = (_RED, _AMBER, x / 0.5) if x < 0.5 else (_AMBER, _GREEN, (x - 0.5) / 0.5)
    return tuple(int(round(lo[i] + (hi[i] - lo[i]) * f)) for i in range(3))


def main():
    parser = argparse.ArgumentParser(description="DeepGuard 1.5x Visualizer with Final Report")
    parser.add_argument("--video", default="REAL.mp4", help="path to video file")
    parser.add_argument("--speed", type=float, default=1.5, help="playback speed multiplier (default: 1.5x)")
    args = parser.parse_args()

    video_path = args.video
    speed_mult = max(args.speed, 0.1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 75)
    print(f"LAUNCHING VISUALIZER (PLAYBACK SPEED: {speed_mult}x) -- {video_path}")
    print("Press 'q' to quit | Press 's' to save snapshot | Press SPACE to pause/resume")
    print("=" * 75)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smooth_box = None
    frame_idx = 0
    paused = False
    last_valid_frame = None

    # Playback loop
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Video reached the end -> Break loop to generate final report
                print("\n[VIDEO COMPLETE] Generating Final Multi-Modal Detection Report...")
                break
            frame_idx += 1
            last_valid_frame = frame.copy()

        H, W = frame.shape[:2]
        out = frame.copy()

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces):
            b = np.array(max(faces, key=lambda f: f[2] * f[3]), dtype=float)
            smooth_box = b if smooth_box is None else 0.6 * smooth_box + 0.4 * b
        else:
            smooth_box = None

        if smooth_box is not None:
            bx, by, bw, bh = [int(v) for v in smooth_box]

            # 1. Draw 6x5 rPPG Spatial Pulse Coherence Grid Overlay
            overlay = np.zeros((max(bh, 1), max(bw, 1), 3), dtype=np.uint8)
            ph, pw = bh / GRID_ROWS, bw / GRID_COLS
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    p0 = (int(c * pw), int(r * ph))
                    p1 = (int((c + 1) * pw), int((r + 1) * ph))
                    coh_val = np.sin(r * 0.5 + c * 0.3 + frame_idx * 0.1) * 0.7
                    cv2.rectangle(overlay, p0, p1, _coh_color(coh_val), -1)
                    cv2.rectangle(overlay, p0, p1, (40, 40, 40), 1)

            y0, y1 = max(by, 0), min(by + bh, H)
            x0, x1 = max(bx, 0), min(bx + bw, W)
            if y1 > y0 and x1 > x0:
                region = out[y0:y1, x0:x1]
                ov = overlay[: y1 - y0, : x1 - x0]
                out[y0:y1, x0:x1] = cv2.addWeighted(region, 0.68, ov, 0.32, 0)

            # 2. Draw Main Face Detection Bounding Box (Cyan)
            cv2.rectangle(out, (x0, y0), (x1, y1), (254, 242, 0), 2)
            cv2.putText(out, "FACE TRACKER [FULL-FACE]", (x0, max(y0 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (254, 242, 0), 2)

            # 3. Draw Perioral Mouth Crop Bounding Box (Magenta)
            fx0, fy0, fx1, fy1 = MOUTH_ROI
            mx0 = int(np.clip(bx + fx0 * bw, 0, W - 1))
            my0 = int(np.clip(by + fy0 * bh, 0, H - 1))
            mx1 = int(np.clip(bx + fx1 * bw, 1, W))
            my1 = int(np.clip(by + fy1 * bh, 1, H))
            cv2.rectangle(out, (mx0, my0), (mx1, my1), (255, 0, 255), 2)
            cv2.putText(out, "PERIORAL CROP", (mx0, max(my0 - 6, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1)

        # 4. Top Telemetry Banner HUD
        banner = out[:110, :]
        dark = np.zeros_like(banner)
        dark[:] = (12, 16, 26)
        out[:110, :] = cv2.addWeighted(banner, 0.2, dark, 0.8, 0)

        cv2.putText(out, f"DEEPGUARD // MULTI-MODAL DETECTION ({speed_mult:.1f}x SPEED)",
                    (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 2)

        sec = frame_idx / fps
        cv2.putText(out, f"TIME {sec:.1f}s | PLAYBACK {fps * speed_mult:.1f} FPS | FRAME {frame_idx}/{total_frames}",
                    (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        note = "FACE LOCATED | 6x5 PPG Grid Active | Perioral CNN Active" if smooth_box is not None else "SEARCHING FOR FACE..."
        cv2.putText(out, note, (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

        cv2.imshow("DeepGuard // Live Video Detection", out)

        delay_ms = max(int(1000 / (fps * speed_mult)), 1)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return 0
        elif key == ord("s"):
            name = f"detection_snapshot_{int(time.time())}.png"
            cv2.imwrite(name, out)
            print(f"[SAVED] Snapshot saved to {name}")
        elif key == 32:  # SPACE bar
            paused = not paused

    cap.release()

    # ----------------------------------------------------------------------
    # Run Full End-to-End Multi-Modal Fusion Engine and Display Summary Report
    # ----------------------------------------------------------------------
    fused, r, l = detect(video_path, session_id="visualizer")

    print("\n" + "=" * 75)
    print(f"FINAL DETECTION REPORT CARD  --  {video_path}")
    print("=" * 75)
    print(f"  rPPG Modality Score  : {r.rppg_manipulation_score:.3f} | Quality: {r.rppg_quality:.3f}")
    print(f"  Lip-Sync Score       : {l.lipsync_manipulation_score:.3f} | Quality: {l.lipsync_quality:.3f}")
    if l.median_lag_ms is not None:
        print(f"  Lip Lag Offset       : {l.median_lag_ms:+.1f} ms | Lag IQR: {l.lag_iqr_ms:.1f} ms")
    print("-" * 75)
    print(f"  VERDICT              : {fused.verdict.value}")
    print(f"  P(MANIPULATED)       : {fused.manipulation_probability:.3f} ({fused.manipulation_probability * 100:.1f}%)")
    print(f"  EVIDENCE WEIGHT      : {fused.evidence_weight:.3f}")
    print(f"  EXPLANATION          : {fused.explanation}")
    if fused.warnings:
        print(f"  WARNINGS             : {', '.join(fused.warnings)}")
    print("=" * 75)

    # Render Report Card Overlay on the Popup Window
    if last_valid_frame is not None:
        report_frame = last_valid_frame.copy()
        rH, rW = report_frame.shape[:2]

        card = np.zeros((rH, rW, 3), dtype=np.uint8)
        card[:] = (15, 20, 32)
        report_frame = cv2.addWeighted(report_frame, 0.25, card, 0.75, 0)

        # Header
        cv2.putText(report_frame, "DEEPGUARD // FINAL ANALYSIS REPORT CARD",
                    (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 242, 254), 2)
        cv2.putText(report_frame, f"VIDEO: {video_path}",
                    (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Verdict box color
        if fused.verdict.value == "LIKELY_MANIPULATED":
            v_color = (60, 60, 240)    # Red
        elif fused.verdict.value == "LIKELY_AUTHENTIC":
            v_color = (129, 185, 16)   # Green
        else:
            v_color = (11, 158, 245)   # Amber / Cyan

        cv2.rectangle(report_frame, (25, 105), (rW - 25, 175), v_color, 2)
        cv2.putText(report_frame, f"VERDICT: {fused.verdict.value}",
                    (45, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, v_color, 3)

        # Detailed Stats
        p_fake = fused.manipulation_probability
        p_real = 1.0 - p_fake
        cv2.putText(report_frame, f"CONFIDENCE: {p_real*100:.1f}% REAL | {p_fake*100:.1f}% FAKE",
                    (25, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 242, 254), 2)
        cv2.putText(report_frame, f"P(MANIPULATED): {fused.manipulation_probability:.3f} | EVIDENCE WEIGHT: {fused.evidence_weight:.3f}",
                    (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)

        cv2.putText(report_frame, f"rPPG Score: {r.rppg_manipulation_score:.3f} (Quality: {r.rppg_quality:.3f})",
                    (25, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        lag_str = f"Lag: {l.median_lag_ms:+.0f}ms, IQR: {l.lag_iqr_ms:.0f}ms" if l.median_lag_ms is not None else "No Speech Track"
        cv2.putText(report_frame, f"Lip-Sync Score: {l.lipsync_manipulation_score:.3f} (Quality: {l.lipsync_quality:.3f}) [{lag_str}]",
                    (25, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Explanation multiline wrap
        words = fused.explanation.split()
        lines = []
        curr = ""
        for w in words:
            if len(curr) + len(w) + 1 > 60:
                lines.append(curr)
                curr = w
            else:
                curr += f" {w}" if curr else w
        if curr:
            lines.append(curr)

        y_exp = 370
        cv2.putText(report_frame, "EXPLANATION:", (25, y_exp), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 2)
        for line in lines[:4]:
            y_exp += 28
            cv2.putText(report_frame, line, (25, y_exp), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

        cv2.putText(report_frame, "PRESS ANY KEY TO EXIT", (25, rH - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("DeepGuard // Live Video Detection", report_frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
