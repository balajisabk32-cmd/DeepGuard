"""SyncNet lip-sync scoring — fixed for this environment.

    python Wav2Lip/run_sync_test.py --video ../REAL.mp4

WHAT WAS BROKEN
---------------
1. `mp.solutions.face_mesh` (was line 41). mediapipe 1.0.0 REMOVED `mp.solutions`
   entirely — verified: "AttributeError: module 'mediapipe' has no attribute
   'solutions'". Only `mediapipe.tasks` survives, and FaceLandmarker needs a
   .task weights file that is not vendored. Face detection now goes through
   src.rppg.backends, which already handles both and falls back to OpenCV Haar,
   so this runs today with no downloads.

2. Temporal window mismatch. `SAMPLE_EVERY_N_FRAMES = 3` meant the 5-crop SyncNet
   window spanned 15 source frames — about 500 ms at 30 fps. SyncNet is trained on
   5 CONSECUTIVE frames at 25 fps, i.e. ~200 ms. Feeding it a 500 ms window is
   off-distribution: scores degrade in a way that looks like bad lip-sync rather
   than a sampling bug. Crops are now taken consecutively; the stride applies only
   to where each window STARTS.

3. `torch.load` without `weights_only`. Torch 2.6+ flipped that default to True,
   which refuses a checkpoint dict carrying non-tensor entries. This checkpoint is
   ours, so it is loaded explicitly with weights_only=False.

4. ffmpeg subprocess for audio. There is no ffmpeg binary on PATH here. Audio now
   comes from PyAV in-process, which also yields the true A/V stream offset —
   measured at -167.7 ms on WIN_20260807_14_02_13_Pro.mp4, four times the plan's
   "suspicious" threshold. Uncorrected it lands straight in the sync estimate.

CHECKPOINT: needs Wav2Lip/checkpoints/lipsync_expert.pth (~50 MB). Not present ->
the script explains and exits 2 rather than throwing a FileNotFoundError.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)          # Wav2Lip's own `models`, `audio`
sys.path.insert(0, _REPO)          # DeepGuard's `src`

CKPT = os.path.join(_HERE, "checkpoints", "lipsync_expert.pth")
SYNCNET_FPS = 25.0                 # SyncNet's training frame rate
WINDOW = 5                         # consecutive frames per SyncNet window
MOUTH_ROI = (0.28, 0.60, 0.72, 0.95)   # fractions of the face box


def mouth_crops(frames, boxes):
    """96x48 mouth crops, one per frame. None where the face was not usable."""
    out = []
    for frame, box in zip(frames, boxes):
        if box is None or np.isnan(box).any():
            out.append(None)
            continue
        x, y, w, h = box
        H, W = frame.shape[:2]
        fx0, fy0, fx1, fy1 = MOUTH_ROI
        x0, x1 = int(np.clip(x + fx0 * w, 0, W - 1)), int(np.clip(x + fx1 * w, 1, W))
        y0, y1 = int(np.clip(y + fy0 * h, 0, H - 1)), int(np.clip(y + fy1 * h, 1, H))
        if x1 - x0 < 8 or y1 - y0 < 8:
            out.append(None)
            continue
        out.append(cv2.resize(frame[y0:y1, x0:x1], (96, 48)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="SyncNet lip-sync score")
    ap.add_argument("--video", default=os.path.join(_REPO, "REAL.mp4"))
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=5,
                    help="frames between window STARTS (windows stay consecutive)")
    args = ap.parse_args()

    if not os.path.exists(CKPT):
        print(f"[BLOCKED] SyncNet checkpoint missing: {CKPT}\n"
              "  Download `lipsync_expert.pth` (~50 MB) from the Wav2Lip project and\n"
              "  place it in Wav2Lip/checkpoints/.\n"
              "  NOTE: checkpoints are gitignored — GitHub rejects blobs over 100 MB.\n"
              "  Until then use the classical path:  python -m src.pipeline.detect <video>",
              file=sys.stderr)
        return 2

    import torch
    from models import SyncNet_color                       # noqa: E402
    import audio as wav2lip_audio                          # noqa: E402
    from src.lipsync.audio_io import load_audio            # noqa: E402
    from src.rppg import backends                          # noqa: E402
    from src.rppg.analyze import read_frames               # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    t0 = time.time()

    frames, t, nominal, _ = read_frames(args.video, max_sec=args.max_sec)
    if len(frames) < WINDOW * 2:
        print("[ERROR] too few frames", file=sys.stderr)
        return 1
    print(f"frames: {len(frames)}  ({t[-1]-t[0]:.1f}s @ {nominal:.2f} fps nominal)")

    track = load_audio(args.video, target_sr=16000, max_sec=args.max_sec)
    if not track.ok:
        print(f"[ERROR] audio unavailable: {track.reason}", file=sys.stderr)
        return 1
    print(f"audio: {track.duration:.1f}s   A/V stream offset {track.av_start_offset_sec*1000:+.1f} ms")

    boxes, counts = backends.OpenCVBackend().detect_boxes(frames)
    det = float(np.mean(~np.isnan(boxes[:, 0])))
    print(f"face detection: {det:.1%}")
    if det < 0.3:
        print("[ERROR] face not reliably detected", file=sys.stderr)
        return 1
    boxes = backends.smooth_boxes(boxes)

    crops = mouth_crops(frames, boxes)
    mel = wav2lip_audio.melspectrogram(track.samples.astype(np.float64))
    # Offset-corrected mapping from frame index to mel column.
    duration = max(t[-1] - t[0], 1e-6)
    mel_per_sec = mel.shape[1] / duration

    model = SyncNet_color().to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.eval()

    dists = []
    with torch.no_grad():
        for start in range(0, len(crops) - WINDOW, max(args.stride, 1)):
            window = crops[start:start + WINDOW]          # CONSECUTIVE frames
            if any(c is None for c in window):
                continue

            face = np.concatenate(window, axis=2).transpose(2, 0, 1)
            face_t = torch.FloatTensor(face).unsqueeze(0).to(device) / 255.0

            centre_sec = (t[start] - t[0]) - track.av_start_offset_sec
            m0 = int(centre_sec * mel_per_sec)
            chunk = mel[:, m0:m0 + 16]
            if chunk.shape[1] < 16:
                chunk = np.pad(chunk, ((0, 0), (0, 16 - chunk.shape[1])), mode="edge")
            mel_t = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)

            a, v = model(mel_t, face_t)
            dists.append(float(torch.nn.functional.cosine_similarity(a, v).item()))

    if len(dists) < 3:
        print("[ERROR] too few scored windows", file=sys.stderr)
        return 1

    d = np.array(dists)
    print("-" * 58)
    print(f"windows scored     {len(d)}")
    print(f"mean confidence    {d.mean():+.4f}   (higher = better lip/audio match)")
    print(f"median             {np.median(d):+.4f}")
    print(f"std (consistency)  {d.std():.4f}   <- drift matters more than level")
    print(f"elapsed            {time.time()-t0:.1f}s")
    print("-" * 58)
    print("Interpretation: a constant offset is normal (encoders introduce it).")
    print("Wandering match quality across windows is the lip-sync deepfake signature.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
