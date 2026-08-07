import os
import sys
import argparse

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import cv2
import numpy as np
import subprocess
import time
import mediapipe as mp
from models import SyncNet_color
import audio as wav2lip_audio

def main():
    parser = argparse.ArgumentParser(description="DeepGuard: Video Deepfake Detector (Lip-Sync)")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    args = parser.parse_args()

    VIDEO_PATH = args.video_path
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Could not find video file at '{VIDEO_PATH}'")
        sys.exit(1)

    AUDIO_PATH = 'temp_audio.wav'
    device = 'cpu'
    print(f"Analyzing Video: {VIDEO_PATH}")
    print(f"Using device: {device}")
    
    t0 = time.time()

    # Step 1: extract audio using MoviePy (doesn't require global ffmpeg install)
    print("\n[1/5] Extracting audio from video...")
    try:
        from moviepy import VideoFileClip
        video_clip = VideoFileClip(VIDEO_PATH)
        if video_clip.audio is None:
            print("Error: Could not extract audio. Make sure the video has an audio track!")
            sys.exit(1)
        video_clip.audio.write_audiofile(AUDIO_PATH, fps=16000, nbytes=2, codec='pcm_s16le')
    except Exception as e:
        print(f"Error extracting audio: {e}")
        sys.exit(1)

    # Step 2: read frames perfectly time-aligned
    print("[2/5] Reading video frames (auto-correcting VFR/framerate)...")
    try:
        # moviepy.iter_frames natively handles Variable Framerate (VFR) and 
        # accurately resamples the video to exactly 25.0 FPS using real timestamps!
        frames = list(video_clip.iter_frames(fps=25.0, dtype='uint8'))
        fps = 25.0
    except Exception as e:
        print(f"Error reading frames: {e}")
        sys.exit(1)
    finally:
        video_clip.close()

    print(f"      Extracted {len(frames)} perfectly time-aligned frames at 25.0 FPS")

    if len(frames) == 0:
        print("Error: Video contains no frames.")
        sys.exit(1)

    # Step 3: MediaPipe face mesh
    print("[3/5] Detecting and tracking faces...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    MOUTH_IDX = [61, 291, 0, 17, 78, 308, 13, 14, 87, 317]
    mouth_crops = []
    valid_frame_indices = []
    
    # We analyze every frame to get the most accurate sync score
    for i in range(len(frames)):
        frame = frames[i]
        h, w, _ = frame.shape
        frame_small = cv2.resize(frame, (320, int(320 * h / w)))
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        sh, sw, _ = frame_small.shape
        xs = [landmarks[idx].x * sw for idx in MOUTH_IDX]
        ys = [landmarks[idx].y * sh for idx in MOUTH_IDX]

        scale_x = w / sw
        scale_y = h / sh
        x1, x2 = int(min(xs) * scale_x), int(max(xs) * scale_x)
        y1, y2 = int(min(ys) * scale_y), int(max(ys) * scale_y)

        pad = 25
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (96, 48))
        mouth_crops.append(crop)
        valid_frame_indices.append(i)

    face_mesh.close()
    print(f"      Found faces in {len(mouth_crops)} frames")

    if len(mouth_crops) < 5:
        print("Error: Could not detect enough face frames to analyze lip sync.")
        sys.exit(1)

    # Step 4: audio features
    print("[4/5] Extracting audio features...")
    wav = wav2lip_audio.load_wav(AUDIO_PATH, 16000)
    mel = wav2lip_audio.melspectrogram(wav)
    
    # The crucial fix: calculate steps per frame dynamically based on actual fps
    mel_steps_per_frame = 80 / fps

    # Step 5: load SyncNet & Score
    print("[5/5] Analyzing audio-visual sync...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, 'checkpoints', 'lipsync_expert.pth')
    
    model = SyncNet_color().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    window = 5
    scores = []

    for start in range(0, len(mouth_crops) - window, window):
        crops = mouth_crops[start:start+window]
        stacked = np.concatenate(crops, axis=2)
        stacked = stacked.transpose(2, 0, 1)
        face_tensor = torch.FloatTensor(stacked).unsqueeze(0) / 255.0

        frame_idx = valid_frame_indices[start]
        mel_start = int(frame_idx * mel_steps_per_frame)
        mel_chunk = mel[:, mel_start:mel_start+16]
        
        if mel_chunk.shape[1] < 16:
            pad_width = 16 - mel_chunk.shape[1]
            mel_chunk = np.pad(mel_chunk, ((0,0),(0,pad_width)), mode='constant')

        mel_tensor = torch.FloatTensor(mel_chunk).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            audio_embed, face_embed = model(mel_tensor.to(device), face_tensor.to(device))
            sim = torch.nn.functional.cosine_similarity(audio_embed, face_embed)
        scores.append(sim.item())

    scores = np.array(scores)
    mean_score = scores.mean()
    
    total_time = time.time() - t0
    
    print("\n" + "=" * 60)
    print("                      RESULTS                      ")
    print("=" * 60)
    print(f"Analyzed {len(scores)} window segments in {total_time:.1f} seconds")
    print(f"Mean Lip-Sync Confidence Score: {mean_score:.4f}")
    print("-" * 60)
    
    # Threshold for video deepfakes 
    # Smartphone videos often suffer from heavy compression or micro-stutters, lowering the raw score.
    if mean_score > 0.20:
        print("[REAL] RESULT: REAL HUMAN")
        print("       The audio and lip movements match appropriately.")
    elif mean_score > 0.10:
        print("[WARNING] RESULT: INCONCLUSIVE / SUSPICIOUS")
        print("          The audio sync is questionable. Might be a low-quality deepfake,")
        print("          a dubbed video, or heavily compressed.")
    else:
        print("[FAKE] RESULT: FAKE OR MANIPULATED (AI/Deepfake)")
        print("       The audio and lip movements are completely out of sync.")
    print("=" * 60 + "\n")
    
    # Cleanup
    if os.path.exists(AUDIO_PATH):
        os.remove(AUDIO_PATH)

if __name__ == '__main__':
    main()
