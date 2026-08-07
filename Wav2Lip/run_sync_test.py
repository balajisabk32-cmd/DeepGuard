import os
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

device = 'cpu'
print(f"Using device: {device}")

VIDEO_PATH = 'test_video.mp4'
AUDIO_PATH = 'test_audio.wav'
SAMPLE_EVERY_N_FRAMES = 3   # skip frames to speed up detection

t0 = time.time()

# Step 1: extract audio
print("Extracting audio...")
subprocess.call(f'ffmpeg -y -i {VIDEO_PATH} -ar 16000 -ac 1 {AUDIO_PATH}', shell=True)

# Step 2: read frames
print("Reading video frames...")
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f"Total frames read: {len(frames)}")

# Step 3: MediaPipe face mesh (fast, CPU-friendly)
print("Detecting faces with MediaPipe...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mouth landmark indices in MediaPipe's 468-point face mesh
MOUTH_IDX = [61, 291, 0, 17, 78, 308, 13, 14, 87, 317]

mouth_crops = []
valid_frame_indices = []

for i in range(0, len(frames), SAMPLE_EVERY_N_FRAMES):
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

    # scale back up to original frame size
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
print(f"Faces detected in {len(mouth_crops)}/{len(frames)//SAMPLE_EVERY_N_FRAMES} sampled frames")
print(f"Detection took {time.time()-t0:.2f}s so far")

# Step 4: audio features (Wav2Lip's own normalization)
print("Extracting audio features...")
wav = wav2lip_audio.load_wav(AUDIO_PATH, 16000)
mel = wav2lip_audio.melspectrogram(wav)
mel_steps_per_frame = mel.shape[1] / len(frames)

# Step 5: load SyncNet
print("Loading SyncNet...")
model = SyncNet_color().to(device)
checkpoint = torch.load('checkpoints/lipsync_expert.pth', map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Step 6: score in windows of 5 consecutive DETECTED frames
print("Computing sync scores...")
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
total_time = time.time() - t0

print(f"\n--- RESULTS ---")
print(f"Total windows scored: {len(scores)}")
print(f"Mean sync score: {scores.mean():.4f}")
print(f"Std deviation: {scores.std():.4f}")
print(f"Min: {scores.min():.4f}  Max: {scores.max():.4f}")
print(f"Total pipeline time: {total_time:.2f} seconds")
print("\nPer-window scores:")
print(scores)