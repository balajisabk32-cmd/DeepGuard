import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import cv2
import numpy as np
import subprocess
import time
import mediapipe as mp
import sounddevice as sd
import soundfile as sf
from models import SyncNet_color
import audio as wav2lip_audio

def record_av(duration=5, fps=25, sr=16000):
    print(f"Opening camera for {duration} seconds... Please speak into the camera!")
    cap = cv2.VideoCapture(0)
    
    # Try to set fps
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    frames = []
    
    # Start audio recording
    print("Recording started...")
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    
    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        # We need the clean frame for analysis, without text
        clean_frame = frame.copy()
        frames.append(clean_frame)

        # Put text on frame to indicate recording
        time_left = duration - (time.time() - start_time)
        cv2.putText(frame, f"Recording: {time_left:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("DeepGuard - Live Deepfake Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    sd.wait() # wait for audio to finish
    cap.release()
    cv2.destroyAllWindows()
    print("Recording finished!")
    return frames, audio_data, sr

def main():
    device = 'cpu'
    print(f"Using device: {device}")
    
    # 1. Record 
    duration = 4
    fps = 25
    sr = 16000
    frames, audio_data, sr = record_av(duration=duration, fps=fps, sr=sr)
    
    if len(frames) == 0:
        print("No video frames captured! Please check your webcam.")
        return
        
    # Save temp audio
    AUDIO_PATH = 'temp_audio.wav'
    sf.write(AUDIO_PATH, audio_data, sr)
    
    # Step 3: MediaPipe face mesh
    print("Detecting faces with MediaPipe...")
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
    SAMPLE_EVERY_N_FRAMES = 1
    
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
    print(f"Faces detected in {len(mouth_crops)} frames")
    
    if len(mouth_crops) < 5:
        print("Not enough face frames detected. Please make sure your face is visible.")
        return
        
    # Step 4: audio features
    wav = wav2lip_audio.load_wav(AUDIO_PATH, 16000)
    mel = wav2lip_audio.melspectrogram(wav)
    mel_steps_per_frame = mel.shape[1] / len(frames)
    
    # Step 5: load SyncNet
    print("Loading SyncNet...")
    model = SyncNet_color().to(device)
    checkpoint = torch.load('checkpoints/lipsync_expert.pth', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Step 6: score
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
    mean_score = scores.mean()
    
    print(f"\n--- RESULTS ---")
    print(f"Mean lip-sync confidence score: {mean_score:.4f}")
    
    # Typical syncnet cosine similarity is higher than 0.6 for real synced audio
    is_real = mean_score > 0.55
    
    print("\n" + "=" * 50)
    if is_real:
        print("[REAL] RESULT: REAL HUMAN")
        print("    (Lip movements match the audio perfectly!)")
    else:
        print("[FAKE] RESULT: FAKE OR MANIPULATED (AI/Deepfake)")
        print("    (Lip movements DO NOT match the audio or out-of-sync!)")
    print("=" * 50 + "\n")
    
if __name__ == '__main__':
    main()
