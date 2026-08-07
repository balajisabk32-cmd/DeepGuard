import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import cv2
import numpy as np
import time
import mediapipe as mp
import sounddevice as sd
import soundfile as sf
import threading
from models import SyncNet_color
import audio as wav2lip_audio

# Global state variables
app_state = "IDLE"  # Can be IDLE, RECORDING, PROCESSING
last_result = "Press SPACE to run AI scan"
recorded_frames = []
audio_data = None
start_time = 0
duration = 4
device = 'cpu'
model = None

def load_model():
    """Load the SyncNet model once at startup."""
    global model
    if model is None:
        print("Loading AI Detection Model... please wait.")
        model = SyncNet_color().to(device)
        checkpoint = torch.load('checkpoints/lipsync_expert.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print("Model loaded successfully!")

def process_clip(frames, audio, sr):
    """Background thread to process the recorded video and audio clip."""
    global app_state, last_result
    
    try:
        # Save temp audio for Wav2Lip processing
        AUDIO_PATH = 'temp_audio.wav'
        sf.write(AUDIO_PATH, audio, sr)
        
        # Load MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        
        MOUTH_IDX = [61, 291, 0, 17, 78, 308, 13, 14, 87, 317]
        mouth_crops = []
        valid_frame_indices = []
        
        # Extract mouth regions from each frame
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
            if crop.size == 0: continue
            crop = cv2.resize(crop, (96, 48))
            mouth_crops.append(crop)
            valid_frame_indices.append(i)
        
        face_mesh.close()
        
        if len(mouth_crops) < 5:
            last_result = "ERROR: Face not detected clearly. Try again."
            app_state = "IDLE"
            return
            
        # Extract audio features
        wav = wav2lip_audio.load_wav(AUDIO_PATH, 16000)
        mel = wav2lip_audio.melspectrogram(wav)
        mel_steps_per_frame = mel.shape[1] / len(frames)
        
        window = 5
        scores = []
        
        # Calculate lip-sync scores over 5-frame windows
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
        
        # Evaluate result based on the sync score
        if mean_score > 0.55:
            last_result = f"REAL HUMAN DETECTED (Sync Score: {mean_score:.2f})"
        else:
            last_result = f"FAKE / AI DETECTED (Sync Score: {mean_score:.2f})"
            
    except Exception as e:
        last_result = f"Error during processing: {str(e)}"
    
    # Return to IDLE state so the user can record again
    app_state = "IDLE"

def main():
    global app_state, last_result, recorded_frames, audio_data, start_time
    
    load_model()
    
    # Open the default camera
    cap = cv2.VideoCapture(0)
    sr = 16000
    
    # Create the interface window
    cv2.namedWindow("DeepGuard Live Interface", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DeepGuard Live Interface", 800, 600)
    
    print("Camera interface started. Look at the window!")
    print("Press SPACE to start detection, Q to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
            
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        if app_state == "IDLE":
            # Set text color based on the result
            color = (0, 255, 0) if "REAL" in last_result else (0, 0, 255)
            if "Press" in last_result or "ERROR" in last_result:
                color = (255, 255, 255)
            
            # Display the result
            cv2.putText(display_frame, last_result, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # Display instruction at the bottom
            cv2.putText(display_frame, "[ SPACE: Scan ]  [ Q: Quit ]", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
        elif app_state == "RECORDING":
            # Save the raw frame to our recording list
            recorded_frames.append(frame.copy())
            
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)
            
            # Show recording progress
            cv2.putText(display_frame, f"RECORDING... Speak now! ({remaining:.1f}s)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # Check if duration is met
            if elapsed >= duration:
                sd.wait() # Ensure audio recording is complete
                app_state = "PROCESSING"
                # Launch the processing in a background thread to keep the camera feed live
                threading.Thread(target=process_clip, args=(list(recorded_frames), audio_data.copy(), sr), daemon=True).start()
                
        elif app_state == "PROCESSING":
            cv2.putText(display_frame, "PROCESSING AI DETECTION...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display_frame, "Please wait a moment...", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Show the camera feed continuously
        cv2.imshow("DeepGuard Live Interface", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and app_state == "IDLE":
            # Start recording
            app_state = "RECORDING"
            recorded_frames = []
            start_time = time.time()
            # Start recording audio asynchronously
            audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
