import cv2
import numpy as np
import scipy.signal as signal
from collections import deque
import time
import sys

def process_chrom(R_buffer, G_buffer, B_buffer, fps):
    R = np.array(R_buffer)
    G = np.array(G_buffer)
    B = np.array(B_buffer)
    
    # Normalize
    R_n = R / np.mean(R)
    G_n = G / np.mean(G)
    B_n = B / np.mean(B)
    
    # CHROM formula
    X = 3 * R_n - 2 * G_n
    Y = 1.5 * R_n + G_n - 1.5 * B_n
    
    # Filter
    nyq = 0.5 * fps
    
    # Need enough samples for the filter order
    if len(X) < 15:
        return None
        
    b, a = signal.butter(3, [0.7 / nyq, 4.0 / nyq], btype='bandpass')
    try:
        X_f = signal.filtfilt(b, a, X)
        Y_f = signal.filtfilt(b, a, Y)
    except ValueError:
        return None
        
    alpha = np.std(X_f) / np.std(Y_f)
    pulse_signal = X_f - alpha * Y_f
    
    # Welch's method to find strongest frequency
    nperseg = min(len(pulse_signal), int(fps * 10))
    f, Pxx = signal.welch(pulse_signal, fs=fps, nperseg=nperseg)
    
    valid_idx = np.where((f >= 0.7) & (f <= 4.0))[0]
    if len(valid_idx) == 0:
        return None, 0.0
    best_idx = valid_idx[np.argmax(Pxx[valid_idx])]
    
    detected_freq = f[best_idx]
    
    # Calculate SNR (Signal to Noise Ratio)
    peak_power = 0.0
    noise_power = 0.0
    for i in valid_idx:
        # Include the peak and its immediate neighbors to account for slight leakage
        if abs(i - best_idx) <= 1:
            peak_power += Pxx[i]
        else:
            noise_power += Pxx[i]
            
    if noise_power == 0:
        snr = 10.0
    else:
        snr = 10 * np.log10(peak_power / noise_power)
        
    return detected_freq * 60, snr

def main():
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        cap = cv2.VideoCapture(video_path)
        print(f"Analyzing video: {video_path}")
    else:
        # Connect to the first webcam (CAP_DSHOW fixes MSMF errors on Windows)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("Starting webcam for rPPG CHROM. Press 'q' to quit.")
    
    # Try to get the actual FPS of the camera, default to 30 if unavailable
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
        
    window_sec = 10
    max_frames = int(fps * window_sec)
    
    R_buffer = deque(maxlen=max_frames)
    G_buffer = deque(maxlen=max_frames)
    B_buffer = deque(maxlen=max_frames)
    
    last_bpm = 0.0
    last_snr = 0.0
    status_text = "ANALYZING..."
    status_color = (0, 255, 255) # Yellow
    
    # Variables for smoothing the bounding box
    smooth_x, smooth_y, smooth_w, smooth_h = 0, 0, 0, 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # Convert BGR to RGB for MediaPipe and rPPG processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(image_rgb)
        
        if results.detections:
            # Take the most prominent face
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Smooth the bounding box to reduce jitter (which creates massive noise)
            if smooth_w == 0:
                smooth_x, smooth_y, smooth_w, smooth_h = x, y, w, h
            else:
                alpha_box = 0.1 # Smoothing factor (lower = smoother)
                smooth_x = int(alpha_box * x + (1 - alpha_box) * smooth_x)
                smooth_y = int(alpha_box * y + (1 - alpha_box) * smooth_y)
                smooth_w = int(alpha_box * w + (1 - alpha_box) * smooth_w)
                smooth_h = int(alpha_box * h + (1 - alpha_box) * smooth_h)
            
            # Define ROIs (Region of Interest)
            
            # 1. Forehead
            fh_x = max(0, smooth_x + int(smooth_w * 0.3))
            fh_y = max(0, smooth_y + int(smooth_h * 0.1))
            fh_w = int(smooth_w * 0.4)
            fh_h = int(smooth_h * 0.2)
            
            # 2. Left Cheek
            lc_x = max(0, smooth_x + int(smooth_w * 0.15))
            lc_y = max(0, smooth_y + int(smooth_h * 0.55))
            lc_w = int(smooth_w * 0.25)
            lc_h = int(smooth_h * 0.25)
            
            # 3. Right Cheek
            rc_x = max(0, smooth_x + int(smooth_w * 0.6))
            rc_y = max(0, smooth_y + int(smooth_h * 0.55))
            rc_w = int(smooth_w * 0.25)
            rc_h = int(smooth_h * 0.25)
            
            # 4. Eyes (Per user request, though usually noisy due to blinking)
            eye_x = max(0, smooth_x + int(smooth_w * 0.2))
            eye_y = max(0, smooth_y + int(smooth_h * 0.35))
            eye_w = int(smooth_w * 0.6)
            eye_h = int(smooth_h * 0.15)
            
            rois = [
                (fh_x, fh_y, fh_w, fh_h),
                (lc_x, lc_y, lc_w, lc_h),
                (rc_x, rc_y, rc_w, rc_h),
                (eye_x, eye_y, eye_w, eye_h)
            ]
            
            avg_colors = []
            
            for (rx, ry, rw, rh) in rois:
                # Draw the ROI on the image for visualization
                cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                roi = image_rgb[ry:ry+rh, rx:rx+rw]
                if roi.size > 0:
                    avg_colors.append(np.average(np.average(roi, axis=0), axis=0))
                    
            if len(avg_colors) > 0:
                # Average all the ROIs together
                final_avg = np.mean(avg_colors, axis=0)
                
                R_buffer.append(final_avg[0])
                G_buffer.append(final_avg[1])
                B_buffer.append(final_avg[2])
                
                # Compute BPM if we have at least 3 seconds of data
                if len(R_buffer) > fps * 3:
                    result = process_chrom(R_buffer, G_buffer, B_buffer, fps)
                    if result[0] is not None:
                        bpm, snr = result
                        last_bpm = bpm
                        last_snr = snr
                        
                        # Threshold for SNR to determine if it's a real pulse or random noise
                        if snr > 1.0:
                            status_text = "REAL"
                            status_color = (0, 255, 0) # Green
                        else:
                            status_text = "DEEPFAKE (No Pulse)"
                            status_color = (0, 0, 255) # Red
                            
            # Display the BPM, SNR and Status
            cv2.putText(image, f"BPM: {last_bpm:.1f} (SNR: {last_snr:.1f}dB)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
            
        # Display the resulting frame
        cv2.imshow('rPPG CHROM Test', image)
        
        # Press 'q' to exit.
        # Calculate proper delay so the video plays at normal speed (not hyper-speed)
        delay = max(1, int(1000 / fps))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
            
    print("\n============================================================")
    print("                     rPPG ANALYSIS RESULT                   ")
    print("============================================================")
    print(f"Final Heart Rate (BPM): {last_bpm:.1f}")
    print(f"Signal-to-Noise Ratio:  {last_snr:.1f} dB")
    print("-" * 60)
    print(f"CONCLUSION: {status_text}")
    print("============================================================\n")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
