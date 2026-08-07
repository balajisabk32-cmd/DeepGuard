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
    
    # Need enough samples for the filter order.
    # Returned a bare `None` here while every other path returns a 2-tuple, so the
    # caller's `result[0]` raised TypeError. Unreachable in practice only because
    # the caller happens to gate on len(buffer) > fps*3.
    if len(X) < 15:
        return None, 0.0

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
    # Initialize OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
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
    if fps == 0 or np.isnan(fps):
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
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Convert BGR to RGB for rPPG processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if len(faces) > 0:
            # Sort faces by size (largest first) to avoid jumping to background faces
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            (x, y, w, h) = faces[0]
            
            # Smooth the bounding box to reduce jitter (which creates massive noise)
            if smooth_w == 0:
                smooth_x, smooth_y, smooth_w, smooth_h = x, y, w, h
            else:
                alpha_box = 0.1 # Smoothing factor (lower = smoother)
                smooth_x = int(alpha_box * x + (1 - alpha_box) * smooth_x)
                smooth_y = int(alpha_box * y + (1 - alpha_box) * smooth_y)
                smooth_w = int(alpha_box * w + (1 - alpha_box) * smooth_w)
                smooth_h = int(alpha_box * h + (1 - alpha_box) * smooth_h)
            
            # Define ROI (Region of Interest): Forehead
            # This avoids glasses, eyes blinking, and mouth movement!
            roi_x = max(0, smooth_x + int(smooth_w * 0.3))
            roi_y = max(0, smooth_y + int(smooth_h * 0.1))
            roi_w = int(smooth_w * 0.4)
            roi_h = int(smooth_h * 0.2)
            
            # Draw the ROI on the image for visualization
            cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            
            roi = image_rgb[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            if roi.size > 0:
                # Average colors across the ROI
                avg_color_per_row = np.average(roi, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                
                R_buffer.append(avg_color[0])
                G_buffer.append(avg_color[1])
                B_buffer.append(avg_color[2])
                
                # Compute BPM if we have at least 3 seconds of data
                if len(R_buffer) > fps * 3:
                    result = process_chrom(R_buffer, G_buffer, B_buffer, fps)
                    if result[0] is not None:
                        bpm, snr = result
                        last_bpm = bpm
                        last_snr = snr
                        
                        # DO NOT restore the old "snr > 1.0 -> REAL else DEEPFAKE" logic.
                        #
                        # Two things were wrong with it:
                        #  1. "No pulse => fake" is false for face swaps. The composite
                        #     keeps authentic skin outside the blend mask, so a swap
                        #     retains a pulse. What breaks is agreement BETWEEN facial
                        #     regions — see src/rppg/analyze.py and plan §3.4.
                        #  2. A 1.0 dB threshold means signal power only 26% above
                        #     noise. Any dark or compressed AUTHENTIC clip is accused.
                        #
                        # This script now reports signal quality only. It does not vote.
                        if snr > 6.0:
                            status_text = "PULSE SIGNAL: STRONG"
                            status_color = (0, 255, 0)
                        elif snr > 0.0:
                            status_text = "PULSE SIGNAL: WEAK"
                            status_color = (0, 255, 255)
                        else:
                            status_text = "PULSE SIGNAL: NOT RECOVERABLE"
                            status_color = (0, 165, 255)
                            
            # Display the BPM, SNR and Status
            cv2.putText(image, f"BPM: {last_bpm:.1f} (SNR: {last_snr:.1f}dB)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
            
        # Display the resulting frame
        cv2.imshow('rPPG CHROM Test', image)
        
        # Press 'q' to exit. Use a tiny delay so video plays fast.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
