import numpy as np
import scipy.signal as signal
import sys

def main():
    # 1. Fake a 20-second video signal at 30 fps
    fps = 30
    duration = 20
    num_frames = fps * duration
    t = np.linspace(0, duration, num_frames, endpoint=False)
    
    true_bpm = 72
    freq = true_bpm / 60.0
    
    # Generate sine wave
    pulse = np.sin(2 * np.pi * freq * t)
    
    # Spread across RGB (green reacts most, red least)
    # Using typical average RGB values for skin
    R_base, G_base, B_base = 150, 100, 100
    
    R = R_base + 0.5 * pulse   # Red least
    G = G_base - 5.0 * pulse   # Green most
    B = B_base - 1.5 * pulse   # Blue medium
    
    # 2. Add random noise and slow lighting drift
    np.random.seed(42) # For reproducibility
    noise_level = 0.5
    drift = 10 * np.sin(2 * np.pi * 0.05 * t)
    
    R += np.random.normal(0, noise_level, num_frames) + drift
    G += np.random.normal(0, noise_level, num_frames) + drift
    B += np.random.normal(0, noise_level, num_frames) + drift
    
    # 3. Implement CHROM formula
    # Normalize each color channel
    R_n = R / np.mean(R)
    G_n = G / np.mean(G)
    B_n = B / np.mean(B)
    
    # Compute X and Y
    X = 3 * R_n - 2 * G_n
    Y = 1.5 * R_n + G_n - 1.5 * B_n
    
    # Bandpass filter both between 0.7 and 4 Hz
    nyq = 0.5 * fps
    b, a = signal.butter(3, [0.7 / nyq, 4.0 / nyq], btype='bandpass')
    
    X_f = signal.filtfilt(b, a, X)
    Y_f = signal.filtfilt(b, a, Y)
    
    # Combine using alpha = std(X) / std(Y)
    alpha = np.std(X_f) / np.std(Y_f)
    pulse_signal = X_f - alpha * Y_f
    
    # 4. Use Welch's method to find strongest frequency
    # We set nperseg to a large value (e.g. 10 seconds of data) for better frequency resolution
    f, Pxx = signal.welch(pulse_signal, fs=fps, nperseg=fps * 10)
    
    # Find strongest frequency peak in plausible human heart rate range (0.7 Hz to 4.0 Hz)
    valid_idx = np.where((f >= 0.7) & (f <= 4.0))[0]
    best_idx = valid_idx[np.argmax(Pxx[valid_idx])]
    
    detected_freq = f[best_idx]
    detected_bpm = detected_freq * 60
    
    # 5. Compare detected bpm to 72 bpm
    error = abs(detected_bpm - true_bpm)
    if error <= 2.0:
        print(f"PASS")
        print(f"Detected BPM: {detected_bpm:.2f} (Target: {true_bpm})")
        print(f"Error: {error:.2f} bpm")
    else:
        print(f"FAIL")
        print(f"Detected BPM: {detected_bpm:.2f} (Target: {true_bpm})")
        print(f"Error: {error:.2f} bpm")

if __name__ == "__main__":
    main()
