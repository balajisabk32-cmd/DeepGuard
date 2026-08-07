@echo off
echo ============================================================
echo                    DEEPGUARD ANALYSIS
echo ============================================================
echo Analyzing Video: %1
echo.
echo [TEST 1] Running Lip-Sync Confidence Detection...
deepfake_env\Scripts\python Wav2Lip\detect_video.py %1

echo.
echo [TEST 2] Running rPPG Heartbeat Detection...
deepfake_env\Scripts\python rppg_test\test_chrom_real.py %1
