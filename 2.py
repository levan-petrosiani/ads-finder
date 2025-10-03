import librosa
import cupy as cp  # Replace numpy with cupy
import numpy as np
from scipy.signal import find_peaks  # Keep scipy for peak finding (CPU)
from cupyx.scipy.signal import fftconvolve  # CuPy's fftconvolve
import time

start_time = time.perf_counter()

# # File paths
long_path = "mp3_files/chunk.mp3"
ads_path = "mp3_files/ad_clip.mp3"

# Load audio with Librosa (runs on CPU)
ad_audio, sr = librosa.load(ads_path, mono=True, sr=None)
tv_audio, _ = librosa.load(long_path, mono=True, sr=sr)

# Convert to CuPy arrays for GPU processing
ad_audio = cp.asarray(ad_audio)
tv_audio = cp.asarray(tv_audio)

# Normalize audio on GPU
ad_audio = ad_audio / cp.max(cp.abs(ad_audio))
tv_audio = tv_audio / cp.max(cp.abs(tv_audio))

# Perform cross-correlation using CuPy's fftconvolve
# Note: fftconvolve expects ad_audio reversed for convolution
correlation = fftconvolve(tv_audio, ad_audio[::-1], mode='valid')
lags = cp.arange(len(correlation))

# Convert correlation to NumPy for peak finding (since find_peaks is CPU-based)
correlation_np = cp.asnumpy(correlation)  # Transfer to CPU
lags_np = cp.asnumpy(lags)

# Convert lag indices to time (seconds)
time_lags = lags_np / sr

# Find peaks in correlation (on CPU)
ad_duration_samples = len(ad_audio)
threshold = 0.8 * np.max(correlation_np)  # Use NumPy max for CPU array
peaks, properties = find_peaks(correlation_np, height=threshold, distance=ad_duration_samples)
peak_times = peaks / sr  # Convert to seconds

# Get ad duration
duration = librosa.get_duration(y=cp.asnumpy(ad_audio), sr=sr)  # Librosa needs NumPy array

# Output results
if len(peak_times) > 0:
    print(f"Found {len(peak_times)} distinct matches at:")
    for i, t in enumerate(peak_times, 1):
        print(f"Match {i} at {t:.2f} seconds, End at {t + duration:.2f} seconds")
else:
    print("No matches found above threshold.")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.6f} seconds")

# Free GPU memory explicitly
del ad_audio, tv_audio, correlation, lags
cp.get_default_memory_pool().free_all_blocks()