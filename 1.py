import librosa
import numpy as np
from scipy.signal import correlate, find_peaks, fftconvolve
import time

start_time = time.perf_counter()

long_path = "mp3_files/chunk.mp3"
ads_path = "mp3_files/ad_clip.mp3"

ad_audio, sr = librosa.load(ads_path, mono=True, sr=None)
tv_audio, _ = librosa.load(long_path, mono=True, sr=sr)

# Normalize audio to reduce amplitude differences
ad_audio = ad_audio / np.max(np.abs(ad_audio))
tv_audio = tv_audio / np.max(np.abs(tv_audio))


# Perform cross-correlation
# correlation = correlate(tv_audio, ad_audio, mode='valid')
correlation = fftconvolve(tv_audio, ad_audio[::-1], mode='valid')
lags = np.arange(len(correlation))

# Convert lag indices to time (seconds)
time_lags = lags / sr

# Find peaks in correlation (potential matches)
# Adjust threshold based on your audio (e.g., 0.8 is a high correlation)
ad_duration_samples = len(ad_audio)
threshold = 0.8 * np.max(correlation)  # Adjust threshold if needed
peaks, properties = find_peaks(correlation, height=threshold, distance=ad_duration_samples)
peak_times = peaks / sr  # Convert to seconds


duration = librosa.get_duration(y=ad_audio, sr=sr)
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
