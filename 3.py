import numpy as np
from scipy.io.wavfile import read
import time
try:
    import cupy as cp
    use_gpu = True
except ImportError:
    import numpy as cp
    use_gpu = False
import os

def cross_correlation(x, y):
    """Compute cross-correlation using GPU or CPU."""
    if use_gpu:
        x = cp.array(x)
        y = cp.array(y)
        corr = cp.correlate(x, y, mode='full')
        corr = cp.asnumpy(corr)  # Convert back to numpy
    else:
        corr = np.correlate(x, y, mode='full')
    return corr

def find_matches(long_audio, ad_audio, sr, threshold=0.65):
    """Find matches using cross-correlation with chunking."""
    ad_len = len(ad_audio)
    chunk_size = sr * 8  # 8-second chunks
    chunk_overlap = sr * 4  # 4-second overlap
    matches = []

    for start in range(0, len(long_audio) - ad_len, chunk_size - chunk_overlap):
        chunk = long_audio[start:start + chunk_size]
        if len(chunk) < ad_len:
            break
        corr = cross_correlation(chunk, ad_audio)
        # Normalize correlation
        norm_factor = np.sqrt(np.sum(chunk**2) * np.sum(ad_audio**2))
        if norm_factor > 0:
            corr = corr / norm_factor
        # Dynamic threshold based on local signal energy
        local_threshold = max(threshold, 0.5 * np.max(corr))
        peaks = np.where(corr > local_threshold)[0]
        for peak in peaks:
            match_time = (start + peak) / sr
            matches.append(match_time)
    
    # Cluster matches within ad duration, report median time
    ad_duration = ad_len / sr
    if not matches:
        return []
    matches = np.sort(matches)
    clusters = []
    current_cluster = [matches[0]]
    for t in matches[1:]:
        if t - current_cluster[-1] <= ad_duration:
            current_cluster.append(t)
        else:
            if len(current_cluster) >= 3:  # Require at least 3 hits
                clusters.append(np.median(current_cluster))  # Use median time
            current_cluster = [t]
    if len(current_cluster) >= 3:
        clusters.append(np.median(current_cluster))
    
    return np.unique(np.round(clusters, 2))

start_time = time.perf_counter()

# File paths
long_path = "wav_files/chunk_22k.wav"
ads_path = "wav_files/ad_clip_22k.wav"

# Load audio
try:
    sr, ad_audio = read(ads_path)
    _, long_audio = read(long_path)
except FileNotFoundError:
    print(f"Error: WAV files not found. Ensure {ads_path} and {long_path} exist.")
    exit(1)
load_time = time.perf_counter()
print(f"Loading: {load_time - start_time:.2f} seconds")

# Convert to float32 and normalize
ad_audio = ad_audio.astype(np.float32) / np.max(np.abs(ad_audio))
long_audio = long_audio.astype(np.float32) / np.max(np.abs(long_audio))
norm_time = time.perf_counter()
print(f"Normalization: {norm_time - load_time:.2f} seconds")

# Find matches
matches = find_matches(long_audio, ad_audio, sr, threshold=0.65)
match_time = time.perf_counter()
print(f"Correlation and matching: {match_time - norm_time:.2f} seconds")

# Output results
ad_duration = len(ad_audio) / sr
if len(matches) > 0:
    print(f"Found {len(matches)} distinct matches at:")
    for i, t in enumerate(matches, 1):
        print(f"Match {i} at {t:.2f} seconds, End at {t + ad_duration:.2f} seconds")
else:
    print("No matches found.")

end_time = time.perf_counter()
print(f"Total execution time: {end_time - start_time:.6f} seconds")