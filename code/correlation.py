import numpy as np

# Try GPU acceleration
try:
    import cupy as cp
    use_gpu = True
except ImportError:
    import numpy as cp
    use_gpu = False

def cross_correlation(x, y):
    """Compute cross-correlation with GPU (cupy) if available."""
    if use_gpu:
        x = cp.array(x)
        y = cp.array(y)
        corr = cp.correlate(x, y, mode="full")
        return cp.asnumpy(corr)
    else:
        return np.correlate(x, y, mode="full")

def find_matches(long_audio, ad_audio, sr, threshold=0.55):
    """Find matches of ad_audio inside long_audio using cross-correlation."""
    ad_len = len(ad_audio)
    matches = []

    # Step through in chunks
    chunk_size = sr * 12
    overlap = sr * 6

    for start in range(0, len(long_audio) - ad_len, chunk_size - overlap):
        chunk = long_audio[start:start + chunk_size]
        if len(chunk) < ad_len:
            break

        corr = cross_correlation(chunk, ad_audio)

        # Normalize correlation
        norm_factor = np.sqrt(np.sum(chunk**2) * np.sum(ad_audio**2))
        if norm_factor > 0:
            corr = corr / norm_factor

        # Find peaks above threshold
        peaks = np.where(corr > threshold)[0]
        for peak in peaks:
            match_time = (start + peak) / sr
            matches.append(match_time)

    # Adjust for ad duration
    ad_duration = len(ad_audio) / sr
    adjusted_matches = [t - ad_duration for t in matches if t - ad_duration >= 0]
    return np.unique(np.round(adjusted_matches, 2))
