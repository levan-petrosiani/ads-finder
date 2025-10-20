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
        result = cp.asnumpy(corr)
        # Explicitly free GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        return result
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

    # Deduplicate matches within a small time window (e.g., 0.1 seconds)
    if adjusted_matches:
        adjusted_matches = np.sort(adjusted_matches)
        deduplicated_matches = []
        tolerance = 0.1  # Merge matches within 0.1 seconds
        current_match = adjusted_matches[0]
        deduplicated_matches.append(current_match)

        for t in adjusted_matches[1:]:
            if t - current_match > tolerance:
                deduplicated_matches.append(t)
                current_match = t

        adjusted_matches = np.round(deduplicated_matches, 2)
    else:
        adjusted_matches = np.array([])

    return np.unique(adjusted_matches)