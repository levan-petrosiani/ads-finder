import numpy as np
try:
    import cupy as cp
    use_gpu = True
except ImportError:
    import numpy as cp
    use_gpu = False
from scipy import signal


def create_constellation(audio, sr):
    """Generate constellation map from audio samples using STFT and peak finding."""
    window_length_seconds = 0.5
    window_length_samples = int(window_length_seconds * sr)
    window_length_samples += window_length_samples % 2
    num_peaks = 10  # Reduced for faster processing

    amount_to_pad = window_length_samples - audio.size % window_length_samples
    song_input = np.pad(audio, (0, amount_to_pad))

    if use_gpu:
        song_input = cp.array(song_input)
        nperseg = window_length_samples
        noverlap = nperseg // 2
        nfft = nperseg
        window = cp.hanning(nperseg)
        hop_size = nperseg - noverlap
        n_frames = (len(song_input) - noverlap) // hop_size
        stft = cp.zeros((nfft // 2 + 1, n_frames), dtype=cp.complex64)

        for i in range(n_frames):
            start = i * hop_size
            frame = song_input[start:start + nperseg]
            if len(frame) == nperseg:
                frame = frame * window
                stft[:, i] = cp.fft.rfft(frame, n=nfft)

        frequencies = cp.linspace(0, sr / 2, stft.shape[0])
        times = cp.arange(n_frames) * hop_size / sr
        stft = cp.abs(stft)

        constellation_map = []
        for time_idx in range(stft.shape[1]):
            spectrum = stft[:, time_idx]
            peaks = cp.argsort(spectrum)[-num_peaks:]
            for peak in peaks:
                if spectrum[peak] > 0.01:  # Stricter prominence
                    frequency = frequencies[peak]
                    constellation_map.append([time_idx, float(frequency)])
        constellation_map = np.array(constellation_map)
        cp.get_default_memory_pool().free_all_blocks()
    else:
        frequencies, times, stft = signal.stft(
            song_input, sr, nperseg=window_length_samples, nfft=window_length_samples, return_onesided=True
        )
        constellation_map = []
        for time_idx, window in enumerate(stft.T):
            spectrum = np.abs(window)
            peaks, props = signal.find_peaks(spectrum, prominence=0.01, distance=200)
            n_peaks = min(num_peaks, len(peaks))
            largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
            for peak in peaks[largest_peaks]:
                frequency = frequencies[peak]
                constellation_map.append([time_idx, frequency])

    return constellation_map


def create_hashes(constellation_map, item_id=None):
    """Generate combinatorial hashes from constellation map."""
    hashes = {}
    upper_frequency = 23000
    frequency_bits = 10

    for idx, (time, freq) in enumerate(constellation_map):
        for other_time, other_freq in constellation_map[idx : idx + 100]:
            diff = other_time - time
            if diff <= 1 or diff > 10:
                continue

            freq_binned = freq / upper_frequency * (2 ** frequency_bits)
            other_freq_binned = other_freq / upper_frequency * (2 ** frequency_bits)

            hash_val = int(freq_binned) | (int(other_freq_binned) << 10) | (int(diff) << 20)
            hashes[hash_val] = (time, item_id)
    return hashes


def build_database(ads):
    """Build fingerprint database for all ads."""
    database = {}
    song_name_index = {}

    for index, (ad_name, ad_audio, ad_sr) in enumerate(ads):
        song_name_index[index] = ad_name
        constellation = create_constellation(ad_audio, ad_sr)
        hashes = create_hashes(constellation, index)

        for hash_val, time_index_pair in hashes.items():
            if hash_val not in database:
                database[hash_val] = []
            database[hash_val].append(time_index_pair)

    return database, song_name_index


def match_clip(hashes, database, song_name_index, clip_duration):
    """Match clip hashes against database and return detected ads with offsets."""
    matches_per_ad = {}
    for hash_val, (clip_time, _) in hashes.items():
        if hash_val in database:
            matching_occurrences = database[hash_val]
            for source_time, ad_index in matching_occurrences:
                if ad_index not in matches_per_ad:
                    matches_per_ad[ad_index] = []
                matches_per_ad[ad_index].append((hash_val, clip_time, source_time))

    scores = {}
    for ad_index, matches in matches_per_ad.items():
        song_scores_by_offset = {}
        for hash_val, clip_time, source_time in matches:
            delta = clip_time - source_time
            if 0 <= delta <= clip_duration:  # Constrain offset to clip duration
                song_scores_by_offset[delta] = song_scores_by_offset.get(delta, 0) + 1

        if song_scores_by_offset:
            max_score = (0, 0)
            for offset, score in song_scores_by_offset.items():
                if score > max_score[1]:
                    max_score = (offset, score)
            scores[ad_index] = max_score
            print(f"[DEBUG] Ad {song_name_index[ad_index]}: Best offset {max_score[0]:.2f}s, score {max_score[1]}")

    sorted_scores = sorted(scores.items(), key=lambda x: x[1][1], reverse=True)
    detected_ads = []
    for ad_index, (offset, score) in sorted_scores:
        if score > 50:  # Higher threshold to reduce false positives
            detected_ads.append((song_name_index[ad_index], offset, score))

    return detected_ads