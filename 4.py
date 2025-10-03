import numpy as np
from scipy.io.wavfile import read
import mysql.connector
import time
import pickle
from scipy.signal import find_peaks
import librosa  # For spectral feature extraction

# MySQL connection
try:
    db = mysql.connector.connect(
        host="localhost",
        user="your_username",
        password="your_password",
        database="audio_processing"
    )
    cursor = db.cursor()
except mysql.connector.Error as e:
    print(f"Error connecting to MySQL: {e}")
    exit(1)

def extract_fingerprints(audio, sr, hop_length=512):
    """Extract spectral peak fingerprints."""
    # Compute spectrogram
    S = np.abs(librosa.stft(audio, hop_length=hop_length))
    # Find peaks in frequency domain
    fingerprints = []
    for t in range(S.shape[1]):
        peaks, _ = find_peaks(S[:, t], height=0.1 * np.max(S[:, t]))
        if len(peaks) > 0:
            # Store top 5 peaks as fingerprint
            top_peaks = peaks[np.argsort(S[peaks, t])[-5:]]
            hash_value = hash(tuple(top_peaks))
            fingerprints.append((hash_value, t * hop_length / sr))
    return fingerprints

def store_fingerprints(file_name, fingerprints, audio_id):
    """Store fingerprints in MySQL."""
    query = "INSERT INTO ad_fingerprints (ad_id, fingerprint_hash, time_offset) VALUES (%s, %s, %s)"
    cursor.executemany(query, [(audio_id, str(f[0]), f[1]) for f in fingerprints])
    db.commit()

def match_fingerprints(chunk_fingerprints, ad_id):
    """Match chunk fingerprints against stored ad fingerprints."""
    matches = []
    for hash_value, chunk_time in chunk_fingerprints:
        cursor.execute(
            "SELECT time_offset FROM ad_fingerprints WHERE ad_id = %s AND fingerprint_hash = %s",
            (ad_id, str(hash_value))
        )
        results = cursor.fetchall()
        for ad_time, in results:
            matches.append((chunk_time - ad_time, 1.0))  # Simplified score
    return matches

start_time = time.perf_counter()

# File paths
long_path = "wav_files/chunk_22k.wav"
ads_path = "wav_files/ad_clip_22k.wav"

# Load audio
try:
    sr, ad_audio = read(ads_path)
    _, tv_audio = read(long_path)
except FileNotFoundError:
    print(f"Error: WAV files not found. Ensure {ads_path} and {long_path} exist.")
    exit(1)
load_time = time.perf_counter()
print(f"Loading: {load_time - start_time:.2f} seconds")

# Check/store audio in MySQL
cursor.execute("SELECT id FROM audio_files WHERE file_name = %s", (ads_path,))
result = cursor.fetchone()
if result:
    ad_id = result[0]
else:
    duration = len(ad_audio) / sr
    audio_blob = pickle.dumps(ad_audio)
    cursor.execute(
        "INSERT INTO audio_files (file_name, sample_rate, audio_data, duration) VALUES (%s, %s, %s, %s)",
        (ads_path, sr, audio_blob, duration)
    )
    db.commit()
    ad_id = cursor.lastrowid

cursor.execute("SELECT id FROM audio_files WHERE file_name = %s", (long_path,))
result = cursor.fetchone()
if result:
    chunk_id = result[0]
else:
    duration = len(tv_audio) / sr
    audio_blob = pickle.dumps(tv_audio)
    cursor.execute(
        "INSERT INTO audio_files (file_name, sample_rate, audio_data, duration) VALUES (%s, %s, %s, %s)",
        (long_path, sr, audio_blob, duration)
    )
    db.commit()
    chunk_id = cursor.lastrowid

# Extract fingerprints
ad_fingerprints = extract_fingerprints(ad_audio, sr)
store_fingerprints(ads_path, ad_fingerprints, ad_id)
chunk_fingerprints = extract_fingerprints(tv_audio, sr)  # Cache this for reuse
fp_time = time.perf_counter()
print(f"Fingerprint extraction: {fp_time - load_time:.2f} seconds")

# Match fingerprints
matches = match_fingerprints(chunk_fingerprints, ad_id)
match_time = time.perf_counter()
print(f"Fingerprint matching: {match_time - fp_time:.2f} seconds")

# Process matches (simplified: group by time offset)
peak_times = []
for time_offset, score in matches:
    if 0 <= time_offset <= len(tv_audio) / sr - len(ad_audio) / sr:
        peak_times.append(time_offset)
peak_times = np.unique(np.round(peak_times, decimals=2))
peak_time = time.perf_counter()
print(f"Peak processing: {peak_time - match_time:.2f} seconds")

# Store results in MySQL
for peak_time in peak_times:
    cursor.execute(
        "INSERT INTO ad_matches (chunk_id, ad_id, peak_time, correlation_score) VALUES (%s, %s, %s, %s)",
        (chunk_id, ad_id, float(peak_time), 1.0)  # Placeholder score
    )
db.commit()

# Output results
duration = len(ad_audio) / sr
if len(peak_times) > 0:
    print(f"Found {len(peak_times)} distinct matches at:")
    for i, t in enumerate(peak_times, 1):
        print(f"Match {i} at {t:.2f} seconds, End at {t + duration:.2f} seconds")
else:
    print("No matches found.")

end_time = time.perf_counter()
print(f"Total execution time: {end_time - start_time:.6f} seconds")

# Close database
cursor.close()
db.close()