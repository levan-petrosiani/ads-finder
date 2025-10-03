import librosa
import numpy as np
import mysql.connector

# -----------------------------
# MySQL Connection Settings
# -----------------------------
DB_PARAMS = {
    'host': 'localhost',
    'user': 'root',
    'password': '232018',
    'database': 'audio_db'
}

# -----------------------------
# Fingerprint Computation
# -----------------------------
def compute_fingerprint(audio_path, sr=None):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return mfcc  # keep 2D

# -----------------------------
# Store Fingerprint in MySQL
# -----------------------------
def store_fingerprint(clip_name, fingerprint):
    fingerprint_bytes = fingerprint.astype(np.float32).tobytes()
    conn = mysql.connector.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO audio_fingerprints (clip_name, fingerprint) VALUES (%s, %s)",
        (clip_name, fingerprint_bytes)
    )
    conn.commit()
    cur.close()
    conn.close()

# -----------------------------
# Load Fingerprints from MySQL
# -----------------------------
def load_fingerprints():
    conn = mysql.connector.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT clip_name, fingerprint FROM audio_fingerprints")
    data = []
    for clip_name, fp_bytes in cur.fetchall():
        fp = np.frombuffer(fp_bytes, dtype=np.float32)
        fp = fp.reshape((20, -1))  # reshape back to (n_mfcc, n_frames)
        data.append((clip_name, fp))
    cur.close()
    conn.close()
    return data

# -----------------------------
# Recognize Clips in Long Audio
# -----------------------------
def recognize_clip(long_clip_path, threshold=0.7, step_frames=20, dedup_seconds=1.0):
    y_long, sr_long = librosa.load(long_clip_path, sr=None)
    long_mfcc = librosa.feature.mfcc(y=y_long, sr=sr_long, n_mfcc=20)
    fingerprints_db = load_fingerprints()
    hop_length = 512

    for clip_name, ad_fp in fingerprints_db:
        ad_len = ad_fp.shape[1]
        detected_times = []

        for start in range(0, long_mfcc.shape[1] - ad_len, step_frames):
            window = long_mfcc[:, start:start + ad_len]
            sim = np.sum(window * ad_fp) / (np.linalg.norm(window) * np.linalg.norm(ad_fp) + 1e-6)

            if sim > threshold:
                time_sec = start * hop_length / sr_long
                # Only add if far enough from previous detection
                if not detected_times or time_sec - detected_times[-1] > dedup_seconds:
                    detected_times.append(time_sec)
                    print(f"Detected '{clip_name}' at ~{time_sec:.2f}s! Confidence: {sim:.2f}")


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # 1️⃣ Store fingerprints of short ads
    ad_files = ["mp3_files/ad_clip.mp3"]  # add more ad clips here
    for ad_file in ad_files:
        fp = compute_fingerprint(ad_file)
        store_fingerprint(ad_file, fp)

    # 2️⃣ Recognize ads in a long TV clip
    recognize_clip("mp3_files/chunk.mp3")
