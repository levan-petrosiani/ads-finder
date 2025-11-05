import os
import numpy as np
from pydub import AudioSegment

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio between -1 and 1."""
    max_val = np.max(np.abs(audio))
    return audio.astype(np.float32) / max_val if max_val > 0 else audio.astype(np.float32)

def load_mp3(path: str, sample_rate: int = 22050) -> tuple[int, np.ndarray]:
    """Load MP3 file directly as mono float32 PCM."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        audio = AudioSegment.from_mp3(path)
        audio = audio.set_channels(1).set_frame_rate(sample_rate)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = normalize_audio(samples)
        return sample_rate, samples
    except Exception as e:
        print(f"[WARN] Could not decode {path}: {e}")
        return None, None

def process_audio(input_mp3: str, sample_rate: int = 22050):
    """
    Full pipeline:
    1. Load MP3 directly.
    2. Normalize audio.
    """
    return load_mp3(input_mp3, sample_rate)
