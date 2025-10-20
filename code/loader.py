import os
import numpy as np
from scipy.io.wavfile import read
from pydub import AudioSegment
import io


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Convert audio to float32 and normalize between -1 and 1."""
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio


def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 22050) -> str:
    """Convert any audio file to mono WAV with given sample rate."""
    audio = AudioSegment.from_file(input_path)
    audio_mono = audio.set_channels(1).set_frame_rate(sample_rate)
    audio_mono.export(output_path, format="wav")
    print(f"[INFO] Converted {input_path} → {output_path} ({sample_rate}Hz mono)")
    return output_path


def load_audio(path: str, sample_rate: int = 22050) -> tuple[int, np.ndarray]:
    """
    Load audio file (MP3 or WAV), convert if needed, and normalize.
    Returns (sample_rate, normalized_audio).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # If input is MP3 → convert to temp WAV first
    if path.lower().endswith(".mp3"):
        wav_path = path.replace(".mp3", "_temp.wav")
        convert_to_wav(path, wav_path, sample_rate)
        path = wav_path

    sr, audio = read(path)
    return sr, normalize_audio(audio)


def load_audio_in_memory(path: str, sample_rate: int = 22050) -> tuple[int, np.ndarray]:
    """
    Load and process audio file in memory without saving to disk.
    Handles MP3 or other formats, converts to mono at given sample rate, normalizes.
    Returns (sample_rate, normalized_audio).
    """
    try:
        audio = AudioSegment.from_file(path)
        audio_mono = audio.set_channels(1).set_frame_rate(sample_rate)

        buffer = io.BytesIO()
        audio_mono.export(buffer, format="wav")
        buffer.seek(0)

        sr, audio_data = read(buffer)
        normalized_audio = normalize_audio(audio_data)
        return sr, normalized_audio
    except Exception as e:
        # Re-raise the exception to be handled by the caller
        raise Exception(f"Failed to process audio file {path}: {e}")