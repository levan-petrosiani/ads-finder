# 🎵 Ad Finder – Audio Advertisement Detection with Cross-Correlation

Detect audio advertisements inside long recordings (TV or radio) using **cross-correlation**. Supports MP3/WAV files and GPU acceleration via **CuPy**.

---

## 📂 Project Structure

```
├── code/                  # All Python source files
│   ├── main.py            # Entry point for running detection
│   ├── loader.py          # Audio loading, conversion, normalization
│   ├── correlation.py     # Cross-correlation and matching logic
│   ├── results.py         # Pretty-print detection results
├── mp3_files/             # Input folder for MP3 files (ads + long recordings)
├── wav_files/             # Optional pre-converted WAV files
├── environment.yml        # Conda environment file
└── README.md              # Project documentation
```

---

## ⚙️ Installation (with Conda)

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/ad-finder.git
   cd ad-finder
   ```

2. Create a conda environment from `environment.yml`:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate ad-finder
   ```

4. Run the code from the `code/` folder:

   ```bash
   python code/main.py
   ```

---

## 📦 environment.yml

```yaml
name: ad-finder
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - scipy
  - pydub
  - pip
  - pip:
      - cupy-cuda12x  # optional, GPU acceleration
```

⚠️ **Note:** Adjust the `cupy` version to match your CUDA version or remove it if running on CPU.

---

## ▶️ Usage

1. Place your audio files in `mp3_files/` or `wav_files/`.
2. Edit `code/main.py` to select your long audio file and ad clip:

```python
LONG_AUDIO = "wav_files/chunk_22k_mono.wav"
AD_AUDIO = [
    "mp3_files/presidenti_ad.mp3",
    "mp3_files/sandoni_ad.mp3",
    "mp3_files/ninja_ad.mp3",
    "mp3_files/fino_ad.mp3"
]

run_matching(LONG_AUDIO, AD_AUDIO[3], threshold=0.55)
```

3. Run detection:

```bash
python code/main.py
```

---

## 🔧 How It Works

1. **Audio Preprocessing**

   * Converts all inputs to **mono 22kHz WAV**
   * Normalizes amplitude between -1 and 1

2. **Cross-Correlation Matching**

   * Slides the ad clip over the long recording
   * Computes normalized correlation
   * Peaks above threshold are considered matches

3. **Results**

   * Prints start/end times of matches
   * Adjusts matches by ad duration

---

## ⚡ GPU Acceleration

* Automatic if `cupy` is installed and compatible GPU is present.
* Falls back to **NumPy** on CPU.

---

## 📌 Notes

* Threshold (`threshold=0.55`) can be adjusted to reduce false positives/negatives.
* Audio is processed in **12-second chunks with 6-second overlap** for performance.
* Can be extended for **batch ad detection**.

---

## 👤 Author

Developed by **Levan Petrosiani** – built for detecting ads in media recordings.

---
