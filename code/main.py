import time
import os
import gc
import csv
from loader import load_audio_in_memory
from correlation import find_matches
from results import print_results


def save_to_csv(tv_clip_path, ad_name, matches, ad_duration):
    """Save successful matches to matches.csv."""
    csv_file = "matches.csv"
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(["tv_clip", "ad_name", "start_time", "end_time", "duration"])
        
        for t in matches:
            # Convert start time to MM:SS
            start_time = round(t)
            start_minutes = int(start_time // 60)
            start_seconds = int(start_time % 60)
            start_formatted = f"{start_minutes:02d}:{start_seconds:02d}"
            
            # Convert end time to MM:SS
            end_time = round(t + ad_duration)
            end_minutes = int(end_time // 60)
            end_seconds = int(end_time % 60)
            end_formatted = f"{end_minutes:02d}:{end_seconds:02d}"
            
            # Write match to CSV
            writer.writerow([tv_clip_path, ad_name, start_formatted, end_formatted, round(ad_duration)])


def perform_matching(long_audio, sr, ad_name, ad_audio, ad_sr, tv_clip_path, threshold=0.55):
    """Perform matching for a single ad against preloaded long audio."""
    start_time = time.perf_counter()

    if ad_sr != sr:
        print(f"[WARNING] Sample rate mismatch for {ad_name}: long audio ({sr} Hz) vs ad ({ad_sr} Hz)")
        return []

    print(f"[INFO] Processing ad: {ad_name}")
    matches = find_matches(long_audio, ad_audio, sr, threshold)
    print(f"[INFO] Matching completed for {ad_name}.")
    print_results(matches, ad_audio, sr)

    # Save matches to CSV if any are found
    if len(matches) > 0:
        ad_duration = len(ad_audio) / sr
        save_to_csv(tv_clip_path, ad_name, matches, ad_duration)

    print(f"[INFO] Total time for {ad_name}: {time.perf_counter() - start_time:.2f}s")
    return matches


if __name__ == "__main__":
    # Define input directories
    TV_CLIPS_DIR = "1TVb"
    ADS_DIR = "Subwaves"
    SAMPLE_RATE = 22050
    THRESHOLD = 0.55

    # Load all valid ads into memory once
    start_time = time.perf_counter()
    ads = []
    ad_files = [f for f in os.listdir(ADS_DIR) if f.lower().endswith(".mp3")]
    for ad_file in ad_files:
        ad_path = os.path.join(ADS_DIR, ad_file)
        try:
            ad_sr, ad_audio = load_audio_in_memory(ad_path, SAMPLE_RATE)
            ads.append((ad_file, ad_audio, ad_sr))
            print(f"[INFO] Loaded ad in memory: {ad_file}")
        except Exception as e:
            print(f"[ERROR] Skipping corrupted or invalid ad file {ad_file}: {e}")
            continue
    print(f"[INFO] Loaded {len(ads)} ads into memory in: {time.perf_counter() - start_time:.2f}s")

    if not ads:
        print(f"[ERROR] No valid ads found in {ADS_DIR}. Exiting.")
        exit(1)

    # Loop over each TV clip in 1TVb directory
    overall_start_time = time.perf_counter()
    tv_clips = [f for f in os.listdir(TV_CLIPS_DIR) if f.lower().endswith(".mp3")]
    if not tv_clips:
        print(f"[ERROR] No valid TV clips found in {TV_CLIPS_DIR}. Exiting.")
        exit(1)

    for tv_clip in tv_clips:
        tv_clip_path = os.path.join(TV_CLIPS_DIR, tv_clip)
        print(f"\n[INFO] Processing TV clip: {tv_clip_path}")

        # Load TV clip in memory
        tv_start_time = time.perf_counter()
        try:
            sr, tv_audio = load_audio_in_memory(tv_clip_path, SAMPLE_RATE)
            print(f"[INFO] Loaded TV clip in memory in: {time.perf_counter() - tv_start_time:.2f}s")
        except Exception as e:
            print(f"[ERROR] Skipping corrupted or invalid TV clip {tv_clip_path}: {e}")
            continue

        # Matching loop for all ads against this TV clip
        correlation_start_time = time.perf_counter()
        for ad_name, ad_audio, ad_sr in ads:
            perform_matching(tv_audio, sr, ad_name, ad_audio, ad_sr, tv_clip_path, THRESHOLD)
        print(f"[INFO] Correlation time for {tv_clip_path}: {time.perf_counter() - correlation_start_time:.2f}s")

        # clean up TV audio memory
        del tv_audio
        gc.collect()

    # clear ads memory at the end
    del ads
    gc.collect()

    print(f"[INFO] Total processing time: {time.perf_counter() - overall_start_time:.2f}s")