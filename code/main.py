
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from loader import process_audio
from correlation import find_matches
from results import format_time  # import formatting function
import time

TV_DIR = "1TVb"
ADS_DIR = "Subwaves"
RESULTS_FILE = "results.csv"
SAMPLE_RATE = 22050
THRESHOLD = 0.45
MAX_WORKERS = 8  # Adjust to your CPU cores

def match_single_ad(tv_audio, sr, tv_file, ad_file):
    """Match a single ad against the loaded TV clip."""
    ad_path = os.path.join(ADS_DIR, ad_file)
    _, ad_audio = process_audio(ad_path, sr)
    if ad_audio is None:
        return tv_file, ad_file, None
    matches = find_matches(tv_audio, ad_audio, sr, THRESHOLD)

    # Convert to "MM:SS - MM:SS" format
    ad_duration = len(ad_audio) / sr
    formatted_matches = [
        f"{format_time(t)} - {format_time(t + ad_duration)}"
        for t in matches
    ] if matches is not None else []
    
    return tv_file, ad_file, formatted_matches

if __name__ == "__main__":
    program_start_time = time.perf_counter()

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tv_clip", "ad", "found"])

    # Iterate over all TV clips
    for tv_file in os.listdir(TV_DIR):
        if not tv_file.endswith(".mp3"):
            continue

        tv_path = os.path.join(TV_DIR, tv_file)
        print(f"[INFO] Loading TV clip: {tv_file}")
        sr, tv_audio = process_audio(tv_path, SAMPLE_RATE)
        if tv_audio is None:
            print(f"[WARNING] Skipping {tv_file} (failed to load).")
            continue

        results_list = []
        # Match all ads in parallel using threads
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(match_single_ad, tv_audio, sr, tv_file, ad_file): ad_file
                for ad_file in os.listdir(ADS_DIR) if ad_file.endswith(".mp3")
            }

            for fut in as_completed(futures):
                results_list.append(fut.result())

        # Write results to CSV
        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            for tv_file, ad_file, found in results_list:
                # Join multiple matches with comma if more than one
                writer.writerow([tv_file, ad_file, ", ".join(found) if found else "No match"])
        
        print(f"[INFO] Finished TV clip: {tv_file}")
    
    print(f"[INFO] Program Total time: {time.perf_counter() - program_start_time:.2f}s")
