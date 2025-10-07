import time
from loader import process_audio
from correlation import find_matches
from results import print_results


def run_matching(long_path, ad_path, threshold=0.55):
    start_time = time.perf_counter()

    sr, ad_audio = process_audio(ad_path, "temp_ad.wav", 22050)
    _, long_audio = process_audio(long_path, "temp_long.wav", 22050)
    print(f"[INFO] Loaded audio files.")

    matches = find_matches(long_audio, ad_audio, sr, threshold)
    print(f"[INFO] Matching completed.")
    print_results(matches, ad_audio, sr)

    print(f"[INFO] Total time: {time.perf_counter() - start_time:.2f}s")
    return matches


if __name__ == "__main__":
    LONG_AUDIO = "mp3_files/chunk.mp3"

    AD_AUDIO = ["mp3_files/presidenti_ad.mp3",
                "mp3_files/sandoni_ad.mp3",
                "mp3_files/ninja_ad.mp3",
                "mp3_files/fino_ad.mp3"
                ]


    run_matching(LONG_AUDIO, AD_AUDIO[3], threshold=0.55)


