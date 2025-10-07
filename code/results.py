def print_results(matches, ad_audio, sr):
    """Pretty-print detection results."""
    ad_duration = len(ad_audio) / sr
    if len(matches) > 0:
        print(f"[RESULT] Found {len(matches)} matches:")
        for i, t in enumerate(matches, 1):
            print(f"  - Match {i}: Start {t:.2f}s | End {t + ad_duration:.2f}s")
    else:
        print("[RESULT] No matches found.")
