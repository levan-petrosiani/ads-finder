def print_results(matches, ad_audio, sr):
    """Pretty-print detection results in MM:SS format."""
    ad_duration = len(ad_audio) / sr
    if len(matches) > 0:
        print(f"[RESULT] Found {len(matches)} matches:")
        for i, t in enumerate(matches, 1):
            # Convert start time to MM:SS
            start_time = round(t)  # Round to whole seconds
            start_minutes = int(start_time // 60)
            start_seconds = int(start_time % 60)
            start_formatted = f"{start_minutes:02d}:{start_seconds:02d}"
            
            # Convert end time to MM:SS
            end_time = round(t + ad_duration)  # Round to whole seconds
            end_minutes = int(end_time // 60)
            end_seconds = int(end_time % 60)
            end_formatted = f"{end_minutes:02d}:{end_seconds:02d}"
            
            print(f"  - Match {i}: Start {start_formatted} | End {end_formatted}")
    else:
        print("[RESULT] No matches found.")