def format_time(seconds):
    """Convert seconds to MM:SS.s format."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"

def print_results(matches, ad_audio, sr):
    """Pretty-print detection results in minutes:seconds format."""
    ad_duration = len(ad_audio) / sr
    formatted_matches = []

    if len(matches) > 0:
        print(f"[RESULT] Found {len(matches)} matches:")
        for i, t in enumerate(matches, 1):
            start_time = format_time(t)
            end_time = format_time(t + ad_duration)
            print(f"  - Match {i}: Start {start_time} | End {end_time}")
            formatted_matches.append(f"{start_time}-{end_time}")
    else:
        print("[RESULT] No matches found.")
    
    return formatted_matches

