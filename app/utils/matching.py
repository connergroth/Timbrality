from thefuzz import process

def find_best_match(track_title, track_list, score_cutoff=60):
    """
    Find the closest match for a track title in a given list of tracks.

    Args:
        track_title (str): The track title to match.
        track_list (list): List of track titles to compare against.
        score_cutoff (int): Minimum match score (default: 60).

    Returns:
        str | None: Best matching track title or None if no good match is found.
    """
    if not track_list:  # Handle empty track list case
        print(f"Warning: No tracks available for matching against '{track_title}'")
        return None

    # Try to find multiple close matches
    matches = process.extract(track_title, track_list, limit=3)

    # Pick the best match above cutoff
    for match in matches:
        best_match, score = match
        if score >= score_cutoff:
            return best_match

    print(f"Warning: No close match found for track: {track_title}")
    return None
