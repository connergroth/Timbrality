from thefuzz import process
import re

import re

def clean_title(title: str, is_album=False) -> str:
    """
    Normalize track and album titles.
    - Removes featured artists from track titles.
    - Tries both base album title and deluxe/extended versions.
    """
    if is_album:
        # Normalize different formatting styles (parentheses & brackets)
        title = title.replace("(", "").replace(")", "").replace("[", "").replace("]", "")

    else:
        # Remove featured artist info for track titles
        title = re.split(r"\s*\(feat\.?|\s*\(ft\.?|\s*\(featuring|\s*\(with", title, maxsplit=1)[0]

    return title.strip()


def find_best_match(query_title, title_list, score_cutoff=70, is_album=False):
    """
    Find the closest match for a track or album title in a given list.

    Args:
        query_title (str): The title to match (track or album).
        title_list (list): List of titles to compare against.
        score_cutoff (int): Minimum match score (default: 70).
        is_album (bool): Whether the input is an album title (default: False).

    Returns:
        str | None: Best matching title or None if no good match is found.
    """
    if not title_list:
        print(f"Warning: No titles available for matching against '{query_title}'")
        return None

    # Clean query and list titles
    cleaned_query = clean_title(query_title, is_album)
    cleaned_title_list = [clean_title(t, is_album) for t in title_list]

    print(f"\nSearching for: '{cleaned_query}' in Titles: {cleaned_title_list}\n")

    # Check for direct match first
    if cleaned_query in cleaned_title_list:
        matched_title = title_list[cleaned_title_list.index(cleaned_query)]
        print(f"Exact Match Found: '{query_title}' -> '{matched_title}'")
        return matched_title  # Return original uncleaned match

    # Use fuzzy matching if no exact match is found
    matches = process.extract(cleaned_query, cleaned_title_list, limit=3)

    for best_match, score in matches:
        original_match = title_list[cleaned_title_list.index(best_match)]
        print(f"Fuzzy Match Attempt: '{cleaned_query}' -> '{original_match}' (Score: {score})")

        if score >= score_cutoff:
            print(f"Fuzzy Matched Title: '{query_title}' -> '{original_match}' (Score: {score})")
            return original_match  # Return original uncleaned match

    print(f"No close match found for title: '{query_title}'\n")
    return None
