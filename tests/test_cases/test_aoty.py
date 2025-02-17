import sys
import os
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.scraper.aoty_scraper import get_album_url, scrape_album
from app.utils.matching import find_best_match, clean_title
import asyncpg
import time
import random

async def fetch_tracks():
    """Fetch track details from the database."""
    conn = await asyncpg.connect("postgresql://postgres:postgronner34@localhost:5432/Sonance")

    query = """
    SELECT t.id, t.title, a.name AS artist, al.title AS album
    FROM tracks t
    JOIN artists a ON t.artist_id = a.id
    JOIN albums al ON t.album_id = al.id
    LIMIT 30;
    """

    rows = await conn.fetch(query)  
    await conn.close()
    return rows  

async def test_fetch_aoty_scores():
    """Fetch tracks from DB and match them with AOTY album and track ratings."""
    tracks = await fetch_tracks()

    for track in tracks:
        artist, album, title = track["artist"], track["album"], track["title"]
        
        # ✅ Normalize album title formatting
        cleaned_title = clean_title(title)
        cleaned_album = clean_title(album, is_album=True)

        # ✅ Sleep before each request to avoid rate limiting
        sleep_time = random.uniform(2, 5)
        print(f"Sleeping for {sleep_time:.2f} seconds before searching AOTY...")
        time.sleep(sleep_time)

        # Fetch AOTY album URL (Deluxe versions first)
        album_data = await get_album_url(artist, cleaned_album)
        if not album_data:
            print(f"Album not found on AOTY: {artist} - {album}. Trying fuzzy matching.")
            
            # Attempt to find the best album match using fuzzy search
            possible_albums = await get_album_url(artist, "")  # Fetch multiple albums by artist
            if possible_albums:
                aoty_album_titles = [clean_title(a[2], is_album=True) for a in possible_albums]
                best_album_match = find_best_match(cleaned_album, aoty_album_titles, is_album=True)

                if best_album_match:
                    album_data = next((a for a in possible_albums if clean_title(a[2], is_album=True) == best_album_match), None)

        if not album_data:
            print(f"No matching album found on AOTY for: {artist} - {album}")
            continue

        album_url, fetched_artist, fetched_album = album_data
        print(f"Found AOTY URL: {album_url} for {artist} - {album} (Matched: {fetched_album})")

        # ✅ Sleep before scraping album details
        sleep_time = random.uniform(2, 5)
        print(f"Sleeping for {sleep_time:.2f} seconds before scraping album...")
        time.sleep(sleep_time)

        # Scrape the album details
        album_obj = await scrape_album(album_url, fetched_artist, fetched_album)
        
        # ✅ Find the best track match after cleaning
        aoty_track_titles = [clean_title(aoty_track.title) for aoty_track in album_obj.tracks]
        best_match = find_best_match(cleaned_title, aoty_track_titles)

        if best_match:
            matched_track = next((t for t in album_obj.tracks if clean_title(t.title) == best_match), None)
            if matched_track:
                print(f"Matched Track: {title} -> {matched_track.title} | AOTY Rating: {matched_track.rating}")
            else:
                print(f"No exact match found in AOTY for {title} (Best fuzzy match: {best_match})")
        else:
            print(f"No rating found for track: {title}")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_fetch_aoty_scores())  
