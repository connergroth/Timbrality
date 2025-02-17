import sys
import os
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.scraper.aoty_scraper import get_album_url, scrape_album
from app.utils.matching import find_best_match
import asyncpg

async def fetch_tracks():
    conn = await asyncpg.connect("postgresql://postgres:postgronner34@localhost:5432/Sonance")

    query = """
    SELECT t.id, t.title, a.name AS artist, al.title AS album
    FROM tracks t
    JOIN artists a ON t.artist_id = a.id
    JOIN albums al ON t.album_id = al.id
    LIMIT 10;
    """

    rows = await conn.fetch(query)  
    await conn.close()
    return rows  


async def test_fetch_aoty_scores():
    tracks = await fetch_tracks()

    for track in tracks:
        artist, album, title = track["artist"], track["album"], track["title"]
        
        # Fetch AOTY album URL
        album_data = await get_album_url(artist, album)
        if not album_data:
            print(f"Album not found on AOTY: {artist} - {album}")
            continue

        album_url, fetched_artist, fetched_title = album_data
        print(f"Found AOTY URL: {album_url} for {artist} - {album}")

        # Scrape the album details
        album_obj = await scrape_album(album_url, fetched_artist, fetched_title)
        
        # Find the track rating
        for aoty_track in album_obj.tracks:
            if aoty_track.title.lower() == title.lower():
                print(f"Track: {title} | AOTY Rating: {aoty_track.rating}")
                break
        else:
            print(f"No rating found for track: {title}")

# Run the test
if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_fetch_aoty_scores())
