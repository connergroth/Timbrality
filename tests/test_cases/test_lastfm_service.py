import os
import sys
import asyncio
import asyncpg
import aiohttp
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.services.lastfm_service import fetch_track_metadata

load_dotenv()

LASTFM_API_KEY = os.getenv("API_KEY")
LASTFM_USERNAME = "connergroth"

BASE_URL = "http://ws.audioscrobbler.com/2.0/"

# ✅ Fetch Tracks from Database
async def fetch_tracks():
    """Fetch track details from the database."""
    conn = await asyncpg.connect("postgresql://postgres:postgronner34@localhost:5432/Sonance")

    query = """
    SELECT t.id, t.title, a.name AS artist
    FROM tracks t
    JOIN artists a ON t.artist_id = a.id
    LIMIT 10;
    """

    rows = await conn.fetch(query)
    await conn.close()
    return rows

# ✅ Test Last.fm API for Stored Tracks
async def test_lastfm():
    """Test Last.fm API with tracks from the database."""
    tracks = await fetch_tracks()

    async with aiohttp.ClientSession() as session:
        for track in tracks:
            track_id, title, artist = track["id"], track["title"], track["artist"]
            print(f"\nFetching Last.fm Data for: {artist} - {title}")

            # ✅ Fetch metadata (play count + tags)
            metadata = fetch_track_metadata(title)

            # ✅ Extract data
            play_count = metadata.get("playcount", "N/A")
            tags = metadata.get("tags", [])

            print(f"Play Count: {play_count}")
            print(f"Tags: {', '.join(tags) if tags else 'None'}")

# ✅ Run Test Script
if __name__ == "__main__":
    asyncio.run(test_lastfm())