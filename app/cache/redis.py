import os
import json
import httpx

from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

if not REDIS_URL or not REDIS_TOKEN:
    raise ValueError(
        "Redis configuration missing. Please set UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN"
    )

HEADERS = {"Authorization": f"Bearer {REDIS_TOKEN}"}

# Cache TTLs (in seconds)
ALBUM_TTL = 604800
SIMILAR_TTL = 86400
USER_TTL = 3600


async def set_cache(key: str, value: Any, expire_seconds: int = 3600) -> None:
    """Set a value in Redis cache with expiration"""
    try:
        value_str = json.dumps(value)
        pipeline_data = [["SET", key, value_str], ["EXPIRE", key, str(expire_seconds)]]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{REDIS_URL}/pipeline", headers=HEADERS, json=pipeline_data
            )
            response.raise_for_status()

    except Exception as e:
        print(f"Error setting cache: {str(e)}")


async def get_cache(key: str) -> Optional[Any]:
    """Get a value from Redis cache"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{REDIS_URL}/get/{key}", headers=HEADERS)
            response.raise_for_status()

            data = response.json()
            if data["result"]:
                return json.loads(data["result"])

    except Exception as e:
        print(f"Error getting cache: {str(e)}")

    return None


async def delete_cache(key: str) -> None:
    """Delete a value from Redis cache"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{REDIS_URL}/del/{key}", headers=HEADERS)
            response.raise_for_status()

    except Exception as e:
        print(f"Error deleting cache: {str(e)}")

# Album data caching


async def cache_album_data(album_id: str, album_data: dict) -> None:
    """Cache album data in Redis"""
    try:
        # Set a TTL of 1 week (604800 seconds)
        await set_cache(f"album:{album_id}", album_data, expire_seconds=ALBUM_TTL)
        print(f"Album data cached for {album_id}.")
    except Exception as e:
        print(f"Error caching album data for {album_id}: {str(e)}")


async def get_cached_album(album_id: str) -> Optional[dict]:
    """Retrieve cached album data from Redis"""
    try:
        cached_album = await get_cache(f"album:{album_id}")
        if cached_album:
            print(f"Album data retrieved from cache for {album_id}.")
            return cached_album
        else:
            print(f"No cached data found for album {album_id}.")
            return None
    except Exception as e:
        print(f"Error retrieving cached album data for {album_id}: {str(e)}")
    return None


# User data caching


async def cache_user_profile(user_id: str, user_data: dict) -> None:
    """Cache user profile data in Redis"""
    try:
        # Set a TTL of 1 hour (3600 seconds)
        await set_cache(f"user:{user_id}", user_data, expire_seconds=USER_TTL)
        print(f"User profile cached for {user_id}.")
    except Exception as e:
        print(f"Error caching user profile for {user_id}: {str(e)}")


async def get_cached_user_profile(user_id: str) -> Optional[dict]:
    """Retrieve cached user profile data from Redis"""
    try:
        cached_user_profile = await get_cache(f"user:{user_id}")
        if cached_user_profile:
            print(f"User profile retrieved from cache for {user_id}.")
            return cached_user_profile
        else:
            print(f"No cached data found for user {user_id}.")
            return None
    except Exception as e:
        print(f"Error retrieving cached user profile for {user_id}: {str(e)}")
    return None


# Song data caching


async def cache_song_data(song_id: str, song_data: dict) -> None:
    """Cache song data in Redis"""
    try:
        # Set a TTL of 1 week (604800 seconds) for song data
        await set_cache(f"song:{song_id}", song_data, expire_seconds=ALBUM_TTL)
        print(f"Song data cached for {song_id}.")
    except Exception as e:
        print(f"Error caching song data for {song_id}: {str(e)}")


async def get_cached_song(song_id: str) -> Optional[dict]:
    """Retrieve cached song data from Redis"""
    try:
        cached_song = await get_cache(f"song:{song_id}")
        if cached_song:
            print(f"Song data retrieved from cache for {song_id}.")
            return cached_song
        else:
            print(f"No cached data found for song {song_id}.")
            return None
    except Exception as e:
        print(f"Error retrieving cached song data for {song_id}: {str(e)}")
    return None


# Similar Songs data caching


async def cache_similar_songs(song_id: str, similar_songs: list) -> None:
    """Cache similar songs data in Redis"""
    try:
        # Set a TTL of 1 day (86400 seconds) for similar songs
        await set_cache(f"similar_songs:{song_id}", similar_songs, expire_seconds=SIMILAR_TTL)
        print(f"Similar songs cached for {song_id}.")
    except Exception as e:
        print(f"Error caching similar songs for {song_id}: {str(e)}")


async def get_cached_similar_songs(song_id: str) -> Optional[list]:
    """Retrieve cached similar songs data from Redis"""
    try:
        cached_similar_songs = await get_cache(f"similar_songs:{song_id}")
        if cached_similar_songs:
            print(f"Similar songs retrieved from cache for {song_id}.")
            return cached_similar_songs
        else:
            print(f"No cached similar songs found for {song_id}.")
            return None
    except Exception as e:
        print(f"Error retrieving cached similar songs for {song_id}: {str(e)}")
    return None


# Similar Albums data caching


async def cache_similar_albums(album_id: str, similar_albums: list) -> None:
    """Cache similar albums data in Redis"""
    try:
        # Set a TTL of 1 day (86400 seconds) for similar albums
        await set_cache(f"similar_albums:{album_id}", similar_albums, expire_seconds=SIMILAR_TTL)
        print(f"Similar albums cached for {album_id}.")
    except Exception as e:
        print(f"Error caching similar albums for {album_id}: {str(e)}")


async def get_cached_similar_albums(album_id: str) -> Optional[list]:
    """Retrieve cached similar albums data from Redis"""
    try:
        cached_similar_albums = await get_cache(f"similar_albums:{album_id}")
        if cached_similar_albums:
            print(f"Similar albums retrieved from cache for {album_id}.")
            return cached_similar_albums
        else:
            print(f"No cached similar albums found for {album_id}.")
            return None
    except Exception as e:
        print(f"Error retrieving cached similar albums for {album_id}: {str(e)}")
    return None


# Album Cover data caching


async def cache_album_cover(album_id: str, cover_url: str) -> None:
    """Cache the album cover URL in Redis"""
    try:
        await set_cache(f"album_cover:{album_id}", cover_url, expire_seconds=ALBUM_TTL)
        print(f"Album cover cached for {album_id}.")
    except Exception as e:
        print(f"Error caching album cover for {album_id}: {str(e)}")


async def get_cached_album_cover(album_id: str, default_cover_url: Optional[str] = "/static/images/sonance-logo.PNG") -> Optional[str]:
    """Retrieve cached album cover URL from Redis"""
    try:
        cached_cover_url = await get_cache(f"album_cover:{album_id}")
        if cached_cover_url:
            print(f"Album cover URL retrieved from cache for {album_id}.")
            return cached_cover_url
        else:
            print(f"No cached album cover URL found for {album_id}.")
            return default_cover_url  # Return the default cover URL if not found in cache
    except Exception as e:
        print(f"Error retrieving cached album cover URL for {album_id}. Returning default.: {str(e)}")
        return default_cover_url  # Return the default cover URL in case of an error


# Song Cover data caching


async def cache_song_cover(song_id: str, cover_url: str) -> None:
    """Cache the song cover URL in Redis"""
    try:
        await set_cache(f"song_cover:{song_id}", cover_url, expire_seconds=ALBUM_TTL)
        print(f"Song cover cached for {song_id}.")
    except Exception as e:
        print(f"Error caching song cover for {song_id}: {str(e)}")


async def get_cached_song_cover(song_id: str, default_cover_url: Optional[str] = "/static/images/sonance-logo.PNG") -> Optional[str]:
    """Retrieve cached song cover URL from Redis"""
    try:
        cached_cover_url = await get_cache(f"song_cover:{song_id}")
        if cached_cover_url:
            print(f"Song cover URL retrieved from cache for {song_id}.")
            return cached_cover_url
        else:
            print(f"No cached song cover URL found for {song_id}.")
            return default_cover_url  # Return the default cover URL if not found in cache
    except Exception as e:
        print(f"Error retrieving cached song cover URL for {song_id}. Returning default.: {str(e)}")
        return default_cover_url  # Return the default cover URL in case of an error


# Playlist Cover data caching 


async def cache_playlist_cover(playlist_id: str, cover_url: str) -> None:
    """Cache the playlist cover URL in Redis"""
    try:
        await set_cache(f"playlist_cover:{playlist_id}", cover_url, expire_seconds=ALBUM_TTL)
        print(f"Playlist cover cached for {playlist_id}.")
    except Exception as e:
        print(f"Error caching playlist cover for {playlist_id}: {str(e)}")



async def get_cached_playlist_cover(playlist_id: str, default_cover_url: Optional[str] = "/static/images/sonance-logo.PNG") -> Optional[str]:
    """Retrieve cached playlist cover URL from Redis"""
    try:
        cached_cover_url = await get_cache(f"playlist_cover:{playlist_id}")
        if cached_cover_url:
            print(f"Playlist cover URL retrieved from cache for {playlist_id}.")
            return cached_cover_url
        else:
            print(f"No cached playlist cover URL found for {playlist_id}.")
            return default_cover_url  # Return the default cover URL if not found in cache
    except Exception as e:
        print(f"Error retrieving cached playlist cover URL for {playlist_id}. Returning default.: {str(e)}")
        return default_cover_url  # Return the default cover URL in case of an error