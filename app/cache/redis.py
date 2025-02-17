import os
import json
import httpx
import asyncio
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
ALBUM_TTL = 604800  # 7 days
SIMILAR_TTL = 86400  # 1 day
USER_TTL = 3600  # 1 hour
RECOMMENDATION_TTL = 604800  # 7 days

# --- Redis Helper Functions ---

async def set_cache(key: str, value: Any, expire_seconds: int) -> None:
    """Set a value in Redis cache with expiration."""
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
    """Get a value from Redis cache."""
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
    """Delete a value from Redis cache."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{REDIS_URL}/del/{key}", headers=HEADERS)
            response.raise_for_status()
    except Exception as e:
        print(f"Error deleting cache: {str(e)}")

# --- Album Data Caching ---

async def cache_album_data(album_id: str, album_data: dict) -> None:
    await set_cache(f"album:{album_id}", album_data, expire_seconds=ALBUM_TTL)

async def get_cached_album(album_id: str) -> Optional[dict]:
    return await get_cache(f"album:{album_id}")

# --- User Data Caching ---

async def cache_user_profile(user_id: str, user_data: dict) -> None:
    await set_cache(f"user:{user_id}", user_data, expire_seconds=USER_TTL)

async def get_cached_user_profile(user_id: str) -> Optional[dict]:
    return await get_cache(f"user:{user_id}")

# --- Song Data Caching ---

async def cache_song_data(song_id: str, song_data: dict) -> None:
    await set_cache(f"song:{song_id}", song_data, expire_seconds=ALBUM_TTL)

async def get_cached_song(song_id: str) -> Optional[dict]:
    return await get_cache(f"song:{song_id}")

# --- Similar Songs Caching ---

async def cache_similar_songs(song_id: str, similar_songs: list) -> None:
    await set_cache(f"similar_songs:{song_id}", similar_songs, expire_seconds=SIMILAR_TTL)

async def get_cached_similar_songs(song_id: str) -> Optional[list]:
    return await get_cache(f"similar_songs:{song_id}")

# --- Recommendations Caching ---

async def cache_user_recommendations(user_id: str, recommendations: list) -> None:
    await set_cache(f"recommendations:{user_id}", recommendations, expire_seconds=RECOMMENDATION_TTL)

async def get_cached_user_recommendations(user_id: str) -> Optional[list]:
    return await get_cache(f"recommendations:{user_id}")



