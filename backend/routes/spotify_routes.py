from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from services.spotify_service import SpotifyService
from typing import Optional
import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/spotify", tags=["spotify"])

def get_client_credentials_token():
    """Get access token using client credentials flow."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Spotify credentials not configured")
    
    # Create base64 encoded credentials
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    
    # Request token
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data={
            "grant_type": "client_credentials"
        }
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to get Spotify access token")
    
    return response.json()["access_token"]

@router.get("/current-playing/{username}")
async def get_current_playing_track(username: str):
    """Get demo currently playing track using client credentials (no user auth needed)."""
    try:
        # Since client credentials can't access user's current playing track,
        # we'll return a demo track with proper album art for display purposes
        access_token = get_client_credentials_token()
        
        # Search for a popular track to demonstrate album art display
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers={
                "Authorization": f"Bearer {access_token}"
            },
            params={
                "q": "Kendrick Lamar good kid",
                "type": "track",
                "limit": 1
            }
        )
        
        if response.status_code == 200:
            search_data = response.json()
            tracks = search_data["tracks"]["items"]
            
            if tracks:
                track = tracks[0]
                return JSONResponse(content={
                    "is_playing": True,
                    "track": {
                        "id": track["id"],
                        "name": track["name"],
                        "artist": track["artists"][0]["name"],
                        "artists": [artist["name"] for artist in track["artists"]],
                        "album": track["album"]["name"],
                        "album_id": track["album"]["id"],
                        "artwork_url": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                        "preview_url": track.get("preview_url"),
                        "external_urls": track["external_urls"],
                        "duration_ms": track["duration_ms"],
                        "progress_ms": 120000  # Demo progress
                    },
                    "message": f"Demo: {track['name']} by {track['artists'][0]['name']}"
                })
        
        # Fallback if search fails
        return JSONResponse(content={
            "is_playing": False,
            "track": None,
            "message": "Not Playing - Spotify"
        })
        
    except Exception as e:
        return JSONResponse(content={
            "is_playing": False,
            "track": None,
            "message": f"Error: {str(e)}"
        })

@router.get("/album/{album_id}")
async def get_album_info(album_id: str):
    """Get album information using client credentials."""
    try:
        access_token = get_client_credentials_token()
        
        response = requests.get(
            f"https://api.spotify.com/v1/albums/{album_id}",
            headers={
                "Authorization": f"Bearer {access_token}"
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Album not found")
        
        album_data = response.json()
        
        return JSONResponse(content={
            "id": album_data["id"],
            "name": album_data["name"],
            "artist": album_data["artists"][0]["name"],
            "images": album_data["images"],
            "external_urls": album_data["external_urls"]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search-album")
async def search_album(q: str):
    """Search for an album by name."""
    try:
        access_token = get_client_credentials_token()
        
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers={
                "Authorization": f"Bearer {access_token}"
            },
            params={
                "q": q,
                "type": "album",
                "limit": 1
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Search failed")
        
        search_data = response.json()
        albums = search_data["albums"]["items"]
        
        if not albums:
            raise HTTPException(status_code=404, detail="Album not found")
        
        album = albums[0]
        
        return JSONResponse(content={
            "id": album["id"],
            "name": album["name"],
            "artist": album["artists"][0]["name"],
            "images": album["images"],
            "external_urls": album["external_urls"]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search-track")
async def search_track(q: str):
    """Search for a track by name and artist."""
    try:
        access_token = get_client_credentials_token()
        
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers={
                "Authorization": f"Bearer {access_token}"
            },
            params={
                "q": q,
                "type": "track",
                "limit": 1
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Search failed")
        
        search_data = response.json()
        tracks = search_data["tracks"]["items"]
        
        if not tracks:
            raise HTTPException(status_code=404, detail="Track not found")
        
        track = tracks[0]
        
        return JSONResponse(content={
            "id": track["id"],
            "name": track["name"],
            "artist": track["artists"][0]["name"],
            "artists": [artist["name"] for artist in track["artists"]],
            "album": track["album"]["name"],
            "album_id": track["album"]["id"],
            "artwork_url": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
            "preview_url": track.get("preview_url"),
            "external_urls": track["external_urls"],
            "duration_ms": track["duration_ms"],
            "popularity": track["popularity"]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 