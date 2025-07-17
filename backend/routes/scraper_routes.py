from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
from slowapi import Limiter
from slowapi.util import get_remote_address

try:
    # Use CloudScraper-based scraper (much more reliable)
    from ..scraper.aoty_cloudscraper import (
        get_album_url,
        search_albums,
        close_browser
    )
    # Fallback to old scraper for full scraping features
    from ..scraper.aoty_scraper import (
        scrape_album,
        get_similar_albums,
        get_user_profile,
    )
    from ..models.aoty_models import Album, SearchResult, UserProfile
except ImportError:
    # Fall back to absolute imports
    from scraper.aoty_cloudscraper import (
        get_album_url,
        search_albums,
        close_browser
    )
    from scraper.aoty_scraper import (
        scrape_album,
        get_similar_albums,
        get_user_profile,
    )
    from models.aoty_models import Album, SearchResult, UserProfile

# Create the router
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/search", summary="Search for albums", response_model=List[SearchResult])
@limiter.limit("30/minute")
async def search_albums_endpoint(
    request,
    query: str = Query(..., description="Search query for albums"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results to return")
):
    """
    Search for albums on AOTY by query string.
    
    Returns a list of search results with basic album information.
    """
    try:
        results = await search_albums(query, limit)
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/album/find", summary="Find specific album URL")
@limiter.limit("30/minute")
async def find_album_url_endpoint(
    request,
    artist: str = Query(..., description="Artist name"),
    album: str = Query(..., description="Album title")
):
    """
    Find the AOTY URL for a specific album by artist and title.
    
    Returns the album URL, confirmed artist name, and album title.
    """
    try:
        result = await get_album_url(artist, album)
        if result is None:
            raise HTTPException(status_code=404, detail="Album not found")
        
        url, artist_name, album_title = result
        return {
            "url": url,
            "artist": artist_name,
            "album": album_title
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Album search failed: {str(e)}")


@router.get("/album/scrape", summary="Scrape full album data", response_model=Album)
@limiter.limit("20/minute")
async def scrape_album_endpoint(
    request,
    url: str = Query(..., description="AOTY album URL to scrape"),
    artist: str = Query(..., description="Artist name"),
    title: str = Query(..., description="Album title")
):
    """
    Scrape complete album information from AOTY.
    
    Returns detailed album data including tracks, reviews, metadata, and more.
    """
    try:
        album = await scrape_album(url, artist, title)
        return album
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Album scraping failed: {str(e)}")


@router.get("/album/scrape-by-search", summary="Find and scrape album", response_model=Album)
@limiter.limit("15/minute")
async def scrape_album_by_search_endpoint(
    request,
    artist: str = Query(..., description="Artist name"),
    album: str = Query(..., description="Album title")
):
    """
    Find an album by artist/title and scrape its complete data.
    
    This is a convenience endpoint that combines finding the album URL and scraping it.
    """
    try:
        # First find the album URL
        result = await get_album_url(artist, album)
        if result is None:
            raise HTTPException(status_code=404, detail="Album not found")
        
        url, artist_name, album_title = result
        
        # Then scrape the album data
        album = await scrape_album(url, artist_name, album_title)
        return album
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Album scraping failed: {str(e)}")


@router.get("/album/similar", summary="Get similar albums", response_model=List[Album])
@limiter.limit("20/minute")
async def get_similar_albums_endpoint(
    request,
    url: str = Query(..., description="AOTY album URL to find similar albums for"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of similar albums to return")
):
    """
    Get similar albums for a given album.
    
    Returns a list of albums similar to the provided album.
    """
    try:
        similar = await get_similar_albums(url, limit)
        return similar
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar albums search failed: {str(e)}")


@router.get("/user/{username}", summary="Get user profile", response_model=UserProfile)
@limiter.limit("20/minute")
async def get_user_profile_endpoint(
    request,
    username: str
):
    """
    Get complete user profile information from AOTY.
    
    Returns user stats, favorite albums, recent reviews, and more.
    """
    try:
        profile = await get_user_profile(username)
        return profile
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User profile retrieval failed: {str(e)}")


@router.post("/cleanup", summary="Close browser instances")
@limiter.limit("5/minute")
async def cleanup_browser_endpoint(request):
    """
    Close any open browser instances to free up resources.
    
    This endpoint is useful for cleaning up after scraping operations.
    """
    try:
        await close_browser()
        return {"message": "Browser instances closed successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Cleanup failed: {str(e)}"}
        )


@router.get("/status", summary="Check scraper status")
async def scraper_status():
    """
    Check if the scraper is operational.
    
    Returns basic status information about the scraper service.
    """
    return {
        "status": "operational",
        "service": "AOTY Scraper",
        "version": "1.0.0",
        "features": [
            "Album search",
            "Full album scraping",
            "Similar albums discovery",
            "User profile retrieval"
        ]
    } 