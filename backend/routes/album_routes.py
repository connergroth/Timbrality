from fastapi import APIRouter, HTTPException, Query, Request
from time import time
from typing import List

try:
    from models.aoty_models import Album, SearchResult
    from services.aoty_scraper_service import (
        get_album_url,
        scrape_album,
        search_albums,
        get_similar_albums
    )
    from utils.cache import (
        get_cache,
        set_cache,
        ALBUM_TTL,
        SIMILAR_TTL,
        SEARCH_TTL,
    )
    from utils.metrics import metrics
except ImportError:
    # Fallback for development - create simple mock objects
    
    # Mock classes
    class Album:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def dict(self):
            return self.__dict__
    
    class SearchResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def dict(self):
            return self.__dict__
    
    # Mock functions
    async def get_album_url(*args, **kwargs):
        return None
    
    async def scrape_album(*args, **kwargs):
        return Album(title="Mock Album", artist="Mock Artist", error="Service not available")
    
    async def search_albums(*args, **kwargs):
        return []
    
    async def get_similar_albums(*args, **kwargs):
        return []
    
    async def get_cache(*args, **kwargs):
        return None
    
    async def set_cache(*args, **kwargs):
        pass
    
    ALBUM_TTL = 3600
    SIMILAR_TTL = 3600
    SEARCH_TTL = 3600
    
    class MockMetrics:
        def record_request(self, *args, **kwargs): pass
        def record_response_time(self, *args, **kwargs): pass
        def record_error(self, *args, **kwargs): pass
    
    metrics = MockMetrics()
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Dependency to get the database session


@router.get(
    "/",
    summary="Get Album Details",
    description="Retrieve detailed information about an album.",
    response_description="Detailed album information including tracks, reviews, and more.",
    responses={
        404: {"description": "Album not found"},
        503: {"description": "Error accessing album site"},
    },
)
@limiter.limit("30/minute")
async def get_album_endpoint(
    request: Request,
    artist: str = Query(..., description="Name of the artist", example="Radiohead"),
    album: str = Query(..., description="Name of the album", example="OK Computer"),
    refresh: bool = Query(False, description="Force refresh the cache"),
):
    start_time = time()
    try:
        cache_key = f"album:{artist}:{album}"
        
        # Check cache unless refresh is requested
        if not refresh and (cached_result := await get_cache(cache_key)):
            metrics.record_request(cache_hit=True, endpoint="album")
            return Album(**cached_result)

        metrics.record_request(cache_hit=False, endpoint="album")
        result = await get_album_url(artist, album)
        if not result:
            raise HTTPException(status_code=404, detail="Album not found")

        url, artist_name, title = result
        album_data = await scrape_album(url, artist_name, title)

        await set_cache(cache_key, album_data.dict(), ALBUM_TTL)
        metrics.record_response_time(time() - start_time)
        return album_data

    except HTTPException:
        raise
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=503, detail=str(e))


@router.get(
    "/similar/",
    summary="Get Similar Albums",
    description="Find albums similar to a specified album.",
    response_description="List of albums similar to the specified album",
    responses={
        404: {"description": "Album not found"},
        503: {"description": "Error accessing album site"},
    },
)
@limiter.limit("30/minute")
async def get_similar_albums_endpoint(
    request: Request,
    artist: str = Query(..., description="Name of the artist", example="Radiohead"),
    album: str = Query(..., description="Name of the album", example="OK Computer"),
    refresh: bool = Query(False, description="Force refresh the cache"),
    limit: int = Query(5, description="Maximum number of similar albums to return", ge=1, le=10),
):
    start_time = time()
    try:
        cache_key = f"similar:{artist}:{album}:{limit}"
        
        # Check cache unless refresh is requested
        if not refresh and (cached_result := await get_cache(cache_key)):
            metrics.record_request(cache_hit=True, endpoint="similar")
            return [Album(**album_data) for album_data in cached_result]

        metrics.record_request(cache_hit=False, endpoint="similar")
        result = await get_album_url(artist, album)
        if not result:
            raise HTTPException(status_code=404, detail="Album not found")

        url, _, _ = result
        similar_albums = await get_similar_albums(url, limit)

        # Cache the list of albums as dictionaries
        await set_cache(
            cache_key, 
            [album.dict() for album in similar_albums], 
            SIMILAR_TTL
        )
        
        metrics.record_response_time(time() - start_time)
        return similar_albums

    except HTTPException:
        raise
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=503, detail=str(e))


@router.get(
    "/search/",
    summary="Search Albums",
    description="Search for albums matching the query.",
    response_description="List of albums matching the search query",
    responses={
        503: {"description": "Error accessing album site"},
    },
)
@limiter.limit("30/minute")
async def search_albums_endpoint(
    request: Request,
    query: str = Query(..., description="Search query", example="Radiohead OK Computer"),
    limit: int = Query(10, description="Maximum number of results to return", ge=1, le=20),
):
    start_time = time()
    try:
        cache_key = f"search:{query}:{limit}"
        
        # Check cache first
        if cached_result := await get_cache(cache_key):
            metrics.record_request(cache_hit=True, endpoint="search")
            return [SearchResult(**result) for result in cached_result]

        metrics.record_request(cache_hit=False, endpoint="search")
        results = await search_albums(query, limit)

        # Cache the search results
        await set_cache(
            cache_key,
            [result.dict() for result in results],
            SEARCH_TTL
        )
        
        metrics.record_response_time(time() - start_time)
        return results

    except HTTPException:
        raise
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=503, detail=str(e))


# Legacy endpoints for backward compatibility
@router.get("/enrich/{album_id}")
async def enrich_album(album_id: str):
    """Legacy endpoint - Enrich an album with AOTY metadata."""
    # This could be updated to use the new AOTY functionality
    # For now, return a placeholder response
    return {"message": "Album enrichment functionality has been updated. Use the new /album/ endpoint instead."}


@router.get("/similar/{album_id}")
async def get_similar_albums_legacy(
    album_id: str, 
    limit: int = Query(5, gt=0, le=10)
):
    """Legacy endpoint - Get similar albums for a given album."""
    # This could be updated to use the new AOTY functionality
    # For now, return a placeholder response
    return {"message": "Similar albums functionality has been updated. Use the new /album/similar/ endpoint instead."}