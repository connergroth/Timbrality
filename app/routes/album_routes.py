from fastapi import APIRouter, HTTPException, Query, Request
from app.utils.scraper import get_album_url, scrape_album, get_similar_albums
from app.utils.redis import get_cache, set_cache, ALBUM_TTL, SIMILAR_TTL
from app.models.aoty import Album
from app.utils.metrics import metrics
from slowapi.util import get_remote_address
from slowapi import Limiter
from time import time

router = APIRouter()

limiter = Limiter(key_func=get_remote_address)

@router.get(
    "/",
    response_model=Album,
    summary="Get Album Details",
    description="Retrieve information about an album.",
    responses={
        404: {"description": "Album not found"},
        503: {"description": "Error accessing album site"},
    },
)
@limiter.limit("30/minute")
async def get_album(
    request: Request,
    artist: str = Query(..., description="Name of the artist", example="Radiohead"),
    album: str = Query(..., description="Name of the album", example="In Rainbows"),
):
    start_time = time()
    try:
        cache_key = f"album:{artist}:{album}"
        if cached_result := await get_cache(cache_key):
            metrics.record_request(cache_hit=True)
            return Album(**cached_result)

        metrics.record_request(cache_hit=False)
        result = await get_album_url(artist, album)
        if not result:
            raise HTTPException(status_code=404, detail="Album not found")

        url, artist, title = result
        album_data = await scrape_album(url, artist, title)

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
    response_model=list[Album],
    summary="Get Similar Albums",
    description="Find albums similar to a specified album.",
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
):
    start_time = time()
    try:
        cache_key = f"similar:{artist}:{album}"
        if cached_result := await get_cache(cache_key):
            metrics.record_request(cache_hit=True)
            return [Album(**album_data) for album_data in cached_result]

        metrics.record_request(cache_hit=False)
        result = await get_album_url(artist, album)
        if not result:
            raise HTTPException(status_code=404, detail="Album not found")

        url, _, _ = result
        similar_albums = await get_similar_albums(url)

        # Cache the list of albums as dictionaries
        await set_cache(cache_key, [album.dict() for album in similar_albums], SIMILAR_TTL)
        metrics.record_response_time(time() - start_time)
        return similar_albums

    except HTTPException:
        raise
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=503, detail=str(e))
