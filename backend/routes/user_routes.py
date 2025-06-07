from fastapi import APIRouter, HTTPException, Query, Request, Depends
from sqlalchemy.orm import Session
from time import time

from app.models.database import SessionLocal
from app.models.aoty_models import UserProfile
from app.services.aoty_scraper_service import get_user_profile
from app.utils.cache import (
    get_cache,
    set_cache,
    USER_TTL,
)
from app.utils.metrics import metrics
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get(
    "/",
    response_model=UserProfile,
    summary="Get User Profile",
    description="Retrieve a user's profile information from Album of the Year.",
    response_description="User profile information",
    responses={
        404: {"description": "User not found"},
        503: {"description": "Error accessing user profile"},
    },
)
@limiter.limit("30/minute")
async def get_user_endpoint(
    request: Request,
    username: str = Query(
        ..., description="Username on albumoftheyear.org", example="evrynoiseatonce"
    ),
    refresh: bool = Query(False, description="Force refresh the cache"),
):
    start_time = time()
    try:
        cache_key = f"user:{username}"
        
        # Check cache unless refresh is requested
        if not refresh and (cached_result := await get_cache(cache_key)):
            metrics.record_request(cache_hit=True, endpoint="user")
            return UserProfile(**cached_result)

        metrics.record_request(cache_hit=False, endpoint="user")
        user_profile = await get_user_profile(username)

        await set_cache(cache_key, user_profile.dict(), USER_TTL)
        metrics.record_response_time(time() - start_time)
        return user_profile

    except HTTPException:
        raise
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=503, detail=str(e))


# Legacy endpoint for backward compatibility
@router.get(
    "/aoty-profile/",
    response_model=UserProfile,
    summary="Get AOTY User Profile (Legacy)",
    description="Legacy endpoint - Retrieve a user's profile information from Album of the Year.",
    responses={
        404: {"description": "User not found"},
        503: {"description": "Error accessing user profile"},
    },
)
async def get_aoty_user_legacy(
    request: Request,
    username: str = Query(..., description="Username on albumoftheyear.org", example="evrynoiseatonce"),
    db: Session = Depends(get_db)
):
    """Legacy endpoint - redirects to the new user profile endpoint"""
    return await get_user_endpoint(request, username)