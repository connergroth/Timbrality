from fastapi import APIRouter, HTTPException, Query, Request, Depends
from sqlalchemy.orm import Session
from app.services.aoty_service import AOTYService
from app.cache.redis import get_cached_user_profile, cache_user_profile
from app.models.aoty import UserProfile
from app.utils.metrics import metrics
from app.models.database import SessionLocal

router = APIRouter()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get(
    "/aoty-profile/",
    response_model=UserProfile,
    summary="Get AOTY User Profile",
    description="Retrieve a user's profile information from Album of the Year.",
    responses={
        404: {"description": "User not found"},
        503: {"description": "Error accessing user profile"},
    },
)
async def get_aoty_user(
    request: Request,
    username: str = Query(..., description="Username on albumoftheyear.org", example="evrynoiseatonce"),
    db: Session = Depends(get_db)
):
    # Check cache first
    cached_profile = await get_cached_user_profile(username)
    if cached_profile:
        metrics.record_request(cache_hit=True)
        return UserProfile(**cached_profile)

    # Create AOTY service
    aoty_service = AOTYService()
    
    try:
        # Call the API endpoint instead of the scraper
        response = await aoty_service.client.get(
            f"{aoty_service.base_url}/user/",
            params={"username": username}
        )
        response.raise_for_status()
        
        # Process the API response
        user_profile = response.json()
        
        # Cache the result
        await cache_user_profile(username, user_profile)
        
        metrics.record_request(cache_hit=False)
        return UserProfile(**user_profile)
    
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=503, detail=str(e))
    
    finally:
        await aoty_service.close()