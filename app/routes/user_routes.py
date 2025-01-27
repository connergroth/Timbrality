from fastapi import APIRouter, HTTPException, Query, Request
from app.utils.scraper import get_user_profile
from app.utils.redis import get_cache, set_cache, USER_TTL
from app.models.aoty import UserProfile
from app.utils.metrics import metrics

@app.get(
    "/user/",
    response_model=UserProfile,
    summary="Get User Profile",
    description="Retrieve a user's profile information.",
    response_description="User profile information",
    responses={
        404: {"description": "User not found"},
        503: {"description": "Error accessing user profile"},
    },
)
@limiter.limit("30/minute")
async def get_user(
    request: Request,
    username: str = Query(
        ..., description="Username on albumoftheyear.org", example="evrynoiseatonce"
    ),
):
    start_time = time()
    try:
        cache_key = f"user:{username}"
        if cached_result := await get_cache(cache_key):
            metrics.record_request(cache_hit=True)
            return UserProfile(**cached_result)

        metrics.record_request(cache_hit=False)
        user_profile = await get_user_profile(username)

        await set_cache(cache_key, user_profile.dict(), USER_TTL)
        metrics.record_response_time(time() - start_time)
        return user_profile

    except HTTPException:
        raise
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=503, detail=str(e))
