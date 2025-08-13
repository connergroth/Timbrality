"""
Timbral ML Service Proxy Routes

Proxies requests from the main Timbre backend to the Timbral ML microservice,
providing a unified API while keeping ML logic isolated.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import httpx
import logging
import os
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Create router
timbral_router = APIRouter(prefix="/timbral", tags=["Timbral ML Service"])

# ML service configuration
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8001")
ML_SERVICE_TIMEOUT = 30.0


class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    seed_track_id: Optional[str] = None
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None


class TrackMetadata(BaseModel):
    track_id: str
    title: str
    artist: str
    album: str
    genres: Optional[List[str]] = None
    moods: Optional[List[str]] = None


async def get_ml_client():
    """Get HTTP client for ML service communication"""
    return httpx.AsyncClient(timeout=ML_SERVICE_TIMEOUT)


# ========== Health & Status ==========

@timbral_router.get("/health")
async def check_ml_service_health():
    """Check if Timbral ML service is healthy"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ML_SERVICE_URL}/api/v1/health")
            response.raise_for_status()
            return {
                "status": "healthy",
                "ml_service": response.json(),
                "service_url": ML_SERVICE_URL
            }
    except Exception as e:
        logger.error(f"ML service health check failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"ML service unavailable: {str(e)}"
        )


@timbral_router.get("/")
async def ml_service_info():
    """Get ML service information"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ML_SERVICE_URL}/")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get ML service info: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"ML service unavailable: {str(e)}"
        )


# ========== Recommendations ==========

@timbral_router.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations from Timbral ML service"""
    try:
        async with httpx.AsyncClient(timeout=ML_SERVICE_TIMEOUT) as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/api/v1/recommendations",
                json=request.dict()
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"ML service returned error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"ML service error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get recommendations: {str(e)}"
        )


@timbral_router.get("/recommendations/{user_id}")
async def get_user_recommendations(user_id: int, top_k: int = 10):
    """Get recommendations for a specific user"""
    try:
        async with httpx.AsyncClient(timeout=ML_SERVICE_TIMEOUT) as client:
            response = await client.get(
                f"{ML_SERVICE_URL}/api/v1/recommendations/{user_id}",
                params={"top_k": top_k}
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"ML service returned error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to get user recommendations"
        )
    except Exception as e:
        logger.error(f"Error getting user recommendations: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get user recommendations: {str(e)}"
        )


@timbral_router.get("/similar/{item_id}")
async def get_similar_items(item_id: int, top_k: int = 10):
    """Get items similar to a given item"""
    try:
        async with httpx.AsyncClient(timeout=ML_SERVICE_TIMEOUT) as client:
            response = await client.get(
                f"{ML_SERVICE_URL}/api/v1/similar/{item_id}",
                params={"top_k": top_k}
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"ML service returned error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to get similar items"
        )
    except Exception as e:
        logger.error(f"Error getting similar items: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get similar items: {str(e)}"
        )


# ========== Explainability ==========

@timbral_router.get("/explain/{user_id}/{item_id}")
async def explain_recommendation(user_id: int, item_id: int):
    """Get explanation for why an item was recommended"""
    try:
        async with httpx.AsyncClient(timeout=ML_SERVICE_TIMEOUT) as client:
            response = await client.get(
                f"{ML_SERVICE_URL}/api/v1/explain/{user_id}/{item_id}"
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"ML service returned error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to get recommendation explanation"
        )
    except Exception as e:
        logger.error(f"Error getting explanation: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get explanation: {str(e)}"
        )


# ========== Feedback ==========

@timbral_router.post("/feedback")
async def submit_feedback(user_id: int, item_id: int, rating: float):
    """Submit user feedback to improve recommendations"""
    try:
        async with httpx.AsyncClient(timeout=ML_SERVICE_TIMEOUT) as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/api/v1/feedback",
                params={
                    "user_id": user_id,
                    "item_id": item_id, 
                    "rating": rating
                }
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"ML service returned error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to submit feedback"
        )
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to submit feedback: {str(e)}"
        )


# ========== Training & Data Management ==========

@timbral_router.post("/train")
async def trigger_model_training():
    """Trigger ML model retraining (admin endpoint)"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 min timeout for training
            response = await client.post(f"{ML_SERVICE_URL}/api/v1/train")
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"ML service returned error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to trigger model training"
        )
    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to trigger training: {str(e)}"
        )


@timbral_router.post("/sync-tracks")
async def sync_track_metadata(tracks: List[TrackMetadata]):
    """Sync track metadata to ML service"""
    try:
        async with httpx.AsyncClient(timeout=ML_SERVICE_TIMEOUT) as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/api/v1/sync-tracks",
                json=[track.dict() for track in tracks]
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"ML service returned error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to sync track metadata"
        )
    except Exception as e:
        logger.error(f"Error syncing tracks: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to sync tracks: {str(e)}"
        )


# ========== Cache Management ==========

@timbral_router.post("/cache/clear")
async def clear_ml_cache():
    """Clear ML service cache"""
    try:
        async with httpx.AsyncClient(timeout=ML_SERVICE_TIMEOUT) as client:
            response = await client.post(f"{ML_SERVICE_URL}/api/v1/cache/clear")
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"ML service returned error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to clear ML cache"
        )
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to clear cache: {str(e)}"
        )


@timbral_router.get("/cache/stats")
async def get_cache_stats():
    """Get ML service cache statistics"""
    try:
        async with httpx.AsyncClient(timeout=ML_SERVICE_TIMEOUT) as client:
            response = await client.get(f"{ML_SERVICE_URL}/api/v1/cache/stats")
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"ML service returned error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to get cache stats"
        )
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get cache stats: {str(e)}"
        )