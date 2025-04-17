from fastapi import APIRouter, Response, Depends
from app.utils.metrics import metrics
from app.services.aoty_service import AOTYService
from typing import Optional
import httpx

router = APIRouter()

@router.get(
    "/metrics",
    summary="API Usage Metrics",
    description="Get current API usage statistics for Sonance",
    response_description="Current API usage statistics",
    responses={
        200: {
            "description": "API metrics including request counts and response times",
            "content": {
                "application/json": {
                    "example": {
                        "total_requests": 7,
                        "cache_hits": 5,
                        "cache_misses": 2,
                        "errors": 0,
                        "avg_response_time": 0.158826896122524,
                        "last_reset": "2025-02-30T05:17:38.585115",
                    }
                }
            },
        }
    },
)
async def get_sonance_metrics():
    """Get Sonance application metrics."""
    return metrics.get_metrics()

@router.get(
    "/aoty-metrics",
    summary="AOTY API Metrics",
    description="Get metrics from the AOTY API",
    response_description="AOTY API usage statistics",
)
async def get_aoty_metrics():
    """Get metrics from the AOTY API service."""
    aoty_service = AOTYService()
    try:
        # Call the AOTY API metrics endpoint
        response = await aoty_service.client.get(f"{aoty_service.base_url}/metrics")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Failed to fetch AOTY API metrics: {str(e)}"}
    finally:
        await aoty_service.close()

@router.get(
    "/combined-metrics",
    summary="Combined Metrics",
    description="Get combined metrics from both Sonance and AOTY API",
    response_description="Combined metrics from both systems",
)
async def get_combined_metrics():
    """Get combined metrics from both Sonance and AOTY API."""
    sonance_metrics = metrics.get_metrics()
    
    # Get AOTY metrics
    aoty_service = AOTYService()
    aoty_metrics = {}
    
    try:
        response = await aoty_service.client.get(f"{aoty_service.base_url}/metrics")
        if response.status_code == 200:
            aoty_metrics = response.json()
    except Exception:
        aoty_metrics = {"error": "Failed to fetch AOTY API metrics"}
    finally:
        await aoty_service.close()
    
    # Combine both metrics
    return {
        "sonance": sonance_metrics,
        "aoty_api": aoty_metrics
    }