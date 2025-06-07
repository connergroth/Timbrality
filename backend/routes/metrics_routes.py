from fastapi import APIRouter, Response
from app.utils.metrics import metrics

router = APIRouter()

@router.get(
    "/",
    summary="API Usage Metrics",
    description="Get current API usage statistics",
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
                        "max_response_time": 1.234567,
                        "last_reset": "2025-03-11T05:17:38.585115",
                        "endpoint_hits": {
                            "album": 3,
                            "similar": 2,
                            "search": 1,
                            "user": 1,
                            "metrics": 0
                        }
                    }
                }
            },
        }
    },
)
async def get_metrics_endpoint():
    """Get current API usage statistics"""
    metrics.record_request(cache_hit=True, endpoint="metrics")
    return metrics.get_metrics()

@router.post(
    "/reset",
    summary="Reset Metrics",
    description="Reset all API usage statistics",
    response_description="Confirmation of metrics reset",
)
async def reset_metrics_endpoint():
    """Reset all metrics counters"""
    metrics.reset()
    return {"message": "Metrics have been reset successfully"}

# Legacy endpoints for backward compatibility
@router.get(
    "/metrics",
    summary="API Usage Metrics (Legacy)",
    description="Legacy endpoint - Get current API usage statistics",
    response_description="Current API usage statistics",
)
async def get_sonance_metrics_legacy():
    """Legacy endpoint - Get application metrics."""
    return await get_metrics_endpoint()

@router.get(
    "/aoty-metrics",
    summary="AOTY API Metrics (Legacy)",
    description="Legacy endpoint - Now returns integrated metrics",
    response_description="Integrated API usage statistics",
)
async def get_aoty_metrics_legacy():
    """Legacy endpoint - Returns integrated metrics instead of external AOTY API metrics."""
    return {
        "message": "AOTY functionality is now integrated. Use /metrics/ for current statistics.",
        "metrics": metrics.get_metrics()
    }

@router.get(
    "/combined-metrics",
    summary="Combined Metrics (Legacy)",
    description="Legacy endpoint - Now returns integrated metrics",
    response_description="Integrated metrics",
)
async def get_combined_metrics_legacy():
    """Legacy endpoint - Returns integrated metrics."""
    return {
        "message": "All metrics are now integrated into a single system.",
        "integrated_metrics": metrics.get_metrics()
    }