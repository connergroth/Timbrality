from fastapi import APIRouter
from app.utils.metrics import metrics

@app.get(
    "/metrics",
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
                        "last_reset": "2024-12-30T05:17:38.585115",
                    }
                }
            },
        }
    },
)
async def get_metrics():
    return metrics.get_metrics()

