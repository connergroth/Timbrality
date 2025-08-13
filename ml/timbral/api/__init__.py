"""
FastAPI route handlers for Timbral music recommendation system.

This package contains all API endpoints for the recommendation service,
including recommendation generation, model management, and health checks.
"""

from .routes import router
from .models import RecommendationRequest, RecommendationResponse

__all__ = ["router", "RecommendationRequest", "RecommendationResponse"] 