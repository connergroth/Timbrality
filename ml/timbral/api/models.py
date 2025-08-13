"""
Pydantic models for API request and response schemas.

This module defines the data models used for API communication,
including recommendation requests, responses, and validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class RecommendationRequest(BaseModel):
    """
    Request model for recommendation generation.
    """
    user_id: int = Field(..., description="User ID to generate recommendations for")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations to return")
    include_metadata: bool = Field(default=True, description="Include item metadata in response")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters for recommendations")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 12345,
                "top_k": 10,
                "include_metadata": True,
                "filters": {"genre": "rock", "min_rating": 4.0}
            }
        }


class RecommendationItem(BaseModel):
    """
    Model for a single recommended item.
    """
    item_id: int = Field(..., description="Item ID")
    score: float = Field(..., description="Recommendation score")
    title: Optional[str] = Field(None, description="Item title")
    artist: Optional[str] = Field(None, description="Artist name")
    genre: Optional[str] = Field(None, description="Music genre")
    explanation: Optional[str] = Field(None, description="Explanation for recommendation")
    
    class Config:
        schema_extra = {
            "example": {
                "item_id": 67890,
                "score": 0.85,
                "title": "Bohemian Rhapsody",
                "artist": "Queen",
                "genre": "Rock",
                "explanation": "Based on your preference for classic rock"
            }
        }


class RecommendationResponse(BaseModel):
    """
    Response model for recommendation requests.
    """
    user_id: int = Field(..., description="User ID")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommended items")
    total_count: int = Field(..., description="Total number of recommendations")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of generation")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 12345,
                "recommendations": [
                    {
                        "item_id": 67890,
                        "score": 0.85,
                        "title": "Bohemian Rhapsody",
                        "artist": "Queen",
                        "genre": "Rock"
                    }
                ],
                "total_count": 1,
                "generated_at": "2024-01-01T12:00:00Z"
            }
        }


class ExplanationResponse(BaseModel):
    """
    Response model for recommendation explanations.
    """
    user_id: int = Field(..., description="User ID")
    item_id: int = Field(..., description="Item ID")
    explanation: str = Field(..., description="Explanation text")
    factors: Dict[str, float] = Field(..., description="Contributing factors and their weights")
    similar_items: List[int] = Field(..., description="List of similar item IDs")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 12345,
                "item_id": 67890,
                "explanation": "You might like this because you enjoy classic rock and have listened to similar artists",
                "factors": {"genre_similarity": 0.8, "artist_similarity": 0.6, "collaborative_score": 0.7},
                "similar_items": [123, 456, 789]
            }
        }


class FeedbackRequest(BaseModel):
    """
    Request model for user feedback submission.
    """
    user_id: int = Field(..., description="User ID")
    item_id: int = Field(..., description="Item ID")
    rating: float = Field(..., ge=1.0, le=5.0, description="User rating (1-5 scale)")
    feedback_type: str = Field(default="rating", description="Type of feedback")
    additional_data: Optional[Dict[str, Any]] = Field(default=None, description="Additional feedback data")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 12345,
                "item_id": 67890,
                "rating": 4.5,
                "feedback_type": "rating",
                "additional_data": {"listened_duration": 180}
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Status of individual components")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "timbral",
                "version": "0.1.0",
                "timestamp": "2024-01-01T12:00:00Z",
                "components": {
                    "redis": "healthy",
                    "models": "healthy",
                    "api": "healthy"
                }
            }
        } 