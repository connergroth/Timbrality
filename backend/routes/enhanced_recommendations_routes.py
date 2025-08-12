"""
API routes for enhanced collaborative filtering recommendations
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from slowapi import Limiter
from slowapi.util import get_remote_address
from datetime import datetime
import logging

from services.enhanced_collaborative_filtering import EnhancedCollaborativeFilteringService
from scripts.library_ingestion_script import LibraryIngestionScript

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations/enhanced", tags=["enhanced-recommendations"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/generate")
@limiter.limit("10/minute")  # Conservative rate limit for expensive operation
async def generate_enhanced_recommendations(
    request,
    user_id: str = Query(..., description="User ID to generate recommendations for"),
    num_recommendations: int = Query(20, ge=1, le=100, description="Number of recommendations"),
    include_explanations: bool = Query(True, description="Include recommendation explanations"),
    min_neighbors: int = Query(25, ge=5, le=50, description="Minimum number of neighbors"),
    max_neighbors: int = Query(125, ge=25, le=200, description="Maximum number of neighbors"),
    diversity_lambda: float = Query(0.7, ge=0.0, le=1.0, description="Diversity weight (0=relevance only, 1=diversity only)")
):
    """
    Generate enhanced collaborative filtering recommendations with 2-hop neighbors
    
    This endpoint uses the advanced collaborative filtering system with:
    - 2-hop neighbor discovery  
    - Adaptive k selection based on similarity threshold
    - Popularity de-biasing
    - MMR diversity re-ranking
    - Hard caps and gentle boosts
    - Content-based blending
    """
    try:
        # Initialize service with custom parameters
        enhanced_service = EnhancedCollaborativeFilteringService(
            min_neighbors=min_neighbors,
            max_neighbors=max_neighbors,
            diversity_lambda=diversity_lambda
        )
        
        # Generate recommendations
        recommendations = await enhanced_service.generate_recommendations(
            target_user_id=user_id,
            num_recommendations=num_recommendations,
            include_explanations=include_explanations
        )
        
        if "error" in recommendations:
            raise HTTPException(status_code=500, detail=recommendations["error"])
        
        return {
            "user_id": user_id,
            "recommendations": recommendations["recommendations"],
            "explanations": recommendations["explanations"],
            "metadata": recommendations["metadata"],
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating enhanced recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


@router.get("/user/{user_id}/quality-metrics")
@limiter.limit("20/minute")
async def get_recommendation_quality_metrics(
    request,
    user_id: str,
    num_recommendations: int = Query(20, ge=1, le=100)
):
    """
    Get quality metrics for a user's recommendations without full generation
    
    Returns metrics like:
    - Intra-List Diversity (ILD)
    - Artist/genre entropy  
    - Long-tail fraction
    - Repeat artist rate
    """
    try:
        enhanced_service = EnhancedCollaborativeFilteringService()
        
        # Generate a small set for quality analysis
        recommendations = await enhanced_service.generate_recommendations(
            target_user_id=user_id,
            num_recommendations=num_recommendations,
            include_explanations=False
        )
        
        if "error" in recommendations:
            raise HTTPException(status_code=500, detail=recommendations["error"])
        
        quality_metrics = recommendations.get("metadata", {}).get("quality_metrics", {})
        
        return {
            "user_id": user_id,
            "num_recommendations": len(recommendations.get("recommendations", [])),
            "quality_metrics": quality_metrics,
            "calculated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating quality metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/neighbors")
@limiter.limit("30/minute")
async def get_user_neighbors(
    request,
    user_id: str,
    include_two_hop: bool = Query(True, description="Include 2-hop neighbors"),
    max_neighbors: int = Query(125, ge=10, le=200)
):
    """
    Get the neighbors that would be used for a user's recommendations
    
    Useful for debugging and understanding the collaborative filtering process
    """
    try:
        enhanced_service = EnhancedCollaborativeFilteringService(max_neighbors=max_neighbors)
        
        if include_two_hop:
            neighbors = await enhanced_service._find_two_hop_neighbors(user_id)
        else:
            neighbors = await enhanced_service._find_direct_neighbors(user_id)
        
        # Organize by hop distance
        direct_neighbors = [n for n in neighbors if n.get('hop') == 1]
        two_hop_neighbors = [n for n in neighbors if n.get('hop') == 2]
        
        return {
            "user_id": user_id,
            "total_neighbors": len(neighbors),
            "direct_neighbors": {
                "count": len(direct_neighbors),
                "neighbors": direct_neighbors
            },
            "two_hop_neighbors": {
                "count": len(two_hop_neighbors), 
                "neighbors": two_hop_neighbors
            },
            "similarity_stats": {
                "min_similarity": min([n['similarity'] for n in neighbors]) if neighbors else 0,
                "max_similarity": max([n['similarity'] for n in neighbors]) if neighbors else 0,
                "avg_similarity": sum([n['similarity'] for n in neighbors]) / len(neighbors) if neighbors else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting user neighbors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/personal-library")
@limiter.limit("1/hour")  # Very conservative - this is expensive
async def ingest_personal_library(
    request,
    background_tasks: BackgroundTasks,
    lastfm_username: Optional[str] = Query(None, description="Last.fm username"),
    spotify_username: Optional[str] = Query(None, description="Spotify username"),
    max_neighbors: int = Query(50, ge=10, le=100, description="Max neighbors to ingest"),
    similarity_threshold: float = Query(0.1, ge=0.05, le=0.5, description="Min similarity for neighbors"),
    expand_neighbors: bool = Query(True, description="Find and ingest neighbors")
):
    """
    Ingest personal library and expand training data with similar users
    
    This is a background operation that can take a long time.
    Use this to bootstrap your collaborative filtering system.
    """
    try:
        if not lastfm_username and not spotify_username:
            raise HTTPException(
                status_code=400, 
                detail="Must provide at least one of lastfm_username or spotify_username"
            )
        
        # Run ingestion in background
        ingestion_script = LibraryIngestionScript()
        
        background_tasks.add_task(
            ingestion_script.ingest_personal_library,
            lastfm_username=lastfm_username,
            spotify_username=spotify_username,
            expand_neighbors=expand_neighbors,
            max_neighbors=max_neighbors,
            similarity_threshold=similarity_threshold
        )
        
        return {
            "message": "Personal library ingestion started in background",
            "parameters": {
                "lastfm_username": lastfm_username,
                "spotify_username": spotify_username,
                "max_neighbors": max_neighbors,
                "similarity_threshold": similarity_threshold,
                "expand_neighbors": expand_neighbors
            },
            "started_at": datetime.now().isoformat(),
            "estimated_completion": "5-30 minutes depending on data size"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting library ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/algorithm/parameters")
async def get_algorithm_parameters():
    """Get current algorithm parameters and their explanations"""
    return {
        "parameters": {
            "min_neighbors": {
                "default": 25,
                "range": "5-50",
                "description": "Minimum number of neighbors to use, ensures recommendations aren't too narrow"
            },
            "max_neighbors": {
                "default": 125, 
                "range": "25-200",
                "description": "Maximum number of neighbors, caps computational cost"
            },
            "similarity_threshold": {
                "default": 0.05,
                "range": "0.01-0.3", 
                "description": "Minimum similarity for 2-hop neighbors (sim_2hop = sim(u,n1) * sim(n1,n2))"
            },
            "cumulative_similarity_target": {
                "default": 0.85,
                "range": "0.5-0.95",
                "description": "Target cumulative similarity for adaptive k selection"  
            },
            "diversity_lambda": {
                "default": 0.7,
                "range": "0.0-1.0",
                "description": "MMR diversity weight: 0=relevance only, 1=diversity only"
            },
            "content_blend_alpha": {
                "default": 0.6,
                "range": "0.0-1.0", 
                "description": "CF vs content blend: 1=CF only, 0=content only"
            }
        },
        "algorithm_steps": [
            "1. Find 2-hop neighbors with similarity decay",
            "2. Apply adaptive k selection (cumulative similarity ≥ 0.85)",
            "3. Generate candidates with weighted scoring: Σ sim(u,n) * log(1 + plays)",  
            "4. Apply popularity de-biasing: divide by log(1 + global_pop)",
            "5. MMR re-ranking for diversity on top ~500 candidates",
            "6. Apply hard caps (≤3 tracks/artist) and gentle boosts",
            "7. Blend with content-based to prevent echo chambers"
        ],
        "quality_metrics": [
            "Intra-List Diversity (ILD): Mean 1 - cosine similarity between items",
            "Artist entropy: Diversity of artists in recommendations", 
            "Long-tail fraction: Share of below-median popularity items",
            "Repeat artist rate: Fraction of recommendations with duplicate artists",
            "Catalog coverage: Unique items recommended across users"
        ]
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        enhanced_service = EnhancedCollaborativeFilteringService()
        
        # Simple health check - just initialize service
        return {
            "status": "healthy",
            "service": "Enhanced Collaborative Filtering",
            "version": "1.0.0",
            "features": [
                "2-hop neighbor discovery",
                "Adaptive k selection", 
                "Popularity de-biasing",
                "MMR diversity re-ranking",
                "Hard caps and gentle boosts",
                "Content-based blending",
                "Quality metrics calculation"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")