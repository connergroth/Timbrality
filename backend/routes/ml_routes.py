"""
ML Routes - API endpoints for machine learning operations

Provides RESTful endpoints for ingestion, data preparation, and ML training data access.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import logging

try:
    from services.ml_service import ml_service
    from models.ingestion_models import IngestionStats, MLTrainingData
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.ml_service import ml_service
    from models.ingestion_models import IngestionStats, MLTrainingData

logger = logging.getLogger(__name__)

# Create router
ml_router = APIRouter(prefix="/ml", tags=["Machine Learning"])


# Pydantic models for requests/responses
class AlbumIngestionRequest(BaseModel):
    album_name: str = Field(..., description="Name of the album")
    artist_name: str = Field(..., description="Name of the artist")


class BatchIngestionRequest(BaseModel):
    albums: List[Tuple[str, str]] = Field(..., description="List of (album_name, artist_name) tuples")


class IngestionResponse(BaseModel):
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class TrainingDataResponse(BaseModel):
    total_tracks: int
    training_data: List[MLTrainingData]
    feature_summary: Dict[str, Any]


class AnalyticsResponse(BaseModel):
    genre_distribution: Dict[str, int]
    mood_distribution: Dict[str, int]
    popularity_insights: Dict[str, Any]
    ingestion_stats: IngestionStats


# ========== Ingestion Endpoints ==========

@ml_router.post("/ingest/album")
async def ingest_single_album(request: AlbumIngestionRequest):
    """Ingest a single album into the database"""
    try:
        success = await ml_service.ingest_album_async(request.album_name, request.artist_name)
        
        if success:
            return IngestionResponse(
                success=True,
                message=f"Successfully ingested album '{request.album_name}' by {request.artist_name}"
            )
        else:
            return IngestionResponse(
                success=False,
                message=f"Failed to ingest album '{request.album_name}' by {request.artist_name}"
            )
    
    except Exception as e:
        logger.error(f"Error in album ingestion endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.post("/ingest/batch")
def ingest_batch_albums(request: BatchIngestionRequest, background_tasks: BackgroundTasks):
    """Ingest multiple albums in batch (runs in background)"""
    try:
        # Run batch ingestion in background
        background_tasks.add_task(ml_service.ingest_albums_batch, request.albums)
        
        return IngestionResponse(
            success=True,
            message=f"Started batch ingestion of {len(request.albums)} albums",
            details={"albums_count": len(request.albums)}
        )
    
    except Exception as e:
        logger.error(f"Error in batch ingestion endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Data Access Endpoints ==========

@ml_router.get("/stats")
def get_ingestion_statistics():
    """Get comprehensive ingestion statistics"""
    try:
        return ml_service.get_ingestion_stats()
    except Exception as e:
        logger.error(f"Error getting ingestion stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/training-data")
def get_training_data(
    limit: int = Query(default=1000, description="Maximum number of tracks to return"),
    include_audio_features: bool = Query(default=True, description="Include tracks with audio features")
):
    """Get ML training data with feature summary"""
    try:
        # Get training data
        training_data = ml_service.get_ml_training_data(limit, include_audio_features)
        
        if not training_data:
            raise HTTPException(status_code=404, detail="No training data available")
        
        # Create feature summary
        feature_summary = {
            "total_tracks": len(training_data),
            "tracks_with_genres": sum(1 for t in training_data if t.genres),
            "tracks_with_moods": sum(1 for t in training_data if t.moods),
            "tracks_with_audio_features": sum(1 for t in training_data if t.audio_features),
            "tracks_with_aoty_scores": sum(1 for t in training_data if t.aoty_score is not None),
            "average_genres_per_track": sum(len(t.genres or []) for t in training_data) / len(training_data),
            "average_moods_per_track": sum(len(t.moods or []) for t in training_data) / len(training_data)
        }
        
        return TrainingDataResponse(
            total_tracks=len(training_data),
            training_data=training_data,
            feature_summary=feature_summary
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/feature-matrix")
def get_feature_matrix(
    limit: int = Query(default=1000, description="Maximum number of tracks to process")
):
    """Get prepared feature matrix for ML training"""
    try:
        training_data = ml_service.get_ml_training_data(limit)
        
        if not training_data:
            raise HTTPException(status_code=404, detail="No training data available")
        
        X, y = ml_service.prepare_feature_matrix(training_data)
        
        if X.empty:
            raise HTTPException(status_code=500, detail="Failed to prepare feature matrix")
        
        return {
            "shape": {"samples": X.shape[0], "features": X.shape[1]},
            "features": list(X.columns),
            "target_stats": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max())
            },
            "sample_data": X.head(5).to_dict('records')  # First 5 rows as example
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Analytics Endpoints ==========

@ml_router.get("/analytics")
def get_ml_analytics():
    """Get comprehensive ML analytics and insights"""
    try:
        return AnalyticsResponse(
            genre_distribution=ml_service.get_genre_distribution(),
            mood_distribution=ml_service.get_mood_distribution(),
            popularity_insights=ml_service.get_popularity_insights(),
            ingestion_stats=ml_service.get_ingestion_stats()
        )
    
    except Exception as e:
        logger.error(f"Error getting ML analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/genres")
def get_genre_distribution(limit: int = Query(default=20, description="Number of top genres to return")):
    """Get distribution of genres across tracks"""
    try:
        return ml_service.get_genre_distribution(limit)
    except Exception as e:
        logger.error(f"Error getting genre distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/moods")
def get_mood_distribution(limit: int = Query(default=20, description="Number of top moods to return")):
    """Get distribution of moods across tracks"""
    try:
        return ml_service.get_mood_distribution(limit)
    except Exception as e:
        logger.error(f"Error getting mood distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Export Endpoints ==========

@ml_router.post("/export/training-data")
def export_training_data(
    filename: str = Query(..., description="Output filename for the CSV export"),
    limit: int = Query(default=10000, description="Maximum number of tracks to export")
):
    """Export training data to CSV file"""
    try:
        success = ml_service.export_training_data(filename, limit)
        
        if success:
            return {
                "success": True,
                "message": f"Training data exported to {filename}",
                "filename": filename
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to export training data")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.post("/export/feature-matrix")
def export_feature_matrix(
    filename: str = Query(..., description="Output filename for the CSV export"),
    limit: int = Query(default=10000, description="Maximum number of tracks to process")
):
    """Export prepared feature matrix to CSV file"""
    try:
        success = ml_service.export_feature_matrix(filename, limit)
        
        if success:
            return {
                "success": True,
                "message": f"Feature matrix exported to {filename}",
                "filename": filename
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to export feature matrix")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting feature matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Recommendation Endpoints ==========

@ml_router.get("/recommendations/{track_id}")
def get_similar_tracks(
    track_id: str,
    limit: int = Query(default=10, description="Number of similar tracks to return")
):
    """Get tracks similar to the specified track (basic implementation)"""
    try:
        similar_tracks = ml_service.find_similar_tracks_by_features(track_id, limit)
        
        if not similar_tracks:
            raise HTTPException(status_code=404, detail="No similar tracks found or track not found")
        
        return {
            "track_id": track_id,
            "similar_tracks": [
                {
                    "id": track.id,
                    "title": track.title,
                    "artist": track.artist,
                    "album": track.album,
                    "genres": track.genres,
                    "popularity": track.popularity,
                    "aoty_score": track.aoty_score
                }
                for track in similar_tracks
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similar tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/embeddings/{track_id}")
def get_track_embedding_data(track_id: str):
    """Get track data formatted for embedding generation"""
    try:
        embedding_data = ml_service.get_track_embeddings_data([track_id])
        
        if not embedding_data:
            raise HTTPException(status_code=404, detail="Track not found")
        
        return embedding_data[0]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting track embedding data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@ml_router.get("/health")
def ml_health_check():
    """Health check for ML service"""
    try:
        stats = ml_service.get_ingestion_stats()
        return {
            "status": "healthy",
            "ml_service": "operational",
            "total_tracks": stats.total_tracks,
            "total_albums": stats.total_albums,
            "total_artists": stats.total_artists
        }
    except Exception as e:
        logger.error(f"ML service health check failed: {e}")
        return {
            "status": "unhealthy",
            "ml_service": "error",
            "error": str(e)
        } 