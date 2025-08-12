"""
API routes for collaborative filtering management
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from models.database import get_db
from services.collaborative_filtering_service import CollaborativeFilteringService

router = APIRouter(prefix="/collaborative-filtering", tags=["collaborative-filtering"])
service = CollaborativeFilteringService()


@router.post("/users")
async def add_lastfm_user(username: str, display_name: str = None):
    """Add a new Last.fm user for collaborative filtering"""
    try:
        result = await service.add_lastfm_user(username, display_name)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users")
async def get_active_users():
    """Get list of active Last.fm users being tracked"""
    try:
        users = await service.get_active_users()
        return {"users": users, "count": len(users)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{username}/fetch")
async def fetch_user_data(username: str, fetch_types: List[str] = None):
    """Fetch data for a specific Last.fm user"""
    try:
        if fetch_types is None:
            fetch_types = ['profile', 'tracks', 'albums', 'artists']
        
        result = await service.fetch_user_data(username, fetch_types)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/bulk-fetch")
async def fetch_multiple_users_data(usernames: List[str]):
    """Fetch data for multiple Last.fm users concurrently"""
    try:
        result = await service.fetch_multiple_users_data(usernames)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_collaborative_filtering_stats(db: Session = Depends(get_db)):
    """Get statistics about collaborative filtering data"""
    try:
        from models.collaborative_filtering import (
            LastfmUser, UserTrackInteraction, UserAlbumInteraction, 
            UserArtistInteraction, UserSimilarity, CollaborativeRecommendation
        )
        
        # Count users
        total_users = db.query(LastfmUser).count()
        active_users = db.query(LastfmUser).filter(LastfmUser.is_active == True).count()
        
        # Count interactions
        total_track_interactions = db.query(UserTrackInteraction).count()
        total_album_interactions = db.query(UserAlbumInteraction).count()
        total_artist_interactions = db.query(UserArtistInteraction).count()
        
        # Count similarities and recommendations
        total_similarities = db.query(UserSimilarity).count()
        total_recommendations = db.query(CollaborativeRecommendation).count()
        
        return {
            "users": {
                "total": total_users,
                "active": active_users
            },
            "interactions": {
                "tracks": total_track_interactions,
                "albums": total_album_interactions,
                "artists": total_artist_interactions
            },
            "ml_data": {
                "user_similarities": total_similarities,
                "recommendations": total_recommendations
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/users/{username}")
async def remove_lastfm_user(username: str, db: Session = Depends(get_db)):
    """Remove a Last.fm user and all their data"""
    try:
        from models.collaborative_filtering import LastfmUser
        
        user = db.query(LastfmUser).filter(
            LastfmUser.lastfm_username == username
        ).first()
        
        if not user:
            raise HTTPException(status_code=404, detail=f"User {username} not found")
        
        # Delete user (cascading will remove all related data)
        db.delete(user)
        db.commit()
        
        return {"message": f"Successfully removed user {username} and all associated data"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))




