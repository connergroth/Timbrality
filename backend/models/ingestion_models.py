from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class EnhancedTrack(BaseModel):
    track_id: str
    track_name: str
    artist_id: str
    artist_name: str
    album_id: Optional[str] = None
    album_name: Optional[str] = None
    duration_ms: Optional[int] = None
    popularity: Optional[int] = None
    explicit: Optional[bool] = None
    audio_features: Optional[Dict[str, Any]] = None
    genres: Optional[List[str]] = None
    release_date: Optional[str] = None
    source: str = "spotify"

class EnhancedAlbum(BaseModel):
    album_id: str
    album_name: str
    artist_id: str
    artist_name: str
    release_date: Optional[str] = None
    total_tracks: Optional[int] = None
    album_type: Optional[str] = None
    genres: Optional[List[str]] = None
    popularity: Optional[int] = None
    source: str = "spotify"

class EnhancedArtist(BaseModel):
    artist_id: str
    artist_name: str
    genres: Optional[List[str]] = None
    popularity: Optional[int] = None
    followers: Optional[int] = None
    source: str = "spotify"

class MLTrainingData(BaseModel):
    user_id: str
    track_id: str
    track_name: str
    artist_name: str
    album_name: Optional[str] = None
    interaction_type: str  # 'saved', 'played', 'skipped', etc.
    interaction_count: int = 1
    audio_features: Optional[Dict[str, Any]] = None
    created_at: datetime = datetime.now()

class IngestionStats(BaseModel):
    total_tracks: int = 0
    total_albums: int = 0
    total_artists: int = 0
    successful_inserts: int = 0
    failed_inserts: int = 0
    processing_time: float = 0.0
    last_updated: datetime = datetime.now()


def get_ml_training_data(user_id: str, limit: int = 1000) -> List[MLTrainingData]:
    """
    Retrieve ML training data for a specific user.
    This is a placeholder implementation that should be replaced with actual database queries.
    """
    return []