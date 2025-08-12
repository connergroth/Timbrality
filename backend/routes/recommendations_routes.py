"""
Recommendations API Routes
Aligns with architecture.md Section 6 API Surface
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.db import db_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recs", tags=["recommendations"])


class Track(BaseModel):
    """Track model for API responses"""
    id: str
    title: str
    artist: str
    album: Optional[str] = None
    spotify_id: Optional[str] = None
    spotify_url: Optional[str] = None
    aoty_score: Optional[float] = None
    popularity: Optional[int] = None
    duration_ms: Optional[int] = None
    genres: List[str] = []
    moods: List[str] = []


class Carousel(BaseModel):
    """Carousel model for home recommendations"""
    title: str
    description: str
    tracks: List[Track]


class HomeRecommendations(BaseModel):
    """Home page recommendations response"""
    carousels: List[Carousel]
    user_id: Optional[str] = None


class SimilarTracksResponse(BaseModel):
    """Similar tracks API response"""
    seed_track: Track
    similar_tracks: List[Track]
    algorithm: str = "content_similarity"


class SearchRequest(BaseModel):
    """Search request model"""
    query: str
    limit: int = 50
    offset: int = 0
    filters: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    total_results: int
    tracks: List[Track]
    algorithm: str = "semantic_search"


class TasteSummary(BaseModel):
    """User taste profile summary"""
    user_id: str
    top_genres: List[Dict[str, Any]]
    mood_distribution: Dict[str, float]
    listening_stats: Dict[str, Any]


@router.get("/home", response_model=HomeRecommendations)
async def get_home_recommendations(
    user_id: Optional[str] = None,
    limit: int = Query(default=5, le=10)
) -> HomeRecommendations:
    """
    Get home page recommendations with blended carousels
    
    Architecture alignment: GET /recs/home
    Returns 5 blended carousels (For You, New & Hype, Genre Mixes…)
    """
    try:
        # TODO: Implement actual recommendation logic
        # For now, return popular tracks from database
        
        # Get some tracks from database
        popular_tracks = await _get_popular_tracks(limit * 5)
        
        # Create mock carousels
        carousels = [
            Carousel(
                title="For You",
                description="Personalized recommendations based on your taste",
                tracks=popular_tracks[:limit]
            ),
            Carousel(
                title="New & Trending",
                description="Fresh tracks gaining popularity",
                tracks=popular_tracks[limit:limit*2]
            ),
            Carousel(
                title="Discover Weekly",
                description="New music tailored to your preferences",
                tracks=popular_tracks[limit*2:limit*3]
            ),
            Carousel(
                title="Genre Mix",
                description="Explore different genres you might like",
                tracks=popular_tracks[limit*3:limit*4]
            ),
            Carousel(
                title="Deep Cuts",
                description="Hidden gems and lesser known tracks",
                tracks=popular_tracks[limit*4:limit*5]
            )
        ]
        
        return HomeRecommendations(
            carousels=carousels,
            user_id=user_id
        )
        
    except Exception as e:
        logger.error(f"Failed to get home recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/track/{track_id}", response_model=SimilarTracksResponse)
async def get_similar_tracks(
    track_id: str,
    limit: int = Query(default=20, le=50)
) -> SimilarTracksResponse:
    """
    Get tracks similar to a given track
    
    Architecture alignment: GET /recs/track/{track_id}
    Tracks similar to X (KNN on track_vector)
    """
    try:
        # Get the seed track
        seed_track = await _get_track_by_id(track_id)
        if not seed_track:
            raise HTTPException(status_code=404, detail="Track not found")
        
        # TODO: Implement actual vector similarity search
        # For now, return tracks with similar genres/moods
        similar_tracks = await _get_similar_tracks_by_content(seed_track, limit)
        
        return SimilarTracksResponse(
            seed_track=seed_track,
            similar_tracks=similar_tracks,
            algorithm="content_similarity"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get similar tracks for {track_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search_tracks(request: SearchRequest) -> SearchResponse:
    """
    Free-text search for tracks
    
    Architecture alignment: POST /search
    Free-text query → semantic search results
    """
    try:
        # TODO: Implement semantic search with embeddings
        # For now, do basic text search
        tracks = await _search_tracks_by_text(
            request.query,
            request.limit,
            request.offset,
            request.filters
        )
        
        return SearchResponse(
            query=request.query,
            total_results=len(tracks),  # TODO: Get actual total count
            tracks=tracks,
            algorithm="semantic_search"
        )
        
    except Exception as e:
        logger.error(f"Failed to search tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/taste/summary/{user_id}", response_model=TasteSummary)
async def get_taste_summary(user_id: str) -> TasteSummary:
    """
    Get user taste profile summary
    
    Architecture alignment: GET /taste/summary
    User taste profile JSON (top genres, mood distribution)
    """
    try:
        # TODO: Implement actual taste profiling
        # For now, return mock data
        
        return TasteSummary(
            user_id=user_id,
            top_genres=[
                {"genre": "indie rock", "weight": 0.4},
                {"genre": "electronic", "weight": 0.3},
                {"genre": "jazz", "weight": 0.2},
                {"genre": "ambient", "weight": 0.1}
            ],
            mood_distribution={
                "energetic": 0.3,
                "chill": 0.4,
                "melancholy": 0.2,
                "upbeat": 0.1
            },
            listening_stats={
                "total_tracks": 1234,
                "avg_energy": 0.65,
                "avg_valence": 0.55,
                "dominant_genres": ["indie rock", "electronic"]
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get taste summary for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions

async def _get_popular_tracks(limit: int) -> List[Track]:
    """Get popular tracks from database"""
    try:
        # Query tracks ordered by popularity
        result = db_client.client.from_("tracks").select("*").order("popularity", desc=True).limit(limit).execute()
        
        tracks = []
        for row in result.data:
            tracks.append(Track(
                id=row["id"],
                title=row["title"],
                artist=row["artist"],
                album=row.get("album"),
                spotify_id=row.get("spotify_id"),
                spotify_url=row.get("spotify_url"),
                aoty_score=row.get("aoty_score"),
                popularity=row.get("popularity"),
                duration_ms=row.get("duration_ms"),
                genres=row.get("genres", []),
                moods=row.get("moods", [])
            ))
        
        return tracks
        
    except Exception as e:
        logger.error(f"Failed to get popular tracks: {e}")
        return []


async def _get_track_by_id(track_id: str) -> Optional[Track]:
    """Get a single track by ID"""
    try:
        result = db_client.client.from_("tracks").select("*").eq("id", track_id).limit(1).execute()
        
        if not result.data:
            return None
        
        row = result.data[0]
        return Track(
            id=row["id"],
            title=row["title"],
            artist=row["artist"],
            album=row.get("album"),
            spotify_id=row.get("spotify_id"),
            spotify_url=row.get("spotify_url"),
            aoty_score=row.get("aoty_score"),
            popularity=row.get("popularity"),
            duration_ms=row.get("duration_ms"),
            genres=row.get("genres", []),
            moods=row.get("moods", [])
        )
        
    except Exception as e:
        logger.error(f"Failed to get track {track_id}: {e}")
        return None


async def _get_similar_tracks_by_content(seed_track: Track, limit: int) -> List[Track]:
    """Get similar tracks based on content (genres, moods, etc.)"""
    # TODO: Implement proper content-based similarity
    # For now, return tracks with overlapping genres
    try:
        if not seed_track.genres:
            return await _get_popular_tracks(limit)  # Fallback
        
        # Search for tracks with similar genres
        genre_conditions = " OR ".join([f"genres.cs.{{{genre}}}" for genre in seed_track.genres[:2]])
        
        result = db_client.client.from_("tracks").select("*").or_(genre_conditions).neq("id", seed_track.id).limit(limit).execute()
        
        tracks = []
        for row in result.data:
            tracks.append(Track(
                id=row["id"],
                title=row["title"],
                artist=row["artist"],
                album=row.get("album"),
                spotify_id=row.get("spotify_id"),
                spotify_url=row.get("spotify_url"),
                aoty_score=row.get("aoty_score"),
                popularity=row.get("popularity"),
                duration_ms=row.get("duration_ms"),
                genres=row.get("genres", []),
                moods=row.get("moods", [])
            ))
        
        return tracks
        
    except Exception as e:
        logger.error(f"Failed to get similar tracks: {e}")
        return []


async def _search_tracks_by_text(
    query: str,
    limit: int,
    offset: int,
    filters: Optional[Dict[str, Any]]
) -> List[Track]:
    """Search tracks by text query"""
    try:
        # Basic text search on title and artist
        search_query = f"%{query}%"
        
        supabase_query = db_client.client.from_("tracks").select("*")
        
        # Add text search conditions
        supabase_query = supabase_query.or_(
            f"title.ilike.{search_query},artist.ilike.{search_query}"
        )
        
        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                if key == "genre" and value:
                    supabase_query = supabase_query.contains("genres", [value])
                elif key == "mood" and value:
                    supabase_query = supabase_query.contains("moods", [value])
        
        # Apply pagination
        result = supabase_query.range(offset, offset + limit - 1).execute()
        
        tracks = []
        for row in result.data:
            tracks.append(Track(
                id=row["id"],
                title=row["title"],
                artist=row["artist"],
                album=row.get("album"),
                spotify_id=row.get("spotify_id"),
                spotify_url=row.get("spotify_url"),
                aoty_score=row.get("aoty_score"),
                popularity=row.get("popularity"),
                duration_ms=row.get("duration_ms"),
                genres=row.get("genres", []),
                moods=row.get("moods", [])
            ))
        
        return tracks
        
    except Exception as e:
        logger.error(f"Failed to search tracks: {e}")
        return []