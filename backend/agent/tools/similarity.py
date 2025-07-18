from typing import Dict, Any, List
import numpy as np
from .base import BaseTool, ToolResult
from ..embedding.engine import EmbeddingEngine
from services.ml_service import MLService
from utils.database_utils import get_database_connection


class SimilarityTool(BaseTool):
    """Tool for finding similar tracks based on embeddings and ML models."""
    
    def __init__(self):
        super().__init__(
            name="similarity",
            description="Find tracks similar to a given seed track, album, or playlist"
        )
        self.embedding_engine = EmbeddingEngine()
        self.ml_service = MLService()
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Execute similarity search based on intent results."""
        try:
            entities = intent_result.entities
            
            # Get seed information
            seed_track_id = entities.get("track_id")
            seed_artist = entities.get("artist")
            seed_mood = entities.get("mood")
            similarity_count = entities.get("count", 10)
            
            if seed_track_id:
                tracks = await self._find_similar_by_track_id(seed_track_id, similarity_count)
                explanation = f"Found tracks similar to the specified track"
            elif seed_artist:
                tracks = await self._find_similar_by_artist(seed_artist, similarity_count)
                explanation = f"Found tracks similar to {seed_artist}'s style"
            elif seed_mood:
                tracks = await self._find_similar_by_mood(seed_mood, similarity_count)
                explanation = f"Found tracks matching the '{seed_mood}' vibe"
            else:
                # Use user's listening history as seed
                tracks = await self._find_similar_by_user_profile(context.user_id, similarity_count)
                explanation = "Found tracks based on your listening history"
            
            if not tracks:
                return self._create_error_result("No similar tracks found")
            
            return self._create_success_result({
                "tracks": tracks,
                "explanation": explanation,
                "similarity_method": "hybrid_embedding_ml"
            }, confidence=0.85)
            
        except Exception as e:
            return self._create_error_result(f"Similarity search failed: {str(e)}")
    
    async def _find_similar_by_track_id(self, track_id: str, count: int) -> List[Dict[str, Any]]:
        """Find tracks similar to a specific track ID."""
        # Get track metadata
        track_data = await self._get_track_metadata(track_id)
        if not track_data:
            return []
        
        # Generate embedding for the seed track
        seed_embedding = await self.embedding_engine.embed_track_metadata(track_data)
        
        # Get candidate tracks from database
        candidate_tracks = await self._get_candidate_tracks(limit=1000)
        
        # Generate embeddings for candidates and compute similarities
        similar_tracks = await self.embedding_engine.find_similar_tracks(
            seed_embedding.vector, candidate_tracks, top_k=count * 2
        )
        
        # Apply ML model for reranking
        reranked_tracks = await self.ml_service.rerank_recommendations(
            track_data, similar_tracks
        )
        
        return reranked_tracks[:count]
    
    async def _find_similar_by_artist(self, artist: str, count: int) -> List[Dict[str, Any]]:
        """Find tracks similar to an artist's style."""
        # Get tracks by the artist
        artist_tracks = await self._get_tracks_by_artist(artist, limit=20)
        if not artist_tracks:
            return []
        
        # Create composite embedding from artist's tracks
        track_embeddings = []
        for track in artist_tracks:
            embedding = await self.embedding_engine.embed_track_metadata(track)
            track_embeddings.append(embedding.vector)
        
        # Average embeddings to represent artist style
        artist_style_embedding = np.mean(track_embeddings, axis=0)
        
        # Find similar tracks
        candidate_tracks = await self._get_candidate_tracks(
            exclude_artist=artist, limit=1000
        )
        
        similar_tracks = await self.embedding_engine.find_similar_tracks(
            artist_style_embedding, candidate_tracks, top_k=count
        )
        
        return similar_tracks
    
    async def _find_similar_by_mood(self, mood: str, count: int) -> List[Dict[str, Any]]:
        """Find tracks matching a specific mood or vibe."""
        # Generate embedding for mood
        mood_embedding = await self.embedding_engine.embed_mood_description(mood)
        
        # Get candidate tracks
        candidate_tracks = await self._get_candidate_tracks(limit=1000)
        
        # Find tracks matching the mood
        similar_tracks = await self.embedding_engine.find_similar_tracks(
            mood_embedding.vector, candidate_tracks, top_k=count * 2
        )
        
        # Filter by audio features that match the mood
        filtered_tracks = self._filter_by_mood_features(similar_tracks, mood)
        
        return filtered_tracks[:count]
    
    async def _find_similar_by_user_profile(self, user_id: str, count: int) -> List[Dict[str, Any]]:
        """Find tracks similar to user's listening profile."""
        # Get user's listening history
        user_history = await self._get_user_listening_history(user_id, limit=50)
        if not user_history:
            return []
        
        # Generate embeddings for user's tracks
        user_track_embeddings = []
        for track in user_history:
            embedding = await self.embedding_engine.embed_track_metadata(track)
            user_track_embeddings.append(embedding.vector)
        
        # Create user profile embedding (weighted by recency/frequency)
        weights = self._calculate_track_weights(user_history)
        user_profile_embedding = np.average(user_track_embeddings, weights=weights, axis=0)
        
        # Find similar tracks
        candidate_tracks = await self._get_candidate_tracks(
            exclude_user_history=user_id, limit=1000
        )
        
        similar_tracks = await self.embedding_engine.find_similar_tracks(
            user_profile_embedding, candidate_tracks, top_k=count
        )
        
        return similar_tracks
    
    async def _get_track_metadata(self, track_id: str) -> Dict[str, Any]:
        """Get metadata for a specific track."""
        # This would query the database for track metadata
        # For now, return empty dict
        return {}
    
    async def _get_candidate_tracks(
        self, 
        exclude_artist: str = None, 
        exclude_user_history: str = None, 
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get candidate tracks for similarity comparison."""
        # This would query the database for candidate tracks
        # with appropriate filters and embeddings
        return []
    
    async def _get_tracks_by_artist(self, artist: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get tracks by a specific artist."""
        # This would query the database for tracks by artist
        return []
    
    async def _get_user_listening_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's recent listening history."""
        # This would query the database for user's listening history
        return []
    
    def _calculate_track_weights(self, tracks: List[Dict[str, Any]]) -> List[float]:
        """Calculate weights for tracks based on recency and frequency."""
        weights = []
        for i, track in enumerate(tracks):
            # More recent tracks get higher weight
            recency_weight = 1.0 - (i / len(tracks)) * 0.5
            
            # More frequently played tracks get higher weight
            play_count = track.get("play_count", 1)
            frequency_weight = min(play_count / 10, 2.0)
            
            total_weight = recency_weight * frequency_weight
            weights.append(total_weight)
        
        return weights
    
    def _filter_by_mood_features(
        self, 
        tracks: List[Dict[str, Any]], 
        mood: str
    ) -> List[Dict[str, Any]]:
        """Filter tracks by audio features that match the mood."""
        mood_filters = {
            "energetic": lambda f: f.get("energy", 0) > 0.6 and f.get("danceability", 0) > 0.5,
            "chill": lambda f: f.get("energy", 0) < 0.5 and f.get("valence", 0) > 0.3,
            "sad": lambda f: f.get("valence", 0) < 0.4 and f.get("energy", 0) < 0.6,
            "happy": lambda f: f.get("valence", 0) > 0.6 and f.get("energy", 0) > 0.4,
            "dark": lambda f: f.get("valence", 0) < 0.3 and f.get("mode", 0) == 0,
            "upbeat": lambda f: f.get("tempo", 0) > 120 and f.get("energy", 0) > 0.7
        }
        
        filter_func = None
        for mood_key, func in mood_filters.items():
            if mood_key in mood.lower():
                filter_func = func
                break
        
        if not filter_func:
            return tracks
        
        filtered = []
        for track in tracks:
            audio_features = track.get("audio_features", {})
            if filter_func(audio_features):
                filtered.append(track)
        
        return filtered