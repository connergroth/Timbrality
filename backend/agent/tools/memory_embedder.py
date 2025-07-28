from typing import Dict, Any, List
from .base import BaseTool, ToolResult
from ..embedding.engine import EmbeddingEngine
from ..memory.store import MemoryStore


class MemoryEmbedderTool(BaseTool):
    """Tool for updating user memory and preference embeddings."""
    
    def __init__(self):
        super().__init__(
            name="memory_embedder",
            description="Update user profile based on interactions and feedback"
        )
        self.embedding_engine = EmbeddingEngine()
        self.memory_store = MemoryStore()
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Update user memory with interaction data."""
        try:
            entities = intent_result.entities
            interaction_type = entities.get("interaction_type", "query")
            tracks = entities.get("tracks", [])
            feedback = entities.get("feedback", {})
            user_input = entities.get("user_input", "")
            
            # Update user preference embeddings
            if tracks:
                await self._update_preference_embeddings(
                    context.user_id, tracks, user_input, feedback
                )
            
            # Store interaction for future reference
            await self._store_interaction(
                context.user_id, interaction_type, user_input, tracks, feedback
            )
            
            # Update user tags and characteristics
            await self._update_user_characteristics(
                context.user_id, tracks, user_input, feedback
            )
            
            return self._create_success_result({
                "updated_tracks": len(tracks),
                "interaction_type": interaction_type,
                "memory_updated": True,
                "explanation": f"Updated user profile based on interaction with {len(tracks)} tracks"
            }, confidence=0.9)
            
        except Exception as e:
            return self._create_error_result(f"Memory update failed: {str(e)}")
    
    async def _update_preference_embeddings(
        self,
        user_id: str,
        tracks: List[Dict[str, Any]],
        user_input: str,
        feedback: Dict[str, Any]
    ):
        """Update user preference embeddings based on interaction."""
        # Generate embeddings for the tracks
        track_embeddings = []
        for track in tracks:
            embedding = await self.embedding_engine.embed_track_metadata(track)
            track_embeddings.append(embedding.vector)
        
        if not track_embeddings:
            return
        
        # Weight embeddings based on feedback
        weights = self._calculate_feedback_weights(tracks, feedback)
        
        # Update user profile embedding
        await self.embedding_engine.update_user_profile(
            user_id, tracks, user_input
        )
        
        # Store weighted preference vectors
        await self._store_preference_vectors(user_id, track_embeddings, weights)
    
    async def _store_interaction(
        self,
        user_id: str,
        interaction_type: str,
        user_input: str,
        tracks: List[Dict[str, Any]],
        feedback: Dict[str, Any]
    ):
        """Store interaction data in memory store."""
        interaction_data = {
            "type": interaction_type,
            "input": user_input,
            "tracks": [
                {
                    "id": track.get("id"),
                    "name": track.get("name"),
                    "artist": track.get("artist"),
                    "feedback": feedback.get(track.get("id", ""), {})
                }
                for track in tracks
            ],
            "feedback_summary": self._summarize_feedback(feedback),
            "timestamp": context.timestamp.isoformat() if hasattr(context, 'timestamp') else None
        }
        
        await self.memory_store.add_interaction(user_id, interaction_data)
    
    async def _update_user_characteristics(
        self,
        user_id: str,
        tracks: List[Dict[str, Any]],
        user_input: str,
        feedback: Dict[str, Any]
    ):
        """Update user characteristics based on interaction."""
        # Extract characteristics from tracks
        genres = set()
        artists = set()
        audio_features = []
        
        for track in tracks:
            if track.get("genres"):
                genres.update(track["genres"])
            if track.get("artist"):
                artists.add(track["artist"])
            if track.get("audio_features"):
                audio_features.append(track["audio_features"])
        
        # Calculate preference scores for genres and features
        genre_scores = self._calculate_genre_scores(genres, feedback, tracks)
        feature_preferences = self._calculate_feature_preferences(audio_features, feedback, tracks)
        
        # Update user profile
        await self._update_user_profile_characteristics(
            user_id, genre_scores, feature_preferences, list(artists)
        )
    
    def _calculate_feedback_weights(
        self, 
        tracks: List[Dict[str, Any]], 
        feedback: Dict[str, Any]
    ) -> List[float]:
        """Calculate weights for track embeddings based on feedback."""
        weights = []
        
        for track in tracks:
            track_id = track.get("id", "")
            track_feedback = feedback.get(track_id, {})
            
            # Base weight
            weight = 1.0
            
            # Adjust based on explicit feedback
            if track_feedback.get("liked"):
                weight = 2.0
            elif track_feedback.get("disliked"):
                weight = 0.1
            elif track_feedback.get("skipped"):
                weight = 0.3
            elif track_feedback.get("played_full"):
                weight = 1.5
            elif track_feedback.get("repeated"):
                weight = 2.5
            
            # Adjust based on rating
            rating = track_feedback.get("rating")
            if rating is not None:
                weight = max(rating / 5.0, 0.1)  # Convert 1-5 rating to 0.2-1.0 weight
            
            weights.append(weight)
        
        return weights
    
    def _calculate_genre_scores(
        self,
        genres: set,
        feedback: Dict[str, Any],
        tracks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate preference scores for genres."""
        genre_scores = {}
        
        for genre in genres:
            positive_interactions = 0
            negative_interactions = 0
            total_interactions = 0
            
            for track in tracks:
                if genre in track.get("genres", []):
                    track_feedback = feedback.get(track.get("id", ""), {})
                    total_interactions += 1
                    
                    if track_feedback.get("liked") or track_feedback.get("rating", 0) >= 4:
                        positive_interactions += 1
                    elif track_feedback.get("disliked") or track_feedback.get("rating", 0) <= 2:
                        negative_interactions += 1
            
            if total_interactions > 0:
                score = (positive_interactions - negative_interactions) / total_interactions
                genre_scores[genre] = max(min(score, 1.0), -1.0)  # Clamp to [-1, 1]
        
        return genre_scores
    
    def _calculate_feature_preferences(
        self,
        audio_features_list: List[Dict[str, Any]],
        feedback: Dict[str, Any],
        tracks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate preferences for audio features."""
        if not audio_features_list:
            return {}
        
        feature_names = ["danceability", "energy", "valence", "acousticness", "instrumentalness"]
        feature_preferences = {}
        
        for feature in feature_names:
            positive_values = []
            negative_values = []
            
            for i, track in enumerate(tracks):
                if i < len(audio_features_list):
                    features = audio_features_list[i]
                    track_feedback = feedback.get(track.get("id", ""), {})
                    feature_value = features.get(feature, 0.5)
                    
                    if track_feedback.get("liked") or track_feedback.get("rating", 0) >= 4:
                        positive_values.append(feature_value)
                    elif track_feedback.get("disliked") or track_feedback.get("rating", 0) <= 2:
                        negative_values.append(feature_value)
            
            # Calculate preferred range for this feature
            if positive_values:
                avg_positive = sum(positive_values) / len(positive_values)
                feature_preferences[feature] = avg_positive
            elif negative_values:
                avg_negative = sum(negative_values) / len(negative_values)
                # Invert negative preference
                feature_preferences[feature] = 1.0 - avg_negative
        
        return feature_preferences
    
    async def _store_preference_vectors(
        self,
        user_id: str,
        embeddings: List,
        weights: List[float]
    ):
        """Store weighted preference vectors for the user."""
        import numpy as np
        
        if len(embeddings) != len(weights):
            return
        
        # Calculate weighted average embedding
        weighted_embeddings = [emb * weight for emb, weight in zip(embeddings, weights)]
        if weighted_embeddings:
            avg_embedding = np.mean(weighted_embeddings, axis=0)
            
            # Store in memory store
            await self.memory_store.update_user_embedding(user_id, avg_embedding.tolist())
    
    async def _update_user_profile_characteristics(
        self,
        user_id: str,
        genre_scores: Dict[str, float],
        feature_preferences: Dict[str, float],
        artists: List[str]
    ):
        """Update user profile with calculated characteristics."""
        profile_update = {
            "genre_preferences": genre_scores,
            "audio_feature_preferences": feature_preferences,
            "recent_artists": artists[-20:],  # Keep last 20 artists
            "last_updated": context.timestamp.isoformat() if hasattr(context, 'timestamp') else None
        }
        
        await self.memory_store.update_user_profile(user_id, profile_update)
    
    def _summarize_feedback(self, feedback: Dict[str, Any]) -> Dict[str, int]:
        """Summarize feedback for storage."""
        summary = {
            "total_tracks": len(feedback),
            "liked": 0,
            "disliked": 0,
            "skipped": 0,
            "played_full": 0,
            "repeated": 0
        }
        
        for track_feedback in feedback.values():
            if track_feedback.get("liked"):
                summary["liked"] += 1
            if track_feedback.get("disliked"):
                summary["disliked"] += 1
            if track_feedback.get("skipped"):
                summary["skipped"] += 1
            if track_feedback.get("played_full"):
                summary["played_full"] += 1
            if track_feedback.get("repeated"):
                summary["repeated"] += 1
        
        return summary