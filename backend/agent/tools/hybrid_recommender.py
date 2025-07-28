from typing import Dict, Any, List
import numpy as np
from .base import BaseTool, ToolResult
from ..embedding.engine import EmbeddingEngine
from services.ml_service import MLService


class HybridRecommenderTool(BaseTool):
    """Tool for hybrid ML-driven music recommendations."""
    
    def __init__(self):
        super().__init__(
            name="hybrid_recommender",
            description="Generate personalized recommendations using hybrid ML models"
        )
        self.embedding_engine = EmbeddingEngine()
        self.ml_service = MLService()
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Execute hybrid recommendation based on intent results."""
        try:
            entities = intent_result.entities
            recommendation_type = entities.get("recommendation_type", "general")
            count = entities.get("count", 10)
            seed_data = entities.get("seed_data", {})
            
            # Check if this is a mood-based request
            mood = entities.get("mood")
            if mood:
                recommendation_type = "mood"
            
            # Get user profile and history
            user_profile = await self._get_user_profile(context.user_id)
            
            # Generate recommendations based on type
            if recommendation_type == "similar":
                tracks = await self._recommend_similar(seed_data, user_profile, count)
            elif recommendation_type == "discovery":
                tracks = await self._recommend_discovery(user_profile, count)
            elif recommendation_type == "mood":
                tracks = await self._recommend_by_mood(mood, user_profile, count)
            else:
                tracks = await self._recommend_general(user_profile, count)
            
            if not tracks:
                return self._create_error_result("No recommendations generated")
            
            # Add explanation for recommendations
            explanation = self._generate_recommendation_explanation(
                tracks, recommendation_type, user_profile
            )
            
            return self._create_success_result({
                "tracks": tracks,
                "explanation": explanation,
                "recommendation_type": recommendation_type,
                "personalization_score": self._calculate_personalization_score(user_profile)
            }, confidence=0.85)
            
        except Exception as e:
            return self._create_error_result(f"Hybrid recommendation failed: {str(e)}")
    
    async def _recommend_similar(
        self, 
        seed_data: Dict[str, Any], 
        user_profile: Dict[str, Any], 
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate recommendations similar to seed data."""
        # Get seed embedding
        if "track_id" in seed_data:
            seed_track = await self._get_track_by_id(seed_data["track_id"])
            seed_embedding = await self.embedding_engine.embed_track_metadata(seed_track)
        elif "playlist_data" in seed_data:
            seed_embedding = await self._get_playlist_embedding(seed_data["playlist_data"])
        else:
            return []
        
        # Get candidate tracks
        candidates = await self._get_candidate_tracks(exclude_user_history=True)
        
        # Use embedding similarity for initial filtering
        similar_tracks = await self.embedding_engine.find_similar_tracks(
            seed_embedding.vector, candidates, top_k=count * 3
        )
        
        # Apply collaborative filtering
        cf_scores = await self.ml_service.get_collaborative_scores(
            user_profile.get("user_id"), [t["id"] for t in similar_tracks]
        )
        
        # Combine scores
        hybrid_tracks = self._combine_similarity_scores(similar_tracks, cf_scores)
        
        # Personalize based on user profile
        personalized_tracks = self._apply_user_preferences(hybrid_tracks, user_profile)
        
        return personalized_tracks[:count]
    
    async def _recommend_discovery(
        self, 
        user_profile: Dict[str, Any], 
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate discovery recommendations for exploration."""
        # Get user's preference embedding
        user_embedding = user_profile.get("preference_embedding")
        if not user_embedding:
            return await self._recommend_general(user_profile, count)
        
        user_vector = np.array(user_embedding)
        
        # Get diverse candidate tracks
        candidates = await self._get_diverse_candidates(user_profile)
        
        # Find tracks that are somewhat similar but introduce novelty
        discovery_tracks = []
        
        for candidate in candidates:
            candidate_embedding = await self.embedding_engine.embed_track_metadata(candidate)
            similarity = await self.embedding_engine.compute_similarity(
                user_vector, candidate_embedding.vector
            )
            
            # Sweet spot: similar enough to like, different enough to discover
            if 0.4 <= similarity <= 0.7:
                candidate["discovery_score"] = similarity
                candidate["novelty_score"] = 1.0 - similarity
                discovery_tracks.append(candidate)
        
        # Sort by balanced discovery + novelty score
        discovery_tracks.sort(
            key=lambda x: x.get("discovery_score", 0) * 0.6 + x.get("novelty_score", 0) * 0.4,
            reverse=True
        )
        
        return discovery_tracks[:count]
    
    async def _recommend_by_mood(
        self, 
        mood: str, 
        user_profile: Dict[str, Any], 
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate mood-based recommendations."""
        if not mood:
            return []
        
        # Get mood embedding
        mood_embedding = await self.embedding_engine.embed_mood_description(mood)
        
        # Get candidates that match the mood
        candidates = await self._get_mood_candidates(mood)
        
        # Find tracks matching the mood
        mood_tracks = await self.embedding_engine.find_similar_tracks(
            mood_embedding.vector, candidates, top_k=count * 2
        )
        
        # Filter by audio features that match mood
        filtered_tracks = self._filter_by_mood_features(mood_tracks, mood)
        
        # Apply user personalization
        personalized_tracks = self._apply_user_preferences(filtered_tracks, user_profile)
        
        return personalized_tracks[:count]
    
    async def _recommend_general(
        self, 
        user_profile: Dict[str, Any], 
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate general personalized recommendations."""
        # Use collaborative filtering as primary method
        cf_recommendations = await self.ml_service.get_user_recommendations(
            user_profile.get("user_id"), count * 2
        )
        
        if not cf_recommendations:
            # Fallback to content-based recommendations
            return await self._content_based_recommendations(user_profile, count)
        
        # Apply content-based filtering for reranking
        reranked_tracks = await self._rerank_with_content(cf_recommendations, user_profile)
        
        return reranked_tracks[:count]
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user profile."""
        # This would integrate with the memory store
        # For now, return minimal profile
        return {
            "user_id": user_id,
            "preference_embedding": None,
            "genre_preferences": {},
            "audio_feature_preferences": {},
            "listening_history": []
        }
    
    async def _get_candidate_tracks(self, exclude_user_history: bool = False) -> List[Dict[str, Any]]:
        """Get candidate tracks for recommendation."""
        # This would query the database for candidate tracks
        return []
    
    async def _get_diverse_candidates(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get diverse candidates for discovery recommendations."""
        # This would query for tracks across different genres/styles
        return []
    
    async def _get_mood_candidates(self, mood: str) -> List[Dict[str, Any]]:
        """Get candidates that potentially match a mood."""
        # This would query for tracks with appropriate audio features
        return []
    
    async def _get_track_by_id(self, track_id: str) -> Dict[str, Any]:
        """Get track metadata by ID."""
        # This would query the database
        return {}
    
    async def _get_playlist_embedding(self, playlist_data: Dict[str, Any]) -> Any:
        """Get embedding for playlist data."""
        # Generate embedding from playlist tracks
        tracks = playlist_data.get("tracks", [])
        if not tracks:
            return None
        
        embeddings = []
        for track in tracks[:20]:  # Limit for performance
            emb = await self.embedding_engine.embed_track_metadata(track)
            embeddings.append(emb.vector)
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        return None
    
    def _combine_similarity_scores(
        self, 
        similar_tracks: List[Dict[str, Any]], 
        cf_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Combine similarity and collaborative filtering scores."""
        for track in similar_tracks:
            track_id = track.get("id")
            similarity_score = track.get("similarity", 0)
            cf_score = cf_scores.get(track_id, 0)
            
            # Weighted combination
            hybrid_score = 0.6 * similarity_score + 0.4 * cf_score
            track["hybrid_score"] = hybrid_score
        
        # Sort by hybrid score
        return sorted(similar_tracks, key=lambda x: x.get("hybrid_score", 0), reverse=True)
    
    def _apply_user_preferences(
        self, 
        tracks: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply user preferences to rerank tracks."""
        genre_prefs = user_profile.get("genre_preferences", {})
        feature_prefs = user_profile.get("audio_feature_preferences", {})
        
        for track in tracks:
            personalization_boost = 0.0
            
            # Genre preference boost
            track_genres = track.get("genres", [])
            for genre in track_genres:
                if genre in genre_prefs:
                    personalization_boost += genre_prefs[genre] * 0.1
            
            # Audio feature preference boost
            audio_features = track.get("audio_features", {})
            for feature, pref_value in feature_prefs.items():
                if feature in audio_features:
                    feature_value = audio_features[feature]
                    # Boost tracks with features close to user preference
                    similarity = 1.0 - abs(feature_value - pref_value)
                    personalization_boost += similarity * 0.05
            
            # Apply boost to existing score
            current_score = track.get("hybrid_score", track.get("similarity", 0.5))
            track["personalized_score"] = current_score + personalization_boost
        
        # Sort by personalized score
        return sorted(tracks, key=lambda x: x.get("personalized_score", 0), reverse=True)
    
    def _filter_by_mood_features(
        self, 
        tracks: List[Dict[str, Any]], 
        mood: str
    ) -> List[Dict[str, Any]]:
        """Filter tracks by audio features matching mood."""
        mood_filters = {
            "energetic": lambda f: f.get("energy", 0) > 0.6,
            "chill": lambda f: f.get("energy", 0) < 0.5 and f.get("valence", 0) > 0.3,
            "sad": lambda f: f.get("valence", 0) < 0.4,
            "happy": lambda f: f.get("valence", 0) > 0.6,
            "danceable": lambda f: f.get("danceability", 0) > 0.6,
            "acoustic": lambda f: f.get("acousticness", 0) > 0.5,
            "electronic": lambda f: f.get("acousticness", 0) < 0.3
        }
        
        mood_lower = mood.lower()
        applicable_filters = []
        
        for mood_key, filter_func in mood_filters.items():
            if mood_key in mood_lower:
                applicable_filters.append(filter_func)
        
        if not applicable_filters:
            return tracks
        
        filtered_tracks = []
        for track in tracks:
            audio_features = track.get("audio_features", {})
            
            # Track passes if it matches any applicable filter
            if any(filter_func(audio_features) for filter_func in applicable_filters):
                filtered_tracks.append(track)
        
        return filtered_tracks
    
    async def _content_based_recommendations(
        self, 
        user_profile: Dict[str, Any], 
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate content-based recommendations as fallback."""
        # Get user's listening history
        history = user_profile.get("listening_history", [])
        if not history:
            return []
        
        # Create user profile embedding from history
        history_embeddings = []
        for track in history[-20:]:  # Use recent history
            emb = await self.embedding_engine.embed_track_metadata(track)
            history_embeddings.append(emb.vector)
        
        if not history_embeddings:
            return []
        
        user_embedding = np.mean(history_embeddings, axis=0)
        
        # Find similar tracks
        candidates = await self._get_candidate_tracks(exclude_user_history=True)
        similar_tracks = await self.embedding_engine.find_similar_tracks(
            user_embedding, candidates, top_k=count
        )
        
        return similar_tracks
    
    async def _rerank_with_content(
        self, 
        tracks: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rerank collaborative filtering results with content features."""
        # Apply user preference boosting
        reranked = self._apply_user_preferences(tracks, user_profile)
        
        # Add diversity to avoid over-specialization
        diverse_tracks = self._ensure_diversity(reranked)
        
        return diverse_tracks
    
    def _ensure_diversity(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure diversity in recommendations."""
        diverse_tracks = []
        seen_artists = set()
        seen_genres = set()
        
        for track in tracks:
            artist = track.get("artist", "")
            genres = set(track.get("genres", []))
            
            # Add track if it introduces artist or genre diversity
            if (artist not in seen_artists or 
                not genres.intersection(seen_genres) or
                len(diverse_tracks) < 3):  # Always include first 3
                
                diverse_tracks.append(track)
                seen_artists.add(artist)
                seen_genres.update(genres)
                
                if len(diverse_tracks) >= len(tracks):
                    break
        
        return diverse_tracks
    
    def _generate_recommendation_explanation(
        self, 
        tracks: List[Dict[str, Any]], 
        rec_type: str, 
        user_profile: Dict[str, Any]
    ) -> str:
        """Generate explanation for the recommendation set."""
        explanations = {
            "similar": "Based on similarity to your selected track/playlist",
            "discovery": "Curated for musical discovery based on your taste",
            "mood": "Selected to match your requested mood",
            "general": "Personalized recommendations based on your listening history"
        }
        
        base_explanation = explanations.get(rec_type, "Personalized music recommendations")
        
        # Add personalization details
        if user_profile.get("genre_preferences"):
            top_genres = sorted(
                user_profile["genre_preferences"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            if top_genres:
                genre_names = [genre for genre, _ in top_genres]
                base_explanation += f" with emphasis on {', '.join(genre_names)}"
        
        return base_explanation
    
    def _calculate_personalization_score(self, user_profile: Dict[str, Any]) -> float:
        """Calculate how personalized the recommendations are."""
        score = 0.0
        
        if user_profile.get("preference_embedding"):
            score += 0.4
        if user_profile.get("genre_preferences"):
            score += 0.3
        if user_profile.get("listening_history"):
            score += 0.3
        
        return score