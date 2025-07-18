from typing import Dict, Any, List
from .base import BaseTool, ToolResult


class ExplainabilityTool(BaseTool):
    """Tool for explaining why recommendations were made."""
    
    def __init__(self):
        super().__init__(
            name="explainability",
            description="Generate explanations for music recommendations"
        )
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Generate explanations for recommendations."""
        try:
            entities = intent_result.entities
            tracks = entities.get("tracks", [])
            user_query = entities.get("user_query", "")
            recommendation_context = entities.get("context", {})
            
            if not tracks:
                return self._create_error_result("No tracks provided for explanation")
            
            explanations = []
            for track in tracks[:5]:  # Explain top 5 tracks
                explanation = self._generate_track_explanation(
                    track, user_query, recommendation_context, context
                )
                explanations.append(explanation)
            
            # Generate overall explanation
            overall_explanation = self._generate_overall_explanation(
                tracks, user_query, recommendation_context, context
            )
            
            return self._create_success_result({
                "track_explanations": explanations,
                "overall_explanation": overall_explanation,
                "explanation_factors": self._get_explanation_factors(tracks, context)
            }, confidence=0.8)
            
        except Exception as e:
            return self._create_error_result(f"Explanation generation failed: {str(e)}")
    
    def _generate_track_explanation(
        self, 
        track: Dict[str, Any], 
        user_query: str, 
        rec_context: Dict[str, Any],
        context
    ) -> Dict[str, Any]:
        """Generate explanation for a single track recommendation."""
        factors = []
        reasoning_parts = []
        
        # Query matching
        if user_query:
            query_match = self._explain_query_match(track, user_query)
            if query_match:
                factors.append("query_match")
                reasoning_parts.append(query_match)
        
        # Similarity to user preferences
        similarity_reason = self._explain_similarity(track, context)
        if similarity_reason:
            factors.append("user_similarity")
            reasoning_parts.append(similarity_reason)
        
        # Audio features matching
        audio_reason = self._explain_audio_features(track, user_query)
        if audio_reason:
            factors.append("audio_features")
            reasoning_parts.append(audio_reason)
        
        # Popularity and quality
        quality_reason = self._explain_quality_factors(track)
        if quality_reason:
            factors.append("quality")
            reasoning_parts.append(quality_reason)
        
        # Genre/style matching
        style_reason = self._explain_style_match(track, user_query, context)
        if style_reason:
            factors.append("style_match")
            reasoning_parts.append(style_reason)
        
        # Construct explanation
        if reasoning_parts:
            explanation = f"Recommended because {', and '.join(reasoning_parts)}."
        else:
            explanation = "Recommended based on general musical compatibility."
        
        return {
            "track_id": track.get("id"),
            "track_name": track.get("name"),
            "artist": track.get("artist"),
            "explanation": explanation,
            "factors": factors,
            "confidence": len(factors) * 0.2  # Higher confidence with more factors
        }
    
    def _explain_query_match(self, track: Dict[str, Any], user_query: str) -> str:
        """Explain how track matches user query."""
        query_lower = user_query.lower()
        track_name = track.get("name", "").lower()
        artist_name = track.get("artist", "").lower()
        
        # Direct name/artist matches
        if track_name in query_lower or any(word in track_name for word in query_lower.split()):
            return f"it matches your search for '{user_query}'"
        
        if artist_name in query_lower or any(word in artist_name for word in query_lower.split()):
            return f"it's by {track.get('artist')}, which matches your search"
        
        # Mood/genre matches
        mood_keywords = {
            "summer": "it has a summer vibe",
            "chill": "it has a chill, relaxed feel",
            "energetic": "it has high energy",
            "sad": "it has melancholic qualities",
            "happy": "it has an uplifting mood",
            "night": "it has a nighttime atmosphere",
            "party": "it's perfect for parties",
            "workout": "it has driving energy for workouts"
        }
        
        for keyword, explanation in mood_keywords.items():
            if keyword in query_lower:
                # Check if track actually matches this mood
                if self._track_matches_mood(track, keyword):
                    return explanation
        
        return ""
    
    def _explain_similarity(self, track: Dict[str, Any], context) -> str:
        """Explain similarity to user's preferences."""
        # This would analyze user's listening history
        # For now, provide generic similarity explanations
        
        similarity_explanations = [
            "it's similar to music you've liked before",
            "it matches your taste profile",
            "it's by an artist similar to ones you enjoy",
            "it shares characteristics with your favorite tracks"
        ]
        
        # Use track similarity score if available
        if track.get("similarity"):
            sim_score = track["similarity"]
            if sim_score > 0.8:
                return "it's very similar to music you love"
            elif sim_score > 0.6:
                return "it's similar to your music taste"
            elif sim_score > 0.4:
                return "it has elements you typically enjoy"
        
        return ""
    
    def _explain_audio_features(self, track: Dict[str, Any], user_query: str) -> str:
        """Explain audio features that match the request."""
        audio_features = track.get("audio_features", {})
        if not audio_features:
            return ""
        
        explanations = []
        
        # Energy level
        energy = audio_features.get("energy", 0)
        if energy > 0.8:
            explanations.append("it's high-energy")
        elif energy < 0.3:
            explanations.append("it's calm and mellow")
        
        # Danceability
        danceability = audio_features.get("danceability", 0)
        if danceability > 0.7:
            explanations.append("it's very danceable")
        
        # Valence (mood)
        valence = audio_features.get("valence", 0)
        if valence > 0.7:
            explanations.append("it has a positive, uplifting mood")
        elif valence < 0.3:
            explanations.append("it has an emotional, introspective feel")
        
        # Acousticness
        acousticness = audio_features.get("acousticness", 0)
        if acousticness > 0.7:
            explanations.append("it has an acoustic, organic sound")
        elif acousticness < 0.2:
            explanations.append("it has an electronic, produced sound")
        
        if explanations:
            return " and ".join(explanations)
        
        return ""
    
    def _explain_quality_factors(self, track: Dict[str, Any]) -> str:
        """Explain quality and popularity factors."""
        explanations = []
        
        # Spotify popularity
        popularity = track.get("popularity", 0)
        if popularity > 80:
            explanations.append("it's very popular")
        elif popularity > 60:
            explanations.append("it's well-liked by listeners")
        
        # AOTY rating
        aoty_rating = track.get("aoty_rating", 0)
        if aoty_rating > 80:
            explanations.append("it's critically acclaimed")
        elif aoty_rating > 70:
            explanations.append("it's well-reviewed")
        
        # Last.fm listeners
        listeners = track.get("listeners", 0)
        if listeners > 1000000:
            explanations.append("it's widely appreciated")
        
        if explanations:
            return " and ".join(explanations)
        
        return ""
    
    def _explain_style_match(self, track: Dict[str, Any], user_query: str, context) -> str:
        """Explain genre/style matching."""
        genres = track.get("genres", [])
        tags = track.get("tags", [])
        
        # Check if query mentions genres
        query_lower = user_query.lower()
        
        for genre in genres:
            if genre.lower() in query_lower:
                return f"it's in the {genre} genre you requested"
        
        for tag in tags:
            if tag.lower() in query_lower:
                return f"it has the {tag} style you're looking for"
        
        # Generic style explanations
        if genres:
            return f"it fits the {genres[0]} style"
        
        return ""
    
    def _track_matches_mood(self, track: Dict[str, Any], mood: str) -> bool:
        """Check if track actually matches the specified mood."""
        audio_features = track.get("audio_features", {})
        
        mood_criteria = {
            "summer": lambda f: f.get("valence", 0) > 0.5 and f.get("energy", 0) > 0.4,
            "chill": lambda f: f.get("energy", 0) < 0.6 and f.get("valence", 0) > 0.3,
            "energetic": lambda f: f.get("energy", 0) > 0.7,
            "sad": lambda f: f.get("valence", 0) < 0.4,
            "happy": lambda f: f.get("valence", 0) > 0.6,
            "night": lambda f: f.get("energy", 0) < 0.5,
            "party": lambda f: f.get("danceability", 0) > 0.6 and f.get("energy", 0) > 0.6,
            "workout": lambda f: f.get("energy", 0) > 0.7 and f.get("tempo", 0) > 120
        }
        
        criteria_func = mood_criteria.get(mood)
        if criteria_func and audio_features:
            return criteria_func(audio_features)
        
        return True  # Default to true if no specific criteria
    
    def _generate_overall_explanation(
        self, 
        tracks: List[Dict[str, Any]], 
        user_query: str,
        rec_context: Dict[str, Any],
        context
    ) -> str:
        """Generate overall explanation for the recommendation set."""
        if not tracks:
            return "No recommendations available."
        
        explanation_parts = []
        
        # Query-based explanation
        if user_query:
            explanation_parts.append(f"Based on your request for '{user_query}'")
        
        # Method explanation
        methods_used = rec_context.get("methods_used", [])
        if "similarity" in methods_used:
            explanation_parts.append("using similarity analysis")
        if "ml_model" in methods_used:
            explanation_parts.append("with machine learning recommendations")
        if "user_profile" in methods_used:
            explanation_parts.append("considering your listening history")
        
        # Quality note
        track_count = len(tracks)
        explanation_parts.append(f"I found {track_count} tracks that match your preferences")
        
        # Diversity note
        artists = set(track.get("artist", "") for track in tracks)
        if len(artists) > len(tracks) * 0.7:
            explanation_parts.append("with variety across different artists")
        
        return ", ".join(explanation_parts) + "."
    
    def _get_explanation_factors(self, tracks: List[Dict[str, Any]], context) -> List[str]:
        """Get list of all factors used in explanations."""
        factors = set()
        
        for track in tracks:
            if track.get("similarity"):
                factors.add("similarity_matching")
            if track.get("audio_features"):
                factors.add("audio_analysis")
            if track.get("popularity"):
                factors.add("popularity_ranking")
            if track.get("genres"):
                factors.add("genre_matching")
            if track.get("aoty_rating"):
                factors.add("critical_ratings")
        
        return list(factors)