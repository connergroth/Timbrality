from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import asyncio
from dataclasses import dataclass

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
except ImportError:
    # Fallback when ML dependencies are not available
    ML_AVAILABLE = False
    np = None
    SentenceTransformer = None

from utils.cache import cache_manager


@dataclass
class EmbeddingResult:
    vector: Any  # np.ndarray when available, list when fallback
    confidence: float
    metadata: Dict[str, Any]


class EmbeddingEngine:
    """
    Converts user input and track metadata into vector representations
    using Sentence-BERT for semantic understanding.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not ML_AVAILABLE:
            print("Warning: ML dependencies not available. Using fallback embedding engine.")
            self.model = None
            self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        else:
            try:
                self.model = SentenceTransformer(model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"Warning: Could not load ML model ({str(e)}). Using fallback embedding engine.")
                self.model = None
                self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        
    async def embed_user_input(self, text: str) -> EmbeddingResult:
        """Convert user natural language input to embedding vector."""
        # Clean and preprocess text
        processed_text = self._preprocess_text(text)
        
        if not ML_AVAILABLE or self.model is None:
            # Fallback: create a simple hash-based vector
            vector = self._create_fallback_vector(processed_text)
        else:
            # Generate embedding using ML model
            vector = await asyncio.to_thread(
                self.model.encode, processed_text, normalize_embeddings=True
            )
        
        return EmbeddingResult(
            vector=vector,
            confidence=1.0 if ML_AVAILABLE else 0.1,  # Lower confidence for fallback
            metadata={
                "original_text": text,
                "processed_text": processed_text,
                "timestamp": datetime.now().isoformat(),
                "fallback_mode": not ML_AVAILABLE
            }
        )
    
    async def embed_track_metadata(self, track_data: Dict[str, Any]) -> EmbeddingResult:
        """Convert track metadata into embedding vector."""
        cache_key = f"track_embedding:{track_data.get('id', '')}"
        
        # Check cache first
        cached = await cache_manager.get(cache_key)
        if cached:
            return EmbeddingResult(**json.loads(cached))
        
        # Create text representation of track
        track_text = self._track_to_text(track_data)
        
        # Generate embedding
        if not ML_AVAILABLE or self.model is None:
            vector = self._create_fallback_vector(track_text)
        else:
            vector = await asyncio.to_thread(
                self.model.encode, track_text, normalize_embeddings=True
            )
        
        result = EmbeddingResult(
            vector=vector,
            confidence=0.9,  # High confidence for structured data
            metadata={
                "track_id": track_data.get("id"),
                "track_text": track_text,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Cache result
        await cache_manager.set(
            cache_key, 
            json.dumps({
                "vector": vector.tolist(),
                "confidence": result.confidence,
                "metadata": result.metadata
            }),
            expire=86400  # 24 hours
        )
        
        return result
    
    async def embed_mood_description(self, mood: str) -> EmbeddingResult:
        """Convert mood/vibe description to embedding vector."""
        # Enhance mood with contextual keywords
        enhanced_mood = self._enhance_mood_description(mood)
        
        vector = await asyncio.to_thread(
            self.model.encode, enhanced_mood, normalize_embeddings=True
        )
        
        return EmbeddingResult(
            vector=vector,
            confidence=0.8,  # Medium confidence for subjective moods
            metadata={
                "original_mood": mood,
                "enhanced_mood": enhanced_mood,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    async def compute_similarity(
        self, 
        embedding1: Any, 
        embedding2: Any
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        if not ML_AVAILABLE:
            # Fallback: simple dot product for lists
            if isinstance(embedding1, list) and isinstance(embedding2, list):
                return float(sum(a * b for a, b in zip(embedding1, embedding2)))
            return 0.0
        else:
            return float(np.dot(embedding1, embedding2))
    
    async def find_similar_tracks(
        self, 
        query_embedding: Any, 
        track_embeddings: List[Dict[str, Any]], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find most similar tracks to a query embedding."""
        similarities = []
        
        for track_data in track_embeddings:
            if "embedding" in track_data:
                track_embedding = track_data["embedding"]
                if ML_AVAILABLE and not isinstance(track_embedding, list):
                    track_embedding = np.array(track_embedding)
                similarity = await self.compute_similarity(query_embedding, track_embedding)
                
                similarities.append({
                    **track_data,
                    "similarity": similarity
                })
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    async def update_user_profile(
        self, 
        user_id: str, 
        tracks: List[Dict[str, Any]], 
        interaction_context: str
    ):
        """Update user embedding profile based on interactions."""
        # Get current user profile
        user_profile = await self._get_user_profile(user_id)
        
        # Embed interaction context
        context_embedding = await self.embed_user_input(interaction_context)
        
        # Embed tracks the user interacted with
        track_embeddings = []
        for track in tracks:
            track_emb = await self.embed_track_metadata(track)
            track_embeddings.append(track_emb.vector)
        
        # Update user profile with weighted average
        if track_embeddings:
            if ML_AVAILABLE:
                new_preference_vector = np.mean(track_embeddings, axis=0)
            else:
                # Fallback: compute mean manually for lists
                dim = len(track_embeddings[0]) if track_embeddings else self.dimension
                new_preference_vector = [
                    sum(emb[i] for emb in track_embeddings) / len(track_embeddings)
                    for i in range(dim)
                ]
            
            # Blend with existing profile (decay factor for temporal adaptation)
            decay_factor = 0.9
            if user_profile.get("preference_vector"):
                existing_vector = user_profile["preference_vector"]
                if ML_AVAILABLE and not isinstance(existing_vector, list):
                    existing_vector = np.array(existing_vector)
                    blended_vector = (
                        decay_factor * existing_vector + 
                        (1 - decay_factor) * new_preference_vector
                    )
                else:
                    # Fallback: manual blending for lists
                    blended_vector = [
                        decay_factor * existing_vector[i] + (1 - decay_factor) * new_preference_vector[i]
                        for i in range(len(existing_vector))
                    ]
            else:
                blended_vector = new_preference_vector
            
            # Update user profile
            vector_list = blended_vector.tolist() if hasattr(blended_vector, 'tolist') else blended_vector
            await self._update_user_profile(user_id, {
                "preference_vector": vector_list,
                "last_updated": datetime.now().isoformat(),
                "interaction_count": user_profile.get("interaction_count", 0) + 1
            })
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for embedding."""
        # Basic text cleaning
        text = text.lower().strip()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        return text
    
    def _track_to_text(self, track_data: Dict[str, Any]) -> str:
        """Convert track metadata to text representation."""
        parts = []
        
        # Basic track info
        if track_data.get("name"):
            parts.append(f"Song: {track_data['name']}")
        
        if track_data.get("artist"):
            parts.append(f"Artist: {track_data['artist']}")
            
        if track_data.get("album"):
            parts.append(f"Album: {track_data['album']}")
        
        # Genre and tags
        if track_data.get("genres"):
            genres = ", ".join(track_data["genres"])
            parts.append(f"Genres: {genres}")
            
        if track_data.get("tags"):
            tags = ", ".join(track_data["tags"])
            parts.append(f"Tags: {tags}")
        
        # Audio features
        if track_data.get("audio_features"):
            features = track_data["audio_features"]
            feature_text = []
            
            if features.get("danceability", 0) > 0.7:
                feature_text.append("danceable")
            if features.get("energy", 0) > 0.7:
                feature_text.append("energetic")
            if features.get("valence", 0) > 0.7:
                feature_text.append("positive")
            elif features.get("valence", 0) < 0.3:
                feature_text.append("melancholic")
            
            if feature_text:
                parts.append(f"Mood: {', '.join(feature_text)}")
        
        return ". ".join(parts)
    
    def _enhance_mood_description(self, mood: str) -> str:
        """Enhance mood description with contextual keywords."""
        mood_keywords = {
            "summer": ["sunny", "warm", "bright", "energetic", "outdoor"],
            "night": ["dark", "atmospheric", "ambient", "calm", "intimate"],
            "chill": ["relaxed", "laid-back", "mellow", "smooth", "easy"],
            "energetic": ["upbeat", "fast", "dynamic", "powerful", "driving"],
            "sad": ["melancholic", "slow", "emotional", "minor", "contemplative"],
            "happy": ["uplifting", "bright", "major", "cheerful", "positive"]
        }
        
        enhanced_parts = [mood]
        
        for keyword, synonyms in mood_keywords.items():
            if keyword in mood.lower():
                enhanced_parts.extend(synonyms[:3])  # Add top 3 synonyms
        
        return " ".join(enhanced_parts)
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user profile from memory store."""
        # This would integrate with the memory store
        # For now, return empty profile
        return {}
    
    async def _update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Update user profile in memory store."""
        # This would integrate with the memory store
        # For now, just pass
        pass
    
    def _create_fallback_vector(self, text: str) -> List[float]:
        """Create a simple fallback vector when ML dependencies are not available."""
        # Create a simple hash-based vector
        import hashlib
        
        # Create hash from text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numbers and normalize to create a vector
        vector = []
        for i in range(0, len(text_hash), 2):
            # Take pairs of hex digits and convert to float
            hex_pair = text_hash[i:i+2]
            float_val = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            vector.append(float_val)
        
        # Pad or truncate to desired dimension
        while len(vector) < self.dimension:
            vector.extend(vector[:self.dimension - len(vector)])
        
        return vector[:self.dimension]


# Convenience helper used by services that expect a simple embedding function
async def get_embedding(text: str) -> List[float]:
    """Return an embedding vector for the given text.

    Creates a transient EmbeddingEngine instance and returns a list[float].
    """
    engine = EmbeddingEngine()
    result = await engine.embed_user_input(text)
    vector = result.vector
    # Ensure plain list for downstream consumers
    if hasattr(vector, 'tolist'):
        vector = vector.tolist()
    return vector