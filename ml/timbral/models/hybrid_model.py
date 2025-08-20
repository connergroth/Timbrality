"""
Hybrid recommendation model combining NMF collaborative filtering and Two-Tower content-based model.

This module implements a hybrid recommendation system that combines
collaborative filtering (NMF) with content-based features (Two-Tower)
for improved music recommendations.
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import logging

from .nmf_model import NMFModel
from .two_tower_model import TwoTowerModel
from .audio_feature_predictor import AudioFeaturePredictor
from .meta_learning_fusion import MetaLearningFusion, SessionContextExtractor, UncertaintyContextExtractor
from ..config.settings import settings

logger = logging.getLogger(__name__)


class HybridModel(nn.Module):
    """
    Hybrid recommendation model combining NMF and Two-Tower architecture.
    
    This class combines collaborative filtering (NMF) with content-based
    features (Two-Tower + predicted audio features) for comprehensive recommendations.
    """
    
    def __init__(
        self,
        nmf_model: NMFModel,
        two_tower_model: TwoTowerModel,
        audio_predictor: Optional[AudioFeaturePredictor] = None,
        fusion_method: str = "meta_learning",
        cf_weight: float = 0.6,
        content_weight: float = 0.4,
        embedding_dim: int = 64,
        enable_meta_learning: bool = True,
        session_feature_dim: int = 18,
        min_ratings_for_acclaim: int = 100,
        high_acclaim_threshold: int = 500
    ):
        """
        Initialize hybrid model.
        
        Args:
            nmf_model: Trained NMF model for collaborative filtering
            two_tower_model: Trained two-tower model for content-based filtering
            audio_predictor: Optional audio feature predictor
            fusion_method: Method to combine CF and content scores
            cf_weight: Weight for collaborative filtering predictions (fallback)
            content_weight: Weight for content-based predictions (fallback)
            embedding_dim: Embedding dimension for fusion
            enable_meta_learning: Whether to use meta-learning fusion
            session_feature_dim: Dimension of session features for meta-learning
            min_ratings_for_acclaim: Minimum ratings needed for "critically acclaimed" label
            high_acclaim_threshold: Threshold for "widely acclaimed" vs "strong consensus"
        """
        super().__init__()
        
        self.nmf_model = nmf_model
        self.two_tower_model = two_tower_model
        self.audio_predictor = audio_predictor
        self.fusion_method = fusion_method
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.enable_meta_learning = enable_meta_learning
        self.min_ratings_for_acclaim = min_ratings_for_acclaim
        self.high_acclaim_threshold = high_acclaim_threshold
        
        # Meta-learning components (if enabled)
        if enable_meta_learning and fusion_method == "meta_learning":
            self.session_extractor = SessionContextExtractor(
                session_feature_dim=session_feature_dim,
                output_dim=embedding_dim
            )
            self.uncertainty_extractor = UncertaintyContextExtractor(
                output_dim=32
            )
            self.meta_fusion = MetaLearningFusion(
                user_context_dim=embedding_dim,
                item_context_dim=32,
                cf_embedding_dim=embedding_dim,
                content_embedding_dim=embedding_dim
            )
        else:
            # Fallback fusion mechanisms
            if fusion_method == "learned_weights":
                self.fusion_layer = self._build_learned_fusion(embedding_dim)
            elif fusion_method == "adaptive":
                self.fusion_layer = self._build_adaptive_fusion(embedding_dim)
            else:
                self.fusion_layer = None  # Use simple weighted sum
    
    def _build_learned_fusion(self, embedding_dim: int) -> nn.Module:
        """
        Build learned fusion layer for combining CF and content predictions.
        
        Args:
            embedding_dim: Embedding dimension
        
        Returns:
            Fusion neural network layer
        """
        return nn.Sequential(
            nn.Linear(3, 32),  # CF score + content score + confidence
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def _build_adaptive_fusion(self, embedding_dim: int) -> nn.Module:
        """
        Build adaptive fusion layer that learns user-specific weights.
        
        Args:
            embedding_dim: Embedding dimension
            
        Returns:
            Adaptive fusion network
        """
        return nn.Sequential(
            nn.Linear(embedding_dim + 3, 64),  # User embedding + scores + confidence
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Output CF weight and content weight
            nn.Softmax(dim=1)  # Weights sum to 1
        )
    
    def predict(
        self,
        user_data: Dict[str, Any],
        item_data: Dict[str, Any],
        session_features: Optional[Dict[str, Any]] = None,
        use_audio_prediction: bool = True
    ) -> Dict[str, float]:
        """
        Generate hybrid predictions for user-item pairs.
        
        Args:
            user_data: Dictionary containing user information and preferences
            item_data: Dictionary containing item metadata and features
            session_features: Optional session context for meta-learning
            use_audio_prediction: Whether to use audio feature prediction
            
        Returns:
            Dictionary with prediction scores and explanations
        """
        results = {}
        
        # 1. Collaborative Filtering Score (if user has history)
        cf_score = 0.0
        cf_confidence = 0.0
        
        if self.nmf_model.is_fitted and user_data.get('user_matrix_id') is not None:
            try:
                user_matrix_id = user_data['user_matrix_id']
                item_matrix_id = item_data.get('item_matrix_id')
                
                if item_matrix_id is not None:
                    cf_scores = self.nmf_model.predict(
                        np.array([user_matrix_id]),
                        np.array([item_matrix_id])
                    )
                    cf_score = float(cf_scores[0])
                    cf_confidence = self._calculate_cf_confidence(user_data, item_data)
                
            except Exception as e:
                logger.warning(f"CF prediction failed: {e}")
                cf_score = 0.0
                cf_confidence = 0.0
        
        # 2. Content-Based Score
        content_score = 0.0
        content_confidence = 0.0
        
        try:
            # Predict audio features if needed
            audio_features = item_data.get('audio_features', {})
            if use_audio_prediction and self.audio_predictor and not audio_features:
                audio_features = self.audio_predictor.predict_audio_features(
                    title=item_data.get('title', ''),
                    artist=item_data.get('artist', ''),
                    genres=item_data.get('genres', []),
                    moods=item_data.get('moods', []),
                    metadata=self._extract_metadata_features(item_data)
                )
            
            # Get content-based prediction from two-tower model
            content_score, content_confidence = self._get_content_prediction(
                user_data, item_data, audio_features
            )
            
        except Exception as e:
            logger.warning(f"Content prediction failed: {e}")
            content_score = 0.0
            content_confidence = 0.0
        
        # 3. Fusion (Meta-learning or fallback)
        if cf_confidence > 0 and content_confidence > 0:
            # Both available - use appropriate fusion method
            if self.enable_meta_learning and self.fusion_method == "meta_learning" and session_features:
                final_score, fusion_details = self._meta_learning_fusion(
                    cf_score, content_score, cf_confidence, content_confidence,
                    user_data, item_data, session_features, audio_features
                )
                method = "meta_hybrid"
                results.update(fusion_details)
            else:
                # Fallback to traditional fusion
                final_score = self._fuse_predictions(
                    cf_score, content_score, cf_confidence, content_confidence, user_data
                )
                method = "hybrid"
        elif cf_confidence > 0:
            # Only CF available
            final_score = cf_score
            method = "collaborative"
        elif content_confidence > 0:
            # Only content available
            final_score = content_score
            method = "content"
        else:
            # Fallback to popularity or random
            final_score = item_data.get('popularity', 50) / 100.0
            method = "popularity"
        
        results = {
            'final_score': final_score,
            'cf_score': cf_score,
            'content_score': content_score,
            'cf_confidence': cf_confidence,
            'content_confidence': content_confidence,
            'method': method,
            'audio_features_predicted': use_audio_prediction and audio_features
        }
        
        return results
    
    def _calculate_cf_confidence(self, user_data: Dict, item_data: Dict) -> float:
        """Calculate confidence for collaborative filtering prediction."""
        # Base confidence on user's interaction history and item popularity
        user_interactions = user_data.get('total_interactions', 0)
        item_interactions = item_data.get('total_interactions', 0)
        
        # Confidence increases with more interactions (sigmoid function)
        user_conf = 1 / (1 + np.exp(-(user_interactions - 10) / 5))
        item_conf = 1 / (1 + np.exp(-(item_interactions - 5) / 3))
        
        return float(user_conf * item_conf)
    
    def _extract_metadata_features(self, item_data: Dict) -> Dict[str, float]:
        """Extract metadata features for audio prediction (NO popularity data to avoid leakage)."""
        release_year = item_data.get('release_year', 2000)
        return {
            'duration_ms': item_data.get('duration_ms', 180000),
            'release_year': release_year,
            'decade': (release_year // 10) * 10,
            'explicit': float(item_data.get('explicit', False)),
            'track_number': item_data.get('track_number', 1),
            'album_total_tracks': item_data.get('album_total_tracks', 10)
        }
    
    def _get_content_prediction(
        self, 
        user_data: Dict, 
        item_data: Dict, 
        audio_features: Dict
    ) -> Tuple[float, float]:
        """Get content-based prediction from two-tower model."""
        device = next(self.two_tower_model.parameters()).device
        
        # Prepare user features
        user_id = user_data.get('user_id', 0)
        genre_prefs = torch.zeros(1000, device=device)  # Assuming 1000 genres
        mood_prefs = torch.zeros(500, device=device)   # Assuming 500 moods
        
        # Set user preferences based on history
        user_genres = user_data.get('preferred_genres', [])
        for genre_id in user_genres[:10]:  # Top 10 genres
            if isinstance(genre_id, int) and 0 <= genre_id < 1000:
                genre_prefs[genre_id] = 1.0
        
        user_moods = user_data.get('preferred_moods', [])
        for mood_id in user_moods[:10]:  # Top 10 moods
            if isinstance(mood_id, int) and 0 <= mood_id < 500:
                mood_prefs[mood_id] = 1.0
        
        # User activity features
        activity = torch.tensor([
            user_data.get('total_plays', 0) / 1000.0,  # Normalized
            user_data.get('session_length', 30) / 60.0,  # Minutes to hours
            user_data.get('skip_rate', 0.1),
            user_data.get('discovery_rate', 0.2),
            user_data.get('repeat_rate', 0.3),
            user_data.get('avg_rating', 3.5) / 5.0,
            user_data.get('genre_diversity', 0.5),
            user_data.get('recency_bias', 0.5),
            user_data.get('popularity_bias', 0.5),
            float(user_data.get('is_premium', False))
        ], device=device)
        
        # Prepare item features
        track_text = self._create_track_text(item_data)
        artist_id = item_data.get('artist_id', 0)
        
        # Item genres and moods
        item_genre_ids = torch.zeros(10, dtype=torch.long, device=device)
        item_genres = item_data.get('genre_ids', [])
        for i, genre_id in enumerate(item_genres[:10]):
            if isinstance(genre_id, int):
                item_genre_ids[i] = genre_id
        
        item_mood_ids = torch.zeros(10, dtype=torch.long, device=device)
        item_moods = item_data.get('mood_ids', [])
        for i, mood_id in enumerate(item_moods[:10]):
            if isinstance(mood_id, int):
                item_mood_ids[i] = mood_id
        
        # Audio features (predicted or actual)
        audio_tensor = torch.tensor([
            audio_features.get('energy', 0.5),
            audio_features.get('valence', 0.5),
            audio_features.get('danceability', 0.5),
            audio_features.get('acousticness', 0.5),
            audio_features.get('instrumentalness', 0.5),
            audio_features.get('liveness', 0.5),
            audio_features.get('speechiness', 0.5),
            audio_features.get('tempo', 120) / 200.0  # Normalize tempo
        ], device=device)
        
        # Audio confidence scores (higher if actual data, lower if predicted)
        is_predicted = item_data.get('audio_features_predicted', False)
        base_confidence = 0.6 if is_predicted else 0.9
        
        # Adjust confidence based on available metadata
        metadata_completeness = self._calculate_metadata_completeness(item_data)
        adjusted_confidence = base_confidence * metadata_completeness
        
        audio_confidence = torch.tensor([adjusted_confidence] * 8, device=device)
        
        # Rating and popularity features
        rating_tensor = torch.tensor([
            item_data.get('aoty_score', 50) / 100.0,
            item_data.get('popularity', 50) / 100.0,
            item_data.get('aoty_num_ratings', 0) / 1000.0,  # Normalize
            item_data.get('user_rating_avg', 3.5) / 5.0,
            item_data.get('critic_score', 75) / 100.0
        ], device=device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.two_tower_model(
                user_ids=torch.tensor([user_id], device=device),
                user_genre_prefs=genre_prefs.unsqueeze(0),
                user_mood_prefs=mood_prefs.unsqueeze(0),
                user_activity=activity.unsqueeze(0),
                track_texts=[track_text],
                artist_ids=torch.tensor([artist_id], device=device),
                item_genre_ids=item_genre_ids.unsqueeze(0),
                item_mood_ids=item_mood_ids.unsqueeze(0),
                audio_features=audio_tensor.unsqueeze(0),
                audio_confidence=audio_confidence.unsqueeze(0),
                rating_features=rating_tensor.unsqueeze(0)
            )
        
        content_score = float(outputs['scores'][0])
        
        # Calculate confidence based on feature completeness
        feature_completeness = self._calculate_content_confidence(item_data, audio_features)
        
        return content_score, feature_completeness
    
    def _create_track_text(self, item_data: Dict) -> str:
        """Create text representation of track for embedding."""
        parts = []
        
        if item_data.get('title'):
            parts.append(f"Title: {item_data['title']}")
        
        if item_data.get('artist'):
            parts.append(f"Artist: {item_data['artist']}")
        
        if item_data.get('album'):
            parts.append(f"Album: {item_data['album']}")
        
        if item_data.get('genres'):
            genres = ', '.join(item_data['genres'][:5])  # Top 5 genres
            parts.append(f"Genres: {genres}")
        
        if item_data.get('moods'):
            moods = ', '.join(item_data['moods'][:5])  # Top 5 moods
            parts.append(f"Moods: {moods}")
        
        # Add AOTY review snippets if available
        if item_data.get('review_snippet'):
            parts.append(f"Review: {item_data['review_snippet'][:200]}")
        
        return '. '.join(parts)
    
    def _calculate_content_confidence(self, item_data: Dict, audio_features: Dict) -> float:
        """Calculate confidence for content-based prediction."""
        confidence = 0.0
        
        # Text features (always available)
        if item_data.get('title'):
            confidence += 0.2
        if item_data.get('artist'):
            confidence += 0.2
        if item_data.get('genres'):
            confidence += 0.2
        
        # Additional metadata
        if item_data.get('moods'):
            confidence += 0.1
        if item_data.get('review_snippet'):
            confidence += 0.1
        if audio_features:
            confidence += 0.2  # Audio features predicted or available
        
        return min(confidence, 1.0)
    
    def _calculate_metadata_completeness(self, item_data: Dict) -> float:
        """Calculate completeness of metadata for confidence scoring."""
        completeness = 0.0
        
        # Basic metadata
        if item_data.get('title'):
            completeness += 0.2
        if item_data.get('artist'):
            completeness += 0.2
        if item_data.get('album'):
            completeness += 0.1
        
        # Genre and mood data
        if item_data.get('genres'):
            completeness += 0.2
        if item_data.get('moods'):
            completeness += 0.1
        
        # Technical metadata
        if item_data.get('duration_ms'):
            completeness += 0.1
        if item_data.get('release_year'):
            completeness += 0.1
        
        return min(completeness, 1.0)
    
    def _meta_learning_fusion(
        self,
        cf_score: float,
        content_score: float,
        cf_confidence: float,
        content_confidence: float,
        user_data: Dict[str, Any],
        item_data: Dict[str, Any],
        session_features: Dict[str, Any],
        audio_features: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Meta-learning fusion that adaptively weights predictions based on context.
        
        Returns:
            Tuple of (fused_score, fusion_details)
        """
        device = next(self.meta_fusion.parameters()).device
        
        # 1. Extract session context
        session_feature_tensor = torch.tensor([
            session_features.get('total_sessions', 0),
            session_features.get('avg_session_length', 30),
            session_features.get('discovery_rate_per_active_week', 0.5),
            session_features.get('exploration_score', 0.5),
            session_features.get('novelty_seeking', 0.5),
            session_features.get('repeat_rate', 0.3),
            session_features.get('skip_rate', 0.1),
            session_features.get('genre_diversity', 0.5),
            session_features.get('temporal_consistency', 0.5),
            session_features.get('mainstream_score', 0.5),
            session_features.get('early_adopter_score', 0.3),
            session_features.get('plays_per_day', 10),
            session_features.get('active_days', 30),
            session_features.get('longest_streak_percentile', 0.5),
            session_features.get('session_density', 1.0),
            session_features.get('listening_intensity', 60),
            # User type one-hot (simplified)
            float(session_features.get('user_type', 'balanced') == 'explorer'),
            float(session_features.get('user_type', 'balanced') == 'loyalist')
        ], device=device).unsqueeze(0)
        
        user_context, _ = self.session_extractor(session_feature_tensor)
        
        # 2. Extract uncertainty context
        is_predicted = float(item_data.get('audio_features_predicted', False))
        metadata_completeness = self._calculate_metadata_completeness(item_data)
        
        # Create item uncertainty context
        audio_conf_tensor = torch.tensor([
            audio_features.get('energy', 0.5),
            audio_features.get('valence', 0.5),
            audio_features.get('danceability', 0.5),
            audio_features.get('acousticness', 0.5),
            audio_features.get('instrumentalness', 0.5),
            audio_features.get('liveness', 0.5),
            audio_features.get('speechiness', 0.5),
            audio_features.get('tempo', 120) / 200.0
        ], device=device).unsqueeze(0)
        
        item_uncertainty_context = self.uncertainty_extractor(
            audio_features=audio_conf_tensor,
            audio_confidence=audio_conf_tensor,  # Placeholder
            metadata_completeness=torch.tensor([[metadata_completeness]], device=device),
            is_predicted=torch.tensor([[is_predicted]], device=device)
        )
        
        # 3. Create dummy embeddings (in real implementation, get from models)
        cf_embedding = torch.randn(1, 64, device=device)  # Would come from NMF model
        content_embedding = torch.randn(1, 64, device=device)  # Would come from two-tower
        
        # 4. Meta-learning fusion
        with torch.no_grad():
            fusion_output = self.meta_fusion(
                cf_score=torch.tensor([cf_score], device=device),
                content_score=torch.tensor([content_score], device=device),
                cf_embedding=cf_embedding,
                content_embedding=content_embedding,
                user_session_context=user_context,
                item_uncertainty_context=item_uncertainty_context
            )
        
        # 5. Extract results
        fused_score = float(fusion_output['fused_score'][0])
        
        fusion_details = {
            'meta_cf_weight': float(fusion_output['cf_weight'][0]),
            'meta_content_weight': float(fusion_output['content_weight'][0]),
            'meta_cf_confidence': float(fusion_output['cf_confidence'][0]),
            'meta_content_confidence': float(fusion_output['content_confidence'][0]),
            'uncertainty_estimate': fusion_output['uncertainty_estimate'][0].tolist(),
            'fusion_method': 'meta_learning'
        }
        
        return fused_score, fusion_details
    
    def _fuse_predictions(
        self, 
        cf_score: float, 
        content_score: float, 
        cf_confidence: float, 
        content_confidence: float,
        user_data: Dict
    ) -> float:
        """Fuse collaborative and content-based predictions."""
        
        if self.fusion_method == "weighted_sum":
            # Simple weighted sum
            total_weight = cf_confidence + content_confidence
            if total_weight > 0:
                cf_weight = cf_confidence / total_weight
                content_weight = content_confidence / total_weight
                return cf_weight * cf_score + content_weight * content_score
            else:
                return (cf_score + content_score) / 2
        
        elif self.fusion_method == "learned_weights" and self.fusion_layer:
            # Use learned fusion network
            device = next(self.fusion_layer.parameters()).device
            fusion_input = torch.tensor([
                cf_score, content_score, (cf_confidence + content_confidence) / 2
            ], device=device).unsqueeze(0)
            
            with torch.no_grad():
                fused_score = self.fusion_layer(fusion_input).item()
            
            return fused_score
        
        elif self.fusion_method == "adaptive" and self.fusion_layer:
            # Use adaptive weights based on user embedding
            device = next(self.fusion_layer.parameters()).device
            
            # Get user embedding (simplified)
            user_embedding = torch.zeros(64, device=device)  # Placeholder
            
            fusion_input = torch.cat([
                user_embedding,
                torch.tensor([cf_score, content_score, (cf_confidence + content_confidence) / 2], device=device)
            ]).unsqueeze(0)
            
            with torch.no_grad():
                weights = self.fusion_layer(fusion_input)[0]  # [cf_weight, content_weight]
                fused_score = weights[0] * cf_score + weights[1] * content_score
            
            return fused_score.item()
        
        else:
            # Fallback to confidence-weighted average
            total_confidence = cf_confidence + content_confidence
            if total_confidence > 0:
                return (cf_confidence * cf_score + content_confidence * content_score) / total_confidence
            else:
                return (cf_score + content_score) / 2
    
    def get_recommendations(
        self,
        user_data: Dict[str, Any],
        candidate_items: List[Dict[str, Any]],
        top_k: int = 10,
        use_audio_prediction: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get top-k recommendations for a user.
        
        Args:
            user_data: User information and preferences
            candidate_items: List of candidate items with metadata
            top_k: Number of top recommendations to return
            use_audio_prediction: Whether to use audio feature prediction
            
        Returns:
            List of recommended items with scores
        """
        recommendations = []
        
        for item in candidate_items:
            try:
                prediction = self.predict(user_data, item, use_audio_prediction)
                
                item_with_score = item.copy()
                item_with_score.update(prediction)
                recommendations.append(item_with_score)
                
            except Exception as e:
                logger.warning(f"Failed to predict for item {item.get('id', 'unknown')}: {e}")
                continue
        
        # Sort by final score
        recommendations.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return recommendations[:top_k]
    
    def explain_recommendation(
        self,
        user_data: Dict[str, Any],
        item_data: Dict[str, Any],
        prediction_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate explanation for a specific recommendation.
        
        Args:
            user_data: User information and preferences
            item_data: Item information and metadata
            prediction_result: Optional pre-computed prediction result
            
        Returns:
            Dictionary with explanation components
        """
        if prediction_result is None:
            prediction_result = self.predict(user_data, item_data)
        
        explanation = {
            'recommendation_method': prediction_result.get('method', 'unknown'),
            'confidence': (prediction_result.get('cf_confidence', 0) + 
                          prediction_result.get('content_confidence', 0)) / 2,
            'factors': []
        }
        
        # Collaborative filtering explanation
        if prediction_result.get('cf_confidence', 0) > 0:
            cf_explanation = self._explain_collaborative_factors(user_data, item_data)
            explanation['factors'].append({
                'type': 'collaborative',
                'weight': prediction_result.get('cf_confidence', 0),
                'description': cf_explanation
            })
        
        # Content-based explanation
        if prediction_result.get('content_confidence', 0) > 0:
            content_explanation = self._explain_content_factors(user_data, item_data)
            explanation['factors'].append({
                'type': 'content',
                'weight': prediction_result.get('content_confidence', 0),
                'description': content_explanation
            })
        
        # Generate human-readable summary
        explanation['summary'] = self._generate_explanation_summary(
            user_data, item_data, prediction_result, explanation['factors']
        )
        
        return explanation
    
    def _explain_collaborative_factors(self, user_data: Dict, item_data: Dict) -> str:
        """Generate explanation for collaborative filtering component."""
        explanations = []
        
        # Similar users explanation
        if user_data.get('total_interactions', 0) > 10:
            explanations.append("based on users with similar music taste")
        
        # Item popularity
        item_popularity = item_data.get('popularity', 50)
        if item_popularity > 70:
            explanations.append("this is a popular track among users")
        elif item_popularity < 30:
            explanations.append("this is a hidden gem discovered by similar users")
        
        # User activity
        if user_data.get('discovery_rate', 0.2) > 0.5:
            explanations.append("you tend to explore new music")
        
        if not explanations:
            explanations.append("based on collaborative filtering patterns")
        
        return ", ".join(explanations)
    
    def _explain_content_factors(self, user_data: Dict, item_data: Dict) -> str:
        """Generate explanation for content-based component."""
        explanations = []
        
        # Genre preferences
        user_genres = set(user_data.get('preferred_genres', []))
        item_genres = set(item_data.get('genres', []))
        common_genres = user_genres.intersection(item_genres)
        
        if common_genres:
            genre_names = list(common_genres)[:3]  # Top 3
            explanations.append(f"you like {', '.join(genre_names)} music")
        
        # Mood preferences
        user_moods = set(user_data.get('preferred_moods', []))
        item_moods = set(item_data.get('moods', []))
        common_moods = user_moods.intersection(item_moods)
        
        if common_moods:
            mood_names = list(common_moods)[:2]  # Top 2
            explanations.append(f"matches your preference for {', '.join(mood_names)} vibes")
        
        # Artist similarity
        if item_data.get('artist') in user_data.get('preferred_artists', []):
            explanations.append(f"you've enjoyed {item_data['artist']} before")
        
        # Audio features (if predicted)
        if item_data.get('audio_features_predicted'):
            explanations.append("predicted audio characteristics match your taste")
        
        # AOTY rating (with minimum ratings threshold)
        aoty_score = item_data.get('aoty_score', 50)
        aoty_num_ratings = item_data.get('aoty_num_ratings', 0)
        if aoty_score > 80 and aoty_num_ratings >= self.min_ratings_for_acclaim:
            if aoty_num_ratings >= self.high_acclaim_threshold:
                explanations.append("widely acclaimed track with exceptional reviews")
            elif aoty_num_ratings >= (self.min_ratings_for_acclaim * 2):
                explanations.append("critically acclaimed track with strong consensus")
            else:
                explanations.append("critically acclaimed track")
        
        if not explanations:
            explanations.append("content features match your preferences")
        
        return ", ".join(explanations)
    
    def _generate_explanation_summary(
        self, 
        user_data: Dict, 
        item_data: Dict, 
        prediction: Dict, 
        factors: List[Dict]
    ) -> str:
        """Generate human-readable explanation summary."""
        
        track_name = item_data.get('title', 'This track')
        artist_name = item_data.get('artist', 'Unknown artist')
        
        method = prediction.get('method', 'hybrid')
        
        if method == 'hybrid':
            primary_factor = max(factors, key=lambda x: x['weight'])
            summary = f"Recommended '{track_name}' by {artist_name} "
            summary += f"primarily {primary_factor['description']}"
            
            if len(factors) > 1:
                secondary_factor = min(factors, key=lambda x: x['weight'])
                summary += f", and also {secondary_factor['description']}"
        
        elif method == 'collaborative':
            cf_factor = next((f for f in factors if f['type'] == 'collaborative'), None)
            if cf_factor:
                summary = f"Recommended '{track_name}' by {artist_name} {cf_factor['description']}"
            else:
                summary = f"Recommended '{track_name}' by {artist_name} based on collaborative filtering"
        
        elif method == 'content':
            content_factor = next((f for f in factors if f['type'] == 'content'), None)
            if content_factor:
                summary = f"Recommended '{track_name}' by {artist_name} because {content_factor['description']}"
            else:
                summary = f"Recommended '{track_name}' by {artist_name} based on content similarity"
        
        else:
            summary = f"Recommended '{track_name}' by {artist_name} as a popular choice"
        
        return summary
    
    def save(self, filepath: str) -> None:
        """
        Save hybrid model components to disk.
        
        Args:
            filepath: Base path to save the model components
        """
        import os
        import torch
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save NMF model
        nmf_path = f"{filepath}_nmf.pkl"
        if self.nmf_model:
            self.nmf_model.save(nmf_path)
        
        # Save Two-Tower model
        two_tower_path = f"{filepath}_two_tower.pt"
        if self.two_tower_model:
            torch.save(self.two_tower_model.state_dict(), two_tower_path)
        
        # Save Audio Feature Predictor
        audio_pred_path = f"{filepath}_audio_predictor.pt"
        if self.audio_predictor:
            self.audio_predictor.save(audio_pred_path)
        
        # Save hybrid model metadata and fusion layer
        hybrid_metadata = {
            'fusion_method': self.fusion_method,
            'cf_weight': self.cf_weight,
            'content_weight': self.content_weight,
            'fusion_layer_state': self.fusion_layer.state_dict() if self.fusion_layer else None
        }
        
        metadata_path = f"{filepath}_hybrid_metadata.pt"
        torch.save(hybrid_metadata, metadata_path)
        
        logger.info(f"Hybrid model saved to {filepath}")
    
    def load(self, filepath: str) -> 'HybridModel':
        """
        Load hybrid model components from disk.
        
        Args:
            filepath: Base path to load the model components from
            
        Returns:
            Self with loaded models
        """
        import torch
        
        # Load NMF model
        nmf_path = f"{filepath}_nmf.pkl"
        if os.path.exists(nmf_path) and self.nmf_model:
            self.nmf_model.load(nmf_path)
        
        # Load Two-Tower model
        two_tower_path = f"{filepath}_two_tower.pt"
        if os.path.exists(two_tower_path) and self.two_tower_model:
            self.two_tower_model.load_state_dict(torch.load(two_tower_path))
        
        # Load Audio Feature Predictor
        audio_pred_path = f"{filepath}_audio_predictor.pt"
        if os.path.exists(audio_pred_path) and self.audio_predictor:
            self.audio_predictor.load(audio_pred_path)
        
        # Load hybrid metadata and fusion layer
        metadata_path = f"{filepath}_hybrid_metadata.pt"
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
            
            self.fusion_method = metadata.get('fusion_method', self.fusion_method)
            self.cf_weight = metadata.get('cf_weight', self.cf_weight)
            self.content_weight = metadata.get('content_weight', self.content_weight)
            
            if metadata.get('fusion_layer_state') and self.fusion_layer:
                self.fusion_layer.load_state_dict(metadata['fusion_layer_state'])
        
        logger.info(f"Hybrid model loaded from {filepath}")
        return self 