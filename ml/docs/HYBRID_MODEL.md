# Hybrid Recommendation Model

This document describes the Hybrid Recommendation Model, the central orchestrating component that combines collaborative filtering (NMF), content-based filtering (Two-Tower), audio feature prediction, and meta-learning fusion to deliver state-of-the-art music recommendations with sophisticated uncertainty quantification and context adaptation.

## Overview

The Hybrid Model represents the culmination of modern recommendation system design, integrating multiple complementary approaches to overcome the limitations of individual methods. Rather than relying on a single recommendation strategy, this system dynamically combines collaborative filtering, content-based filtering, and predictive modeling through learned fusion mechanisms that adapt to user context and prediction uncertainty.

## System Architecture

### Core Components Integration

The Hybrid Model orchestrates five major components:

```python
class HybridModel(nn.Module):
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
```

#### Component Relationships

**Collaborative Filtering Foundation**

- NMF-based matrix factorization for user-item interactions
- Captures collective wisdom and behavioral patterns
- Provides strong signal for users with substantial interaction history
- Handles implicit feedback through confidence weighting

**Content-Based Enhancement**

- Two-Tower neural network for deep feature understanding
- Processes rich metadata, genres, moods, and audio characteristics
- Enables cold-start recommendations for new items and users
- Incorporates predicted audio features when Spotify data unavailable

**Audio Feature Synthesis**

- AudioFeaturePredictor for missing audio characteristics
- Prevents data leakage by using only intrinsic track properties
- Provides confidence scores for predicted features
- Enables content-based filtering across diverse music catalogs

**Meta-Learning Fusion**

- Adaptive combination of CF and content signals
- Context-aware weight generation based on user behavior and item uncertainty
- Learned confidence calibration for improved reliability
- Dynamic adaptation to recommendation scenarios

**Confidence Calibration**

- Optional CalibrationHead for learned confidence estimation
- Transforms heuristic confidence into calibrated uncertainty estimates
- Enables principled decision-making about recommendation trust
- Supports ensemble and adaptive recalibration strategies

### Data Flow Architecture

```python
def predict(
    self,
    user_data: Dict[str, Any],
    item_data: Dict[str, Any],
    session_features: Optional[Dict[str, Any]] = None,
    use_audio_prediction: bool = True
) -> Dict[str, float]:
```

**Phase 1: Feature Preparation**

1. Extract user preferences, interaction history, and session context
2. Prepare item metadata, ensuring no popularity leakage for audio prediction
3. Generate or retrieve audio features with appropriate confidence scores
4. Normalize and encode all features for neural network consumption

**Phase 2: Base Prediction Generation**

```python
# Collaborative Filtering
cf_score, cf_confidence = self._get_cf_prediction(user_data, item_data)

# Content-Based Filtering
content_score, content_confidence = self._get_content_prediction(
    user_data, item_data, audio_features
)
```

**Phase 3: Fusion and Calibration**

```python
if self.enable_meta_learning and session_features:
    final_score, fusion_details = self._meta_learning_fusion(
        cf_score, content_score, cf_confidence, content_confidence,
        user_data, item_data, session_features, audio_features
    )
else:
    final_score = self._fuse_predictions(
        cf_score, content_score, cf_confidence, content_confidence, user_data
    )
```

**Phase 4: Explanation Generation**

```python
explanation = self.explain_recommendation(user_data, item_data, prediction_result)
```

## Collaborative Filtering Pipeline

### NMF Integration

The hybrid system leverages Non-negative Matrix Factorization for collaborative filtering:

```python
def _get_cf_prediction(self, user_data: Dict, item_data: Dict) -> Tuple[float, float]:
    """Generate collaborative filtering prediction with confidence estimation."""

    user_id = user_data.get('user_id')
    item_id = item_data.get('track_id')

    try:
        # Get NMF prediction
        cf_score = self.nmf_model.predict(user_id, item_id)

        # Calculate confidence based on interaction history
        cf_confidence = self._calculate_cf_confidence(user_data, item_data)

        return cf_score, cf_confidence

    except Exception as e:
        logger.warning(f"CF prediction failed: {e}")
        return 0.0, 0.0
```

#### Confidence Estimation for CF

```python
def _calculate_cf_confidence(self, user_data: Dict, item_data: Dict) -> float:
    """
    Estimate confidence in collaborative filtering prediction.

    Factors:
    - User interaction history volume
    - Item interaction frequency
    - Similarity to training distribution
    - Temporal recency of interactions
    """

    confidence = 0.0

    # User history richness
    user_interactions = user_data.get('total_interactions', 0)
    if user_interactions >= 100:
        confidence += 0.4
    elif user_interactions >= 50:
        confidence += 0.3
    elif user_interactions >= 20:
        confidence += 0.2
    elif user_interactions >= 10:
        confidence += 0.1

    # Item popularity (not too niche, not too mainstream)
    item_interactions = item_data.get('total_user_interactions', 0)
    if 10 <= item_interactions <= 1000:
        confidence += 0.3
    elif 5 <= item_interactions <= 2000:
        confidence += 0.2
    elif item_interactions > 0:
        confidence += 0.1

    # Neighbor similarity quality
    neighbor_similarity = user_data.get('avg_neighbor_similarity', 0.0)
    confidence += min(neighbor_similarity * 0.3, 0.3)

    return min(confidence, 1.0)
```

### Cold Start Handling

```python
def _handle_cf_cold_start(self, user_data: Dict, item_data: Dict) -> Tuple[float, float]:
    """Handle cold start scenarios for collaborative filtering."""

    # New user with some genre preferences
    if user_data.get('total_interactions', 0) < 5:
        if user_data.get('preferred_genres'):
            # Use genre-based similarity to existing users
            genre_similarity_score = self._calculate_genre_based_cf_score(
                user_data, item_data
            )
            return genre_similarity_score, 0.3

    # New item with genre information
    if item_data.get('total_user_interactions', 0) < 3:
        if item_data.get('genres'):
            # Use item-item similarity within genres
            genre_item_score = self._calculate_item_genre_similarity_score(
                user_data, item_data
            )
            return genre_item_score, 0.2

    # Complete cold start
    return 0.0, 0.0
```

## Content-Based Filtering Pipeline

### Two-Tower Integration

The content-based component leverages the Two-Tower neural network:

```python
def _get_content_prediction(
    self,
    user_data: Dict,
    item_data: Dict,
    audio_features: Dict
) -> Tuple[float, float]:
    """Generate content-based prediction using Two-Tower model."""

    try:
        device = next(self.two_tower_model.parameters()).device

        # Prepare user features
        user_features = self._prepare_user_features(user_data, device)

        # Prepare item features with audio confidence
        item_features = self._prepare_item_features(item_data, audio_features, device)

        # Get Two-Tower prediction
        with torch.no_grad():
            outputs = self.two_tower_model(**user_features, **item_features)
            content_score = float(outputs['similarity_scores'][0])

        # Calculate content confidence
        content_confidence = self._calculate_content_confidence(item_data, audio_features)

        return content_score, content_confidence

    except Exception as e:
        logger.warning(f"Content prediction failed: {e}")
        return 0.0, 0.0
```

#### Feature Preparation

```python
def _prepare_user_features(self, user_data: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """Prepare user features for Two-Tower model."""

    # User ID embedding
    user_id = user_data.get('user_id', 0)
    user_ids = torch.tensor([user_id], device=device)

    # Genre preferences (multi-hot)
    preferred_genres = user_data.get('preferred_genres', [])
    genre_prefs = torch.zeros(5, device=device)  # Max 5 genres
    for i, genre in enumerate(preferred_genres[:5]):
        genre_prefs[i] = self.genre_to_idx.get(genre, 0)

    # Mood preferences (multi-hot)
    preferred_moods = user_data.get('preferred_moods', [])
    mood_prefs = torch.zeros(3, device=device)  # Max 3 moods
    for i, mood in enumerate(preferred_moods[:3]):
        mood_prefs[i] = self.mood_to_idx.get(mood, 0)

    # Session activity features (18 dimensions)
    activity_features = self._extract_session_features(user_data)
    user_activity = torch.tensor(activity_features, device=device)

    return {
        'user_ids': user_ids,
        'user_genre_prefs': genre_prefs.unsqueeze(0),
        'user_mood_prefs': mood_prefs.unsqueeze(0),
        'user_activity': user_activity.unsqueeze(0)
    }

def _prepare_item_features(
    self,
    item_data: Dict,
    audio_features: Dict,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Prepare item features for Two-Tower model."""

    # Track text representation
    track_text = self._create_track_text(item_data)
    track_texts = [track_text]

    # Artist ID
    artist_id = item_data.get('artist_id', 0)
    artist_ids = torch.tensor([artist_id], device=device)

    # Genre encoding
    item_genres = item_data.get('genres', [])
    genre_ids = torch.zeros(5, device=device)
    for i, genre in enumerate(item_genres[:5]):
        genre_ids[i] = self.genre_to_idx.get(genre, 0)

    # Mood encoding
    item_moods = item_data.get('moods', [])
    mood_ids = torch.zeros(3, device=device)
    for i, mood in enumerate(item_moods[:3]):
        mood_ids[i] = self.mood_to_idx.get(mood, 0)

    # Audio features with confidence
    audio_tensor = torch.tensor([
        audio_features.get('energy', 0.5),
        audio_features.get('valence', 0.5),
        audio_features.get('danceability', 0.5),
        audio_features.get('acousticness', 0.5),
        audio_features.get('instrumentalness', 0.5),
        audio_features.get('liveness', 0.5),
        audio_features.get('speechiness', 0.5),
        audio_features.get('tempo', 120) / 200.0
    ], device=device)

    # Calculate audio confidence
    is_predicted = item_data.get('audio_features_predicted', False)
    base_confidence = 0.6 if is_predicted else 0.9
    metadata_completeness = self._calculate_metadata_completeness(item_data)
    adjusted_confidence = base_confidence * metadata_completeness
    audio_confidence = torch.tensor([adjusted_confidence] * 8, device=device)

    # Rating features
    rating_tensor = torch.tensor([
        item_data.get('aoty_score', 50) / 100.0,
        item_data.get('popularity', 50) / 100.0,
        item_data.get('aoty_num_ratings', 0) / 1000.0,
        item_data.get('user_rating_avg', 3.5) / 5.0,
        item_data.get('critic_score', 75) / 100.0
    ], device=device)

    return {
        'track_texts': track_texts,
        'artist_ids': artist_ids,
        'item_genre_ids': genre_ids.unsqueeze(0),
        'item_mood_ids': mood_ids.unsqueeze(0),
        'audio_features': audio_tensor.unsqueeze(0),
        'audio_confidence': audio_confidence.unsqueeze(0),
        'rating_features': rating_tensor.unsqueeze(0)
    }
```

### Audio Feature Integration

```python
def _integrate_audio_features(
    self,
    item_data: Dict,
    use_audio_prediction: bool = True
) -> Dict[str, Any]:
    """Integrate actual or predicted audio features."""

    # Check if Spotify features are available
    if self._has_spotify_features(item_data):
        return {
            'energy': item_data.get('energy', 0.5),
            'valence': item_data.get('valence', 0.5),
            'danceability': item_data.get('danceability', 0.5),
            'acousticness': item_data.get('acousticness', 0.5),
            'instrumentalness': item_data.get('instrumentalness', 0.5),
            'liveness': item_data.get('liveness', 0.5),
            'speechiness': item_data.get('speechiness', 0.5),
            'tempo': item_data.get('tempo', 120),
            'predicted': False
        }

    # Use AudioFeaturePredictor if available and enabled
    if use_audio_prediction and self.audio_predictor:
        # Extract metadata for prediction (no popularity data!)
        metadata_features = self._extract_metadata_features(item_data)

        predicted_audio = self.audio_predictor.predict_audio_features(
            title=item_data.get('title', ''),
            artist=item_data.get('artist', ''),
            genres=item_data.get('genres', []),
            moods=item_data.get('moods', []),
            metadata=metadata_features
        )

        predicted_audio['predicted'] = True
        return predicted_audio

    # Fallback to defaults
    return {
        'energy': 0.5, 'valence': 0.5, 'danceability': 0.5,
        'acousticness': 0.5, 'instrumentalness': 0.5,
        'liveness': 0.5, 'speechiness': 0.5, 'tempo': 120,
        'predicted': False
    }
```

## Fusion Mechanisms

### Meta-Learning Fusion

The advanced fusion approach uses meta-learning to adaptively weight predictions:

```python
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
    """Meta-learning fusion with context adaptation."""

    device = next(self.meta_fusion.parameters()).device

    # Extract session context (18-dimensional)
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
        float(session_features.get('user_type', 'balanced') == 'explorer'),
        float(session_features.get('user_type', 'balanced') == 'loyalist')
    ], device=device).unsqueeze(0)

    # Extract user context
    user_context, _ = self.session_extractor(session_feature_tensor)

    # Extract item uncertainty context
    is_predicted = float(item_data.get('audio_features_predicted', False))
    metadata_completeness = self._calculate_metadata_completeness(item_data)

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
        audio_confidence=audio_conf_tensor,  # Simplified for demo
        metadata_completeness=torch.tensor([[metadata_completeness]], device=device),
        is_predicted=torch.tensor([[is_predicted]], device=device)
    )

    # Create embeddings (simplified - would come from actual models)
    cf_embedding = torch.randn(1, 64, device=device)
    content_embedding = torch.randn(1, 64, device=device)

    # Apply meta-learning fusion
    with torch.no_grad():
        fusion_output = self.meta_fusion(
            cf_score=torch.tensor([cf_score], device=device),
            content_score=torch.tensor([content_score], device=device),
            cf_embedding=cf_embedding,
            content_embedding=content_embedding,
            user_session_context=user_context,
            item_uncertainty_context=item_uncertainty_context
        )

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
```

### Traditional Fusion Methods

For fallback scenarios or when meta-learning is disabled:

```python
def _fuse_predictions(
    self,
    cf_score: float,
    content_score: float,
    cf_confidence: float,
    content_confidence: float,
    user_data: Dict
) -> float:
    """Traditional fusion methods for prediction combination."""

    if self.fusion_method == "weighted_sum":
        return self.cf_weight * cf_score + self.content_weight * content_score

    elif self.fusion_method == "confidence_weighted":
        total_confidence = cf_confidence + content_confidence
        if total_confidence > 0:
            cf_weight = cf_confidence / total_confidence
            content_weight = content_confidence / total_confidence
            return cf_weight * cf_score + content_weight * content_score
        else:
            return (cf_score + content_score) / 2

    elif self.fusion_method == "learned_weights" and self.fusion_layer:
        # Use learned fusion network
        device = next(self.fusion_layer.parameters()).device
        fusion_input = torch.tensor([cf_score, content_score,
                                   (cf_confidence + content_confidence) / 2],
                                  device=device).unsqueeze(0)

        with torch.no_grad():
            fused_score = self.fusion_layer(fusion_input)

        return float(fused_score[0])

    elif self.fusion_method == "adaptive" and self.fusion_layer:
        # Use adaptive fusion with user embedding
        device = next(self.fusion_layer.parameters()).device
        user_embedding = torch.randn(64, device=device)  # Would come from user model
        fusion_input = torch.cat([
            user_embedding,
            torch.tensor([cf_score, content_score,
                         (cf_confidence + content_confidence) / 2], device=device)
        ]).unsqueeze(0)

        with torch.no_grad():
            weights = self.fusion_layer(fusion_input)
            cf_weight, content_weight = weights[0]

        return float(cf_weight * cf_score + content_weight * content_score)

    else:
        # Default weighted average
        return self.cf_weight * cf_score + self.content_weight * content_score
```

## Recommendation Generation

### Candidate Scoring

```python
def get_recommendations(
    self,
    user_data: Dict[str, Any],
    candidate_items: List[Dict[str, Any]],
    session_features: Optional[Dict[str, Any]] = None,
    top_k: int = 20,
    use_audio_prediction: bool = True,
    include_explanations: bool = True
) -> List[Dict[str, Any]]:
    """Generate top-k recommendations with scoring and explanations."""

    recommendations = []

    for item in candidate_items:
        try:
            # Generate prediction
            prediction = self.predict(
                user_data,
                item,
                session_features,
                use_audio_prediction
            )

            # Create recommendation object
            item_with_score = item.copy()
            item_with_score.update(prediction)

            # Add explanation if requested
            if include_explanations:
                explanation = self.explain_recommendation(user_data, item, prediction)
                item_with_score['explanation'] = explanation

            recommendations.append(item_with_score)

        except Exception as e:
            logger.warning(f"Failed to score item {item.get('track_id')}: {e}")
            continue

    # Sort by final score
    recommendations.sort(key=lambda x: x.get('final_score', 0), reverse=True)

    # Apply diversity and quality filters
    filtered_recommendations = self._apply_recommendation_filters(
        recommendations, user_data, top_k
    )

    return filtered_recommendations[:top_k]
```

### Recommendation Filtering

```python
def _apply_recommendation_filters(
    self,
    recommendations: List[Dict[str, Any]],
    user_data: Dict[str, Any],
    target_count: int
) -> List[Dict[str, Any]]:
    """Apply quality and diversity filters to recommendations."""

    filtered = []
    artist_counts = {}
    genre_counts = {}

    for rec in recommendations:
        # Quality thresholds
        if rec.get('final_score', 0) < 0.1:
            continue

        if rec.get('cf_confidence', 0) + rec.get('content_confidence', 0) < 0.2:
            continue

        # Artist diversity (max 3 per artist)
        artist = rec.get('artist', 'Unknown')
        if artist_counts.get(artist, 0) >= 3:
            continue

        # Genre diversity (ensure variety)
        item_genres = rec.get('genres', [])
        skip_item = False
        for genre in item_genres:
            if genre_counts.get(genre, 0) >= target_count // 3:
                skip_item = True
                break

        if skip_item:
            continue

        # Update counters
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        for genre in item_genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        filtered.append(rec)

        # Stop when we have enough diverse recommendations
        if len(filtered) >= target_count * 2:  # Get extra for final selection
            break

    return filtered
```

## Explanation Generation

### Multi-Factor Explanations

```python
def explain_recommendation(
    self,
    user_data: Dict[str, Any],
    item_data: Dict[str, Any],
    prediction_result: Dict[str, Any]
) -> str:
    """Generate human-readable explanation for recommendation."""

    explanation_parts = []

    # Determine primary recommendation method
    method = prediction_result.get('method', 'hybrid')
    cf_score = prediction_result.get('cf_score', 0)
    content_score = prediction_result.get('content_score', 0)

    if method == "collaborative" or (method == "hybrid" and cf_score > content_score):
        cf_explanation = self._explain_collaborative_factors(user_data, item_data)
        explanation_parts.append(cf_explanation)

    if method == "content" or (method == "hybrid" and content_score > 0.1):
        content_explanation = self._explain_content_factors(user_data, item_data)
        explanation_parts.append(content_explanation)

    # Add meta-learning insights if available
    if prediction_result.get('fusion_method') == 'meta_learning':
        meta_explanation = self._explain_meta_learning_factors(prediction_result)
        explanation_parts.append(meta_explanation)

    # Combine explanations
    if explanation_parts:
        explanation = self._generate_explanation_summary(explanation_parts, prediction_result)
    else:
        explanation = "Recommended based on overall music preferences"

    return explanation
```

#### Collaborative Filtering Explanations

```python
def _explain_collaborative_factors(self, user_data: Dict, item_data: Dict) -> str:
    """Generate explanation for collaborative filtering component."""
    explanations = []

    # Similar users factor
    similar_users = user_data.get('top_similar_users', [])
    if similar_users:
        user_count = min(3, len(similar_users))
        explanations.append(f"users with similar taste also liked this track")

    # Genre overlap with user history
    user_top_genres = user_data.get('top_genres', [])
    item_genres = item_data.get('genres', [])
    common_genres = set(user_top_genres[:5]).intersection(set(item_genres))

    if common_genres:
        genre_names = list(common_genres)[:2]
        explanations.append(f"matches your {', '.join(genre_names)} listening history")

    # Artist familiarity
    artist = item_data.get('artist', '')
    user_artists = user_data.get('top_artists', [])
    if artist in user_artists[:20]:
        explanations.append(f"you've enjoyed other {artist} tracks")

    # Temporal patterns
    user_listening_time = user_data.get('primary_listening_hours', [])
    current_hour = user_data.get('current_hour', 12)
    if current_hour in user_listening_time:
        explanations.append("fits your current listening pattern")

    return ", ".join(explanations) if explanations else "similar users liked this track"
```

#### Content-Based Explanations

```python
def _explain_content_factors(self, user_data: Dict, item_data: Dict) -> str:
    """Generate explanation for content-based component."""
    explanations = []

    # Genre preferences
    user_genres = set(user_data.get('preferred_genres', []))
    item_genres = set(item_data.get('genres', []))
    common_genres = user_genres.intersection(item_genres)

    if common_genres:
        genre_names = list(common_genres)[:3]
        explanations.append(f"you like {', '.join(genre_names)} music")

    # Mood preferences
    user_moods = set(user_data.get('preferred_moods', []))
    item_moods = set(item_data.get('moods', []))
    common_moods = user_moods.intersection(item_moods)

    if common_moods:
        mood_names = list(common_moods)[:2]
        explanations.append(f"matches your preference for {', '.join(mood_names)} vibes")

    # Artist familiarity
    if item_data.get('artist') in user_data.get('preferred_artists', []):
        explanations.append(f"you've enjoyed {item_data['artist']} before")

    # Audio characteristics (if predicted)
    if item_data.get('audio_features_predicted'):
        explanations.append("predicted audio characteristics match your taste")

    # Critical acclaim (with minimum ratings check)
    aoty_score = item_data.get('aoty_score', 50)
    aoty_num_ratings = item_data.get('aoty_num_ratings', 0)
    if aoty_score > 80 and aoty_num_ratings >= self.min_ratings_for_acclaim:
        if aoty_num_ratings >= self.high_acclaim_threshold:
            explanations.append("widely acclaimed track with exceptional reviews")
        elif aoty_num_ratings >= (self.min_ratings_for_acclaim * 2):
            explanations.append("critically acclaimed track with strong consensus")
        else:
            explanations.append("critically acclaimed track")

    return ", ".join(explanations) if explanations else "content features match your preferences"
```

#### Meta-Learning Explanations

```python
def _explain_meta_learning_factors(self, prediction_result: Dict) -> str:
    """Generate explanation for meta-learning fusion decisions."""

    cf_weight = prediction_result.get('meta_cf_weight', 0.5)
    content_weight = prediction_result.get('meta_content_weight', 0.5)
    uncertainty = prediction_result.get('uncertainty_estimate', [0.5])[0]

    explanations = []

    # Explain fusion weighting
    if cf_weight > 0.7:
        explanations.append("based primarily on similar user preferences")
    elif content_weight > 0.7:
        explanations.append("based primarily on track characteristics")
    else:
        explanations.append("balanced recommendation considering multiple factors")

    # Explain confidence level
    avg_confidence = (prediction_result.get('meta_cf_confidence', 0.5) +
                     prediction_result.get('meta_content_confidence', 0.5)) / 2

    if avg_confidence > 0.8:
        explanations.append("high confidence recommendation")
    elif avg_confidence < 0.4:
        explanations.append("exploratory recommendation")

    # Explain uncertainty
    if uncertainty < 0.2:
        explanations.append("well-established match")
    elif uncertainty > 0.7:
        explanations.append("discovery recommendation")

    return ", ".join(explanations)
```

### Explanation Summary Generation

```python
def _generate_explanation_summary(
    self,
    explanation_parts: List[str],
    prediction_result: Dict[str, Any]
) -> str:
    """Generate coherent explanation summary."""

    # Filter and clean explanation parts
    clean_parts = [part.strip() for part in explanation_parts if part.strip()]

    if not clean_parts:
        return "Recommended based on your music preferences"

    # Create summary based on confidence and method
    confidence_level = prediction_result.get('final_confidence', 0.5)
    method = prediction_result.get('method', 'hybrid')

    if confidence_level > 0.8:
        prefix = "Highly recommended because "
    elif confidence_level > 0.6:
        prefix = "Recommended because "
    elif confidence_level > 0.4:
        prefix = "Suggested because "
    else:
        prefix = "You might like this because "

    # Combine explanation parts intelligently
    if len(clean_parts) == 1:
        summary = prefix + clean_parts[0]
    elif len(clean_parts) == 2:
        summary = prefix + clean_parts[0] + " and " + clean_parts[1]
    else:
        main_explanations = clean_parts[:2]
        summary = prefix + ", ".join(main_explanations)

        if len(clean_parts) > 2:
            summary += f", among other factors"

    # Add method-specific context
    if method == "meta_hybrid":
        summary += " (adaptive recommendation)"

    return summary + "."
```

## Model Persistence and Loading

### Comprehensive Model Saving

```python
def save(self, model_path: str) -> None:
    """Save complete hybrid model with all components."""

    import os
    os.makedirs(model_path, exist_ok=True)

    # Save model state dict
    torch.save(self.state_dict(), os.path.join(model_path, "hybrid_model.pth"))

    # Save individual components
    if self.nmf_model:
        self.nmf_model.save(os.path.join(model_path, "nmf_model"))

    if self.two_tower_model:
        torch.save(
            self.two_tower_model.state_dict(),
            os.path.join(model_path, "two_tower_model.pth")
        )

    if self.audio_predictor:
        torch.save(
            self.audio_predictor.state_dict(),
            os.path.join(model_path, "audio_predictor.pth")
        )

    # Save meta-learning components if present
    if hasattr(self, 'meta_fusion') and self.meta_fusion:
        torch.save(
            self.meta_fusion.state_dict(),
            os.path.join(model_path, "meta_fusion.pth")
        )
        torch.save(
            self.session_extractor.state_dict(),
            os.path.join(model_path, "session_extractor.pth")
        )
        torch.save(
            self.uncertainty_extractor.state_dict(),
            os.path.join(model_path, "uncertainty_extractor.pth")
        )

    # Save calibration head if present
    if hasattr(self, 'calibration_head') and self.calibration_head:
        torch.save(
            self.calibration_head.state_dict(),
            os.path.join(model_path, "calibration_head.pth")
        )

    # Save configuration and mappings
    config = {
        'fusion_method': self.fusion_method,
        'cf_weight': self.cf_weight,
        'content_weight': self.content_weight,
        'enable_meta_learning': self.enable_meta_learning,
        'min_ratings_for_acclaim': self.min_ratings_for_acclaim,
        'high_acclaim_threshold': self.high_acclaim_threshold,
        'genre_to_idx': getattr(self, 'genre_to_idx', {}),
        'mood_to_idx': getattr(self, 'mood_to_idx', {}),
    }

    with open(os.path.join(model_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Hybrid model saved to {model_path}")
```

### Model Loading and Restoration

```python
@classmethod
def load(
    cls,
    model_path: str,
    nmf_model: NMFModel,
    two_tower_model: TwoTowerModel,
    audio_predictor: Optional[AudioFeaturePredictor] = None,
    device: torch.device = None
) -> 'HybridModel':
    """Load complete hybrid model from saved components."""

    import json

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model instance
    model = cls(
        nmf_model=nmf_model,
        two_tower_model=two_tower_model,
        audio_predictor=audio_predictor,
        **{k: v for k, v in config.items() if k in [
            'fusion_method', 'cf_weight', 'content_weight',
            'enable_meta_learning', 'min_ratings_for_acclaim',
            'high_acclaim_threshold'
        ]}
    )

    # Load model state
    model_state_path = os.path.join(model_path, "hybrid_model.pth")
    if os.path.exists(model_state_path):
        model.load_state_dict(torch.load(model_state_path, map_location=device))

    # Load individual components
    two_tower_path = os.path.join(model_path, "two_tower_model.pth")
    if os.path.exists(two_tower_path):
        model.two_tower_model.load_state_dict(
            torch.load(two_tower_path, map_location=device)
        )

    if audio_predictor:
        audio_pred_path = os.path.join(model_path, "audio_predictor.pth")
        if os.path.exists(audio_pred_path):
            model.audio_predictor.load_state_dict(
                torch.load(audio_pred_path, map_location=device)
            )

    # Load meta-learning components
    meta_fusion_path = os.path.join(model_path, "meta_fusion.pth")
    if os.path.exists(meta_fusion_path) and hasattr(model, 'meta_fusion'):
        model.meta_fusion.load_state_dict(
            torch.load(meta_fusion_path, map_location=device)
        )

        session_ext_path = os.path.join(model_path, "session_extractor.pth")
        if os.path.exists(session_ext_path):
            model.session_extractor.load_state_dict(
                torch.load(session_ext_path, map_location=device)
            )

        uncertainty_ext_path = os.path.join(model_path, "uncertainty_extractor.pth")
        if os.path.exists(uncertainty_ext_path):
            model.uncertainty_extractor.load_state_dict(
                torch.load(uncertainty_ext_path, map_location=device)
            )

    # Load calibration head
    calibration_path = os.path.join(model_path, "calibration_head.pth")
    if os.path.exists(calibration_path):
        # Would load calibration head if implemented
        pass

    # Restore mappings
    model.genre_to_idx = config.get('genre_to_idx', {})
    model.mood_to_idx = config.get('mood_to_idx', {})

    model.to(device)
    model.eval()

    logger.info(f"Hybrid model loaded from {model_path}")
    return model
```

## Production Deployment

### Batch Recommendation Generation

```python
def generate_batch_recommendations(
    self,
    users_data: List[Dict[str, Any]],
    candidate_pool: List[Dict[str, Any]],
    batch_size: int = 100,
    top_k: int = 20
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate recommendations for multiple users efficiently."""

    all_recommendations = {}

    # Process users in batches
    for i in range(0, len(users_data), batch_size):
        batch_users = users_data[i:i + batch_size]

        # Generate recommendations for batch
        for user_data in batch_users:
            user_id = user_data['user_id']

            # Filter candidates based on user history
            user_candidates = self._filter_candidates_for_user(
                candidate_pool, user_data
            )

            # Generate recommendations
            recommendations = self.get_recommendations(
                user_data=user_data,
                candidate_items=user_candidates,
                top_k=top_k,
                include_explanations=False  # Skip for batch processing
            )

            all_recommendations[user_id] = recommendations

    return all_recommendations
```

### Real-time Recommendation API

```python
def get_real_time_recommendations(
    self,
    user_id: str,
    context: Dict[str, Any],
    num_recommendations: int = 10,
    diversity_factor: float = 0.3
) -> Dict[str, Any]:
    """Generate real-time recommendations with context awareness."""

    # Load user data with recent activity
    user_data = self._load_user_data_with_context(user_id, context)

    # Extract session features from recent activity
    session_features = self._extract_real_time_session_features(user_data, context)

    # Get candidate pool (cached or dynamically generated)
    candidates = self._get_candidate_pool(user_id, context)

    # Generate recommendations
    recommendations = self.get_recommendations(
        user_data=user_data,
        candidate_items=candidates,
        session_features=session_features,
        top_k=num_recommendations * 3,  # Generate extra for diversity filtering
        use_audio_prediction=True,
        include_explanations=True
    )

    # Apply diversity re-ranking
    diversified_recs = self._apply_diversity_reranking(
        recommendations, diversity_factor, num_recommendations
    )

    # Prepare response
    response = {
        'recommendations': diversified_recs[:num_recommendations],
        'metadata': {
            'user_id': user_id,
            'context': context,
            'session_features': session_features,
            'generation_method': 'hybrid_meta_learning',
            'diversity_factor': diversity_factor,
            'total_candidates_considered': len(candidates)
        }
    }

    return response
```

### Performance Monitoring

```python
def log_recommendation_metrics(
    self,
    user_id: str,
    recommendations: List[Dict[str, Any]],
    context: Dict[str, Any],
    response_time: float
) -> None:
    """Log metrics for recommendation performance monitoring."""

    metrics = {
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'num_recommendations': len(recommendations),
        'response_time_ms': response_time * 1000,
        'context': context
    }

    # Calculate quality metrics
    if recommendations:
        scores = [rec.get('final_score', 0) for rec in recommendations]
        confidences = [
            (rec.get('cf_confidence', 0) + rec.get('content_confidence', 0)) / 2
            for rec in recommendations
        ]

        metrics.update({
            'avg_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'avg_confidence': np.mean(confidences),
            'score_variance': np.var(scores)
        })

        # Diversity metrics
        artists = [rec.get('artist', 'Unknown') for rec in recommendations]
        genres = []
        for rec in recommendations:
            genres.extend(rec.get('genres', []))

        metrics.update({
            'unique_artists': len(set(artists)),
            'unique_genres': len(set(genres)),
            'artist_diversity': len(set(artists)) / len(recommendations),
            'repeat_artist_rate': 1 - (len(set(artists)) / len(recommendations))
        })

        # Method distribution
        methods = [rec.get('method', 'unknown') for rec in recommendations]
        method_counts = {method: methods.count(method) for method in set(methods)}
        metrics['method_distribution'] = method_counts

    # Log metrics (would integrate with monitoring system)
    logger.info(f"Recommendation metrics: {json.dumps(metrics)}")
```

The Hybrid Recommendation Model represents the state-of-the-art in music recommendation systems, combining the strengths of collaborative filtering, content-based filtering, and meta-learning approaches to deliver personalized, context-aware, and explainable recommendations that adapt to user behavior patterns and handle uncertainty gracefully across diverse recommendation scenarios.





