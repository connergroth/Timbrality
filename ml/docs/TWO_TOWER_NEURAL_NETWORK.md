# Two-Tower Neural Network Architecture

This document describes the Two-Tower Neural Network implementation for content-based music recommendation, featuring separate user and item towers with confidence-aware audio feature integration.

## Overview

The Two-Tower architecture consists of two independent neural networks: a User Tower that encodes user preferences and behavior patterns, and an Item Tower that encodes track features. These towers generate embeddings that are combined to produce recommendation scores through dot product similarity.

## Architecture Components

### User Tower

The User Tower processes user-specific features to generate a dense representation of user preferences:

```python
class UserTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_genres: int,
        num_moods: int,
        user_activity_dim: int = 18,
        output_dim: int = 64
    ):
```

#### Input Features

**User Embeddings**

- Learnable user ID embeddings (dimension: 32)
- Captures individual user preference patterns
- Initialized with Xavier uniform distribution

**Genre Preference Vectors**

- Multi-hot encoded genre preferences (up to 5 genres per user)
- Embedded through learnable genre embedding layer
- Aggregated using mean pooling across preferred genres

**Mood Preference Vectors**

- Multi-hot encoded mood preferences (up to 3 moods per user)
- Embedded through learnable mood embedding layer
- Aggregated using mean pooling across preferred moods

**Session Activity Features**

- 18-dimensional normalized session-derived features:
  - `total_sessions`: Total listening sessions
  - `avg_session_length`: Average session duration in minutes
  - `discovery_rate_per_active_week`: New track discovery rate normalized by activity
  - `exploration_score`: Tendency to explore new content (0-1)
  - `novelty_seeking`: Preference for novel/rare tracks (0-1)
  - `repeat_rate`: Fraction of repeated track plays
  - `skip_rate`: Fraction of tracks skipped before completion
  - `genre_diversity`: Shannon entropy of genre distribution
  - `temporal_consistency`: Consistency of listening patterns over time
  - `mainstream_score`: Preference for mainstream vs niche content
  - `early_adopter_score`: Tendency to discover tracks early
  - `plays_per_day`: Average daily listening activity
  - `active_days`: Number of active listening days
  - `longest_streak_percentile`: Percentile rank of longest listening streak
  - `session_density`: Sessions per active day
  - `listening_intensity`: Average minutes listened per day
  - `user_type_explorer`: Binary indicator for explorer user type
  - `user_type_loyalist`: Binary indicator for loyalist user type

#### Network Architecture

```python
def forward(
    self,
    user_ids: torch.Tensor,
    user_genre_prefs: torch.Tensor,
    user_mood_prefs: torch.Tensor,
    user_activity: torch.Tensor
) -> torch.Tensor:
```

1. **Embedding Layer**: User IDs → 32-dimensional embeddings
2. **Genre Processing**: Multi-hot genre preferences → embedded and pooled
3. **Mood Processing**: Multi-hot mood preferences → embedded and pooled
4. **Activity Processing**: 18-dimensional activity features → 32-dimensional representation
5. **Concatenation**: All features concatenated (32 + 16 + 12 + 32 = 92 dimensions)
6. **Dense Layers**:
   - Linear(92, 128) → ReLU → Dropout(0.2)
   - Linear(128, 64) → ReLU
   - Linear(64, output_dim)
7. **Normalization**: L2 normalization of final embeddings

### Item Tower

The Item Tower processes track-specific features including confidence-aware audio features:

```python
class ItemTower(nn.Module):
    def __init__(
        self,
        text_embedding_dim: int = 384,
        num_artists: int = 50000,
        num_genres: int = 500,
        num_moods: int = 100,
        output_dim: int = 64
    ):
```

#### Input Features

**Text Embeddings**

- BERT-encoded track metadata (384 dimensions)
- Combines track title, artist name, album, genres, and moods
- Generated using pre-trained sentence transformer model

**Artist Embeddings**

- Learnable artist ID embeddings (dimension: 32)
- Captures artist-specific style and characteristics
- Shared across all tracks by the same artist

**Genre Embeddings**

- Multi-hot encoded track genres (up to 5 genres per track)
- Embedded through learnable genre embedding layer
- Aggregated using mean pooling

**Mood Embeddings**

- Multi-hot encoded track moods (up to 3 moods per track)
- Embedded through learnable mood embedding layer
- Aggregated using mean pooling

**Confidence-Aware Audio Features**

- 8 predicted/actual audio features: energy, valence, danceability, acousticness, instrumentalness, liveness, speechiness, tempo
- 8 corresponding confidence scores indicating prediction reliability
- Concatenated to form 16-dimensional input for audio processing layer
- Features are normalized to [0, 1] range

**Rating and Popularity Features**

- AOTY score (normalized 0-1)
- General popularity score (normalized 0-1)
- Number of AOTY ratings (normalized by 1000)
- Average user rating (normalized 0-1)
- Critic score (normalized 0-1)

#### Network Architecture

```python
def forward(
    self,
    track_texts: List[str],
    artist_ids: torch.Tensor,
    genre_ids: torch.Tensor,
    mood_ids: torch.Tensor,
    audio_features: torch.Tensor,
    audio_confidence: torch.Tensor,
    rating_features: torch.Tensor
) -> torch.Tensor:
```

1. **Text Encoding**: BERT embedding of concatenated track metadata
2. **Artist Embedding**: Artist IDs → 32-dimensional embeddings
3. **Genre Processing**: Multi-hot genre IDs → embedded and pooled
4. **Mood Processing**: Multi-hot mood IDs → embedded and pooled
5. **Audio Processing**: Concatenated audio features + confidence → 32-dimensional representation
6. **Rating Processing**: 5-dimensional rating/popularity features → 16-dimensional representation
7. **Concatenation**: All features concatenated (384 + 32 + 16 + 12 + 32 + 16 = 492 dimensions)
8. **Dense Layers**:
   - Linear(492, 256) → ReLU → Dropout(0.3)
   - Linear(256, 128) → ReLU → Dropout(0.2)
   - Linear(128, output_dim)
9. **Normalization**: L2 normalization of final embeddings

### Confidence-Aware Audio Features

A key innovation in this architecture is the integration of confidence scores with predicted audio features:

#### Motivation

- Audio features may be predicted using the AudioFeaturePredictor when Spotify data is unavailable
- Predicted features have varying reliability depending on available metadata
- Confidence scores allow the model to learn appropriate weighting of uncertain features

#### Implementation

```python
# Concatenate audio features with their confidence scores
audio_with_confidence = torch.cat([audio_features, audio_confidence], dim=1)  # [batch_size, 16]
audio_emb = self.audio_linear(audio_with_confidence)  # [batch_size, 32]
```

#### Confidence Calculation

Confidence scores are calculated based on:

- **Source reliability**: 0.9 for actual Spotify features, 0.6 for predicted features
- **Metadata completeness**: Multiplied by completeness score (0-1) based on available title, artist, album, genres, moods, duration, and release year
- **Feature-specific confidence**: Each of the 8 audio features has its own confidence score

### Two-Tower Model Integration

The main TwoTowerModel class coordinates both towers:

```python
class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_artists: int,
        num_genres: int,
        num_moods: int,
        text_embedding_dim: int = 384,
        user_activity_dim: int = 18,
        output_dim: int = 64
    ):
```

#### Forward Pass

```python
def forward(
    self,
    user_ids: torch.Tensor,
    user_genre_prefs: torch.Tensor,
    user_mood_prefs: torch.Tensor,
    user_activity: torch.Tensor,
    track_texts: List[str],
    artist_ids: torch.Tensor,
    item_genre_ids: torch.Tensor,
    item_mood_ids: torch.Tensor,
    audio_features: torch.Tensor,
    audio_confidence: torch.Tensor,
    rating_features: torch.Tensor
) -> Dict[str, torch.Tensor]:
```

The model returns:

- `user_embeddings`: User tower output embeddings
- `item_embeddings`: Item tower output embeddings
- `similarity_scores`: Dot product similarities between user and item embeddings

#### Inference Methods

**Batch Prediction**

```python
outputs = model(batch_data)
scores = outputs['similarity_scores']  # [batch_size]
```

**Individual Item Embedding**

```python
item_embedding = model.get_item_embedding(
    track_text="Artist - Track Title",
    artist_id=123,
    genre_ids=torch.tensor([1, 5, 12]),
    mood_ids=torch.tensor([2, 8]),
    audio_feats=torch.tensor([0.7, 0.6, ...]),
    audio_conf=torch.tensor([0.8, 0.9, ...]),
    rating_feats=torch.tensor([0.85, 0.6, ...])
)
```

## Training Process

### Data Preparation

**User Data Preprocessing**

1. Extract user interaction history and compute session features
2. Identify user genre and mood preferences from listening history
3. Normalize session-derived activity features
4. Create user type classifications (explorer, loyalist, etc.)

**Item Data Preprocessing**

1. Generate BERT embeddings for track metadata
2. Predict missing audio features using AudioFeaturePredictor
3. Calculate confidence scores for all audio features
4. Normalize rating and popularity features
5. Create genre and mood multi-hot encodings

### Training Configuration

```python
{
    "learning_rate": 0.001,
    "batch_size": 512,
    "epochs": 50,
    "optimizer": "Adam",
    "weight_decay": 0.0001,
    "dropout_rates": {
        "user_tower": 0.2,
        "item_tower_1": 0.3,
        "item_tower_2": 0.2
    }
}
```

### Loss Function

The model uses a combination of losses:

**Primary Loss**: Cosine Embedding Loss

```python
loss = nn.CosineEmbeddingLoss(margin=0.1)
# Positive pairs: user-track interactions
# Negative pairs: random user-track pairs
```

**Regularization**: L2 regularization on embedding layers to prevent overfitting

### Training Loop

```python
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(**batch)

        # Calculate loss
        user_embeddings = outputs['user_embeddings']
        item_embeddings = outputs['item_embeddings']
        labels = batch['labels']  # 1 for positive, -1 for negative pairs

        loss = cosine_embedding_loss(user_embeddings, item_embeddings, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### Evaluation Metrics

**Embedding Quality**

- Cosine similarity distribution between positive pairs
- Separation between positive and negative pair similarities
- Embedding norm stability across training

**Recommendation Quality**

- Precision@K, Recall@K for various K values
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)

**Diversity Metrics**

- Intra-list diversity of recommended items
- Coverage of genre and mood space
- Artist and album diversity in recommendations

## Production Deployment

### Model Serving

**Batch Inference**

```python
# Pre-compute item embeddings for all tracks
item_embeddings = {}
model.eval()
with torch.no_grad():
    for batch in item_dataloader:
        batch_embeddings = model.item_tower(**batch)
        for i, track_id in enumerate(batch['track_ids']):
            item_embeddings[track_id] = batch_embeddings[i].cpu().numpy()
```

**Real-time User Embedding**

```python
def get_user_recommendations(user_id, num_recommendations=20):
    # Compute user embedding
    user_data = prepare_user_data(user_id)
    user_embedding = model.user_tower(**user_data)

    # Compute similarities with all items
    similarities = cosine_similarity(
        user_embedding.cpu().numpy(),
        item_embeddings_matrix
    )

    # Return top K recommendations
    top_indices = np.argsort(similarities)[::-1][:num_recommendations]
    return [track_ids[i] for i in top_indices]
```

### Performance Optimization

**Memory Management**

- Pre-compute and cache item embeddings
- Use float16 precision for inference to reduce memory usage
- Implement batch processing for large-scale inference

**Computational Efficiency**

- Use approximate nearest neighbor search (FAISS) for large catalogs
- Implement embedding quantization for storage efficiency
- Cache user embeddings for active users

### Monitoring and Maintenance

**Model Performance Monitoring**

- Track embedding drift over time
- Monitor prediction latency and throughput
- Validate embedding quality with periodic offline evaluation

**Data Quality Monitoring**

- Monitor audio feature prediction confidence distributions
- Track metadata completeness trends
- Validate session feature normalization stability

**Retraining Schedule**

- Weekly incremental training with new user interaction data
- Monthly full retraining with updated audio feature predictions
- Quarterly architecture updates based on performance analysis

## Integration with Hybrid System

The Two-Tower model integrates with the broader hybrid recommendation system:

### Input to Hybrid Model

```python
# Get content-based prediction from Two-Tower
content_score, content_confidence = two_tower_model.predict(
    user_data=user_features,
    item_data=item_features,
    audio_features=predicted_audio_features,
    audio_confidence=audio_confidence_scores
)
```

### Confidence Propagation

The Two-Tower model propagates confidence through:

1. Audio feature confidence affects item embedding quality
2. Session feature completeness affects user embedding confidence
3. Overall content confidence is calculated based on feature completeness

### Meta-Learning Integration

The Two-Tower embeddings serve as inputs to the meta-learning fusion network:

- User embeddings provide context for adaptive weight calculation
- Item embeddings inform uncertainty estimation
- Content confidence scores guide fusion decisions

## Advanced Features

### Cold Start Handling

**New Users**

- Use demographic information and onboarding preferences
- Initialize session features with population averages
- Gradually personalize as interaction data accumulates

**New Items**

- Rely on content features and predicted audio characteristics
- Use artist similarity for unknown artists
- Bootstrap with genre and mood similarities

### Temporal Adaptation

**Session Context**

- Incorporate recent listening session information
- Weight recent preferences higher in user embedding calculation
- Adapt to intra-session preference evolution

**Trend Adaptation**

- Update genre and mood embeddings based on temporal popularity shifts
- Incorporate release date information in item representations
- Adapt to seasonal listening pattern changes

This Two-Tower Neural Network architecture provides a robust foundation for content-based music recommendation with sophisticated handling of uncertainty and user behavior patterns.





