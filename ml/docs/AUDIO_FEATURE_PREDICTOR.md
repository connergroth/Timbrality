# Audio Feature Predictor Neural Network

This document describes the Audio Feature Predictor, a specialized neural network designed to predict Spotify-like audio features from available text metadata and track information when direct Spotify audio features are unavailable.

## Overview

The Audio Feature Predictor addresses a critical challenge in music recommendation systems: the absence of rich audio features for tracks not available in Spotify's catalog. By training on datasets like the Million Song Dataset, this model learns to predict eight key audio characteristics that capture the sonic qualities of music tracks.

## Problem Statement

### Missing Audio Features Challenge

Modern music recommendation systems rely heavily on audio features to understand track characteristics:

- Energy, valence, danceability, acousticness, instrumentalness, liveness, speechiness, and tempo
- These features are crucial for content-based filtering and similarity calculations
- Spotify provides these features, but only for tracks in their catalog
- Many tracks from specialized sources (AOTY, Last.fm, independent releases) lack this data

### Data Leakage Prevention

A critical design constraint is preventing data leakage during training:

- Cannot use popularity metrics, user ratings, or play counts as inputs
- Must predict audio features purely from intrinsic track properties
- Ensures the model learns sonic patterns rather than popularity correlations

## Architecture

### Model Structure

```python
class AudioFeaturePredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50000,
        num_genres: int = 500,
        num_moods: int = 100,
        embedding_dim: int = 384,
        hidden_dims: List[int] = [512, 256]
    ):
```

The architecture consists of four main components:

#### Text Embedding Branch

- BERT-based embeddings for track title and artist name
- Vocabulary size: 50,000 tokens
- Embedding dimension: 384 (matching sentence-BERT)
- Processes concatenated text: "{artist} - {title}"

#### Genre Embedding Branch

- Learnable embeddings for genre categories
- Multi-hot encoding supporting up to 5 genres per track
- Genre embedding dimension: 64
- Mean pooling aggregation across active genres

#### Mood Embedding Branch

- Learnable embeddings for mood/tag categories
- Multi-hot encoding supporting up to 3 moods per track
- Mood embedding dimension: 32
- Mean pooling aggregation across active moods

#### Metadata Processing Branch

- Linear transformation of numerical metadata features
- Input features (6 dimensions):
  - `duration_ms`: Track duration in milliseconds
  - `release_year`: Year of release
  - `explicit`: Binary indicator for explicit content
  - `track_number`: Position in album
  - `album_total_tracks`: Total tracks in album
  - `decade`: Derived decade bucket ((year // 10) \* 10)

### Feature Fusion Network

```python
def forward(self, text_embeddings, genre_embeddings, mood_embeddings, metadata_features):
    # Concatenate all feature branches
    combined = torch.cat([
        text_embeddings,      # 384 dimensions
        genre_embeddings,     # 64 dimensions
        mood_embeddings,      # 32 dimensions
        metadata_features     # 32 dimensions (after linear transform)
    ], dim=1)  # Total: 512 dimensions

    # Deep fusion network
    x = self.fusion_layers(combined)  # 512 -> 256 -> 128

    # Separate prediction heads for each audio feature
    predictions = {
        'energy': torch.sigmoid(self.energy_head(x)),
        'valence': torch.sigmoid(self.valence_head(x)),
        'danceability': torch.sigmoid(self.danceability_head(x)),
        'acousticness': torch.sigmoid(self.acousticness_head(x)),
        'instrumentalness': torch.sigmoid(self.instrumentalness_head(x)),
        'liveness': torch.sigmoid(self.liveness_head(x)),
        'speechiness': torch.sigmoid(self.speechiness_head(x)),
        'tempo': self.tempo_head(x) * 200.0  # Scale to BPM range
    }

    return predictions
```

### Prediction Heads

Each audio feature has a dedicated prediction head:

**Probabilistic Features (0-1 range)**

- Energy, valence, danceability, acousticness, instrumentalness, liveness, speechiness
- Single linear layer with sigmoid activation
- Loss: Mean Squared Error between predicted and target values

**Tempo Prediction (BPM range)**

- Linear layer without activation, scaled by 200.0
- Assumes tempo range 0-200 BPM for most music
- Loss: Mean Squared Error with tempo normalized by 200

## Training Process

### Dataset Requirements

**Primary Training Data**

- Million Song Dataset (MSD) with Last.fm tag integration
- Echo Nest audio features as ground truth targets
- Minimum 1 million tracks with complete feature sets

**Data Filtering Criteria**

- Tracks must have at least 3 genre/mood tags
- All 8 target audio features must be available
- Text metadata (artist, title) must be present and non-empty
- Release year must be within reasonable range (1900-2024)

### Data Preprocessing

#### Text Processing

```python
def preprocess_text(artist: str, title: str) -> str:
    # Clean and normalize text
    artist_clean = re.sub(r'[^\w\s-]', '', artist.lower().strip())
    title_clean = re.sub(r'[^\w\s-]', '', title.lower().strip())

    # Create combined text representation
    combined_text = f"{artist_clean} - {title_clean}"

    # Tokenize using BERT tokenizer
    tokens = tokenizer.encode(combined_text, max_length=128, truncation=True)

    return tokens
```

#### Genre/Mood Processing

```python
def process_genres_moods(tags: List[str], genre_to_idx: Dict, mood_to_idx: Dict):
    # Separate tags into genres and moods using predefined mappings
    genres = [tag for tag in tags if tag in genre_to_idx][:5]  # Max 5
    moods = [tag for tag in tags if tag in mood_to_idx][:3]    # Max 3

    # Create multi-hot encodings
    genre_ids = [genre_to_idx[g] for g in genres]
    mood_ids = [mood_to_idx[m] for m in moods]

    return genre_ids, mood_ids
```

#### Metadata Normalization

```python
def normalize_metadata(df: pd.DataFrame) -> np.ndarray:
    feature_cols = [
        'duration_ms', 'release_year', 'explicit',
        'track_number', 'album_total_tracks', 'decade'
    ]

    # Handle missing values
    df['duration_ms'] = df['duration_ms'].fillna(180000)  # 3 minutes default
    df['release_year'] = df['release_year'].fillna(2000)
    df['decade'] = (df['release_year'] // 10) * 10

    # Standardize features using fitted scaler
    metadata_array = df[feature_cols].values
    normalized = self.metadata_scaler.fit_transform(metadata_array)

    return normalized
```

### Training Configuration

```python
{
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 100,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "scheduler": {
        "type": "CosineAnnealingLR",
        "T_max": 100,
        "eta_min": 0.0001
    },
    "dropout_rate": 0.3,
    "gradient_clipping": 1.0
}
```

### Loss Function and Optimization

#### Multi-Task Loss

```python
def compute_loss(predictions, targets, feature_weights):
    total_loss = 0.0
    feature_losses = {}

    for feature in ['energy', 'valence', 'danceability', 'acousticness',
                   'instrumentalness', 'liveness', 'speechiness']:
        mse_loss = F.mse_loss(predictions[feature], targets[feature])
        weighted_loss = feature_weights[feature] * mse_loss
        total_loss += weighted_loss
        feature_losses[f'{feature}_loss'] = mse_loss.item()

    # Tempo loss (scaled)
    tempo_loss = F.mse_loss(predictions['tempo'] / 200.0, targets['tempo'] / 200.0)
    total_loss += feature_weights['tempo'] * tempo_loss
    feature_losses['tempo_loss'] = tempo_loss.item()

    return total_loss, feature_losses
```

#### Feature Weight Balancing

Different audio features have varying prediction difficulty:

- Energy, valence: Weight 1.0 (easier to predict from text/genre)
- Danceability, tempo: Weight 1.2 (moderate difficulty)
- Acousticness, instrumentalness: Weight 1.5 (harder to predict from metadata)
- Liveness, speechiness: Weight 2.0 (most difficult, require audio analysis)

### Training Loop

```python
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    feature_losses = defaultdict(float)

    for batch in dataloader:
        optimizer.zero_grad()

        # Move batch to device
        text_embeds = batch['text_embeddings'].to(device)
        genre_embeds = batch['genre_embeddings'].to(device)
        mood_embeds = batch['mood_embeddings'].to(device)
        metadata = batch['metadata'].to(device)
        targets = {k: v.to(device) for k, v in batch['targets'].items()}

        # Forward pass
        predictions = model(text_embeds, genre_embeds, mood_embeds, metadata)

        # Calculate loss
        loss, batch_feature_losses = compute_loss(predictions, targets, feature_weights)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        for feature, feature_loss in batch_feature_losses.items():
            feature_losses[feature] += feature_loss

    # Return average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_feature_losses = {k: v / num_batches for k, v in feature_losses.items()}

    return avg_loss, avg_feature_losses
```

## Evaluation and Validation

### Evaluation Metrics

#### Regression Metrics

- **Mean Squared Error (MSE)**: Primary metric for each audio feature
- **Mean Absolute Error (MAE)**: Robust to outliers
- **R-squared**: Proportion of variance explained
- **Pearson Correlation**: Linear relationship strength

#### Target Performance Thresholds

- Energy, Valence: MSE ≤ 0.025 (RMSE ≤ 0.158)
- Danceability: MSE ≤ 0.030 (RMSE ≤ 0.173)
- Acousticness, Instrumentalness: MSE ≤ 0.040 (RMSE ≤ 0.200)
- Liveness, Speechiness: MSE ≤ 0.050 (RMSE ≤ 0.224)
- Tempo: MSE ≤ 400 (RMSE ≤ 20 BPM)

#### Cross-Validation Strategy

```python
def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_predictions = defaultdict(list)
    all_targets = defaultdict(list)

    with torch.no_grad():
        for batch in test_dataloader:
            predictions = model(**batch)
            targets = batch['targets']

            for feature in predictions:
                all_predictions[feature].extend(predictions[feature].cpu().numpy())
                all_targets[feature].extend(targets[feature].cpu().numpy())

    # Calculate metrics for each feature
    metrics = {}
    for feature in all_predictions:
        pred_array = np.array(all_predictions[feature])
        target_array = np.array(all_targets[feature])

        metrics[feature] = {
            'mse': mean_squared_error(target_array, pred_array),
            'mae': mean_absolute_error(target_array, pred_array),
            'r2': r2_score(target_array, pred_array),
            'correlation': pearsonr(target_array, pred_array)[0]
        }

    return metrics
```

### Validation Strategies

#### Temporal Split Validation

- Training: Tracks released before 2018
- Validation: Tracks from 2018-2020
- Test: Tracks from 2021+
- Prevents data leakage from future musical trends

#### Genre-Stratified Validation

- Ensure balanced representation across genres in all splits
- Prevent bias toward dominant genres in training data
- Validate model performance across diverse musical styles

#### Artist Holdout Validation

- Reserve entire artist catalogs for validation
- Tests model's ability to generalize to unseen artists
- Critical for real-world deployment scenarios

## Production Deployment

### Model Serving

#### Batch Prediction Pipeline

```python
def predict_batch_audio_features(model, tracks_df, batch_size=1000):
    model.eval()
    all_predictions = []

    for i in range(0, len(tracks_df), batch_size):
        batch_df = tracks_df.iloc[i:i+batch_size]

        # Prepare batch inputs
        text_embeddings = prepare_text_embeddings(batch_df)
        genre_embeddings = prepare_genre_embeddings(batch_df)
        mood_embeddings = prepare_mood_embeddings(batch_df)
        metadata = prepare_metadata(batch_df)

        # Generate predictions
        with torch.no_grad():
            predictions = model(text_embeddings, genre_embeddings,
                              mood_embeddings, metadata)

        # Convert to pandas DataFrame
        batch_predictions = pd.DataFrame({
            feature: predictions[feature].cpu().numpy()
            for feature in predictions
        })

        all_predictions.append(batch_predictions)

    return pd.concat(all_predictions, ignore_index=True)
```

#### Real-time Inference API

```python
def predict_single_track(
    model,
    title: str,
    artist: str,
    genres: List[str],
    moods: List[str],
    metadata: Dict[str, Any]
) -> Dict[str, float]:

    model.eval()

    # Prepare inputs
    text_embedding = encode_text(f"{artist} - {title}")
    genre_embedding = encode_genres(genres)
    mood_embedding = encode_moods(moods)
    metadata_features = normalize_metadata_dict(metadata)

    # Convert to tensors
    text_tensor = torch.FloatTensor(text_embedding).unsqueeze(0)
    genre_tensor = torch.FloatTensor(genre_embedding).unsqueeze(0)
    mood_tensor = torch.FloatTensor(mood_embedding).unsqueeze(0)
    metadata_tensor = torch.FloatTensor(metadata_features).unsqueeze(0)

    # Generate prediction
    with torch.no_grad():
        predictions = model(text_tensor, genre_tensor, mood_tensor, metadata_tensor)

    # Convert to dictionary
    result = {
        feature: float(predictions[feature][0])
        for feature in predictions
    }

    return result
```

### Performance Optimization

#### Model Quantization

```python
# Post-training quantization for production deployment
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Reduces model size by ~75% with minimal accuracy loss
```

#### Caching Strategy

```python
class AudioFeaturePredictorWithCache:
    def __init__(self, model, cache_size=10000):
        self.model = model
        self.cache = LRUCache(max_size=cache_size)

    def predict(self, track_signature):
        # Create cache key from track metadata
        cache_key = hashlib.md5(
            f"{track_signature['artist']}|{track_signature['title']}|"
            f"{track_signature['release_year']}".encode()
        ).hexdigest()

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate prediction
        prediction = self.model.predict_single_track(**track_signature)

        # Cache result
        self.cache[cache_key] = prediction

        return prediction
```

## Confidence Estimation

### Metadata Completeness Scoring

The model generates confidence scores based on input data quality:

```python
def calculate_prediction_confidence(
    title: str,
    artist: str,
    genres: List[str],
    moods: List[str],
    metadata: Dict[str, Any]
) -> float:

    confidence = 0.0

    # Text completeness (40% of total confidence)
    if title and len(title.strip()) > 0:
        confidence += 0.2
    if artist and len(artist.strip()) > 0:
        confidence += 0.2

    # Genre information (30% of total confidence)
    if genres and len(genres) > 0:
        confidence += 0.15 + min(len(genres) * 0.05, 0.15)

    # Mood information (20% of total confidence)
    if moods and len(moods) > 0:
        confidence += 0.1 + min(len(moods) * 0.05, 0.1)

    # Metadata completeness (10% of total confidence)
    metadata_score = 0.0
    if metadata.get('duration_ms'):
        metadata_score += 0.03
    if metadata.get('release_year'):
        metadata_score += 0.03
    if metadata.get('album_total_tracks'):
        metadata_score += 0.02
    if metadata.get('track_number'):
        metadata_score += 0.02

    confidence += metadata_score

    return min(confidence, 1.0)
```

### Prediction Uncertainty Estimation

Advanced confidence estimation using model ensembling:

```python
def estimate_prediction_uncertainty(models, inputs, num_samples=10):
    predictions = []

    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(**inputs)
            predictions.append(pred)

    # Calculate mean and standard deviation across models
    uncertainty_estimates = {}
    for feature in predictions[0]:
        feature_preds = [p[feature] for p in predictions]
        mean_pred = torch.mean(torch.stack(feature_preds), dim=0)
        std_pred = torch.std(torch.stack(feature_preds), dim=0)

        uncertainty_estimates[feature] = {
            'prediction': mean_pred,
            'uncertainty': std_pred,
            'confidence': 1.0 / (1.0 + std_pred)  # Inverse relationship
        }

    return uncertainty_estimates
```

## Integration with Hybrid System

### Connection to Two-Tower Model

The Audio Feature Predictor provides essential inputs to the Two-Tower model:

```python
# In HybridModel.predict()
if use_audio_prediction and self.audio_predictor:
    # Extract metadata for prediction (no popularity data!)
    metadata_features = self._extract_metadata_features(item_data)

    # Predict audio features
    predicted_audio = self.audio_predictor.predict_audio_features(
        title=item_data.get('title', ''),
        artist=item_data.get('artist', ''),
        genres=item_data.get('genres', []),
        moods=item_data.get('moods', []),
        metadata=metadata_features
    )

    # Calculate confidence
    audio_confidence = self._calculate_metadata_completeness(item_data)

    # Mark as predicted for downstream processing
    audio_features = predicted_audio
    audio_features['predicted'] = True
```

### Data Leakage Prevention

Critical safeguards ensure no target-correlated information leaks into predictions:

```python
def _extract_metadata_features(self, item_data: Dict) -> Dict[str, float]:
    """Extract metadata features for audio prediction (NO popularity data to avoid leakage)."""
    release_year = item_data.get('release_year', 2000)

    # ONLY intrinsic track properties - no popularity, ratings, or user data
    return {
        'duration_ms': item_data.get('duration_ms', 180000),
        'release_year': release_year,
        'decade': (release_year // 10) * 10,
        'explicit': float(item_data.get('explicit', False)),
        'track_number': item_data.get('track_number', 1),
        'album_total_tracks': item_data.get('album_total_tracks', 10)
    }

    # Explicitly excluded to prevent leakage:
    # - popularity, aoty_score, aoty_num_ratings
    # - artist_popularity, artist_followers
    # - user_ratings, play_counts
    # - chart_positions, streaming_metrics
```

## Advanced Applications

### Transfer Learning

The Audio Feature Predictor can be adapted for related tasks:

#### Domain Adaptation

- Fine-tune on genre-specific datasets (classical, jazz, electronic)
- Adapt to regional music characteristics
- Customize for specific user demographics

#### Multi-Modal Learning

- Incorporate album artwork features
- Add lyrical content analysis
- Include social media signals

### Continual Learning

The model supports incremental updates as new data becomes available:

```python
def update_model_with_new_data(model, new_dataloader, learning_rate=0.0001):
    # Lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Selective parameter updates
    for name, param in model.named_parameters():
        if 'prediction_heads' in name:
            param.requires_grad = True  # Update prediction heads
        else:
            param.requires_grad = False  # Freeze feature extractors

    # Fine-tune on new data
    for epoch in range(5):  # Limited epochs to prevent forgetting
        train_epoch(model, new_dataloader, optimizer, device)
```

The Audio Feature Predictor provides a robust solution for predicting audio characteristics from limited metadata, enabling comprehensive content-based recommendation even when direct audio features are unavailable.





