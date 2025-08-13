# Enhanced Collaborative Filtering System

This document describes the advanced collaborative filtering system with 2-hop neighbors, diversity optimization, and quality metrics.

## 🚀 Quick Start

### 1. Ingest Your Personal Library

```bash
# Basic ingestion with your Last.fm data
python -m scripts.library_ingestion_script \
    --lastfm-username YOUR_LASTFM_USERNAME \
    --max-neighbors 50 \
    --similarity-threshold 0.1

# Full ingestion with output file
python -m scripts.library_ingestion_script \
    --lastfm-username YOUR_USERNAME \
    --spotify-username YOUR_SPOTIFY \
    --max-neighbors 100 \
    --similarity-threshold 0.05 \
    --output-file ingestion_results.json \
    --test-recommendations USER_ID_HERE
```

### 2. Generate Enhanced Recommendations

```python
from services.enhanced_collaborative_filtering import EnhancedCollaborativeFilteringService

# Initialize service
enhanced_cf = EnhancedCollaborativeFilteringService(
    min_neighbors=25,
    max_neighbors=125,
    diversity_lambda=0.7  # Balance relevance vs diversity
)

# Generate recommendations
recommendations = await enhanced_cf.generate_recommendations(
    target_user_id="user_id_here",
    num_recommendations=20,
    include_explanations=True
)

print(f"Generated {len(recommendations['recommendations'])} recommendations")
print(f"Quality metrics: {recommendations['metadata']['quality_metrics']}")
```

### 3. Use API Endpoints

```bash
# Generate enhanced recommendations
curl -X POST "http://localhost:8000/recommendations/enhanced/generate?user_id=USER_ID&num_recommendations=20&diversity_lambda=0.7"

# Get user's neighbors
curl "http://localhost:8000/recommendations/enhanced/user/USER_ID/neighbors?include_two_hop=true"

# Start personal library ingestion (background task)
curl -X POST "http://localhost:8000/recommendations/enhanced/ingest/personal-library?lastfm_username=YOUR_USERNAME&max_neighbors=50"
```

## 🧠 Algorithm Overview

### 2-Hop Neighbor Discovery

The system finds both direct (1-hop) and indirect (2-hop) neighbors:

- **1-hop neighbors**: Users directly similar to you
- **2-hop neighbors**: Users similar to your similar users, with similarity decay
- **Formula**: `sim_2hop = sim(you, neighbor1) × sim(neighbor1, neighbor2)`
- **Threshold**: Only includes 2-hop neighbors with `sim_2hop ≥ 0.05`

### Adaptive K Selection

Instead of fixed k neighbors, the system adaptively selects neighbors:

- **Minimum**: At least 25 neighbors (prevents narrow recommendations)
- **Maximum**: At most 125 neighbors (caps computational cost)
- **Target**: Include neighbors until cumulative similarity ≥ 0.85
- **Ensures**: Broad but relevant neighbor pool for training

### Scoring with De-biasing

**Base Score**: `score(item) = Σ_neighbors sim(user,neighbor) × log(1 + plays_neighbor,item)`

**De-popularization**: `final_score = base_score / log(1 + global_popularity(item))`

This reduces bias toward popular items while preserving their signal.

### Diversity Re-ranking (MMR)

Uses Maximal Marginal Relevance on top ~500 candidates:

```
MMR_score = λ × relevance - (1-λ) × max_similarity_to_selected
```

- **λ = 0.7**: 70% relevance, 30% diversity penalty
- **Content vectors**: Uses item-item cosine similarity from metadata/audio features
- **Result**: Diverse but relevant recommendations

### Hard Caps & Gentle Boosts

**Artist Cap**: ≤3 tracks per artist in top 20 recommendations
**Long-tail Boost**: +20% score boost for tracks below 70th percentile popularity
**Temporal Mix**: Ensures ≥30% of recommendations from last 24 months
**Content Blending**: `final = α × CF + (1-α) × Content` (α = 0.6 default)

## 📊 Quality Metrics

The system calculates several quality metrics:

### Intra-List Diversity (ILD)
- **Formula**: Mean(1 - cosine_similarity) between all recommendation pairs
- **Higher is better**: More diverse recommendations
- **Target**: ILD ≥ 0.4

### Artist/Genre Entropy  
- **Formula**: Shannon entropy of artist distribution in recommendations
- **Higher is better**: More diverse artists/genres
- **Target**: Entropy ≥ 2.0

### Long-tail Fraction
- **Formula**: Share of recommendations below median popularity
- **Target**: ≥ 0.3 (30% non-mainstream tracks)

### Repeat Artist Rate
- **Formula**: (Total tracks - Unique artists) / Total tracks
- **Lower is better**: Less artist repetition
- **Target**: ≤ 0.25 (max 25% repeats)

## 🛠️ Configuration Options

### Service Parameters

```python
EnhancedCollaborativeFilteringService(
    min_neighbors=25,           # Minimum neighbors (5-50)
    max_neighbors=125,          # Maximum neighbors (25-200)
    similarity_threshold=0.05,  # 2-hop threshold (0.01-0.3)
    cumulative_similarity_target=0.85,  # Adaptive k target (0.5-0.95)
    diversity_lambda=0.7,       # MMR diversity weight (0.0-1.0)  
    content_blend_alpha=0.6     # CF vs content blend (0.0-1.0)
)
```

### API Parameters

All API endpoints accept parameter overrides:

```bash
# High diversity, fewer neighbors
curl -X POST ".../generate?user_id=USER&diversity_lambda=0.9&max_neighbors=75"

# Pure relevance, more neighbors  
curl -X POST ".../generate?user_id=USER&diversity_lambda=0.3&max_neighbors=150"
```

## 📈 Performance & Scaling

### Similarity Calculation Optimization

```python
from utils.similarity_calculator import SimilarityCalculator

# Batch calculate all similarities
calculator = SimilarityCalculator(batch_size=1000)
stats = await calculator.calculate_all_similarities(
    recalculate_existing=False,
    min_shared_tracks=3,
    min_similarity=0.01
)

# Get similarity statistics
sim_stats = await calculator.get_similarity_statistics()
print(f"Total similarities: {sim_stats['total_similarities']}")
```

### Database Optimization

1. **Indexes**: Ensure indexes on `user_track_interactions.lastfm_user_id` and `user_similarities.user_id_1, user_id_2`
2. **Partitioning**: Consider partitioning large tables by date
3. **Caching**: Service includes similarity and content vector caching
4. **Batch Processing**: Similarity calculations use batched operations

### Memory Management

- **User profiles**: Loaded into memory during similarity calculation for speed
- **Content vectors**: Cached to avoid repeated computation
- **Batch size**: Configurable batch size for large-scale operations

## 🔧 Troubleshooting

### Common Issues

**1. No recommendations generated**
- Check if user has enough interaction data
- Verify user similarities exist in database
- Ensure target user has active neighbors

**2. Low diversity scores**
- Increase `diversity_lambda` (0.8-0.9)
- Check if content vectors are properly implemented
- Verify MMR re-ranking is working

**3. Slow performance**
- Use similarity pre-calculation
- Implement content vector caching
- Consider reducing `max_neighbors`

**4. Poor recommendation quality**
- Ensure sufficient training data (neighbors)
- Check similarity calculation quality
- Verify popularity de-biasing is working

### Debug Endpoints

```bash
# Check user's neighbors
curl "http://localhost:8000/recommendations/enhanced/user/USER_ID/neighbors"

# Get quality metrics only
curl "http://localhost:8000/recommendations/enhanced/user/USER_ID/quality-metrics"

# Algorithm parameters
curl "http://localhost:8000/recommendations/enhanced/algorithm/parameters"

# Health check
curl "http://localhost:8000/recommendations/enhanced/health"
```

## 📋 Data Requirements

### Minimum Requirements

- **Users**: At least 10 active users with interaction data
- **Tracks**: At least 1000 unique tracks across users
- **Interactions**: At least 50 interactions per user
- **Overlap**: Users should share some tracks for similarity calculation

### Recommended Scale

- **Users**: 100+ users for effective collaborative filtering
- **Tracks**: 10,000+ unique tracks for diversity
- **Interactions**: 200+ interactions per user for profile richness
- **Coverage**: Each track should have interactions from 2+ users

### Data Quality

- **Clean interactions**: Remove spam, duplicates, very low play counts
- **Balanced users**: Mix of light and heavy listeners
- **Temporal spread**: Include both recent and historical data
- **Genre diversity**: Ensure multiple genres/styles represented

## 🚀 Advanced Usage

### Custom Content Vectors

Replace the placeholder in `_get_content_vectors()`:

```python
async def _get_content_vectors(self, track_ids: List[str]) -> Dict[str, np.ndarray]:
    vectors = {}
    
    # Your implementation using:
    # - Genre tags from database
    # - Audio features from Spotify
    # - AOTY metadata
    # - Custom embedding models
    
    for track_id in track_ids:
        # Build feature vector for this track
        genre_features = self._get_genre_features(track_id)
        audio_features = self._get_audio_features(track_id) 
        metadata_features = self._get_metadata_features(track_id)
        
        # Combine into single vector
        vectors[track_id] = np.concatenate([
            genre_features,
            audio_features,
            metadata_features
        ])
    
    return vectors
```

### Custom Scoring Functions

Override methods for custom scoring:

```python
class CustomCollaborativeFiltering(EnhancedCollaborativeFilteringService):
    
    async def _generate_candidates(self, user_id: str, neighbors: List[Dict]) -> List[Dict]:
        # Custom candidate generation
        # - Include temporal weighting
        # - Add mood/context scoring  
        # - Custom play count transformation
        pass
    
    async def _apply_caps_and_boosts(self, candidates: List[Dict], num_recs: int) -> List[Dict]:
        # Custom caps and boosts
        # - Genre diversity requirements
        # - Label/era balancing
        # - User-specific preferences
        pass
```

### Integration with ML Models

Blend with your existing ML models:

```python
async def _blend_with_content_based(self, user_id: str, cf_recs: List[Dict], num_recs: int):
    # Get predictions from your ML model
    ml_predictions = await your_ml_model.predict(user_id)
    
    # Blend scores
    for rec in cf_recs:
        cf_score = rec['final_score']
        ml_score = ml_predictions.get(rec['track_id'], 0.0)
        
        # Custom blending function
        rec['final_score'] = self._blend_scores(cf_score, ml_score, user_id)
    
    return cf_recs
```

## 📊 Monitoring & Analytics

### Key Metrics to Track

1. **System Performance**
   - Recommendation generation time
   - Similarity calculation time  
   - Cache hit rates

2. **Quality Metrics** 
   - Average ILD across users
   - Artist entropy distribution
   - Long-tail fraction trends
   - Repeat artist rates

3. **User Engagement**
   - Recommendation click-through rates
   - User feedback (likes/dislikes)
   - Recommendation diversity preferences

4. **Data Health**
   - User interaction distribution
   - Track popularity distribution
   - Similarity network connectivity

### Logging Configuration

```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_cf.log'),
        logging.StreamHandler()
    ]
)

# Logger will capture:
# - Recommendation generation steps
# - Quality metric calculations
# - Performance timing
# - Error conditions
```

This enhanced collaborative filtering system provides a comprehensive, production-ready solution for music recommendations with state-of-the-art diversity optimization and quality control.