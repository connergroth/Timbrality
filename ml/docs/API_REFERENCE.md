# API Reference - Collaborative Filtering Components

## Overview

This document provides detailed API reference for all classes and methods in the collaborative filtering implementation.

## Table of Contents

1. [NMFModel](#nmfmodel)
2. [DataLoader](#dataloader)
3. [ModelTrainer](#modeltrainer)
4. [RedisConnector](#redisconnector)
5. [Configuration](#configuration)
6. [Error Handling](#error-handling)

## NMFModel

**Module**: `timbral.models.nmf_model`

The core collaborative filtering model implementing Non-negative Matrix Factorization.

### Class Definition

```python
class NMFModel:
    def __init__(
        self,
        n_components: int = None,
        random_state: int = None,
        max_iter: int = None,
        tol: float = None
    )
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | `int` | `settings.NMF_N_COMPONENTS` | Number of latent factors to learn |
| `random_state` | `int` | `settings.NMF_RANDOM_STATE` | Random seed for reproducibility |
| `max_iter` | `int` | `Constants.NMF_MAX_ITER` | Maximum iterations for convergence |
| `tol` | `float` | `Constants.NMF_TOL` | Tolerance for stopping criterion |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `user_factors` | `np.ndarray` | User latent factor matrix (n_users × n_components) |
| `item_factors` | `np.ndarray` | Item latent factor matrix (n_components × n_items) |
| `is_fitted` | `bool` | Whether the model has been trained |
| `n_users` | `int` | Number of users in training data |
| `n_items` | `int` | Number of items in training data |

### Methods

#### fit(user_item_matrix)

Trains the NMF model on user-item interaction data.

**Parameters:**
- `user_item_matrix` (`np.ndarray` or `pd.DataFrame`): User-item interaction matrix

**Returns:**
- `NMFModel`: Self for method chaining

**Raises:**
- `Exception`: If training fails

**Example:**
```python
model = NMFModel(n_components=50)
model.fit(interaction_matrix)
```

#### predict(user_ids, item_ids)

Generates predictions for user-item pairs.

**Parameters:**
- `user_ids` (`np.ndarray`): Array of user indices
- `item_ids` (`np.ndarray`): Array of item indices

**Returns:**
- `np.ndarray`: Predicted interaction scores

**Raises:**
- `ValueError`: If model not fitted or indices out of bounds

**Example:**
```python
predictions = model.predict([0, 1, 2], [10, 15, 20])
```

#### get_user_embeddings(user_ids)

Extracts latent factor embeddings for users.

**Parameters:**
- `user_ids` (`np.ndarray`): Array of user indices

**Returns:**
- `np.ndarray`: User embeddings matrix

**Raises:**
- `ValueError`: If model not fitted or indices out of bounds

**Example:**
```python
embeddings = model.get_user_embeddings([0, 1, 2])
```

#### get_item_embeddings(item_ids)

Extracts latent factor embeddings for items.

**Parameters:**
- `item_ids` (`np.ndarray`): Array of item indices

**Returns:**
- `np.ndarray`: Item embeddings matrix

**Raises:**
- `ValueError`: If model not fitted or indices out of bounds

**Example:**
```python
embeddings = model.get_item_embeddings([10, 15, 20])
```

#### get_top_recommendations(user_id, top_k, exclude_seen)

Generates top-k recommendations for a user.

**Parameters:**
- `user_id` (`int`): User index for recommendations
- `top_k` (`int`, optional): Number of recommendations to return. Default: 10
- `exclude_seen` (`bool`, optional): Whether to exclude seen items. Default: True

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (item_indices, scores)

**Raises:**
- `ValueError`: If model not fitted or user_id out of bounds

**Example:**
```python
items, scores = model.get_top_recommendations(user_id=5, top_k=10)
```

#### save(filepath)

Saves the trained model to disk.

**Parameters:**
- `filepath` (`str`): Path to save the model

**Raises:**
- `ValueError`: If model not fitted
- `OSError`: If directory creation fails

**Example:**
```python
model.save("models/production_model.pkl")
```

#### load(filepath)

Loads a trained model from disk.

**Parameters:**
- `filepath` (`str`): Path to the saved model

**Returns:**
- `NMFModel`: Self with loaded model state

**Raises:**
- `FileNotFoundError`: If model file doesn't exist
- `Exception`: If loading fails

**Example:**
```python
model = NMFModel()
model.load("models/production_model.pkl")
```

## DataLoader

**Module**: `timbral.utils.data_loader`

Handles data ingestion, preprocessing, and matrix creation.

### Class Definition

```python
class DataLoader:
    def __init__(self)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `supported_formats` | `List[str]` | Supported file formats |
| `user_to_idx` | `Dict[int, int]` | Mapping from original user IDs to matrix indices |
| `item_to_idx` | `Dict[int, int]` | Mapping from original item IDs to matrix indices |
| `idx_to_user` | `Dict[int, int]` | Reverse mapping from matrix indices to user IDs |
| `idx_to_item` | `Dict[int, int]` | Reverse mapping from matrix indices to item IDs |

### Methods

#### load_user_interactions(filepath, format)

Loads user-item interaction data from file.

**Parameters:**
- `filepath` (`str`): Path to the data file
- `format` (`str`, optional): File format ('csv', 'parquet', 'json'). Default: 'csv'

**Returns:**
- `pd.DataFrame`: DataFrame with interaction data

**Raises:**
- `ValueError`: If format unsupported or required columns missing
- `Exception`: If file loading fails

**Required Columns:**
- `user_id` (int): User identifier
- `item_id` (int): Item identifier
- `rating` (float, optional): Interaction strength

**Example:**
```python
loader = DataLoader()
interactions = loader.load_user_interactions("data/ratings.csv")
```

#### create_user_item_matrix(interactions_df, user_col, item_col, value_col)

Creates user-item interaction matrix from DataFrame.

**Parameters:**
- `interactions_df` (`pd.DataFrame`): DataFrame with interaction data
- `user_col` (`str`, optional): User column name. Default: 'user_id'
- `item_col` (`str`, optional): Item column name. Default: 'item_id'
- `value_col` (`str`, optional): Value column name. Default: 'rating'

**Returns:**
- `pd.DataFrame`: User-item matrix with users as rows, items as columns

**Side Effects:**
- Sets `user_to_idx`, `item_to_idx`, `idx_to_user`, `idx_to_item` attributes

**Example:**
```python
matrix = loader.create_user_item_matrix(interactions_df)
```

## ModelTrainer

**Module**: `timbral.logic.trainer`

Orchestrates the training pipeline for collaborative filtering models.

### Class Definition

```python
class ModelTrainer:
    def __init__(self)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_loader` | `DataLoader` | Data loading utility |
| `evaluation_metrics` | `EvaluationMetrics` | Evaluation utility |
| `training_history` | `Dict` | Training history and metadata |

### Methods

#### train_nmf_model(user_item_matrix, n_components, random_state, max_iter, tol)

Trains an NMF model on user-item interaction data.

**Parameters:**
- `user_item_matrix` (`np.ndarray`): User-item interaction matrix
- `n_components` (`int`, optional): Number of latent factors
- `random_state` (`int`, optional): Random seed
- `max_iter` (`int`, optional): Maximum iterations
- `tol` (`float`, optional): Convergence tolerance

**Returns:**
- `NMFModel`: Trained NMF model

**Side Effects:**
- Updates `training_history` with training metadata

**Example:**
```python
trainer = ModelTrainer()
model = trainer.train_nmf_model(
    user_item_matrix=matrix,
    n_components=50,
    random_state=42
)
```

## RedisConnector

**Module**: `timbral.utils.redis_connector`

Provides caching capabilities for recommendations and embeddings.

### Class Definition

```python
class RedisConnector:
    def __init__(self)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `redis_client` | `redis.Redis` | Redis client instance |

### Methods

#### set_recommendations(user_id, recommendations, ttl)

Caches user recommendations in Redis.

**Parameters:**
- `user_id` (`int`): User identifier
- `recommendations` (`List[Dict[str, Any]]`): List of recommendation dictionaries
- `ttl` (`int`, optional): Time to live in seconds. Default: 3600

**Returns:**
- `bool`: Success status

**Cache Key Format:** `recommendations:user:{user_id}`

**Example:**
```python
redis_conn = RedisConnector()
success = redis_conn.set_recommendations(
    user_id=123,
    recommendations=[{"item_id": 1, "score": 0.95}],
    ttl=3600
)
```

#### get_recommendations(user_id)

Retrieves cached recommendations for a user.

**Parameters:**
- `user_id` (`int`): User identifier

**Returns:**
- `Optional[List[Dict[str, Any]]]`: Cached recommendations or None

**Example:**
```python
recommendations = redis_conn.get_recommendations(user_id=123)
```

#### set_embeddings(key, embeddings, ttl)

Caches embeddings in Redis using pickle serialization.

**Parameters:**
- `key` (`str`): Cache key identifier
- `embeddings` (`Dict[str, Any]`): Embeddings data
- `ttl` (`int`, optional): Time to live in seconds. Default: 86400

**Returns:**
- `bool`: Success status

**Cache Key Format:** `embeddings:{key}`

**Example:**
```python
embeddings = {"user_embeddings": user_factors}
success = redis_conn.set_embeddings("model_v1_users", embeddings)
```

#### get_embeddings(key)

Retrieves cached embeddings.

**Parameters:**
- `key` (`str`): Cache key identifier

**Returns:**
- `Optional[Dict[str, Any]]`: Cached embeddings or None

**Example:**
```python
embeddings = redis_conn.get_embeddings("model_v1_users")
```

#### delete_key(key)

Deletes a key from Redis.

**Parameters:**
- `key` (`str`): Key to delete

**Returns:**
- `bool`: Success status

#### health_check()

Checks Redis connection health.

**Returns:**
- `Dict[str, Any]`: Health status information

#### close()

Closes Redis connection.

## Configuration

**Module**: `timbral.config.settings`

### Settings Class

Global configuration settings for the ML service.

#### Redis Configuration
- `REDIS_URL` (`Optional[str]`): Redis connection URL
- `REDIS_PASSWORD` (`Optional[str]`): Redis password
- `REDIS_DB` (`int`): Redis database number (default: 0)

#### Model Configuration
- `NMF_N_COMPONENTS` (`int`): Default number of NMF components (default: 100)
- `NMF_RANDOM_STATE` (`int`): Default random state (default: 42)

#### API Configuration
- `API_HOST` (`str`): API host address (default: "0.0.0.0")
- `API_PORT` (`int`): API port (default: 8000)
- `API_WORKERS` (`int`): Number of API workers (default: 4)
- `DEBUG` (`bool`): Debug mode flag (default: False)

### Constants Class

Global constants used throughout the application.

#### Data Processing
- `MIN_PLAYS_THRESHOLD`: Minimum plays threshold (5)
- `MAX_PLAYS_THRESHOLD`: Maximum plays threshold (10000)

#### Model Parameters
- `NMF_MAX_ITER`: Maximum NMF iterations (200)
- `NMF_TOL`: NMF convergence tolerance (1e-4)

#### Cache Keys
- `USER_EMBEDDINGS_KEY`: User embeddings cache key
- `ITEM_EMBEDDINGS_KEY`: Item embeddings cache key
- `RECOMMENDATIONS_KEY`: Recommendations cache key

#### File Extensions
- `SUPPORTED_AUDIO_FORMATS`: Supported audio file formats
- `SUPPORTED_DATA_FORMATS`: Supported data file formats

## Error Handling

### Exception Hierarchy

The system uses a structured approach to error handling:

```
Exception
├── ValueError: Invalid parameters or input data
├── FileNotFoundError: Missing model or data files
├── ConnectionError: Redis connection issues
└── RuntimeError: Model training or prediction failures
```

### Common Error Scenarios

#### Model Not Fitted
```python
# Error
predictions = model.predict([0], [0])  # Model not trained

# Exception
ValueError: Model must be fitted before making predictions
```

#### Index Out of Bounds
```python
# Error
predictions = model.predict([1000], [0])  # User 1000 doesn't exist

# Exception
ValueError: User IDs must be in range [0, 50)
```

#### Missing Required Columns
```python
# Error
df = pd.DataFrame({"user": [1, 2], "item": [1, 2]})  # Wrong column names
loader.create_user_item_matrix(df)

# Exception
ValueError: Missing required columns: ['user_id', 'item_id']
```

#### Redis Connection Failure
```python
# Error scenario: Redis server down
redis_conn = RedisConnector()

# Behavior: Graceful degradation
# - redis_client attribute set to None
# - All cache operations return False/None
# - No exceptions thrown, allows fallback to direct computation
```

### Error Handling Best Practices

1. **Graceful Degradation**: Services continue operating when dependencies fail
2. **Detailed Logging**: All errors logged with context information
3. **Input Validation**: Early validation of parameters and data
4. **Fallback Mechanisms**: Alternative code paths when external services fail

### Debug Information

Enable detailed logging for troubleshooting:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get logger for specific component
logger = logging.getLogger('timbral.models.nmf_model')
```

This provides comprehensive API documentation for all collaborative filtering components, enabling developers to effectively integrate and extend the system.