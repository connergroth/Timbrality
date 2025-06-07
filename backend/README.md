# Tensoe Backend - Complete ML Music Platform

A comprehensive music discovery and machine learning platform that integrates Spotify, Last.fm, and Album of the Year data to create rich training datasets for ML models.

## 🚀 Features

### Core Capabilities

- **Music Data Ingestion**: Automated collection from Spotify, Last.fm, and AOTY
- **ML Training Pipeline**: Complete feature engineering and data preparation
- **Real-time Analytics**: Genre/mood distribution and popularity insights
- **RESTful API**: Full FastAPI-based interface with automatic documentation
- **Scalable Architecture**: Async processing, connection pooling, and caching

### Data Sources

- **Spotify**: Track metadata, audio features, popularity metrics
- **Last.fm**: Genre tags, mood classifications, social data
- **AOTY**: Album ratings, critic scores, release information

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL database
- Redis (optional, for caching)
- API keys for Spotify and Last.fm

## ⚡ Quick Start

### 1. Environment Setup

Create a `.env` file with required configuration:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/tensoe_db

# Spotify API (Required)
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret

# Last.fm API (Required)
LASTFM_API_KEY=your_lastfm_api_key
LASTFM_API_SECRET=your_lastfm_secret

# Supabase (Optional)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key

# Redis Cache (Optional)
REDIS_URL=redis://localhost:6379/0

# API Configuration
ENVIRONMENT=development
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000
```

### 2. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database tables
python -m backend.startup
```

### 3. Run the Server

```bash
# Using the startup script
python -m backend.startup

# Or directly with uvicorn
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/
- **System Info**: http://localhost:8000/system

## 🎵 Usage Examples

### Ingest Music Data

```python
import requests

# Ingest a single album
response = requests.post("http://localhost:8000/ml/ingest/album", json={
    "album_name": "The Dark Side of the Moon",
    "artist_name": "Pink Floyd"
})

# Batch ingestion
response = requests.post("http://localhost:8000/ml/ingest/batch", json={
    "albums": [
        ["Abbey Road", "The Beatles"],
        ["Kind of Blue", "Miles Davis"],
        ["Nevermind", "Nirvana"]
    ]
})
```

### Get Training Data

```python
# Get ML training data
response = requests.get("http://localhost:8000/ml/training-data?limit=1000")
training_data = response.json()

# Get prepared feature matrix
response = requests.get("http://localhost:8000/ml/feature-matrix?limit=1000")
feature_matrix = response.json()
```

### Analytics & Insights

```python
# Get comprehensive analytics
response = requests.get("http://localhost:8000/ml/analytics")
analytics = response.json()

# Genre distribution
response = requests.get("http://localhost:8000/ml/genres?limit=20")
genres = response.json()

# Ingestion statistics
response = requests.get("http://localhost:8000/ml/stats")
stats = response.json()
```

## 🏗️ Architecture

### Directory Structure

```
backend/
├── config/
│   └── settings.py          # Centralized configuration
├── models/
│   ├── database.py          # Database connection & base
│   ├── track.py            # Track model
│   ├── album.py            # Album model
│   ├── artist.py           # Artist model
│   └── ingestion_models.py # ML-optimized models
├── services/
│   └── ml_service.py       # Core ML service
├── routes/
│   ├── ml_routes.py        # ML & ingestion endpoints
│   ├── album_routes.py     # Album endpoints
│   ├── user_routes.py      # User endpoints
│   └── metrics_routes.py   # Metrics endpoints
├── ingestion/
│   ├── spotify_fetcher.py  # Spotify API integration
│   ├── lastfm_fetcher.py   # Last.fm API integration
│   ├── aoty_scraper.py     # AOTY scraper wrapper
│   ├── normalizer.py       # Data normalization
│   ├── insert_to_supabase.py # Database insertion
│   └── ingest_runner.py    # Main ingestion orchestrator
├── utils/
├── cache/
├── metrics/
├── startup.py              # Complete system startup
├── main.py                 # FastAPI application
└── requirements.txt
```

### Data Schema

The system uses an enhanced schema optimized for ML training:

```sql
-- Enhanced tracks table
CREATE TABLE enhanced_tracks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    genres TEXT[],              -- Array of genres
    moods TEXT[],               -- Array of moods from Last.fm
    audio_features JSONB,       -- Spotify audio features
    popularity INTEGER,
    duration_ms INTEGER,
    track_number INTEGER,
    release_date DATE,
    aoty_score REAL,           -- Album of the Year score
    explicit BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- GIN indexes for efficient array searches
CREATE INDEX idx_tracks_genres ON enhanced_tracks USING GIN(genres);
CREATE INDEX idx_tracks_moods ON enhanced_tracks USING GIN(moods);
```

## 🔧 Configuration

### Environment Variables

| Variable                | Description                  | Required | Default        |
| ----------------------- | ---------------------------- | -------- | -------------- |
| `DATABASE_URL`          | PostgreSQL connection string | ✅       | -              |
| `SPOTIFY_CLIENT_ID`     | Spotify API client ID        | ✅       | -              |
| `SPOTIFY_CLIENT_SECRET` | Spotify API client secret    | ✅       | -              |
| `LASTFM_API_KEY`        | Last.fm API key              | ✅       | -              |
| `LASTFM_API_SECRET`     | Last.fm API secret           | ❌       | -              |
| `SUPABASE_URL`          | Supabase project URL         | ❌       | -              |
| `SUPABASE_ANON_KEY`     | Supabase anonymous key       | ❌       | -              |
| `REDIS_URL`             | Redis connection string      | ❌       | localhost:6379 |
| `ENVIRONMENT`           | Environment (dev/prod/test)  | ❌       | development    |
| `DEBUG`                 | Enable debug mode            | ❌       | false          |
| `API_HOST`              | API server host              | ❌       | 0.0.0.0        |
| `API_PORT`              | API server port              | ❌       | 8000           |

### Advanced Configuration

```bash
# Ingestion settings
INGESTION_BATCH_SIZE=50
INGESTION_MAX_CONCURRENT=10
MAX_GENRES_PER_TRACK=10
MAX_MOODS_PER_TRACK=15

# ML settings
ML_TRAINING_DATA_LIMIT=50000
ML_FEATURE_MATRIX_LIMIT=20000
ML_MIN_FEATURES_PER_TRACK=3

# Cache settings
CACHE_DEFAULT_TTL=3600
CACHE_ALBUM_TTL=7200
CACHE_USER_TTL=1800

# Database settings
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
```

## 📚 API Reference

### ML & Ingestion Endpoints

| Endpoint                         | Method | Description                 |
| -------------------------------- | ------ | --------------------------- |
| `/ml/ingest/album`               | POST   | Ingest single album         |
| `/ml/ingest/batch`               | POST   | Batch album ingestion       |
| `/ml/training-data`              | GET    | Get ML training data        |
| `/ml/feature-matrix`             | GET    | Get prepared feature matrix |
| `/ml/analytics`                  | GET    | Comprehensive analytics     |
| `/ml/genres`                     | GET    | Genre distribution          |
| `/ml/moods`                      | GET    | Mood distribution           |
| `/ml/stats`                      | GET    | Ingestion statistics        |
| `/ml/export/training-data`       | POST   | Export training data to CSV |
| `/ml/recommendations/{track_id}` | GET    | Get similar tracks          |

### System Endpoints

| Endpoint   | Method | Description                    |
| ---------- | ------ | ------------------------------ |
| `/`        | GET    | Health check with data summary |
| `/system`  | GET    | Detailed system information    |
| `/docs`    | GET    | Interactive API documentation  |
| `/metrics` | GET    | API usage metrics              |

## 🤖 ML Integration

### Feature Engineering

The system automatically creates ML-ready features:

```python
from backend.services.ml_service import get_feature_matrix

# Get prepared feature matrix
X, y = get_feature_matrix(limit=10000)

# Features include:
# - Numerical: duration_ms, popularity, track_number, release_year
# - Audio: danceability, energy, valence, tempo, acousticness, etc.
# - Categorical: top genres (one-hot encoded)
# - Mood: top moods (one-hot encoded)
# - Binary: explicit content

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {list(X.columns)}")
```

### Training Data Export

```python
# Export for external ML frameworks
import requests

# Export training data
response = requests.post("http://localhost:8000/ml/export/training-data",
                        params={"filename": "training_data.csv", "limit": 10000})

# Export feature matrix
response = requests.post("http://localhost:8000/ml/export/feature-matrix",
                        params={"filename": "features.csv", "limit": 10000})
```

## 🧪 Development

### Running Tests

```bash
# Run ingestion tests
python backend/test_ingestion.py

# Test individual components
python -m pytest tests/
```

### Database Migrations

```bash
# Generate migration
alembic revision --autogenerate -m "Add new features"

# Apply migration
alembic upgrade head
```

### Adding New Data Sources

1. Create fetcher in `ingestion/` directory
2. Add to `normalizer.py` for data standardization
3. Update `ingest_runner.py` to include new source
4. Add configuration to `config/settings.py`

## 📊 Performance

### Optimization Features

- **Connection Pooling**: Efficient database connections
- **Async Processing**: Non-blocking I/O operations
- **Batch Processing**: Bulk operations for large datasets
- **Caching**: Redis/in-memory caching with configurable TTL
- **Rate Limiting**: API protection and external service compliance

### Monitoring

The system provides comprehensive metrics:

```python
# Get system performance metrics
response = requests.get("http://localhost:8000/metrics")
metrics = response.json()

# Monitor ingestion progress
response = requests.get("http://localhost:8000/ml/stats")
stats = response.json()
```

## 🔐 Security

- Environment-based configuration
- API rate limiting
- SQL injection prevention via ORM
- CORS configuration
- Input validation and sanitization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Database Connection Errors**

```bash
# Check DATABASE_URL format
DATABASE_URL=postgresql://user:password@host:port/database
```

**API Key Issues**

```bash
# Verify Spotify credentials
curl -X POST "https://accounts.spotify.com/api/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "grant_type=client_credentials&client_id=YOUR_ID&client_secret=YOUR_SECRET"
```

**Import Errors**

```bash
# Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project"
```

### Getting Help

- Check `/docs` endpoint for API documentation
- View logs with `LOG_LEVEL=DEBUG`
- Use `/system` endpoint for system diagnostics

---

**Built with ❤️ for the music and ML community**
