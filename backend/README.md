# Timbre Data Ingestion Pipeline

A high-performance, async data ingestion pipeline for the Timbre music recommendation engine. Pulls 10,000 tracks from Last.fm, enriches them with Spotify and AOTY metadata, deduplicates using canonical IDs, and bulk-loads into PostgreSQL via Supabase.

## 🎯 Overview

This pipeline implements the complete data flow for Timbre's ML-powered music recommendations:

1. **Last.fm Integration**: Pulls user's top tracks with rate limiting and caching
2. **Spotify Enrichment**: Adds basic metadata, popularity, and album information 
3. **AOTY Scraping**: Extracts album ratings and genres with anti-bot measures
4. **Intelligent Deduplication**: Uses canonical IDs (ISRC > Spotify ID > hash)
5. **Bulk Database Operations**: Efficient PostgreSQL insertion via Supabase

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Last.fm   │───▶│   Pipeline   │───▶│  Supabase   │
│  (30k raw)  │    │ Orchestrator │    │ PostgreSQL  │
└─────────────┘    │              │    └─────────────┘
                   │              │
┌─────────────┐    │              │    ┌─────────────┐
│   Spotify   │───▶│  (10k final) │───▶│    Redis    │
│  Metadata   │    │              │    │   Cache     │
└─────────────┘    │              │    └─────────────┘
                   │              │
┌─────────────┐    │              │
│    AOTY     │───▶│              │
│  Ratings    │    │              │
└─────────────┘    └──────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- PostgreSQL (or Supabase account)
- Redis (optional, for caching)
- API keys for Last.fm, Spotify, and Supabase

### Installation

1. **Clone and setup**:
   ```bash
   cd backend
   poetry install
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

3. **Database setup**:
   ```bash
   # Apply database schema
   psql -f migrations/enhanced_schema.sql
   ```

4. **Run pipeline**:
   ```bash
   poetry run python scripts/run_pipeline.py run
   ```

### Docker Deployment

```bash
# Development with local services
docker-compose -f docker-compose.ingestion.yml up

# Production deployment
docker-compose -f docker-compose.ingestion.yml --profile pipeline up
```

## 📋 Environment Variables

### Required Credentials

```env
# Last.fm API (https://www.last.fm/api)
LASTFM_API_KEY=your_api_key
LASTFM_USERNAME=your_username

# Spotify API (https://developer.spotify.com/)
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret

# Supabase (https://supabase.com/)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### Pipeline Configuration

```env
MAX_SONGS=10000                    # Target song count
SCRAPE_CONCURRENCY=4               # AOTY concurrent requests
SCRAPE_DELAY_SEC=2.0              # Delay between AOTY requests
BATCH_SIZE=50                     # Processing batch size
DB_BATCH_SIZE=2000                # Database insertion batch size

# Rate Limiting (requests per minute)
SPOTIFY_RATE_LIMIT=100
LASTFM_RATE_LIMIT=200
AOTY_RATE_LIMIT=30
```

## 🗄️ Database Schema

The pipeline uses a normalized PostgreSQL schema with canonical IDs:

```sql
-- Core songs table
CREATE TABLE songs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artist TEXT NOT NULL,
    title TEXT NOT NULL,
    canonical_id TEXT UNIQUE NOT NULL,  -- ISRC > Spotify > Hash
    isrc TEXT,
    spotify_id TEXT
);

-- Spotify basic metadata
CREATE TABLE spotify_attrs (
    song_id UUID REFERENCES songs(id),
    duration_ms INTEGER,
    popularity INTEGER,
    album_id TEXT,
    artist_id TEXT,
    album_name TEXT,
    release_date TEXT,
    explicit BOOLEAN,
    track_number INTEGER
);

-- Last.fm user statistics
CREATE TABLE lastfm_stats (
    song_id UUID REFERENCES songs(id),
    playcount INTEGER,
    tags JSONB
);

-- Album of the Year ratings
CREATE TABLE aoty_attrs (
    song_id UUID REFERENCES songs(id),
    user_score NUMERIC,
    rating_count INTEGER,
    tags JSONB,
    genres JSONB
);
```

## 🔧 CLI Usage

### Basic Commands

```bash
# Run complete pipeline
poetry run python scripts/run_pipeline.py run

# Quick test with 100 songs
poetry run python scripts/run_pipeline.py test --quick

# Dry run (no database writes)
poetry run python scripts/run_pipeline.py run --dry-run

# Skip AOTY scraping for speed
poetry run python scripts/run_pipeline.py run --skip-aoty

# Custom song count
poetry run python scripts/run_pipeline.py run --max-songs 5000
```

### Configuration and Validation

```bash
# Show current configuration
poetry run python scripts/run_pipeline.py config

# Validate environment setup
poetry run python scripts/run_pipeline.py validate
```

## 🌐 API Endpoints

When running as a service, the pipeline exposes these endpoints:

```bash
# Health check
GET /health

# Pipeline statistics
GET /stats

# Start pipeline (background)
POST /pipeline/start
{
  "max_tracks": 10000,
  "skip_aoty": false,
  "dry_run": false
}

# Pipeline status
GET /pipeline/status

# Database counts
GET /database/counts
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test categories
poetry run pytest -m unit          # Unit tests only
poetry run pytest -m integration   # Integration tests
poetry run pytest -m "not slow"    # Skip slow tests
```

## 📊 Performance & Monitoring

### Expected Performance

- **Runtime**: ~60 minutes for 10,000 songs (4-core machine)
- **Memory**: ~2GB peak usage
- **Storage**: ~500MB for raw data, logs, and exports

### Rate Limiting

The pipeline respects API rate limits:

- **Last.fm**: 5 requests/second (with jitter)
- **Spotify**: 100 requests/minute 
- **AOTY**: 30 requests/minute (with proxy rotation)

### Monitoring

```bash
# View pipeline logs
tail -f logs/pipeline_summary_*.json

# Check coverage statistics
curl http://localhost:8000/stats

# Monitor Redis cache hits
redis-cli monitor
```

## 🔍 Data Quality Features

### Canonical ID Hierarchy

1. **ISRC** (International Standard Recording Code) - Most reliable
2. **Spotify ID** - Platform specific but stable
3. **Hash** - Fallback based on normalized artist::title

### Deduplication Logic

- Case-insensitive artist/title matching
- Removes common variations (remaster, live, acoustic)
- Handles "The" prefix and featuring credits
- Selects best version based on metadata completeness and playcount

### Data Validation

- Required field validation before insertion
- Rate limit compliance monitoring
- Coverage percentage tracking
- Sanity checks on final counts

## 🐳 Deployment

### Local Development

```bash
# Install dependencies
poetry install

# Run development server
poetry run uvicorn app.main:app --reload

# Run background pipeline
poetry run python scripts/run_pipeline.py run
```

### Docker Production

```bash
# Build image
docker build -f Dockerfile.ingestion -t timbre-ingestion .

# Run with docker-compose
docker-compose -f docker-compose.ingestion.yml up -d

# Check logs
docker-compose logs -f timbre-ingestion
```

### GitHub Actions

The pipeline includes automated CI/CD:

- **Pull Requests**: Dry-run testing
- **Main Branch**: Limited production run
- **Scheduled**: Nightly full pipeline execution
- **Manual**: Configurable runs via workflow dispatch

## 🔧 Advanced Configuration

### Proxy Setup (for AOTY)

```env
AOTY_PROXY_URL=http://your-proxy:port
```

### Custom Data Directories

```env
DATA_DIR=./custom_data
LOGS_DIR=./custom_logs  
EXPORTS_DIR=./custom_exports
```

### Database Tuning

```env
DB_BATCH_SIZE=5000          # Larger batches for faster insertion
DB_RETRY_ATTEMPTS=5         # More retries for reliability
DB_RETRY_DELAY=2.0          # Longer delays between retries
```

## 📁 Project Structure

```
backend/
├── app/                    # Main application code
│   ├── config.py          # Pydantic settings
│   ├── models.py          # Data models
│   ├── db.py              # Database client
│   ├── main.py            # FastAPI application
│   ├── pipelines.py       # Main orchestrator
│   └── tasks/             # Pipeline modules
│       ├── lastfm.py      # Last.fm integration
│       ├── spotify.py     # Spotify enrichment
│       ├── aoty.py        # AOTY scraping
│       ├── unify.py       # Deduplication logic
│       └── ingest.py      # Database operations
├── scripts/               # CLI utilities
│   └── run_pipeline.py    # Main CLI script
├── migrations/            # Database schema
├── tests/                 # Test suite
├── docker-compose.ingestion.yml
├── Dockerfile.ingestion
├── pyproject.toml         # Poetry dependencies
└── README.md              # This file
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Run tests**: `poetry run pytest`
4. **Check code quality**: `poetry run black . && poetry run flake8`
5. **Submit a pull request**

## 📝 Known Limitations

- **AOTY Rate Limits**: Aggressive rate limiting may require proxy rotation
- **Memory Usage**: Large datasets may require chunked processing
- **API Changes**: External APIs may change without notice
- **Regional Restrictions**: Some APIs have geographic limitations

## 🔮 Future Enhancements

- [ ] Additional music service integrations (Apple Music, Bandcamp)
- [ ] Caching tiers by volatility (7-day vs 24-hour)
- [ ] Redis Streams for distributed processing
- [ ] Real-time pipeline monitoring dashboard
- [ ] ML model integration for data quality scoring
- [ ] Enhanced genre classification using AOTY data

## 📜 License

This project is part of the Timbre music recommendation engine. See the main repository for license information.

## 🆘 Support

- **Issues**: Report bugs and feature requests in the main repository
- **Documentation**: See `/docs` in the main repository
- **API Reference**: Available at `/docs` when running the service

---

**Built with ❤️ for music discovery**
