# Tensoe Ingestion Pipeline

This directory contains the complete ingestion pipeline for Tensoe, designed to collect music data from multiple sources (Spotify, Last.fm, AOTY) and normalize it for machine learning training.

## 🎼 Overview

The ingestion pipeline follows this flow:

1. **Spotify**: Get track/album metadata (title, artist, release date, duration, popularity)
2. **Last.fm**: Enrich tracks with tags and moods
3. **AOTY**: Scrape album/track ratings and genres
4. **Normalize**: Merge & align data into unified format
5. **Store**: Insert normalized tracks into Supabase database

## 📁 File Structure

```
ingestion/
├── __init__.py              # Package initialization
├── spotify_fetcher.py       # Spotify API integration
├── lastfm_fetcher.py        # Last.fm API integration
├── aoty_scraper.py          # AOTY scraper wrapper
├── normalizer.py            # Data normalization & cleaning
├── insert_to_supabase.py    # Database operations
├── ingest_runner.py         # Main pipeline orchestrator
├── config.py                # Configuration settings
└── README.md                # This file
```

## 🔧 Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the backend directory with:

```env
# Spotify API Credentials
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret

# Last.fm API Credentials
LASTFM_API_KEY=your_lastfm_api_key
LASTFM_API_SECRET=your_lastfm_api_secret

# Supabase Database Credentials
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# Optional: Logging level
LOG_LEVEL=INFO
```

### 3. Database Setup

The pipeline will automatically create the necessary tables in Supabase. The track schema includes:

```sql
CREATE TABLE tracks (
    id TEXT PRIMARY KEY,                 -- Spotify track ID
    title TEXT NOT NULL,                 -- Track title
    artist TEXT NOT NULL,                -- Artist name
    album TEXT,                          -- Album name
    release_date DATE,                   -- Release date
    duration_ms INTEGER,                 -- Track duration in ms
    popularity INTEGER,                  -- Spotify popularity (0-100)
    genres TEXT[],                       -- Array of genres from AOTY
    moods TEXT[],                        -- Array of moods from Last.fm
    aoty_score REAL,                     -- AOTY rating score
    spotify_url TEXT,                    -- Spotify track URL
    explicit BOOLEAN DEFAULT FALSE,      -- Explicit content flag
    track_number INTEGER,                -- Track number in album
    album_total_tracks INTEGER,          -- Total tracks in album
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## 🚀 Usage

### Single Album Ingestion

```python
from ingestion import run_ingestion

# Ingest a single album
success = run_ingestion("Kid A", "Radiohead")
print(f"Ingestion {'successful' if success else 'failed'}")
```

### Batch Ingestion

```python
from ingestion import run_batch_ingestion

# Ingest multiple albums
album_list = [
    ("Kid A", "Radiohead"),
    ("OK Computer", "Radiohead"),
    ("In Rainbows", "Radiohead")
]

results = run_batch_ingestion(album_list, use_async=True)
print(f"Results: {results['successful']} successful, {results['failed']} failed")
```

### CSV Batch Ingestion

Create a CSV file with album and artist names:

```csv
album_name,artist_name
Metaphorical Music,Nujabes
OK Computer,Radiohead
In Rainbows,Radiohead
Blonde,Frank Ocean
To Pimp a Butterfly,Kendrick Lamar
```

Then run:

```python
from ingestion import run_batch_from_csv

results = run_batch_from_csv("albums.csv", use_async=True)
```

### Command Line Usage

```bash
# Single album
python -m ingestion.ingest_runner "Kid A" "Radiohead"

# From CSV file
python -m ingestion.ingest_runner --csv albums.csv

# Search and ingest
python -m ingestion.ingest_runner --search "Radiohead"
```

### Search and Ingest

```python
from ingestion import search_and_ingest

# Search for albums and ingest top results
results = search_and_ingest("indie rock 2023", limit=10)
```

## 📊 Data Export

Export ingested data for ML training:

```python
from ingestion.insert_to_supabase import export_to_csv, get_training_dataset

# Export to CSV
export_to_csv("training_data.csv", limit=10000)

# Get data as Python objects
training_data = get_training_dataset(limit=5000, offset=0)
```

## 🎯 Unified Track Schema

The normalized track format used for ML training:

```python
{
    'track_id': 'spotify:track:abc123',
    'title': 'Motion Picture Soundtrack',
    'artist': 'Radiohead',
    'album': 'Kid A',
    'release_date': '2000-10-02',
    'duration_ms': 230000,
    'popularity': 65,
    'genres': ['art rock', 'ambient'],
    'moods': ['dreamy', 'melancholy'],
    'aoty_score': 84.5,
    'spotify_url': 'https://open.spotify.com/track/...',
    'explicit': False,
    'track_number': 8,
    'album_total_tracks': 10
}
```

## ⚡ Performance Features

- **Async Processing**: Parallel data fetching for faster ingestion
- **Intelligent Caching**: Reuses AOTY album data for multiple tracks
- **Batch Operations**: Efficient database insertions
- **Error Handling**: Graceful handling of API failures
- **Rate Limiting**: Respects API rate limits
- **Deduplication**: Prevents duplicate track entries

## 🔍 Monitoring

Track ingestion progress and results:

```python
from ingestion.insert_to_supabase import get_track_count

# Get total tracks in database
total_tracks = get_track_count()
print(f"Total tracks in database: {total_tracks}")

# Get tracks by artist
tracks = get_tracks_by_artist("Radiohead", limit=50)

# Get tracks by genre
indie_tracks = get_tracks_by_genre("indie", limit=100)
```

## 🚨 Error Handling

The pipeline includes comprehensive error handling:

- **API Failures**: Continues processing other tracks if one API fails
- **Data Validation**: Validates track data before database insertion
- **Retry Logic**: Retries failed operations with exponential backoff
- **Logging**: Detailed logging for debugging and monitoring

## 🎛 Configuration

Customize the pipeline behavior in `config.py`:

- Rate limits for each API
- Batch sizes for processing
- Maximum tags/genres per track
- Database retry settings
- Export formats

## 📈 Scaling

For large-scale ingestion:

1. **Use Async Mode**: Set `use_async=True` for parallel processing
2. **Batch Processing**: Process albums in batches to manage memory
3. **Rate Limiting**: Respect API limits to avoid throttling
4. **Database Batching**: Insert tracks in batches for efficiency

## 🐛 Troubleshooting

Common issues and solutions:

### API Credentials

Ensure all API credentials are correctly set in `.env` file.

### Rate Limiting

If you hit rate limits, increase delays between requests or reduce batch sizes.

### Database Issues

Check Supabase connection and ensure the tracks table exists.

### Missing Data

Some tracks may not have AOTY scores or Last.fm tags - this is normal.

## 🔮 Future Enhancements

Planned improvements:

- Support for additional music services (Apple Music, Bandcamp)
- Advanced mood detection using audio analysis
- Genre prediction using ML models
- Real-time ingestion from streaming platforms
- Data quality scoring and validation
