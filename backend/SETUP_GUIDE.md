# Tensoe ML Setup Guide

## 🚀 Quick Start (5 minutes)

### 1. Environment Setup

Create `.env` file in `backend/` directory:

```bash
# Required for ML ingestion
DATABASE_URL=postgresql://postgres:[password]@db.[project-id].supabase.co:5432/postgres
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
LASTFM_API_KEY=your_lastfm_api_key

# Optional
SUPABASE_URL=https://[project-id].supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
REDIS_URL=redis://localhost:6379/0
```

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
pip install pydantic-settings  # If not already installed
```

### 3. Start Server

```bash
# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test ML Integration

```bash
# In another terminal
python test_ml.py
```

## 🎵 Testing Data Ingestion

### Method 1: API Documentation (Recommended)

1. Open http://localhost:8000/docs
2. Find "Machine Learning" section
3. Try `POST /ml/ingest/album`:
   ```json
   {
     "album_name": "Kind of Blue",
     "artist_name": "Miles Davis"
   }
   ```

### Method 2: Command Line

```bash
# Test single album ingestion
curl -X POST "http://localhost:8000/ml/ingest/album" \
     -H "Content-Type: application/json" \
     -d '{"album_name": "Abbey Road", "artist_name": "The Beatles"}'

# Check stats
curl "http://localhost:8000/ml/stats"
```

### Method 3: Python Script

```python
import requests

# Ingest album
response = requests.post("http://localhost:8000/ml/ingest/album", json={
    "album_name": "Nevermind",
    "artist_name": "Nirvana"
})
print(response.json())

# Get stats
stats = requests.get("http://localhost:8000/ml/stats").json()
print(f"Total tracks: {stats['total_tracks']}")
```

## 📊 Available Endpoints

| Endpoint                | Purpose                       |
| ----------------------- | ----------------------------- |
| `GET /`                 | Health check with system info |
| `GET /docs`             | Interactive API documentation |
| `POST /ml/ingest/album` | Ingest single album           |
| `GET /ml/stats`         | Ingestion statistics          |
| `GET /ml/analytics`     | Genre/mood analytics          |
| `GET /ml/training-data` | Get ML training data          |
| `GET /ml/genres`        | Genre distribution            |
| `GET /ml/health`        | ML service health             |

## 🔧 Troubleshooting

### Import Errors

- **Solution**: The system now has fallback imports. Just make sure your `.env` file has the required API keys.

### Database Connection

- **Check**: Your `DATABASE_URL` in `.env` is correct
- **Test**: Visit `/ml/health` to see if database is connected

### API Keys Missing

- **Spotify**: Get from https://developer.spotify.com/dashboard/
- **Last.fm**: Get from https://www.last.fm/api/account/create

### Server Won't Start

```bash
# Check if port 8000 is in use
netstat -an | grep 8000

# Try different port
uvicorn main:app --reload --port 8001
```

## 🎯 Next Steps

1. **Ingest Data**: Use the endpoints to populate your database
2. **View Analytics**: Check `/ml/analytics` for data insights
3. **Export Data**: Use `/ml/export/training-data` for ML training
4. **Build Models**: Use the feature matrix for your ML model training

## 📈 Sample Albums to Test

```json
[
  ["Kind of Blue", "Miles Davis"],
  ["Abbey Road", "The Beatles"],
  ["The Dark Side of the Moon", "Pink Floyd"],
  ["Nevermind", "Nirvana"],
  ["OK Computer", "Radiohead"]
]
```

Copy this list to `/ml/ingest/batch` for bulk testing!
