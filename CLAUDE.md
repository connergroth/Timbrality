# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Timbre is a machine learning-powered music recommendation engine that integrates Spotify, Last.fm, and Album of the Year (AOTY) data. It's a monorepo with separate backend (FastAPI/Python) and frontend (React/TypeScript) applications.

## Common Development Commands

### Backend (Python/FastAPI)
- **Run development server**: `cd backend && uvicorn main:app --reload`
- **Install dependencies**: `cd backend && pip install -r requirements.txt`
- **Run tests**: `cd backend && pytest`
- **Code formatting**: `cd backend && black .`
- **Linting**: `cd backend && flake8`
- **Database migrations**: `cd backend && alembic upgrade head`

### Frontend (React)
- **Main landing site** (Vite + React):
  - Development: `cd frontend && npm run dev`
  - Build: `cd frontend && npm run build`
  - Lint: `cd frontend && npm run lint`

- **Next.js app** (authentication flow):
  - Development: `cd frontend/app && npm run dev`
  - Build: `cd frontend/app && npm run build`
  - Lint: `cd frontend/app && npm run lint`

## Architecture Overview

### Backend Structure
- **FastAPI application** with async/await support
- **Multi-tier caching**: Redis primary + in-memory fallback
- **Rate limiting**: 30 requests/minute default via SlowAPI
- **Database**: PostgreSQL with SQLAlchemy ORM and Alembic migrations
- **Web scraping**: Playwright browser automation for AOTY data
- **External APIs**: Spotify, Last.fm, and custom AOTY scraper

### Key Backend Components
- `/routes/`: API endpoints (album, user, metrics, ML, scraper)
- `/services/`: Business logic (AOTY scraper, Spotify, Last.fm, ML)
- `/ingestion/`: Data pipeline for fetching and normalizing music data
- `/utils/`: Shared utilities (caching, metrics, database, matching)
- `/models/`: Database schema and Pydantic models
- `/middleware/`: Custom middleware for AOTY integration

### Frontend Structure
- **Main site**: Vite + React + shadcn/ui components
- **Auth app**: Next.js application for OAuth flows
- **Shared UI**: shadcn/ui component library with Tailwind CSS
- **State management**: React Context for theme, Supabase for auth

### Database & External Services
- **PostgreSQL**: User data, music metadata, recommendation history
- **Redis**: Caching layer for API responses and model results
- **Supabase**: Authentication and user management
- **Spotify API**: User listening history and playlists
- **Last.fm API**: Listening data and social features
- **AOTY scraper**: Album ratings, reviews, and similar albums

## Development Workflow

1. **Environment setup**: Both backend and frontend require separate `.env` files
2. **Database**: Run PostgreSQL locally or use remote instance
3. **Cache**: Redis is optional (falls back to in-memory)
4. **API keys**: Required for Spotify, Last.fm, and Supabase integrations

## Testing

- **Backend**: pytest with async support (`pytest-asyncio`)
- **Frontend**: No specific test framework configured yet
- **Test files**: Located in `/backend/test_*.py` and `/tests/` directory

## Configuration

### Backend Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN`: Redis cache
- `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET`: Spotify API
- `LASTFM_API_KEY` / `LASTFM_API_SECRET`: Last.fm API
- `SUPABASE_URL` / `SUPABASE_ANON_KEY`: Supabase integration

### Frontend Environment Variables
- Supabase configuration for authentication
- API endpoint URLs for backend communication

## Data Pipeline

The ingestion pipeline (`/backend/ingestion/`) handles:
1. **Spotify data**: User playlists, saved tracks, listening history
2. **Last.fm data**: Scrobbles, loved tracks, similar artists
3. **AOTY data**: Album ratings, reviews, tags, similar albums
4. **Normalization**: Matching and deduplication across sources
5. **Database insertion**: Structured storage in PostgreSQL

## Machine Learning

- **Hybrid model**: Collaborative filtering + content-based filtering
- **External repo**: [Timbral](https://github.com/connergroth/timbral)
- **Model serving**: FastAPI endpoints in `/routes/ml_routes.py`
- **Features**: User-item interactions, audio features, tag vectors

## Performance Considerations

- **Async everywhere**: Full async/await support in backend
- **Browser reuse**: Shared Playwright instances for scraping
- **Intelligent caching**: Multi-tier with appropriate TTLs
- **Rate limiting**: Prevents abuse and ensures fair usage
- **Connection pooling**: Efficient database connection management