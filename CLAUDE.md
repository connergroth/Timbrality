# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Timbrality is a machine learning-powered music recommendation engine that integrates Spotify, Last.fm, and Album of the Year (AOTY) data. It's a clean, well-organized monorepo with microservice architecture featuring separate backend (FastAPI/Python), ML service (Timbral), and frontend (React/TypeScript) applications.

**Architecture**: Clean microservice design with consolidated components, no redundant code, and optimal structure for both development and production deployment.

## Common Development Commands

### Backend (Python/FastAPI)
- **Run development server**: `cd backend && uvicorn main:app --reload`
- **Install dependencies**: `cd backend && pip install -r requirements.txt`
- **Run tests**: `cd backend && pytest`
- **Code formatting**: `cd backend && black .`
- **Linting**: `cd backend && flake8`
- **Database migrations**: `cd backend && alembic upgrade head`

### ML Service (Timbral)
- **Run ML service**: `cd ml && python main.py`
- **Install dependencies**: `cd ml && pip install -r requirements.txt`
- **Train models**: `cd ml && python scripts/train_models.py`
- **Populate Redis cache**: `cd ml && python scripts/populate_redis.py`

### Frontend (React)
- **Main landing site** (Vite + React):
  - Development: `cd frontend && npm run dev`
  - Build: `cd frontend && npm run build`
  - Lint: `cd frontend && npm run lint`

- **Next.js app** (authentication flow):
  - Development: `cd frontend/app && npm run dev`
  - Build: `cd frontend/app && npm run build`
  - Lint: `cd frontend/app && npm run lint`

### Full Stack Development
- **All services**: `docker-compose up` (from project root)
- **Development mode**: `docker-compose -f docker-compose.dev.yml up`

## Architecture Overview

### Microservice Architecture
- **Backend** (port 8000): Main FastAPI application
- **ML Service** (port 8001): Timbral recommendation engine 
- **Frontend Auth** (port 3000): Next.js authentication flow
- **Frontend Main** (port 3001): Vite landing site
- **Redis** (port 6379): Shared caching layer
- **PostgreSQL**: Supabase-hosted database

### Backend Structure
- **FastAPI application** with async/await support
- **Multi-tier caching**: Redis primary + in-memory fallback
- **Rate limiting**: 30 requests/minute default via SlowAPI
- **Database**: PostgreSQL with SQLAlchemy ORM and Alembic migrations
- **Web scraping**: Playwright browser automation for AOTY data
- **External APIs**: Spotify, Last.fm, and custom AOTY scraper
- **ML Integration**: HTTP proxy to Timbral ML service

### Key Backend Components
- `/routes/`: API endpoints (album, user, metrics, ml, scraper, timbral, agent, playlist, spotify)
- `/services/`: Business logic (AOTY scraper, Spotify, Last.fm, ML, collaborative filtering)
- `/ingestion/`: Data pipeline for fetching and normalizing music data
- `/utils/`: Shared utilities (caching, metrics, database, matching)
- `/models/`: Database schema and Pydantic models
- `/middleware/`: Custom middleware for AOTY integration
- `/tests/`: Consolidated test suite for all backend functionality

### ML Service (Timbral) Structure
- **Hybrid recommendation engine**: NMF collaborative + BERT content-based filtering
- **FastAPI service**: Dedicated ML endpoints with async processing
- **Model components**: `/timbral/models/` (NMF, BERT encoder, hybrid fusion)
- **Core logic**: `/timbral/core/` (recommendation service, scoring, explainability)
- **Data processing**: `/timbral/logic/` (training pipeline, embedding builder)
- **API layer**: `/timbral/api/` (routes and request/response models)
- **Shared utilities**: `/timbral/utils/` (Redis connector, evaluation, file utils)

### Frontend Structure
- **Main landing site**: Vite + React + shadcn/ui components (port 3001)
- **Authentication app**: Next.js application for OAuth flows (port 3000)
- **Shared UI**: shadcn/ui component library with Tailwind CSS
- **State management**: React Context for theme, Supabase for auth
- **Clean component architecture**: No duplicates, well-organized structure

### Database & External Services
- **PostgreSQL**: User data, music metadata, recommendation history
- **Redis**: Caching layer for API responses and model results
- **Supabase**: Authentication and user management
- **Spotify API**: User listening history and playlists
- **Last.fm API**: Listening data and social features
- **AOTY scraper**: Album ratings, reviews, and similar albums

## Development Workflow

1. **Environment setup**: Backend, ML service, and frontend apps require `.env` files
2. **Database**: PostgreSQL via Supabase (remote instance)
3. **Cache**: Redis shared across backend and ML service
4. **API keys**: Required for Spotify, Last.fm, Supabase, and OpenAI integrations
5. **Clean architecture**: All redundant files removed, optimized for development and production

## Testing

- **Backend**: pytest with async support (`pytest-asyncio`)
- **Test structure**: Consolidated in `/backend/tests/` directory
- **Test coverage**: Database, API endpoints, services, and integrations
- **Frontend**: No specific test framework configured yet
- **Clean test organization**: All standalone test files removed, unified structure

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

## Machine Learning (Timbral Service)

- **Hybrid recommendation engine**: NMF collaborative filtering + BERT content-based filtering
- **Dedicated microservice**: Runs on port 8001, communicates via HTTP
- **Model serving**: FastAPI endpoints exposed through `/timbral/*` proxy routes
- **Features**: User-item interactions, Spotify audio features, genre/mood embeddings
- **Training pipeline**: Automated model retraining with new interaction data
- **Explainability**: Built-in recommendation explanation system
- **Caching**: Redis-based precomputed recommendations for fast serving

### ML API Endpoints (via /timbral/ proxy)
- `GET /timbral/recommendations/{user_id}` - User-specific recommendations
- `POST /timbral/recommendations` - Flexible recommendation requests
- `GET /timbral/similar/{item_id}` - Similar item discovery
- `GET /timbral/explain/{user_id}/{item_id}` - Recommendation explanations
- `POST /timbral/feedback` - User feedback for model improvement
- `POST /timbral/train` - Trigger model retraining
- `GET /timbral/health` - ML service health check

## Performance Considerations

- **Async everywhere**: Full async/await support in backend and ML service
- **Microservice architecture**: Independent scaling of backend, ML, and frontend services
- **Browser reuse**: Shared Playwright instances for scraping efficiency
- **Intelligent caching**: Multi-tier Redis caching with appropriate TTLs
- **Rate limiting**: Prevents abuse and ensures fair usage across services
- **Connection pooling**: Efficient database connection management
- **Clean codebase**: Optimized structure with no redundant files or dependencies

### ✅ **Current Clean Structure**
```
Timbrality/
├── backend/              # FastAPI service (port 8000)
├── frontend/             # Vite landing site (port 3001)
├── frontend/app/         # Next.js auth app (port 3000)
├── ml/                   # Timbral ML engine (port 8001)
├── docker-compose.yml    # Service orchestration
└── README.md            # Project overview
```

### ✅ **Quality Assurance**
- **No duplicate code**: All redundant components removed
- **Single source of truth**: Each feature has one clear implementation
- **Production-ready**: Optimized for both development and deployment
- **Recruiter-friendly**: Professional, well-organized codebase structure