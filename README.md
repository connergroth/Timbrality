<div align="center">
  <img src="https://github.com/user-attachments/assets/5498a533-8497-46e6-a4f5-46d8cfacd6f3" alt="Timbrality" width="350px" />
  <h1>Timbrality - AI Powered Music Discovery</h1>
</div>

> **Timbrality** — a machine learning-powered music recommendation engine that uses AI agents to create personalized music experiences.

Timbrality is an intelligent music recommendation platform that combines data from **Spotify**, **Last.fm**, and **Album of the Year (AOTY)** to provide personalized music suggestions through conversational AI agents. The platform features a hybrid recommendation system powered by the **Timbral** ML engine and modern web interface built with React and Next.js.

---

## Features

- **🤖 AI-Powered Music Agent**  
  Conversational AI agent that understands music preferences and provides intelligent recommendations through natural language interactions.

- **🎧 Personalized Recommendations**  
  Hybrid recommendation system combining collaborative filtering and content-based approaches using listening behavior and audio features.

- **🔗 Multi-Platform Integration**  
  Seamlessly connects with **Spotify**, **Last.fm**, and **Album of the Year** to gather comprehensive music data and preferences.

- **📱 Modern Web Interface**  
  Clean, responsive UI built with React/Next.js featuring chat interface, playlist management, and real-time music discovery.

- **🎵 Smart Playlist Creation**  
  AI-generated playlists with Spotify integration for seamless music discovery and playlist management.

- **⚡ High-Performance Backend**  
  FastAPI-powered backend with multi-tier caching (Redis + in-memory), rate limiting, and async processing.

- **📊 Rich Music Metadata**  
  Enhanced with AOTY ratings, reviews, tags, and similar album data through sophisticated CloudScraper-based pipeline with rating count extraction.

---

## Architecture Overview

### Backend (FastAPI)
- **Multi-tier caching**: Redis primary + in-memory fallback
- **Rate limiting**: 30 requests/minute via SlowAPI  
- **Database**: PostgreSQL with SQLAlchemy ORM and Alembic migrations
- **Web scraping**: CloudScraper with async processing for comprehensive AOTY data extraction
- **AI Agent**: NLP processor with tool registry for music recommendations

### ML Service (Timbral Engine)
- **Hybrid recommendation engine**: NMF collaborative + BERT content-based filtering
- **Dedicated FastAPI service**: Port 8001 with ML-specific endpoints
- **Model serving**: Redis-cached recommendations with explainability
- **HTTP integration**: Proxied through main backend at `/timbral/*` routes

### Frontend 
- **Main site**: Vite + React + shadcn/ui components
- **Auth app**: Next.js application for OAuth flows
- **State management**: React Context + Supabase auth

### Key Components
- `/backend/agent/`: AI agent core, tools, and NLP processing
- `/backend/routes/`: API endpoints (agent, albums, playlists, users, timbral)
- `/backend/services/`: Business logic (Spotify, Last.fm, ML, AOTY)
- `/backend/ingestion/`: Data pipeline for music metadata
- `/ml/timbral/`: Timbral ML engine (models, training, inference)
- `/frontend/app/`: Next.js authentication and chat interface

---

## Tech Stack

### 💻 Backend Technologies

- **FastAPI** – Async Python web framework with automatic OpenAPI docs
- **PostgreSQL + SQLAlchemy** – Relational database with async ORM
- **Redis** – High-performance caching layer
- **CloudScraper** – Advanced web scraping with anti-bot protection bypass for AOTY data
- **Pydantic** – Data validation and serialization

### 📊 Data Sources & APIs

- **Spotify Web API** – User listening data, playlists, and audio features
- **Last.fm API** – Scrobbling data and music discovery
- **AOTY Custom Scraper** – Album ratings, reviews, rating counts, and comprehensive metadata
- **Supabase** – Authentication and user management

### 🤖 AI & Machine Learning

- **AI Agent Architecture** – Tool-based agent for music recommendations
- **NLP Processing** – Natural language understanding for music queries
- **Timbral Engine** – Dedicated ML microservice with hybrid recommendation engine
- **NMF Collaborative Filtering** – User-item matrix factorization for personalized suggestions
- **BERT Content-Based Filtering** – Semantic understanding of music metadata and genres
- **Model Explainability** – Built-in recommendation reasoning and explanations

---

## Model Design

### 🔸 Collaborative Filtering (CF)

- Built from play counts and listening behavior
- Uses Non-negative Matrix Factorization (NMF)
- Predicts latent user-track affinities

### 🔹 Content-Based Filtering (CBF)

- Embeds mood, genre, and tags using Sentence-BERT
- Computes track similarity with cosine distance
- Useful for cold-starts and fallback recs

### 🔶 Hybrid Fusion

- Weighted blending of CF + CBF scores
- Tunable or learnable fusion logic
- Produces rich, explainable recs per user or seed

---

## AOTY Data Scraper

Timbrality includes a sophisticated web scraper that extracts rich music metadata from **Album of the Year (AOTY)**, one of the most comprehensive music databases available. This custom scraper enhances the platform's recommendation capabilities with detailed album ratings, reviews, and metadata.

### 🎯 What It Scrapes

**Albums:**
- User scores and rating counts (e.g., "Based on 37,040 ratings")
- Critic reviews from major publications
- Popular user reviews with like counts
- Genre tags and metadata
- Similar album recommendations
- "Must Hear" designations

**Artists:**
- Overall user ratings and rating counts
- Biography and formation details
- Geographic location data
- Complete discography listings
- Genre classifications

**Tracks:**
- Individual track ratings and rating counts
- Track-level metadata and features
- Featured artist information
- Track length and positioning data

### 🛠 Technical Implementation

**Web Scraping Engine:**
- **CloudScraper** for bypassing anti-bot protection
- **BeautifulSoup** for robust HTML parsing
- **Async/await** processing for high performance
- **Custom retry logic** with exponential backoff
- **Rate limiting** to respect AOTY's servers

**Data Models:**
```python
class Album(BaseModel):
    title: str
    artist: str
    user_score: Optional[float]
    num_ratings: int              
    tracks: List[Track]
    critic_reviews: List[CriticReview]
    popular_reviews: List[AlbumUserReview]
    
class Track(BaseModel):
    title: str
    rating: Optional[int]
    num_ratings: int              
    featured_artists: List[str]
```

**API Endpoints:**
```bash
GET /scraper/album?artist=Radiohead&album=OK+Computer
GET /scraper/similar?artist=Radiohead&album=OK+Computer
GET /scraper/artist?name=Radiohead
```

### 🔄 Data Pipeline Integration

**Automated Population:**
```bash
# Add rating count columns to existing tables
psql $DATABASE_URL -f backend/add_aoty_rating_counts.sql

# Populate rating counts for all entities
python backend/populate_aoty_rating_counts.py --type all --batch-size 10
```

**Database Enhancement:**
- Adds `aoty_num_ratings` columns to albums, artists, and tracks tables
- Batch processing with configurable limits
- Resume capability for interrupted runs
- Error handling and logging for production use

**Caching Strategy:**
- **Redis caching** for scraped data with configurable TTL
- **In-memory fallback** when Redis is unavailable
- **Smart cache keys** based on artist/album combinations
- **Cache warming** for popular albums and artists

### 🎵 Use Cases

**Recommendation Enhancement:**
- Weight recommendations by AOTY rating popularity
- Surface critically acclaimed but undiscovered albums
- Filter by minimum rating thresholds
- Include review-based reasoning in AI responses

**Music Discovery:**
- "Similar Albums" recommendations from AOTY's algorithm
- Genre-based exploration using AOTY's tagging system
- Critical consensus analysis for new releases
- User review sentiment for recommendation explanations

**Data Quality:**
- Cross-reference Spotify/Last.fm data with AOTY metadata
- Resolve artist/album name discrepancies
- Enrich sparse metadata with comprehensive AOTY details
- Validate music catalog completeness

---

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+
- PostgreSQL
- Redis (optional, falls back to in-memory cache)

### Full Stack Setup (Docker)
```bash
docker-compose up
```

### Manual Setup

#### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload  # Port 8000
```

#### ML Service
```bash
cd ml
pip install -r requirements.txt
python main.py  # Port 8001
```

#### Frontend
```bash
# Main site
cd frontend && npm install && npm run dev  # Port 3001

# Auth app  
cd frontend/app && npm install && npm run dev  # Port 3000
```

### Environment Variables
Configure `.env` files in `backend/`, `ml/`, and `frontend/app/` directories with your API keys for Spotify, Last.fm, Supabase, and OpenAI.

---

## Current Status

✅ **Completed:**
- AI agent architecture with conversational interface
- Multi-platform data integration (Spotify, Last.fm, AOTY)
- Modern React/Next.js frontend with chat interface
- FastAPI backend with caching and rate limiting

🚧 **In Progress:**
- Enhanced playlist management features
- Performance optimizations and deployment preparation
- Advanced ML model training and fine-tuning
