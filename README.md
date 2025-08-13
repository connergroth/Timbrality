<div align="center">
  <img src="https://github.com/user-attachments/assets/ff6a407e-cefa-4855-8a90-22ae328b8e95" alt="Timbrality" width="350px" />
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
  Enhanced with AOTY ratings, reviews, tags, and similar album data through custom web scraping pipeline.

---

## Architecture Overview

### Backend (FastAPI)
- **Multi-tier caching**: Redis primary + in-memory fallback
- **Rate limiting**: 30 requests/minute via SlowAPI  
- **Database**: PostgreSQL with SQLAlchemy ORM and Alembic migrations
- **Web scraping**: Playwright browser automation for AOTY data
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
- **Playwright** – Browser automation for AOTY web scraping
- **Pydantic** – Data validation and serialization

### 📊 Data Sources & APIs

- **Spotify Web API** – User listening data, playlists, and audio features
- **Last.fm API** – Scrobbling data and music discovery
- **AOTY Custom Scraper** – Album ratings, reviews, and metadata
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
