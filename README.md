<div align="center">
  <img src="https://github.com/user-attachments/assets/ff6a407e-cefa-4855-8a90-22ae328b8e95" alt="Timbre" width="350px" />
  <h1>Timbre - AI Powered Music Discovery</h1>
</div>

> **Timbre** — a machine learning-powered music recommendation engine that uses AI agents to create personalized music experiences.

Timbre is an intelligent music recommendation platform that combines data from **Spotify**, **Last.fm**, and **Album of the Year (AOTY)** to provide personalized music suggestions through conversational AI agents. The platform features a hybrid recommendation system and modern web interface built with React and Next.js.

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

### Frontend 
- **Main site**: Vite + React + shadcn/ui components
- **Auth app**: Next.js application for OAuth flows
- **State management**: React Context + Supabase auth

### Key Components
- `/backend/agent/`: AI agent core, tools, and NLP processing
- `/backend/routes/`: API endpoints (agent, albums, playlists, users)
- `/backend/services/`: Business logic (Spotify, Last.fm, ML, AOTY)
- `/backend/ingestion/`: Data pipeline for music metadata
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
- **Hybrid Recommender** – Collaborative + content-based filtering
- **External ML Models** – Integration with [Timbral](https://github.com/connergroth/timbral) recommendation engine

---

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+
- PostgreSQL
- Redis (optional, falls back to in-memory cache)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup
```bash
# Main site
cd frontend && npm install && npm run dev

# Auth app  
cd frontend/app && npm install && npm run dev
```

### Environment Variables
Configure `.env` files in both `backend/` and `frontend/app/` directories with your API keys for Spotify, Last.fm, and Supabase.

---

## Current Status

✅ **Completed:**
- AI agent architecture with conversational interface
- Multi-platform data integration (Spotify, Last.fm, AOTY)
- Modern React/Next.js frontend with chat interface
- FastAPI backend with caching and rate limiting

🚧 **In Progress:**
- ML model integration and recommendation refinement
- Enhanced playlist management features
- Performance optimizations and deployment preparation
