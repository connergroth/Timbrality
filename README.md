# Sonance 🎵  
## [In-Progress]

**Sonance** is a machine learning–powered music recommendation engine that generates personalized suggestions for songs, albums, and artists.  
It integrates listening data from **Spotify** and **Last.fm**, and enriches recommendations using **user ratings, tags, and review metadata from albumoftheyear.org (AOTY)** via a custom FastAPI-powered scraper.

---

## ✨ Features

- **🎧 Personalized Music Recommendations**  
  Uses a hybrid of collaborative filtering and content-based filtering to suggest relevant songs and albums based on listening behavior and track metadata.

- **🔗 Spotify & Last.fm Integration**  
  Fetches real user listening history, favorite artists, and playlists for input into the recommendation engine.

- **📈 AOTY Ratings & Similar Albums**  
  Leverages the custom [AOTY-API](https://github.com/connergroth/aoty-api) to enrich music data with user scores, tags, and similar albums to influence recommendation quality and scoring.

- **🧠 Music Compatibility Scores**  
  Calculates similarity between users based on their listening profiles, useful for future social discovery features.

- **⚡ Redis Caching**  
  Caches model results, recommendation snapshots, and API responses for performance.

- **🗃 PostgreSQL Database**  
  Persists user data, song metadata, tag vectors, and recommendation history.

- **📡 Real-Time Data Sync**  
  Dynamically fetches and processes new listening history to reflect evolving user preferences.

---

## 🛠 Roadmap & Milestones

- [x] Integrate Spotify, Last.fm, and AOTY data sources  
- [x] Build AOTY-API to scrape and serve album metadata & reviews  
- [ ] Insert and normalize all data into PostgreSQL  
- [ ] Train collaborative filtering & content-based models  
- [ ] Build hybrid scoring engine and recommendation API  
- [ ] Integrate ChatGPT for explainability and playlist logic  
- [ ] Build a React-based frontend UI  
- [ ] Support playlist generation and song-based discovery tools  
- [ ] Deploy backend and frontend for live usage

---

## ⚙️ Tech Stack

### 💻 Backend
- **Python (FastAPI)** – High-performance API for recommendations  
- **SQLAlchemy + Alembic** – ORM & migration tools for PostgreSQL  
- **PostgreSQL** – Stores user listening history, track metadata, and model results  
- **Redis** – Caches frequently accessed data for faster response times  
- **Docker** – Containerized app for reproducible dev & deployment

### 📊 Data Sources
- **Spotify API** – Fetches user listening data, saved tracks, and playlists  
- **Last.fm API** – Pulls listening history, neighbor data, and tag clouds  
- **[AOTY-API](https://github.com/connergroth/aoty-api)** – Custom scraper for Albumoftheyear.org album ratings, reviews, tags, and similar albums

### 🤖 Machine Learning & Recommendation
- **Collaborative Filtering (NMF)** – Learns user-item relationships from listening data  
- **Content-Based Filtering (TF-IDF + cosine similarity)** – Recommends similar tracks using tag vectors  
- **Hybrid Model** – Blends both approaches for highly personalized recommendations  
- **GPT-4 Integration** – Powers explanations, playlist naming, and feedback analysis

---

## 📌 Status

> Sonance is currently in development. The backend ingestion pipeline and AOTY integration are in progress.  
> Next steps include ML model training, hybrid rec logic, and frontend implementation.



