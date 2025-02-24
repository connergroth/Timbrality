# Sonance 🎵
## [In-Progress]  

A powerful music recommendation engine that integrates Spotify, Last.fm, and Albumoftheyear.org data to provide personalized recommendations for songs, albums, and artists.

## ✨ Features
- **Personalized Music Recommendations** – Uses listening history, trending music, and collaborative filtering to suggest songs & albums.  
- **Spotify & Last.fm Integration** – Fetches listening data from both platforms for hybrid recommendation models.  
- **Redis Caching** – Stores frequently accessed data for faster responses.  
- **PostgreSQL Database** – Persists user interactions, recommendations, and metadata.  
- **Music Compatibility Scores** – Calculates similarity between users based on listening history.  
- **Track & Album Ratings** – Integrates Albumoftheyear.org (AOTY) for critic/user scores to prioritize highly-rated music.  
- **Real-Time Updates** – Fetches & processes data dynamically to reflect user preferences.  

## 🛠 Roadmap & Future Improvements
- [x] Implement data fetching with APIs  
- [ ] Insert data into PostgreSQL database
- [ ] Train recommendation engine  
- [ ] Build a frontend using React  
- [ ] Support music discovery based on user-generated playlists  

# Tech Stack
## 💻 Backend
- Python (FastAPI) – High-performance API backend.  
- PostgreSQL – Stores persistent data like user profiles, listening history, and recommendations.  
- Redis – Caches frequently accessed data for fast retrieval.  
- SQLAlchemy – ORM for database operations.  
- Alembic – Handles database migrations.  
## 📊 Data Sources  
- Spotify API – Fetches user listening data & playlists.  
- Last.fm API – Retrieves listening history & trends.  
- Albumoftheyear.org – Scrapes album & track ratings for better recommendation scoring.  
## 🤖 Machine Learning & Recommendation  
- Collaborative Filtering – Suggests music based on user interactions.  
- Content-Based Filtering – Recommends similar songs/albums using track metadata.  
- Hybrid Model – Combines both for highly accurate recommendations.  
## ☁️ Caching & Performance
- Redis – Stores user recommendations, trending songs, and listening history snapshots.  
- Docker – Containerized environment for easier deployment.


## 📜 License
Sonance is open-source under the MIT License.

## 👨‍💻 Author
💡 Created by [Conner Groth](https://www.linkedin.com/in/conner-groth-978228260/)


