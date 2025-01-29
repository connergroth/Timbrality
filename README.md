# Sonance 🎵
A powerful music recommendation engine that integrates Spotify, Last.fm, and Albumoftheyear.org to provide personalized recommendations for songs, albums, and artists.

## ✨ Features
✔ Personalized Music Recommendations – Uses listening history, trending music, and collaborative filtering to suggest songs & albums.
✔ Spotify & Last.fm Integration – Fetches listening data from both platforms for hybrid recommendation models.
✔ Redis Caching – Stores frequently accessed data for faster responses.
✔ PostgreSQL Database – Persists user interactions, recommendations, and metadata.
✔ Music Compatibility Scores – Calculates similarity between users based on listening history.
✔ Track & Album Ratings – Integrates Albumoftheyear.org (AOTY) for critic/user scores to prioritize highly-rated music.
✔ Real-Time Updates – Fetches & processes data dynamically to reflect user preferences.

##🛠 Roadmap & Future Improvements
✅ Implement hybrid recommendation models (Spotify + Last.fm + AOTY)
✅ Enhance caching strategy for better performance
🔜 Build a frontend using React or Next.js
🔜 Integrate user preferences & manual ratings into recommendations
🔜 Support music discovery based on user-generated playlists

## 🛠 Tech Stack
Backend
Python (FastAPI) – High-performance API backend.
PostgreSQL – Stores persistent data like user profiles, listening history, and recommendations.
Redis – Caches frequently accessed data for fast retrieval.
SQLAlchemy – ORM for database operations.
Alembic – Handles database migrations.
Data Sources
Spotify API – Fetches user listening data & playlists.
Last.fm API – Retrieves listening history & trends.
Albumoftheyear.org – Scrapes album & track ratings for better recommendation scoring.
Machine Learning & Recommendation
Collaborative Filtering – Suggests music based on user interactions.
Content-Based Filtering – Recommends similar songs/albums using track metadata.
Hybrid Model – Combines both for highly accurate recommendations.
Caching & Performance
Redis – Stores user recommendations, trending songs, and listening history snapshots.
Docker – Containerized environment for easier deployment.
## 📦 Installation
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/sonance.git
cd sonance
2️⃣ Set Up Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate      # For Windows
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Set Up Environment Variables
Create a .env file with the following variables:

env
Copy
Edit
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/sonance
REDIS_URL=your_redis_url
REDIS_TOKEN=your_redis_token
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
LASTFM_API_KEY=your_lastfm_api_key
5️⃣ Start PostgreSQL & Redis (via Docker)
bash
Copy
Edit
docker-compose up -d
6️⃣ Apply Migrations
bash
Copy
Edit
alembic upgrade head
7️⃣ Run the Application
bash
Copy
Edit
uvicorn app.main:app --reload
8️⃣ API Documentation (Swagger UI)
Once running, visit:
🔗 http://127.0.0.1:8000/docs

##🚀 Usage
🎧 Fetch User Listening History
bash
Copy
Edit
curl -X GET "http://127.0.0.1:8000/listening-history?user_id=1"
##🔥 Get Personalized Recommendations
bash
Copy
Edit
curl -X GET "http://127.0.0.1:8000/recommendations?user_id=1"
##🎼 Search for a Song or Album
bash
Copy
Edit
curl -X GET "http://127.0.0.1:8000/search?q=Pink Floyd"
##🧪 Running Tests
Run Unit Tests
bash
Copy
Edit
pytest tests --cov=app
Check Code Quality
bash
Copy
Edit
flake8 .


##📜 License
Sonance is open-source under the MIT License.

##👥 Author
💡 Created by Conner Groth
Want to contribute? Feel free to submit a PR!

