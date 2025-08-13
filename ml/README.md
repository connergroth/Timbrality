<p align="center">
  <img src="https://github.com/user-attachments/assets/41799144-166f-4f51-989f-a461f1732760" alt="Timbral" width="355"/>
</p>

# Timbral - Hybrid Music Recommendation Engine

**Timbral** is the machine learning engine behind [Timbre](https://github.com/connergroth/timbre), a personalized, explainable music recommendation system. It fuses collaborative filtering, BERT-based content similarity, and user interaction data to deliver smart, human-feeling suggestions — fast.

> _timbral /ˈtɪm.brəl/ — adj.
> Relating to the unique character or quality of sound; in this context, where machine learning meets musical nuance._

---

## 🤖 Overview

This repository contains all ML logic powering Timbral, including:

- User-track interaction modeling
- Track metadata embedding and indexing
- Score fusion and reranking
- Redis-based recommendation serving
- Optional GPT agent hooks for explainability and feedback

---

## 🧠 Model Design

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

## 📂 Project Structure

```bash
timbral-recommender/
├── data/
│   ├── bronze/                # raw interaction & metadata
│   ├── silver/                # cleaned matrices and embeddings
│   └── gold/                  # final user-track matrix, similarity cache
├── timbral/
│   ├── models/                # nmf.py, content_encoder.py, fusion.py
│   ├── api/                   # recommend.py (FastAPI endpoints)
│   ├── core/                  # ranking.py, explain.py, scoring.py
│   ├── logic/                 # train_nmf.py, build_embeddings.py
│   ├── utils/                 # redis_client.py, metrics.py
│   └── config/                # config.yaml, constants
├── scripts/
│   ├── precompute_recs.py
│   └── populate_redis.py
├── notebooks/
│   └── evaluation.ipynb
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the Repo

```bash
git clone https://github.com/connergroth/timbral-recommender.git
cd timbral-recommender
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file:

```env
REDIS_URL=redis://localhost:6379
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_KEY=your-supabase-api-key
```

---

## 💻 Training

```bash
python timbral/logic/train_nmf.py
```

This will:

- Build the user-item interaction matrix
- Train the NMF model
- Save user and track vectors to disk

```bash
python timbral/logic/build_embeddings.py
```

- Generates Sentence-BERT embeddings from metadata
- Saves cosine similarity matrix

---

## 🔍 Inference

```python
from timbral.api.recommend import get_recommendations

recs = get_recommendations(user_id="123", top_k=20)
```

For guests (seeded by track):

```python
get_seed_recommendations(seed_track_id="spotify:abc123")
```

---

## ✖️ Batch Precomputation

```bash
python scripts/precompute_recs.py
```

- Computes top-K per user
- Caches to Redis or Supabase for fast serving

---

## 📊 Evaluation

```bash
python timbral/models/evaluation.py
```

Includes:

- Precision\@k
- Recall\@k
- nDCG

---

## 🏗️ Roadmap

- 🎵 Audio preview + tag embeddings
- 🧠 GPT-powered agent feedback loop
- 💬 Natural language explainability
- 🌟 LightGBM final reranker
- 📜 A/B testing engine

---

## 📰 Credits

Built by [Conner Groth](https://www.connergroth.com) for the Timbral ML system.

Powered by real-world music intelligence from Spotify, Last.fm, and AlbumOfTheYear.org.
