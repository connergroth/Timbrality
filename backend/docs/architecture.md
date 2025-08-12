1. Dataset Pipeline
Step	What Happens	Key Tables / Buckets
1.1 Pull Last.fm data	Cron / user-triggered job hits user.getRecentTracks, getTopArtists/Tracks, library.getTracks. Store raw JSON ➜ S3-style bucket, then upsert to DB.	user_tracks_raw, user_scrobbles_raw
1.2 Pull Spotify metadata	For every unique track (artist+title fingerprint) call: search?q={...}&type=track, then audio-features (if allowed). Persist.	tracks_spotify, audio_features_spotify
1.3 Scrape AOTY	Existing AOTY-API scraper dumps album & review JSON. Publish Kafka/Redis stream ➜ consumer writes.	albums_aoty, album_reviews_aoty
1.4 Canonical track mapping	Deterministic hash on artist-title-duration to resolve duplicates. Produce track_id used everywhere.	tracks_core
1.5 Tag aggregation	Deduplicate & weight tags from Last.fm + AOTY + Spotify genres.	track_tags
1.6 Feature snapshot	Materialized view joins everything into a single row per track_id.	track_feature_view

2. Data-Enrichment Layer
Text embeddings

Sentence-BERT (“all-mpnet-base-v2”) on: track tags, album description, AOTY review snippets.

Store 768-d vector ➜ track_text_embedding.

Predicted mood/energy (Model 1)

Inference job runs nightly on any track lacking pred_energy/pred_valence.

Output floats 0-1 + categorical flags (upbeat, chill, dark, …).

Persist ➜ track_mood_pred.

Track vector assembly
Concatenate / PCA-reduce: text-embedding || predicted-mood || normalized audio features (if Spotify gave them).
→ 256-d track_vector stored in Redis for ANN search (eg, Milvus or Redis Vector).

3. Model 1 – Mood / Energy Predictor
Item	Details
Training data	Million Song Dataset + Last.fm tag subset. Filter tracks with ≥3 mood tags and valid Echo-Nest energy/valence.
Input features	Bag-of-tags TF-IDF (20k vocab)
Genre one-hot (1 k)
Artist popularity bucket
Year bucket
Targets	energy, valence (regression) and 6 one-vs-rest mood labels (multi-label clf).
Model	XGBoost (n_estimators=800, max_depth=10) for regression branch; LightGBM classifier for mood tags.
Offline eval	5-fold CV: RMSE ≤ 0.10 on energy; micro-F1 ≥ 0.62 on mood tags.
Outputs	pred_energy, pred_valence, prob_upbeat, prob_chill, etc.
Artifacts	model_energy.pkl, model_valence.pkl, model_mood.bin + requirements.txt.
Serving	Lightweight FastAPI internal endpoint (/internal/mood/predict) loaded once per worker.

4. Model 2 – Hybrid Recommender
4.1 Collaborative-Filtering Component
Build implicit feedback matrix (user × track) where confidence = log(1+playcount).

Use Alternating Least Squares (implicit library) with 128-d latent factors.

4.2 Content-Similarity Component
KNN (FAISS) over track_vector for cold-start & diversification.

4.3 Rank Fusion
makefile
Copy
Edit
score = α · CF_score
      + β · cosine(track_vector, user_vector)
      + γ · (1 - popularity_penalty)
Start with α=0.6, β=0.3, γ=0.1; tune offline.

4.4 User Vector Construction
If Last.fm history exists: mean of CF latent vectors weighted by playcount.

Else if Spotify Top N exists: same using those tracks.

Else: cold-start vector from seed picks / liked songs UI.

5. Search & NLP Layer
Query encoder – same SBERT model as tracks.

ANN search against track_vector + lexical filter (genre, year).

Optional reranker: cross-encoder (ms-marco-MiniLM-L-6-v2) for top 100 candidates.

6. API Surface (FastAPI, GraphQL sub-layer)
Route	Purpose
POST /ingest/lastfm	User token → fetch & schedule scrobble import job
GET /recs/home	Returns 5 blended carousels (For You, New & Hype, Genre Mixes…)
GET /recs/track/{track_id}	Tracks similar to X (KNN on track_vector)
POST /search	Free-text query → semantic search results
GET /taste/summary	User taste profile JSON (top genres, mood distribution)

7. Playlist Autofill (Post-MVP ready)
User posts seed playlist ID.

Aggregate centroid of seed tracks’ track_vector.

KNN search (exclude seeds + user library).

Diversify via Maximal Marginal Relevance (λ = 0.3).

Return top k; optional “Add to Spotify” with single /batch addTracks call.

8. Evaluation & Monitoring
Offline metrics: Precision@10, Recall@50, NDCG using held-out 3-month slice.

Online (after launch): implicit click-through rate, long-play rate (>60 s), playlist-save rate.

Dashboards: Grafana panel reading Postgres job logs + Redis hit/miss + model inference latency.

ML retraining cadence:

CF matrix - nightly

Mood model - quarterly (or when tag drift > 5 %)

9. Cold-Start & Fallback Logic
Scenario	Action
No Last.fm, no Spotify	Show onboarding picker (genres + sample tracks). Build user vector from selections.
Track has only genre	Use genre→energy prior table; mark mood_confidence = low.
New release with zero tags	Run web-scrape enrichment micro-service only if album’s weekly Scrobbles > 1000 or explicit user search triggers.

10. DevOps / Repo Notes
Keep all ML code inside Timbral; expose as Docker image timbre-recs:v1.

Timbre (monorepo) imports recs via internal gRPC or REST.

CI: model unit tests (pytest) + smoke inference test on sample track.

CD: GitHub Actions pushes new model image ➜ Fly.io deploy hook, zero-downtime rollout.