# Collaborative Filtering Setup Guide

This guide explains how to set up collaborative filtering for Timbre using multiple Last.fm users' listening data.

## 🎯 What This Enables

- **User-Based Collaborative Filtering**: Find users with similar music tastes
- **Item-Based Recommendations**: Suggest tracks based on what similar users listen to
- **Hybrid Recommendations**: Combine collaborative filtering with your existing content-based approach
- **Scalable Data Collection**: Fetch data from multiple Last.fm users concurrently

## 🗄️ Database Schema

The system creates these tables automatically:

- `lastfm_users` - User profiles and metadata
- `user_track_interactions` - Individual track plays, loves, skips
- `user_album_interactions` - Album-level listening data
- `user_artist_interactions` - Artist-level listening data
- `user_similarities` - Pre-calculated user similarity scores
- `collaborative_recommendations` - Generated recommendations
- `data_fetch_logs` - Monitor data collection progress

## 🚀 Quick Start

### 1. Run the Database Migration

```bash
cd backend
# Connect to your Supabase database and run:
psql $SUPABASE_DB_URL -f migrations/collaborative_filtering_setup.sql
```

### 2. Add Last.fm Users

Edit the script to add your target users:

```bash
# Edit the usernames in the script
nano scripts/setup_collaborative_filtering.py

# Run the setup script
python scripts/setup_collaborative_filtering.py
```

### 3. Use the API

The system provides these endpoints:

```bash
# Add a new user
POST /collaborative-filtering/users?username=some_user&display_name=Some User

# Get all active users
GET /collaborative-filtering/users

# Fetch data for a specific user
POST /collaborative-filtering/users/some_user/fetch

# Fetch data for multiple users
POST /collaborative-filtering/users/bulk-fetch
Body: ["user1", "user2", "user3"]

# Get system statistics
GET /collaborative-filtering/stats

# Remove a user
DELETE /collaborative-filtering/users/some_user
```

## 📊 Data Collection Strategy

### Recommended User Selection

Choose users who:

- Have diverse music tastes (not just one genre)
- Are active on Last.fm (regular listening)
- Have substantial play counts (1000+ tracks)
- Represent different demographics/age groups

### Data Fetching Limits

- **Tracks**: Up to 1000 per user
- **Albums**: Up to 500 per user
- **Artists**: Up to 500 per user
- **Rate Limiting**: Built-in to respect Last.fm API limits

## 🔧 Configuration

### Environment Variables

Ensure these are set in your `.env`:

```bash
LASTFM_API_KEY=your_api_key
LASTFM_API_SECRET=your_api_secret
SUPABASE_DB_URL=your_database_url
```

### Last.fm API Limits

- **Free tier**: 3000 requests/hour
- **Paid tier**: 5000 requests/hour
- The system automatically handles rate limiting

## 📈 Monitoring & Analytics

### Data Fetch Logs

Check `data_fetch_logs` table for:

- Fetch success/failure status
- Number of items fetched
- Processing time
- Error messages

### User Statistics

Use the `/collaborative-filtering/stats` endpoint to monitor:

- Total users tracked
- Interaction counts
- ML data generation progress

## 🧠 Next Steps

### 1. Similarity Calculations

After collecting data, calculate user similarities:

```python
# This will be implemented next
from services.similarity_calculator import SimilarityCalculator

calculator = SimilarityCalculator()
await calculator.calculate_all_user_similarities()
```

### 2. Recommendation Generation

Generate collaborative filtering recommendations:

```python
# This will be implemented next
from services.recommendation_engine import CollaborativeRecommendationEngine

engine = CollaborativeRecommendationEngine()
recommendations = await engine.generate_recommendations(user_id)
```

### 3. Integration

Integrate with your existing recommendation system:

```python
# Combine collaborative and content-based approaches
hybrid_recommendations = await engine.get_hybrid_recommendations(
    user_id,
    content_weight=0.6,
    collaborative_weight=0.4
)
```

## 🚨 Troubleshooting

### Common Issues

1. **User not found**: Check Last.fm username spelling
2. **API rate limits**: Wait for rate limit reset
3. **Database errors**: Verify migration ran successfully
4. **Missing tracks**: Some tracks may not exist in your database

### Debug Commands

```bash
# Check database tables
psql $SUPABASE_DB_URL -c "\dt"

# View user data
psql $SUPABASE_DB_URL -c "SELECT * FROM lastfm_users LIMIT 5;"

# Check interactions
psql $SUPABASE_DB_URL -c "SELECT COUNT(*) FROM user_track_interactions;"
```

## 📚 API Documentation

Once running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🤝 Contributing

To add new features:

1. Extend the models in `models/collaborative_filtering.py`
2. Add methods to `services/collaborative_filtering_service.py`
3. Create new routes in `routes/collaborative_filtering_routes.py`
4. Update this README

## 📞 Support

For issues or questions:

1. Check the logs in `data_fetch_logs` table
2. Review Last.fm API documentation
3. Check database connectivity
4. Verify environment variables are set correctly




