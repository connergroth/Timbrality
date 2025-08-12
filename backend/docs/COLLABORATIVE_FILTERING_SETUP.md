# Collaborative Filtering Setup Guide

This guide explains how to set up collaborative filtering for your music recommendation system using multiple users' Last.fm data.

## Overview

Collaborative filtering works by finding users with similar music taste and recommending tracks that similar users have enjoyed. This system will:

1. Fetch data from multiple Last.fm users
2. Store user-track interactions in the database
3. Calculate user similarity scores
4. Generate collaborative recommendations

## Database Schema

The migration file `collaborative_filtering_setup.sql` creates several new tables:

### Core Tables

- **`lastfm_users`** - Stores Last.fm user profiles
- **`user_track_interactions`** - User-track plays, loves, and tags
- **`user_album_interactions`** - User-album plays and loves
- **`user_artist_interactions`** - User-artist plays and loves

### Analysis Tables

- **`user_similarities`** - Calculated similarity scores between users
- **`collaborative_recommendations`** - Generated recommendations
- **`data_fetch_logs`** - Monitoring and debugging logs

### Views

- **`user_listening_stats`** - Summary statistics per user
- **`track_popularity_across_users`** - Track popularity across all users

## Setup Steps

### 1. Run the Database Migration

```bash
cd backend
psql -h your_host -U your_user -d your_database -f migrations/collaborative_filtering_setup.sql
```

### 2. Configure Last.fm Users

Edit `config/collaborative_users.py` and add real Last.fm usernames:

```python
COLLABORATIVE_USERS = [
    'your_actual_username',  # Replace with your real Last.fm username
    'musiccritic123',        # Add real usernames here
    'indie_lover',
    'jazz_enthusiast',
    # Add more users...
]
```

**Important**: Only add users with public Last.fm profiles. Private profiles cannot be accessed via the API.

### 3. Install Dependencies

Make sure you have the required packages:

```bash
pip install psycopg2-binary python-dotenv
```

### 4. Run the Data Population Script

```bash
cd backend
python scripts/populate_collaborative_data.py
```

This script will:

- Fetch top tracks, albums, and artists for each user
- Store the data in the new tables
- Handle rate limiting and errors gracefully

## How It Works

### Data Collection

1. **User Discovery**: The system fetches data from the configured Last.fm users
2. **Interaction Storage**: Stores plays, loves, and tags for tracks/albums/artists
3. **Data Quality**: Filters out low-quality data (e.g., tracks with 0 plays)

### Similarity Calculation

The system calculates user similarity using:

- **Cosine Similarity**: Based on shared track/album/artist preferences
- **Jaccard Similarity**: Based on overlap of favorite items
- **Pearson Correlation**: Based on rating patterns

### Recommendation Generation

1. Find users similar to the target user
2. Identify tracks that similar users love but the target user hasn't heard
3. Score recommendations based on similarity and popularity
4. Store results in `collaborative_recommendations` table

## Configuration Options

### Fetch Limits

```python
FETCH_LIMITS = {
    'tracks_per_user': 100,    # Top 100 tracks per user
    'albums_per_user': 50,     # Top 50 albums per user
    'artists_per_user': 30,    # Top 30 artists per user
}
```

### Rate Limiting

```python
RATE_LIMITING = {
    'delay_between_users': 1.0,  # 1 second between users
    'max_requests_per_minute': 30,  # Last.fm API limit
}
```

### Algorithm Settings

```python
ALGORITHM_SETTINGS = {
    'similarity_threshold': 0.1,  # Minimum similarity to consider
    'max_similar_users': 50,      # Max similar users to analyze
    'min_shared_items': 5,        # Min shared items for similarity
}
```

## Monitoring and Debugging

### Data Fetch Logs

Check the `data_fetch_logs` table to monitor:

- Success/failure of data fetches
- Number of items fetched per user
- Error messages and timing information

### User Statistics

Use the `user_listening_stats` view to see:

- How many tracks/albums/artists each user has
- Total play counts and loved items
- Last update timestamps

### Track Popularity

Use the `track_popularity_across_users` view to see:

- Which tracks are popular across multiple users
- Average plays per user
- Total love counts

## Best Practices

### User Selection

1. **Diverse Taste**: Include users with different musical preferences
2. **Active Users**: Choose users with substantial listening history
3. **Public Profiles**: Only include users with public Last.fm profiles
4. **Quality Over Quantity**: 10-20 high-quality users is better than 100 inactive ones

### Data Management

1. **Regular Updates**: Run the script weekly to get fresh data
2. **Monitor Quality**: Check for users with very few tracks
3. **Error Handling**: The script logs errors for debugging
4. **Rate Limiting**: Respect Last.fm's API limits

### Performance

1. **Indexes**: The migration creates necessary database indexes
2. **Batch Processing**: Process users sequentially to avoid overwhelming the API
3. **Connection Pooling**: The script manages database connections efficiently

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Increase delays between users
2. **Database Connection**: Check your database credentials
3. **User Not Found**: Verify Last.fm usernames exist and are public
4. **Empty Results**: Some users may have very limited listening history

### Debug Commands

```sql
-- Check user data
SELECT * FROM user_listening_stats;

-- Check recent fetch logs
SELECT * FROM data_fetch_logs ORDER BY started_at DESC LIMIT 10;

-- Check user similarities
SELECT * FROM user_similarities ORDER BY similarity_score DESC LIMIT 10;

-- Check recommendations
SELECT * FROM collaborative_recommendations ORDER BY recommendation_score DESC LIMIT 10;
```

## Next Steps

After setting up the collaborative filtering system:

1. **Train Models**: Use the collected data to train recommendation models
2. **A/B Testing**: Compare collaborative vs. content-based recommendations
3. **Hybrid Approaches**: Combine collaborative and content-based methods
4. **Real-time Updates**: Implement incremental updates for new user data

## API Integration

The system is designed to integrate with your existing recommendation API:

```python
# Example: Get collaborative recommendations for a user
def get_collaborative_recommendations(user_id: str, limit: int = 10):
    # Query the collaborative_recommendations table
    # Return tracks with highest recommendation scores
    pass
```

## Support

If you encounter issues:

1. Check the logs in `data_fetch_logs`
2. Verify Last.fm API credentials
3. Check database connectivity
4. Review user profile privacy settings

The system is designed to be robust and handle errors gracefully, so most issues will be logged for easy debugging.




