-- Collaborative Filtering Tables for Supabase
-- Run these SQL statements in Supabase SQL Editor

-- LastFM Users table
CREATE TABLE IF NOT EXISTS lastfm_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lastfm_username TEXT UNIQUE NOT NULL,
    display_name TEXT,
    real_name TEXT,
    country TEXT,
    age INTEGER,
    gender TEXT,
    subscriber BOOLEAN DEFAULT FALSE,
    playcount_total INTEGER DEFAULT 0,
    playlists_count INTEGER DEFAULT 0,
    registered_date TIMESTAMP,
    last_updated TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    data_fetch_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User Track Interactions table
CREATE TABLE IF NOT EXISTS user_track_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lastfm_user_id UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
    track_id TEXT REFERENCES tracks(id) ON DELETE CASCADE,
    interaction_type TEXT DEFAULT 'play',
    play_count INTEGER DEFAULT 0,
    user_loved BOOLEAN DEFAULT FALSE,
    last_played TIMESTAMP,
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(lastfm_user_id, track_id)
);

-- User Album Interactions table  
CREATE TABLE IF NOT EXISTS user_album_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lastfm_user_id UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
    album_title TEXT NOT NULL,
    album_artist TEXT NOT NULL,
    play_count INTEGER DEFAULT 0,
    user_loved BOOLEAN DEFAULT FALSE,
    last_played TIMESTAMP,
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(lastfm_user_id, album_title, album_artist)
);

-- User Artist Interactions table
CREATE TABLE IF NOT EXISTS user_artist_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lastfm_user_id UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
    artist_name TEXT NOT NULL,
    play_count INTEGER DEFAULT 0,
    user_loved BOOLEAN DEFAULT FALSE,
    last_played TIMESTAMP,
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(lastfm_user_id, artist_name)
);

-- User Similarities table
CREATE TABLE IF NOT EXISTS user_similarities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id_1 UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
    user_id_2 UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
    similarity_score FLOAT NOT NULL CHECK (similarity_score >= -1 AND similarity_score <= 1),
    similarity_type TEXT DEFAULT 'cosine',
    shared_tracks_count INTEGER DEFAULT 0,
    shared_albums_count INTEGER DEFAULT 0,
    shared_artists_count INTEGER DEFAULT 0,
    calculated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id_1, user_id_2)
);

-- Collaborative Recommendations table
CREATE TABLE IF NOT EXISTS collaborative_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_user_id UUID REFERENCES lastfm_users(id) ON DELETE CASCADE,
    track_id TEXT REFERENCES tracks(id) ON DELETE CASCADE,
    recommendation_score FLOAT NOT NULL CHECK (recommendation_score >= 0 AND recommendation_score <= 1),
    algorithm_type TEXT DEFAULT 'user_based',
    confidence_score FLOAT DEFAULT 0.0 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    reason TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(target_user_id, track_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_track_interactions_user ON user_track_interactions(lastfm_user_id);
CREATE INDEX IF NOT EXISTS idx_user_track_interactions_track ON user_track_interactions(track_id);
CREATE INDEX IF NOT EXISTS idx_user_track_interactions_play_count ON user_track_interactions(play_count);

CREATE INDEX IF NOT EXISTS idx_user_album_interactions_user ON user_album_interactions(lastfm_user_id);
CREATE INDEX IF NOT EXISTS idx_user_album_interactions_artist ON user_album_interactions(album_artist);

CREATE INDEX IF NOT EXISTS idx_user_artist_interactions_user ON user_artist_interactions(lastfm_user_id);
CREATE INDEX IF NOT EXISTS idx_user_artist_interactions_artist ON user_artist_interactions(artist_name);

CREATE INDEX IF NOT EXISTS idx_user_similarities_user1 ON user_similarities(user_id_1);
CREATE INDEX IF NOT EXISTS idx_user_similarities_user2 ON user_similarities(user_id_2);
CREATE INDEX IF NOT EXISTS idx_user_similarities_score ON user_similarities(similarity_score);

CREATE INDEX IF NOT EXISTS idx_collaborative_recommendations_user ON collaborative_recommendations(target_user_id);
CREATE INDEX IF NOT EXISTS idx_collaborative_recommendations_track ON collaborative_recommendations(track_id);
CREATE INDEX IF NOT EXISTS idx_collaborative_recommendations_score ON collaborative_recommendations(recommendation_score);

-- Enable Row Level Security (RLS) if needed
ALTER TABLE lastfm_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_track_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_album_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_artist_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_similarities ENABLE ROW LEVEL SECURITY;
ALTER TABLE collaborative_recommendations ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (adjust based on your authentication needs)
-- For now, allow all operations (you can restrict this later)
CREATE POLICY "Allow all operations on lastfm_users" ON lastfm_users FOR ALL USING (true);
CREATE POLICY "Allow all operations on user_track_interactions" ON user_track_interactions FOR ALL USING (true);
CREATE POLICY "Allow all operations on user_album_interactions" ON user_album_interactions FOR ALL USING (true);
CREATE POLICY "Allow all operations on user_artist_interactions" ON user_artist_interactions FOR ALL USING (true);
CREATE POLICY "Allow all operations on user_similarities" ON user_similarities FOR ALL USING (true);
CREATE POLICY "Allow all operations on collaborative_recommendations" ON collaborative_recommendations FOR ALL USING (true);