/**
 * Unified track interface for all recommendation sources
 * (Agent, ML, Spotify, Last.fm, AOTY, etc.)
 */
export interface UnifiedTrack {
  // Core identifiers
  id: string;
  spotify_id?: string;
  lastfm_id?: string;
  
  // Basic metadata
  name: string;
  artist: string;
  artists?: string[]; // Additional artists for collaborations
  album?: string;
  album_id?: string;
  
  // Media
  artwork_url?: string;
  preview_url?: string;
  
  // Links
  spotify_url?: string;
  lastfm_url?: string;
  youtube_url?: string;
  
  // Audio characteristics
  duration_ms?: number;
  explicit?: boolean;
  popularity?: number;
  release_date?: string;
  
  // Recommendation context
  source: RecommendationSource;
  similarity_score?: number;
  confidence_score?: number;
  recommendation_reason?: string;
  
  // Classification
  genres?: string[];
  tags?: string[];
  mood?: string[];
  
  // Audio features (for ML recommendations)
  audio_features?: {
    danceability?: number;
    energy?: number;
    valence?: number;
    acousticness?: number;
    instrumentalness?: number;
    liveness?: number;
    speechiness?: number;
    tempo?: number;
    key?: number;
    mode?: number;
    time_signature?: number;
  };
  
  // Metadata
  created_at?: string;
  updated_at?: string;
  metadata?: Record<string, any>;
}

export type RecommendationSource = 
  | 'agent'
  | 'ml_hybrid'
  | 'ml_collaborative'
  | 'ml_content'
  | 'spotify_search'
  | 'spotify_playlist'
  | 'spotify_recommendations'
  | 'lastfm_similar'
  | 'lastfm_user'
  | 'aoty_similar'
  | 'aoty_charts'
  | 'database'
  | 'fallback';

/**
 * Recommendation context for displaying groups of tracks
 */
export interface RecommendationSet {
  id: string;
  title: string;
  description?: string;
  tracks: UnifiedTrack[];
  source: RecommendationSource;
  confidence: number;
  created_at: string;
  metadata?: {
    query?: string;
    user_input?: string;
    tools_used?: string[];
    explanation?: string;
    [key: string]: any;
  };
}

/**
 * User feedback on recommendations
 */
export interface TrackFeedback {
  track_id: string;
  user_id: string;
  feedback_type: 'like' | 'dislike' | 'save' | 'skip' | 'play_full' | 'share';
  timestamp: string;
  context?: {
    source: RecommendationSource;
    recommendation_set_id?: string;
    listening_session_id?: string;
    [key: string]: any;
  };
}