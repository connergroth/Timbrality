import type { UnifiedTrack, RecommendationSet, RecommendationSource, TrackFeedback } from '@/lib/types/track';
import type { Track as AgentTrack } from '@/lib/agent';

// ML API Integration Types
export interface MLRecommendationRequest {
  user_id: string
  algorithm: 'collaborative' | 'content' | 'hybrid'
  top_k: number
  filters?: {
    genres?: string[]
    min_year?: number
    max_year?: number
    min_rating?: number
    energy_range?: [number, number]
    danceability_range?: [number, number]
    include_mood_analysis?: boolean
    diversity_weight?: number
    popularity_bias?: number
  }
  include_metadata?: boolean
  include_explanations?: boolean
}

export interface MLRecommendationExplanation {
  collaborative_score: number
  content_score: number
  hybrid_score: number
  reasons: string[]
  confidence: number
  similar_users?: number[]
  content_factors?: {
    genre_similarity: number
    audio_feature_similarity: number
    lyrical_similarity?: number
  }
}

export interface MLAudioFeatures {
  danceability: number
  energy: number
  valence: number
  acousticness: number
  instrumentalness: number
  liveness: number
  speechiness: number
  tempo: number
  loudness: number
  key: number
  mode: number
  time_signature: number
}

export interface MLRecommendationItem {
  item_id: string
  title: string
  artist: string
  album?: string
  year: number
  genres: string[]
  rating?: number
  popularity?: number
  cover_url?: string
  spotify_url?: string
  preview_url?: string
  score: number
  explanation?: MLRecommendationExplanation
  audio_features?: MLAudioFeatures
  metadata?: {
    duration_ms: number
    explicit: boolean
    is_local: boolean
    track_number?: number
    disc_number?: number
  }
}

export interface MLRecommendationResponse {
  user_id: string
  algorithm: string
  recommendations: MLRecommendationItem[]
  total_count: number
  generated_at: string
  cache_hit: boolean
  generation_time_ms: number
  model_version?: string
}

/**
 * Service for normalizing track data from different sources into unified format
 * and handling recommendation display logic
 */
export class RecommendationService {
  
  /**
   * Convert agent track to unified format
   */
  static fromAgentTrack(track: AgentTrack, reason?: string): UnifiedTrack {
    return {
      id: track.id,
      spotify_id: track.id,
      name: track.name,
      artist: track.artist,
      album: track.album,
      artwork_url: track.artwork_url,
      spotify_url: track.spotify_url,
      preview_url: track.preview_url,
      source: 'agent',
      similarity_score: track.similarity,
      genres: track.genres,
      audio_features: track.audio_features,
      recommendation_reason: reason,
      created_at: new Date().toISOString()
    };
  }

  /**
   * Convert Spotify search result to unified format
   */
  static fromSpotifySearch(spotifyTrack: any, source: RecommendationSource = 'spotify_search'): UnifiedTrack {
    return {
      id: spotifyTrack.id,
      spotify_id: spotifyTrack.id,
      name: spotifyTrack.name,
      artist: spotifyTrack.artists?.[0]?.name || spotifyTrack.artist,
      artists: spotifyTrack.artists?.map((a: any) => a.name),
      album: spotifyTrack.album?.name || spotifyTrack.album,
      album_id: spotifyTrack.album?.id || spotifyTrack.album_id,
      artwork_url: spotifyTrack.album?.images?.[0]?.url || spotifyTrack.cover_art,
      spotify_url: spotifyTrack.external_urls?.spotify || spotifyTrack.spotify_url,
      preview_url: spotifyTrack.preview_url,
      duration_ms: spotifyTrack.duration_ms,
      popularity: spotifyTrack.popularity,
      explicit: spotifyTrack.explicit,
      release_date: spotifyTrack.album?.release_date || spotifyTrack.release_date,
      source,
      created_at: new Date().toISOString()
    };
  }

  /**
   * Convert ML recommendation to unified format
   */
  static fromMLRecommendation(
    mlTrack: any, 
    source: 'ml_hybrid' | 'ml_collaborative' | 'ml_content' = 'ml_hybrid'
  ): UnifiedTrack {
    return {
      id: mlTrack.track_id || mlTrack.id,
      spotify_id: mlTrack.spotify_id,
      name: mlTrack.name || mlTrack.track_name,
      artist: mlTrack.artist || mlTrack.artist_name,
      album: mlTrack.album || mlTrack.album_name,
      artwork_url: mlTrack.artwork_url || mlTrack.cover_art,
      spotify_url: mlTrack.spotify_url,
      preview_url: mlTrack.preview_url,
      source,
      confidence_score: mlTrack.confidence || mlTrack.score,
      similarity_score: mlTrack.similarity,
      genres: mlTrack.genres,
      audio_features: mlTrack.audio_features,
      recommendation_reason: mlTrack.reason || `Recommended by ${source.replace('ml_', '').replace('_', ' ')} model`,
      created_at: new Date().toISOString()
    };
  }

  /**
   * Convert Last.fm track to unified format
   */
  static fromLastfmTrack(lastfmTrack: any): UnifiedTrack {
    return {
      id: `lastfm_${lastfmTrack.mbid || `${lastfmTrack.artist?.name}_${lastfmTrack.name}`.replace(/\s+/g, '_')}`,
      lastfm_id: lastfmTrack.mbid,
      name: lastfmTrack.name,
      artist: lastfmTrack.artist?.name || lastfmTrack.artist,
      album: lastfmTrack.album?.title || lastfmTrack.album,
      artwork_url: lastfmTrack.image?.find((img: any) => img.size === 'large')?.['#text'],
      lastfm_url: lastfmTrack.url,
      source: 'lastfm_similar',
      tags: lastfmTrack.toptags?.tag?.map((t: any) => t.name) || [],
      recommendation_reason: 'Similar artists on Last.fm',
      created_at: new Date().toISOString()
    };
  }

  /**
   * Create a recommendation set from multiple tracks
   */
  static createRecommendationSet(
    tracks: UnifiedTrack[],
    title: string,
    description?: string,
    metadata?: any
  ): RecommendationSet {
    const sources = [...new Set(tracks.map(t => t.source))];
    const avgConfidence = tracks.reduce((sum, t) => sum + (t.confidence_score || t.similarity_score || 0.5), 0) / tracks.length;

    return {
      id: `rec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title,
      description,
      tracks,
      source: sources.length === 1 ? sources[0] : 'ml_hybrid',
      confidence: avgConfidence,
      created_at: new Date().toISOString(),
      metadata
    };
  }

  /**
   * Format track for display with enhanced recommendation context
   */
  static formatTrackForDisplay(track: UnifiedTrack, context?: {
    query?: string;
    mood?: string;
    previousTracks?: string[];
  }): UnifiedTrack {
    let enhancedReason = track.recommendation_reason;

    // Enhance recommendation reason based on context
    if (context?.query && !enhancedReason) {
      enhancedReason = `Found matching "${context.query}"`;
    }

    if (context?.mood && track.source === 'agent') {
      enhancedReason = `Matches your ${context.mood} mood`;
    }

    if (track.similarity_score && track.similarity_score > 0.8) {
      enhancedReason = `${enhancedReason} • High similarity match`;
    }

    return {
      ...track,
      recommendation_reason: enhancedReason
    };
  }

  /**
   * Submit user feedback on track recommendation
   */
  static async submitFeedback(
    feedback: TrackFeedback,
    userId: string,
    apiUrl: string = process.env.NEXT_PUBLIC_AGENT_API_URL || 'http://localhost:8000'
  ): Promise<{ success: boolean; message: string }> {
    try {
      const response = await fetch(`${apiUrl}/agent/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          track_id: feedback.track_id,
          feedback_type: feedback.feedback_type,
          feedback_data: {
            ...feedback.context,
            timestamp: new Date().toISOString()
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`Feedback submission failed: ${response.statusText}`);
      }

      const result = await response.json();
      return { success: true, message: result.message || 'Feedback submitted successfully' };
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      return { success: false, message: 'Failed to submit feedback' };
    }
  }

  /**
   * Get recommendation source display information
   */
  static getSourceInfo(source: RecommendationSource): { label: string; color: string; description: string } {
    const sourceMap: Record<RecommendationSource, { label: string; color: string; description: string }> = {
      'agent': { label: 'AI Agent', color: 'blue', description: 'Recommended by AI assistant' },
      'ml_hybrid': { label: 'Smart Rec', color: 'purple', description: 'Hybrid ML recommendation' },
      'ml_collaborative': { label: 'Users Like You', color: 'green', description: 'Based on similar users' },
      'ml_content': { label: 'Similar Sound', color: 'orange', description: 'Based on audio features' },
      'spotify_search': { label: 'Spotify', color: 'green', description: 'Direct Spotify search' },
      'spotify_playlist': { label: 'Playlist', color: 'green', description: 'From Spotify playlist' },
      'spotify_recommendations': { label: 'Spotify Recs', color: 'green', description: 'Spotify recommendations' },
      'lastfm_similar': { label: 'Last.fm', color: 'red', description: 'Similar on Last.fm' },
      'lastfm_user': { label: 'Last.fm Profile', color: 'red', description: 'From your Last.fm' },
      'aoty_similar': { label: 'AOTY', color: 'yellow', description: 'Album of the Year' },
      'aoty_charts': { label: 'AOTY Charts', color: 'yellow', description: 'AOTY trending' },
      'database': { label: 'Database', color: 'gray', description: 'From local database' },
      'fallback': { label: 'Search', color: 'gray', description: 'Fallback search result' }
    };

    return sourceMap[source] || { label: source, color: 'gray', description: 'Unknown source' };
  }

  /**
   * ML API Integration Methods
   */
  static async getMLRecommendations(
    request: MLRecommendationRequest,
    apiUrl: string = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8001'
  ): Promise<MLRecommendationResponse> {
    try {
      const response = await fetch(`${apiUrl}/api/v1/recommendations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`ML API Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to get ML recommendations:', error);
      throw error;
    }
  }

  /**
   * Convert ML recommendation to unified track format
   */
  static fromMLRecommendationItem(
    mlItem: MLRecommendationItem, 
    source: 'ml_hybrid' | 'ml_collaborative' | 'ml_content' = 'ml_hybrid'
  ): UnifiedTrack {
    return {
      id: mlItem.item_id,
      spotify_id: mlItem.item_id.startsWith('spotify:') ? mlItem.item_id.split(':')[2] : mlItem.item_id,
      name: mlItem.title,
      artist: mlItem.artist,
      album: mlItem.album,
      artwork_url: mlItem.cover_url,
      spotify_url: mlItem.spotify_url,
      preview_url: mlItem.preview_url,
      source,
      confidence_score: mlItem.score,
      similarity_score: mlItem.explanation?.hybrid_score || mlItem.score,
      genres: mlItem.genres,
      audio_features: mlItem.audio_features ? {
        danceability: mlItem.audio_features.danceability,
        energy: mlItem.audio_features.energy,
        valence: mlItem.audio_features.valence,
        acousticness: mlItem.audio_features.acousticness,
        instrumentalness: mlItem.audio_features.instrumentalness,
        liveness: mlItem.audio_features.liveness,
        speechiness: mlItem.audio_features.speechiness,
        tempo: mlItem.audio_features.tempo,
        key: mlItem.audio_features.key,
        mode: mlItem.audio_features.mode,
        time_signature: mlItem.audio_features.time_signature
      } : undefined,
      recommendation_reason: mlItem.explanation?.reasons?.[0] || `Recommended by ${source.replace('ml_', '').replace('_', ' ')} model`,
      popularity: mlItem.popularity || mlItem.rating,
      release_date: mlItem.year?.toString(),
      duration_ms: mlItem.metadata?.duration_ms,
      explicit: mlItem.metadata?.explicit,
      created_at: new Date().toISOString()
    };
  }

  /**
   * Get ML-based recommendations and convert to unified format
   */
  static async getMLRecommendationsAsUnifiedTracks(
    userId: string,
    algorithm: 'collaborative' | 'content' | 'hybrid' = 'hybrid',
    topK: number = 10,
    filters?: MLRecommendationRequest['filters']
  ): Promise<UnifiedTrack[]> {
    try {
      const request: MLRecommendationRequest = {
        user_id: userId,
        algorithm,
        top_k: topK,
        filters,
        include_metadata: true,
        include_explanations: true
      };

      const response = await this.getMLRecommendations(request);
      const source = `ml_${algorithm}` as 'ml_hybrid' | 'ml_collaborative' | 'ml_content';

      return response.recommendations.map(item => 
        this.fromMLRecommendationItem(item, source)
      );
    } catch (error) {
      console.error('Failed to get ML recommendations as unified tracks:', error);
      // Return empty array instead of throwing to allow graceful fallback
      return [];
    }
  }

  /**
   * Explain a specific recommendation
   */
  static async explainMLRecommendation(
    userId: string,
    itemId: string,
    algorithm: 'collaborative' | 'content' | 'hybrid' = 'hybrid',
    apiUrl: string = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8001'
  ): Promise<MLRecommendationExplanation | null> {
    try {
      const response = await fetch(
        `${apiUrl}/api/v1/explain/${userId}/${itemId}?algorithm=${algorithm}`
      );

      if (!response.ok) {
        throw new Error(`Explanation API Error: ${response.statusText}`);
      }

      const data = await response.json();
      return data.explanation;
    } catch (error) {
      console.error('Failed to get ML explanation:', error);
      return null;
    }
  }

  /**
   * Submit feedback to ML system
   */
  static async submitMLFeedback(
    userId: string,
    itemId: string,
    rating: number,
    feedbackType: string = 'rating',
    apiUrl: string = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8001'
  ): Promise<{ success: boolean; message: string }> {
    try {
      const response = await fetch(`${apiUrl}/api/v1/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          item_id: itemId,
          rating,
          feedback_type: feedbackType
        }),
      });

      if (!response.ok) {
        throw new Error(`Feedback API Error: ${response.statusText}`);
      }

      const result = await response.json();
      return { success: true, message: result.message || 'Feedback submitted successfully' };
    } catch (error) {
      console.error('Failed to submit ML feedback:', error);
      return { success: false, message: 'Failed to submit feedback' };
    }
  }

  /**
   * Get ML service health status
   */
  static async getMLHealthStatus(
    apiUrl: string = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8001'
  ): Promise<{ status: string; models: any; cache: any } | null> {
    try {
      const response = await fetch(`${apiUrl}/api/v1/health`);
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to get ML health status:', error);
      return null;
    }
  }

  /**
   * Generate mock ML recommendations for development
   */
  static generateMockMLRecommendations(count: number = 10): MLRecommendationItem[] {
    const mockTracks = [
      { title: 'Blue in Green', artist: 'Miles Davis', album: 'Kind of Blue', genres: ['Jazz', 'Cool Jazz'], year: 1959 },
      { title: 'Paranoid Android', artist: 'Radiohead', album: 'OK Computer', genres: ['Alternative Rock', 'Art Rock'], year: 1997 },
      { title: 'So What', artist: 'Miles Davis', album: 'Kind of Blue', genres: ['Jazz', 'Modal Jazz'], year: 1959 },
      { title: 'Karma Police', artist: 'Radiohead', album: 'OK Computer', genres: ['Alternative Rock'], year: 1997 },
      { title: 'Take Five', artist: 'Dave Brubeck', album: 'Time Out', genres: ['Jazz', 'Cool Jazz'], year: 1959 },
      { title: 'Let Down', artist: 'Radiohead', album: 'OK Computer', genres: ['Alternative Rock'], year: 1997 },
      { title: 'All Blues', artist: 'Miles Davis', album: 'Kind of Blue', genres: ['Jazz', 'Blues'], year: 1959 },
      { title: 'Exit Music (For a Film)', artist: 'Radiohead', album: 'OK Computer', genres: ['Alternative Rock', 'Art Rock'], year: 1997 }
    ];

    return Array.from({ length: count }, (_, i) => {
      const base = mockTracks[i % mockTracks.length];
      return {
        item_id: `mock_ml_${i}`,
        title: base.title,
        artist: base.artist,
        album: base.album,
        year: base.year,
        genres: base.genres,
        rating: Math.floor(Math.random() * 20) + 80,
        score: Math.random() * 0.3 + 0.7,
        cover_url: `https://via.placeholder.com/300x300/1a1a1a/ffffff?text=${encodeURIComponent(base.title.slice(0, 20))}`,
        explanation: {
          collaborative_score: Math.random() * 0.3 + 0.6,
          content_score: Math.random() * 0.3 + 0.6,
          hybrid_score: Math.random() * 0.3 + 0.7,
          confidence: Math.random() * 0.3 + 0.7,
          reasons: [
            `Similar to your ${base.genres[0].toLowerCase()} preferences`,
            `High ratings from users with similar taste`,
            `Matches your preference for ${base.year < 1980 ? 'classic' : 'modern'} music`
          ]
        },
        audio_features: {
          danceability: Math.random(),
          energy: Math.random(),
          valence: Math.random(),
          acousticness: Math.random(),
          instrumentalness: Math.random(),
          liveness: Math.random(),
          speechiness: Math.random(),
          tempo: Math.random() * 100 + 80,
          loudness: Math.random() * 30 - 20,
          key: Math.floor(Math.random() * 12),
          mode: Math.round(Math.random()),
          time_signature: 4
        }
      };
    });
  }

  /**
   * Format audio feature for display
   */
  static formatAudioFeature(feature: string, value: number): string {
    const formatters: Record<string, (v: number) => string> = {
      danceability: (v) => `${Math.round(v * 100)}% danceable`,
      energy: (v) => `${Math.round(v * 100)}% energy`,
      valence: (v) => v > 0.5 ? 'Positive mood' : 'Melancholic mood',
      acousticness: (v) => v > 0.5 ? 'Acoustic' : 'Electronic',
      tempo: (v) => `${Math.round(v)} BPM`,
      loudness: (v) => `${v.toFixed(1)} dB`,
      key: (v) => {
        const keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        return keys[v] || 'Unknown';
      },
      mode: (v) => v === 1 ? 'Major' : 'Minor'
    };

    return formatters[feature]?.(value) || `${value}`;
  }

  /**
   * Get confidence score color for UI display
   */
  static getConfidenceColor(score: number): string {
    if (score >= 0.8) return 'text-green-600 dark:text-green-400';
    if (score >= 0.6) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  }
}