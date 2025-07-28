import type { UnifiedTrack, RecommendationSet, RecommendationSource, TrackFeedback } from '@/lib/types/track';
import type { Track as AgentTrack } from '@/lib/agent';

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
}