import { useState } from 'react';
import { Heart, Play, ExternalLink, ThumbsUp, ThumbsDown, SkipForward, Share2, Plus, ListPlus } from 'lucide-react';
import type { UnifiedTrack, TrackFeedback } from '@/lib/types/track';

interface TrackRecommendationCardProps {
  track: UnifiedTrack;
  onFeedback?: (feedback: Omit<TrackFeedback, 'user_id' | 'timestamp'>) => void;
  onAddToPlaylist?: (trackId: string) => void;
  showFeedback?: boolean;
  showSource?: boolean;
  size?: 'compact' | 'standard' | 'detailed';
  className?: string;
}

export function TrackRecommendationCard({ 
  track, 
  onFeedback, 
  onAddToPlaylist,
  showFeedback = true,
  showSource = true,
  size = 'standard',
  className = '' 
}: TrackRecommendationCardProps) {
  const [feedback, setFeedback] = useState<'like' | 'dislike' | null>(null);
  const [isSaved, setIsSaved] = useState(false);

  const handleFeedback = (type: 'like' | 'dislike' | 'skip' | 'save') => {
    if (type === 'save') {
      setIsSaved(!isSaved);
    } else if (type !== 'skip') {
      setFeedback(feedback === type ? null : type);
    }
    
    onFeedback?.({
      track_id: track.id,
      feedback_type: type === 'save' ? (isSaved ? 'save' : 'save') : type,
      context: {
        source: track.source,
        recommendation_reason: track.recommendation_reason
      }
    });
  };

  const handlePlay = () => {
    onFeedback?.({
      track_id: track.id,
      feedback_type: 'play_full',
      context: { source: track.source }
    });
  };

  const handleShare = () => {
    onFeedback?.({
      track_id: track.id,
      feedback_type: 'share',
      context: { source: track.source }
    });
  };

  const handleAddToPlaylist = () => {
    onAddToPlaylist?.(track.spotify_id || track.id);
  };

  const openSpotify = () => {
    if (track.spotify_url) {
      window.open(track.spotify_url, '_blank');
    }
  };

  const getSourceLabel = (source: string) => {
    const labels: Record<string, string> = {
      'agent': 'AI Agent',
      'ml_hybrid': 'ML Hybrid',
      'ml_collaborative': 'Collaborative',
      'ml_content': 'Content-Based',
      'spotify_search': 'Spotify',
      'spotify_recommendations': 'Spotify Recs',
      'lastfm_similar': 'Last.fm',
      'aoty_similar': 'AOTY',
      'database': 'Database',
      'fallback': 'Search'
    };
    return labels[source] || source;
  };

  const cardSizeClasses = {
    compact: 'p-3',
    standard: 'p-4',
    detailed: 'p-5'
  };

  const imageSizeClasses = {
    compact: 'w-12 h-12',
    standard: 'w-16 h-16', 
    detailed: 'w-20 h-20'
  };

  return (
    <div className={`bg-card border border-border rounded-lg hover:shadow-md transition-all ${cardSizeClasses[size]} ${className}`}>
      <div className="flex items-start space-x-3">
        {/* Album Art */}
        <div className={`${imageSizeClasses[size]} bg-muted rounded-md flex-shrink-0 overflow-hidden`}>
          {track.artwork_url ? (
            <img 
              src={track.artwork_url} 
              alt={`${track.album || track.name} cover`}
              className="w-full h-full object-cover"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                target.nextElementSibling?.classList.remove('hidden');
              }}
            />
          ) : null}
          <div className={`w-full h-full flex items-center justify-center bg-muted ${track.artwork_url ? 'hidden' : ''}`}>
            <Play className="w-6 h-6 text-muted-foreground" />
          </div>
        </div>

        {/* Track Info */}
        <div className="flex-1 min-w-0">
          <h3 className="font-inter font-semibold text-foreground truncate text-base">
            {track.name}
          </h3>
          <p className="text-muted-foreground truncate mb-1 font-inter text-sm">
            {track.artists?.join(', ') || track.artist}
          </p>
          {track.album && (
            <p className="text-xs text-muted-foreground truncate mb-2 font-inter">
              {track.album}
            </p>
          )}

          {/* Recommendation Reason */}
          {track.recommendation_reason && size !== 'compact' && (
            <div className="mb-2">
              <span className="text-xs text-blue-600 font-inter font-medium">
                Recommended by AI Agent
              </span>
              <p className="text-xs text-muted-foreground/80 mt-1 font-inter line-clamp-2">
                {track.recommendation_reason}
              </p>
            </div>
          )}

          {/* Genres/Tags */}
          {track.genres && track.genres.length > 0 && size === 'detailed' && (
            <div className="flex flex-wrap gap-1 mb-2">
              {track.genres.slice(0, 3).map((genre, index) => (
                <span 
                  key={index}
                  className="px-2 py-1 bg-muted text-xs rounded-full text-muted-foreground font-inter"
                >
                  {genre}
                </span>
              ))}
            </div>
          )}

          {/* Similarity/Confidence Score */}
          {(track.similarity_score || track.confidence_score) && size !== 'compact' && (
            <div className="mb-2">
              <span className="text-xs text-muted-foreground font-mono font-medium">
                {Math.round((track.similarity_score || track.confidence_score || 0) * 100)}% match
              </span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex items-center justify-between mt-3">
            {/* Left side buttons */}
            <div className="flex items-center space-x-3">
              {/* Spotify Link */}
              {track.spotify_url && (
                <button
                  onClick={openSpotify}
                  className="flex items-center space-x-1 text-sm text-green-600 hover:text-green-700 transition-colors font-inter font-medium"
                >
                  <ExternalLink className="w-4 h-4" />
                  <span className="font-inter">Open in Spotify</span>
                </button>
              )}
              
              {/* Add to Playlist Button */}
              {onAddToPlaylist && track.spotify_id && (
                <button
                  onClick={handleAddToPlaylist}
                  className="flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-700 transition-colors font-inter font-medium"
                  title="Add to playlist"
                >
                  <ListPlus className="w-4 h-4" />
                  <span className="font-inter">Add to Playlist</span>
                </button>
              )}
            </div>

            {/* Feedback Buttons - Only thumbs up/down */}
            {showFeedback && (
              <div className="flex items-center space-x-2">
                {/* Like Button */}
                <button
                  onClick={() => handleFeedback('like')}
                  className={`p-2 rounded-full transition-colors ${
                    feedback === 'like' 
                      ? 'text-green-600 bg-green-100 dark:bg-green-900/20' 
                      : 'text-muted-foreground hover:text-green-600 hover:bg-green-50 dark:hover:bg-green-900/10'
                  }`}
                  title="Like this recommendation"
                >
                  <ThumbsUp className="w-4 h-4" />
                </button>

                {/* Dislike Button */}
                <button
                  onClick={() => handleFeedback('dislike')}
                  className={`p-2 rounded-full transition-colors ${
                    feedback === 'dislike' 
                      ? 'text-red-600 bg-red-100 dark:bg-red-900/20' 
                      : 'text-muted-foreground hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/10'
                  }`}
                  title="Dislike this recommendation"
                >
                  <ThumbsDown className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Popularity Badge */}
      {track.popularity && (
        <div className="mt-3 flex justify-end">
          <span className="text-xs text-muted-foreground font-mono">
            {track.popularity}% popular
          </span>
        </div>
      )}
    </div>
  );
}