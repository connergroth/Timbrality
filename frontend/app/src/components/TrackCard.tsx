import { useState } from 'react';
import { Heart, Play, ExternalLink, ThumbsUp, ThumbsDown, SkipForward } from 'lucide-react';
import type { Track } from '@/lib/agent';

interface TrackCardProps {
  track: Track;
  onFeedback?: (trackId: string, type: 'like' | 'dislike' | 'skip' | 'play_full') => void;
  showFeedback?: boolean;
}

export function TrackCard({ track, onFeedback, showFeedback = true }: TrackCardProps) {
  const [feedback, setFeedback] = useState<'like' | 'dislike' | null>(null);

  const handleFeedback = (type: 'like' | 'dislike' | 'skip') => {
    setFeedback(type === 'skip' ? null : type);
    onFeedback?.(track.id, type);
  };

  const handlePlay = () => {
    if (track.preview_url || track.spotify_url) {
      onFeedback?.(track.id, 'play_full');
    }
  };

  const openSpotify = () => {
    if (track.spotify_url) {
      window.open(track.spotify_url, '_blank');
    }
  };

  return (
    <div className="bg-card border border-border rounded-lg p-4 hover:shadow-md transition-all">
      <div className="flex items-start space-x-3">
        {/* Album Art */}
        <div className="w-16 h-16 bg-muted rounded-md flex-shrink-0 overflow-hidden">
          {track.artwork_url ? (
            <img 
              src={track.artwork_url} 
              alt={`${track.album} cover`}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-muted">
              <Play className="w-6 h-6 text-muted-foreground" />
            </div>
          )}
        </div>

        {/* Track Info */}
        <div className="flex-1 min-w-0">
          <h3 className="font-inter font-semibold text-foreground truncate text-base">
            {track.name}
          </h3>
          <p className="text-muted-foreground truncate mb-1 font-inter text-sm">
            {track.artist}
          </p>
          {track.album && (
            <p className="text-xs text-muted-foreground truncate mb-2 font-inter">
              {track.album}
            </p>
          )}

          {/* Genres */}
          {track.genres && track.genres.length > 0 && (
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

          {/* Similarity Score */}
          {track.similarity && (
            <div className="mb-2">
              <span className="text-xs text-muted-foreground font-mono font-medium">
                {Math.round(track.similarity * 100)}% match
              </span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex items-center justify-between mt-3">
            <div className="flex items-center space-x-2">
              {/* Play Button */}
              {(track.preview_url || track.spotify_url) && (
                <button
                  onClick={handlePlay}
                  className="flex items-center space-x-1 text-sm text-primary hover:text-primary/80 transition-colors font-inter"
                >
                  <Play className="w-4 h-4" />
                  <span className="font-inter">Play</span>
                </button>
              )}

              {/* Spotify Link */}
              {track.spotify_url && (
                <button
                  onClick={openSpotify}
                  className="flex items-center space-x-1 text-sm text-muted-foreground hover:text-foreground transition-colors font-inter"
                >
                  <ExternalLink className="w-4 h-4" />
                  <span className="font-inter">Spotify</span>
                </button>
              )}
            </div>

            {/* Feedback Buttons */}
            {showFeedback && (
              <div className="flex items-center space-x-1">
                <button
                  onClick={() => handleFeedback('like')}
                  className={`p-1 rounded transition-colors ${
                    feedback === 'like' 
                      ? 'text-green-600 bg-green-100 dark:bg-green-900/20' 
                      : 'text-muted-foreground hover:text-green-600'
                  }`}
                >
                  <ThumbsUp className="w-4 h-4" />
                </button>
                <button
                  onClick={() => handleFeedback('dislike')}
                  className={`p-1 rounded transition-colors ${
                    feedback === 'dislike' 
                      ? 'text-red-600 bg-red-100 dark:bg-red-900/20' 
                      : 'text-muted-foreground hover:text-red-600'
                  }`}
                >
                  <ThumbsDown className="w-4 h-4" />
                </button>
                <button
                  onClick={() => handleFeedback('skip')}
                  className="p-1 rounded text-muted-foreground hover:text-foreground transition-colors"
                >
                  <SkipForward className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Source Badge */}
      <div className="mt-3 flex justify-between items-center">
        <span className="text-xs text-muted-foreground capitalize font-inter">
          via {track.source}
        </span>
      </div>
    </div>
  );
}