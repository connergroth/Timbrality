import { useState } from 'react';
import { Globe, Clock, Calendar, Star, Plus, Info, X } from 'lucide-react';
import type { Track } from '@/lib/agent';

interface TrackCardProps {
  track: Track;
  onFeedback?: (trackId: string, type: 'like' | 'dislike' | 'skip' | 'play_full') => void;
  showFeedback?: boolean;
}

export function TrackCard({ track, onFeedback, showFeedback = true }: TrackCardProps) {
  const [showPlaylistModal, setShowPlaylistModal] = useState(false);
  const [showInfoModal, setShowInfoModal] = useState(false);

  const openSpotify = () => {
    if (track.spotify_url) {
      window.open(track.spotify_url, '_blank');
    }
  };

  // Helper function to format duration from milliseconds to MM:SS
  const formatDuration = (durationMs: number) => {
    const minutes = Math.floor(durationMs / 60000);
    const seconds = Math.floor((durationMs % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  // Helper function to format release date from full date
  const formatReleaseDate = (releaseDate: string) => {
    if (!releaseDate) return 'Unknown';
    const date = new Date(releaseDate);
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const year = date.getFullYear();
    return `${month}-${year}`;
  };

  // Helper function to get duration (use real Spotify data)
  const getTrackDuration = () => {
    if (track.duration_ms) {
      return formatDuration(track.duration_ms);
    }
    return '3:20'; // Fallback
  };

  // Helper function to get release date (use real Spotify data)
  const getReleaseDate = () => {
    if (track.release_date) {
      return formatReleaseDate(track.release_date);
    }
    return 'Unknown';
  };

  // Helper function to get rating (use database rating, convert to 0-100 scale)
  const getRating = () => {
    if (track.rating) {
      // If rating is already 0-100 scale, use as-is, otherwise convert from 0-10 scale
      return track.rating > 10 ? track.rating : Math.round(track.rating * 10);
    }
    if (track.aoty_score) {
      // AOTY scores are typically 0-100
      return Math.round(track.aoty_score);
    }
    if (track.similarity) {
      return Math.round(track.similarity * 100);
    }
    return 85; // Fallback
  };

  // Helper function to get popularity (use real Spotify popularity)
  const getPopularity = () => {
    if (track.popularity !== undefined) {
      return track.popularity; // Spotify popularity is already 0-100
    }
    if (track.similarity) {
      return Math.round(track.similarity * 100);
    }
    return 75; // Fallback
  };

  return (
    <>
      <div className="w-[480px] h-[320px] bg-neutral-800 rounded-3xl p-6 shadow-xl transition-all duration-300 hover:scale-[1.02] cursor-pointer group relative">
        {/* Top section with artist and action buttons */}
        <div className="flex items-center justify-between mb-1">
          {/* Artist pill with real artist image or fallback */}
          <div className="bg-neutral-700 rounded-full px-3 py-1 flex items-center gap-1.5">
            <div className="w-4 h-4 rounded-full overflow-hidden flex-shrink-0">
              {track.artist_image_url ? (
                <img 
                  src={track.artist_image_url} 
                  alt={track.artist}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full bg-neutral-600 rounded-full flex items-center justify-center">
                  <span className="text-white text-[8px] font-bold">
                    {track.artist.charAt(0)}
                  </span>
                </div>
              )}
            </div>
            <span className="text-white font-medium text-xs">{track.artist}</span>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            {/* Info button */}
            <button
              onClick={() => setShowInfoModal(true)}
              className="w-6 h-6 flex items-center justify-center text-white hover:bg-neutral-700 rounded-full transition-colors duration-200"
            >
              <Info className="w-3.5 h-3.5" />
            </button>

            {/* Add to playlist button */}
            <button
              onClick={() => setShowPlaylistModal(true)}
              className="w-6 h-6 flex items-center justify-center text-white hover:bg-neutral-700 rounded-full transition-colors duration-200"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Song title - bigger */}
        <h3 className="font-playfair text-4xl font-bold text-white mb-2 leading-tight">
          {track.name}
        </h3>

        {/* Content section with album cover and stats */}
        <div className="flex gap-4 mb-3">
          {/* Album cover - use actual artwork */}
          <div className="w-36 h-36 rounded-xl flex items-center justify-center relative flex-shrink-0 overflow-hidden">
            {track.artwork_url ? (
              <img 
                src={track.artwork_url} 
                alt={`${track.album} cover`}
                className="w-full h-full object-cover"
              />
            ) : (
              <div 
                className="w-full h-full flex items-center justify-center"
                style={{ backgroundColor: '#FF7A38' }}
              >
                <div className="text-white font-bold text-sm text-center px-2">
                  {track.album || 'Unknown Album'}
                </div>
              </div>
            )}
            {/* Parental advisory label */}
            <div className="absolute bottom-1 right-1 bg-white text-black text-[9px] px-1.5 py-0.5 rounded">
              PARENTAL<br/>ADVISORY
            </div>
          </div>

          {/* Stats section - more compact */}
          <div className="flex-1 space-y-2">
            {/* Popularity */}
            <div>
              <div className="flex items-center gap-1.5 mb-0.5">
                <Globe className="w-3 h-3 text-white" />
                <span className="text-white font-medium text-xs">Popularity</span>
              </div>
              <div className="w-full bg-neutral-700 rounded-full h-4">
                <div 
                  className="bg-green-500 h-4 rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${getPopularity()}%` }}
                ></div>
              </div>
            </div>

            {/* Duration, Release Date, and Rating - positioned below and compact */}
            <div className="flex gap-3 pt-1">
              {/* Duration */}
              <div className="flex-1">
                <div className="bg-neutral-700 rounded-lg px-2.5 py-1.5 w-fit">
                  <div className="text-white text-sm font-bold">{getTrackDuration()}</div>
                </div>
                <div className="flex items-center gap-1 mt-1">
                  <Clock className="w-3 h-3 text-white" />
                  <span className="text-white text-[10px]">Length</span>
                </div>
              </div>

              {/* Release Date */}
              <div className="flex-1">
                <div className="bg-neutral-700 rounded-lg px-2.5 py-1.5 w-fit">
                  <div className="text-white text-sm font-bold">{getReleaseDate()}</div>
                </div>
                <div className="flex items-center gap-1 mt-1">
                  <Calendar className="w-3 h-3 text-white" />
                  <span className="text-white text-[10px]">Release date</span>
                </div>
              </div>

              {/* Rating */}
              <div className="flex-1">
                <div className="bg-neutral-700 rounded-lg px-2.5 py-1.5 w-fit">
                  <div className="text-white text-sm font-bold">{getRating()}</div>
                </div>
                <div className="flex items-center gap-1 mt-1">
                  <Star className="w-3 h-3 text-white" />
                  <span className="text-white text-[10px]">Rating</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Spotify button */}
        <button 
          onClick={openSpotify}
          className="w-full bg-neutral-700 hover:bg-neutral-600 rounded-xl py-3 flex items-center justify-center gap-3 transition-colors duration-200"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor" style={{ minWidth: '20px' }}>
            <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.299.421-1.02.599-1.559.3z" />
          </svg>
          <span className="text-white font-medium">Open in Spotify</span>
        </button>
      </div>

      {/* Modals */}
      {showPlaylistModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div 
            className="absolute inset-0 bg-black/50" 
            onClick={() => setShowPlaylistModal(false)}
          />
          <div className="relative bg-neutral-800 rounded-xl p-4 w-72 border border-neutral-700 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-white font-semibold text-sm">Add to Playlist</h3>
              <button 
                onClick={() => setShowPlaylistModal(false)}
                className="text-neutral-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-2">
              <button className="w-full text-left px-3 py-2 bg-neutral-700 hover:bg-neutral-600 rounded-lg text-white text-sm transition-colors">
                My Favorites
              </button>
              <button className="w-full text-left px-3 py-2 bg-neutral-700 hover:bg-neutral-600 rounded-lg text-white text-sm transition-colors">
                Chill Vibes
              </button>
              <button className="w-full text-left px-3 py-2 bg-neutral-700 hover:bg-neutral-600 rounded-lg text-white text-sm transition-colors">
                Discover Weekly
              </button>
              <button className="w-full text-left px-3 py-2 border-2 border-dashed border-neutral-600 hover:border-neutral-500 rounded-lg text-neutral-300 text-sm transition-colors">
                + Create New Playlist
              </button>
            </div>
          </div>
        </div>
      )}

      {showInfoModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div 
            className="absolute inset-0 bg-black/50" 
            onClick={() => setShowInfoModal(false)}
          />
          <div className="relative bg-neutral-800 rounded-xl p-4 w-80 border border-neutral-700 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-white font-semibold text-sm">Why This Track?</h3>
              <button 
                onClick={() => setShowInfoModal(false)}
                className="text-neutral-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="text-neutral-300 text-sm leading-relaxed">
              <p className="mb-2">
                <strong className="text-white">Recommended because:</strong>
              </p>
              <ul className="space-y-1 text-xs">
                <li>• Similar to your recently played {track.genres?.[0] || 'music'} tracks</li>
                <li>• Matches your preference for {track.artist?.toLowerCase().includes('ocean') ? 'soulful vocals' : 'this artist\'s style'}</li>
                <li>• Popular among users with similar taste profiles</li>
                <li>• High audio feature compatibility ({track.similarity ? Math.round(track.similarity * 100) : 89}% match)</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </>
  );
}