import { useState, useEffect } from 'react';

interface Track {
  id: number;
  title: string;
  artist: string;
  album?: string;
  artwork_url?: string | null;
  duration?: string;
  opacity: number;
  scale: number;
}

interface SpotifyTrackData {
  id: string;
  name: string;
  artist: string;
  album: string;
  artwork_url: string | null;
  duration_ms: number;
  external_urls: {
    spotify: string;
  };
}

export const PlaylistAnimation = () => {
  // Initial tracks with diverse artists - will be enriched with Spotify data
  const initialTracks: Track[] = [
    { id: 1, title: "vampire", artist: "Olivia Rodrigo", opacity: 0, scale: 0.8 },
    { id: 2, title: "THUNDRRR", artist: "Quadeca", opacity: 0, scale: 0.8 },
    { id: 3, title: "All I Need", artist: "Radiohead", opacity: 0, scale: 0.8 },
    { id: 4, title: "As It Was", artist: "Harry Styles", opacity: 0, scale: 0.8 },
    { id: 5, title: "Anti-Hero", artist: "Taylor Swift", opacity: 0, scale: 0.8 },
  ];

  const [tracks, setTracks] = useState<Track[]>(initialTracks);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showSpotify, setShowSpotify] = useState(false);
  const [dataLoaded, setDataLoaded] = useState(false);

  // Function to fetch Spotify track data
  const fetchSpotifyTrackData = async (trackName: string, artistName: string): Promise<SpotifyTrackData | null> => {
    try {
      const query = `${trackName} ${artistName}`;
      const response = await fetch(`/api/spotify/search-track?q=${encodeURIComponent(query)}`);
      
      if (!response.ok) {
        console.warn(`Failed to fetch Spotify data for ${trackName} by ${artistName}`);
        return null;
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`Error fetching Spotify data for ${trackName}:`, error);
      return null;
    }
  };

  // Function to enrich tracks with Spotify data
  const enrichTracksWithSpotifyData = async () => {
    const enrichedTracks = await Promise.all(
      initialTracks.map(async (track) => {
        const spotifyData = await fetchSpotifyTrackData(track.title, track.artist);
        
        if (spotifyData) {
          // Convert duration from ms to MM:SS format
          const minutes = Math.floor(spotifyData.duration_ms / 60000);
          const seconds = Math.floor((spotifyData.duration_ms % 60000) / 1000);
          const formattedDuration = `${minutes}:${seconds.toString().padStart(2, '0')}`;
          
          return {
            ...track,
            album: spotifyData.album,
            artwork_url: spotifyData.artwork_url,
            duration: formattedDuration,
          };
        }
        
        return { ...track, duration: '3:20' }; // Fallback duration
      })
    );
    
    setTracks(enrichedTracks);
    setDataLoaded(true);
  };

  // Load Spotify data on component mount
  useEffect(() => {
    enrichTracksWithSpotifyData();
  }, []);

  useEffect(() => {
    if (!dataLoaded) return; // Wait for Spotify data to load

    const interval = setInterval(() => {
      // Start generation cycle
      setIsGenerating(true);
      setShowSpotify(false);
      
      // Reset all tracks
      setTracks(prev => prev.map(track => ({
        ...track,
        opacity: 0,
        scale: 0.8
      })));

      // Animate tracks in sequence
      tracks.forEach((_, index) => {
        setTimeout(() => {
          setTracks(prev => prev.map((track, i) => 
            i === index 
              ? { ...track, opacity: 1, scale: 1 }
              : track
          ));
        }, 300 + index * 200);
      });

      // Show Spotify export button
      setTimeout(() => {
        setIsGenerating(false);
        setShowSpotify(true);
      }, 300 + tracks.length * 200 + 500);

    }, 4000);

    return () => clearInterval(interval);
  }, [tracks.length, dataLoaded]);

  return (
    <div className="w-full h-48 flex items-center justify-center">
      <div className="relative w-80 h-40 bg-neutral-800/60 rounded-2xl border border-neutral-600/30 p-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-white text-sm font-medium">Your Playlist</span>
          </div>
          {isGenerating && (
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 flex items-center justify-center flex-shrink-0">
                <img 
                  src="/soundwhite.png" 
                  alt="Timbrality" 
                  className="w-4 h-4 object-contain opacity-50"
                />
              </div>
              <div className="relative inline-block">
                <span className="text-xs font-inter text-neutral-400">
                  {'Creating'.split('').map((char, i) => (
                    <span
                      key={i}
                      className="inline-block"
                      style={{
                        opacity: 0.3,
                        animation: `creating-wave 1.2s ease-in-out infinite`,
                        animationDelay: `${0.1 + i * 0.08}s`
                      }}
                    >
                      {char}
                    </span>
                  ))}
                </span>
                <style jsx>{`
                  @keyframes creating-wave {
                    0%, 70%, 100% {
                      opacity: 0.3;
                    }
                    35% {
                      opacity: 1;
                    }
                  }
                `}</style>
                <span className="ml-1 text-xs font-inter text-neutral-400 animate-bounce">...</span>
              </div>
            </div>
          )}
        </div>

        {/* Track List */}
        <div className="space-y-1">
          {tracks.slice(0, 4).map((track, index) => (
            <div
              key={track.id}
              className="flex items-center gap-3 p-1 rounded transition-all duration-500"
              style={{
                opacity: track.opacity,
                transform: `scale(${track.scale})`,
              }}
            >
              <div className="w-6 h-6 rounded flex-shrink-0 overflow-hidden">
                {track.artwork_url ? (
                  <img 
                    src={track.artwork_url} 
                    alt={`${track.album} cover`}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-orange-400 to-orange-600 rounded flex items-center justify-center">
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  </div>
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-white text-xs font-medium truncate">{track.title}</div>
                <div className="text-neutral-400 text-xs truncate">{track.artist}</div>
              </div>
              <div className="text-neutral-500 text-xs">{track.duration || '3:20'}</div>
            </div>
          ))}
        </div>

        {/* Spotify Export Button */}
        {showSpotify && (
          <div 
            className="absolute bottom-2 right-2 bg-green-500 hover:bg-green-600 px-3 py-1 rounded-full flex items-center gap-1 transition-all duration-300 animate-in fade-in slide-in-from-bottom-2"
          >
            <svg className="w-3 h-3 text-white" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
            </svg>
            <span className="text-white text-xs font-medium">Export</span>
          </div>
        )}

        {/* Background glow effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 via-transparent to-blue-500/5 pointer-events-none rounded-2xl"></div>
      </div>
    </div>
  );
};