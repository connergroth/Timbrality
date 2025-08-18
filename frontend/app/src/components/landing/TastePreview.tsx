import { useState, useRef, useEffect } from 'react';
import { Globe, Clock, Calendar, User, Star, Plus, Info, X } from 'lucide-react';

interface SpotifyTrackData {
  id: string;
  name: string;
  artist: string;
  artists: string[];
  album: string;
  album_id: string;
  artwork_url: string | null;
  preview_url: string | null;
  external_urls: {
    spotify: string;
  };
  duration_ms: number;
  popularity: number;
  release_date: string;
}

interface EnrichedTrack {
  id: number;
  name: string;
  artist: string;
  album: string;
  cover_color: string;
  duration: string;
  release_date: string;
  popularity: number;
  rank: number;
  rating: number;
  avatar: string;
  artwork_url?: string | null;
  artist_image_url?: string | null;
  spotify_url?: string;
}

export const TastePreview = () => {
  const carouselRef = useRef<HTMLDivElement>(null);

  // Initial track data - will be enriched with Spotify data
  const initialTracks: EnrichedTrack[] = [
    {
      id: 1,
      name: "Pyramids",
      artist: "Frank Ocean",
      album: "channel ORANGE",
      cover_color: "#FF7A38",
      duration: "03:20",
      release_date: "07-2012",
      popularity: 85,
      rank: 3,
      rating: 97,
      avatar: "👤"
    },
    {
      id: 2,
      name: "Ensalada",
      artist: "Freddie Gibbs",
      album: "channel ORANGE",
      cover_color: "#FF7A38",
      duration: "09:53",
      release_date: "07-2012",
      popularity: 78,
      rank: 1,
      rating: 94,
      avatar: "👤"  
    },
    {
      id: 3,
      name: "VCRs",
      artist: "JID",
      album: "God Does Like Ugly",
      cover_color: "#FF7A38",
      duration: "05:04",
      release_date: "07-2012",
      popularity: 82,
      rank: 2,
      rating: 88,
      avatar: "👤"
    },
    {
      id: 4,
      name: "Toronto 2014",
      artist: "Daniel Caesar",
      album: "NEVER ENOUGH",
      cover_color: "#FF7A38",
      duration: "04:23",
      release_date: "07-2012",
      popularity: 75,
      rank: 4,
      rating: 83,
      avatar: "👤"
    },
    {
      id: 5,
      name: "Aria Math",
      artist: "C418",
      album: "Minecraft: Volume Beta",
      cover_color: "#FF7A38",
      duration: "03:55",
      release_date: "07-2012",
      popularity: 88,
      rank: 5,
      rating: 98,
      avatar: "👤"
    }
  ];

  const [tracks, setTracks] = useState<EnrichedTrack[]>(initialTracks);

  const [loading, setLoading] = useState(true);
  const [showPlaylistModal, setShowPlaylistModal] = useState<number | null>(null);
  const [showInfoModal, setShowInfoModal] = useState<number | null>(null);

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

  // Function to fetch artist image
  const fetchArtistImage = async (artistName: string): Promise<string | null> => {
    try {
      const response = await fetch(`/api/spotify/search-artist?q=${encodeURIComponent(artistName)}`);
      
      if (!response.ok) {
        return null;
      }
      
      const data = await response.json();
      return data.profile_picture || null;
    } catch (error) {
      console.error(`Error fetching artist image for ${artistName}:`, error);
      return null;
    }
  };

  // Function to enrich tracks with Spotify data
  const enrichTracksWithSpotifyData = async () => {
    const enrichedTracks = await Promise.all(
      initialTracks.map(async (track) => {
        const [spotifyData, artistImage] = await Promise.all([
          fetchSpotifyTrackData(track.name, track.artist),
          fetchArtistImage(track.artist)
        ]);
        
        if (spotifyData) {
          // Convert duration from ms to MM:SS format
          const minutes = Math.floor(spotifyData.duration_ms / 60000);
          const seconds = Math.floor((spotifyData.duration_ms % 60000) / 1000);
          const formattedDuration = `${minutes}:${seconds.toString().padStart(2, '0')}`;
          
          // Format release date to MM-YYYY
          const formatReleaseDate = (releaseDate: string) => {
            if (!releaseDate) return '12-2023';
            const date = new Date(releaseDate);
            const month = (date.getMonth() + 1).toString().padStart(2, '0');
            const year = date.getFullYear();
            return `${month}-${year}`;
          };
          
          return {
            ...track,
            artwork_url: spotifyData.artwork_url,
            artist_image_url: artistImage,
            spotify_url: spotifyData.external_urls.spotify,
            duration: formattedDuration,
            popularity: spotifyData.popularity,
            release_date: formatReleaseDate(spotifyData.release_date),
          };
        }
        
        return track;
      })
    );
    
    setTracks(enrichedTracks);
    setLoading(false);
  };

  useEffect(() => {
    // Start with a delay to show loading animation
    const timer = setTimeout(() => {
      enrichTracksWithSpotifyData();
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    <section id="taste-preview" className="py-24 bg-neutral-900">
      <div className="container mx-auto px-6">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="font-playfair text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-4">
            Discovery Engine
          </h2>
          <p className="text-xl text-neutral-300 font-inter leading-relaxed max-w-3xl mx-auto mb-6">
          Continuously finds new tracks tailored to your musical profile
          </p>
        </div>

        {/* Carousel Container */}
        <div className="relative overflow-hidden">
          {/* Gradient fade edges */}
          <div className="absolute left-0 top-0 bottom-0 w-24 bg-gradient-to-r from-neutral-900 to-transparent z-10 pointer-events-none"></div>
          <div className="absolute right-0 top-0 bottom-0 w-24 bg-gradient-to-l from-neutral-900 to-transparent z-10 pointer-events-none"></div>
          
          <div
            ref={carouselRef}
            className="flex gap-6 animate-scroll-left-smooth"
            style={{
              width: `${tracks.length * 480 * 2}px`,
            }}
          >
            {/* First set of tracks */}
            {tracks.length > 0 && !loading && tracks.map((track, index) => (
              <div key={`first-${track.id}-${index}`} className="flex-shrink-0">
                <div
                  className="w-[460px] h-[320px] bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-6 shadow-xl transition-all duration-300 hover:scale-[1.02] cursor-pointer group relative"
                  style={{ '--delay': index } as React.CSSProperties}
                >
                  {/* Top section with artist and action buttons */}
                  <div className="flex items-center justify-between mb-1">
                    {/* Artist pill - smaller */}
                    <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-full px-3 py-1 flex items-center gap-1.5">
                      <div className="w-4 h-4 rounded-full overflow-hidden flex-shrink-0">
                        {track.artist_image_url ? (
                          <img 
                            src={track.artist_image_url} 
                            alt={track.artist}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full bg-neutral-600/60 backdrop-blur-sm rounded-full flex items-center justify-center">
                            <User className="w-2.5 h-2.5 text-white" />
                          </div>
                        )}
                      </div>
                      <span className="text-white font-medium text-xs">{track.artist}</span>
                    </div>

                    {/* Action buttons */}
                    <div className="flex items-center gap-2">
                      {/* Info button */}
                      <button
                        onClick={() => setShowInfoModal(track.id)}
                        className="w-6 h-6 flex items-center justify-center text-white hover:bg-neutral-700/60 backdrop-blur-sm rounded-full transition-colors duration-200"
                      >
                        <Info className="w-3.5 h-3.5" />
                      </button>

                      {/* Add to playlist button */}
                      <button
                        onClick={() => setShowPlaylistModal(track.id)}
                        className="w-6 h-6 flex items-center justify-center text-white hover:bg-neutral-700/60 backdrop-blur-sm rounded-full transition-colors duration-200"
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
                    {/* Album cover */}
                    <div className="w-32 h-32 rounded-xl flex items-center justify-center relative flex-shrink-0 overflow-hidden">
                      {track.artwork_url ? (
                        <img 
                          src={track.artwork_url} 
                          alt={`${track.album} cover`}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div 
                          className="w-full h-full flex items-center justify-center"
                          style={{ backgroundColor: track.cover_color }}
                        >
                          <div className="text-white font-bold text-sm text-center px-2">{track.album}</div>
                        </div>
                      )}
                    </div>

                    {/* Stats section */}
                    <div className="flex-1 space-y-3">
                      {/* Popularity */}
                      <div>
                        <div className="flex items-center gap-1 mb-1">
                          <Globe className="w-3 h-3 text-white" />
                          <span className="text-white font-medium text-sm">Popularity</span>
                        </div>
                        <div className="w-full bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-full h-6">
                          <div 
                            className="bg-purple-500 h-6 rounded-full transition-all duration-700 ease-out"
                            style={{ width: `${track.popularity}%` }}
                          ></div>
                        </div>
                      </div>

                      {/* Duration, Release Date, and Rating */}
                      <div className="flex gap-3">
                        {/* Duration */}
                        <div>
                          <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-xl px-3 py-2 inline-block">
                            <div className="text-white text-lg font-bold">{track.duration}</div>
                          </div>
                          <div className="flex items-center gap-1 mt-0.5">
                            <Clock className="w-2.5 h-2.5 text-white" />
                            <span className="text-white text-xs">Length</span>
                          </div>
                        </div>

                        {/* Release Date */}
                        <div>
                          <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-xl px-3 py-2 inline-block">
                            <div className="text-white text-lg font-bold">{track.release_date}</div>
                          </div>
                          <div className="flex items-center gap-1 mt-0.5">
                            <Calendar className="w-2.5 h-2.5 text-white" />
                            <span className="text-white text-xs whitespace-nowrap">Release date</span>
                          </div>
                        </div>

                        {/* Rating */}
                        <div>
                          <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-xl px-3 py-2 inline-block">
                            <div className="text-white text-lg font-bold">{track.rating}</div>
                          </div>
                          <div className="flex items-center gap-1 mt-0.5">
                            <Star className="w-2.5 h-2.5 text-white" />
                            <span className="text-white text-xs">Rating</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Spotify button */}
                  <button 
                    onClick={() => track.spotify_url && window.open(track.spotify_url, '_blank')}
                    className="w-full bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 hover:bg-neutral-600/60 rounded-xl py-3 flex items-center justify-center gap-3 transition-colors duration-200"
                  >
                    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor" style={{ minWidth: '20px' }}>
                      <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.299.421-1.02.599-1.559.3z" />
                    </svg>
                    <span className="text-white font-medium">Open in Spotify</span>
                  </button>
                </div>
              </div>
            ))}
            
            {/* Duplicate set for seamless loop */}
            {tracks.length > 0 && !loading && tracks.map((track, index) => (
              <div key={`second-${track.id}-${index}`} className="flex-shrink-0">
                <div className="w-[460px] h-[320px] bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-6 shadow-xl transition-all duration-300 hover:scale-[1.02] cursor-pointer group relative">
                  {/* Top section with artist and action buttons */}
                  <div className="flex items-center justify-between mb-1">
                    {/* Artist pill - smaller */}
                    <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-full px-3 py-1 flex items-center gap-1.5">
                      <div className="w-4 h-4 rounded-full overflow-hidden flex-shrink-0">
                        {track.artist_image_url ? (
                          <img 
                            src={track.artist_image_url} 
                            alt={track.artist}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full bg-neutral-600/60 backdrop-blur-sm rounded-full flex items-center justify-center">
                            <User className="w-2.5 h-2.5 text-white" />
                          </div>
                        )}
                      </div>
                      <span className="text-white font-medium text-xs">{track.artist}</span>
                    </div>

                    {/* Action buttons */}
                    <div className="flex items-center gap-2">
                      {/* Info button */}
                      <button
                        onClick={() => setShowInfoModal(track.id)}
                        className="w-6 h-6 flex items-center justify-center text-white hover:bg-neutral-700/60 backdrop-blur-sm rounded-full transition-colors duration-200"
                      >
                        <Info className="w-3.5 h-3.5" />
                      </button>

                      {/* Add to playlist button */}
                      <button
                        onClick={() => setShowPlaylistModal(track.id)}
                        className="w-6 h-6 flex items-center justify-center text-white hover:bg-neutral-700/60 backdrop-blur-sm rounded-full transition-colors duration-200"
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
                    {/* Album cover */}
                    <div className="w-32 h-32 rounded-xl flex items-center justify-center relative flex-shrink-0 overflow-hidden">
                      {track.artwork_url ? (
                        <img 
                          src={track.artwork_url} 
                          alt={`${track.album} cover`}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div 
                          className="w-full h-full flex items-center justify-center"
                          style={{ backgroundColor: track.cover_color }}
                        >
                          <div className="text-white font-bold text-sm text-center px-2">{track.album}</div>
                        </div>
                      )}

                    </div>

                    {/* Stats section */}
                    <div className="flex-1 space-y-3">
                      {/* Popularity */}
                      <div>
                        <div className="flex items-center gap-1 mb-1">
                          <Globe className="w-3 h-3 text-white" />
                          <span className="text-white font-medium text-sm">Popularity</span>
                        </div>
                        <div className="w-full bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-full h-6">
                          <div 
                            className="bg-purple-500 h-6 rounded-full transition-all duration-700 ease-out"
                            style={{ width: `${track.popularity}%` }}
                          ></div>
                        </div>
                      </div>

                      {/* Duration, Release Date, and Rating */}
                      <div className="flex gap-3">
                        {/* Duration */}
                        <div>
                          <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-xl px-3 py-2 inline-block">
                            <div className="text-white text-lg font-bold">{track.duration}</div>
                          </div>
                          <div className="flex items-center gap-1 mt-0.5">
                            <Clock className="w-2.5 h-2.5 text-white" />
                            <span className="text-white text-xs">Length</span>
                          </div>
                        </div>

                        {/* Release Date */}
                        <div>
                          <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-xl px-3 py-2 inline-block">
                            <div className="text-white text-lg font-bold">{track.release_date}</div>
                          </div>
                          <div className="flex items-center gap-1 mt-0.5">
                            <Calendar className="w-2.5 h-2.5 text-white" />
                            <span className="text-white text-xs whitespace-nowrap">Release date</span>
                          </div>
                        </div>

                        {/* Rating */}
                        <div>
                          <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-xl px-3 py-2 inline-block">
                            <div className="text-white text-lg font-bold">{track.rating}</div>
                          </div>
                          <div className="flex items-center gap-1 mt-0.5">
                            <Star className="w-2.5 h-2.5 text-white" />
                            <span className="text-white text-xs">Rating</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Spotify button */}
                  <button 
                    onClick={() => track.spotify_url && window.open(track.spotify_url, '_blank')}
                    className="w-full bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 hover:bg-neutral-600/60 rounded-xl py-3 flex items-center justify-center gap-3 transition-colors duration-200"
                  >
                    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor" style={{ minWidth: '20px' }}>
                      <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.299.421-1.02.599-1.559.3z" />
                    </svg>
                    <span className="text-white font-medium">Open in Spotify</span>
                  </button>
                </div>
              </div>
            ))}

            {/* Loading state */}
            {loading && (
              <>
                {[...Array(3)].map((_, index) => (
                  <div key={`loading-${index}`} className="flex-shrink-0">
                    <div className="w-[460px] h-[320px] bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-6 shadow-xl animate-pulse">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="w-32 h-6 bg-neutral-700 rounded-full"></div>
                      </div>
                      <div className="w-3/4 h-10 bg-neutral-700 rounded mb-4"></div>
                      <div className="flex gap-4 mb-4">
                        <div className="w-32 h-32 bg-neutral-700 rounded-xl"></div>
                        <div className="flex-1 space-y-3">
                          <div className="w-3/4 h-4 bg-neutral-700 rounded"></div>
                          <div className="flex gap-3">
                            <div>
                              <div className="h-10 bg-neutral-700 rounded-xl w-16"></div>
                              <div className="flex items-center gap-1 mt-0.5">
                                <div className="w-2.5 h-2.5 bg-neutral-700 rounded"></div>
                                <div className="w-10 h-2.5 bg-neutral-700 rounded"></div>
                              </div>
                            </div>
                            <div>
                              <div className="h-10 bg-neutral-700 rounded-xl w-20"></div>
                              <div className="flex items-center gap-1 mt-0.5">
                                <div className="w-2.5 h-2.5 bg-neutral-700 rounded"></div>
                                <div className="w-16 h-2.5 bg-neutral-700 rounded"></div>
                              </div>
                            </div>
                            <div>
                              <div className="h-10 bg-neutral-700 rounded-xl w-12"></div>
                              <div className="flex items-center gap-1 mt-0.5">
                                <div className="w-2.5 h-2.5 bg-neutral-700 rounded"></div>
                                <div className="w-12 h-2.5 bg-neutral-700 rounded"></div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="w-full h-10 bg-neutral-700 rounded-xl"></div>
                    </div>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>

      </div>


      {/* Modals */}
      {showPlaylistModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div 
            className="absolute inset-0 bg-black/50" 
            onClick={() => setShowPlaylistModal(null)}
          />
          <div className="relative bg-neutral-800 rounded-xl p-4 w-72 border border-neutral-700 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-white font-semibold text-sm">Add to Playlist</h3>
              <button 
                onClick={() => setShowPlaylistModal(null)}
                className="text-neutral-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-2">
              <button className="w-full text-left px-3 py-2 bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 hover:bg-neutral-600/60 rounded-lg text-white text-sm transition-colors">
                My Favorites
              </button>
              <button className="w-full text-left px-3 py-2 bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 hover:bg-neutral-600/60 rounded-lg text-white text-sm transition-colors">
                Chill Vibes
              </button>
              <button className="w-full text-left px-3 py-2 bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 hover:bg-neutral-600/60 rounded-lg text-white text-sm transition-colors">
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
            onClick={() => setShowInfoModal(null)}
          />
          <div className="relative bg-neutral-800 rounded-xl p-4 w-80 border border-neutral-700 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-white font-semibold text-sm">Why This Track?</h3>
              <button 
                onClick={() => setShowInfoModal(null)}
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
                <li>• Similar to your recently played R&B tracks</li>
                <li>• Matches your preference for soulful vocals</li>
                <li>• Popular among users with similar taste profiles</li>
                <li>• High audio feature compatibility (89% match)</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};    