import { Badge } from "@/components/ui/badge";
import { useState, useEffect, useRef } from "react";

// Easy to edit demo songs list
const DEMO_SONGS = [
  {
    query: "Kendrick Lamar good kid maad city",
    explanation: "You might like this because of your love for jazz and hip-hop fusion.",
    tags: ["Jazz", "Hip-Hop", "Conscious Rap", "West Coast"]
  },
  {
    query: "Frank Ocean Blonde",
    explanation: "This matches your taste for emotional R&B and alternative sounds.",
    tags: ["R&B", "Alternative", "Lo-Fi", "Experimental"]
  },
  {
    query: "Tyler The Creator Igor",
    explanation: "Perfect for your eclectic taste in innovative hip-hop production.",
    tags: ["Hip-Hop", "Neo-Soul", "Experimental", "Alternative"]
  },
  {
    query: "Nujabes Metaphorical Music",
    explanation: "Recommended for your preference for contemporary R&B vibes.",
    tags: ["R&B", "Neo-Soul", "Chill", "Contemporary"]
  },
  {
    query: "C418 Minecraft Volume Alpha",
    explanation: "This aligns with your appreciation for introspective rap music.",
    tags: ["Hip-Hop", "Alternative Rap", "Jazz", "Introspective"]
  },
  {
    query: "Mac Miller Swimming",
    explanation: "Perfect match for your taste in introspective and melodic rap.",
    tags: ["Hip-Hop", "Alternative Rap", "R&B", "Melancholic"]
  },
  {
    query: "Daniel Caesar Freudian",
    explanation: "This aligns with your love for soulful R&B and contemporary sounds.",
    tags: ["R&B", "Neo-Soul", "Contemporary", "Soulful"]
  },
  {
    query: "Childish Gambino Because the Internet",
    explanation: "Recommended for your appreciation of experimental hip-hop.",
    tags: ["Hip-Hop", "Experimental", "Alternative", "R&B"]
  }
];

interface SpotifyTrack {
  id: string;
  name: string;
  artist: string;
  album: string;
  cover_art: string | null;
  explanation: string;
  tags: string[];
}

// Cache for track data
const trackCache = new Map<string, SpotifyTrack>();

export const TastePreview = () => {
  const [tracks, setTracks] = useState<SpotifyTrack[]>([]);
  const [loading, setLoading] = useState(true);
  const [isPaused, setIsPaused] = useState(false);
  const carouselRef = useRef<HTMLDivElement>(null);

  // Create a cache key based on the song query and explanation
  const getCacheKey = (songQuery: string, explanation: string) => {
    return `${songQuery}|${explanation}`;
  };

  const fetchTrackData = async (songQuery: string, explanation: string, tags: string[]) => {
    const cacheKey = getCacheKey(songQuery, explanation);
    
    // Check cache first
    if (trackCache.has(cacheKey)) {
      console.log('Using cached data for:', songQuery);
      return trackCache.get(cacheKey)!;
    }

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(
        `${backendUrl}/spotify/search-album?q=${encodeURIComponent(songQuery)}`
      );
      
      if (!response.ok) {
        throw new Error('Failed to fetch track data');
      }
      
      const data = await response.json();
      console.log('TastePreview API response:', data);
      
      const trackData = {
        id: data.id,
        name: data.name,
        artist: data.artist,
        album: data.name,
        cover_art: data.images?.[0]?.url || null,
        explanation,
        tags
      };

      // Cache the result
      trackCache.set(cacheKey, trackData);
      console.log('Cached data for:', songQuery);
      
      return trackData;
    } catch (error) {
      console.error('Error fetching track data:', error);
      // Return fallback data
      const fallbackData = {
        id: 'fallback',
        name: songQuery.split(' ').slice(0, -1).join(' '),
        artist: songQuery.split(' ').slice(-1)[0],
        album: songQuery,
        cover_art: null,
        explanation,
        tags
      };
      
      // Cache fallback data too
      trackCache.set(cacheKey, fallbackData);
      
      return fallbackData;
    }
  };

  const loadAllTracks = async () => {
    const trackPromises = DEMO_SONGS.map(song => 
      fetchTrackData(song.query, song.explanation, song.tags)
    );
    
    const loadedTracks = await Promise.all(trackPromises);
    setTracks(loadedTracks);
    setLoading(false);
  };

  // Initial load
  useEffect(() => {
    loadAllTracks();
  }, []);


  return (
    <section id="your-dna" className="py-20 scroll-mt-24">
      <div className="max-w-6xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="font-playfair text-4xl md:text-5xl font-bold text-foreground mb-4">
            Taste preview
          </h2>
          <p className="text-xl text-muted-foreground font-playfair">
            Here's what personalized discovery looks like
          </p>
        </div>

        {/* Carousel Container */}
        <div className="relative overflow-hidden">
          <div
            ref={carouselRef}
            className={`flex gap-6 ${
              isPaused ? '' : 'animate-scroll-left'
            } hover:pause-animation`}
            onMouseEnter={() => setIsPaused(true)}
            onMouseLeave={() => setIsPaused(false)}
            style={{
              width: `${tracks.length * 400 * 2}px`, // Double width for seamless loop
            }}
          >
            {/* First set of tracks */}
            {tracks.length > 0 && !loading && tracks.map((track, index) => (
              <div
                key={`first-${track.id}-${index}`}
                className="flex-shrink-0 w-[380px] bg-card border border-border rounded-xl p-6 space-y-4 shadow-lg transition-transform hover:scale-105 cursor-pointer"
              >
                <div className="flex items-start space-x-4">
                  {/* Album Cover */}
                  <div className="w-20 h-20 rounded-lg overflow-hidden bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center flex-shrink-0">
                    {track.cover_art ? (
                      <img
                        src={track.cover_art}
                        alt={`${track.album} cover`}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          const target = e.currentTarget as HTMLImageElement;
                          target.style.display = 'none';
                          const sibling = target.nextElementSibling as HTMLElement;
                          if (sibling) {
                            sibling.style.display = 'flex';
                          }
                        }}
                      />
                    ) : null}
                    <span 
                      className="text-xl font-playfair font-bold text-primary"
                      style={{ display: track.cover_art ? 'none' : 'flex' }}
                    >
                      T
                    </span>
                  </div>

                  {/* Track Info */}
                  <div className="flex-1 space-y-1 min-w-0">
                    <h3 className="font-playfair text-lg font-semibold text-foreground leading-tight">
                      {track.name}
                    </h3>
                    <p className="font-playfair text-base text-muted-foreground leading-tight">
                      {track.artist}
                    </p>
                    <p className="text-muted-foreground italic text-sm leading-relaxed line-clamp-2">
                      "{track.explanation}"
                    </p>
                  </div>
                </div>

                {/* Tags */}
                <div className="flex flex-wrap gap-2">
                  {track.tags.slice(0, 3).map((tag) => (
                    <Badge key={tag} variant="secondary" className="font-playfair text-xs px-2 py-1">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            ))}
            
            {/* Duplicate set for seamless loop */}
            {tracks.length > 0 && !loading && tracks.map((track, index) => (
              <div
                key={`second-${track.id}-${index}`}
                className="flex-shrink-0 w-[380px] bg-card border border-border rounded-xl p-6 space-y-4 shadow-lg transition-transform hover:scale-105 cursor-pointer"
              >
                <div className="flex items-start space-x-4">
                  {/* Album Cover */}
                  <div className="w-20 h-20 rounded-lg overflow-hidden bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center flex-shrink-0">
                    {track.cover_art ? (
                      <img
                        src={track.cover_art}
                        alt={`${track.album} cover`}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          const target = e.currentTarget as HTMLImageElement;
                          target.style.display = 'none';
                          const sibling = target.nextElementSibling as HTMLElement;
                          if (sibling) {
                            sibling.style.display = 'flex';
                          }
                        }}
                      />
                    ) : null}
                    <span 
                      className="text-xl font-playfair font-bold text-primary"
                      style={{ display: track.cover_art ? 'none' : 'flex' }}
                    >
                      T
                    </span>
                  </div>

                  {/* Track Info */}
                  <div className="flex-1 space-y-1 min-w-0">
                    <h3 className="font-playfair text-lg font-semibold text-foreground leading-tight">
                      {track.name}
                    </h3>
                    <p className="font-playfair text-base text-muted-foreground leading-tight">
                      {track.artist}
                    </p>
                    <p className="text-muted-foreground italic text-sm leading-relaxed line-clamp-2">
                      "{track.explanation}"
                    </p>
                  </div>
                </div>

                {/* Tags */}
                <div className="flex flex-wrap gap-2">
                  {track.tags.slice(0, 3).map((tag) => (
                    <Badge key={tag} variant="secondary" className="font-playfair text-xs px-2 py-1">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            ))}
          </div>
          
          {/* Loading state */}
          {loading && (
            <div className="flex gap-6">
              {[1, 2, 3].map((i) => (
                <div key={i} className="flex-shrink-0 w-[380px] bg-card border border-border rounded-xl p-6 space-y-4">
                  <div className="flex items-start space-x-4">
                    <div className="w-20 h-20 bg-muted-foreground/20 animate-pulse rounded-lg" />
                    <div className="flex-1 space-y-2">
                      <div className="h-5 bg-muted-foreground/20 rounded animate-pulse" />
                      <div className="h-4 bg-muted-foreground/15 rounded animate-pulse w-2/3" />
                      <div className="h-3 bg-muted-foreground/10 rounded animate-pulse w-3/4" />
                    </div>
                  </div>
                  <div className="flex gap-2">
                    {[1, 2, 3].map((j) => (
                      <div key={j} className="h-6 w-16 bg-muted-foreground/20 rounded animate-pulse" />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
};