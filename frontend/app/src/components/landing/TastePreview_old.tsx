import { useState, useRef, useEffect } from 'react';
import { Heart, Music, ChevronLeft, ChevronRight, Play, Pause, BarChart3, TrendingUp, Sparkles } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

export const TastePreview = () => {
  const [isPaused, setIsPaused] = useState(false);
  const carouselRef = useRef<HTMLDivElement>(null);

  // Sample track data for demo with enhanced metadata
  const tracks = [
    {
      id: 1,
      name: "Midnight City",
      artist: "M83",
      album: "Hurry Up, We're Dreaming",
      cover_art: null,
      explanation: "Atmospheric synth-pop with dreamy vocals that matches your love for ambient electronic music",
      tags: ["Electronic", "Ambient", "Synth-pop"],
      confidence: 96,
      reason: "Based on your Spotify listening history",
      year: "2011",
      duration: "4:04",
      popularity: 85
    },
    {
      id: 2,
      name: "Re: Stacks",
      artist: "Bon Iver",
      album: "For Emma, Forever Ago",
      cover_art: null,
      explanation: "Intimate folk with raw emotion, perfect for your contemplative listening moments",
      tags: ["Folk", "Indie", "Acoustic"],
      confidence: 92,
      reason: "Similar to your saved tracks",
      year: "2007",
      duration: "6:41",
      popularity: 78
    },
    {
      id: 3,
      name: "Teardrop",
      artist: "Massive Attack",
      album: "Mezzanine",
      cover_art: null,
      explanation: "Trip-hop masterpiece that aligns with your taste for atmospheric, moody music",
      tags: ["Trip-hop", "Electronic", "Atmospheric"],
      confidence: 89,
      reason: "Matches your evening listening patterns",
      year: "1998",
      duration: "5:29",
      popularity: 82
    },
    {
      id: 4,
      name: "Everything in Its Right Place",
      artist: "Radiohead",
      album: "Kid A",
      cover_art: null,
      explanation: "Experimental electronic rock that matches your appreciation for innovative soundscapes",
      tags: ["Experimental", "Electronic", "Rock"],
      confidence: 94,
      reason: "Algorithm detected similar audio features",
      year: "2000",
      duration: "4:11",
      popularity: 88
    },
    {
      id: 5,
      name: "Holocene",
      artist: "Bon Iver",
      album: "Bon Iver",
      cover_art: null,
      explanation: "Ethereal folk with rich textures, perfect for your atmospheric music preferences",
      tags: ["Folk", "Indie", "Ethereal"],
      confidence: 91,
      reason: "High compatibility with your taste profile",
      year: "2011",
      duration: "5:36",
      popularity: 76
    },
    {
      id: 6,
      name: "Strobe",
      artist: "Deadmau5",
      album: "For Lack of a Better Name",
      cover_art: null,
      explanation: "Progressive house epic that builds beautifully, matching your taste for electronic journeys",
      tags: ["Progressive House", "Electronic", "Ambient"],
      confidence: 87,
      reason: "Based on your late-night listening habits",
      year: "2009",
      duration: "10:32",
      popularity: 79
    }
  ];

  const [loading, setLoading] = useState(true);
  const [hoveredTrack, setHoveredTrack] = useState<number | null>(null);
  const [playingTrack, setPlayingTrack] = useState<number | null>(null);

  useEffect(() => {
    // Simulate loading with realistic timing
    const timer = setTimeout(() => setLoading(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  const handlePlayPause = (trackId: number) => {
    if (playingTrack === trackId) {
      setPlayingTrack(null);
    } else {
      setPlayingTrack(trackId);
    }
  };

  return (
    <section id="taste-preview" className="py-24 bg-neutral-900">
      <div className="container mx-auto px-6">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="font-playfair text-4xl md:text-5xl font-bold text-white mb-4">
            Taste Preview
          </h2>
          <p className="text-xl text-neutral-300 font-inter leading-relaxed max-w-3xl mx-auto mb-6">
            See how our AI understands your unique taste and discovers music that resonates with your soul
          </p>
              
        </div>

        {/* Carousel Container */}
        <div className="relative overflow-hidden">
          {/* Gradient fade edges */}
          <div className="absolute left-0 top-0 bottom-0 w-24 bg-gradient-to-r from-neutral-900 to-transparent z-10 pointer-events-none"></div>
          <div className="absolute right-0 top-0 bottom-0 w-24 bg-gradient-to-l from-neutral-900 to-transparent z-10 pointer-events-none"></div>
          
          {/* Speed control indicator */}
          <div className="absolute top-4 right-6 z-20">
            <div className="bg-neutral-800/80 backdrop-blur-md border border-neutral-700/50 rounded-full px-3 py-1.5 text-xs text-neutral-300 font-inter">
              {isPaused ? 'Paused' : 'Auto-scroll'}
            </div>
          </div>
          
          <div
            ref={carouselRef}
            className={`flex gap-8 ${
              isPaused ? '' : 'animate-scroll-left-smooth'
            } hover:pause-animation`}
            onMouseEnter={() => setIsPaused(true)}
            onMouseLeave={() => setIsPaused(false)}
            style={{
              width: `${tracks.length * 280 * 2}px`, // Adjusted for smaller cards
            }}
          >
            {/* First set of tracks */}
            {tracks.length > 0 && !loading && tracks.map((track, index) => (
              <div
                key={`first-${track.id}-${index}`}
                className="flex-shrink-0 w-[260px] bg-gradient-to-br from-neutral-800/50 to-neutral-900/60 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-4 space-y-3 shadow-xl transition-all duration-300 hover:scale-[1.02] hover:bg-neutral-800/70 hover:border-neutral-600/50 cursor-pointer group relative overflow-hidden"
                style={{ '--delay': index } as React.CSSProperties}
                onMouseEnter={() => setHoveredTrack(track.id)}
                onMouseLeave={() => setHoveredTrack(null)}
              >
                {/* Animated background gradient */}
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/3 via-transparent to-violet-500/3 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                
                {/* Confidence indicator */}
                <div className="absolute top-0 left-0 right-0 h-1 bg-neutral-700/30 rounded-t-2xl overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-indigo-400 to-violet-400 rounded-t-2xl transition-all duration-700 ease-out"
                    style={{ width: `${track.confidence}%` }}
                  ></div>
                </div>

                <div className="flex items-start space-x-5">
                  {/* Enhanced Album Cover with Play Button */}
                  <div className="relative w-20 h-20 rounded-2xl overflow-hidden bg-gradient-to-br from-neutral-700/60 to-neutral-800/60 flex items-center justify-center flex-shrink-0 border border-neutral-600/40 group-hover:border-neutral-500/60 transition-all duration-300 shadow-lg group-hover:shadow-xl">
                    {track.cover_art ? (
                      <img
                        src={track.cover_art}
                        alt={`${track.album} cover`}
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
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
                      className="text-2xl font-playfair font-bold text-white"
                      style={{ display: track.cover_art ? 'none' : 'flex' }}
                    >
                      <Music className="h-10 w-10" />
                    </span>
                    
                    {/* Enhanced Play button overlay */}
                    {hoveredTrack === track.id && (
                      <div 
                        className="absolute inset-0 bg-gradient-to-br from-black/70 to-black/50 flex items-center justify-center transition-all duration-300 backdrop-blur-sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handlePlayPause(track.id);
                        }}
                      >
                        <div className="bg-white/20 backdrop-blur-md rounded-full p-3 border border-white/30 transition-all duration-300 hover:scale-110">
                          {playingTrack === track.id ? (
                            <Pause className="h-6 w-6 text-white" />
                          ) : (
                            <Play className="h-6 w-6 text-white ml-0.5" />
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Enhanced Track Info */}
                  <div className="flex-1 space-y-3 min-w-0">
                    <div className="flex items-center justify-between">
                      <h3 className="font-playfair text-lg font-bold text-white leading-tight group-hover:text-neutral-100 transition-colors truncate">
                        {track.name}
                      </h3>
                      <span className="text-sm text-neutral-400 font-mono bg-neutral-800/60 px-2 py-1 rounded-md border border-neutral-700/40">{track.duration}</span>
                    </div>
                    <p className="font-inter text-base text-neutral-300 leading-tight font-medium">
                      {track.artist} • {track.year}
                    </p>
                    <p className="text-neutral-400 italic text-sm leading-relaxed line-clamp-2 font-inter bg-neutral-800/30 px-3 py-2 rounded-lg border border-neutral-700/20">
                      "{track.explanation}"
                    </p>
                  </div>
                </div>

                {/* Enhanced AI Reasoning */}
                <div className="bg-gradient-to-br from-neutral-700/30 to-neutral-800/30 rounded-2xl p-4 border border-neutral-600/30 backdrop-blur-sm">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="p-2 bg-green-500/20 rounded-lg border border-green-500/30">
                      <BarChart3 className="h-4 w-4 text-green-400" />
                    </div>
                    <span className="text-sm font-bold text-green-400 font-inter">{track.confidence}% Match</span>
                    <span className="text-sm text-neutral-400 font-inter">• {track.reason}</span>
                  </div>
                  
                  {/* Enhanced Tags */}
                  <div className="flex flex-wrap gap-2">
                    {track.tags.slice(0, 3).map((tag) => (
                      <Badge 
                        key={tag} 
                        variant="secondary" 
                        className="font-inter text-xs px-3 py-1.5 bg-gradient-to-r from-neutral-700/60 to-neutral-800/60 border-neutral-600/40 text-neutral-200 hover:from-neutral-700/80 hover:to-neutral-800/80 hover:border-neutral-500/60 transition-all duration-300 hover:scale-105"
                      >
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Enhanced Popularity indicator */}
                <div className="flex items-center justify-between text-sm text-neutral-400">
                  <div className="flex items-center gap-2 bg-neutral-800/40 px-3 py-2 rounded-lg border border-neutral-700/30">
                    <TrendingUp className="h-4 w-4 text-blue-400" />
                    <span className="font-inter font-medium">Popularity: {track.popularity}/100</span>
                  </div>
                  {playingTrack === track.id && (
                    <div className="flex items-center gap-2 bg-green-500/20 px-3 py-2 rounded-lg border border-green-500/30">
                      <div className="w-2.5 h-2.5 bg-green-400 rounded-full animate-pulse shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
                      <span className="text-green-400 font-inter font-medium">Playing</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {/* Duplicate set for seamless loop */}
            {tracks.length > 0 && !loading && tracks.map((track, index) => (
              <div
                key={`second-${track.id}-${index}`}
                className="flex-shrink-0 w-[380px] bg-gradient-to-br from-neutral-800/60 via-neutral-800/40 to-neutral-900/60 backdrop-blur-2xl border border-neutral-700/40 rounded-3xl p-6 space-y-5 shadow-2xl transition-all duration-500 hover:scale-105 hover:bg-neutral-800/80 hover:border-neutral-600/60 cursor-pointer group relative overflow-hidden hover:shadow-[0_25px_50px_-12px_rgba(0,0,0,0.8)] hover-lift hover-glow"
              >
                {/* Animated background gradient */}
                <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 via-transparent to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
                
                <div className="flex items-start space-x-5">
                  {/* Enhanced Album Cover */}
                  <div className="w-20 h-20 rounded-2xl overflow-hidden bg-gradient-to-br from-neutral-700/60 to-neutral-800/60 flex items-center justify-center flex-shrink-0 border border-neutral-600/40 group-hover:border-neutral-500/60 transition-all duration-300 shadow-lg group-hover:shadow-xl">
                    {track.cover_art ? (
                      <img
                        src={track.cover_art}
                        alt={`${track.album} cover`}
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
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
                      className="text-2xl font-playfair font-bold text-white"
                      style={{ display: track.cover_art ? 'none' : 'flex' }}
                    >
                      <Music className="h-10 w-10" />
                    </span>
                  </div>

                  {/* Enhanced Track Info */}
                  <div className="flex-1 space-y-3 min-w-0">
                    <h3 className="font-playfair text-lg font-bold text-white leading-tight group-hover:text-neutral-100 transition-colors">
                      {track.name}
                    </h3>
                    <p className="font-inter text-base text-neutral-300 leading-tight font-medium">
                      {track.artist}
                    </p>
                    <p className="text-neutral-400 italic text-sm leading-relaxed line-clamp-2 bg-neutral-800/30 px-3 py-2 rounded-lg border border-neutral-700/20">
                      "{track.explanation}"
                    </p>
                  </div>
                </div>

                {/* Enhanced Tags */}
                <div className="flex flex-wrap gap-2">
                  {track.tags.slice(0, 3).map((tag) => (
                    <Badge 
                      key={tag} 
                      variant="secondary" 
                      className="font-inter text-xs px-3 py-1.5 bg-gradient-to-r from-neutral-700/60 to-neutral-800/60 border-neutral-600/40 text-neutral-200 hover:from-neutral-700/80 hover:to-neutral-800/80 hover:border-neutral-500/60 transition-all duration-300 hover:scale-105"
                    >
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            ))}

            {/* Enhanced Loading state */}
            {loading && (
              <>
                {[...Array(5)].map((_, index) => (
                  <div
                    key={`loading-${index}`}
                    className="flex-shrink-0 w-[380px] bg-gradient-to-br from-neutral-800/60 via-neutral-800/40 to-neutral-900/60 backdrop-blur-2xl border border-neutral-700/40 rounded-3xl p-6 space-y-5 shadow-2xl animate-pulse"
                  >
                    <div className="flex items-start space-x-5">
                      <div className="w-20 h-20 rounded-2xl bg-neutral-700/40 flex-shrink-0"></div>
                      <div className="flex-1 space-y-3">
                        <div className="h-6 bg-neutral-700/40 rounded w-3/4"></div>
                        <div className="h-5 bg-neutral-700/40 rounded w-1/2"></div>
                        <div className="h-16 bg-neutral-700/40 rounded w-full"></div>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <div className="h-8 bg-neutral-700/40 rounded w-20"></div>
                      <div className="h-8 bg-neutral-700/40 rounded w-24"></div>
                      <div className="h-8 bg-neutral-700/40 rounded w-16"></div>
                    </div>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>

        {/* Bottom CTA */}
        <div className="text-center mt-16">
          <div className="bg-gradient-to-br from-neutral-800/60 to-neutral-900/80 backdrop-blur-xl border border-white/20 rounded-3xl p-8 shadow-2xl inline-block border-glow-animation relative overflow-hidden max-w-2xl">
            {/* Gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-transparent to-blue-500/10 pointer-events-none rounded-3xl"></div>
            
            <div className="relative z-10">
              <h3 className="text-2xl font-playfair font-bold text-white mb-3">
                Ready to discover your musical DNA?
              </h3>
              <p className="text-neutral-300 font-inter mb-6 leading-relaxed">
                Join thousands of music lovers who've already found their perfect sound. Connect your Spotify and let our AI curate your next favorite song.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <a
                  href="/auth"
                  className="inline-flex items-center gap-2 bg-white text-neutral-900 px-8 py-3 rounded-lg font-inter font-semibold hover:bg-neutral-100 transition-all duration-300 hover:scale-105"
                >
                  <Sparkles className="w-5 h-5" />
                  Start Your Journey
                  <ChevronRight className="w-4 h-4" />
                </a>
                <a
                  href="/pricing"
                  className="inline-flex items-center gap-2 border border-white text-white px-8 py-3 rounded-lg font-inter font-semibold hover:bg-white hover:text-neutral-900 transition-all duration-300 hover:scale-105"
                >
                  View Pricing
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};