import { NetworkGraph } from "./NetworkGraph";
import { AudioWaveformAnimation } from "./AudioWaveformAnimation";
import { MLGraphAnimation } from "./MLGraphAnimation";
import { PlaylistAnimation } from "./PlaylistAnimation";

export const Features = () => {
  return (
    <section id="features" className="py-32 bg-neutral-900">
      <div className="container mx-auto px-8">
        <div className="text-center mb-20">
          <h2 className="font-playfair text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-8">
            Features
          </h2>
          <p className="text-xl text-neutral-300 font-inter max-w-4xl mx-auto leading-relaxed">
            Experience music discovery that goes beyond simple genre matching
          </p>
        </div>

        {/* Hero Features Row - AI-Powered and Community-Driven */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16 max-w-7xl mx-auto">
          {/* AI-Powered - Hero Card with Flashy Visuals */}
          <div 
            className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-6 text-center shadow-2xl transition-all duration-300 hover:bg-neutral-800/60 hover:border-neutral-700/50 animate-card-enter"
            style={{ '--delay': 0 } as React.CSSProperties}
          >
            <div className="mb-8">
              <h3 className="text-2xl md:text-3xl lg:text-4xl font-playfair font-bold text-white mb-6">
                AI-Powered
              </h3>
              <p className="text-sm md:text-base text-neutral-400 font-inter font-medium leading-relaxed mb-8">
                Advanced machine learning algorithms that continuously learn and improve your recommendations. Our AI analyzes thousands of musical features to understand your unique taste profile.
              </p>
            </div>
            
            {/* AI Processing Visualization */}
            <div className="mt-8">
              <div className="relative transform scale-125">
                <MLGraphAnimation />
              </div>
            </div>
          </div>

          {/* Community-Driven Discovery - Hero Card with Flashy Visuals */}
          <div 
            className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-6 text-center shadow-2xl transition-all duration-300 hover:bg-neutral-800/60 hover:border-neutral-700/50 animate-card-enter"
            style={{ '--delay': 1 } as React.CSSProperties}
          >
            <div className="mb-8">
              <h3 className="text-2xl md:text-3xl lg:text-4xl font-playfair font-bold text-white mb-6">
                Community-Driven Discovery
              </h3>
              <p className="text-sm md:text-base text-neutral-400 font-inter font-medium leading-relaxed mb-8">
                Join a vibrant network of music enthusiasts, share recommendations, and explore trending tracks within your musical community. Our collaborative approach helps surface hidden gems.
              </p>
            </div>
            
            {/* Network Graph */}
            <div className="mt-8">
              <div className="relative transform scale-125">
                <NetworkGraph />
              </div>
            </div>
          </div>
        </div>

        {/* Supporting Features Row - More Subtle */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl mx-auto">
          {/* Listening Trends - Subtle Card */}
          <div 
            className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-6 text-center shadow-2xl transition-all duration-300 hover:bg-neutral-800/60 hover:border-neutral-700/50 animate-card-enter"
            style={{ '--delay': 2 } as React.CSSProperties}
          >
            <div className="mb-8">
              <h3 className="text-2xl md:text-3xl lg:text-4xl font-playfair font-bold text-white mb-6">
                Audio Analysis
              </h3>
              <p className="text-sm md:text-base text-neutral-400 font-inter font-medium leading-relaxed mb-8">
                Deep dive into the sonic DNA of tracks with advanced audio feature analysis including tempo, energy, valence, and acousticness.
              </p>
            </div>
            
            {/* Audio Waveform Animation */}
            <div className="mt-8">
              <div className="relative">
                <AudioWaveformAnimation />
              </div>
            </div>
          </div>

          {/* Mood & Vibe Detection - Subtle Card */}
          <div 
            className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-6 text-center shadow-2xl transition-all duration-300 hover:bg-neutral-800/60 hover:border-neutral-700/50 animate-card-enter"
            style={{ '--delay': 3 } as React.CSSProperties}
          >
            <div className="mb-8">
              <h3 className="text-2xl md:text-3xl lg:text-4xl font-playfair font-bold text-white mb-6">
                Smart Playlist Creation
              </h3>
              <p className="text-sm md:text-base text-neutral-400 font-inter font-medium leading-relaxed mb-8">
                Automatically generate curated playlists based on your preferences, listening patterns, and discovered music. Export directly to Spotify with one click.
              </p>
            </div>
            
            {/* Playlist Creation Visualization */}
            <div className="mt-8">
              <div className="relative">
                <PlaylistAnimation />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};