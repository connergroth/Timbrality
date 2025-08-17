import { Brain, Network, BarChart3, Palette } from "lucide-react";
import { NetworkGraph } from "./NetworkGraph";
import { AudioWaveformAnimation } from "./AudioWaveformAnimation";
import { MLGraphAnimation } from "./MLGraphAnimation";
import { MoodVibeAnimation } from "./MoodVibeAnimation";

export const Features = () => {
  return (
    <section id="features" className="py-32 bg-neutral-900">
      <div className="container mx-auto px-8">
        <div className="text-center mb-20">
          <h2 className="font-playfair text-4xl md:text-6xl font-bold text-white mb-8">
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
            <div className="mb-6">
              <div className="inline-flex items-center justify-center w-14 h-14 bg-neutral-700/40 rounded-xl mb-4">
                <Brain className="h-7 w-7 text-neutral-300" />
              </div>
              <h3 className="text-xl font-playfair font-bold text-white mb-4">
                AI-Powered
              </h3>
              <p className="text-xs text-neutral-300 font-inter leading-relaxed mb-6">
                Advanced machine learning algorithms that continuously learn and improve your recommendations. Our AI analyzes thousands of musical features to understand your unique taste profile.
              </p>
            </div>
            
            {/* AI Processing Visualization */}
            <div className="mt-6">
              <div className="relative">
                <MLGraphAnimation />
              </div>
              
              {/* ML Algorithm Details */}
              <div className="mt-6">
                <p className="text-xs text-neutral-300 font-inter leading-relaxed">
                  We use collaborative filtering, content-based filtering, neural networks, and embedding models to learn your music taste.
                </p>
              </div>
            </div>
          </div>

          {/* Community-Driven Discovery - Hero Card with Flashy Visuals */}
          <div 
            className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-6 text-center shadow-2xl transition-all duration-300 hover:bg-neutral-800/60 hover:border-neutral-700/50 animate-card-enter"
            style={{ '--delay': 1 } as React.CSSProperties}
          >
            <div className="mb-6">
              <div className="inline-flex items-center justify-center w-14 h-14 bg-neutral-700/40 rounded-xl mb-4">
                <Network className="h-7 w-7 text-neutral-300" />
              </div>
              <h3 className="text-xl font-playfair font-bold text-white mb-4">
                Community-Driven Discovery
              </h3>
              <p className="text-xs text-neutral-300 font-inter leading-relaxed mb-6">
                Join a vibrant network of music enthusiasts, share recommendations, and explore trending tracks within your musical community. Our collaborative approach helps surface hidden gems.
              </p>
            </div>
            
            {/* Network Graph */}
            <div className="mt-6">
              <div className="relative">
                <NetworkGraph />
              </div>
              
              {/* Community Features Details */}
              <div className="mt-6">
                <p className="text-xs text-neutral-300 font-inter leading-relaxed">
                  Connect with music lovers who share your taste and discover trending tracks through collaborative recommendations.
                </p>
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
            <div className="mb-6">
              <div className="inline-flex items-center justify-center w-14 h-14 bg-neutral-700/40 rounded-xl mb-4">
                <BarChart3 className="h-7 w-7 text-neutral-300" />
              </div>
              <h3 className="text-xl font-playfair font-bold text-white mb-4">
                Listening Trends
              </h3>
              <p className="text-xs text-neutral-300 font-inter leading-relaxed mb-6">
                Visualize your musical journey through real-time audio waveform analysis. See how your listening patterns evolve and discover the unique rhythms that define your taste.
              </p>
            </div>
            
            {/* Audio Waveform Animation */}
            <div className="mt-6">
              <div className="relative">
                <AudioWaveformAnimation />
              </div>
              
              {/* Listening Trends Details */}
              <div className="mt-4">
                <div className="flex flex-wrap justify-center gap-4 text-xs font-inter">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{backgroundColor: '#6366F1', opacity: 0.8}}></div>
                    <span className="text-neutral-400">Morning (40%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{backgroundColor: '#7C3AED', opacity: 0.8}}></div>
                    <span className="text-neutral-400">Afternoon (60%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{backgroundColor: '#6366F1', opacity: 0.8}}></div>
                    <span className="text-neutral-400">Evening (80%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{backgroundColor: '#7C3AED', opacity: 0.8}}></div>
                    <span className="text-neutral-400">Night (30%)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Mood & Vibe Detection - Subtle Card */}
          <div 
            className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-6 text-center shadow-2xl transition-all duration-300 hover:bg-neutral-800/60 hover:border-neutral-700/50 animate-card-enter"
            style={{ '--delay': 3 } as React.CSSProperties}
          >
            <div className="mb-6">
              <div className="inline-flex items-center justify-center w-14 h-14 bg-neutral-700/40 rounded-xl mb-4">
                <Palette className="h-7 w-7 text-neutral-300" />
              </div>
              <h3 className="text-xl font-playfair font-bold text-white mb-4">
                Mood & Vibe Detection
              </h3>
              <p className="text-xs text-neutral-300 font-inter leading-relaxed mb-6">
                Explore music tailored to your energy, vibe, or moment — from late-night ambient to high-energy hip-hop.
              </p>
            </div>
            
            {/* Mood Detection Visualization */}
            <div className="mt-6">
              <div className="relative">
                <MoodVibeAnimation />
                
                {/* Mood and Energy Level Indicators */}
                <div className="mt-4 flex flex-wrap justify-center gap-4 text-xs font-inter">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{backgroundColor: '#6366F1', opacity: 0.7}}></div>
                    <span className="text-neutral-400">Ambient (30%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{backgroundColor: '#7C3AED', opacity: 0.7}}></div>
                    <span className="text-neutral-400">Chill (50%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{backgroundColor: '#F472B6', opacity: 0.7}}></div>
                    <span className="text-neutral-400">Groovy (70%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{backgroundColor: '#F472B6', opacity: 0.7}}></div>
                    <span className="text-neutral-400">Energetic (80%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{backgroundColor: '#F472B6', opacity: 0.7}}></div>
                    <span className="text-neutral-400">Intense (100%)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};