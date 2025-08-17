import { Music, Brain, Sparkles } from "lucide-react";

export const HowItWorks = () => {
  return (
    <section id="how-it-works" className="py-24 bg-neutral-900">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="font-playfair text-4xl md:text-5xl font-bold text-white mb-6">
            How It Works
          </h2>
          <p className="text-lg text-neutral-300 font-inter max-w-3xl mx-auto">
            Discover your musical DNA through our advanced AI-powered recommendation engine
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {/* Step 1 */}
          <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-8 text-center shadow-2xl transition-all duration-300 hover:scale-105 hover:bg-neutral-800/60 hover:border-neutral-700/50">
            <div className="w-16 h-16 bg-neutral-700/40 rounded-full flex items-center justify-center mx-auto mb-6">
              <Music className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-2xl font-playfair font-bold text-white mb-4">
              Connect Your Music
            </h3>
            <p className="text-neutral-300 font-inter leading-relaxed">
              Link your Spotify and Last.fm accounts to give our AI insights into your listening patterns, favorite artists, and musical preferences.
            </p>
          </div>

          {/* Step 2 */}
          <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-8 text-center shadow-2xl transition-all duration-300 hover:scale-105 hover:bg-neutral-800/60 hover:border-neutral-700/50">
            <div className="w-16 h-16 bg-neutral-700/40 rounded-full flex items-center justify-center mx-auto mb-6">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-2xl font-playfair font-bold text-white mb-4">
              AI Analysis
            </h3>
            <p className="text-neutral-300 font-inter leading-relaxed">
              Our machine learning algorithms analyze your musical DNA, identifying patterns in genres, moods, and listening habits to understand your unique taste.
            </p>
          </div>

          {/* Step 3 */}
          <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-8 text-center shadow-2xl transition-all duration-300 hover:scale-105 hover:bg-neutral-800/60 hover:border-neutral-700/50">
            <div className="w-16 h-16 bg-neutral-700/40 rounded-full flex items-center justify-center mx-auto mb-6">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-2xl font-playfair font-bold text-white mb-4">
              Discover New Music
            </h3>
            <p className="text-neutral-300 font-inter leading-relaxed">
              Receive personalized recommendations that match your taste profile, helping you discover hidden gems and expand your musical horizons.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};