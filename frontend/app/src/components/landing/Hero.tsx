import { Button } from "@/components/ui/button";
import { MLAnimationCard } from "@/components/recommend/MLAnimationCard";

export const Hero = () => {
  return (
    <section id="home" className="relative flex-1 bg-neutral-900 pt-16 scroll-mt-24 min-h-screen">
      {/* Large ML Animation Card spanning most of the page */}
      <div className="w-full px-4 md:px-8 lg:px-12 mt-4 md:mt-8 h-full">
        <div className="w-full h-full min-h-[calc(100vh-8rem)] md:min-h-[calc(100vh-6rem)] rounded-2xl md:rounded-3xl overflow-hidden border border-neutral-700/30 shadow-2xl relative">
          {/* ML Animation Card - Full size background */}
          <div className="absolute inset-0 w-full h-full">
            <MLAnimationCard />
          </div>
          
          {/* Header content overlay inside the card */}
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center text-center px-4 md:px-6 py-8 md:py-12">
            {/* Logo/Brand */}
            <div className="mb-6 md:mb-8">
              <h1 className="font-playfair text-5xl sm:text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-bold text-white tracking-tight">
                <span className="block">AI-Powered</span>
                <span className="block">Music Discovery</span>
              </h1>
            </div>

            {/* Subtitle */}
            <div className="space-y-4 md:space-y-6 lg:space-y-8 mb-6 md:mb-8">
              <p className="text-lg sm:text-lg md:text-xl text-neutral-300 font-inter max-w-xl md:max-w-2xl mx-auto leading-relaxed font-medium">
              Timbrality helps you discover music that reflects your unique taste, shaped by your patterns, moods, and preferences.
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row gap-3 md:gap-4 justify-center items-center">
                <a href="/auth" className="px-6 md:px-8 py-3 md:py-3 text-lg md:text-lg font-playfair bg-white text-neutral-900 hover:bg-neutral-100 transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2 w-full sm:w-auto rounded-lg">
                  <svg width="24" height="24" className="md:w-6 md:h-6" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                  </svg>
                  Connect with Spotify
                </a>
                <a href="/auth" className="px-6 md:px-8 py-3 md:py-3 text-lg md:text-lg font-playfair border border-white text-white hover:bg-white hover:text-neutral-900 transition-all duration-300 hover:scale-105 w-full sm:w-auto rounded-lg flex items-center justify-center">
                  Explore as Guest
                </a>
              </div>

              {/* Subtle tagline */}
              <div className="opacity-60 hidden sm:block">
                <p className="text-xs md:text-sm font-playfair text-neutral-400 italic">
                /ˈtambər/ · The quality of a sound that distinguishes different types of musical instruments or voices
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};