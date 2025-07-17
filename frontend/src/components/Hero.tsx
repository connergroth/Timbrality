import { Button } from "@/components/ui/button";
import { VinylShader } from "./VinylShader";

export const Hero = () => {
  return (
    <section id="home" className="relative min-h-screen flex items-center justify-center overflow-hidden bg-background pt-16 scroll-mt-24">
      <VinylShader />
      
      {/* Main content */}
      <div className="relative z-10 text-center px-6 max-w-4xl mx-auto">
        {/* Logo/Brand */}
        <div className="mb-8">
          <h1 className="font-playfair text-6xl md:text-8xl font-bold text-primary tracking-tight">
            Timbre
          </h1>
        </div>

        {/* Main headline */}
        <div className="mb-12 space-y-4">
          <h2 className="font-playfair text-3xl md:text-5xl font-medium text-foreground leading-tight">
            Discover music intelligently.
          </h2>
          <p className="text-xl md:text-2xl text-muted-foreground font-playfair max-w-2xl mx-auto leading-relaxed">
            Timbre helps you discover music that reflects your unique taste, shaped by your patterns, moods, and preferences.
          </p>
        </div>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <Button 
            size="lg" 
            className="px-8 py-4 text-lg font-playfair bg-primary text-primary-foreground hover:bg-primary/90 transition-all duration-300 hover:scale-105 flex items-center gap-2"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
            </svg>
            Connect with Spotify
          </Button>
          <Button 
            variant="outline" 
            size="lg" 
            className="px-8 py-4 text-lg font-playfair border-primary text-primary hover:bg-primary hover:text-primary-foreground transition-all duration-300 hover:scale-105"
          >
            Explore as Guest
          </Button>
        </div>

        {/* Subtle tagline */}
        <div className="mt-16 opacity-60">
          <p className="text-sm font-playfair text-muted-foreground italic">
          /ˈtambər/ · The quality of a sound that distinguishes different types of musical instruments or voices
          </p>
        </div>
      </div>

      {/* Gradient overlay for depth */}
      <div className="absolute inset-0 bg-gradient-to-b from-background/50 via-transparent to-background/80 pointer-events-none" />
    </section>
  );
};