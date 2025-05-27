import React from 'react';
import { SoundWave } from './SoundWave';

export function HeroSection() {
  return <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <SoundWave />
      <div className="relative z-10 text-center px-6 max-w-4xl mx-auto">
        <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-[#00C6FF] to-[#0099FF] bg-clip-text text-transparent animate-fade-in-up">
          Tensoe
        </h1>
        <h2 className="text-2xl md:text-4xl font-semibold mb-4 text-gray-100 animate-fade-in-up delay-100">
          Discover music intelligently
        </h2>
        <p className="text-lg md:text-xl text-gray-300 mb-8 animate-fade-in-up delay-200">
          Shaped by machine learning. Tuned for your taste.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center animate-fade-in-up delay-300">
          <button className="flex items-center gap-3 px-8 py-3 bg-gradient-to-r from-[#00C6FF] to-[#0099FF] rounded-full font-semibold hover:from-[#0099FF] hover:to-[#00C6FF] transition-all duration-300 transform hover:scale-105">
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor" style={{ minWidth: '20px' }}>
              <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.299.421-1.02.599-1.559.3z" />
            </svg>
            Connect Spotify
          </button>
          <button className="px-8 py-3 border border-gray-600 rounded-full font-semibold hover:border-gray-400 transition-all duration-300 animate-fade-in-up delay-400" onClick={() => {
            document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' });
          }}>
            Learn More
          </button>
        </div>
      </div>
    </section>;
}