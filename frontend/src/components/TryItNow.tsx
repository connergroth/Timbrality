import React from 'react';
import { useScrollAnimation } from '../hooks/useScrollAnimation';

export function TryItNow() {
  const sectionRef = useScrollAnimation();
  
  return <section ref={sectionRef} className="pt-16 pb-8 px-6">
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-6 text-gray-100 scroll-fade-up">
          Ready to discover your next favorite song?
        </h2>
        <p className="text-lg text-gray-300 mb-12 max-w-2xl mx-auto scroll-fade-up scroll-delay-100">
          Choose how you want to get personalized recommendations
        </p>
        <div className="flex flex-col gap-6">
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center scroll-fade-up scroll-delay-200">
            <button className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-[#00C6FF] to-[#0099FF] rounded-full font-semibold hover:from-[#0099FF] hover:to-[#00C6FF] transition-all duration-300 transform hover:scale-105">
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor" style={{ minWidth: '20px' }}>
                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.299.421-1.02.599-1.559.3z" />
              </svg>
              Connect Spotify
            </button>
            <button className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-[#00C6FF] to-[#0099FF] rounded-full font-semibold hover:from-[#0099FF] hover:to-[#00C6FF] transition-all duration-300 transform hover:scale-105">
              <img src="/logos/lastfm.png" alt="Last.fm" className="w-6 h-6" />
              Connect Last.fm
            </button>
            <button className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-[#00C6FF] to-[#0099FF] rounded-full font-semibold hover:from-[#0099FF] hover:to-[#00C6FF] transition-all duration-300 transform hover:scale-105">
              <img src="/logos/aoty.png" alt="AOTY" className="w-7 h-7" />
              Enter AOTY Username
            </button>
          </div>
          <div className="flex justify-center items-center scroll-fade-up scroll-delay-300">
            <button className="flex items-center gap-3 px-8 py-4 border border-gray-600 hover:border-gray-400 rounded-full font-semibold transition-all duration-300 transform hover:scale-105">
              <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4M10 17l5-5-5-5M13.8 12H3"/>
              </svg>
              Explore as Guest
            </button>
          </div>
        </div>
        <p className="text-sm text-gray-400 mt-6 scroll-fade-up scroll-delay-400">
          Your data is secure and never shared. We only use it to improve your
          recommendations.
        </p>
      </div>
    </section>;
}