import React from 'react';
import { HeroSection } from './components/HeroSection';
import { HowItWorks } from './components/HowItWorks';
import { TryItNow } from './components/TryItNow';
import { Footer } from './components/Footer';
export function App() {
  return <div className="min-h-screen bg-gray-900 text-white">
      <HeroSection />
      <HowItWorks />
      <TryItNow />
      <Footer />
    </div>;
}