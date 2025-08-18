'use client';

import { Button } from "@/components/ui/button";
import { Menu, X, Sun, Moon } from "lucide-react";
import { useState, useEffect } from "react";

export const LandingNavbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');

  useEffect(() => {
    // Simple theme detection
    const isDark = document.documentElement.classList.contains('dark');
    setTheme(isDark ? 'dark' : 'light');
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    document.documentElement.classList.toggle('dark');
  };

  const handleLogoClick = () => {
    if (window.location.pathname === '/') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      window.location.href = '/';
    }
  };

  return (
    <header className="w-full px-8 max-w-7xl mx-auto relative -mb-16 z-50">
      <div className="flex items-center justify-between relative bg-background/ backdrop-blur-sm py-4">
            {/* Left: Logo (moved in from edge) */}
            <button 
              onClick={handleLogoClick}
              className="flex items-center space-x-1 hover:opacity-80 transition-opacity cursor-pointer"
            >
              <div className="w-10 h-10 flex items-center justify-center">
                <img 
                  src="/soundwhite.png" 
                  alt="Timbrality" 
                  className="w-7 h-7 object-contain"
                />
              </div>
              <span className="font-playfair text-2xl font-bold text-primary">Timbrality</span>
            </button>

            {/* Center: Navigation Links */}
            <div className="hidden md:flex items-center space-x-8">
              <a 
                href="/#how-it-works" 
                className="text-muted-foreground hover:text-foreground transition-colors duration-200 text-base font-medium relative group"
              >
                How it works
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary transition-all duration-200 group-hover:w-full"></span>
              </a>
              <a 
                href="/#agent-demo" 
                className="text-muted-foreground hover:text-foreground transition-colors duration-200 text-base font-medium relative group"
              >
                AI Curator
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary transition-all duration-200 group-hover:w-full"></span>
              </a>
              <a 
                href="/#your-dna" 
                className="text-muted-foreground hover:text-foreground transition-colors duration-200 text-base font-medium relative group"
              >
                Your DNA
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary transition-all duration-200 group-hover:w-full"></span>
              </a>
            </div>

            {/* Right: Buttons (moved in from edge) */}
            <div className="hidden md:flex items-center space-x-3 mr-4">
              <Button 
                variant="outline" 
                size="sm"
                className="px-4 py-2 text-sm font-inter border-primary text-primary hover:bg-primary hover:text-primary-foreground transition-colors duration-200"
                asChild
              >
                <a href="/auth">Log in</a>  
              </Button>
              <Button 
                size="sm"
                className="px-4 py-2 text-sm font-inter bg-white text-black hover:bg-gray-100 transition-colors duration-200"
                asChild
              >
                <a href="/auth">Get Started</a>
              </Button>
            </div>

            {/* Mobile Menu Button */}
            <button
              className="md:hidden text-muted-foreground hover:text-foreground p-1"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
      </div>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <div className="md:hidden fixed inset-0 bg-neutral-900 z-50">
          {/* Header */}
          <div className="flex items-center justify-between p-6">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 flex items-center justify-center">
                <img 
                  src="/soundwhite.png" 
                  alt="Timbrality" 
                  className="w-6 h-6 object-contain"
                />
              </div>
              <span className="font-playfair text-2xl font-bold text-white">Timbrality</span>
            </div>
            <button
              onClick={() => setIsMenuOpen(false)}
              className="text-white p-2"
            >
              <X size={24} />
            </button>
          </div>

          {/* Navigation Links - Centered */}
          <div className="flex-1 flex flex-col items-center justify-center space-y-8 mt-20">
            <a 
              href="#how-it-works" 
              className="text-5xl md:text-6xl font-inter font-medium text-white hover:text-neutral-300 transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              How it works
            </a>
            <a 
              href="#agent-demo" 
              className="text-5xl md:text-6xl font-inter font-medium text-white hover:text-neutral-300 transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              AI Curator
            </a>
            <a 
              href="#your-dna" 
              className="text-5xl md:text-6xl font-inter font-medium text-white hover:text-neutral-300 transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              Your DNA
            </a>
          </div>

          {/* Bottom CTA Buttons */}
          <div className="p-6 space-y-4">
            <Button 
              variant="outline" 
              size="lg"
              className="w-full py-4 text-lg font-inter border-white text-white hover:bg-white hover:text-neutral-900 transition-colors rounded-xl"
              asChild
            >
              <a href="/auth">Log in</a>
            </Button>
            <Button 
              size="lg"
              className="w-full py-4 text-lg font-inter bg-white text-neutral-900 hover:bg-gray-100 transition-colors rounded-xl"
              asChild
            >
              <a href="/auth">Get Started</a>
            </Button>
          </div>
        </div>
      )}
    </header>
  );
};