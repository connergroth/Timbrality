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

  return (
    <header className="w-full px-8 max-w-7xl mx-auto relative -mb-16 z-50">
      <div className="flex items-center justify-between relative bg-background/ backdrop-blur-sm py-4">
            {/* Left: Logo (moved in from edge) */}
            <div className="flex items-center space-x-1 ml-4">
              <div className="w-10 h-10 flex items-center justify-center">
                <img 
                  src="/soundwhite.png" 
                  alt="Timbrality" 
                  className="w-7 h-7 object-contain"
                />
              </div>
              <span className="font-playfair text-2xl font-bold text-primary">Timbrality</span>
            </div>

            {/* Center: Navigation Links */}
            <div className="hidden md:flex items-center space-x-8">
              <a 
                href="#how-it-works" 
                className="text-muted-foreground hover:text-foreground transition-colors duration-200 text-base font-medium relative group"
              >
                How it works
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary transition-all duration-200 group-hover:w-full"></span>
              </a>
              <a 
                href="#agent-demo" 
                className="text-muted-foreground hover:text-foreground transition-colors duration-200 text-base font-medium relative group"
              >
                AI Curator
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary transition-all duration-200 group-hover:w-full"></span>
              </a>
              <a 
                href="#your-dna" 
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
        <div className="md:hidden absolute top-full left-0 right-0 mt-4 bg-background/95 backdrop-blur-xl border border-border/50 rounded-2xl shadow-lg">
          <div className="px-6 py-4 space-y-3">
            <a 
              href="#how-it-works" 
              className="block text-muted-foreground hover:text-foreground transition-colors duration-200 text-sm font-medium text-center"
              onClick={() => setIsMenuOpen(false)}
            >
              How it works
            </a>
            <a 
              href="#agent-demo" 
              className="block text-muted-foreground hover:text-foreground transition-colors duration-200 text-sm font-medium text-center"
              onClick={() => setIsMenuOpen(false)}
            >
              AI Curator
            </a>
            <a 
              href="#your-dna" 
              className="block text-muted-foreground hover:text-foreground transition-colors duration-200 text-sm font-medium text-center"
              onClick={() => setIsMenuOpen(false)}
            >
              Your DNA
            </a>
            <div className="pt-2 space-y-2">
              {/* Mobile Dark Mode Toggle */}
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleTheme}
                className="w-full text-muted-foreground hover:text-foreground hover:bg-background/50 flex items-center justify-center space-x-1"
              >
                {theme === 'light' ? (
                  <>
                    <Moon className="w-4 h-4" />
                    <span>Dark Mode</span>
                  </>
                ) : (
                  <>
                    <Sun className="w-4 h-4" />
                    <span>Light Mode</span>
                  </>
                )}
              </Button>
              
              <Button 
                variant="ghost" 
                size="sm"
                className="w-full text-muted-foreground hover:text-foreground hover:bg-background/50"
                asChild
              >
                <a href="/auth">Log in</a>
              </Button>
              <Button 
                size="sm"
                className="w-full bg-primary hover:bg-primary/90 text-primary-foreground rounded-full"
                asChild
              >
                <a href="/auth">Get Started</a>
              </Button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};