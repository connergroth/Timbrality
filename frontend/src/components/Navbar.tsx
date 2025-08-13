import { Button } from "@/components/ui/button";
import { Menu, X, Sun, Moon } from "lucide-react";
import { useState } from "react";
import { useTheme } from "@/contexts/ThemeContext";
import { Link } from "react-router-dom";

export const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();

  return (
    <nav className="fixed top-8 left-1/2 transform -translate-x-1/2 z-50">
      <div className="bg-background/90 backdrop-blur-xl border border-border/50 rounded-full px-8 py-4 shadow-lg">
        <div className="flex items-center justify-between space-x-8">
          {/* Logo */}
          <div className="font-playfair text-xl font-bold text-primary">
            Timbrality
          </div>
          
          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-6">
            <a 
              href="#how-it-works" 
              className="text-muted-foreground hover:text-foreground transition-colors duration-200 text-sm font-medium relative group"
            >
              How it works
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary transition-all duration-200 group-hover:w-full"></span>
            </a>
            <a 
              href="#agent-demo" 
              className="text-muted-foreground hover:text-foreground transition-colors duration-200 text-sm font-medium relative group"
            >
              AI Curator
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary transition-all duration-200 group-hover:w-full"></span>
            </a>
            <a 
              href="#your-dna" 
              className="text-muted-foreground hover:text-foreground transition-colors duration-200 text-sm font-medium relative group"
            >
              Your DNA
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary transition-all duration-200 group-hover:w-full"></span>
            </a>
          </div>

          {/* Desktop Buttons */}
          <div className="hidden md:flex items-center space-x-3">
            <Button 
              variant="ghost" 
              size="sm"
              className="text-muted-foreground hover:text-foreground hover:bg-background/50"
              asChild
            >
              <a href="/auth">Log in</a>  
            </Button>
            <Button 
              size="sm"
              className="bg-primary hover:bg-primary/90 text-primary-foreground rounded-full px-6"
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
                  className="w-full text-muted-foreground hover:text-foreground hover:bg-background/50 flex items-center justify-center space-x-2"
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
      </div>
    </nav>
  );
};