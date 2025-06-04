
import { useState, useEffect } from 'react';
import { Github } from 'lucide-react';

const Header = () => {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled
          ? 'bg-tensoe-navy/80 backdrop-blur-md border-b border-tensoe-blue/20'
          : 'bg-transparent'
      }`}
    >
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="text-2xl font-bold gradient-text">
            Tensoe
          </div>
          
          <nav className="hidden md:flex items-center space-x-8">
            <a href="#how-it-works" className="text-tensoe-blue-light hover:text-tensoe-blue transition-colors">
              How It Works
            </a>
            <a href="#connect" className="text-tensoe-blue-light hover:text-tensoe-blue transition-colors">
              Connect
            </a>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-tensoe-blue-light hover:text-tensoe-blue transition-colors">
              <Github size={20} />
            </a>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;
