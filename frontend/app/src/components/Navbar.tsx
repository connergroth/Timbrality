'use client';

import { User } from '@supabase/supabase-js';
import { BrainCircuit } from 'lucide-react';
import { useRouter } from 'next/navigation';

interface NavbarProps {
  user: User;
  onSignOut: () => Promise<void>;
  onToggleAlgorithmSidebar?: () => void;
}

export function Navbar({ user, onSignOut, onToggleAlgorithmSidebar }: NavbarProps) {
  const router = useRouter();

  const handleLogoClick = () => {
    router.push('/');
  };

  return (
    <nav className="bg-background/80 backdrop-blur-sm sticky top-0 z-40">
      <div className="w-full px-6 py-4 flex items-center justify-between">
        {/* Left: Logo */}
        <button 
          onClick={handleLogoClick}
          className="flex items-center space-x-2 hover:opacity-80 transition-opacity cursor-pointer"
        >
          <div className="w-8 h-8 flex items-center justify-center">
            <img 
              src="/soundwhite.png" 
              alt="Timbrality" 
              className="w-5 h-5 object-contain"
            />
          </div>
          <span className="font-playfair text-xl font-bold text-primary">Timbrality</span>
        </button>

        {/* Right: Brain icon only */}
        <div className="flex items-center space-x-4">
          <button
            onClick={onToggleAlgorithmSidebar}
            className="flex items-center justify-center h-10 w-10 text-muted-foreground hover:text-foreground transition-colors focus:outline-none"
            title="Behind the Algorithm"
            type="button"
          >
            <BrainCircuit className="h-5 w-5" />
          </button>
        </div>
      </div>
    </nav>
  );
} 