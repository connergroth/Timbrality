'use client'

import { Moon, Sun } from 'lucide-react'
import { useTheme } from 'next-themes'
import { Button } from '@/components/ui/button'

export function Footer() {
  const { theme, setTheme } = useTheme()
  return (
    <footer className="border-t border-border bg-card/50 mt-auto">
      <div className="container mx-auto px-6 py-10">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-6 md:space-y-0">
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 bg-primary rounded flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-sm">T</span>
            </div>
            <span className="font-playfair font-semibold text-primary">Timbrality</span>
          </div>
          
          <div className="flex items-center space-x-8 text-sm text-muted-foreground">
            <a href="/terms" className="hover:text-foreground transition-colors font-inter">
              Terms of Service
            </a>
            <a href="/privacy" className="hover:text-foreground transition-colors font-inter">
              Privacy Policy
            </a>
            <a href="/about" className="hover:text-foreground transition-colors font-inter">
              About
            </a>
            <a href="/contact" className="hover:text-foreground transition-colors font-inter">
              Contact
            </a>
            
            {/* Dark Mode Toggle */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="h-8 w-8 p-0"
            >
              <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
          
          <div className="text-sm text-muted-foreground font-inter">
            © {new Date().getFullYear()} Timbrality. All rights reserved.
          </div>
        </div>
      </div>
    </footer>
  );
} 