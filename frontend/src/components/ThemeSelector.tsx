import { useState, useEffect } from 'react';
import { Monitor, Sun, Moon } from 'lucide-react';

type Theme = 'light' | 'dark' | 'system';

interface ThemeSelectorProps {
  className?: string;
}

export const ThemeSelector = ({ className = '' }: ThemeSelectorProps) => {
  const [theme, setTheme] = useState<Theme>('system');

  useEffect(() => {
    // Get initial theme from localStorage or system preference
    const savedTheme = localStorage.getItem('theme') as Theme;
    if (savedTheme && ['light', 'dark', 'system'].includes(savedTheme)) {
      setTheme(savedTheme);
    }
  }, []);

  const handleThemeChange = (newTheme: Theme) => {
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Apply theme to document
    const root = document.documentElement;
    
    if (newTheme === 'system') {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      root.classList.remove('light', 'dark');
      root.classList.add(systemTheme);
    } else {
      root.classList.remove('light', 'dark');
      root.classList.add(newTheme);
    }
  };

  return (
    <div className={`flex items-center justify-start ${className}`}>
      <div className="bg-muted/30 border border-border rounded-lg p-1 flex items-center">
        <button
          onClick={() => handleThemeChange('light')}
          className={`p-2 rounded-md transition-all duration-200 ${
            theme === 'light' 
              ? 'bg-background text-foreground shadow-sm' 
              : 'text-muted-foreground hover:text-foreground'
          }`}
          title="Light mode"
        >
          <Sun className="w-4 h-4" />
        </button>
        
        <button
          onClick={() => handleThemeChange('system')}
          className={`p-2 rounded-md transition-all duration-200 ${
            theme === 'system' 
              ? 'bg-background text-foreground shadow-sm' 
              : 'text-muted-foreground hover:text-foreground'
          }`}
          title="System theme"
        >
          <Monitor className="w-4 h-4" />
        </button>
        
        <button
          onClick={() => handleThemeChange('dark')}
          className={`p-2 rounded-md transition-all duration-200 ${
            theme === 'dark' 
              ? 'bg-background text-foreground shadow-sm' 
              : 'text-muted-foreground hover:text-foreground'
          }`}
          title="Dark mode"
        >
          <Moon className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}; 