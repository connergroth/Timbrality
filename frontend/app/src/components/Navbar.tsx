'use client';

import { User } from '@supabase/supabase-js';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { LogOut, Settings, User as UserIcon, Brain, PanelLeft, RefreshCw } from 'lucide-react';
import { forceSpotifyReauth, hasRequiredSpotifyScopes } from '@/lib/spotify-auth';

interface NavbarProps {
  user: User;
  onOpenAlgorithmSidebar: () => void;
  onOpenNavigationSidebar: () => void;
  onSignOut: () => Promise<void>;
}

export function Navbar({ user, onOpenAlgorithmSidebar, onOpenNavigationSidebar, onSignOut }: NavbarProps) {
  const handleSignOut = async () => {
    try {
      await onSignOut();
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const handleSpotifyReauth = async () => {
    try {
      await forceSpotifyReauth();
    } catch (error) {
      console.error('Error re-authenticating Spotify:', error);
    }
  };

  return (
    <nav className="bg-background/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="w-full px-6 py-4 flex items-center justify-between">
        {/* Left corner - Sidebar Button and Logo */}
        <div className="flex items-center space-x-4">
          <button
            onClick={onOpenNavigationSidebar}
            className="flex items-center justify-center h-8 w-8 text-muted-foreground hover:text-foreground transition-colors focus:outline-none"
            title="Navigation Menu"
            type="button"
          >
            <PanelLeft className="h-5 w-5" />
          </button>
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 flex items-center justify-center">
              <img 
                src="/soundwhite.png" 
                alt="Timbre" 
                className="w-5 h-5 object-contain"
              />
            </div>
            <span className="font-playfair text-xl font-bold text-primary">Timbre</span>
          </div>
        </div>

        {/* Right corner - Brain Icon and User Profile */}
        <div className="flex items-center space-x-4">
          {/* Brain Icon Button with Circle */}
          <button
            onClick={onOpenAlgorithmSidebar}
            className="flex items-center justify-center h-10 w-10 rounded-full bg-muted/50 hover:bg-muted transition-colors focus:outline-none"
            title="Behind the Algorithm"
            type="button"
          >
            <Brain className="h-5 w-5 text-muted-foreground" />
          </button>
          
          {/* User Profile */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="relative h-8 w-8 p-0">
                <Avatar className="h-8 w-8">
                  <AvatarImage src={user.user_metadata?.avatar_url} alt={user.email || 'User'} />
                  <AvatarFallback>
                    {user.email?.charAt(0).toUpperCase() || 'U'}
                  </AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56 align-end">
              <div className="flex items-center justify-start gap-2 p-2">
                <div className="flex flex-col space-y-1">
                  {user.user_metadata?.full_name && (
                    <p className="font-inter font-medium text-sm">{user.user_metadata.full_name}</p>
                  )}
                  {user.email && (
                    <p className="w-[200] truncate text-xs text-muted-foreground font-inter">
                      {user.email}
                    </p>
                  )}
                </div>
              </div>
              <DropdownMenuItem>
                <UserIcon className="mr-2 h-4 w-4"/>
                <span className="font-inter text-sm">Profile</span>
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="mr-2 h-4 w-4"/>
                <span className="font-inter text-sm">Settings</span>
              </DropdownMenuItem>
              {!hasRequiredSpotifyScopes(user) && (
                <DropdownMenuItem onClick={handleSpotifyReauth}>
                  <RefreshCw className="mr-2 h-4 w-4"/>
                  <span className="font-inter text-sm">Re-authenticate Spotify</span>
                </DropdownMenuItem>
              )}
              <DropdownMenuItem onClick={handleSignOut}>
                <LogOut className="mr-2 h-4 w-4"/>
                <span className="font-inter text-sm">Log out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </nav>
  );
} 