'use client';

import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Separator } from '../../components/ui/separator';
import { ArrowRight } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { useSupabase } from '@/components/SupabaseProvider';  

const Auth = () => {
  const [formData, setFormData] = useState({ email: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const searchParams = useSearchParams();
  const { signInWithOAuth } = useSupabase();

  useEffect(() => {
    const errorParam = searchParams.get('error');
    
    if (errorParam) {
      const errorMessages = {
        spotify_denied: 'Spotify connection was denied. Please try again.',
        invalid_request: 'Invalid request. Please try again.',
        invalid_state: 'Security validation failed. Please try again.',
        configuration_error: 'Service configuration error. Please contact support.',
        not_authenticated: 'Please sign in first before connecting services.',
        database_error: 'Database error occurred. Please try again.',
        spotify_error: 'Spotify connection failed. Please try again.',
        user_creation_failed: 'Failed to create user account. Please try again.',
        signin_failed: 'Failed to sign in after OAuth. Please try again.',
        session_not_established: 'Session not established. Please try again.'
      };
      setError(errorMessages[errorParam as keyof typeof errorMessages] || 'An error occurred. Please try again.');
    }
  }, [searchParams]);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSpotifyOAuth = async () => {
    setLoading(true);
    setError(null);
    try {
      await signInWithOAuth('spotify', {
        options: {
          redirectTo: `${window.location.origin}/`
        }
      });
    } catch (error) {
      console.error('Error connecting Spotify:', error);
      setLoading(false);
      setError('Failed to connect to Spotify. Please try again.');
    }
  };

  const handleContinue = () => {
    // Handle form submission
    console.log('Form data:', formData);
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="font-playfair text-4xl md:text-5xl font-bold text-primary tracking-tight">
            Timbrality
          </h1>
          <h2 className="font-playfair text-2xl font-medium text-foreground">
            Connect Your Music Profile
          </h2>
          <p className="text-muted-foreground font-playfair leading-relaxed">
            Sign in with Spotify to discover your unique musical DNA
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 text-center">
            <p className="text-destructive text-sm font-medium">{error}</p>
          </div>
        )}

        {/* Spotify OAuth Card */}
        <div
          className="bg-card border border-border rounded-xl p-3 hover:bg-accent/30 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={!loading ? handleSpotifyOAuth : undefined}
        >
          <div className="flex items-center justify-center space-x-4">
            <div className="w-6 h-6 flex items-center justify-center">
              {loading ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-500"></div>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" className="text-green-500">
                  <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                </svg>
              )}
            </div>
            <span className="text-foreground font-medium">
              {loading ? 'Connecting...' : 'Continue with Spotify'}
            </span>
          </div>
        </div>

        {/* OR Separator */}
        <div className="relative">
          <Separator className="bg-border" />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="bg-background px-4 text-sm text-muted-foreground font-medium">OR</span>
          </div>
        </div>

        {/* Email Input */}
        <div className="space-y-4">
          <Input
            type="email"
            placeholder="frank.ocean@example.com"
            value={formData.email}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleInputChange('email', e.target.value)}
            className="bg-card border-border text-foreground placeholder:text-muted-foreground rounded-xl p-3"
          />

          {/* Continue Button */}
          <Button
            onClick={handleContinue}
            size="lg"
            className="w-full px-8 py-4 text-lg font-playfair bg-primary text-primary-foreground hover:bg-primary/90 transition-all duration-300 hover:scale-105"
          >
            <span>Continue</span>
            <ArrowRight className="h-5 w-5 ml-2" />
          </Button>
        </div>

        {/* Legal Disclaimer */}
        <p className="text-xs text-muted-foreground text-center leading-relaxed font-playfair">
          By continuing, you agree to our{' '}
          <a href="/terms" className="text-blue-500 hover:underline">
            Terms of Service
          </a>{' '}
          and{' '}
          <a href="/privacy" className="text-blue-500 hover:underline">
            Privacy Policy
          </a>
        </p>
      </div>
    </div>
  );
};

export default Auth; 
 