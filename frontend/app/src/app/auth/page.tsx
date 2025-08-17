'use client';

import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Separator } from '../../components/ui/separator';
import { ArrowRight, Sparkles } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { useSupabase } from '@/components/SupabaseProvider';  
import Image from 'next/image';

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
    <div className="min-h-screen bg-neutral-900 relative flex items-center justify-center">
      {/* Background overlay */}
      <div className="fixed inset-0 bg-neutral-900"></div>
      
      <div className="relative z-10 px-4 md:px-8 lg:px-12 max-w-2xl mx-auto">
        {/* Main Auth Card */}
        <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-8 md:p-12 shadow-2xl">
          {/* Header */}
          <div className="text-center space-y-6 mb-8">
            <div className="flex items-center justify-center gap-3 mb-6">
                <Image
                  src="/soundwhite.png"
                  alt="Timbrality Logo"
                  width={100}
                  height={100}
                  className="w-16 h-16"
                />
            </div>
            <h1 className="font-playfair text-5xl md:text-6xl font-bold text-white tracking-tight">
              Timbrality
            </h1>
            <h2 className="font-playfair text-2xl md:text-3xl font-medium text-white mb-4">
              Connect Your Music Profile
            </h2>
            <p className="text-neutral-300 font-inter text-lg leading-relaxed max-w-lg mx-auto">
              Sign in with Spotify to discover your unique musical DNA and get personalized recommendations
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-4 text-center backdrop-blur-sm mb-6">
              <p className="text-red-400 text-sm font-medium font-inter">{error}</p>
            </div>
          )}

          {/* Spotify OAuth Card */}
          <div
            className="bg-neutral-700/40 backdrop-blur-xl border border-neutral-600/30 rounded-2xl p-2 hover:bg-neutral-700/60 hover:border-neutral-600/50 transition-all duration-300 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed shadow-xl mb-6"
            onClick={!loading ? handleSpotifyOAuth : undefined}
          >
            <div className="flex items-center justify-center space-x-4">
              <div className="w-8 h-8 flex items-center justify-center">
                {loading ? (
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-green-500"></div>
                ) : (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" className="text-green-500">
                    <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                  </svg>
                )}
              </div>
              <span className="text-white font-semibold font-inter text-lg">
                {loading ? 'Connecting...' : 'Continue with Spotify'}
              </span>
            </div>
          </div>

          {/* OR Separator */}
          <div className="relative mb-6">
            <Separator className="bg-neutral-600" />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="bg-neutral-800 px-4 text-sm text-neutral-400 font-medium font-inter">OR</span>
            </div>
          </div>

          {/* Email Input */}
          <div className="space-y-6">
            <Input
              type="email"
              placeholder="frank.ocean@example.com"
              value={formData.email}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleInputChange('email', e.target.value)}
              className="bg-neutral-700/40 backdrop-blur-xl border-neutral-600/30 text-white placeholder:text-neutral-400 rounded-2xl p-4 text-lg focus:border-neutral-500/50 focus:ring-neutral-500/50 transition-all duration-300 font-inter"
            />

            {/* Continue Button */}
            <Button
              onClick={handleContinue}
              size="lg"
              className="w-full px-8 py-4 text-lg font-inter font-semibold bg-white text-neutral-900 hover:bg-neutral-100 transition-all duration-300 hover:scale-105 backdrop-blur-sm shadow-xl rounded-2xl"
            >
              <span>Continue</span>
              <ArrowRight className="h-5 w-5 ml-2" />
            </Button>
          </div>

          {/* Legal Disclaimer */}
          <p className="text-sm text-neutral-400 text-center leading-relaxed font-inter mt-8">
            By continuing, you agree to our{' '}
            <a href="/terms" className="text-blue-400 hover:text-blue-300 hover:underline transition-colors">
              Terms of Service
            </a>{' '}
            and{' '}
            <a href="/privacy" className="text-blue-400 hover:text-blue-300 hover:underline transition-colors">
              Privacy Policy
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Auth; 
 