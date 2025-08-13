'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { Sparkles } from 'lucide-react';

export function MoodInput() {
  const [moodText, setMoodText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleAskTimbrality = async () => {
    if (!moodText.trim()) return;
    
    setIsLoading(true);
    try {
      // This would call your backend API to get mood-based recommendations
      console.log('Asking Timbrality for:', moodText);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // Handle response here
      console.log('Got recommendations for mood:', moodText);
    } catch (error) {
      console.error('Error getting mood recommendations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAskTimbrality();
    }
  };

  return (
    <Card className="bg-gradient-to-r from-primary/5 to-accent/5 border-primary/20">
      <CardContent className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="h-5 w-5 text-primary" />     <h3 className="text-lg font-playfair font-semibold text-foreground">
            💬 What are you in the mood for?
          </h3>
        </div>
        
        <div className="flex gap-3">
          <Input
            type="text"       placeholder="I want something moody but upbeat..."
            value={moodText}
            onChange={(e) => setMoodText(e.target.value)}
            onKeyPress={handleKeyPress}
            className="flex-1 text-base"
            disabled={isLoading}
          />
          <Button
            onClick={handleAskTimbrality}
            disabled={!moodText.trim() || isLoading}
            className="px-6 font-medium"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 order-white"></div>
                <span>Asking...</span>
              </div>
            ) : (
              <span>Ask Timbrality →</span>
            )}
          </Button>
        </div>
        
        <p className="text-sm text-muted-foreground mt-3">
          Describe your mood, energy level, or what you're looking for. Timbrality will find the perfect music for you.
        </p>
      </CardContent>
    </Card>
  );
} 