'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Heart, ThumbsUp, ThumbsDown, Plus, Info } from 'lucide-react';

interface Track {
  id: number;
  title: string;
  artist: string;
  album: string;
  cover: string;
  why: string;
}

interface TrackGridProps {
  tracks: Track[];
}

export function TrackGrid({ tracks }: TrackGridProps) {
  const [likedTracks, setLikedTracks] = useState<Set<number>>(new Set());
  const [dislikedTracks, setDislikedTracks] = useState<Set<number>>(new Set());

  const handleLike = (trackId: number) => {
    setLikedTracks(prev => {
      const newSet = new Set(prev);
      newSet.add(trackId);
      return newSet;
    });
    setDislikedTracks(prev => {
      const newSet = new Set(prev);
      newSet.delete(trackId);
      return newSet;
    });
  };

  const handleDislike = (trackId: number) => {
    setDislikedTracks(prev => {
      const newSet = new Set(prev);
      newSet.add(trackId);
      return newSet;
    });
    setLikedTracks(prev => {
      const newSet = new Set(prev);
      newSet.delete(trackId);
      return newSet;
    });
  };

  const handleAddToPlaylist = (trackId: number) => {
    console.log('Add track to playlist:', trackId);
  };

  if (tracks.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">No recommendations yet. Connect your music profiles to get started!</p>
      </div>
    );
  }

  return (
    <TooltipProvider>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {tracks.map((track) => (
          <Card key={track.id} className="overflow-hidden hover:shadow-lg transition-shadow">
            <CardContent className="p-0">
              {/* Album Cover */}
              <div className="relative aspect-square bg-muted">
                <img
                  src={track.cover}
                  alt={`${track.album} cover`}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.currentTarget.src = 'https://via.placeholder.com/300x300/FFFFFF?text=Album';
                  }}
                />
              </div>

              {/* Track Info */}
              <div className="p-4">
                <h3 className="font-semibold text-foreground truncate">{track.title}</h3>
                <p className="text-sm text-muted-foreground truncate">{track.artist}</p>
                <p className="text-xs text-muted-foreground truncate mb-3">{track.album}</p>

                {/* Action Buttons */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {/* Why? Button */}
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0"
                        >
                          <Info className="h-4 w-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-sm">{track.why}</p>
                      </TooltipContent>
                    </Tooltip>

                    {/* Like/Dislike Buttons */}
                    <Button
                      variant="ghost"
                      size="sm"
                      className={`h-8 w-8 p-0 ${
                        likedTracks.has(track.id) ? 'text-green-50' : 'text-muted-foreground'
                      }`}
                      onClick={() => handleLike(track.id)}
                    >
                      <ThumbsUp className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className={`h-8 w-8 p-0 ${
                        dislikedTracks.has(track.id) ? 'text-red-50' : 'text-muted-foreground'
                      }`}
                      onClick={() => handleDislike(track.id)}
                    >
                      <ThumbsDown className="h-4 w-4" />
                    </Button>
                  </div>

                  {/* Add to Playlist Button */}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 text-muted-foreground hover:text-foreground"
                    onClick={() => handleAddToPlaylist(track.id)}
                  >
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </TooltipProvider>
  );
} 