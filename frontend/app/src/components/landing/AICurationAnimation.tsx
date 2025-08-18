import { useState, useEffect } from 'react';
import { Globe, Clock, Calendar, Star, Info, Plus, User } from 'lucide-react';
import type { Track } from '@/lib/agent';

interface Message {
  id: string;
  type: 'user' | 'agent';
  content: string;
  isTyping?: boolean;
  track?: Track & { artist_image_url?: string | null };
}

interface Conversation {
  id: string;
  messages: Message[];
}

export const AICurationAnimation = () => {
  const [currentConversationIndex, setCurrentConversationIndex] = useState(0);
  const [visibleMessages, setVisibleMessages] = useState<Message[]>([]);
  const [isThinking, setIsThinking] = useState(false);

  const [conversations, setConversations] = useState<Conversation[]>([
    {
      id: '1',
      messages: [
        { id: '1', type: 'user', content: 'I want a song for a late night drive' },
        { 
          id: '2', 
          type: 'agent', 
          content: 'Perfect for late night vibes! This track captures that dreamy, contemplative mood perfectly.',
          track: {
            id: '1',
            name: 'MY EYES',
            artist: 'Travis Scott',
            album: 'UTOPIA',
            artwork_url: undefined,
            spotify_url: 'https://open.spotify.com/track/4kjI1gwQZRKNDkw1nI475M',
            source: 'agent',
            duration_ms: 175000,
            popularity: 85,
            release_date: '2018-06-01',
            rating: 92,
            similarity: 0.94
          }
        }
      ]
    },
    {
      id: '2', 
      messages: [
        { id: '1', type: 'user', content: 'Something melancholic but beautiful' },
        { 
          id: '2', 
          type: 'agent', 
          content: 'Here\'s a hauntingly beautiful track that perfectly captures that bittersweet feeling.',
          track: {
            id: '2',
            name: 'Let Down',
            artist: 'Radiohead',
            album: 'OK Computer',
            artwork_url: undefined,
            spotify_url: 'https://open.spotify.com/track/4Oun2ylbjFKMPTiaSbbCih',
            source: 'agent',
            duration_ms: 299000,
            popularity: 78,
            release_date: '1997-06-16',
            rating: 97,
            similarity: 0.91
          }
        }
      ]
    },
    {
      id: '3',
      messages: [
        { id: '1', type: 'user', content: 'I need focus music for studying' },
        { 
          id: '2', 
          type: 'agent', 
          content: 'This atmospheric track creates the perfect ambient soundscape for deep concentration.',
          track: {
            id: '3',
            name: 'City of Tears',
            artist: 'Christopher Larkin',
            album: 'Hollow Knight (Original Soundtrack)',
            artwork_url: undefined,
            spotify_url: 'https://open.spotify.com/track/4s8ieOhpVsYL3VtQH1Hu2O',
            source: 'agent',
            duration_ms: 214000,
            popularity: 65,
            release_date: '2017-02-24',
            rating: 98,
            similarity: 0.88
          }
        }
      ]
    }
  ]);

  const typeMessage = async (message: string, callback: (char: string) => void) => {
    for (let i = 0; i <= message.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 30));
      callback(message.slice(0, i));
    }
  };

  // Function to fetch Spotify track data
  const fetchSpotifyTrackData = async (trackName: string, artistName: string) => {
    try {
      const query = `${trackName} ${artistName}`;
      // Add cache busting for fresh data
      const cacheBuster = Date.now();
      const response = await fetch(`/api/spotify/search-track?q=${encodeURIComponent(query)}&_t=${cacheBuster}`, {
        cache: 'no-store'
      });
      
      if (!response.ok) {
        console.warn(`Failed to fetch Spotify data for ${trackName} by ${artistName}`);
        return null;
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`Error fetching Spotify data for ${trackName}:`, error);
      return null;
    }
  };

  // Function to fetch artist image
  const fetchArtistImage = async (artistName: string) => {
    try {
      // Add cache busting for fresh data
      const cacheBuster = Date.now();
      const response = await fetch(`/api/spotify/search-artist?q=${encodeURIComponent(artistName)}&_t=${cacheBuster}`, {
        cache: 'no-store'
      });
      
      if (!response.ok) {
        return null;
      }
      
      const data = await response.json();
      return data.profile_picture || null;
    } catch (error) {
      console.error(`Error fetching artist image for ${artistName}:`, error);
      return null;
    }
  };

  // Function to enrich conversations with Spotify data
  const enrichConversationsWithSpotifyData = async () => {
    const enrichedConversations = await Promise.all(
      conversations.map(async (conversation) => {
        const enrichedMessages = await Promise.all(
          conversation.messages.map(async (message) => {
            if (message.track) {
              const [spotifyData, artistImage] = await Promise.all([
                fetchSpotifyTrackData(message.track.name, message.track.artist),
                fetchArtistImage(message.track.artist)
              ]);
              
              if (spotifyData) {
                return {
                  ...message,
                  track: {
                    ...message.track,
                    artwork_url: spotifyData.artwork_url,
                    artist_image_url: artistImage,
                    spotify_url: spotifyData.external_urls.spotify,
                    duration_ms: spotifyData.duration_ms,
                    popularity: spotifyData.popularity,
                    release_date: spotifyData.release_date,
                  }
                };
              }
            }
            return message;
          })
        );
        
        return {
          ...conversation,
          messages: enrichedMessages
        };
      })
    );
    
    setConversations(enrichedConversations);
  };

  // Enrich conversations with Spotify data on mount
  useEffect(() => {
    enrichConversationsWithSpotifyData();
  }, []);

  useEffect(() => {
    let isMounted = true;

    const runConversation = async () => {
      if (!isMounted) return;

      const conversation = conversations[currentConversationIndex];
      setVisibleMessages([]);

      // Type user message
      const userMessage = conversation.messages[0];
      let currentUserContent = '';
      
      setVisibleMessages([{ ...userMessage, content: '', isTyping: true }]);
      
      await typeMessage(userMessage.content, (content) => {
        if (!isMounted) return;
        currentUserContent = content;
        setVisibleMessages([{ ...userMessage, content, isTyping: content.length < userMessage.content.length }]);
      });

      if (!isMounted) return;

      // Show thinking
      await new Promise(resolve => setTimeout(resolve, 800));
      if (!isMounted) return;
      
      setIsThinking(true);
      await new Promise(resolve => setTimeout(resolve, 1500));
      if (!isMounted) return;
      
      setIsThinking(false);

      // Type agent response
      const agentMessage = conversation.messages[1];
      let currentAgentContent = '';

      setVisibleMessages(prev => [...prev, { ...agentMessage, content: '', isTyping: true }]);

      await typeMessage(agentMessage.content, (content) => {
        if (!isMounted) return;
        currentAgentContent = content;
        setVisibleMessages(prev => [
          prev[0],
          { ...agentMessage, content, isTyping: content.length < agentMessage.content.length }
        ]);
      });

      if (!isMounted) return;

      // Wait before next conversation
      await new Promise(resolve => setTimeout(resolve, 3000));
      if (!isMounted) return;

      setCurrentConversationIndex((prev) => (prev + 1) % conversations.length);
    };

    runConversation();

    return () => {
      isMounted = false;
    };
  }, [currentConversationIndex]);

  const ThinkingAnimation = () => (
    <div className="flex justify-start mb-4">
      <div className="flex items-center space-x-2">
        <div className="w-6 h-6 flex items-center justify-center flex-shrink-0 mr-2">
          <img 
            src="/soundwhite.png" 
            alt="Timbrality Agent" 
            className="w-6 h-6 object-contain opacity-50"
          />
        </div>
        <div className="relative inline-block">
          <span className="text-base font-inter text-neutral-400">
            {'Thinking'.split('').map((char, i) => (
              <span
                key={i}
                className="inline-block"
                style={{
                  opacity: 0.3,
                  animation: `thinking-wave 1.2s ease-in-out infinite`,
                  animationDelay: `${0.1 + i * 0.08}s`
                }}
              >
                {char}
              </span>
            ))}
          </span>
          <style jsx>{`
            @keyframes thinking-wave {
              0%, 70%, 100% {
                opacity: 0.3;
              }
              35% {
                opacity: 1;
              }
            }
          `}</style>
          <span className="ml-1 text-base font-inter text-neutral-400 animate-bounce">...</span>
        </div>
      </div>
    </div>
  );

  return (
    <div className="w-full h-[500px] flex items-center justify-center">
      <div className="w-full max-w-6xl h-[450px] bg-neutral-800/60 rounded-2xl border border-neutral-600/30 p-6 overflow-hidden">
        <div className="h-full overflow-y-auto space-y-3">
          {visibleMessages.map((message, messageIndex) => {
            const isUser = message.type === 'user';
            
            return (
              <div key={message.id}>
                <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
                  {isUser ? (
                    <div className="flex max-w-[80%] flex-row-reverse items-start space-x-2">
                      <div className="rounded-xl p-3 bg-neutral-700 text-white">
                        <p className="text-base font-inter whitespace-pre-wrap">
                          {message.content}
                          {message.isTyping && <span className="animate-pulse">|</span>}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex max-w-[80%] flex-row items-center space-x-2">
                      <div className="w-6 h-6 flex items-center justify-center flex-shrink-0 mr-2">
                        <img 
                          src="/soundwhite.png" 
                          alt="Timbrality Agent" 
                          className="w-6 h-6 object-contain"
                        />
                      </div>
                      <div className="flex-1">
                        <div className="text-base font-inter text-white">
                          <p className="leading-relaxed">
                            {message.content}
                            {message.isTyping && <span className="animate-pulse">|</span>}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Track Card for agent messages with tracks */}
                {!isUser && message.track && !message.isTyping && (
                  <div className="flex justify-start mb-3">
                    <div className="flex max-w-full flex-row items-start space-x-2">
                      <div className="w-6 h-6 flex items-center justify-center flex-shrink-0 mr-2">
                        <img 
                          src="/soundwhite.png" 
                          alt="Timbrality Agent" 
                          className="w-6 h-6 object-contain"
                        />
                      </div>
                      <div className="flex-1">
                        <div className="transform scale-[0.8] origin-top-left -mb-12">
                          <div className="w-[460px] h-[320px] bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-6 shadow-xl transition-all duration-300 hover:scale-[1.02] cursor-pointer group relative">
                            {/* Top section with artist and action buttons */}
                            <div className="flex items-center justify-between mb-1">
                              {/* Artist pill */}
                              <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-full px-3 py-1 flex items-center gap-1.5">
                                <div className="w-4 h-4 rounded-full overflow-hidden flex-shrink-0">
                                  {message.track.artist_image_url ? (
                                    <img 
                                      src={message.track.artist_image_url} 
                                      alt={message.track.artist}
                                      className="w-full h-full object-cover"
                                    />
                                  ) : (
                                    <div className="w-full h-full bg-neutral-600/60 backdrop-blur-sm rounded-full flex items-center justify-center">
                                      <User className="w-2.5 h-2.5 text-white" />
                                    </div>
                                  )}
                                </div>
                                <span className="text-white font-medium text-xs">{message.track.artist}</span>
                              </div>

                              {/* Action buttons */}
                              <div className="flex items-center gap-2">
                                <button className="w-6 h-6 flex items-center justify-center text-white hover:bg-neutral-700/60 backdrop-blur-sm rounded-full transition-colors duration-200">
                                  <Info className="w-3.5 h-3.5" />
                                </button>
                                <button className="w-6 h-6 flex items-center justify-center text-white hover:bg-neutral-700/60 backdrop-blur-sm rounded-full transition-colors duration-200">
                                  <Plus className="w-3.5 h-3.5" />
                                </button>
                              </div>
                            </div>

                            {/* Song title */}
                            <h3 className="font-playfair text-4xl font-bold text-white mb-2 leading-tight">
                              {message.track.name}
                            </h3>

                            {/* Content section with album cover and stats */}
                            <div className="flex gap-4 mb-3">
                              {/* Album cover */}
                              <div className="w-32 h-32 rounded-xl flex items-center justify-center relative flex-shrink-0 overflow-hidden">
                                {message.track.artwork_url ? (
                                  <img 
                                    src={message.track.artwork_url} 
                                    alt={`${message.track.album} cover`}
                                    className="w-full h-full object-cover"
                                  />
                                ) : (
                                  <div 
                                    className="w-full h-full flex items-center justify-center"
                                    style={{ backgroundColor: '#FF7A38' }}
                                  >
                                    <div className="text-white font-bold text-sm text-center px-2">
                                      {message.track.album || 'Unknown Album'}
                                    </div>
                                  </div>
                                )}
                              </div>

                              {/* Stats section */}
                              <div className="flex-1 space-y-3">
                                {/* Popularity */}
                                <div>
                                  <div className="flex items-center gap-1 mb-1">
                                    <Globe className="w-3 h-3 text-white" />
                                    <span className="text-white font-medium text-sm">Popularity</span>
                                  </div>
                                  <div className="w-full bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-full h-6">
                                    <div 
                                      className="bg-purple-500 h-6 rounded-full transition-all duration-700 ease-out"
                                      style={{ width: `${message.track.popularity || 75}%` }}
                                    ></div>
                                  </div>
                                </div>

                                {/* Duration, Release Date, and Rating */}
                                <div className="flex gap-3">
                                  {/* Duration */}
                                  <div>
                                    <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-xl px-3 py-2 inline-block">
                                      <div className="text-white text-lg font-bold">
                                        {message.track.duration_ms ? 
                                          `${Math.floor(message.track.duration_ms / 60000)}:${Math.floor((message.track.duration_ms % 60000) / 1000).toString().padStart(2, '0')}` 
                                          : '3:20'}
                                      </div>
                                    </div>
                                    <div className="flex items-center gap-1 mt-0.5">
                                      <Clock className="w-2.5 h-2.5 text-white" />
                                      <span className="text-white text-xs">Length</span>
                                    </div>
                                  </div>

                                  {/* Release Date */}
                                  <div>
                                    <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-xl px-3 py-2 inline-block">
                                      <div className="text-white text-lg font-bold">
                                        {message.track.release_date ? 
                                          (() => {
                                            const date = new Date(message.track.release_date);
                                            const month = (date.getMonth() + 1).toString().padStart(2, '0');
                                            const year = date.getFullYear();
                                            return `${month}-${year}`;
                                          })()
                                          : '12-2023'}
                                      </div>
                                    </div>
                                    <div className="flex items-center gap-1 mt-0.5">
                                      <Calendar className="w-2.5 h-2.5 text-white" />
                                      <span className="text-white text-xs whitespace-nowrap">Release date</span>
                                    </div>
                                  </div>

                                  {/* Rating */}
                                  <div>
                                    <div className="bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 rounded-xl px-3 py-2 inline-block">
                                      <div className="text-white text-lg font-bold">
                                        {(() => {
                                          if (message.track.rating) {
                                            return message.track.rating > 10 ? message.track.rating : Math.round(message.track.rating * 10);
                                          }
                                          if (message.track.aoty_score) {
                                            return Math.round(message.track.aoty_score);
                                          }
                                          if (message.track.similarity) {
                                            return Math.round(message.track.similarity * 100);
                                          }
                                          return 85;
                                        })()}
                                      </div>
                                    </div>
                                    <div className="flex items-center gap-1 mt-0.5">
                                      <Star className="w-2.5 h-2.5 text-white" />
                                      <span className="text-white text-xs">Rating</span>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* Spotify button */}
                            <button 
                              onClick={() => message.track?.spotify_url && window.open(message.track.spotify_url, '_blank')}
                              className="w-full bg-neutral-700/60 backdrop-blur-sm border border-neutral-600/30 hover:bg-neutral-600/60 rounded-xl py-3 flex items-center justify-center gap-3 transition-colors duration-200"
                            >
                              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor" style={{ minWidth: '20px' }}>
                                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.299.421-1.02.599-1.559.3z" />
                                </svg>
                              <span className="text-white font-medium">Open in Spotify</span>
                              </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
          
          {isThinking && <ThinkingAnimation />}
        </div>

        {/* Background glow effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 via-transparent to-blue-500/5 pointer-events-none rounded-2xl"></div>
      </div>
    </div>
  );
};