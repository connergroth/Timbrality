import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Loader2, Music, User, Settings, X, ArrowUp, BarChart3, Tag, TrendingUp, Info, Activity, Target, Sliders, Clock, Star, Zap, Plus } from 'lucide-react';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { useAgent, type AgentMessage } from '@/hooks/useAgent';
import { TrackCard } from './TrackCard';
import { TrackRecommendationCard } from './TrackRecommendationCard';
import { RecommendationService } from '@/lib/services/recommendations';
import { PlaylistModal } from './PlaylistModal';
import type { Track } from '@/lib/agent';
import type { UnifiedTrack, TrackFeedback } from '@/lib/types/track';

interface AgentChatProps {
  userId: string;
  onTrackRecommendations?: (tracks: Track[]) => void;
  className?: string;
  chatId?: string;
  onOpenChat?: () => void;
  onClose?: () => void;
  isInline?: boolean;
  onStartChat?: (message: string) => void;
  user?: any; // Add user prop for PlaylistModal
  onSelectTool?: (tool: string | null) => void;
  selectedTool?: string | null;
  showMusicDepthSlider?: boolean;
  onGetClearFunction?: (clearFn: () => void) => void;
  onMessagesUpdate?: (messageCount: number, lastUserMessage?: string) => void;
  onTitleGenerated?: (title: string) => void;
  initialMessage?: string;
}

export function AgentChat({ 
  userId, 
  onTrackRecommendations, 
  className = '', 
  chatId, 
  onOpenChat, 
  onClose,
  isInline = false,
  onStartChat,
  user,
  onSelectTool,
  selectedTool,
  showMusicDepthSlider = false,
  onGetClearFunction,
  onMessagesUpdate,
  onTitleGenerated,
  initialMessage
}: AgentChatProps) {
  const [input, setInput] = useState('');
  const [hasStartedChat, setHasStartedChat] = useState(false);
  const [showChatArea, setShowChatArea] = useState(false);
  const hasProcessedInitialMessageRef = useRef<string | null>(null);
  const [playlistModalOpen, setPlaylistModalOpen] = useState(false);
  const [selectedTrack, setSelectedTrack] = useState<{ id: string; name: string; artist: string } | null>(null);
  const [feedTuning, setFeedTuning] = useState(50); // 0-100, 50 is center
  const [musicDepth, setMusicDepth] = useState([75]); // 0-100, 75 is default
  const inputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const { 
    messages, 
    isLoading, 
    sendMessage, 
    submitFeedback, 
    clearConversation,
    cancelRequest
  } = useAgent({
    userId,
    chatId,
    onTrackRecommendations,
    onMessagesUpdate,
    onTitleGenerated,
    onError: (error) => {
      console.error('Agent error:', error);
    },
  });

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Expose clear function through ref or imperative handle instead of callback
  // This pattern was causing infinite re-renders

  // Auto-submit initial message if provided (track by message content to prevent duplicates)
  useEffect(() => {
    if (initialMessage && hasProcessedInitialMessageRef.current !== initialMessage && !isLoading) {
      hasProcessedInitialMessageRef.current = initialMessage;
      setHasStartedChat(true);
      setShowChatArea(true);
      sendMessage(initialMessage, { streaming: true });
    }
  }, [initialMessage, isLoading, isInline, sendMessage]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const message = input.trim();

    // If this is the first message and we have onStartChat callback, trigger it
    if (!hasStartedChat && onStartChat) {
      setHasStartedChat(true);
      setShowChatArea(true);
      onStartChat(message);
      // Don't send message on home page - navigation will handle it
      if (!isInline) {
        setInput('');
        return;
      }
    }

    setInput('');

    await sendMessage(message, { streaming: true });
  };

  const handleFeedback = (trackId: string, type: 'like' | 'dislike' | 'skip' | 'play_full') => {
    submitFeedback(trackId, type);
  };

  const handleUnifiedFeedback = async (feedback: Omit<TrackFeedback, 'user_id' | 'timestamp'>) => {
    try {
      const result = await RecommendationService.submitFeedback(
        { ...feedback, user_id: userId, timestamp: new Date().toISOString() },
        userId
      );
      if (!result.success) {
        console.error('Failed to submit feedback:', result.message);
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  const handleAddToPlaylist = (trackId: string) => {
    // Find the track details from recent messages
    const allTracks = messages
      .filter(m => m.tracks && m.tracks.length > 0)
      .flatMap(m => m.tracks || []);
    
         const track = allTracks.find(t => t.id === trackId);
    
    if (track) {
             setSelectedTrack({
         id: track.id,
         name: track.name,
         artist: track.artist
       });
      setPlaylistModalOpen(true);
    }
  };

  const handleToolsClick = () => {
    // Dropdown handles this now
  };

  const handleQuickModuleClick = (moduleType: string) => {
    let promptText = '';
    
    switch (moduleType) {
      case 'continue-listening':
        promptText = 'Show me music similar to what I\'ve been listening to lately';
        break;
      case 'because-you-played':
        promptText = 'Give me recommendations based on my recent plays';
        break;
      case 'fresh-for-you':
        promptText = 'Show me fresh releases that match my taste';
        break;
    }

    setInput(promptText);
    // Auto-submit the prompt
    if (promptText && !isLoading) {
      if (!hasStartedChat && onStartChat) {
        setHasStartedChat(true);
        setShowChatArea(true);
        onStartChat(promptText);
        // Don't send message on home page - navigation will handle it
        if (!isInline) {
          setInput('');
          return;
        }
      }
      setInput('');
      sendMessage(promptText, { streaming: true });
    }
  };

  const ThinkingEffect = () => (
    <div className="flex justify-start mb-4">
      <div className="flex items-center space-x-2">
        {/* Avatar space to align with agent messages */}
        <div className="w-6 h-6 flex items-center justify-center flex-shrink-0 mr-2">
          <img 
            src="/soundwhite.png" 
            alt="Timbrality Agent" 
            className="w-6 h-6 object-contain opacity-50"
          />
        </div>
        {/* Thinking text */}
        <div className="relative inline-block">
          <span className="text-lg font-inter text-muted-foreground">
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
          <span className="ml-1 text-lg font-inter text-muted-foreground animate-bounce">...</span>
        </div>
      </div>
    </div>
  );

  const ToolMessage = ({ message }: { message: AgentMessage }) => {
    const getIcon = () => {
      if (message.toolStatus === 'complete') return '✓';
      if (message.toolStatus === 'error') return '✗';
      return '•';
    };

    const getColor = () => {
      if (message.toolStatus === 'complete') return 'text-green-600';
      if (message.toolStatus === 'error') return 'text-red-600';
      return 'text-blue-600';
    };

    return (
      <div className="flex justify-start py-1">
        <div className="flex items-center space-x-2">
          {/* Avatar space to align with agent messages */}
          <div className="w-6 h-6 flex items-center justify-center flex-shrink-0 mr-2">
            <span className={`font-mono text-xs ${getColor()}`}>{getIcon()}</span>
          </div>
          {/* Tool message */}
          <span className="font-inter text-muted-foreground text-sm">{message.content}</span>
        </div>
      </div>
    );
  };

  const MessageBubble = ({ message }: { message: AgentMessage }) => {
    // Handle different message types
    if (message.type === 'system' && message.content === 'Thinking') {
      return <ThinkingEffect />;
    }
    
    if (message.type === 'tool') {
      return <ToolMessage message={message} />;
    }
    
    // Regular user/agent messages
    const isUser = message.type === 'user';
    
    return (
      <>
        {/* Text Message */}
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
          {isUser ? (
            <div className="flex max-w-[80%] flex-row-reverse items-start space-x-2">
              {/* Message Content for user - keep bubble */}
              <div className="rounded-lg p-3 bg-muted text-foreground">
                <p className="text-[15px] font-inter whitespace-pre-wrap">{message.content}</p>
              </div>
            </div>
          ) : (
            <div className="flex max-w-[80%] flex-row items-center space-x-2">
              {/* Avatar for agent */}
              <div className="w-6 h-6 flex items-center justify-center flex-shrink-0 mr-2">
                <img 
                  src="/soundwhite.png" 
                  alt="Timbrality Agent" 
                  className="w-6 h-6 object-contain"
                />
              </div>
              {/* Message Content for agent - no bubble, just text */}
              <div className="flex-1">
                <p className="text-[15px] font-inter whitespace-pre-wrap text-foreground">{message.content}</p>
              </div>
            </div>
          )}
        </div>

        {/* Track Recommendations - Show after agent messages that have tracks */}
        {!isUser && message.tracks && message.tracks.length > 0 && (
          <>
            {message.tracks.slice(0, 6).map((track, index) => (
              <div key={`${message.id}-track-${index}`} className="flex justify-start mb-4">
                <div className="flex max-w-[80%] flex-row items-start space-x-2">
                  {/* Avatar space to align with agent messages */}
                  <div className="w-6 h-6 flex items-center justify-center flex-shrink-0 mr-2">
                    <img 
                      src="/soundwhite.png" 
                      alt="Timbrality Agent" 
                      className="w-6 h-6 object-contain"
                    />
                  </div>
                  
                  {/* Track Card */}
                  <div className="flex-1">
                                         <TrackRecommendationCard
                       track={RecommendationService.fromAgentTrack(
                         track, 
                         "A personalized recommendation based on your preferences"
                       )}
                      onFeedback={handleUnifiedFeedback}
                      onAddToPlaylist={handleAddToPlaylist}
                      size="standard"
                      showSource={false}
                    />
                  </div>
                </div>
              </div>
            ))}
          </>
        )}
      </>
    );
  };

  return (
    <div className={className}>
      {/* Chat Messages Area - Show when chat is active or has messages */}
      {(isInline || messages.length > 0 || showChatArea) && (
        <div className={`w-full ${isInline ? 'pb-40' : 'mb-6'}`}>
          <div 
            ref={chatContainerRef}
            className={isInline ? "p-4 space-y-4" : "space-y-4"}
          >
            <div className={isInline ? "max-w-2xl mx-auto px-4" : "max-w-2xl"}>
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              
              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>
      )}

      {/* Main Input */}
      <div className={isInline ? "fixed bottom-0 left-16 right-0 bg-background/95 backdrop-blur-sm p-4 shadow-lg transition-all duration-300 ease-in-out" : "mb-6"}>
          {!isInline && (
          <h2 className="text-xl font-inter font-semibold mb-4 tracking-tight">
            What are you in the mood for?
          </h2>
        )}
        <div className={isInline ? "max-w-2xl mx-auto" : "max-w-2xl"}>
          <form onSubmit={handleSubmit}>
            <div 
              className="bg-card rounded-xl px-4 pt-3 pb-2 relative cursor-text border border-border/20 border-glow-animation"
              onClick={(e) => {
                // Don't focus input if clicking on slider area
                if (!(e.target as HTMLElement).closest('[data-slider-area]')) {
                  inputRef.current?.focus()
                }
              }}
            >
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Songs that feel like a summer night..."
                autoComplete="off"
                disabled={isLoading}
                className="w-full bg-transparent text-foreground placeholder:text-muted-foreground outline-none border-none text-base font-inter mb-8 disabled:opacity-50"
              />
              
              {/* Music Depth Slider */}
              {showMusicDepthSlider && (
                <div className="mb-6 px-2 cursor-default" data-slider-area>
                  <div className="flex items-center justify-between mb-3">
                    <label className="text-sm font-medium text-foreground">Music Depth</label>
                    <span className="text-xs text-muted-foreground">{musicDepth[0]}%</span>
                  </div>
                  <Slider
                    value={musicDepth}
                    onValueChange={setMusicDepth}
                    max={100}
                    min={0}
                    step={1}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-2">
                    <span>Surface</span>
                    <span>Deep</span>
                  </div>
                </div>
              )}
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <DropdownMenu onOpenChange={(open) => {
                    // Remove focus when dropdown closes
                    if (!open) {
                      setTimeout(() => {
                        const activeElement = document.activeElement as HTMLElement;
                        if (activeElement && activeElement.blur) {
                          activeElement.blur();
                        }
                      }, 0);
                    }
                  }}>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors text-sm font-inter h-auto p-1 focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
                      >
                        <Settings className="w-4 h-4" />
                        <span>Tools</span>
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="w-56" align="start">
                      <DropdownMenuItem onClick={() => onSelectTool?.('nmf-weights')} className="relative">
                        <BarChart3 className="mr-2 h-4 w-4" />
                        <span className="font-inter text-sm">NMF Weights</span>
                        {selectedTool === 'nmf-weights' && <div className="absolute right-2 w-2 h-2 bg-primary rounded-full" />}
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => onSelectTool?.('bert-tags')} className="relative">
                        <Tag className="mr-2 h-4 w-4" />
                        <span className="font-inter text-sm">BERT Tag Similarities</span>
                        {selectedTool === 'bert-tags' && <div className="absolute right-2 w-2 h-2 bg-primary rounded-full" />}
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => onSelectTool?.('tag-influences')} className="relative">
                        <TrendingUp className="mr-2 h-4 w-4" />
                        <span className="font-inter text-sm">Tag Influences</span>
                        {selectedTool === 'tag-influences' && <div className="absolute right-2 w-2 h-2 bg-primary rounded-full" />}
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => onSelectTool?.('aoty-influences')} className="relative">
                        <Target className="mr-2 h-4 w-4" />
                        <span className="font-inter text-sm">AOTY Influences</span>
                        {selectedTool === 'aoty-influences' && <div className="absolute right-2 w-2 h-2 bg-primary rounded-full" />}
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => onSelectTool?.('lastfm-tags')} className="relative">
                        <Activity className="mr-2 h-4 w-4" />
                        <span className="font-inter text-sm">Last.fm Tags</span>
                        {selectedTool === 'lastfm-tags' && <div className="absolute right-2 w-2 h-2 bg-primary rounded-full" />}
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => onSelectTool?.('music-depth')} className="relative">
                        <Sliders className="mr-2 h-4 w-4" />
                        <span className="font-inter text-sm">Music Depth</span>
                        {selectedTool === 'music-depth' && <div className="absolute right-2 w-2 h-2 bg-primary rounded-full" />}
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => onSelectTool?.('algorithm-info')} className="relative">
                        <Info className="mr-2 h-4 w-4" />
                        <span className="font-inter text-sm">How it works</span>
                        {selectedTool === 'algorithm-info' && <div className="absolute right-2 w-2 h-2 bg-primary rounded-full" />}
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => onSelectTool?.(null)} className="relative">
                        <X className="mr-2 h-4 w-4" />
                        <span className="font-inter text-sm">Clear selection</span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    type="button"
                    className="flex items-center space-x-2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <Music className="w-5 h-5" />
                  </button>
                  <button
                    type={isLoading ? "button" : "submit"}
                    disabled={!input.trim() && !isLoading}
                    onClick={isLoading ? cancelRequest : undefined}
                    className="flex items-center justify-center w-12 h-12 rounded-full transition-colors disabled:opacity-50"
                  >
                    {isLoading ? (
                      <div className="w-6 h-6 border border-white rounded-full"></div>
                    ) : (
                      <div className="w-6 h-6 bg-white rounded-full flex items-center justify-center">
                        <ArrowUp className="w-4 h-4 text-black" />
                      </div>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </form>
          
          {/* Quick Modules - Only show when chat is not active */}
          {!isInline && !hasStartedChat && (
            <div className="mt-6 flex flex-wrap gap-3 justify-center">
              <button 
                onClick={() => handleQuickModuleClick('continue-listening')}
                className="bg-card hover:bg-card/80 transition-colors rounded-full px-4 py-2 text-sm font-inter font-medium text-foreground border border-border/50 flex items-center gap-2"
              >
                <Clock className="w-4 h-4" />
                Based on Recent Listening
              </button>
              <button 
                onClick={() => handleQuickModuleClick('because-you-played')}
                className="bg-card hover:bg-card/80 transition-colors rounded-full px-4 py-2 text-sm font-inter font-medium text-foreground border border-border/50 flex items-center gap-2"
              >
                <Star className="w-4 h-4" />
                Highly Rated
              </button>
              <button 
                onClick={() => handleQuickModuleClick('fresh-for-you')}
                className="bg-card hover:bg-card/80 transition-colors rounded-full px-4 py-2 text-sm font-inter font-medium text-foreground border border-border/50 flex items-center gap-2"
              >
                <Zap className="w-4 h-4" />
                Fresh for You
              </button>
              <button 
                onClick={() => handleQuickModuleClick('fresh-for-you')}
                className="bg-card hover:bg-card/80 transition-colors rounded-full px-4 py-2 text-sm font-inter font-medium text-foreground border border-border/50 flex items-center gap-2"
              >
                <Plus className="w-4 h-4" />
                New Playlist
              </button>
            </div>
          )}
        </div>
      </div>


      {/* Playlist Modal */}
             {selectedTrack && (
         <PlaylistModal
           isOpen={playlistModalOpen}
           onClose={() => {
             setPlaylistModalOpen(false);
             setSelectedTrack(null);
           }}
           trackId={selectedTrack.id}
           trackName={selectedTrack.name}
           artistName={selectedTrack.artist}
           userId={userId}
           user={user}
         />
       )}
    </div>
  );
}