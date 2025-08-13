'use client'

import { useSupabase } from '@/components/SupabaseProvider'
import { useState, useEffect } from 'react'
import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { AgentChat } from '@/components/AgentChat'
import { NavigationSidebar } from '@/components/NavigationSidebar'
import { AlgorithmSidebar } from '@/components/AlgorithmSidebar'
import { SoundBar } from '@/components/SoundBar'
import { VinylShader } from '@/components/VinylShader'
import type { Track as AgentTrack } from '@/lib/agent'

// Define the track type to fix TypeScript error
interface Track {
  id: number
  title: string
  artist: string
  album: string
  cover: string
  why: string
}

export default function HomePage() {
  const { user, loading, signOut } = useSupabase();
  const [userProfile, setUserProfile] = useState<any>(null)
  const [isNavigationSidebarOpen, setIsNavigationSidebarOpen] = useState(false)
  const [isAlgorithmSidebarOpen, setIsAlgorithmSidebarOpen] = useState(false)
  const [agentRecommendations, setAgentRecommendations] = useState<AgentTrack[]>([])
  const [isChatActive, setIsChatActive] = useState(false)
  const [selectedTool, setSelectedTool] = useState<string | null>(null)
  const [showMusicDepthSlider, setShowMusicDepthSlider] = useState(false)

  useEffect(() => {
    if (user && !loading) {
      // Fetch user profile data
      fetchUserProfile()
    }
  }, [user, loading])

  const fetchUserProfile = async () => {
    try {
      // Define the tracks we want to fetch
      const trackQueries = [
        { query: 'Nights Frank Ocean', why: 'Atmospheric R&B production matches your dreamy soundscape preference' },
        { query: 'THANK GOD Travis Scott', why: 'Experimental hip-hop production aligns with your taste for innovative sounds' },
        { query: 'Sk8 JID', why: 'Creative lyricism and unique flow patterns match your preference for artistic expression' }
      ];

      // Fetch track data from backend
      const trackPromises = trackQueries.map(async ({ query, why }, index) => {
        try {
          const response = await fetch(`/api/spotify/search-track?q=${encodeURIComponent(query)}`);
          if (response.ok) {
            const trackData = await response.json();
            return {
              id: index + 1,
              title: trackData.name,
              artist: trackData.artist,
              album: trackData.album,
              cover: trackData.artwork_url,
              why
            };
          }
        } catch (error) {
          console.error(`Error fetching ${query}:`, error);
        }
        return null;
      });

      const tracks = await Promise.all(trackPromises);
      const validTracks = tracks.filter(track => track !== null);

      // Use fetched tracks or fallback to hardcoded data
      setUserProfile({
        name: 'Conner',
        taste: ['Atmospheric', 'Experimental', 'Lo-fi'],
        recentTracks: validTracks.length > 0 ? validTracks : [
          {
            id: 1,
            title: 'Nights',
            artist: 'Frank Ocean',
            album: 'Blonde',
            cover: 'https://i.scdn.co/image/ab67616d0000b2738343d6fc2866e9e52acd74df',
            why: 'Atmospheric R&B production matches your dreamy soundscape preference'
          },
          {
            id: 2,
            title: 'THANK GOD',
            artist: 'Travis Scott',
            album: 'UTOPIA',
            cover: 'https://i.scdn.co/image/ab67616d0000b273881d8d8378cd01099babcd44',
            why: 'Experimental hip-hop production aligns with your taste for innovative sounds'
          },
          {
            id: 3,
            title: 'Sk8',
            artist: 'JID',
            album: 'The Never Story',
            cover: 'https://i.scdn.co/image/ab67616d0000b273230dde08404b4e4c3b5a3b13',
            why: 'Creative lyricism and unique flow patterns match your preference for artistic expression'
          }
        ]
      });
    } catch (error) {
      console.error('Error fetching user profile:', error);
      // Fallback to hardcoded data
      setUserProfile({
        name: 'Conner',
        taste: ['Atmospheric', 'Experimental', 'Lo-fi'],
        recentTracks: [
          {
            id: 1,
            title: 'Nights',
            artist: 'Frank Ocean',
            album: 'Blonde',
            cover: 'https://i.scdn.co/image/ab67616d0000b2738343d6fc2866e9e52acd74df',
            why: 'Atmospheric R&B production matches your dreamy soundscape preference'
          },
          {
            id: 2,
            title: 'THANK GOD',
            artist: 'Travis Scott',
            album: 'UTOPIA',
            cover: 'https://i.scdn.co/image/ab67616d0000b273881d8d8378cd01099babcd44',
            why: 'Experimental hip-hop production aligns with your taste for innovative sounds'
          },
          {
            id: 3,
            title: 'Sk8 Head',
            artist: 'JID',
            album: 'The Never Story',
            cover: 'https://i.scdn.co/image/ab67616d0000b273230dde08404b4e4c3b5a3b13',
            why: 'Creative lyricism and unique flow patterns match your preference for artistic expression'
          }
        ]
      });
    }
  }

  const handleAgentRecommendations = (tracks: AgentTrack[]) => {
    setAgentRecommendations(tracks)
  }

  const handleStartChat = () => {
    setIsChatActive(true)
  }

  const handleCloseChat = () => {
    setIsChatActive(false)
  }

  const handleToolSelect = (tool: string) => {
    setSelectedTool(tool)
    
    if (tool === 'music-depth') {
      setShowMusicDepthSlider(!showMusicDepthSlider)
    } else {
      // For other tools, you could show a modal or inline component here
      console.log(`Selected tool: ${tool}`)
    }
  }

  const handleToggleAlgorithmSidebar = () => {
    setIsAlgorithmSidebarOpen(!isAlgorithmSidebarOpen)
  }

  // Show loading state while checking authentication
  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="relative w-16 h-16">
          <div className="absolute inset-0 rounded-full border-4 border-white/20"></div>
          <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-white animate-spin"></div>
          <div className="absolute inset-2 rounded-full border-2 border-transparent border-t-white/60 animate-spin" style={{animationDirection: 'reverse', animationDuration: '0.8s'}}></div>
          <div className="absolute top-1/2 left-1/2 w-2 h-2 bg-white rounded-full transform -translate-x-1/2 -translate-y-1/2 animate-pulse"></div>
        </div>
      </div>
    )
  }

  // Show auth page if no user
  if (!user) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-inter font-semibold mb-4 tracking-tight">Please sign in to continue</h1>
          <p className="text-muted-foreground mb-4 font-inter">You need to authenticate to access Timbrality.</p>
          <button 
            onClick={() => window.location.href = '/auth'}
            className="bg-primary text-primary-foreground px-6 py-3 rounded-lg font-inter font-medium hover:bg-primary/90 transition-colors"
          >
            Go to Auth Page
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-background">
      {/* Navigation Sidebar */}
      <NavigationSidebar 
        user={user}
        onSignOut={signOut}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col ml-16">
        <Navbar 
          user={user} 
          onOpenNavigationSidebar={() => setIsNavigationSidebarOpen(!isNavigationSidebarOpen)}
          onSignOut={signOut}
          onToggleAlgorithmSidebar={handleToggleAlgorithmSidebar}
        />
        
        <main className={`flex-1 container mx-auto px-4 py-8 max-w-7xl ${isChatActive ? 'pb-24' : ''}`}>
          {/* Welcome Section and Track Grid - Only show when chat is not active */}
          {!isChatActive && (
            <>
              {/* Welcome Section */}
              <div className="mb-8">
                {/* Sound Bar aligned with text */}
                <div className="mb-4 flex justify-start">
                  <SoundBar className="" barCount={9} />
                </div>
                
                <h1 className="text-3xl font-inter font-semibold text-foreground mb-2 tracking-tight">
                  Welcome back, {userProfile?.name || user.email?.split('@')[0]}.
                </h1>
                <p className="text-base text-muted-foreground font-inter font-medium">
                  Your taste leans: {userProfile?.taste?.join(' • ') || 'Loading...'}
                </p>
              </div>

              {/* Track Grid */}
              <div className="mb-8">
                <h2 className="text-xl font-inter font-semibold mb-6 tracking-tight">Recent Recommendations</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {userProfile?.recentTracks?.map((track: Track) => (
                    <div key={track.id} className="bg-card rounded-lg p-3 border border-border/50">
                      <div className="flex items-center space-x-3">
                        <img 
                          src={track.cover} 
                          alt={`${track.album} cover`}
                          className="w-12 h-12 rounded-md flex-shrink-0 object-cover"
                        />
                        <div className="flex-1 min-w-0">
                          <h3 className="font-inter font-semibold text-foreground truncate text-sm">{track.title}</h3>
                          <p className="text-xs text-muted-foreground truncate font-inter">{track.artist}</p>
                          <p className="text-xs text-muted-foreground/70 truncate font-inter">{track.album}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* AI Agent Chat */}
          <AgentChat 
            userId={user.id} 
            onTrackRecommendations={handleAgentRecommendations}
            onStartChat={handleStartChat}
            isInline={isChatActive}
            onClose={isChatActive ? handleCloseChat : undefined}
            user={user}
            onSelectTool={handleToolSelect}
            showMusicDepthSlider={showMusicDepthSlider}
          />
        </main>
      </div>

      {/* Algorithm Sidebar */}
      <AlgorithmSidebar 
        isOpen={isAlgorithmSidebarOpen} 
        onClose={() => setIsAlgorithmSidebarOpen(false)} 
      />
    </div>
  )
} 