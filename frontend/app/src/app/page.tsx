'use client'

import { useSupabase } from '@/components/SupabaseProvider'
import { useState, useEffect } from 'react'
import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { AlgorithmSidebar } from '@/components/AlgorithmSidebar'
import { AgentChat } from '@/components/AgentChat'
import { NavigationSidebar } from '@/components/NavigationSidebar'
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
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isNavigationSidebarOpen, setIsNavigationSidebarOpen] = useState(false)
  const [agentRecommendations, setAgentRecommendations] = useState<AgentTrack[]>([])
  const [isChatActive, setIsChatActive] = useState(false)

  useEffect(() => {
    if (user && !loading) {
      // Fetch user profile data
      fetchUserProfile()
    }
  }, [user, loading])

  const fetchUserProfile = async () => {
    // This would fetch user's music taste profile from your backend
    // For now, using mock data
    setUserProfile({
      name: 'Conner',
      taste: ['Atmospheric', 'Experimental', 'Lo-fi'],
      recentTracks: [
        {
          id: 1,
          title: 'Midnight City',
          artist: 'M83',
          album: 'Hurry Up, We\'re Dreaming',
          cover: '/api/placeholder/3000',
          why: 'High atmospheric score, matches your preference for dreamy soundscapes'
        },
        {
          id: 2,
          title: 'Motion',
          artist: 'Calvin Harris',
          album: 'Motion',
          cover: '/api/placeholder/3000',
          why: 'Experimental electronic elements align with your taste'
        },
        {
          id: 3,
          title: 'Teardrop',
          artist: 'Massive Attack',
          album: 'Mezzanine',
          cover: '/api/placeholder/3000',
          why: 'Lo-fi production style matches your preference for raw, intimate sounds'
        }
      ]
    })
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

  // Show loading state while checking authentication
  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
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
          <p className="text-muted-foreground mb-4 font-inter">You need to authenticate to access Timbre.</p>
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
        isOpen={isNavigationSidebarOpen}
        onClose={() => setIsNavigationSidebarOpen(false)}
        userId={user.id}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        <Navbar 
          user={user} 
          onOpenAlgorithmSidebar={() => setIsSidebarOpen(true)}
          onOpenNavigationSidebar={() => setIsNavigationSidebarOpen(!isNavigationSidebarOpen)}
          onSignOut={signOut}
        />
        
        <main className={`flex-1 container mx-auto px-4 py-8 max-w-7xl ${isChatActive ? 'pb-24' : ''}`}>
          {/* Welcome Section and Track Grid - Only show when chat is not active */}
          {!isChatActive && (
            <>
              {/* Welcome Section */}
              <div className="mb-8">
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
                    <div key={track.id} className="bg-card border border-border rounded-lg p-3">
                      <div className="flex items-center space-x-3">
                        <div className="w-12 h-12 bg-muted rounded-md flex-shrink-0"></div>
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
          />
        </main>
      </div>

      {/* Algorithm Sidebar */}
      <AlgorithmSidebar 
        isOpen={isSidebarOpen} 
        onClose={() => setIsSidebarOpen(false)} 
      />
    </div>
  )
} 