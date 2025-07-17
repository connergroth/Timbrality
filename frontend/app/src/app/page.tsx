'use client'

import { useSupabase } from '@/components/SupabaseProvider'
import { useState, useEffect, useRef } from 'react'
import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { AlgorithmSidebar } from '@/components/AlgorithmSidebar'
import { Plus, Settings, Mic, Music, Send } from 'lucide-react'

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
  const { user, loading } = useSupabase();
  const [userProfile, setUserProfile] = useState<any>(null)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

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
      taste: ['Atmospheric,Experimental, fi'],
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
          <h1 className="text-2l font-bold mb-4">Please sign in to continue</h1>
          <p className="text-muted-foreground mb-4">You need to authenticate to access Timbre.</p>
          <button 
            onClick={() => window.location.href = '/auth'}
            className="bg-primary text-primary-foreground px-6 py-3 rounded-lg font-medium hover:bg-primary/90 transition-colors"
          >
            Go to Auth Page
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar user={user} onOpenAlgorithmSidebar={() => setIsSidebarOpen(true)} />
      
      <main className="flex-1 container mx-auto px-4 py-8 max-w-7xl">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-playfair font-bold text-foreground mb-2">
            Welcome back, {userProfile?.name || user.email?.split('@')[0]}.
          </h1>
          <p className="text-lg text-muted-foreground font-playfair">
            Your taste leans: {userProfile?.taste?.join(' •') || 'Loading...'}
          </p>
        </div>

        {/* Track Grid */}
        <div className="mb-8">
          <h2 className="text-2xl font-playfair font-semibold mb-4">Recent Recommendations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {userProfile?.recentTracks?.map((track: Track) => (
              <div key={track.id} className="bg-card border border-border rounded-lg p-3">
                <div className="flex items-center space-x-3">
                  <div className="w-12 h-12 bg-muted rounded-md flex-shrink-0"></div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-foreground truncate">{track.title}</h3>
                    <p className="text-sm text-muted-foreground truncate">{track.artist}</p>
                    <p className="text-xs text-muted-foreground truncate">{track.album}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Mood Input */}
        <div className="mb-8">
          <h2 className="text-2xl font-playfair font-semibold mb-4">What are you in the mood for?</h2>
          <div className="max-w-2xl">
            <div 
              className="bg-card border border-border rounded-xl p-4 relative cursor-text"
              onClick={() => inputRef.current?.focus()}
            >
              <input
                ref={inputRef}
                type="text"
                placeholder="Songs that feel like a summer night"
                autoComplete="off"
                className="w-full bg-transparent text-foreground placeholder:text-muted-foreground outline-none border-none text-lg mb-8"
              />
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <button className="flex items-center space-x-2 text-muted-foreground hover:text-foreground transition-colors">
                    <Plus className="w-5 h-5" />
                  </button>
                  <button className="flex items-center space-x-2 text-muted-foreground hover:text-foreground transition-colors">
                    <Settings className="w-5 h-5" />
                    <span>Tools</span>
                  </button>
                </div>
                <div className="flex items-center space-x-4">
                  <button className="flex items-center space-x-2 text-muted-foreground hover:text-foreground transition-colors">
                    <Music className="w-5 h-5" />
                  </button>
                  <button className="flex items-center space-x-2 text-muted-foreground hover:text-foreground transition-colors">
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

      </main>

      <Footer />

      {/* Algorithm Sidebar */}
      <AlgorithmSidebar 
        isOpen={isSidebarOpen} 
        onClose={() => setIsSidebarOpen(false)} 
      />
    </div>
  )
} 