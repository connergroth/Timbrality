'use client'

import { useSupabase } from '@/components/SupabaseProvider'
import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { AgentChat } from '@/components/AgentChat'
import { NavigationSidebar } from '@/components/NavigationSidebar'
import { AlgorithmSidebar } from '@/components/AlgorithmSidebar'
import { SoundBar } from '@/components/SoundBar'
import { VinylShader } from '@/components/VinylShader'
import { useSidebar } from '@/contexts/SidebarContext'
import { supabase } from '@/lib/supabase'
import type { Track as AgentTrack } from '@/lib/agent'
import { LandingNavbar } from "@/components/landing/LandingNavbar";
import { Hero } from "@/components/landing/Hero";
import { HowItWorks } from "@/components/landing/HowItWorks";
import { Features } from "@/components/landing/Features";
import { MusicalDNA } from "@/components/landing/MusicalDNA";
import { TastePreview } from "@/components/landing/TastePreview";
import { LandingFooter } from "@/components/landing/LandingFooter";

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
  const { isExpanded } = useSidebar();
  const router = useRouter()
  const [userProfile, setUserProfile] = useState<any>(null)
  const [isNavigationSidebarOpen, setIsNavigationSidebarOpen] = useState(false)
  const [isAlgorithmSidebarOpen, setIsAlgorithmSidebarOpen] = useState(false)
  const [agentRecommendations, setAgentRecommendations] = useState<AgentTrack[]>([])
  const [isChatActive, setIsChatActive] = useState(false)
  const [selectedTool, setSelectedTool] = useState<string | null>(null)
  const [showMusicDepthSlider, setShowMusicDepthSlider] = useState(false)
  // Removed clearChatFunction state as it was causing infinite re-renders

  useEffect(() => {
    if (user && !loading) {
      // Try to load from cache first
      const cachedProfile = localStorage.getItem(`user-profile-${user.id}`)
      if (cachedProfile) {
        try {
          const parsedProfile = JSON.parse(cachedProfile)
          setUserProfile(parsedProfile)
        } catch (error) {
          console.error('Error parsing cached profile:', error)
        }
      }
      // Fetch user profile data
      fetchUserProfile()
    }
  }, [user, loading])

  const fetchUserProfile = async () => {
    if (!user) return
    
    try {
      
      // Try to get profile from database first
      const { data, error } = await supabase
        .from('users')
        .select('display_name')
        .eq('id', user.id)
        .single()
      
      if (error && error.code === 'PGRST116') {
        // User doesn't exist in our custom users table, create them
        
        const newUserData = {
          id: user.id,
          username: user.email?.split('@')[0] || 'user',
          email: user.email,
          display_name: user.user_metadata?.full_name || user.user_metadata?.name || user.email?.split('@')[0] || 'User',
          provider: 'spotify',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }
        
        const { data: insertData, error: insertError } = await supabase
          .from('users')
          .insert([newUserData])
          .select('display_name')
          .single()
        
        if (insertError) {
          // Fallback to user metadata
          setUserProfile({ 
            display_name: user.user_metadata?.full_name || user.user_metadata?.name || null
          })
        } else {
          setUserProfile(insertData)
          // Cache the new profile
          localStorage.setItem(`user-profile-${user.id}`, JSON.stringify(insertData))
        }
      } else if (error) {
        // Fallback to user metadata if database query fails
        setUserProfile({ 
          display_name: user.user_metadata?.full_name || user.user_metadata?.name || null
        })
      } else {
        setUserProfile(data)
        // Cache the profile
        localStorage.setItem(`user-profile-${user.id}`, JSON.stringify(data))
      }
    } catch (error) {
      // Fallback to user metadata if there's an exception
      const fallbackProfile = { 
        display_name: user.user_metadata?.full_name || user.user_metadata?.name || null
      }
      setUserProfile(fallbackProfile)
      // Cache the fallback
      localStorage.setItem(`user-profile-${user.id}`, JSON.stringify(fallbackProfile))
    }
  }

  // Extract first word from display_name for welcome message
  const getUserDisplayName = () => {
    if (userProfile?.display_name) {
      return userProfile.display_name.split(' ')[0]
    }
    // Only show fallback if we've tried to load profile but it's null/empty
    if (userProfile !== null) {
      return user?.user_metadata?.full_name?.split(' ')[0] || user?.user_metadata?.name?.split(' ')[0] || user?.email?.split('@')[0] || 'there'
    }
    // While loading, show generic greeting
    return 'there'
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

  const handleToolSelect = (tool: string | null) => {
    // If clicking the same tool, deselect it
    if (tool === selectedTool) {
      setSelectedTool(null)
      if (tool === 'music-depth') {
        setShowMusicDepthSlider(false)
      }
      return
    }
    
    setSelectedTool(tool)
    
    if (tool === 'music-depth') {
      setShowMusicDepthSlider(!showMusicDepthSlider)
    } else if (tool === null) {
      // Clear selection - hide any active tool UI
      setShowMusicDepthSlider(false)
    } else {
      // For other tools, you could show a modal or inline component here
    }
  }

  const handleToggleAlgorithmSidebar = () => {
    setIsAlgorithmSidebarOpen(!isAlgorithmSidebarOpen)
  }

  const handleLogoClick = () => {
    // Reset chat state to return to full home page
    setIsChatActive(false)
    setSelectedTool(null)
    setShowMusicDepthSlider(false)
    // Note: Chat clearing removed to fix infinite re-render issue
  }

  const handleStartChatWithNavigation = (firstMessage: string) => {
    // Generate a unique chat ID
    const chatId = `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    // Navigate to new chat page with the first message
    router.push(`/chat/${chatId}?message=${encodeURIComponent(firstMessage)}`)
    
    // Dispatch custom event to notify sidebar to refresh chat history
    window.dispatchEvent(new CustomEvent('chatHistoryUpdated'))
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

  // Show landing page if no user
  if (!user) {
    return (
      <div className="min-h-screen">
        <LandingNavbar />
        <Hero />
        <HowItWorks />
        <Features />
        <MusicalDNA />
        <TastePreview />
        <LandingFooter />
      </div>
    )
  }

  return (
    <div className="flex min-h-screen relative">
      {/* Solid Dark Gray/Black Background */}
      <div className="fixed inset-0 bg-neutral-900"></div>
      {/* Navigation Sidebar */}
      <NavigationSidebar 
        user={user}
        onSignOut={signOut}
      />

      {/* Main Content */}
      <div className={`flex-1 flex flex-col transition-all duration-300 ease-in-out relative z-10 ${
        isExpanded ? 'ml-40' : 'ml-16'
      }`}>
        <Navbar 
          user={user} 
          onOpenNavigationSidebar={() => setIsNavigationSidebarOpen(!isNavigationSidebarOpen)}
          onSignOut={signOut}
          onToggleAlgorithmSidebar={handleToggleAlgorithmSidebar}
          onLogoClick={handleLogoClick}
        />
        
        <main className={`flex-1 ${isChatActive ? 'pb-24' : 'px-6 py-10'}`}>
          <div className={`max-w-4xl mx-auto transition-all duration-300 ease-in-out`}>
            {/* Welcome Section and Track Grid - Only show when chat is not active */}
            {!isChatActive && (
              <>
                {/* Welcome Section */}
                <div className="mb-10">
                  {/* Sound Bar aligned with text */}
                  <div className="mb-5 flex justify-start">
                    <SoundBar className="" barCount={9} />
                  </div>
                  
                  <h1 className="text-4xl font-playfair font-semibold text-white mb-3 tracking-tight">
                    Welcome back, {getUserDisplayName()}.
                  </h1>
                  <p className="text-base text-slate-300 font-inter font-medium">
                    Discovering your musical preferences...
                  </p>
                </div>

                {/* Track Grid */}
                <div className="mb-10">
                  <h2 className="text-xl font-inter font-semibold mb-7 tracking-tight text-white">Recent Recommendations</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                    {userProfile?.recentTracks?.map((track: Track) => (
                      <div key={track.id} className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-5 shadow-2xl">
                        <div className="flex items-center space-x-4">
                          <img 
                            src={track.cover} 
                            alt={`${track.album} cover`}
                            className="w-14 h-14 rounded-lg flex-shrink-0 object-cover"
                          />
                          <div className="flex-1 min-w-0">
                            <h3 className="font-inter font-semibold text-white truncate text-sm">{track.title}</h3>
                            <p className="text-xs text-slate-300 truncate font-inter">{track.artist}</p>
                            <p className="text-xs text-slate-400 truncate font-inter">{track.album}</p>
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
              onStartChat={handleStartChatWithNavigation}
              isInline={isChatActive}
              onClose={isChatActive ? handleCloseChat : undefined}
              user={user}
              onSelectTool={handleToolSelect}
              selectedTool={selectedTool}
              showMusicDepthSlider={showMusicDepthSlider}
            />
          </div>
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