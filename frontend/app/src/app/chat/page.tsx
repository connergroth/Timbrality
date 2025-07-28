'use client'

import { useSupabase } from '@/components/SupabaseProvider'
import { useState, useEffect } from 'react'
import { Navbar } from '@/components/Navbar'
import { AlgorithmSidebar } from '@/components/AlgorithmSidebar'
import { AgentChat } from '@/components/AgentChat'
import { ChatSidebar } from '@/components/ChatSidebar'
import type { Track as AgentTrack } from '@/lib/agent'

export default function ChatPage() {
  const { user, loading, signOut } = useSupabase();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isChatSidebarOpen, setIsChatSidebarOpen] = useState(true)
  const [agentRecommendations, setAgentRecommendations] = useState<AgentTrack[]>([])

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

  const handleAgentRecommendations = (tracks: AgentTrack[]) => {
    setAgentRecommendations(tracks)
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Chat Sidebar */}
      <ChatSidebar 
        isOpen={isChatSidebarOpen}
        onToggle={() => setIsChatSidebarOpen(!isChatSidebarOpen)}
        userId={user.id}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Navbar */}
        <Navbar 
          user={user} 
          onOpenAlgorithmSidebar={() => setIsSidebarOpen(true)}
          onOpenNavigationSidebar={() => {}} // Not used in chat page
          onSignOut={signOut}
        />
        
        {/* Chat Content */}
        <div className="flex-1 flex flex-col">
          <AgentChat 
            userId={user.id} 
            onTrackRecommendations={handleAgentRecommendations}
            className="flex-1 flex flex-col"
          />
        </div>
      </div>

      {/* Algorithm Sidebar */}
      <AlgorithmSidebar 
        isOpen={isSidebarOpen} 
        onClose={() => setIsSidebarOpen(false)} 
      />
    </div>
  )
} 