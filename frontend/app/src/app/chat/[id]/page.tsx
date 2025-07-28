'use client'

import { useSupabase } from '@/components/SupabaseProvider'
import { useState, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { Navbar } from '@/components/Navbar'
import { AlgorithmSidebar } from '@/components/AlgorithmSidebar'
import { AgentChat } from '@/components/AgentChat'
import { ChatSidebar } from '@/components/ChatSidebar'
import type { Track as AgentTrack } from '@/lib/agent'

interface ChatSession {
  id: string
  title: string
  lastMessage: string
  timestamp: Date
  isActive: boolean
}

export default function ChatSessionPage() {
  const { user, loading, signOut } = useSupabase();
  const params = useParams()
  const router = useRouter()
  const chatId = params.id as string
  
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isChatSidebarOpen, setIsChatSidebarOpen] = useState(true)
  const [agentRecommendations, setAgentRecommendations] = useState<AgentTrack[]>([])
  const [currentChat, setCurrentChat] = useState<ChatSession | null>(null)

  useEffect(() => {
    if (user && chatId) {
      loadChatSession()
    }
  }, [user, chatId])

  const loadChatSession = () => {
    // Load chat session from localStorage
    const savedSessions = localStorage.getItem(`timbre-chats-${user?.id}`)
    if (savedSessions) {
      const sessions: ChatSession[] = JSON.parse(savedSessions)
      const chat = sessions.find(session => session.id === chatId)
      if (chat) {
        setCurrentChat(chat)
      } else {
        // Chat not found, redirect to main chat page
        router.push('/chat')
      }
    } else {
      // No chats found, redirect to main chat page
      router.push('/chat')
    }
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
          {currentChat && (
            <AgentChat 
              userId={user.id} 
              onTrackRecommendations={handleAgentRecommendations}
              className="flex-1 flex flex-col"
              chatId={chatId}
            />
          )}
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