'use client'

import { useSupabase } from '@/components/SupabaseProvider'
import { useState, useEffect, useCallback, useRef } from 'react'
import { useParams, useRouter, useSearchParams } from 'next/navigation'
import { ArrowLeft } from 'lucide-react'
import { Navbar } from '@/components/Navbar'
import { AlgorithmSidebar } from '@/components/AlgorithmSidebar'
import { AgentChat } from '@/components/AgentChat'
import { NavigationSidebar } from '@/components/NavigationSidebar'
import { Button } from '@/components/ui/button'
import { agentService } from '@/lib/agent'
import type { Track as AgentTrack } from '@/lib/agent'
import { useSidebar } from '@/contexts/SidebarContext'

interface ChatSession {
  id: string
  title: string
  lastMessage: string
  timestamp: Date
  messageCount: number
}

export default function ChatSessionPage() {
  const { user, loading, signOut } = useSupabase();
  const { isExpanded } = useSidebar();
  const router = useRouter()
  const params = useParams()
  const searchParams = useSearchParams()
  const chatId = params.id as string || 'default'
  const initialMessage = searchParams.get('message')
  
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [agentRecommendations, setAgentRecommendations] = useState<AgentTrack[]>([])
  const [currentChat, setCurrentChat] = useState<ChatSession | null>(null)
  const [isNewChat, setIsNewChat] = useState(false)
  const [hasAutoSubmitted, setHasAutoSubmitted] = useState(false)

  const createNewChatSession = useCallback(() => {
    const newChat: ChatSession = {
      id: chatId,
      title: 'New Chat',
      lastMessage: '',
      timestamp: new Date(),
      messageCount: 0
    }
    setCurrentChat(newChat)
    setIsNewChat(true)
  }, [chatId])

  const loadOrCreateChatSession = useCallback(() => {
    const savedSessions = localStorage.getItem(`timbre-chats-${user?.id}`)
    if (savedSessions) {
      const sessions: ChatSession[] = JSON.parse(savedSessions)
      const chat = sessions.find(session => session.id === chatId)
      if (chat) {
        setCurrentChat(chat)
        setIsNewChat(false)
      } else {
        // Create new chat session
        createNewChatSession()
      }
    } else {
      // Create new chat session
      createNewChatSession()
    }
  }, [user?.id, chatId, createNewChatSession])

  const updateChatSession = useCallback((title?: string, lastMessage?: string, messageCount?: number) => {
    if (!currentChat || !user) return

    const updatedChat: ChatSession = {
      ...currentChat,
      title: title || currentChat.title,
      lastMessage: lastMessage || currentChat.lastMessage,
      timestamp: new Date(),
      messageCount: messageCount !== undefined ? messageCount : currentChat.messageCount
    }

    // Update current state
    setCurrentChat(updatedChat)

    // Update localStorage
    const savedSessions = localStorage.getItem(`timbre-chats-${user.id}`)
    let sessions: ChatSession[] = savedSessions ? JSON.parse(savedSessions) : []
    
    const existingIndex = sessions.findIndex(s => s.id === chatId)
    if (existingIndex >= 0) {
      sessions[existingIndex] = updatedChat
    } else {
      sessions.unshift(updatedChat)
    }

    localStorage.setItem(`timbre-chats-${user.id}`, JSON.stringify(sessions))
    
    // Dispatch custom event to notify sidebar to refresh chat history
    window.dispatchEvent(new CustomEvent('chatHistoryUpdated'))
  }, [currentChat, user, chatId])

  useEffect(() => {
    if (user && chatId) {
      loadOrCreateChatSession()
    }
  }, [user, chatId, loadOrCreateChatSession])

  // Clean up URL parameter after initial message is processed
  useEffect(() => {
    if (initialMessage && !hasAutoSubmitted) {
      // Remove the message parameter from URL for cleaner experience
      const newUrl = `/chat/${chatId}`
      window.history.replaceState({}, '', newUrl)
      setHasAutoSubmitted(true)
    }
  }, [initialMessage, chatId, hasAutoSubmitted])

  const handleMessagesUpdate = useCallback((messageCount: number, lastUserMessage?: string) => {
    if (lastUserMessage) {
      updateChatSession(undefined, lastUserMessage, messageCount)
    }
  }, [updateChatSession])

  const handleTitleGenerated = useCallback((title: string) => {
    updateChatSession(title, undefined, undefined)
  }, [updateChatSession])

  if (loading) {
    return (
      <div className="min-h-screen bg-neutral-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-neutral-900 flex items-center justify-center">
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

  const handleAgentRecommendations = (tracks: AgentTrack[]) => {
    setAgentRecommendations(tracks)
  }

  const handleChatStart = (firstMessage: string) => {
    // Just update with temporary title and first message
    updateChatSession('New Chat', firstMessage, 1)
  }

  


  const handleBackToHistory = () => {
    router.push('/chat')
  }

  return (
    <div className="flex h-screen bg-neutral-900">
      {/* Persistent Sidebar */}
      <NavigationSidebar user={user} onSignOut={signOut} />

      {/* Main Chat Area */}
      <div className={`flex-1 flex flex-col transition-all duration-300 ease-in-out ${
        isExpanded ? 'ml-40' : 'ml-16'
      }`}>
        {/* Navbar */}
        <Navbar 
          user={user} 
          onSignOut={signOut}
          onToggleAlgorithmSidebar={() => setIsSidebarOpen(true)}
        />
        
        {/* Chat Content */}
        <div className="flex-1 overflow-y-auto pl-0 pr-0">
          {currentChat && (
            <AgentChat 
              userId={user.id} 
              onTrackRecommendations={handleAgentRecommendations}
              className="min-h-full"
              chatId={chatId}
              isInline={true}
              onStartChat={handleChatStart}
              user={user}
              onMessagesUpdate={handleMessagesUpdate}
              onTitleGenerated={handleTitleGenerated}
              initialMessage={initialMessage || undefined}
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