'use client'

import { useSupabase } from '@/components/SupabaseProvider'
import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Navbar } from '@/components/Navbar'
import { AlgorithmSidebar } from '@/components/AlgorithmSidebar'
import { NavigationSidebar } from '@/components/NavigationSidebar'
import { ChatHistory } from '@/components/ChatHistory'
import { useSidebar } from '@/contexts/SidebarContext'

export default function ChatPage() {
  const { user, loading, signOut } = useSupabase();
  const { isExpanded } = useSidebar();
  const router = useRouter()
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

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

  const handleChatSelect = (chatId: string) => {
    router.push(`/chat/${chatId}`)
  }

  const handleNewChat = () => {
    const newChatId = `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    router.push(`/chat/${newChatId}`)
  }

  return (
    <div className="flex h-screen bg-background">
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
        
        {/* Chat History Content */}
        <main className="flex-1 container mx-auto px-4 py-8 max-w-7xl">
          <ChatHistory 
            userId={user.id}
            onChatSelect={handleChatSelect}
            onNewChat={handleNewChat}
            className="flex-1 flex flex-col"
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