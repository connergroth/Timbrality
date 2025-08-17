'use client'

import { useState, useEffect } from 'react'
import { MessageCircle, Plus, Clock, Music, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface ChatSession {
  id: string
  title: string
  lastMessage: string
  timestamp: Date
  messageCount: number
}

interface ChatHistoryProps {
  userId: string
  onChatSelect: (chatId: string) => void
  onNewChat: () => void
  className?: string
}

export function ChatHistory({ userId, onChatSelect, onNewChat, className = '' }: ChatHistoryProps) {
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([])

  useEffect(() => {
    loadChatSessions()
  }, [userId])

  const loadChatSessions = () => {
    const savedSessions = localStorage.getItem(`timbre-chats-${userId}`)
    if (savedSessions) {
      const sessions: ChatSession[] = JSON.parse(savedSessions)
      setChatSessions(sessions.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()))
    }
  }

  const deleteChatSession = (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    const updatedSessions = chatSessions.filter(session => session.id !== chatId)
    setChatSessions(updatedSessions)
    localStorage.setItem(`timbre-chats-${userId}`, JSON.stringify(updatedSessions))
    
    // Also remove the specific chat messages
    localStorage.removeItem(`timbre-chat-${chatId}-${userId}`)
  }

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date()
    const chatDate = new Date(timestamp)
    const diffInHours = (now.getTime() - chatDate.getTime()) / (1000 * 60 * 60)
    
    if (diffInHours < 24) {
      return chatDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } else if (diffInHours < 48) {
      return 'Yesterday'
    } else if (diffInHours < 168) { // 7 days
      return chatDate.toLocaleDateString([], { weekday: 'long' })
    } else {
      return chatDate.toLocaleDateString([], { month: 'short', day: 'numeric' })
    }
  }

  const truncateMessage = (message: string, maxLength: number = 60) => {
    if (message.length <= maxLength) return message
    return message.substring(0, maxLength) + '...'
  }

  return (
    <div className={className}>
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-4xl font-playfair font-semibold tracking-tight mb-2">Chat History</h1>
          <p className="text-muted-foreground font-inter">Your conversations with Timbre</p>
        </div>
        <Button 
          onClick={onNewChat}
          className="flex items-center gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </Button>
      </div>

      {/* Chat Sessions */}
      {chatSessions.length === 0 ? (
        <div className="mb-8">
          <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mb-4">
            <MessageCircle className="w-8 h-8 text-muted-foreground" />
          </div>
          <h3 className="text-xl font-inter font-semibold mb-2">No conversations yet</h3>
          <p className="text-muted-foreground mb-6 font-inter">Start a new chat to get music recommendations</p>
          <Button 
            onClick={onNewChat}
            className="flex items-center gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
          >
            <Plus className="w-4 h-4" />
            Start Your First Chat
          </Button>
        </div>
      ) : (
        <div className="grid gap-4">
          {chatSessions.map((session) => (
            <div
              key={session.id}
              onClick={() => onChatSelect(session.id)}
              className="group relative bg-card rounded-lg p-3 hover:bg-card/80 transition-colors cursor-pointer border border-border/30"
            >
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0">
                  <Music className="w-3 h-3 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-inter font-medium text-foreground truncate text-sm">
                    {session.title || 'Untitled Chat'}
                  </h3>
                </div>
                <div className="text-xs text-muted-foreground flex-shrink-0">
                  {formatTimestamp(session.timestamp)}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => deleteChatSession(session.id, e)}
                  className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive h-6 w-6 p-0"
                >
                  <Trash2 className="w-3 h-3" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}