'use client'

import { useState, useEffect } from 'react'
import { Plus, MessageSquare, Trash2, MoreVertical } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'

interface ChatSession {
  id: string
  title: string
  lastMessage: string
  timestamp: Date
  isActive: boolean
}

interface ChatSidebarProps {
  isOpen: boolean
  onToggle: () => void
  userId: string
}

export function ChatSidebar({ isOpen, onToggle, userId }: ChatSidebarProps) {
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([])
  const [activeChatId, setActiveChatId] = useState<string | null>(null)

  useEffect(() => {
    // Load chat sessions from localStorage or API
    loadChatSessions()
  }, [userId])

  const loadChatSessions = () => {
    // For now, load from localStorage
    const savedSessions = localStorage.getItem(`timbre-chats-${userId}`)
    if (savedSessions) {
      const sessions = JSON.parse(savedSessions)
      setChatSessions(sessions)
      if (sessions.length > 0 && !activeChatId) {
        setActiveChatId(sessions[0].id)
      }
    }
  }

  const createNewChat = () => {
    const newChat: ChatSession = {
      id: Date.now().toString(),
      title: 'New Chat',
      lastMessage: '',
      timestamp: new Date(),
      isActive: true
    }

    const updatedSessions = [newChat, ...chatSessions]
    setChatSessions(updatedSessions)
    setActiveChatId(newChat.id)
    
    // Save to localStorage
    localStorage.setItem(`timbre-chats-${userId}`, JSON.stringify(updatedSessions))
    
    // Don't automatically navigate - let user click on the chat to open it
  }

  const deleteChat = (chatId: string) => {
    const updatedSessions = chatSessions.filter(session => session.id !== chatId)
    setChatSessions(updatedSessions)
    
    // Save to localStorage
    localStorage.setItem(`timbre-chats-${userId}`, JSON.stringify(updatedSessions))
    
    // If deleted chat was active, switch to first available chat
    if (activeChatId === chatId && updatedSessions.length > 0) {
      setActiveChatId(updatedSessions[0].id)
      window.location.href = `/chat/${updatedSessions[0].id}`
    } else if (updatedSessions.length === 0) {
      // If no chats left, go to main chat page
      window.location.href = '/chat'
    }
  }

  const selectChat = (chatId: string) => {
    setActiveChatId(chatId)
    window.location.href = `/chat/${chatId}`
  }

  if (!isOpen) {
    return (
      <div className="w-16 bg-sidebar flex flex-col items-center py-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggle}
          className="w-10 h-10 p-0"
        >
          <MessageSquare className="h-5 w-5" />
        </Button>
      </div>
    )
  }

  return (
    <div className="w-80 bg-sidebar flex flex-col">
      {/* Header */}
      <div className="p-4">
        <div className="flex items-center justify-between">
          <h2 className="font-inter font-semibold text-lg">Chats</h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={createNewChat}
            className="h-8 w-8 p-0"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Chat List */}
      <div className="flex-1 overflow-y-auto">
        {chatSessions.length === 0 ? (
          <div className="p-4 text-center">
            <p className="text-muted-foreground text-sm">No chats yet</p>
            <Button
              variant="outline"
              size="sm"
              onClick={createNewChat}
              className="mt-2"
            >
              Start a new chat
            </Button>
          </div>
        ) : (
          <div className="p-2">
            {chatSessions.map((session) => (
              <div
                key={session.id}
                className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors ${
                  activeChatId === session.id
                    ? 'bg-sidebar-accent text-sidebar-accent-foreground'
                    : 'hover:bg-sidebar-accent/50'
                }`}
                onClick={() => selectChat(session.id)}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <MessageSquare className="h-4 w-4 flex-shrink-0" />
                    <p className="font-inter font-medium text-sm truncate">
                      {session.title}
                    </p>
                  </div>
                  {session.lastMessage && (
                    <p className="text-xs text-muted-foreground truncate mt-1">
                      {session.lastMessage}
                    </p>
                  )}
                </div>
                
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <MoreVertical className="h-3 w-3" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => deleteChat(session.id)}>
                      <Trash2 className="mr-2 h-4 w-4" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
} 