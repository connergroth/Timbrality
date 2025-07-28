'use client'

import { useEffect } from 'react'
import { X } from 'lucide-react'
import { AgentChat } from './AgentChat'
import type { Track } from '@/lib/agent'

interface ChatModalProps {
  isOpen: boolean
  onClose: () => void
  userId: string
  chatId: string
}

export function ChatModal({ isOpen, onClose, userId, chatId }: ChatModalProps) {
  // Close modal on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden'
    }

    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = 'unset'
    }
  }, [isOpen, onClose])

  if (!isOpen) return null

  const handleTrackRecommendations = (tracks: Track[]) => {
    // Handle track recommendations in modal context
    console.log('Track recommendations in modal:', tracks)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal Content */}
      <div className="relative w-full h-full max-w-6xl max-h-[90vh] bg-background rounded-lg shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 flex items-center justify-center">
              <img 
                src="/soundwhite.png" 
                alt="Timbre" 
                className="w-5 h-5 object-contain"
              />
            </div>
            <span className="font-playfair text-lg font-bold text-primary">Timbre Chat</span>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-muted rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Chat Content */}
        <div className="flex-1 overflow-hidden">
          <AgentChat 
            userId={userId} 
            onTrackRecommendations={handleTrackRecommendations}
            className="h-full"
            chatId={chatId}
          />
        </div>
      </div>
    </div>
  )
} 