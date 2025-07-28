'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { 
  Home, 
  MessageSquare, 
  Music, 
  Settings, 
  User
} from 'lucide-react'
import Link from 'next/link'

interface NavigationSidebarProps {
  isOpen: boolean
  onClose: () => void
  userId: string
}

export function NavigationSidebar({ isOpen, onClose, userId }: NavigationSidebarProps) {
  const navigationItems = [
    {
      icon: Home,
      label: 'Home',
      href: '/',
      description: 'Your music dashboard'
    },
    {
      icon: MessageSquare,
      label: 'Chat',
      href: '/chat',
      description: 'AI music assistant'
    },
    {
      icon: Music,
      label: 'Recommendations',
      href: '/recommend',
      description: 'Discover new music'
    },
    {
      icon: User,
      label: 'Profile',
      href: '/profile',
      description: 'Your music profile'
    },
    {
      icon: Settings,
      label: 'Settings',
      href: '/settings',
      description: 'App preferences'
    }
  ]

  if (!isOpen) {
    return null
  }

  return (
    <div className="w-80 bg-sidebar flex flex-col">
      {/* Navigation Items */}
      <div className="flex-1 p-4">
        <nav className="space-y-2">
          {navigationItems.map((item) => {
            const Icon = item.icon
            return (
              <Link key={item.href} href={item.href}>
                <Button
                  variant="ghost"
                  className="w-full justify-start h-12 px-4 hover:bg-sidebar-accent/50"
                  onClick={onClose}
                >
                  <Icon className="h-5 w-5 mr-3" />
                  <div className="flex flex-col items-start">
                    <span className="font-inter font-medium text-sm">
                      {item.label}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {item.description}
                    </span>
                  </div>
                </Button>
              </Link>
            )
          })}
        </nav>
      </div>
    </div>
  )
} 