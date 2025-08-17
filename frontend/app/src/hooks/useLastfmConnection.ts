import { useState, useEffect } from 'react'
import { User } from '@supabase/supabase-js'

export function useLastfmConnection(user: User | null) {
  const [lastfmConnected, setLastfmConnected] = useState(false)
  const [showBanner, setShowBanner] = useState(true)

  useEffect(() => {
    const checkLastfmConnection = async () => {
      if (!user?.id) return

      try {
        // Check if user has Last.fm integration
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/user/${user.id}/lastfm/status`)
        if (response.ok) {
          const data = await response.json()
          setLastfmConnected(data.connected || false)
        } else {
          setLastfmConnected(false)
        }
      } catch (error) {
        console.error('Failed to check Last.fm connection:', error)
        setLastfmConnected(false)
      }
    }

    if (user) {
      checkLastfmConnection()
    }
  }, [user])

  const connectLastfm = () => {
    window.location.href = '/api/auth/lastfm'
  }

  const dismissBanner = () => {
    setShowBanner(false)
  }

  return {
    lastfmConnected,
    showBanner,
    connectLastfm,
    dismissBanner
  }
}