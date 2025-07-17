'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useSupabase } from '@/components/SupabaseProvider'

export default function HomePage() {
  const { user, loading } = useSupabase()
  const router = useRouter()

  useEffect(() => {
    if (!loading) {
      if (user) {
        router.push('/recommend')
      } else {
        router.push('/auth')
      }
    }
  }, [user, loading, router])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    )
  }

  return null
} 