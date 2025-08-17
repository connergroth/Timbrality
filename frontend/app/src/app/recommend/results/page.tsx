'use client'

import { useSupabase } from '@/components/SupabaseProvider'
import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Navbar } from '@/components/Navbar'
import { NavigationSidebar } from '@/components/NavigationSidebar'
import { useSidebar } from '@/contexts/SidebarContext'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ArrowLeft, Shuffle, Filter, Download, Share2 } from 'lucide-react'
import { RecommendationCard } from '@/components/recommend/RecommendationCard'
import { useRecommendations } from '@/hooks/useRecommendations'

export default function RecommendationResultsPage() {
  const { user, loading, signOut } = useSupabase()
  const { isExpanded } = useSidebar()
  const router = useRouter()
  
  const {
    recommendations,
    activeAlgorithm,
    generateRecommendations,
    submitFeedback
  } = useRecommendations(user)

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="relative w-16 h-16">
          <div className="absolute inset-0 rounded-full border-4 border-white/20"></div>
          <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-white animate-spin"></div>
          <div className="absolute inset-2 rounded-full border-2 border-transparent border-t-white/60 animate-spin" style={{animationDirection: 'reverse', animationDuration: '0.8s'}}></div>
          <div className="absolute top-1/2 left-1/2 w-2 h-2 bg-white rounded-full transform -translate-x-1/2 -translate-y-1/2 animate-pulse"></div>
        </div>
      </div>
    )
  }

  // Auth check
  if (!user) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-inter font-semibold mb-4 tracking-tight">Please sign in to continue</h1>
          <p className="text-muted-foreground mb-4 font-inter">You need to authenticate to access personalized recommendations.</p>
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

  return (
    <div className="flex min-h-screen bg-background relative">
      {/* Navigation Sidebar */}
      <NavigationSidebar user={user} onSignOut={signOut} />

      {/* Main Content */}
      <div className={`flex-1 flex flex-col transition-all duration-300 ease-in-out relative z-10 ${
        isExpanded ? 'ml-40' : 'ml-16'
      }`}>
        <Navbar user={user} onSignOut={signOut} />
        
        <main className="flex-1 container mx-auto px-4 py-8 max-w-7xl">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-4 mb-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => router.push('/recommend')}
                className="flex items-center gap-2"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to Recommendations
              </Button>
            </div>
            
            <div className="flex items-start justify-between">
              <div>
                <h1 className="text-3xl font-playfair font-bold mb-2">Your AI-Generated Recommendations</h1>
                <p className="text-muted-foreground">
                  Discover your next favorite songs, powered by advanced machine learning
                </p>
              </div>
              
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="text-sm">
                  {activeAlgorithm === 'hybrid' ? 'Hybrid AI' : 
                   activeAlgorithm === 'collaborative' ? 'Collaborative' : 'Content-Based'} Model
                </Badge>
                <Button variant="outline" size="sm">
                  <Share2 className="h-4 w-4 mr-1" />
                  Share
                </Button>
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-1" />
                  Export
                </Button>
              </div>
            </div>
          </div>

          {/* Stats Bar */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-muted/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-foreground">{recommendations.length}</div>
              <div className="text-sm text-muted-foreground">Recommendations</div>
            </div>
            <div className="bg-muted/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-foreground">
                {Math.round((recommendations.reduce((sum, rec) => sum + rec.confidence, 0) / recommendations.length) * 100) || 0}%
              </div>
              <div className="text-sm text-muted-foreground">Avg Confidence</div>
            </div>
            <div className="bg-muted/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-foreground">
                {[...new Set(recommendations.flatMap(rec => rec.genre))].length}
              </div>
              <div className="text-sm text-muted-foreground">Genres</div>
            </div>
            <div className="bg-muted/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-foreground">
                {Math.round(recommendations.reduce((sum, rec) => sum + rec.rating, 0) / recommendations.length) || 0}
              </div>
              <div className="text-sm text-muted-foreground">Avg Rating</div>
            </div>
          </div>

          {/* Action Bar */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                onClick={generateRecommendations}
                className="flex items-center gap-2"
              >
                <Shuffle className="h-4 w-4" />
                Generate New
              </Button>
              <Button variant="ghost" size="sm">
                <Filter className="h-4 w-4 mr-1" />
                Filter Results
              </Button>
            </div>
            
            <div className="text-sm text-muted-foreground">
              Generated just now
            </div>
          </div>

          {/* Recommendations Grid */}
          {recommendations.length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
              {recommendations.map((rec) => (
                <RecommendationCard
                  key={rec.id}
                  recommendation={rec}
                  onLike={(id: string) => submitFeedback(id, 1)}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-16">
              <div className="w-24 h-24 mx-auto mb-6 bg-muted/50 rounded-full flex items-center justify-center">
                <Shuffle className="h-12 w-12 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-medium mb-2">No Recommendations Yet</h3>
              <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                It looks like you haven't generated any recommendations yet. Go back and click "Generate AI Recommendations" to get started.
              </p>
              <Button onClick={() => router.push('/recommend')}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Recommendations
              </Button>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}