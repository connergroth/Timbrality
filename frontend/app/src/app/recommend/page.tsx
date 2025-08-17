'use client'

import { useSupabase } from '@/components/SupabaseProvider'
import { useState } from 'react'
import { Navbar } from '@/components/Navbar'
import { NavigationSidebar } from '@/components/NavigationSidebar'
import { useSidebar } from '@/contexts/SidebarContext'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Shuffle, Sparkles, RefreshCw, Music } from 'lucide-react'

// Custom hooks
import { useRecommendations } from '@/hooks/useRecommendations'
import { useLastfmConnection } from '@/hooks/useLastfmConnection'

// Components
import { LastfmBanner } from '@/components/recommend/LastfmBanner'
import { AlgorithmSelector } from '@/components/recommend/AlgorithmSelector'
import { AdvancedFiltersModal } from '@/components/recommend/AdvancedFiltersModal'
import { RecommendationCard } from '@/components/recommend/RecommendationCard'
import { GenerationOverlay } from '@/components/recommend/GenerationOverlay'
import { MLAnimationCard } from '@/components/recommend/MLAnimationCard'
import { FullScreenMLGeneration } from '@/components/recommend/FullScreenMLGeneration'

export default function RecommendPage() {
  const { user, loading, signOut } = useSupabase()
  const { isExpanded } = useSidebar()
  const [showFiltersModal, setShowFiltersModal] = useState(false)

  // Custom hooks
  const {
    recommendations,
    isGenerating,
    generationStage,
    filters,
    setFilters,
    activeAlgorithm,
    setActiveAlgorithm,
    generateRecommendations,
    submitFeedback
  } = useRecommendations(user)

  const {
    lastfmConnected,
    showBanner,
    connectLastfm,
    dismissBanner
  } = useLastfmConnection(user)

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-neutral-900 flex items-center justify-center">
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
      <div className="min-h-screen bg-neutral-900 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-inter font-semibold mb-4 tracking-tight text-white">Please sign in to continue</h1>
          <p className="text-neutral-300 mb-4 font-inter">You need to authenticate to access personalized recommendations.</p>
          <button 
            onClick={() => window.location.href = '/auth'}
            className="bg-neutral-800 text-white px-6 py-3 rounded-lg font-inter font-medium hover:bg-neutral-700 transition-colors"
          >
            Go to Auth Page
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen relative">
      {/* Solid Dark Gray/Black Background */}
      <div className="fixed inset-0 bg-neutral-900"></div>
      
      {/* Navigation Sidebar */}
      <NavigationSidebar user={user} onSignOut={signOut} />

      {/* Main Content */}
      <div className={`flex-1 flex flex-col transition-all duration-300 ease-in-out relative z-10 ${
        isExpanded ? 'ml-40' : 'ml-16'
      }`}>
        <Navbar user={user} onSignOut={signOut} />
        
        <main className="flex-1 px-6 py-10 max-w-4xl mx-auto relative">
          {/* Last.fm Connection Banner */}
          <LastfmBanner
            isVisible={showBanner && !lastfmConnected}
            onConnect={connectLastfm}
            onDismiss={dismissBanner}
          />

          {/* Premium Header */}
          <div className="mb-10 text-left">
            <h1 className="text-4xl font-playfair font-bold mb-3 text-white tracking-tight">
              AI Music Discovery
            </h1>
            <p className="text-lg text-slate-300 max-w-2xl leading-relaxed">
              Discover your next favorite songs with our advanced machine learning algorithms
            </p>
          </div>

          {/* Algorithm Selector */}
          <AlgorithmSelector
            activeAlgorithm={activeAlgorithm}
            onAlgorithmChange={setActiveAlgorithm}
            onOpenFilters={() => setShowFiltersModal(true)}
          />

          {/* Generate Button */}
          <div className="mb-10 text-center">
            <Button
              onClick={generateRecommendations}
              disabled={isGenerating}
              size="lg"
              className="bg-white text-black px-8 py-3 text-lg font-semibold rounded-xl hover:bg-gray-100 transition-colors"
            >
              {isGenerating ? (
                <>
                  <RefreshCw className="h-5 w-5 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="h-5 w-5 mr-2" />
                  Generate AI Recommendations
                </>
              )}
            </Button>
          </div>
        </main>
      </div>

      {/* Full Screen ML Generation */}
      <FullScreenMLGeneration 
        isVisible={isGenerating} 
        onComplete={() => {}}
        recommendations={recommendations}
      />
      
      {/* Advanced Filters Modal */}
      <AdvancedFiltersModal
        isOpen={showFiltersModal}
        onClose={() => setShowFiltersModal(false)}
        filters={filters}
        onFiltersChange={setFilters}
      />
    </div>
  )
}