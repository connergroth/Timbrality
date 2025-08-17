import { useState, useEffect, useCallback } from 'react'
import { RecommendationService } from '@/lib/services/recommendations'
import { User } from '@supabase/supabase-js'

export interface Recommendation {
  id: string
  title: string
  artist: string
  album?: string
  rating: number
  year: number
  genre: string[]
  cover: string
  spotifyUrl?: string
  confidence: number
  explanation: {
    collaborative: number
    content: number
    hybrid: number
    reasons: string[]
  }
  audioFeatures?: {
    danceability: number
    energy: number
    valence: number
    acousticness: number
  }
}

export interface RecommendationFilters {
  genres: string[]
  minYear: number
  maxYear: number
  minRating: number
  energyRange: [number, number]
  danceabilityRange: [number, number]
  includeMoodAnalysis: boolean
  diversityWeight: number
  popularityBias: number
}

export const defaultFilters: RecommendationFilters = {
  genres: [],
  minYear: 1950,
  maxYear: new Date().getFullYear(),
  minRating: 0,
  energyRange: [0, 100],
  danceabilityRange: [0, 100],
  includeMoodAnalysis: true,
  diversityWeight: 0.7,
  popularityBias: 0.3
}

const mockRecommendations: Recommendation[] = [
  {
    id: '1',
    title: 'Blue in Green',
    artist: 'Miles Davis',
    album: 'Kind of Blue',
    rating: 92,
    year: 1959,
    genre: ['Jazz', 'Cool Jazz'],
    cover: 'https://i.scdn.co/image/ab67616d0000b273e6f7b4c6c8d9e5f4a3b2c1d0',
    confidence: 0.89,
    explanation: {
      collaborative: 0.75,
      content: 0.82,
      hybrid: 0.89,
      reasons: [
        'Similar to jazz albums you\'ve rated highly',
        'Matches your preference for instrumental music',
        'High ratings from users with similar taste'
      ]
    },
    audioFeatures: {
      danceability: 0.3,
      energy: 0.4,
      valence: 0.6,
      acousticness: 0.9
    }
  },
  {
    id: '2',
    title: 'Paranoid Android',
    artist: 'Radiohead',
    album: 'OK Computer',
    rating: 95,
    year: 1997,
    genre: ['Alternative Rock', 'Art Rock'],
    cover: 'https://i.scdn.co/image/ab67616d0000b273c8b444df094279e70d0ed856',
    confidence: 0.94,
    explanation: {
      collaborative: 0.88,
      content: 0.91,
      hybrid: 0.94,
      reasons: [
        'Aligns with your alternative rock preferences',
        'Complex arrangements match your listening patterns',
        'Highly rated by users who like experimental music'
      ]
    },
    audioFeatures: {
      danceability: 0.4,
      energy: 0.8,
      valence: 0.3,
      acousticness: 0.2
    }
  }
]

// Enhanced mock data generator
function generateMockRecommendations(
  algorithm: 'collaborative' | 'content' | 'hybrid',
  filters: RecommendationFilters
): Recommendation[] {
  const mockTracks = [
    { title: 'Bohemian Rhapsody', artist: 'Queen', album: 'A Night at the Opera', year: 1975, genres: ['Rock', 'Progressive Rock'], rating: 96 },
    { title: 'Hotel California', artist: 'Eagles', album: 'Hotel California', year: 1976, genres: ['Rock', 'Classic Rock'], rating: 94 },
    { title: 'Stairway to Heaven', artist: 'Led Zeppelin', album: 'Led Zeppelin IV', year: 1971, genres: ['Rock', 'Hard Rock'], rating: 98 },
    { title: 'Imagine', artist: 'John Lennon', album: 'Imagine', year: 1971, genres: ['Pop', 'Soft Rock'], rating: 93 },
    { title: 'Billie Jean', artist: 'Michael Jackson', album: 'Thriller', year: 1982, genres: ['Pop', 'R&B'], rating: 95 },
    { title: 'Like a Rolling Stone', artist: 'Bob Dylan', album: 'Highway 61 Revisited', year: 1965, genres: ['Folk Rock', 'Rock'], rating: 97 },
    { title: 'Smells Like Teen Spirit', artist: 'Nirvana', album: 'Nevermind', year: 1991, genres: ['Grunge', 'Alternative Rock'], rating: 92 },
    { title: 'What\'s Going On', artist: 'Marvin Gaye', album: 'What\'s Going On', year: 1971, genres: ['Soul', 'R&B'], rating: 94 },
    { title: 'Purple Rain', artist: 'Prince', album: 'Purple Rain', year: 1984, genres: ['Rock', 'Pop'], rating: 93 },
    { title: 'Good Vibrations', artist: 'The Beach Boys', album: 'Smiley Smile', year: 1966, genres: ['Pop', 'Psychedelic'], rating: 91 },
    { title: 'Yesterday', artist: 'The Beatles', album: 'Help!', year: 1965, genres: ['Pop', 'Baroque Pop'], rating: 89 },
    { title: 'Respect', artist: 'Aretha Franklin', album: 'I Never Loved a Man', year: 1967, genres: ['Soul', 'R&B'], rating: 92 },
    { title: 'Johnny B. Goode', artist: 'Chuck Berry', album: 'Chuck Berry Is on Top', year: 1958, genres: ['Rock and Roll', 'Blues'], rating: 90 },
    { title: 'Superstition', artist: 'Stevie Wonder', album: 'Talking Book', year: 1972, genres: ['Funk', 'Soul'], rating: 91 },
    { title: 'The Sound of Silence', artist: 'Simon & Garfunkel', album: 'Sounds of Silence', year: 1964, genres: ['Folk', 'Pop'], rating: 88 }
  ]

  // Filter tracks based on year range
  const filteredTracks = mockTracks.filter(track => 
    track.year >= filters.minYear && track.year <= filters.maxYear
  )

  // Generate recommendations based on algorithm
  const recommendations: Recommendation[] = []
  const tracksToUse = filteredTracks.length > 0 ? filteredTracks : mockTracks
  
  for (let i = 0; i < 10; i++) {
    const baseTrack = tracksToUse[i % tracksToUse.length]
    
    // Algorithm-specific scoring
    let collaborative = 0.6 + Math.random() * 0.3
    let content = 0.6 + Math.random() * 0.3
    let hybrid = (collaborative + content) / 2 + Math.random() * 0.1
    
    if (algorithm === 'collaborative') {
      collaborative += 0.1
      hybrid = collaborative * 0.7 + content * 0.3
    } else if (algorithm === 'content') {
      content += 0.1
      hybrid = collaborative * 0.3 + content * 0.7
    } else {
      hybrid += 0.1
    }

    // Generate algorithm-specific reasons
    const algorithmReasons = {
      collaborative: [
        'Popular among users with similar listening history',
        'Highly rated by users who liked your recent favorites',
        'Trending among users with similar taste profiles'
      ],
      content: [
        'Audio features match your preferred sound signature',
        'Genre and mood align with your listening patterns',
        'Similar musical complexity to your liked tracks'
      ],
      hybrid: [
        'Perfect blend of collaborative and content signals',
        'Optimized recommendation from neural network fusion',
        'High confidence from multi-modal ML analysis'
      ]
    }

    recommendations.push({
      id: `mock_${i}`,
      title: baseTrack.title,
      artist: baseTrack.artist,
      album: baseTrack.album,
      rating: baseTrack.rating + Math.floor(Math.random() * 6 - 3), // ±3 variance
      year: baseTrack.year,
      genre: baseTrack.genres,
      cover: `https://picsum.photos/300/300?random=${i}`,
      spotifyUrl: `https://open.spotify.com/track/mock_${i}`,
      confidence: Math.min(0.99, hybrid),
      explanation: {
        collaborative: Math.min(0.99, collaborative),
        content: Math.min(0.99, content),
        hybrid: Math.min(0.99, hybrid),
        reasons: algorithmReasons[algorithm]
      },
      audioFeatures: {
        danceability: Math.random(),
        energy: filters.energyRange[0]/100 + Math.random() * (filters.energyRange[1] - filters.energyRange[0])/100,
        valence: Math.random(),
        acousticness: Math.random()
      }
    })
  }

  return recommendations.sort((a, b) => b.confidence - a.confidence)
}

export function useRecommendations(user: User | null) {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [generationStage, setGenerationStage] = useState('')
  const [filters, setFilters] = useState<RecommendationFilters>(defaultFilters)
  const [activeAlgorithm, setActiveAlgorithm] = useState<'collaborative' | 'content' | 'hybrid'>('hybrid')

  const generateRecommendations = useCallback(async () => {
    if (!user?.id) return

    setIsGenerating(true)
    setRecommendations([])

    // Check for mock mode (if NEXT_PUBLIC_MOCK_ML is set or localStorage flag)
    const mockMode = process.env.NEXT_PUBLIC_MOCK_ML === 'true' || 
                     localStorage.getItem('timbre-mock-ml') === 'true'

    try {
      if (mockMode) {
        // Mock mode: Generate more realistic recommendations with full loading experience
        await new Promise(resolve => setTimeout(resolve, 15000)) // 15 second delay for testing

        // Generate varied mock recommendations based on algorithm
        const mockRecs = generateMockRecommendations(activeAlgorithm, filters)
        setRecommendations(mockRecs)
      } else {
        // Production mode: Actual ML API calls
        // Magical loading animation stages
        const stages = [
          'Analyzing your musical DNA...',
          'Scanning collaborative patterns...',
          'Processing content features...',
          'Applying hybrid algorithms...',
          'Curating perfect matches...',
          'Finalizing recommendations...'
        ]

        // Show loading stages
        for (let i = 0; i < stages.length; i++) {
          setGenerationStage(stages[i])
          await new Promise(resolve => setTimeout(resolve, 800))
        }

        // Prepare ML request filters
        const mlFilters = {
          genres: filters.genres,
          min_year: filters.minYear,
          max_year: filters.maxYear,
          min_rating: filters.minRating,
          energy_range: [filters.energyRange[0] / 100, filters.energyRange[1] / 100] as [number, number],
          danceability_range: [filters.danceabilityRange[0] / 100, filters.danceabilityRange[1] / 100] as [number, number],
          include_mood_analysis: filters.includeMoodAnalysis,
          diversity_weight: filters.diversityWeight,
          popularity_bias: filters.popularityBias
        }

        // Call ML API
        const mlTracks = await RecommendationService.getMLRecommendationsAsUnifiedTracks(
          user.id,
          activeAlgorithm,
          10,
          mlFilters
        )

        // Convert to recommendation format
        const recs: Recommendation[] = mlTracks.map(track => ({
          id: track.id,
          title: track.name,
          artist: track.artist,
          album: track.album,
          rating: track.popularity || track.confidence_score ? Math.round((track.confidence_score || 0) * 100) : 85,
          year: parseInt(track.release_date || '2020'),
          genre: track.genres || ['Unknown'],
          cover: track.artwork_url || 'https://via.placeholder.com/80x80/333/fff?text=♪',
          spotifyUrl: track.spotify_url,
          confidence: track.confidence_score || track.similarity_score || 0.8,
          explanation: {
            collaborative: 0.7,
            content: 0.7,
            hybrid: track.confidence_score || 0.8,
            reasons: [track.recommendation_reason || 'AI recommended']
          },
          audioFeatures: track.audio_features ? {
            danceability: track.audio_features.danceability ?? 0.5,
            energy: track.audio_features.energy ?? 0.5,
            valence: track.audio_features.valence ?? 0.5,
            acousticness: track.audio_features.acousticness ?? 0.5
          } : undefined
        }))

        setRecommendations(recs.length > 0 ? recs : mockRecommendations)
      }
    } catch (error) {
      console.error('Failed to generate recommendations:', error)
      // Always fallback to mock data
      const mockRecs = generateMockRecommendations(activeAlgorithm, filters)
      setRecommendations(mockRecs)
    } finally {
      setIsGenerating(false)
      setGenerationStage('')
    }
  }, [user?.id, filters, activeAlgorithm])

  const submitFeedback = useCallback(async (recommendationId: string, rating: number) => {
    if (!user?.id) return

    try {
      await RecommendationService.submitMLFeedback(user.id, recommendationId, rating, 'like')
    } catch (error) {
      console.error('Failed to submit feedback:', error)
    }
  }, [user?.id])

  return {
    recommendations,
    isGenerating,
    generationStage,
    filters,
    setFilters,
    activeAlgorithm,
    setActiveAlgorithm,
    generateRecommendations,
    submitFeedback
  }
}