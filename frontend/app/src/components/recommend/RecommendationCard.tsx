import { Brain, Play, ExternalLink, Heart } from 'lucide-react'
import type { Recommendation } from '@/hooks/useRecommendations'

interface RecommendationCardProps {
  recommendation: Recommendation
  onLike?: (id: string) => void
}

export function RecommendationCard({ recommendation: rec, onLike }: RecommendationCardProps) {
  return (
    <div className="bg-neutral-800 rounded-3xl p-6 shadow-xl transition-all duration-300 hover:scale-[1.02] cursor-pointer group relative">
      <div className="flex gap-4">
        {/* Album Cover */}
        <div className="relative">
          <div className="w-20 h-20 rounded-xl overflow-hidden">
            <img
              src={rec.cover}
              alt={rec.title}
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              onError={(e) => {
                const target = e.target as HTMLImageElement
                target.src = 'https://via.placeholder.com/80x80/333/fff?text=♪'
              }}
            />
          </div>
          <div className="absolute top-1 right-1 bg-neutral-700 text-white text-xs px-1.5 py-0.5 rounded-full">
            {rec.rating}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between mb-1">
            <div className="flex-1 min-w-0">
              <h3 className="font-medium text-white truncate">{rec.title}</h3>
              <p className="text-sm text-neutral-300 truncate">{rec.artist}</p>
              {rec.album && (
                <p className="text-xs text-neutral-400 truncate">{rec.album} • {rec.year}</p>
              )}
            </div>
            <div className="ml-2 bg-neutral-700 text-white text-xs px-2 py-1 rounded-full">
              {Math.round(rec.confidence * 100)}% match
            </div>
          </div>

          {/* Genres */}
          <div className="flex flex-wrap gap-1 mb-2">
            {rec.genre.map((g, idx) => (
              <div key={idx} className="bg-neutral-700 text-white text-xs px-2 py-0.5 rounded">
                {g}
              </div>
            ))}
          </div>

          {/* AI Explanation */}
          <div className="mb-2">
            <div className="flex items-center gap-2 mb-1">
              <Brain className="h-3 w-3 text-neutral-400" />
              <span className="text-xs font-medium text-neutral-400">AI Analysis</span>
            </div>
            <div className="flex gap-2 text-xs">
              <span className="text-neutral-400">
                Collaborative: <span className="text-white font-medium">{Math.round(rec.explanation.collaborative * 100)}%</span>
              </span>
              <span className="text-neutral-400">
                Content: <span className="text-white font-medium">{Math.round(rec.explanation.content * 100)}%</span>
              </span>
              <span className="text-neutral-400">
                Hybrid: <span className="text-white font-medium">{Math.round(rec.explanation.hybrid * 100)}%</span>
              </span>
            </div>
          </div>

          {/* Audio Features */}
          {rec.audioFeatures && (
            <div className="mb-2">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-neutral-400">Energy:</span>
                  <span className="font-medium text-white">{Math.round(rec.audioFeatures.energy * 100)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-400">Dance:</span>
                  <span className="font-medium text-white">{Math.round(rec.audioFeatures.danceability * 100)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-400">Mood:</span>
                  <span className="font-medium text-white">{Math.round(rec.audioFeatures.valence * 100)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-400">Acoustic:</span>
                  <span className="font-medium text-white">{Math.round(rec.audioFeatures.acousticness * 100)}%</span>
                </div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2">
            <button 
              className="flex-1 bg-neutral-700 hover:bg-neutral-600 text-white text-sm py-2 px-3 rounded-lg transition-colors duration-200 flex items-center justify-center gap-1"
              onClick={() => {
                if (rec.spotifyUrl) {
                  window.open(rec.spotifyUrl, '_blank')
                }
              }}
            >
              <Play className="h-3 w-3" />
              Preview
            </button>
            {rec.spotifyUrl && (
              <button 
                className="bg-neutral-700 hover:bg-neutral-600 text-white text-sm py-2 px-3 rounded-lg transition-colors duration-200"
                onClick={() => window.open(rec.spotifyUrl, '_blank')}
                title="Open in Spotify"
              >
                <ExternalLink className="h-3 w-3" />
              </button>
            )}
            <button 
              className="bg-neutral-700 hover:bg-neutral-600 text-white text-sm py-2 px-3 rounded-lg transition-colors duration-200"
              onClick={() => onLike?.(rec.id)}
              title="Like this recommendation"
            >
              <Heart className="h-3 w-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Explanation Reasons */}
      <div className="mt-3 pt-3 border-t border-neutral-700">
        <div className="text-xs text-neutral-400 space-y-1">
          {rec.explanation.reasons.slice(0, 2).map((reason, idx) => (
            <div key={idx} className="flex items-center gap-1">
              <div className="w-1 h-1 bg-neutral-500 rounded-full"></div>
              <span>{reason}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}