import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Brain, Play, ExternalLink, Heart } from 'lucide-react'
import type { Recommendation } from '@/hooks/useRecommendations'

interface RecommendationCardProps {
  recommendation: Recommendation
  onLike?: (id: string) => void
}

export function RecommendationCard({ recommendation: rec, onLike }: RecommendationCardProps) {
  return (
    <Card className="group hover:shadow-lg transition-all duration-300 hover:scale-[1.02]">
      <CardContent className="p-3">
        <div className="flex gap-4">
          {/* Album Cover */}
          <div className="relative">
            <img
              src={rec.cover}
              alt={rec.title}
              className="w-20 h-20 rounded-lg object-cover group-hover:scale-105 transition-transform duration-300"
              onError={(e) => {
                const target = e.target as HTMLImageElement
                target.src = 'https://via.placeholder.com/80x80/333/fff?text=♪'
              }}
            />
            <div className="absolute top-1 right-1 bg-primary/80 text-primary-foreground text-xs px-1.5 py-0.5 rounded-full">
              {rec.rating}
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between mb-1">
              <div className="flex-1 min-w-0">
                <h3 className="font-medium text-foreground truncate">{rec.title}</h3>
                <p className="text-sm text-muted-foreground truncate">{rec.artist}</p>
                {rec.album && (
                  <p className="text-xs text-muted-foreground/70 truncate">{rec.album} • {rec.year}</p>
                )}
              </div>
              <Badge variant="secondary" className="ml-2">
                {Math.round(rec.confidence * 100)}% match
              </Badge>
            </div>

            {/* Genres */}
            <div className="flex flex-wrap gap-1 mb-2">
              {rec.genre.map((g, idx) => (
                <Badge key={idx} variant="outline" className="text-xs">
                  {g}
                </Badge>
              ))}
            </div>

            {/* AI Explanation */}
            <div className="mb-2">
              <div className="flex items-center gap-2 mb-1">
                <Brain className="h-3 w-3 text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">AI Analysis</span>
              </div>
              <div className="flex gap-2 text-xs">
                <span className="text-muted-foreground">
                  Collaborative: <span className="text-foreground font-medium">{Math.round(rec.explanation.collaborative * 100)}%</span>
                </span>
                <span className="text-muted-foreground">
                  Content: <span className="text-foreground font-medium">{Math.round(rec.explanation.content * 100)}%</span>
                </span>
                <span className="text-muted-foreground">
                  Hybrid: <span className="text-foreground font-medium">{Math.round(rec.explanation.hybrid * 100)}%</span>
                </span>
              </div>
            </div>

            {/* Audio Features */}
            {rec.audioFeatures && (
              <div className="mb-2">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Energy:</span>
                    <span className="font-medium">{Math.round(rec.audioFeatures.energy * 100)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Dance:</span>
                    <span className="font-medium">{Math.round(rec.audioFeatures.danceability * 100)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Mood:</span>
                    <span className="font-medium">{Math.round(rec.audioFeatures.valence * 100)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Acoustic:</span>
                    <span className="font-medium">{Math.round(rec.audioFeatures.acousticness * 100)}%</span>
                  </div>
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-2">
              <Button 
                size="sm" 
                variant="default" 
                className="flex-1"
                onClick={() => {
                  if (rec.spotifyUrl) {
                    window.open(rec.spotifyUrl, '_blank')
                  }
                }}
              >
                <Play className="h-3 w-3 mr-1" />
                Preview
              </Button>
              {rec.spotifyUrl && (
                <Button 
                  size="sm" 
                  variant="outline"
                  onClick={() => window.open(rec.spotifyUrl, '_blank')}
                  title="Open in Spotify"
                >
                  <ExternalLink className="h-3 w-3" />
                </Button>
              )}
              <Button 
                size="sm" 
                variant="outline"
                onClick={() => onLike?.(rec.id)}
                title="Like this recommendation"
              >
                <Heart className="h-3 w-3" />
              </Button>
            </div>
          </div>
        </div>

        {/* Explanation Reasons */}
        <div className="mt-2 pt-2 border-t border-border/50">
          <div className="text-xs text-muted-foreground space-y-1">
            {rec.explanation.reasons.slice(0, 2).map((reason, idx) => (
              <div key={idx} className="flex items-center gap-1">
                <div className="w-1 h-1 bg-primary rounded-full"></div>
                <span>{reason}</span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}