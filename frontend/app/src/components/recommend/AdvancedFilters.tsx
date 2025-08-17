import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Filter } from 'lucide-react'
import type { RecommendationFilters } from '@/hooks/useRecommendations'

interface AdvancedFiltersProps {
  filters: RecommendationFilters
  onFiltersChange: (filters: RecommendationFilters) => void
}

export function AdvancedFilters({ filters, onFiltersChange }: AdvancedFiltersProps) {
  const updateFilters = (updates: Partial<RecommendationFilters>) => {
    onFiltersChange({ ...filters, ...updates })
  }

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Filter className="h-5 w-5" />
          Advanced Filters
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Year Range */}
        <div>
          <label className="text-sm font-medium mb-2 block">Release Year Range</label>
          <div className="px-2">
            <Slider
              value={[filters.minYear, filters.maxYear]}
              onValueChange={([min, max]) => updateFilters({ minYear: min, maxYear: max })}
              min={1950}
              max={new Date().getFullYear()}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>{filters.minYear}</span>
              <span>{filters.maxYear}</span>
            </div>
          </div>
        </div>

        {/* Energy Range */}
        <div>
          <label className="text-sm font-medium mb-2 block">Energy Level</label>
          <div className="px-2">
            <Slider
              value={filters.energyRange}
              onValueChange={(value: [number, number]) => updateFilters({ energyRange: value })}
              min={0}
              max={100}
              step={5}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Calm</span>
              <span>Energetic</span>
            </div>
          </div>
        </div>

        {/* Diversity vs Similarity */}
        <div>
          <label className="text-sm font-medium mb-2 block">Diversity Weight</label>
          <div className="px-2">
            <Slider
              value={[filters.diversityWeight * 100]}
              onValueChange={([value]) => updateFilters({ diversityWeight: value / 100 })}
              min={0}
              max={100}
              step={5}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Similar</span>
              <span>Diverse</span>
            </div>
          </div>
        </div>

        {/* Switches */}
        <div className="flex items-center justify-between">
          <div>
            <label className="text-sm font-medium">Include Mood Analysis</label>
            <p className="text-xs text-muted-foreground">Use BERT embeddings for emotional context</p>
          </div>
          <Switch
            checked={filters.includeMoodAnalysis}
            onCheckedChange={(checked) => updateFilters({ includeMoodAnalysis: checked })}
          />
        </div>
      </CardContent>
    </Card>
  )
}