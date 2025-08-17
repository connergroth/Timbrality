import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Filter, X } from 'lucide-react'
import type { RecommendationFilters } from '@/hooks/useRecommendations'

interface AdvancedFiltersModalProps {
  isOpen: boolean
  onClose: () => void
  filters: RecommendationFilters
  onFiltersChange: (filters: RecommendationFilters) => void
}

export function AdvancedFiltersModal({ 
  isOpen, 
  onClose, 
  filters, 
  onFiltersChange 
}: AdvancedFiltersModalProps) {
  const updateFilters = (updates: Partial<RecommendationFilters>) => {
    onFiltersChange({ ...filters, ...updates })
  }

  const resetFilters = () => {
    onFiltersChange({
      genres: [],
      minYear: 1950,
      maxYear: new Date().getFullYear(),
      minRating: 0,
      energyRange: [0, 100],
      danceabilityRange: [0, 100],
      includeMoodAnalysis: true,
      diversityWeight: 0.7,
      popularityBias: 0.3
    })
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-md bg-neutral-900 border-neutral-700">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <div>
              <DialogTitle className="flex items-center gap-2 text-white">
                <Filter className="h-5 w-5" />
                Advanced Filters
              </DialogTitle>
              <DialogDescription className="text-neutral-300">
                Fine-tune your recommendation preferences
              </DialogDescription>
            </div>
            <Button variant="ghost" size="sm" onClick={onClose} className="text-white hover:bg-neutral-800">
              <X className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        <div className="space-y-6">
          {/* Year Range */}
          <div>
            <label className="text-sm font-medium mb-3 block text-white">Release Year Range</label>
            <div className="px-2">
              <Slider
                value={[filters.minYear, filters.maxYear]}
                onValueChange={([min, max]) => updateFilters({ minYear: min, maxYear: max })}
                min={1950}
                max={new Date().getFullYear()}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-neutral-400 mt-2">
                <span>{filters.minYear}</span>
                <span>{filters.maxYear}</span>
              </div>
            </div>
          </div>

          {/* Energy Range */}
          <div>
            <label className="text-sm font-medium mb-3 block text-white">Energy Level</label>
            <div className="px-2">
              <Slider
                value={filters.energyRange}
                onValueChange={(value: [number, number]) => updateFilters({ energyRange: value })}
                min={0}
                max={100}
                step={5}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-neutral-400 mt-2">
                <span>Calm</span>
                <span>Energetic</span>
              </div>
            </div>
          </div>

          {/* Danceability Range */}
          <div>
            <label className="text-sm font-medium mb-3 block text-white">Danceability</label>
            <div className="px-2">
              <Slider
                value={filters.danceabilityRange}
                onValueChange={(value: [number, number]) => updateFilters({ danceabilityRange: value })}
                min={0}
                max={100}
                step={5}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-neutral-400 mt-2">
                <span>Chill</span>
                <span>Danceable</span>
              </div>
            </div>
          </div>

          {/* Diversity vs Similarity */}
          <div>
            <label className="text-sm font-medium mb-3 block text-white">Diversity Weight</label>
            <div className="px-2">
              <Slider
                value={[filters.diversityWeight * 100]}
                onValueChange={([value]) => updateFilters({ diversityWeight: value / 100 })}
                min={0}
                max={100}
                step={5}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-neutral-400 mt-2">
                <span>Similar</span>
                <span>Diverse</span>
              </div>
            </div>
          </div>

          {/* Popularity Bias */}
          <div>
            <label className="text-sm font-medium mb-3 block text-white">Popularity Preference</label>
            <div className="px-2">
              <Slider
                value={[filters.popularityBias * 100]}
                onValueChange={([value]) => updateFilters({ popularityBias: value / 100 })}
                min={0}
                max={100}
                step={5}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-neutral-400 mt-2">
                <span>Hidden Gems</span>
                <span>Popular Hits</span>
              </div>
            </div>
          </div>

          {/* Mood Analysis Switch */}
          <div className="flex items-center justify-between py-2">
            <div>
              <label className="text-sm font-medium text-white">Include Mood Analysis</label>
              <p className="text-xs text-neutral-400">Use BERT embeddings for emotional context</p>
            </div>
            <Switch
              checked={filters.includeMoodAnalysis}
              onCheckedChange={(checked) => updateFilters({ includeMoodAnalysis: checked })}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 pt-4">
            <Button variant="outline" onClick={resetFilters} className="flex-1 border-neutral-600 text-white hover:bg-neutral-800">
              Reset to Defaults
            </Button>
            <Button onClick={onClose} className="flex-1 bg-white text-black hover:bg-neutral-100">
              Apply Filters
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}