import { Button } from '@/components/ui/button'
import { TrendingUp, Music, Zap, Filter } from 'lucide-react'

interface AlgorithmSelectorProps {
  activeAlgorithm: 'collaborative' | 'content' | 'hybrid'
  onAlgorithmChange: (algorithm: 'collaborative' | 'content' | 'hybrid') => void
  onOpenFilters: () => void
}

export function AlgorithmSelector({ activeAlgorithm, onAlgorithmChange, onOpenFilters }: AlgorithmSelectorProps) {
  const algorithms = [
    {
      id: 'hybrid' as const,
      name: 'Hybrid AI',
      icon: Zap,
      description: 'Combines collaborative filtering and content analysis with neural network fusion for the most accurate recommendations.'
    },
    {
      id: 'collaborative' as const,
      name: 'Collaborative',
      icon: TrendingUp,
      description: 'Recommends music based on listening patterns of users with similar taste. Uses Non-negative Matrix Factorization to find latent musical preferences.'
    },
    {
      id: 'content' as const,
      name: 'Content-Based',
      icon: Music,
      description: 'Analyzes musical attributes, genres, and metadata using BERT embeddings to find songs that match your preferences.'
    }
  ]

  return (
    <div className="mb-12">
      {/* Dark Gray Card Container */}
      <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-6 shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="flex items-center gap-3 text-2xl font-playfair font-bold mb-3 text-white">
              <TrendingUp className="h-6 w-6" />
              AI Recommendation Engine
            </h2>
            <p className="text-slate-300 text-lg leading-relaxed">
              Choose your recommendation algorithm. Hybrid mode combines collaborative filtering with content analysis for optimal results.
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onOpenFilters}
            className="flex items-center gap-2 bg-neutral-700/40 border-neutral-600/40 text-white hover:bg-neutral-700/60 hover:border-neutral-600/60 backdrop-blur-sm"
          >
            <Filter className="h-4 w-4" />
            Advanced Filters
          </Button>
        </div>

        {/* Algorithm Buttons */}
        <div className="flex gap-4 mb-6">
          {algorithms.map((algo) => (
            <button
              key={algo.id}
              onClick={() => onAlgorithmChange(algo.id)}
              className={`flex items-center gap-3 px-4 py-3 rounded-2xl font-semibold transition-colors ${
                activeAlgorithm === algo.id
                  ? 'bg-white text-neutral-900'
                  : 'border border-neutral-600/40 text-neutral-300 hover:bg-neutral-700/40 hover:border-neutral-600/60 hover:text-white backdrop-blur-sm'
              }`}
            >
              <algo.icon className="h-4 w-4" />
              {algo.name}
            </button>
          ))}
        </div>

        {/* Selected Algorithm Description */}
        {activeAlgorithm && (
          <div className="p-4 bg-neutral-700/30 backdrop-blur-sm rounded-2xl border border-neutral-600/30">
            <h4 className="flex items-center gap-3 font-semibold mb-3 text-white text-lg">
              {(() => {
                const algo = algorithms.find(a => a.id === activeAlgorithm)
                const Icon = algo?.icon
                return Icon ? <Icon className="h-5 w-5" /> : null
              })()}
              {algorithms.find(a => a.id === activeAlgorithm)?.name}
            </h4>
            <p className="text-slate-300 leading-relaxed">
              {algorithms.find(a => a.id === activeAlgorithm)?.description}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}