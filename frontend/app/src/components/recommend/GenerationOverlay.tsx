import { Sparkles } from 'lucide-react'

interface GenerationOverlayProps {
  isVisible: boolean
  stage: string
}

const stages = [
  'Analyzing your musical DNA...',
  'Scanning collaborative patterns...',
  'Processing content features...',
  'Applying hybrid algorithms...',
  'Curating perfect matches...',
  'Finalizing recommendations...'
]

export function GenerationOverlay({ isVisible, stage }: GenerationOverlayProps) {
  if (!isVisible) return null

  return (
    <div className="fixed inset-0 bg-background/90 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="text-center max-w-md mx-auto px-6">
        {/* Magical Animation */}
        <div className="relative mb-8">
          <div className="w-32 h-32 mx-auto relative">
            {/* Outer ring */}
            <div className="absolute inset-0 rounded-full border-4 border-primary/20 animate-spin" style={{ animationDuration: '3s' }}></div>
            
            {/* Middle ring */}
            <div className="absolute inset-4 rounded-full border-4 border-primary/40 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '2s' }}></div>
            
            {/* Inner ring */}
            <div className="absolute inset-8 rounded-full border-4 border-primary/60 animate-spin" style={{ animationDuration: '1.5s' }}></div>
            
            {/* Center icon */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <Sparkles className="h-8 w-8 text-primary animate-pulse" />
            </div>
            
            {/* Floating particles */}
            <div className="absolute top-2 left-1/2 w-2 h-2 bg-primary rounded-full animate-ping" style={{ animationDelay: '0s' }}></div>
            <div className="absolute top-1/2 right-2 w-1.5 h-1.5 bg-primary rounded-full animate-ping" style={{ animationDelay: '0.5s' }}></div>
            <div className="absolute bottom-4 left-4 w-1 h-1 bg-primary rounded-full animate-ping" style={{ animationDelay: '1s' }}></div>
          </div>
        </div>

        {/* Text */}
        <h2 className="text-2xl font-playfair font-semibold mb-4">Crafting Your Perfect Playlist</h2>
        <p className="text-muted-foreground mb-6">{stage}</p>
        
        {/* Progress bar */}
        <div className="w-full bg-muted rounded-full h-2 mb-4">
          <div 
            className="bg-primary h-2 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${(stage ? (stages.indexOf(stage) + 1) / stages.length : 0) * 100}%` }}
          ></div>
        </div>
        
        <p className="text-xs text-muted-foreground">
          AI is analyzing millions of songs to find your perfect matches...
        </p>
      </div>
    </div>
  )
}