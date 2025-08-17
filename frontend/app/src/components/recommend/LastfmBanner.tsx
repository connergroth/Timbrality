import { Button } from '@/components/ui/button'
import { X } from 'lucide-react'

interface LastfmBannerProps {
  isVisible: boolean
  onConnect: () => void
  onDismiss: () => void
}

export function LastfmBanner({ isVisible, onConnect, onDismiss }: LastfmBannerProps) {
  if (!isVisible) return null

  return (
    <div className="mb-6 bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-4 shadow-2xl">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <img 
            src="/lastfm.png" 
            alt="Last.fm" 
            className="h-8 w-8 object-contain"
          />
          <div>
            <h3 className="font-semibold text-white mb-1">Connect Your Last.fm</h3>
            <p className="text-sm text-slate-300">
              Import your listening history for personalized music recommendations
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Button 
            variant="default" 
            size="sm"
            onClick={onConnect}
            className="bg-neutral-700/40 hover:bg-neutral-700/60 text-white border border-neutral-600/40"
          >
            Connect
          </Button>
          <Button 
            variant="ghost" 
            size="sm"
            onClick={onDismiss}
            className="text-slate-400 hover:text-white hover:bg-neutral-700/40"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}