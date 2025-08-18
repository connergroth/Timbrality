import { X } from 'lucide-react'

interface LastfmBannerProps {
  isVisible: boolean
  onConnect: () => void
  onDismiss: () => void
}

export function LastfmBanner({ isVisible, onConnect, onDismiss }: LastfmBannerProps) {
  if (!isVisible) return null

  return (
    <div className="mb-6 bg-neutral-800 rounded-3xl p-6 shadow-xl">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <img 
            src="/lastfm.png" 
            alt="Last.fm" 
            className="h-8 w-8 object-contain"
          />
          <div>
            <h3 className="font-semibold text-white mb-1">Connect Your Last.fm</h3>
            <p className="text-sm text-neutral-300">
              Import your listening history for personalized music recommendations
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button 
            onClick={onConnect}
            className="bg-neutral-700 hover:bg-neutral-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200"
          >
            Connect
          </button>
          <button 
            onClick={onDismiss}
            className="text-neutral-400 hover:text-white hover:bg-neutral-700 p-2 rounded-lg transition-colors duration-200"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}