'use client'

import { useEffect, useState } from 'react'

interface SoundBarProps {
  className?: string
  barCount?: number
  isPlaying?: boolean
}

export function SoundBar({ 
  className = '', 
  barCount = 9, 
  isPlaying = true 
}: SoundBarProps) {
  const [heights, setHeights] = useState<number[]>([])

  useEffect(() => {
    // Initialize all bars at short baseline height
    const initialHeights = Array.from({ length: barCount }, () => 0.15)
    setHeights(initialHeights)

    if (!isPlaying) return

    // Realistic song-like animation with natural, unsynchronized movement
    const interval = setInterval(() => {
      setHeights(prev => {
        const newHeights = [...prev]
        const time = Date.now() * 0.002
        
        newHeights.forEach((_, index) => {
          // Each bar has its own rhythm and timing
          const barTime = time + index * 0.8
          const rhythm = Math.sin(barTime) * 0.4 + 0.6 // Base rhythm between 0.2 and 1.0
          
          // Add natural variation and "hits" like real music
          const hit = Math.sin(barTime * 3) > 0.7 ? 0.8 : 0 // Occasional "hits"
          const variation = Math.sin(barTime * 0.5 + index) * 0.2 // Slow variation
          
          // Combine all factors for natural movement
          let newHeight = rhythm + hit + variation
          
          // Ensure bars return to baseline when not "active"
          if (newHeight < 0.3) {
            newHeight = 0.15 + Math.random() * 0.1 // Small random baseline variation
          }
          
          newHeights[index] = Math.max(0.15, Math.min(1.0, newHeight))
        })
        
        return newHeights
      })
    }, 50) // Faster updates for smoother, more realistic movement

    return () => clearInterval(interval)
  }, [barCount, isPlaying])

  return (
    <div className={`flex items-end space-x-1 ${className}`} style={{ height: '24px' }}>
      {heights.map((height, index) => (
        <div
          key={index}
          className="bg-white rounded-full transition-all duration-150 ease-out"
          style={{
            width: '4px',
            height: `${Math.max(height * 24, 4)}px`,
            minHeight: '4px',
            maxHeight: '24px',
            transformOrigin: 'bottom',
          }}
        />
      ))}
    </div>
  )
}
