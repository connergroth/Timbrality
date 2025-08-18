'use client'

import { useEffect, useRef } from 'react'

export const AudioWaveformAnimation = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Setup canvas with device pixel ratio
    const fitDPR = () => {
      const dpr = Math.max(1, window.devicePixelRatio || 1)
      const cssW = canvas.clientWidth
      const cssH = canvas.clientHeight
      canvas.width = Math.round(cssW * dpr)
      canvas.height = Math.round(cssH * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    fitDPR()

    // Audio waveform data points
    const waveforms: Array<{
      x: number
      y: number
      amplitude: number
      frequency: number
      phase: number
      color: string
    }> = []

    // Initialize audio waveforms
    for (let i = 0; i < 6; i++) {
      waveforms.push({
        x: 0,
        y: canvas.clientHeight * 0.2 + i * (canvas.clientHeight * 0.12),
        amplitude: Math.random() * 20 + 15,
        frequency: Math.random() * 0.03 + 0.02,
        phase: Math.random() * Math.PI * 2,
        color: i % 2 === 0 ? '#6366F1' : '#7C3AED' // duotone indigo/violet
      })
    }

    let animationTime = 0
    const startTime = performance.now()

    const drawFrame = (timestamp: number) => {
      const t = (timestamp - startTime) / 1000
      animationTime = t

      // Clear canvas with transparent background
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw audio waveforms (full width)
      ctx.save()
      ctx.globalAlpha = 0.8
      waveforms.forEach((wave, i) => {
        ctx.strokeStyle = wave.color
        ctx.lineWidth = 2
        
        ctx.beginPath()
        for (let x = 0; x < canvas.clientWidth; x += 2) {
          const y = wave.y + Math.sin(x * wave.frequency + t * 2 + wave.phase) * wave.amplitude
          if (x === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        }
        ctx.stroke()

        // Add glow effect
        ctx.save()
        ctx.globalAlpha = 0.3
        ctx.lineWidth = 5
        ctx.stroke()
        ctx.restore()

        // Waveform labels removed
      })
      ctx.restore()

      // Purple data flow lines removed

      requestAnimationFrame(drawFrame)
    }

    requestAnimationFrame(drawFrame)

    const handleResize = () => fitDPR()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  return (
    <canvas 
      ref={canvasRef}
      className="w-full h-48 rounded-2xl"
      style={{ width: '100%', height: '192px' }}
    />
  )
}
