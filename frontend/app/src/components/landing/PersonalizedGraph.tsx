'use client'

import { useEffect, useRef } from 'react'

interface GenreRing {
  label: string
  radius: number
  sweep: number
  jitter: number
  color: string
}

export const PersonalizedGraph = () => {
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

    // User's top genres with different characteristics
    const topGenres = ["Indie Folk", "Electronic", "Jazz", "Rock", "Ambient"]
    
    const palette = [
      "#ef4444", // red
      "#f59e0b", // amber  
      "#10b981", // emerald
      "#3b82f6", // blue
      "#8b5cf6", // violet
    ]

    // Create rings based on genres
    const rings: GenreRing[] = topGenres.map((genre, i) => ({
      label: genre,
      radius: 35 + i * 18,
      sweep: 0.4 + (genre.length % 7) * 0.1, // pseudo uniqueness based on genre name
      jitter: (genre.charCodeAt(0) % 5) * 0.004,
      color: palette[i % palette.length]
    }))

    let animationTime = 0
    const startTime = performance.now()

    const drawFrame = (timestamp: number) => {
      const t = (timestamp - startTime) / 1000
      animationTime = t

      // Clear canvas with dark background
      ctx.fillStyle = 'rgba(38, 38, 38, 0.4)' // neutral-800/40
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      const cx = canvas.clientWidth * 0.5
      const cy = canvas.clientHeight * 0.5

      // Draw faint polar grid
      ctx.save()
      ctx.globalAlpha = 0.1
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 1
      
      for (let r = 30; r <= 30 + rings.length * 18; r += 18) {
        ctx.beginPath()
        ctx.arc(cx, cy, r, 0, Math.PI * 2)
        ctx.stroke()
      }
      
      // Radial grid lines
      for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI * 2
        ctx.beginPath()
        ctx.moveTo(cx, cy)
        ctx.lineTo(
          cx + Math.cos(angle) * (30 + rings.length * 18),
          cy + Math.sin(angle) * (30 + rings.length * 18)
        )
        ctx.stroke()
      }
      ctx.restore()

      // Draw animated genre rings
      rings.forEach((ring, i) => {
        const baseAngle = -Math.PI * 0.5 + i * 0.6
        const wobble = Math.sin(t * 0.8 + i) * ring.jitter * 100
        const startAngle = baseAngle + wobble
        const endAngle = startAngle + ring.sweep + Math.sin(t * 0.6 + i * 1.7) * 0.2

        // Draw the arc
        ctx.beginPath()
        ctx.arc(cx, cy, ring.radius, startAngle, endAngle)
        ctx.strokeStyle = ring.color
        ctx.lineWidth = 4
        ctx.lineCap = 'round'
        ctx.stroke()

        // Add glow effect
        ctx.save()
        ctx.globalAlpha = 0.3
        ctx.lineWidth = 8
        ctx.stroke()
        ctx.restore()
      })

      // Draw genre labels and dots
      ctx.save()
      ctx.font = '11px Inter, system-ui'
      ctx.textBaseline = 'middle'
      
      rings.forEach((ring, i) => {
        const labelAngle = -Math.PI * 0.5 + i * 0.8
        const x = cx + Math.cos(labelAngle) * (ring.radius + 15)
        const y = cy + Math.sin(labelAngle) * (ring.radius + 15)

        // Draw dot
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, Math.PI * 2)
        ctx.fillStyle = ring.color
        ctx.fill()

        // Draw label with background
        const textWidth = ctx.measureText(ring.label).width
        const padding = 4
        
        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
        ctx.fillRect(x + 8 - padding, y - 6, textWidth + padding * 2, 12)
        
        // Text
        ctx.fillStyle = '#ffffff'
        ctx.fillText(ring.label, x + 8, y)
      })
      ctx.restore()

      // Draw center indicator
      ctx.beginPath()
      ctx.arc(cx, cy, 4, 0, Math.PI * 2)
      ctx.fillStyle = '#ffffff'
      ctx.fill()
      
      // Center glow
      ctx.save()
      ctx.globalAlpha = 0.5
      ctx.beginPath()
      ctx.arc(cx, cy, 8, 0, Math.PI * 2)
      ctx.fillStyle = '#ffffff'
      ctx.fill()
      ctx.restore()

      // Pulsing center ring
      const pulseRadius = 12 + Math.sin(t * 2) * 3
      ctx.beginPath()
      ctx.arc(cx, cy, pulseRadius, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
      ctx.lineWidth = 1
      ctx.stroke()

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
      className="w-full h-48 rounded-2xl bg-neutral-800/20 border border-neutral-700/20"
      style={{ width: '100%', height: '192px' }}
    />
  )
}