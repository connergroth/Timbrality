'use client'

import { useEffect, useRef } from 'react'

export const MoodVibeAnimation = () => {
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

    // Mood energy waves
    const moodWaves: Array<{
      x: number
      y: number
      amplitude: number
      frequency: number
      phase: number
      color: string
      energy: number
      speed: number
    }> = []

    // Initialize mood waves with magenta accent over duotone base
    const moods = [
      { color: '#6366F1', energy: 0.3, name: 'ambient' },      // Indigo - low energy
      { color: '#7C3AED', energy: 0.5, name: 'chill' },        // Violet - medium-low
      { color: '#F472B6', energy: 0.7, name: 'groovy' },       // Magenta accent - medium
      { color: '#F472B6', energy: 0.8, name: 'energetic' },    // Magenta accent - high
      { color: '#F472B6', energy: 1.0, name: 'intense' }       // Magenta accent - very high
    ]

    moods.forEach((mood, i) => {
      moodWaves.push({
        x: 0,
        y: canvas.clientHeight * 0.2 + i * (canvas.clientHeight * 0.15),
        amplitude: 15 + mood.energy * 20,
        frequency: 0.02 + mood.energy * 0.03,
        phase: Math.random() * Math.PI * 2,
        color: mood.color,
        energy: mood.energy,
        speed: 1 + mood.energy * 2
      })
    })

    // Floating mood particles
    const particles: Array<{
      x: number
      y: number
      vx: number
      vy: number
      color: string
      size: number
      opacity: number
      life: number
      maxLife: number
    }> = []

    let animationTime = 0
    const startTime = performance.now()
    let lastFrameTime = 0
    const targetFPS = 60
    const frameInterval = 1000 / targetFPS

    const drawFrame = (timestamp: number) => {
      // Frame rate limiting for consistent performance
      if (timestamp - lastFrameTime < frameInterval) {
        requestAnimationFrame(drawFrame)
        return
      }
      lastFrameTime = timestamp
      
      const t = (timestamp - startTime) / 1000
      animationTime = t

      // Clear canvas with duotone gradient background
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height)
      gradient.addColorStop(0, 'rgba(99, 102, 241, 0.03)') // indigo
      gradient.addColorStop(1, 'rgba(124, 58, 237, 0.05)') // violet
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Draw mood energy waves
      moodWaves.forEach((wave, i) => {
        // Update wave position based on energy
        wave.x += wave.speed * 0.5
        
        // Reset wave when it goes off screen
        if (wave.x > canvas.clientWidth + 50) {
          wave.x = -50
        }

        // Draw wave
        ctx.save()
        ctx.globalAlpha = 0.8
        ctx.strokeStyle = wave.color
        ctx.lineWidth = 2 + wave.energy * 2
        
        ctx.beginPath()
        for (let x = 0; x < canvas.clientWidth; x += 5) { // Increased from 3 to 5 for better performance
          const waveX = x + wave.x
          const y = wave.y + Math.sin((waveX * wave.frequency) + t * wave.speed + wave.phase) * wave.amplitude
          
          if (x === 0) {
            ctx.moveTo(waveX, y)
          } else {
            ctx.lineTo(waveX, y)
          }
        }
        ctx.stroke()

        // Add glow effect
        ctx.save()
        ctx.globalAlpha = 0.3
        ctx.lineWidth = 6 + wave.energy * 3
        ctx.stroke()
        ctx.restore()
        ctx.restore()

        // Generate particles occasionally based on energy (optimized)
        if (Math.random() < wave.energy * 0.05) { // Reduced from 0.1
          const particleX = wave.x + Math.random() * canvas.clientWidth
          const particleY = wave.y + Math.sin((particleX * wave.frequency) + t * wave.speed + wave.phase) * wave.amplitude
          
          particles.push({
            x: particleX,
            y: particleY,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            color: wave.color,
            size: 2 + Math.random() * 3,
            opacity: 1,
            life: 0,
            maxLife: 80 + Math.random() * 60 // Reduced from 100 + random 100
          })
        }
      })

      // Update and draw particles
      particles.forEach((particle, index) => {
        // Update particle
        particle.x += particle.vx
        particle.y += particle.vy
        particle.life++
        particle.opacity = 1 - (particle.life / particle.maxLife)

        // Draw particle
        ctx.save()
        ctx.globalAlpha = particle.opacity
        ctx.fillStyle = particle.color
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2)
        ctx.fill()
        
        // Add glow
        ctx.globalAlpha = particle.opacity * 0.5
        ctx.shadowColor = particle.color
        ctx.shadowBlur = 8
        ctx.fill()
        ctx.restore()

        // Remove dead particles
        if (particle.life >= particle.maxLife) {
          particles.splice(index, 1)
        }
      })

      // Energy level indicators and mood labels removed - now displayed in card

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
