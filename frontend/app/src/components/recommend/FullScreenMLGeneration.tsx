'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'

interface FullScreenMLGenerationProps {
  isVisible: boolean
  onComplete: () => void
  recommendations: any[]
}

export function FullScreenMLGeneration({ isVisible, onComplete, recommendations }: FullScreenMLGenerationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const timeRef = useRef(0)
  const [currentStage, setCurrentStage] = useState(0)
  const router = useRouter()

  const stages = [
    "Initializing Neural Networks...",
    "Analyzing Your Musical DNA...",
    "Computing Collaborative Matrices...",
    "Processing Audio Feature Vectors...",
    "Applying BERT Embeddings...",
    "Fusing Recommendation Signals...",
    "Optimizing Similarity Scores...",
    "Calibrating Diversity Weights...",
    "Finalizing Recommendations..."
  ]

  useEffect(() => {
    if (!isVisible) return

    // Stage progression
    const stageInterval = setInterval(() => {
      setCurrentStage(prev => {
        if (prev < stages.length - 1) {
          return prev + 1
        } else {
          // After all stages, check if we have recommendations
          setTimeout(() => {
            if (recommendations.length > 0) {
              // Navigate to results page
              router.push('/recommend/results')
              onComplete()
            }
          }, 1000)
          return prev
        }
      })
    }, 800)

    return () => clearInterval(stageInterval)
  }, [isVisible, recommendations.length, router, onComplete])

  useEffect(() => {
    if (!isVisible) return
    
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas to full screen
    const updateCanvasSize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    updateCanvasSize()
    window.addEventListener('resize', updateCanvasSize)

    // Enhanced ML nodes for full screen
    const mlNodes: Array<{
      x: number
      y: number
      vx: number
      vy: number
      radius: number
      opacity: number
      pulsePhase: number
    }> = []

    // Enhanced music notes
    const musicNotes: Array<{
      x: number
      y: number
      vx: number
      vy: number
      type: string
      opacity: number
      rotation: number
      rotationSpeed: number
      size: number
    }> = []

    // Initialize more nodes for full screen
    for (let i = 0; i < 40; i++) {
      mlNodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: Math.random() * 6 + 2,
        opacity: Math.random() * 0.8 + 0.2,
        pulsePhase: Math.random() * Math.PI * 2
      })
    }

    // Initialize more music notes
    const noteSymbols = ['♪', '♫', '♬', '♩', '♭', '♯', '𝄞', '𝄢']
    for (let i = 0; i < 20; i++) {
      musicNotes.push({
        x: -100,
        y: Math.random() * canvas.height,
        vx: Math.random() * 1.5 + 0.5,
        vy: (Math.random() - 0.5) * 0.3,
        type: noteSymbols[Math.floor(Math.random() * noteSymbols.length)],
        opacity: Math.random() * 0.7 + 0.3,
        rotation: Math.random() * 360,
        rotationSpeed: (Math.random() - 0.5) * 2,
        size: Math.random() * 20 + 20
      })
    }

    const animate = () => {
      timeRef.current += 0.016

      // Dynamic background based on stage
      const progress = currentStage / (stages.length - 1)
      const hue = 220 + progress * 60 // From blue to purple
      
      const gradient = ctx.createRadialGradient(
        canvas.width / 2, canvas.height / 2, 0,
        canvas.width / 2, canvas.height / 2, Math.max(canvas.width, canvas.height) / 2
      )
      gradient.addColorStop(0, `hsla(${hue}, 30%, 5%, 0.95)`)
      gradient.addColorStop(0.5, `hsla(${hue + 20}, 40%, 8%, 0.98)`)
      gradient.addColorStop(1, `hsla(${hue + 40}, 50%, 3%, 0.95)`)
      
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Enhanced ML nodes with pulsing
      mlNodes.forEach((node, i) => {
        // Update position
        node.x += node.vx
        node.y += node.vy

        // Boundary bounce
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1

        // Complex pulsing with individual phases
        const pulse = Math.sin(timeRef.current * 3 + node.pulsePhase) * 0.5 + 0.5
        node.opacity = 0.3 + 0.6 * pulse

        // Draw enhanced node with glow
        ctx.save()
        ctx.globalAlpha = node.opacity
        
        // Glow effect
        const glowGradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius * 3)
        glowGradient.addColorStop(0, `hsl(${hue + Math.sin(timeRef.current + i) * 30}, 70%, 60%)`)
        glowGradient.addColorStop(1, 'transparent')
        ctx.fillStyle = glowGradient
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius * 3, 0, Math.PI * 2)
        ctx.fill()
        
        // Core node
        ctx.fillStyle = `hsl(${hue + Math.sin(timeRef.current + i) * 30}, 80%, 70%)`
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
        ctx.fill()
        ctx.restore()

        // Enhanced connections
        mlNodes.forEach((otherNode, j) => {
          if (i !== j) {
            const dx = node.x - otherNode.x
            const dy = node.y - otherNode.y
            const distance = Math.sqrt(dx * dx + dy * dy)

            if (distance < 150) {
              const opacity = (150 - distance) / 150 * 0.3 * pulse
              ctx.save()
              ctx.globalAlpha = opacity
              ctx.strokeStyle = `hsl(${hue + 20}, 60%, 70%)`
              ctx.lineWidth = 2
              ctx.beginPath()
              ctx.moveTo(node.x, node.y)
              ctx.lineTo(otherNode.x, otherNode.y)
              ctx.stroke()
              ctx.restore()
            }
          }
        })
      })

      // Enhanced music notes
      musicNotes.forEach((note, i) => {
        // Update position with wave motion
        note.x += note.vx
        note.y += note.vy + Math.sin(timeRef.current * 2 + i) * 0.1
        note.rotation += note.rotationSpeed

        // Reset when off screen
        if (note.x > canvas.width + 100) {
          note.x = -100
          note.y = Math.random() * canvas.height
        }

        // Dynamic opacity
        note.opacity = 0.4 + 0.4 * Math.sin(timeRef.current * 2 + i * 0.5)

        // Draw enhanced music note with glow
        ctx.save()
        ctx.globalAlpha = note.opacity
        
        // Glow
        ctx.shadowBlur = 20
        ctx.shadowColor = `hsl(${280 + Math.sin(timeRef.current + i) * 20}, 60%, 70%)`
        
        ctx.fillStyle = `hsl(${280 + Math.sin(timeRef.current + i) * 20}, 70%, 75%)`
        ctx.font = `${note.size + Math.sin(timeRef.current + i) * 4}px serif`
        ctx.textAlign = 'center'
        ctx.translate(note.x, note.y)
        ctx.rotate((note.rotation * Math.PI) / 180)
        ctx.fillText(note.type, 0, 0)
        ctx.restore()
      })

      // Enhanced data flow visualization
      ctx.save()
      ctx.globalAlpha = 0.4
      
      for (let i = 0; i < 12; i++) {
        const y = (canvas.height / 12) * i + 100
        const waveOffset = Math.sin(timeRef.current * 1.5 + i * 0.3) * 30
        
        ctx.strokeStyle = `hsl(${hue + 40}, 60%, 50%)`
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(0, y + waveOffset)
        
        for (let x = 0; x < canvas.width; x += 25) {
          const wave = Math.sin((x + timeRef.current * 60) * 0.01) * 15
          ctx.lineTo(x, y + waveOffset + wave)
        }
        
        ctx.stroke()
      }
      ctx.restore()

      // Particle system
      if (Math.random() < 0.1) {
        ctx.save()
        ctx.globalAlpha = 0.6
        ctx.fillStyle = `hsl(${hue + Math.random() * 60}, 70%, 60%)`
        ctx.beginPath()
        ctx.arc(Math.random() * canvas.width, Math.random() * canvas.height, Math.random() * 3 + 1, 0, Math.PI * 2)
        ctx.fill()
        ctx.restore()
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener('resize', updateCanvasSize)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isVisible, currentStage])

  if (!isVisible) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
      />
      
      {/* Text Overlay */}
      <div className="relative z-10 text-center max-w-2xl mx-auto px-6">
        <div className="mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 mb-6 rounded-full bg-white/10 backdrop-blur-sm">
            <div className="w-8 h-8 border-4 border-white/30 border-t-white rounded-full animate-spin"></div>
          </div>
          
          <h1 className="text-4xl md:text-6xl font-playfair font-bold text-white mb-6 tracking-tight">
            AI Processing
          </h1>
          
          <div className="h-16 flex items-center justify-center">
            <p className="text-xl md:text-2xl text-white/90 font-medium transition-all duration-500">
              {stages[currentStage]}
            </p>
          </div>
        </div>
        
        {/* Progress indicator */}
        <div className="w-full max-w-md mx-auto">
          <div className="w-full bg-white/20 rounded-full h-2 mb-4">
            <div 
              className="bg-white h-2 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${((currentStage + 1) / stages.length) * 100}%` }}
            ></div>
          </div>
          <p className="text-sm text-white/70">
            Step {currentStage + 1} of {stages.length}
          </p>
        </div>
      </div>
    </div>
  )
}