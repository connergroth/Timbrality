'use client'

import { useEffect, useRef } from 'react'

export function MLAnimationCard() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const animationRef = useRef<number>()
  const timeRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size to match container
    const updateCanvasSize = () => {
      const rect = container.getBoundingClientRect()
      canvas.width = rect.width
      canvas.height = rect.height
    }
    
    updateCanvasSize()
    window.addEventListener('resize', updateCanvasSize)

    // ML Animation particles (scaled for larger space)
    const mlNodes: Array<{
      x: number
      y: number
      vx: number
      vy: number
      radius: number
      opacity: number
    }> = []

    // Music notes (scaled for larger space)
    const musicNotes: Array<{
      x: number
      y: number
      vx: number
      vy: number
      type: string
      opacity: number
      rotation: number
      rotationSpeed: number
    }> = []

    // Initialize ML nodes (more for larger space)
    for (let i = 0; i < 20; i++) {
      mlNodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.8,
        vy: (Math.random() - 0.5) * 0.8,
        radius: Math.random() * 6 + 3,
        opacity: Math.random() * 0.3 + 0.2
      })
    }

    // Initialize music notes (more for larger space)
    const noteSymbols = ['♪', '♫', '♬', '♩', '♭', '♯']
    for (let i = 0; i < 12; i++) {
      musicNotes.push({
        x: -50,
        y: Math.random() * canvas.height,
        vx: Math.random() * 1.2 + 0.8,
        vy: (Math.random() - 0.5) * 0.3,
        type: noteSymbols[Math.floor(Math.random() * noteSymbols.length)],
        opacity: Math.random() * 0.4 + 0.3,
        rotation: Math.random() * 360,
        rotationSpeed: (Math.random() - 0.5) * 0.8
      })
    }

    const animate = () => {
      timeRef.current += 0.016

      // Clear canvas with gradient
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height)
      gradient.addColorStop(0, 'rgba(15, 15, 25, 0.95)')
      gradient.addColorStop(0.5, 'rgba(30, 20, 60, 0.98)')
      gradient.addColorStop(1, 'rgba(20, 15, 40, 0.95)')
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Update and draw ML nodes
      mlNodes.forEach((node, i) => {
        // Update position
        node.x += node.vx
        node.y += node.vy

        // Boundary bounce
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1

        // Pulsing opacity
        node.opacity = 0.2 + 0.3 * Math.sin(timeRef.current * 2 + i * 0.5)

        // Draw node
        ctx.save()
        ctx.globalAlpha = node.opacity
        ctx.fillStyle = `hsl(${220 + Math.sin(timeRef.current + i) * 30}, 70%, 60%)`
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
        ctx.fill()
        ctx.restore()

        // Draw connections to nearby nodes
        mlNodes.forEach((otherNode, j) => {
          if (i !== j) {
            const dx = node.x - otherNode.x
            const dy = node.y - otherNode.y
            const distance = Math.sqrt(dx * dx + dy * dy)

            if (distance < 150) {
              const opacity = (150 - distance) / 150 * 0.15
              ctx.save()
              ctx.globalAlpha = opacity
              ctx.strokeStyle = `hsl(${240}, 50%, 70%)`
              ctx.lineWidth = 1.5
              ctx.beginPath()
              ctx.moveTo(node.x, node.y)
              ctx.lineTo(otherNode.x, otherNode.y)
              ctx.stroke()
              ctx.restore()
            }
          }
        })
      })

      // Update and draw music notes
      musicNotes.forEach((note, i) => {
        // Update position
        note.x += note.vx
        note.y += note.vy + Math.sin(timeRef.current * 2 + i) * 0.03
        note.rotation += note.rotationSpeed

        // Reset when off screen
        if (note.x > canvas.width + 50) {
          note.x = -50
          note.y = Math.random() * canvas.height
        }

        // Floating opacity
        const baseOpacity = 0.4
        note.opacity = baseOpacity + 0.2 * Math.sin(timeRef.current * 3 + i * 0.8)

        // Draw music note
        ctx.save()
        ctx.globalAlpha = note.opacity
        ctx.fillStyle = `hsl(${280 + Math.sin(timeRef.current + i) * 20}, 60%, 70%)`
        ctx.font = `${32 + Math.sin(timeRef.current + i) * 6}px serif`
        ctx.textAlign = 'center'
        ctx.translate(note.x, note.y)
        ctx.rotate((note.rotation * Math.PI) / 180)
        ctx.fillText(note.type, 0, 0)
        ctx.restore()
      })

      // Draw subtle data flow lines
      ctx.save()
      ctx.globalAlpha = 0.2
      ctx.strokeStyle = '#4338ca'
      ctx.lineWidth = 2
      
      for (let i = 0; i < 6; i++) {
        const y = (canvas.height / 6) * i + 50
        const waveOffset = Math.sin(timeRef.current * 1.5 + i * 0.5) * 30
        
        ctx.beginPath()
        ctx.moveTo(0, y + waveOffset)
        
        for (let x = 0; x < canvas.width; x += 25) {
          const wave = Math.sin((x + timeRef.current * 40) * 0.008) * 15
          ctx.lineTo(x, y + waveOffset + wave)
        }
        
        ctx.stroke()
      }
      ctx.restore()

      // Draw matrix-like falling characters (subtle)
      if (Math.random() < 0.03) {
        ctx.save()
        ctx.globalAlpha = 0.15
        ctx.fillStyle = '#10b981'
        ctx.font = '16px monospace'
        
        const chars = '01'
        const char = chars[Math.floor(Math.random() * chars.length)]
        const x = Math.random() * canvas.width
        
        ctx.fillText(char, x, 30)
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
  }, [])

  return (
    <div ref={containerRef} className="w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full block"
        style={{
          background: 'transparent'
        }}
      />
    </div>
  )
}