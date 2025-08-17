'use client'

import { useEffect, useRef } from 'react'

export function MLShaderBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const timeRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const updateCanvasSize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    updateCanvasSize()
    window.addEventListener('resize', updateCanvasSize)

    // ML Animation particles (representing neural network nodes)
    const mlNodes: Array<{
      x: number
      y: number
      vx: number
      vy: number
      radius: number
      opacity: number
      connections: number[]
    }> = []

    // Music notes
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

    // Initialize ML nodes
    for (let i = 0; i < 25; i++) {
      mlNodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        radius: Math.random() * 4 + 2,
        opacity: Math.random() * 0.2 + 0.1,
        connections: []
      })
    }

    // Initialize music notes
    const noteSymbols = ['♪', '♫', '♬', '♩', '♭', '♯']
    for (let i = 0; i < 15; i++) {
      musicNotes.push({
        x: -50,
        y: Math.random() * canvas.height,
        vx: Math.random() * 0.5 + 0.2,
        vy: (Math.random() - 0.5) * 0.1,
        type: noteSymbols[Math.floor(Math.random() * noteSymbols.length)],
        opacity: Math.random() * 0.3 + 0.2,
        rotation: Math.random() * 360,
        rotationSpeed: (Math.random() - 0.5) * 0.5
      })
    }

    const animate = () => {
      timeRef.current += 0.016

      // Clear canvas with subtle gradient
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height)
      gradient.addColorStop(0, 'rgba(0, 0, 0, 0.02)')
      gradient.addColorStop(1, 'rgba(100, 50, 200, 0.01)')
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
        node.opacity = 0.1 + 0.15 * Math.sin(timeRef.current * 2 + i * 0.5)

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

            if (distance < 120) {
              const opacity = (120 - distance) / 120 * 0.08
              ctx.save()
              ctx.globalAlpha = opacity
              ctx.strokeStyle = `hsl(${240}, 50%, 70%)`
              ctx.lineWidth = 0.5
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
        note.y += note.vy + Math.sin(timeRef.current * 2 + i) * 0.02
        note.rotation += note.rotationSpeed

        // Reset when off screen
        if (note.x > canvas.width + 50) {
          note.x = -50
          note.y = Math.random() * canvas.height
        }

        // Floating opacity
        const baseOpacity = 0.2
        note.opacity = baseOpacity + 0.1 * Math.sin(timeRef.current * 3 + i * 0.8)

        // Draw music note
        ctx.save()
        ctx.globalAlpha = note.opacity
        ctx.fillStyle = `hsl(${280 + Math.sin(timeRef.current + i) * 20}, 60%, 65%)`
        ctx.font = `${24 + Math.sin(timeRef.current + i) * 4}px serif`
        ctx.textAlign = 'center'
        ctx.translate(note.x, note.y)
        ctx.rotate((note.rotation * Math.PI) / 180)
        ctx.fillText(note.type, 0, 0)
        ctx.restore()
      })

      // Draw subtle data flow lines
      ctx.save()
      ctx.globalAlpha = 0.05
      ctx.strokeStyle = '#4338ca'
      ctx.lineWidth = 1
      
      for (let i = 0; i < 8; i++) {
        const y = (canvas.height / 8) * i + 50
        const waveOffset = Math.sin(timeRef.current * 1.5 + i * 0.5) * 20
        
        ctx.beginPath()
        ctx.moveTo(0, y + waveOffset)
        
        for (let x = 0; x < canvas.width; x += 20) {
          const wave = Math.sin((x + timeRef.current * 50) * 0.01) * 10
          ctx.lineTo(x, y + waveOffset + wave)
        }
        
        ctx.stroke()
      }
      ctx.restore()

      // Draw matrix-like falling characters (subtle)
      if (Math.random() < 0.05) {
        ctx.save()
        ctx.globalAlpha = 0.1
        ctx.fillStyle = '#10b981'
        ctx.font = '12px monospace'
        
        const chars = '01'
        const char = chars[Math.floor(Math.random() * chars.length)]
        const x = Math.random() * canvas.width
        
        ctx.fillText(char, x, 20)
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
    <canvas
      ref={canvasRef}
      className="fixed inset-0 z-0 pointer-events-none opacity-10"
      style={{
        background: 'transparent'
      }}
    />
  )
}