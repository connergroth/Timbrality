'use client'

import { useEffect, useRef } from 'react'

export const MLGraphAnimation = () => {
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

    // Neural network nodes
    const nodes: Array<{
      x: number
      y: number
      radius: number
      color: string
      pulse: number
      connections: number[]
    }> = []

    // Initialize neural network structure
    const layers = [4, 6, 6, 4] // Input, hidden1, hidden2, output
    let nodeIndex = 0
    
    layers.forEach((layerSize, layerIndex) => {
      const layerWidth = canvas.clientWidth / (layers.length + 1)
      const x = layerWidth * (layerIndex + 1)
      
      for (let i = 0; i < layerSize; i++) {
        const y = (canvas.clientHeight / (layerSize + 1)) * (i + 1)
        const color = layerIndex === 0 ? '#6366f1' : // Input layer - indigo
                     layerIndex === layers.length - 1 ? '#5b21b6' : // Output layer - purple
                     '#8b5cf6' // Hidden layers - violet
        
        nodes.push({
          x,
          y,
          radius: 4 + Math.random() * 3,
          color,
          pulse: Math.random() * Math.PI * 2,
          connections: []
        })
        
        // Connect to next layer
        if (layerIndex < layers.length - 1) {
          const nextLayerStart = nodeIndex + layerSize
          const nextLayerSize = layers[layerIndex + 1]
          
          for (let j = 0; j < nextLayerSize; j++) {
            nodes[nodeIndex].connections.push(nextLayerStart + j)
          }
        }
        
        nodeIndex++
      }
    })

    // Data particles flowing through the network
    const particles: Array<{
      x: number
      y: number
      vx: number
      vy: number
      targetX: number
      targetY: number
      color: string
      opacity: number
      size: number
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

      // Clear canvas with gradient background
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height)
      gradient.addColorStop(0, 'rgba(99, 102, 241, 0.05)') // indigo
      gradient.addColorStop(0.5, 'rgba(91, 33, 182, 0.08)') // purple
      gradient.addColorStop(1, 'rgba(139, 92, 246, 0.05)') // violet
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Update and draw nodes
      nodes.forEach((node, i) => {
        // Pulsing effect
        node.pulse += 0.05
        const pulseScale = 1 + 0.2 * Math.sin(node.pulse)
        
        // Draw node
        ctx.save()
        ctx.globalAlpha = 0.9
        ctx.fillStyle = node.color
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius * pulseScale, 0, Math.PI * 2)
        ctx.fill()
        
        // Add glow effect
        ctx.globalAlpha = 0.3
        ctx.shadowColor = node.color
        ctx.shadowBlur = 10
        ctx.fill()
        ctx.restore()

        // Draw connections to next layer
        node.connections.forEach(targetIndex => {
          const target = nodes[targetIndex]
          if (target) {
            const distance = Math.sqrt((node.x - target.x) ** 2 + (node.y - target.y) ** 2)
            const opacity = Math.max(0.1, 0.4 - distance / (canvas.width * 0.8))
            
            ctx.save()
            ctx.globalAlpha = opacity
            ctx.strokeStyle = node.color
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(node.x, node.y)
            ctx.lineTo(target.x, target.y)
            ctx.stroke()
            ctx.restore()
          }
        })
      })

      // Generate new particles occasionally
      if (Math.random() < 0.05) { // Reduced from 0.1
        const inputNode = nodes[Math.floor(Math.random() * layers[0])]
        if (inputNode) {
          particles.push({
            x: inputNode.x,
            y: inputNode.y,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            targetX: inputNode.x,
            targetY: inputNode.y,
            color: inputNode.color,
            opacity: 1,
            size: 2 + Math.random() * 2
          })
        }
      }

      // Update and draw particles
      particles.forEach((particle, index) => {
        // Move particle towards target
        const dx = particle.targetX - particle.x
        const dy = particle.targetY - particle.y
        particle.x += dx * 0.02
        particle.y += dy * 0.02
        
        // Update target to next node in path
        if (Math.abs(dx) < 5 && Math.abs(dy) < 5) {
          const currentNode = nodes.find(node => 
            Math.abs(node.x - particle.x) < 10 && Math.abs(node.y - particle.y) < 10
          )
          if (currentNode && currentNode.connections.length > 0) {
            const nextNode = nodes[currentNode.connections[Math.floor(Math.random() * currentNode.connections.length)]]
            if (nextNode) {
              particle.targetX = nextNode.x
              particle.targetY = nextNode.y
              particle.color = nextNode.color
            }
          }
        }

        // Fade out particles that have traveled far
        const distanceFromStart = Math.sqrt((particle.x - nodes[0].x) ** 2 + (particle.y - nodes[0].y) ** 2)
        if (distanceFromStart > canvas.width * 0.8) {
          particle.opacity *= 0.95
        }

        // Draw particle
        ctx.save()
        ctx.globalAlpha = particle.opacity
        ctx.fillStyle = particle.color
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2)
        ctx.fill()
        
        // Add glow (optimized - only for visible particles)
        if (particle.opacity > 0.3) {
          ctx.globalAlpha = particle.opacity * 0.5
          ctx.shadowColor = particle.color
          ctx.shadowBlur = 8
          ctx.fill()
        }
        ctx.restore()

        // Remove faded particles
        if (particle.opacity < 0.1) {
          particles.splice(index, 1)
        }
      })

      // Draw matrix-like falling characters (reduced frequency)
      if (Math.random() < 0.02) { // Reduced from 0.05
        ctx.save()
        ctx.globalAlpha = 0.15
        ctx.fillStyle = '#6366f1'
        ctx.font = '12px monospace'
        
        const chars = '01'
        const char = chars[Math.floor(Math.random() * chars.length)]
        const x = Math.random() * canvas.width
        const y = Math.random() * canvas.height
        
        ctx.fillText(char, x, y)
        ctx.restore()
      }

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
