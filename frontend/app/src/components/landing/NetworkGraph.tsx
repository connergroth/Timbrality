'use client'

import { useEffect, useRef } from 'react'

interface Node {
  x: number
  y: number
  baseX: number
  baseY: number
  r: number
  phase: number
  color: string
  label?: string
  cluster: string
  pulse: number
  connections: number[]
}

interface Connection {
  from: number
  to: number
  progress: number
  delay: number
  speed: number
  strength: number
  dataFlow: number
}

interface DataParticle {
  x: number
  y: number
  vx: number
  vy: number
  color: string
  size: number
  opacity: number
  life: number
  maxLife: number
  targetX: number
  targetY: number
}

export const NetworkGraph = () => {
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

    // Create nodes representing users with different music tastes
    const nodes: Node[] = [
      // Center node (Timbrality AI) - larger and more prominent
      { 
        x: canvas.clientWidth / 2, 
        y: canvas.clientHeight / 2, 
        baseX: canvas.clientWidth / 2, 
        baseY: canvas.clientHeight / 2, 
        r: 18, 
        phase: 0, 
        color: '#ffffff', 
        label: 'AI', 
        cluster: 'center',
        pulse: 0,
        connections: []
      },
      
      // Folk/Indie cluster (left)
      { x: canvas.clientWidth * 0.25, y: canvas.clientHeight * 0.25, baseX: canvas.clientWidth * 0.25, baseY: canvas.clientHeight * 0.25, r: 10, phase: 0.5, color: '#6366F1', cluster: 'folk', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.2, y: canvas.clientHeight * 0.3, baseX: canvas.clientWidth * 0.2, baseY: canvas.clientHeight * 0.3, r: 8, phase: 1.2, color: '#6366F1', cluster: 'folk', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.3, y: canvas.clientHeight * 0.35, baseX: canvas.clientWidth * 0.3, baseY: canvas.clientHeight * 0.35, r: 9, phase: 0.8, color: '#6366F1', cluster: 'folk', pulse: 0, connections: [] },
      
      // Rock/Metal cluster (bottom-left)
      { x: canvas.clientWidth * 0.25, y: canvas.clientHeight * 0.75, baseX: canvas.clientWidth * 0.25, baseY: canvas.clientHeight * 0.75, r: 10, phase: 2.5, color: '#7C3AED', cluster: 'rock', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.2, y: canvas.clientHeight * 0.7, baseX: canvas.clientWidth * 0.2, baseY: canvas.clientHeight * 0.7, r: 8, phase: 3.1, color: '#7C3AED', cluster: 'rock', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.3, y: canvas.clientHeight * 0.8, baseX: canvas.clientWidth * 0.3, baseY: canvas.clientHeight * 0.8, r: 9, phase: 2.8, color: '#7C3AED', cluster: 'rock', pulse: 0, connections: [] },
      
      // Electronic cluster (right)
      { x: canvas.clientWidth * 0.75, y: canvas.clientHeight * 0.25, baseX: canvas.clientWidth * 0.75, baseY: canvas.clientHeight * 0.25, r: 10, phase: 1.5, color: '#7C3AED', cluster: 'electronic', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.8, y: canvas.clientHeight * 0.3, baseX: canvas.clientWidth * 0.8, baseY: canvas.clientHeight * 0.3, r: 8, phase: 2.1, color: '#7C3AED', cluster: 'electronic', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.7, y: canvas.clientHeight * 0.35, baseX: canvas.clientWidth * 0.7, baseY: canvas.clientHeight * 0.35, r: 9, phase: 1.8, color: '#7C3AED', cluster: 'electronic', pulse: 0, connections: [] },
      
      // Hip-Hop cluster (bottom-right)
      { x: canvas.clientWidth * 0.75, y: canvas.clientHeight * 0.75, baseX: canvas.clientWidth * 0.75, baseY: canvas.clientHeight * 0.75, r: 10, phase: 3.5, color: '#F472B6', cluster: 'hiphop', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.8, y: canvas.clientHeight * 0.7, baseX: canvas.clientWidth * 0.8, baseY: canvas.clientHeight * 0.7, r: 8, phase: 4.1, color: '#F472B6', cluster: 'hiphop', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.7, y: canvas.clientHeight * 0.8, baseX: canvas.clientWidth * 0.7, baseY: canvas.clientHeight * 0.8, r: 9, phase: 3.8, color: '#F472B6', cluster: 'hiphop', pulse: 0, connections: [] },
      
      // Jazz/Classical cluster (bottom-center)
      { x: canvas.clientWidth * 0.5, y: canvas.clientHeight * 0.8, baseX: canvas.clientWidth * 0.5, baseY: canvas.clientHeight * 0.8, r: 10, phase: 4.5, color: '#6366F1', cluster: 'jazz', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.45, y: canvas.clientHeight * 0.85, baseX: canvas.clientWidth * 0.45, baseY: canvas.clientHeight * 0.85, r: 8, phase: 5.1, color: '#6366F1', cluster: 'jazz', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.55, y: canvas.clientHeight * 0.85, baseX: canvas.clientWidth * 0.55, baseY: canvas.clientHeight * 0.85, r: 9, phase: 4.8, color: '#6366F1', cluster: 'jazz', pulse: 0, connections: [] },
      
      // Pop cluster (top-center)
      { x: canvas.clientWidth * 0.5, y: canvas.clientHeight * 0.2, baseX: canvas.clientWidth * 0.5, baseY: canvas.clientHeight * 0.2, r: 10, phase: 5.5, color: '#6366F1', cluster: 'pop', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.45, y: canvas.clientHeight * 0.15, baseX: canvas.clientWidth * 0.45, baseY: canvas.clientHeight * 0.15, r: 8, phase: 6.1, color: '#6366F1', cluster: 'pop', pulse: 0, connections: [] },
      { x: canvas.clientWidth * 0.55, y: canvas.clientHeight * 0.15, baseX: canvas.clientWidth * 0.55, baseY: canvas.clientHeight * 0.15, r: 9, phase: 5.8, color: '#6366F1', cluster: 'pop', pulse: 0, connections: [] }
    ]

    // Create connections from center node to cluster representatives
    const connections: Connection[] = [
      // Main hub connections to cluster leaders
      { from: 0, to: 1, progress: 0, delay: 0, speed: 0.015, strength: 1, dataFlow: 0 },
      { from: 0, to: 4, progress: 0, delay: 0.5, speed: 0.012, strength: 1, dataFlow: 0 },
      { from: 0, to: 7, progress: 0, delay: 1.0, speed: 0.018, strength: 1, dataFlow: 0 },
      { from: 0, to: 10, progress: 0, delay: 1.5, speed: 0.014, strength: 1, dataFlow: 0 },
      { from: 0, to: 13, progress: 0, delay: 2.0, speed: 0.016, strength: 1, dataFlow: 0 },
      { from: 0, to: 16, progress: 0, delay: 2.5, speed: 0.013, strength: 1, dataFlow: 0 },
      
      // Cluster internal connections
      { from: 1, to: 2, progress: 0, delay: 3.0, speed: 0.020, strength: 0.7, dataFlow: 0 },
      { from: 1, to: 3, progress: 0, delay: 3.2, speed: 0.022, strength: 0.7, dataFlow: 0 },
      { from: 4, to: 5, progress: 0, delay: 3.5, speed: 0.018, strength: 0.7, dataFlow: 0 },
      { from: 4, to: 6, progress: 0, delay: 3.8, speed: 0.025, strength: 0.7, dataFlow: 0 },
      { from: 7, to: 8, progress: 0, delay: 4.0, speed: 0.019, strength: 0.7, dataFlow: 0 },
      { from: 7, to: 9, progress: 0, delay: 4.3, speed: 0.021, strength: 0.7, dataFlow: 0 },
      { from: 10, to: 11, progress: 0, delay: 4.5, speed: 0.017, strength: 0.7, dataFlow: 0 },
      { from: 10, to: 12, progress: 0, delay: 4.8, speed: 0.023, strength: 0.7, dataFlow: 0 },
      { from: 13, to: 14, progress: 0, delay: 5.0, speed: 0.020, strength: 0.7, dataFlow: 0 },
      { from: 13, to: 15, progress: 0, delay: 5.3, speed: 0.024, strength: 0.7, dataFlow: 0 },
      { from: 16, to: 17, progress: 0, delay: 5.5, speed: 0.019, strength: 0.7, dataFlow: 0 },
      { from: 16, to: 18, progress: 0, delay: 5.8, speed: 0.021, strength: 0.7, dataFlow: 0 }
    ]

    // Data particles flowing through the network
    const particles: DataParticle[] = []

    let animationTime = 0
    let lastFrameTime = 0
    const targetFPS = 60
    const frameInterval = 1000 / targetFPS

    const drawNode = (node: Node) => {
      // Update pulse animation (reduced intensity)
      node.pulse += 0.03 // Reduced from 0.05 to 0.03 (40% slower)
      const pulseScale = 1 + 0.06 * Math.sin(node.pulse) // Reduced from 0.1 to 0.06 (40% less pulsing)
      
      // Outer glow with cluster color
      ctx.save()
      ctx.shadowColor = node.color
      ctx.shadowBlur = 15
      ctx.beginPath()
      ctx.fillStyle = `${node.color}30`
      ctx.arc(node.x, node.y, node.r + 6, 0, Math.PI * 2)
      ctx.fill()
      ctx.restore()

      // Main node with pulse effect
      ctx.beginPath()
      ctx.fillStyle = node.color
      ctx.arc(node.x, node.y, node.r * pulseScale, 0, Math.PI * 2)
      ctx.fill()

      // Inner highlight
      ctx.beginPath()
      ctx.fillStyle = `${node.color}90`
      ctx.arc(node.x - node.r * 0.3, node.y - node.r * 0.3, node.r * 0.4, 0, Math.PI * 2)
      ctx.fill()

      // Label for center node
      if (node.label) {
        ctx.fillStyle = '#000'
        ctx.font = 'bold 12px Inter'
        ctx.textAlign = 'center'
        ctx.fillText(node.label, node.x, node.y + 4)
      }
    }

    const easeOutCubic = (x: number) => 1 - Math.pow(1 - x, 3)

    const drawFrame = (timestamp: number) => {
      // Frame rate limiting for consistent performance
      if (timestamp - lastFrameTime < frameInterval) {
        requestAnimationFrame(drawFrame)
        return
      }
      lastFrameTime = timestamp
      
      // Clear canvas with transparent background
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      
      animationTime += 0.016

      // Generate data particles occasionally (reduced frequency)
      if (Math.random() < 0.03) { // Reduced from 0.08 to 0.03 (62% fewer particles)
        const centerNode = nodes[0]
        const randomCluster = Math.floor(Math.random() * 6) * 3 + 1
        const targetNode = nodes[randomCluster]
        
        if (centerNode && targetNode) {
          particles.push({
            x: centerNode.x,
            y: centerNode.y,
            vx: (targetNode.x - centerNode.x) * 0.008, // Reduced from 0.02 to 0.008 (60% slower)
            vy: (targetNode.y - centerNode.y) * 0.008, // Reduced from 0.02 to 0.008 (60% slower)
            color: targetNode.color,
            size: 2 + Math.random() * 2, // Reduced from 3 + random 2 to 2 + random 2
            opacity: 1,
            life: 0,
            maxLife: 180 + Math.random() * 120, // Increased from 120 + random 60 to 180 + random 120 (longer lifespan)
            targetX: targetNode.x,
            targetY: targetNode.y
          })
        }
      }

      // Update and draw data particles
      particles.forEach((particle, index) => {
        particle.x += particle.vx
        particle.y += particle.vy
        particle.life++
        particle.opacity = 1 - (particle.life / particle.maxLife)

        // Draw particle with glow effect
        ctx.save()
        ctx.globalAlpha = particle.opacity
        ctx.fillStyle = particle.color
        ctx.shadowColor = particle.color
        ctx.shadowBlur = 8
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2)
        ctx.fill()
        ctx.restore()

        // Remove dead particles
        if (particle.life >= particle.maxLife) {
          particles.splice(index, 1)
        }
      })

      // Update and draw connections
      connections.forEach(conn => {
        if (animationTime > conn.delay) {
          conn.progress = Math.min(1, conn.progress + conn.speed)
          
          // Update data flow for active connections
          if (conn.progress > 0.3) {
            conn.dataFlow += 0.008 // Reduced from 0.02 to 0.008 (60% slower)
          }
        }

        if (conn.progress > 0) {
          const A = nodes[conn.from]
          const B = nodes[conn.to]
          const local = easeOutCubic(conn.progress)
          const x = A.x + (B.x - A.x) * local
          const y = A.y + (B.y - A.y) * local

          // Connection line with neutral color and varying opacity based on strength
          ctx.strokeStyle = `rgba(163, 173, 191, ${0.3 * conn.strength})` // neutral with opacity
          ctx.lineWidth = 2 * conn.strength
          ctx.beginPath()
          ctx.moveTo(A.x, A.y)
          ctx.lineTo(x, y)
          ctx.stroke()

          // Animated progress dot
          if (local < 1) {
            ctx.beginPath()
            ctx.fillStyle = A.color
            ctx.arc(x, y, 4, 0, Math.PI * 2)
            ctx.fill()
          }

          // Data flow animation along completed connections
          if (conn.progress > 0.8 && conn.dataFlow > 0) {
            const flowPos = (conn.dataFlow % 1)
            const flowX = A.x + (B.x - A.x) * flowPos
            const flowY = A.y + (B.y - A.y) * flowPos
            
            ctx.save()
            ctx.globalAlpha = 0.8
            ctx.fillStyle = A.color
            ctx.shadowColor = A.color
            ctx.shadowBlur = 6
            ctx.beginPath()
            ctx.arc(flowX, flowY, 3, 0, Math.PI * 2)
            ctx.fill()
            ctx.restore()
          }
        }
      })

      // Enhanced floating animation for nodes (reduced intensity)
      nodes.forEach(node => {
        node.phase += 0.004 // Reduced from 0.006 to 0.004 (33% slower)
        const floatX = Math.sin(node.phase) * 1.2 // Reduced from 2 to 1.2 (40% less movement)
        const floatY = Math.cos(node.phase * 0.8) * 1.2 // Reduced from 2 to 1.2 (40% less movement)
        
        // Cluster-based movement patterns
        if (node.cluster === 'center') {
          node.x = node.baseX + floatX * 0.3 // Reduced from 0.5 to 0.3 (40% less movement)
          node.y = node.baseY + floatY * 0.3 // Reduced from 0.5 to 0.3 (40% less movement)
        } else {
          node.x = node.baseX + floatX
          node.y = node.baseY + floatY
        }
      })

      // Draw nodes on top
      nodes.forEach(drawNode)

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
      className="w-full h-64 rounded-2xl"
      style={{ width: '100%', height: '256px' }}
    />
  )
}