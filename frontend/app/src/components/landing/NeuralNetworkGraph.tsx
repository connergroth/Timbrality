'use client'

import { useEffect, useRef } from 'react'

interface NetworkNode {
  x: number
  y: number
  baseX: number
  baseY: number
  r: number
  layer: number
  activation: number
  maxActivation: number
  activationTime: number
  label?: string
}

interface NetworkConnection {
  from: number
  to: number
  weight: number
  dataFlow: number
  lastDataTime: number
}

interface DataPacket {
  fromNode: number
  toNode: number
  progress: number
  startTime: number
  speed: number
}

export const NeuralNetworkGraph = () => {
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

    const W = canvas.clientWidth
    const H = canvas.clientHeight

    // Create neural network structure
    const layers = [
      { nodeCount: 4, x: 40, label: 'Audio Features' },    // Input layer
      { nodeCount: 6, x: 120, label: 'Processing' },       // Hidden layer 1
      { nodeCount: 5, x: 200, label: 'Analysis' },         // Hidden layer 2
      { nodeCount: 3, x: 280, label: 'Recommendations' }   // Output layer
    ]

    // Create nodes
    const nodes: NetworkNode[] = []
    let nodeId = 0

    layers.forEach((layer, layerIndex) => {
      const startY = (H - (layer.nodeCount - 1) * 25) / 2
      for (let i = 0; i < layer.nodeCount; i++) {
        const y = startY + i * 25
        nodes.push({
          x: layer.x,
          y: y,
          baseX: layer.x,
          baseY: y,
          r: 6,
          layer: layerIndex,
          activation: 0,
          maxActivation: 0,
          activationTime: 0,
          label: layerIndex === 0 ? ['Tempo', 'Energy', 'Valence', 'Danceability'][i] : undefined
        })
        nodeId++
      }
    })

    // Create connections between adjacent layers
    const connections: NetworkConnection[] = []
    let layerStartIndex = 0

    for (let l = 0; l < layers.length - 1; l++) {
      const currentLayerSize = layers[l].nodeCount
      const nextLayerSize = layers[l + 1].nodeCount
      
      for (let i = 0; i < currentLayerSize; i++) {
        for (let j = 0; j < nextLayerSize; j++) {
          const fromIndex = layerStartIndex + i
          const toIndex = layerStartIndex + currentLayerSize + j
          
          connections.push({
            from: fromIndex,
            to: toIndex,
            weight: 0.3 + Math.random() * 0.7,
            dataFlow: 0,
            lastDataTime: 0
          })
        }
      }
      layerStartIndex += currentLayerSize
    }

    // Data packets for animation
    const dataPackets: DataPacket[] = []
    let animationTime = 0
    let lastPacketTime = 0

    const createDataPacket = (time: number) => {
      // Start from input layer
      const inputLayerSize = layers[0].nodeCount
      const fromNode = Math.floor(Math.random() * inputLayerSize)
      
      // Find connections from this input node
      const validConnections = connections.filter(conn => conn.from === fromNode)
      if (validConnections.length > 0) {
        const randomConnection = validConnections[Math.floor(Math.random() * validConnections.length)]
        
        dataPackets.push({
          fromNode: randomConnection.from,
          toNode: randomConnection.to,
          progress: 0,
          startTime: time,
          speed: 0.8 + Math.random() * 0.4
        })
        
        // Activate the source node
        nodes[fromNode].activation = 1
        nodes[fromNode].maxActivation = 1
        nodes[fromNode].activationTime = time
      }
    }

    const drawNode = (node: NetworkNode, time: number) => {
      // Activation decay
      const timeSinceActivation = time - node.activationTime
      if (timeSinceActivation > 0) {
        node.activation = Math.max(0, node.maxActivation * Math.exp(-timeSinceActivation * 2))
      }

      // Node glow based on activation
      if (node.activation > 0.1) {
        ctx.save()
        ctx.globalAlpha = node.activation * 0.5
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.r + 8, 0, Math.PI * 2)
        ctx.fillStyle = '#fbbf24' // yellow glow
        ctx.fill()
        ctx.restore()
      }

      // Main node
      const intensity = node.activation
      const baseColor = node.layer === 0 ? '#10b981' : // green for input
                       node.layer === layers.length - 1 ? '#8b5cf6' : // purple for output  
                       '#3b82f6' // blue for hidden layers

      ctx.beginPath()
      ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2)
      ctx.fillStyle = intensity > 0.1 ? '#fbbf24' : baseColor
      ctx.fill()

      // Node border
      ctx.beginPath()
      ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
      ctx.lineWidth = 1
      ctx.stroke()

      // Label for input nodes
      if (node.label && node.layer === 0) {
        ctx.save()
        ctx.font = '9px Inter'
        ctx.fillStyle = '#ffffff'
        ctx.textAlign = 'right'
        ctx.fillText(node.label, node.x - 12, node.y + 2)
        ctx.restore()
      }
    }

    const drawConnection = (conn: NetworkConnection, time: number) => {
      const fromNode = nodes[conn.from]
      const toNode = nodes[conn.to]

      // Connection line
      ctx.save()
      ctx.globalAlpha = 0.2 + conn.weight * 0.3
      ctx.beginPath()
      ctx.moveTo(fromNode.x, fromNode.y)
      ctx.lineTo(toNode.x, toNode.y)
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 1
      ctx.stroke()
      ctx.restore()
    }

    const drawDataPacket = (packet: DataPacket, time: number) => {
      const fromNode = nodes[packet.fromNode]
      const toNode = nodes[packet.toNode]

      const x = fromNode.x + (toNode.x - fromNode.x) * packet.progress
      const y = fromNode.y + (toNode.y - fromNode.y) * packet.progress

      // Data packet
      ctx.save()
      ctx.beginPath()
      ctx.arc(x, y, 3, 0, Math.PI * 2)
      ctx.fillStyle = '#fbbf24'
      ctx.fill()
      
      // Glow
      ctx.globalAlpha = 0.6
      ctx.beginPath()
      ctx.arc(x, y, 6, 0, Math.PI * 2)
      ctx.fillStyle = '#fbbf24'
      ctx.fill()
      ctx.restore()
    }

    const drawFrame = (timestamp: number) => {
      animationTime = timestamp / 1000

      // Clear canvas
      ctx.fillStyle = 'rgba(38, 38, 38, 0.4)' // neutral-800/40
      ctx.fillRect(0, 0, W, H)

      // Create new data packets periodically
      if (animationTime - lastPacketTime > 0.8) {
        createDataPacket(animationTime)
        lastPacketTime = animationTime
      }

      // Update and draw connections
      connections.forEach(conn => drawConnection(conn, animationTime))

      // Update data packets
      dataPackets.forEach((packet, index) => {
        const elapsed = animationTime - packet.startTime
        packet.progress = Math.min(1, elapsed * packet.speed)

        if (packet.progress >= 1) {
          // Packet reached destination - activate target node
          const targetNode = nodes[packet.toNode]
          targetNode.activation = 1
          targetNode.maxActivation = 1
          targetNode.activationTime = animationTime

          // Propagate to next layer if not output layer
          if (targetNode.layer < layers.length - 1) {
            setTimeout(() => {
              const nextLayerConnections = connections.filter(conn => conn.from === packet.toNode)
              if (nextLayerConnections.length > 0) {
                const randomNext = nextLayerConnections[Math.floor(Math.random() * nextLayerConnections.length)]
                dataPackets.push({
                  fromNode: randomNext.from,
                  toNode: randomNext.to,
                  progress: 0,
                  startTime: animationTime,
                  speed: 0.8 + Math.random() * 0.4
                })
              }
            }, 100)
          }

          // Remove completed packet
          dataPackets.splice(index, 1)
        } else {
          drawDataPacket(packet, animationTime)
        }
      })

      // Draw nodes
      nodes.forEach(node => drawNode(node, animationTime))

      // Draw layer labels
      ctx.save()
      ctx.font = 'bold 10px Inter'
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
      ctx.textAlign = 'center'
      
      layers.forEach((layer, i) => {
        ctx.fillText(layer.label, layer.x, H - 10)
      })
      ctx.restore()

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