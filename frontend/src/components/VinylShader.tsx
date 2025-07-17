import { useEffect, useState } from "react";

export const VinylShader = () => {
  const [rotation, setRotation] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setRotation((prev) => (prev + 0.2) % 360);
    }, 50);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Main vinyl disc with concentric rings */}
      <div 
        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] opacity-[0.165]"
        style={{ transform: `translate(-50%, -50%) rotate(${rotation}deg)` }}
      >
        {/* Outer ring */}
        <div className="absolute inset-0 rounded-full border-2 border-primary/40 animate-pulse" 
             style={{ animationDuration: '4s' }} />
        
        {/* Multiple concentric rings */}
        {[...Array(12)].map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full border border-primary/30"
            style={{
              top: `${i * 6}%`,
              left: `${i * 6}%`,
              right: `${i * 6}%`,
              bottom: `${i * 6}%`,
              animationDelay: `${i * 0.2}s`,
              animationDuration: `${3 + i * 0.5}s`,
            }}
          />
        ))}

        {/* Center hole */}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-16 h-16 bg-background rounded-full border border-primary/50" />
        
        {/* Groove lines - rotating at different speeds */}
        {[...Array(8)].map((_, i) => (
          <div
            key={`groove-${i}`}
            className="absolute top-1/2 left-1/2 w-full h-full"
            style={{
              transform: `translate(-50%, -50%) rotate(${i * 45 + rotation * 0.5}deg)`,
            }}
          >
            <div className="absolute top-0 left-1/2 w-px h-full bg-gradient-to-b from-transparent via-primary/20 to-transparent" />
          </div>
        ))}
      </div>

      {/* Secondary vinyl disc - smaller and slower */}
      <div 
        className="absolute top-1/4 right-1/4 w-[400px] h-[400px] opacity-[0.13]"
        style={{ transform: `rotate(${rotation * 0.3}deg)` }}
      >
        <div className="absolute inset-0 rounded-full border border-primary/35" />
        {[...Array(8)].map((_, i) => (
          <div
            key={`secondary-${i}`}
            className="absolute rounded-full border border-primary/25"
            style={{
              top: `${i * 8}%`,
              left: `${i * 8}%`,
              right: `${i * 8}%`,
              bottom: `${i * 8}%`,
            }}
          />
        ))}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-background rounded-full border border-primary/40" />
      </div>

      {/* Third vinyl disc - even smaller */}
      <div 
        className="absolute bottom-1/4 left-1/4 w-[300px] h-[300px] opacity-[0.115]"
        style={{ transform: `rotate(${rotation * 0.7}deg)` }}
      >
        <div className="absolute inset-0 rounded-full border border-primary/30" />
        {[...Array(6)].map((_, i) => (
          <div
            key={`tertiary-${i}`}
            className="absolute rounded-full border border-primary/20"
            style={{
              top: `${i * 10}%`,
              left: `${i * 10}%`,
              right: `${i * 10}%`,
              bottom: `${i * 10}%`,
            }}
          />
        ))}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-6 h-6 bg-background rounded-full border border-primary/35" />
      </div>

      {/* Shimmer effects */}
      <div className="absolute inset-0">
        <div 
          className="absolute top-1/3 left-1/3 w-32 h-32 bg-gradient-to-r from-transparent via-primary/12.5 to-transparent rounded-full blur-xl animate-pulse"
          style={{ animationDuration: '6s' }}
        />
        <div 
          className="absolute bottom-1/3 right-1/3 w-24 h-24 bg-gradient-to-r from-transparent via-primary/12.5 to-transparent rounded-full blur-lg animate-pulse"
          style={{ animationDuration: '8s', animationDelay: '2s' }}
        />
      </div>

      {/* Parallax shimmer lines */}
      {[...Array(4)].map((_, i) => (
        <div
          key={`shimmer-${i}`}
          className="absolute w-full h-px bg-gradient-to-r from-transparent via-primary/16.5 to-transparent"
          style={{
            top: `${20 + i * 20}%`,
            transform: `translateX(${Math.sin(rotation * 0.01 + i) * 50}px)`,
            opacity: 0.4 + Math.sin(rotation * 0.02 + i) * 0.25,
          }}
        />
      ))}
    </div>
  );
}; 