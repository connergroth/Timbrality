import { useEffect, useState } from "react";

export const VinylShader = () => {
  const [rotation, setRotation] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setRotation((prev) => (prev + 0.3) % 360);
    }, 40);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Main vinyl record with blue label */}
      <div 
        className="absolute top-1/2 left-1/2 w-[800px] h-[800px] opacity-18"
        style={{ transform: `translate(-50%, -50%) rotate(${rotation}deg)` }}
      >
        <div className="absolute inset-0 rounded-full bg-gradient-radial from-slate-900/70 via-slate-800/60 to-slate-900/80 shadow-2xl">
          {/* Outer edge highlight */}
          <div className="absolute inset-0 rounded-full border border-slate-700/50 shadow-inner" />
          
          {/* Realistic concentric grooves */}
          {[...Array(22)].map((_, i) => {
            const radius = 8 + i * 3.8;
            return (
              <div
                key={`groove-${i}`}
                className="absolute rounded-full border border-slate-600/35"
                style={{
                  top: `${radius}%`,
                  left: `${radius}%`,
                  right: `${radius}%`,
                  bottom: `${radius}%`,
                  borderWidth: i % 3 === 0 ? '1px' : '0.5px',
                  opacity: 0.6 - (i * 0.018),
                }}
              />
            );
          })}
          
          {/* Center label with blue theme */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-28 h-28 bg-gradient-radial from-blue-900/60 via-blue-800/50 to-blue-900/70 rounded-full border border-blue-700/40 shadow-lg">
            <div className="absolute inset-2 rounded-full border border-blue-600/30 flex items-center justify-center">
              <div className="w-2 h-2 bg-blue-400/50 rounded-full" />
            </div>
          </div>

          {/* Center spindle hole */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-black rounded-full border border-slate-600/80 shadow-inner" />
        </div>

        {/* Vinyl reflection/shine effect */}
        <div 
          className="absolute inset-0 rounded-full"
          style={{
            background: `conic-gradient(from ${rotation * 2}deg, transparent 0deg, rgba(255,255,255,0.14) 45deg, transparent 90deg, rgba(255,255,255,0.08) 180deg, transparent 225deg, rgba(255,255,255,0.11) 315deg, transparent 360deg)`,
          }}
        />

        {/* Groove light reflection lines */}
        {[...Array(6)].map((_, i) => (
          <div
            key={`reflection-${i}`}
            className="absolute top-1/2 left-1/2 w-full h-full"
            style={{
              transform: `translate(-50%, -50%) rotate(${i * 60 + rotation * 1.2}deg)`,
            }}
          >
            <div className="absolute top-0 left-1/2 w-0.5 h-full bg-gradient-to-b from-transparent via-white/8 to-transparent opacity-50" />
          </div>
        ))}
      </div>

      {/* Atmospheric light rays */}
      <div className="absolute inset-0">
        {[...Array(3)].map((_, i) => (
          <div
            key={`light-ray-${i}`}
            className="absolute w-full h-0.5 bg-gradient-to-r from-transparent via-primary/8 to-transparent blur-sm"
            style={{
              top: `${30 + i * 25}%`,
              transform: `translateX(${Math.sin(rotation * 0.015 + i * 2) * 100}px) rotate(${Math.sin(rotation * 0.008 + i) * 15}deg)`,
              opacity: 0.6 + Math.sin(rotation * 0.02 + i) * 0.3,
            }}
          />
        ))}
      </div>

      {/* Dust particles effect */}
      {[...Array(8)].map((_, i) => (
        <div
          key={`dust-${i}`}
          className="absolute w-1 h-1 bg-white/20 rounded-full blur-sm"
          style={{
            top: `${15 + Math.sin(rotation * 0.01 + i) * 70}%`,
            left: `${15 + Math.cos(rotation * 0.008 + i * 2) * 70}%`,
            transform: `scale(${0.5 + Math.sin(rotation * 0.02 + i) * 0.5})`,
            opacity: 0.3 + Math.sin(rotation * 0.03 + i) * 0.4,
          }}
        />
      ))}
    </div>
  );
};