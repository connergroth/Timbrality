import { useEffect, useState } from "react";

export const VinylShader = () => {
  const [rotation, setRotation] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setRotation((prev) => (prev + 3) % 360);
    }, 8);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Main vinyl record with blue label */}
      <div 
        className="absolute top-1/2 left-1/2 w-[200px] h-[200px]"
        style={{ transform: `translate(-50%, -50%) rotate(${rotation}deg)` }}
      >
        <div className="absolute inset-0 rounded-full bg-gradient-radial from-slate-900/70 via-slate-800/60 to-slate-900/80 shadow-2xl">
          {/* Outer edge highlight */}
          <div className="absolute inset-0 rounded-full border border-slate-700/50 shadow-inner" />
          
          {/* Realistic concentric grooves */}
          {[...Array(12)].map((_, i) => {
            const radius = 8 + i * 6;
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
                  opacity: 0.6 - (i * 0.03),
                }}
              />
            );
          })}
          
          {/* Center label with blue theme */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-16 h-16 bg-gradient-radial from-blue-900/60 via-blue-800/50 to-blue-900/70 rounded-full border border-blue-700/40 shadow-lg">
            <div className="absolute inset-2 rounded-full border border-blue-600/30 flex items-center justify-center">
              <div className="w-1 h-1 bg-blue-400/50 rounded-full" />
            </div>
          </div>

          {/* Center spindle hole */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-black rounded-full border border-slate-600/80 shadow-inner" />
        </div>

        {/* Vinyl reflection/shine effect */}
        <div 
          className="absolute inset-0 rounded-full"
          style={{
            background: `conic-gradient(from ${rotation * 2}deg, transparent 0deg, rgba(255,255,255,0.14) 45deg, transparent 90deg, rgba(255,255,255,0.08) 180deg, transparent 225deg, rgba(255,255,255,0.11) 315deg, transparent 360deg)`,
          }}
        />

        {/* Groove light reflection lines */}
        {[...Array(4)].map((_, i) => (
          <div
            key={`reflection-${i}`}
            className="absolute top-1/2 left-1/2 w-full h-full"
            style={{
              transform: `translate(-50%, -50%) rotate(${i * 90 + rotation * 0.8}deg)`,
            }}
          >
            <div className="absolute top-0 left-1/2 w-0.5 h-full bg-gradient-to-b from-transparent via-white/8 to-transparent opacity-50" />
          </div>
        ))}
      </div>
    </div>
  );
};