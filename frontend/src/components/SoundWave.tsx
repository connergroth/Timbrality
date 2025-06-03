import React from 'react';
export function SoundWave() {
  return <div className="absolute inset-0 opacity-20">
      <svg className="w-full h-full" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#00C6FF" />
            <stop offset="100%" stopColor="#0099FF" />
          </linearGradient>
        </defs>
        {[...Array(12)].map((_, i) => <rect key={i} x={i * 100 + 50} y="400" width="8" height="40" fill="url(#waveGradient)" className="animate-pulse" style={{
        animationDelay: `${i * 0.1}s`,
        animationDuration: `${1.5 + i % 3 * 0.5}s`,
        transformOrigin: 'center bottom'
      }}>
            <animateTransform attributeName="transform" type="scale" values="1,0.3;1,2;1,0.8;1,1.5;1,0.3" dur={`${2 + i % 4 * 0.3}s`} repeatCount="indefinite" />
          </rect>)}
      </svg>
    </div>;
}