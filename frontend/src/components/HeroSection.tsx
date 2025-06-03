import AnimatedBackground from './AnimatedBackground';
import { FaSpotify } from 'react-icons/fa';

const HeroSection = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center shader-bg overflow-hidden">
      <AnimatedBackground />
      
      <div className="relative z-10 container mx-auto px-6 text-center">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-6xl md:text-8xl font-bold mb-6 animate-fade-in">
            <span className="gradient-text">Tensoe</span>
          </h1>
          
          <p className="text-2xl md:text-3xl text-tensoe-blue-light mb-4 animate-fade-in" style={{ animationDelay: '0.2s' }}>
            Discover music intelligently
          </p>
          
          <p className="text-lg md:text-xl text-gray-300 mb-12 max-w-2xl mx-auto animate-fade-in" style={{ animationDelay: '0.4s' }}>
            Shaped by machine learning. Tuned for your taste.
          </p>

          <div className="flex flex-col sm:flex-row gap-6 justify-center animate-fade-in" style={{ animationDelay: '0.6s' }}>
            <button className="btn-primary text-lg px-8 py-4 flex items-center justify-center gap-3">
              <FaSpotify className="text-black" size={24} />
              Connect Spotify
            </button>
            <button className="btn-secondary text-lg px-8 py-4">
              Learn More
            </button>
          </div>

          {/* Sound wave visualization */}
          <div className="flex justify-center items-end space-x-1 mt-16 opacity-60">
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="bg-tensoe-blue w-1 rounded-full animate-wave"
                style={{
                  height: `${Math.random() * 40 + 10}px`,
                  animationDelay: `${i * 0.1}s`
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
