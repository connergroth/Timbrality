export const MusicalDNA = () => {
  return (
    <section id="musical-dna" className="py-24 bg-neutral-900">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="font-playfair text-4xl md:text-5xl font-bold text-white mb-6">
            Your Musical DNA
          </h2>
          <p className="text-lg text-neutral-300 font-inter max-w-3xl mx-auto">
            Every person has a unique musical fingerprint. Discover yours.
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-8 shadow-2xl">
            <div className="space-y-6">
              {/* DNA Strand Visualization */}
              <div className="flex justify-center">
                <div className="relative">
                  {/* Main DNA strand */}
                  <div className="w-1 h-32 bg-gradient-to-b from-neutral-400 to-neutral-600 rounded-full"></div>
                  
                  {/* DNA rungs */}
                  {[...Array(8)].map((_, i) => (
                    <div
                      key={i}
                      className="absolute left-1/2 transform -translate-x-1/2 w-16 h-0.5 bg-neutral-500/60"
                      style={{ top: `${i * 16}px` }}
                    ></div>
                  ))}
                  
                  {/* Musical notes on DNA */}
                  {['♪', '♫', '♬', '♩'].map((note, i) => (
                    <div
                      key={i}
                      className="absolute text-2xl text-neutral-300 transform -translate-x-1/2"
                      style={{ 
                        top: `${i * 16 + 8}px`,
                        left: i % 2 === 0 ? 'calc(50% - 24px)' : 'calc(50% + 24px)'
                      }}
                    >
                      {note}
                    </div>
                  ))}
                </div>
              </div>

              {/* DNA Description */}
              <div className="text-center space-y-4">
                <h3 className="text-2xl font-playfair font-bold text-white">
                  What Makes You Unique
                </h3>
                <p className="text-neutral-300 font-inter leading-relaxed max-w-2xl mx-auto">
                  Your musical DNA is composed of your listening patterns, emotional responses to music, 
                  preferred genres, and the unique way you experience sound. Our AI analyzes thousands 
                  of data points to map your musical genome.
                </p>
                
                {/* DNA Components */}
                <div className="grid md:grid-cols-3 gap-4 mt-8">
                  <div className="text-center">
                    <div className="w-3 h-3 bg-neutral-400 rounded-full mx-auto mb-2"></div>
                    <p className="text-sm text-neutral-300 font-inter">Listening Patterns</p>
                  </div>
                  <div className="text-center">
                    <div className="w-3 h-3 bg-neutral-500 rounded-full mx-auto mb-2"></div>
                    <p className="text-sm text-neutral-300 font-inter">Emotional Responses</p>
                  </div>
                  <div className="text-center">
                    <div className="w-3 h-3 bg-neutral-600 rounded-full mx-auto mb-2"></div>
                    <p className="text-sm text-neutral-300 font-inter">Genre Preferences</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};