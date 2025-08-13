export const MusicalDNA = () => {
  const traits = [
    { name: "Atmospheric", percentage: 92 },
    { name: "Melodic", percentage: 87 },
    { name: "Experimental", percentage: 74 }
  ];

  return (
    <section id="your-dna" className="py-20 bg-muted/20 scroll-mt-24">
      <div className="max-w-4xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="font-playfair text-4xl md:text-5xl font-bold text-foreground mb-4">
            Your Musical DNA
          </h2>
          <p className="text-xl text-muted-foreground font-playfair">
            Based on your listening patterns, we've identified your unique sonic signature.
          </p>
        </div>

        <div className="space-y-8">
          {traits.map((trait, index) => (
            <div key={index} className="space-y-3">
              <div className="flex justify-between items-center">
                <h3 className="font-playfair text-xl font-medium text-foreground">
                  {trait.name}
                </h3>
                <span className="text-lg font-medium text-primary">
                  {trait.percentage}%
                </span>
              </div>
              <div className="w-full bg-muted rounded-full h-3">
                <div 
                  className="bg-gradient-to-r from-primary to-primary-glow h-3 rounded-full transition-all duration-1000 ease-out"
                  style={{ width: `${trait.percentage}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};