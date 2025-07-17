import { Music, Search, Heart, Info, ChevronDown, ChevronUp, Github, Brain, Target } from "lucide-react";
import { useState } from "react";

export const HowItWorks = () => {
  const [isAlgorithmOpen, setIsAlgorithmOpen] = useState(false);
  
  const steps = [
    {
      icon: Music,
      title: "Connect your profile",
      description: "Your Spotify or Last.fm history becomes the foundation of your personalized recommendations."
    },
    {
      icon: Search,
      title: "Understand your sonic fingerprint", 
      description: "Our hybrid model listens for timbre, mood, and listening patterns — not just genre or charts."
    },
    {
      icon: Heart,
      title: "Receive curated recommendations",
      description: "Discover music that feels like you — textured, personal, and uniquely matched to your listening style."
    }
  ];

  return (
    <section id="how-it-works" className="py-20 bg-muted/20 scroll-mt-24">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="font-playfair text-4xl md:text-5xl font-bold text-foreground mb-4">
            How it works
          </h2>
          <p className="text-xl text-muted-foreground font-playfair">
            Three simple steps to unlock your musical DNA
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {steps.map((step, index) => (
            <div key={index} className="text-center space-y-6">
              <div className="w-16 h-16 mx-auto bg-primary/10 rounded-full flex items-center justify-center">
                <step.icon className="w-8 h-8 text-primary" />
              </div>
              <h3 className="font-playfair text-xl font-semibold text-foreground">
                {step.title}
              </h3>
              <p className="text-muted-foreground leading-relaxed">
                {step.description}
              </p>
            </div>
          ))}
        </div>

        {/* Behind the Algorithm Section */}
        <div className="mt-16 text-center">
          <button
            onClick={() => setIsAlgorithmOpen(!isAlgorithmOpen)}
            className="inline-flex items-center space-x-2 text-sm font-playfair text-muted-foreground hover:text-foreground transition-colors group"
          >
            <Info className="w-4 h-4" />
            <span>Behind the Algorithm</span>
            {isAlgorithmOpen ? (
              <ChevronUp className="w-4 h-4 transition-transform" />
            ) : (
              <ChevronDown className="w-4 h-4 transition-transform" />
            )}
          </button>

          {isAlgorithmOpen && (
            <div className="mt-8 max-w-4xl mx-auto bg-card border border-border rounded-2xl p-8 text-left">
              <h3 className="font-playfair text-2xl font-semibold text-foreground mb-6">
                How Timbre Generates Recommendations
              </h3>
              
              <p className="text-muted-foreground mb-6 leading-relaxed">
                Your recommendations are powered by Timbral, a hybrid ML engine that blends:
              </p>

              <div className="space-y-4 mb-6">
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Brain className="w-3 h-3 text-primary" />
                  </div>
                  <div>
                    <span className="font-medium text-foreground">Collaborative filtering (NMF)</span>
                    <span className="text-muted-foreground"> based on your listening history</span>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Search className="w-3 h-3 text-primary" />
                  </div>
                  <div>
                    <span className="font-medium text-foreground">Content similarity</span>
                    <span className="text-muted-foreground"> using mood, genre, and Sentence-BERT embeddings</span>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Target className="w-3 h-3 text-primary" />
                  </div>
                  <div>
                    <span className="font-medium text-foreground">Smart scoring</span>
                    <span className="text-muted-foreground"> that combines both systems and ranks tracks by musical fit</span>
                  </div>
                </div>
              </div>

              <p className="text-muted-foreground mb-6 leading-relaxed">
                We also factor in metadata from Spotify, Last.fm, and AlbumOfTheYear to reflect both popularity and niche appeal.
              </p>

              <div className="flex items-center justify-between pt-4 border-t border-border">
                <button className="text-sm font-playfair text-primary hover:text-primary/80 transition-colors">
                  Want more technical details? See the model architecture
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};