import { Brain, Music, Star, TrendingUp } from 'lucide-react';

const HowItWorksSection = () => {
  const features = [
    {
      icon: Music,
      title: "Content-Based Analysis",
      description: "Our AI analyzes audio features, genres, and musical patterns to understand what makes each song unique."
    },
    {
      icon: TrendingUp,
      title: "Collaborative Filtering",
      description: "Learn from millions of listening patterns to discover music loved by people with similar tastes."
    },
    {
      icon: Star,
      title: "Personal Ratings",
      description: "Your feedback continuously improves recommendations, creating a truly personalized experience."
    }
  ];

  return (
    <section id="how-it-works" className="py-20 bg-tensoe-navy relative overflow-hidden">
      {/* Subtle background tech elements */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-1/4 left-1/4 w-px h-20 bg-gradient-to-b from-transparent via-tensoe-blue to-transparent"></div>
        <div className="absolute top-1/3 right-1/4 w-px h-16 bg-gradient-to-b from-transparent via-tensoe-blue-light to-transparent"></div>
        <div className="absolute bottom-1/4 left-1/3 w-16 h-px bg-gradient-to-r from-transparent via-tensoe-blue to-transparent"></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6 gradient-text">
            How It Works
          </h2>
          <p className="text-xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
            Our hybrid recommendation engine combines collaborative filtering (NMF), content-based filtering 
            (TF-IDF & audio features), and metadata from Last.fm, Spotify, and Albumoftheyear.org to create 
            a hybrid recommendation model that truly resonates with you.
          </p>
          
          {/* Tech divider */}
          <div className="flex items-center justify-center mt-8">
            <div className="h-px bg-gradient-to-r from-transparent via-tensoe-blue-light to-transparent w-24"></div>
            <div className="mx-3 w-1.5 h-1.5 bg-tensoe-blue-light rounded-full animate-pulse"></div>
            <div className="h-px bg-gradient-to-r from-transparent via-tensoe-blue-light to-transparent w-24"></div>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {features.map((feature, index) => (
            <div
              key={index}
              className="feature-card text-center animate-fade-in"
              style={{ animationDelay: `${index * 0.2}s` }}
            >
              <div className="w-16 h-16 bg-tensoe-blue/20 rounded-full flex items-center justify-center mx-auto mb-6">
                <feature.icon className="text-tensoe-blue" size={32} />
              </div>
              <h3 className="text-xl font-semibold mb-4 text-white">
                {feature.title}
              </h3>
              <p className="text-gray-300 leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        {/* Enhanced central brain visualization */}
        <div className="text-center">
          <div className="relative inline-block">
            {/* Outer ring - properly centered */}
            <div className="absolute top-1/2 left-1/2 w-36 h-36 border border-tensoe-blue/20 rounded-full animate-pulse transform -translate-x-1/2 -translate-y-1/2" style={{ animationDuration: '4s' }}></div>
            
            {/* Main brain container */}
            <div className="w-32 h-32 bg-gradient-to-r from-tensoe-blue to-tensoe-blue-light rounded-full flex items-center justify-center animate-pulse-slow relative">
              <Brain size={64} className="text-tensoe-navy transform translate-x-0 translate-y-0" />
            </div>
            
            {/* Enhanced orbiting elements */}
            <div className="absolute inset-0 animate-spin" style={{ animationDuration: '20s' }}>
              <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                <div className="relative">
                  <Music size={24} className="text-tensoe-blue" />
                  <div className="absolute inset-0 bg-tensoe-blue/20 rounded-full blur-sm scale-150"></div>
                </div>
              </div>
            </div>
            <div className="absolute inset-0 animate-spin" style={{ animationDuration: '15s', animationDirection: 'reverse' }}>
              <div className="absolute top-1/2 -right-4 transform -translate-y-1/2">
                <div className="relative">
                  <Star size={20} className="text-tensoe-blue-light" />
                  <div className="absolute inset-0 bg-tensoe-blue-light/20 rounded-full blur-sm scale-150"></div>
                </div>
              </div>
            </div>
            <div className="absolute inset-0 animate-spin" style={{ animationDuration: '25s' }}>
              <div className="absolute -bottom-4 left-1/2 transform -translate-x-1/2">
                <div className="relative">
                  <TrendingUp size={22} className="text-tensoe-blue" />
                  <div className="absolute inset-0 bg-tensoe-blue/20 rounded-full blur-sm scale-150"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorksSection;
