import { Users, ArrowRight } from 'lucide-react';
import { FaSpotify } from 'react-icons/fa';

const ConnectSection = () => {
  const platforms = [
    {
      name: "Spotify",
      iconComponent: FaSpotify,
      description: "Connect your Spotify account",
      action: "Connect Spotify",
      primary: true
    },
    {
      name: "Last.fm",
      logo: "/logos/lastfm.png",
      description: "Import your Last.fm history",
      action: "Connect Last.fm",
      primary: false
    },
    {
      name: "AOTY",
      logo: "/logos/aoty.png",
      description: "Enter your AOTY username",
      action: "Enter AOTY Username",
      primary: false,
      isLarge: true
    }
  ];

  return (
    <section id="connect" className="py-20 bg-tensoe-navy-light/30 relative">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6 gradient-text leading-relaxed px-4 py-2">
            Ready to discover your next favorite song?
          </h2>
          <p className="text-xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
            Connect your favorite platform to get started.
          </p>
          
          {/* Tech-inspired divider */}
          <div className="flex items-center justify-center mt-8">
            <div className="h-px bg-gradient-to-r from-transparent via-tensoe-blue to-transparent w-32"></div>
            <div className="mx-4 w-2 h-2 bg-tensoe-blue rounded-full animate-pulse"></div>
            <div className="h-px bg-gradient-to-r from-transparent via-tensoe-blue to-transparent w-32"></div>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {platforms.map((platform, index) => (
            <div
              key={index}
              className="feature-card text-center cursor-pointer group ring-2 ring-tensoe-blue"
            >
              <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 bg-tensoe-blue">
                {platform.logo ? (
                  <img 
                    src={platform.logo} 
                    alt={`${platform.name} logo`} 
                    className={platform.isLarge ? "w-14 h-14 object-contain" : "w-8 h-8 object-contain"}
                  />
                ) : platform.iconComponent ? (
                  <platform.iconComponent size={32} className="text-white" />
                ) : null}
              </div>
              <h3 className="text-lg font-semibold mb-3 text-white">
                {platform.name}
              </h3>
              <p className="text-gray-300 mb-6 text-sm">
                {platform.description}
              </p>
              <button className="btn-primary w-full">
                {platform.action}
              </button>
            </div>
          ))}
        </div>

        {/* Standalone Explore as Guest button */}
        <div className="text-center mb-16">
          <button className="inline-flex items-center justify-center px-8 py-4 bg-transparent border border-gray-600 text-gray-300 rounded-full hover:border-gray-500 hover:text-white transition-all duration-300 group">
            <ArrowRight size={20} className="mr-3 group-hover:translate-x-1 transition-transform duration-300" />
            Explore as Guest
          </button>
        </div>

        <div className="text-center">
          {/* Enhanced security message with tech styling */}
          <div className="relative inline-block">
            <div className="absolute -inset-1 bg-gradient-to-r from-tensoe-blue/20 to-transparent rounded-lg blur opacity-30"></div>
            <p className="relative text-gray-400 text-sm max-w-2xl mx-auto bg-tensoe-navy/50 rounded-lg px-6 py-1.5 border border-tensoe-blue/20">
              🔒 Your data is secure and never shared. We only use it to improve your recommendations.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ConnectSection;
