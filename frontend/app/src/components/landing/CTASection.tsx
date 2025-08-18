import { Sparkles, ArrowRight } from "lucide-react";

export const CTASection = () => {
  return (
    <section className="py-24 bg-neutral-900">
      <div className="container mx-auto px-6">
        <div className="bg-gradient-to-br from-neutral-800/60 to-neutral-900/80 backdrop-blur-xl border border-white/20 rounded-3xl p-8 md:p-12 shadow-2xl border-glow-animation text-center relative overflow-hidden max-w-5xl mx-auto">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-transparent to-blue-500/10 pointer-events-none rounded-3xl"></div>
          
          <div className="relative z-10">
            <h3 className="font-playfair text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-8 tracking-tight">
              Get your own personal AI music agent.
            </h3>
            <p className="text-neutral-300 font-inter text-xl md:text-2xl leading-relaxed mb-12 max-w-3xl mx-auto">
              Join thousands of music lovers who've already found their perfect sound. Connect your Spotify and let our AI curate your next favorite song.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-6 justify-center">
              <a
                href="/auth"
                className="inline-flex items-center gap-2 bg-white text-neutral-900 px-12 py-4 rounded-lg font-inter font-semibold hover:bg-neutral-100 transition-all duration-300 hover:scale-105 text-xl"
              >
                <Sparkles className="w-5 h-5" />
                Start Your Journey
                <ArrowRight className="w-5 h-5" />
              </a>
              <a
                href="/pricing"
                className="inline-flex items-center gap-2 border border-white text-white px-12 py-4 rounded-lg font-inter font-semibold hover:bg-white hover:text-neutral-900 transition-all duration-300 hover:scale-105 text-xl"
              >
                View Pricing
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};