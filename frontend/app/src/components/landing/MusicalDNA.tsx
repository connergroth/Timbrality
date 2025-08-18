import { AICurationAnimation } from "./AICurationAnimation";

export const MusicalDNA = () => {
  return (
    <section id="ai-curation" className="py-24 bg-neutral-900">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="font-playfair text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6">
            AI Curation
          </h2>
          <p className="text-lg text-neutral-300 font-inter max-w-3xl mx-auto">
            Have a conversation with your personal AI music agent. Get instant recommendations tailored to your mood, preferences, and context.
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <AICurationAnimation />
        </div>
      </div>
    </section>
  );
};