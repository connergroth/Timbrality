import { Waves, Users, Headphones, Code } from "lucide-react";

export const WhyTimbre = () => {
  const features = [
    {
      icon: Headphones,
      title: "Designed to match how you listen — not just what",
      description: "Beyond simple genre matching, we understand the nuances of your musical preferences."
    },
    {
      icon: Waves,
      title: "Powered by real audio features and fan-tagged data",
      description: "Our hybrid approach combines technical audio analysis with community insights."
    },
    {
      icon: Code,
      title: "Inspired by the nuances of sound: tone, texture, and mood",
      description: "We listen to the subtle elements that make music feel right to you."
    },
    {
      icon: Users,
      title: "Built by musicians and engineers who love music as much as you do",
      description: "Our team understands both the technical and emotional sides of music discovery."
    }
  ];

  return (
    <section id="why-timbre" className="py-20 scroll-mt-24">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="font-playfair text-4xl md:text-5xl font-bold text-foreground mb-4">
            Why Timbre
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-8 lg:gap-12">
          {features.map((feature, index) => (
            <div key={index} className="flex space-x-4">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <feature.icon className="w-6 h-6 text-primary" />
              </div>
              <div className="space-y-2">
                <h3 className="font-playfair text-lg font-semibold text-foreground">
                  {feature.title}
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};