import { Badge } from "@/components/ui/badge";

export const TastePreview = () => {
  const tags = ["Jazz", "Hip-Hop", "R&B", "Lo-Fi"];

  return (
    <section id="your-dna" className="py-20 scroll-mt-24">
      <div className="max-w-4xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="font-playfair text-4xl md:text-5xl font-bold text-foreground mb-4">
            Taste preview
          </h2>
          <p className="text-xl text-muted-foreground font-playfair">
            Here's what personalized discovery looks like
          </p>
        </div>

        <div className="bg-card border border-border rounded-2xl p-8 space-y-6">
          <div className="flex items-start space-x-4">
            <div className="w-16 h-16 bg-gradient-to-br from-primary/20 to-primary-glow/20 rounded-lg flex items-center justify-center">
              <span className="text-2xl font-playfair font-bold text-primary">T</span>
            </div>
            <div className="flex-1 space-y-2">
              <h3 className="font-playfair text-xl font-semibold text-foreground">
                Kendrick Lamar - Sing About Me, I'm Dying of Thirst
              </h3>
              <p className="text-muted-foreground italic">
                "You might like this because of your love for jazz and hip-hop."
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {tags.map((tag) => (
              <Badge key={tag} variant="secondary" className="font-playfair">
                {tag}
              </Badge>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};