import { Navbar } from "@/components/Navbar";
import { Hero } from "@/components/Hero";
import { HowItWorks } from "@/components/HowItWorks";
import { WhyTimbrality } from "@/components/WhyTimbrality";
import { MusicalDNA } from "@/components/MusicalDNA";
import { TastePreview } from "@/components/TastePreview";
import { Footer } from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen">
      <Navbar />
      <Hero />
      <HowItWorks />
      <WhyTimbrality />
      <MusicalDNA />
      <TastePreview />
      <Footer />
    </div>
  );
};

export default Index;
