'use client';

import { LandingNavbar } from "@/components/landing/LandingNavbar";
import { Hero } from "@/components/landing/Hero";
import { HowItWorks } from "@/components/landing/HowItWorks";
import { WhyTimbrality } from "@/components/landing/WhyTimbrality";
import { MusicalDNA } from "@/components/landing/MusicalDNA";
import { TastePreview } from "@/components/landing/TastePreview";
import { LandingFooter } from "@/components/landing/LandingFooter";

const LandingPage = () => {
  return (
    <div className="min-h-screen">
      <LandingNavbar />
      <Hero />
      <HowItWorks />
      <WhyTimbrality />
      <MusicalDNA />
      <TastePreview />
      <LandingFooter />
    </div>
  );
};

export default LandingPage;