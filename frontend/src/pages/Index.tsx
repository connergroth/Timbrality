
import Header from '../components/Header';
import HeroSection from '../components/HeroSection';
import HowItWorksSection from '../components/HowItWorksSection';
import ConnectSection from '../components/ConnectSection';
import Footer from '../components/Footer';

const Index = () => {
  return (
    <div className="min-h-screen bg-tensoe-navy">
      <Header />
      <HeroSection />
      <HowItWorksSection />
      <ConnectSection />
      <Footer />
    </div>
  );
};

export default Index;
