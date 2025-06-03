import { Github } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  return (
    <footer className="bg-tensoe-navy border-t border-tensoe-blue/20 py-12">
      <div className="container mx-auto px-6">
        <div className="text-center mb-8">
          <h3 className="text-2xl font-bold gradient-text mb-4">
            Crafted for deep listening. Powered by intelligent sound.
          </h3>
        </div>

        <div className="flex flex-col md:flex-row justify-between items-center mb-8">
          <div className="flex items-center space-x-8 mb-4 md:mb-0">
            <a href="https://github.com/connergroth/tensoe" target="_blank" rel="noopener noreferrer" className="text-tensoe-blue-light hover:text-tensoe-blue transition-colors flex items-center space-x-2">
              <Github size={20} />
              <span>GitHub</span>
            </a>
            <a href="/privacy" className="text-tensoe-blue-light hover:text-tensoe-blue transition-colors">
              Privacy
            </a>
            <a href="https://connergroth.com/blog/tensoe" className="text-tensoe-blue-light hover:text-tensoe-blue transition-colors">
              Blog
            </a>
          </div>
        </div>

        <div className="border-t border-tensoe-blue/20 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center text-sm text-gray-400">
            <div className="text-center md:text-left">
              <p>&copy; {currentYear} Tensoe. All rights reserved.</p>
              <p className="mt-1">
                By <a href="https://connergroth.com" target="_blank" rel="noopener noreferrer" className="text-tensoe-blue-light hover:text-tensoe-blue transition-colors">Conner Groth</a>
              </p>
            </div>
            <p className="mt-2 md:mt-0 max-w-md text-center md:text-right">
              Tensoe (/ˈtɛn.soʊ/) — a fusion of tensor and tone, using machine learning to shape resonant sound.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
