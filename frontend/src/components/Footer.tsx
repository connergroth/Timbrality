import { Github } from "lucide-react";

export const Footer = () => {
  return (
    <footer className="bg-muted/30 border-t border-border py-12">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid md:grid-cols-4 gap-8">
          <div className="space-y-4">
            <div className="font-playfair text-2xl font-bold text-primary">
              Timbre
            </div>
            <a 
              href="https://github.com/connergroth/timbre" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-2 text-muted-foreground hover:text-foreground transition-colors group"
            >
              <Github className="w-5 h-5 group-hover:scale-110 transition-transform" />
            </a>
          </div>

          <div className="space-y-4">
            <h4 className="font-playfair font-semibold text-foreground">Product</h4>
            <div className="space-y-2 text-sm">
              <div className="block">
                <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                  How it works
                </a>
              </div>
              <div className="block">
                <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                  Features
                </a>
              </div>
              <div className="block">
                <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                  Pricing
                </a>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="font-playfair font-semibold text-foreground">Company</h4>
            <div className="space-y-2 text-sm">
              <div className="block">
                <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                  About
                </a>
              </div>
              <div className="block">
                <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                  Blog
                </a>
              </div>
              <div className="block">
                <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                  Careers
                </a>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="font-playfair font-semibold text-foreground">Support</h4>
            <div className="space-y-2 text-sm">
              <div className="block">
                <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                  Help Center
                </a>
              </div>
              <div className="block">
                <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                  Contact
                </a>
              </div>
              <div className="block">
                <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                  Privacy
                </a>
              </div>
            </div>
          </div>
        </div>

        <div className="border-t border-border mt-8 pt-8 text-center">
          <p className="text-muted-foreground text-sm">
            © {new Date().getFullYear()} Timbre. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};