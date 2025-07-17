import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="text-center space-y-8 max-w-md">
        {/* 404 Number */}
        <div className="space-y-2">
          <h1 className="font-playfair text-8xl md:text-9xl font-bold text-primary tracking-tight">
            404
          </h1>
        </div>

        {/* Main Message */}
        <div className="space-y-4">
          <h2 className="font-playfair text-2xl md:text-3xl font-medium text-foreground">
            Page Not Found
          </h2>
          <p className="text-muted-foreground font-playfair leading-relaxed">
            The page you're looking for doesn't exist or has been moved to a different location.
          </p>
        </div>

        {/* Action Button */}
        <div className="pt-4">
          <Button 
            asChild
            size="lg"
            className="px-8 py-4 text-lg font-playfair bg-primary text-primary-foreground hover:bg-primary/90 transition-all duration-300 hover:scale-105"
          >
            <a href="/" className="flex items-center gap-2">
              <ArrowLeft className="h-5 w-5" />
              <span>Return to Home</span>
            </a>
          </Button>
        </div>

        {/* Subtle tagline */}
        <div className="pt-8 opacity-60">
          <p className="text-sm font-playfair text-muted-foreground italic">
            /ˈtambər/ · The quality of a sound that distinguishes different types of musical instruments or voices
          </p>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
