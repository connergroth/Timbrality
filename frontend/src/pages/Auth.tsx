import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ArrowRight } from "lucide-react";

const Auth = () => {
  const [formData, setFormData] = useState({
    email: "",
  });

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSpotifyOAuth = () => {
    // Handle Spotify OAuth
    console.log("Spotify OAuth initiated");
  };

  const handleLastFMOAuth = () => {
    // Handle LastFM OAuth
    console.log("LastFM OAuth initiated");
  };

  const handleContinue = () => {
    // Handle form submission
    console.log("Form data:", formData);
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="font-playfair text-4xl md:text-5xl font-bold text-primary tracking-tight">
            Timbre
          </h1>
          <h2 className="font-playfair text-2xl font-medium text-foreground">
            Connect Your Music Profiles
          </h2>
          <p className="text-muted-foreground font-playfair leading-relaxed">
            Link your music accounts to discover your unique musical DNA
          </p>
        </div>

        {/* OAuth Cards */}
        <div className="space-y-3">
          {/* Spotify OAuth Card */}
          <div 
            className="bg-card border border-border rounded-xl p-3 hover:bg-accent/30 transition-colors cursor-pointer"
            onClick={handleSpotifyOAuth}
          >
            <div className="flex items-center justify-center space-x-4">
              <div className="w-6 h-6 flex items-center justify-center">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" className="text-green-500">
                  <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                </svg>
              </div>
              <span className="text-foreground font-medium">Continue with Spotify</span>
            </div>
          </div>

          {/* LastFM OAuth Card */}
          <div 
            className="bg-card border border-border rounded-xl p-3 hover:bg-accent/30 transition-colors cursor-pointer"
            onClick={handleLastFMOAuth}
          >
            <div className="flex items-center justify-center space-x-4">
              <div className="w-6 h-6 flex items-center justify-center">
                <img src="/lastfm.png" alt="Last.fm logo" className="w-5 h-3" />
              </div>
              <span className="text-foreground font-medium">Continue with Last.fm</span>
            </div>
          </div>
        </div>

        {/* OR Separator */}
        <div className="relative">
          <Separator className="bg-border" />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="bg-background px-4 text-sm text-muted-foreground font-medium">OR</span>
          </div>
        </div>

        {/* Email Input */}
        <div className="space-y-4">
          <Input
            type="email"
            placeholder="frank.ocean@example.com"
            value={formData.email}
            onChange={(e) => handleInputChange("email", e.target.value)}
            className="bg-card border-border text-foreground placeholder:text-muted-foreground rounded-xl p-3"
          />

          {/* Continue Button */}
          <Button 
            onClick={handleContinue}
            size="lg"
            className="w-full px-8 py-4 text-lg font-playfair bg-primary text-primary-foreground hover:bg-primary/90 transition-all duration-300 hover:scale-105"
          >
            <span>Continue</span>
            <ArrowRight className="h-5 w-5 ml-2" />
          </Button>
        </div>

        {/* Legal Disclaimer */}
        <p className="text-xs text-muted-foreground text-center leading-relaxed font-playfair">
          By continuing, you agree to our{" "}
          <a href="/terms" className="text-blue-500 hover:underline">
            Terms of Service
          </a>{" "}
          and{" "}
          <a href="/privacy" className="text-blue-500 hover:underline">
            Privacy Policy
          </a>
        </p>
      </div>
    </div>
  );
};

export default Auth; 
 