'use client'

import React from 'react'
import { LandingNavbar } from "@/components/landing/LandingNavbar"
import { Button } from "@/components/ui/button"
import { Check, Music, Zap, Users, Shield, Github, Linkedin } from "lucide-react"

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-neutral-900 relative">
      {/* Background overlay similar to landing page */}
      <div className="fixed inset-0 bg-neutral-900"></div>
      
      {/* Header */}
      <LandingNavbar />
      
      <div className="relative z-10 px-4 md:px-8 lg:px-12 py-16 max-w-6xl mx-auto pt-24">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="font-playfair text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6 tracking-tight">
            About Timbrality
          </h1>
          <p className="text-neutral-300 text-lg md:text-xl font-inter font-medium max-w-2xl mx-auto leading-relaxed">
            Your personal music guide for intelligent discovery
          </p>
        </div>

        {/* Mission Statement */}
        <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-8 md:p-12 shadow-2xl mb-12">
          <div className="text-center">
            <h2 className="font-playfair text-3xl md:text-4xl font-bold text-white mb-6 tracking-tight">
              Our Mission
            </h2>
            <p className="text-neutral-300 font-inter text-xl leading-relaxed max-w-4xl mx-auto">
              Timbrality is your personal music guide — helping you discover new sounds through intelligent recommendations, curated vibes, and data-driven insights.
            </p>
          </div>
        </div>

        {/* What Timbrality Does */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-8 shadow-2xl">
            <div className="flex items-center gap-3 mb-6">
              <Music className="h-8 w-8 text-green-400" />
              <h2 className="font-playfair text-3xl font-bold text-white tracking-tight">
                What We Do
              </h2>
            </div>
            
            <ul className="space-y-4 text-neutral-300 font-inter leading-relaxed">
              <li className="flex items-start gap-3">
                <Check className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                <span>Personalized recommendations powered by hybrid ML models (collaborative + content-based)</span>
              </li>
              <li className="flex items-start gap-3">
                <Check className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                <span>Direct integration with Spotify and Last.fm</span>
              </li>
              <li className="flex items-start gap-3">
                <Check className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                <span>Ratings and critic data from Album of the Year</span>
              </li>
              <li className="flex items-start gap-3">
                <Check className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                <span>Weekly discovery (new releases, top-rated albums, hidden gems)</span>
              </li>
              <li className="flex items-start gap-3">
                <Check className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                <span>Listening analytics and Spotify playlist export</span>
              </li>
            </ul>
          </div>

          {/* Why We're Different */}
          <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-8 shadow-2xl">
            <div className="flex items-center gap-3 mb-6">
              <Zap className="h-8 w-8 text-purple-400" />
              <h2 className="font-playfair text-3xl font-bold text-white tracking-tight">
                Why We're Different
              </h2>
            </div>
            
            <ul className="space-y-4 text-neutral-300 font-inter leading-relaxed">
              <li className="flex items-start gap-3">
                <Check className="h-5 w-5 text-purple-400 flex-shrink-0 mt-0.5" />
                <span>Unlike Spotify Wrapped (once a year), Timbrality gives real-time insights</span>
              </li>
              <li className="flex items-start gap-3">
                <Check className="h-5 w-5 text-purple-400 flex-shrink-0 mt-0.5" />
                <span>Combines community-driven data (Last.fm, AOTY) with ML recommendations</span>
              </li>
              <li className="flex items-start gap-3">
                <Check className="h-5 w-5 text-purple-400 flex-shrink-0 mt-0.5" />
                <span>Feels like a personal music agent, not just a playlist generator</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Who Built It */}
        <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-8 md:p-12 shadow-2xl mb-12">
          <div className="flex items-center gap-3 mb-8 justify-center">
            <Users className="h-8 w-8 text-blue-400" />
            <h2 className="font-playfair text-3xl md:text-4xl font-bold text-white tracking-tight">
              Who Built It
            </h2>
          </div>
          
          <div className="text-center max-w-3xl mx-auto">
            <p className="text-neutral-300 font-inter text-lg leading-relaxed mb-6">
              Timbrality was created by <strong className="text-white">Conner Groth</strong>, a Computer Science student at CU Boulder and builder of AI-powered applications. It's part of his ongoing work to merge machine learning with real-world creative discovery.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                variant="outline" 
                className="border-neutral-600 text-neutral-300 hover:bg-neutral-700 hover:text-white transition-all duration-300"
                asChild
              >
                <a href="https://github.com/connergroth" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2">
                  <Github className="h-5 w-5" />
                  View GitHub
                </a>
              </Button>
              <Button 
                variant="outline" 
                className="border-neutral-600 text-neutral-300 hover:bg-neutral-700 hover:text-white transition-all duration-300"
                asChild
              >
                <a href="https://linkedin.com/in/connergroth" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2">
                  <Linkedin className="h-5 w-5" />
                  Connect on LinkedIn
                </a>
              </Button>
            </div>
          </div>
        </div>

        {/* Ethics & Privacy */}
        <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-8 md:p-12 shadow-2xl mb-12">
          <div className="flex items-center gap-3 mb-8 justify-center">
            <Shield className="h-8 w-8 text-emerald-400" />
            <h2 className="font-playfair text-3xl md:text-4xl font-bold text-white tracking-tight">
              Ethics & Privacy Commitment
            </h2>
          </div>
          
          <div className="text-center max-w-3xl mx-auto">
            <p className="text-neutral-300 font-inter text-lg leading-relaxed mb-6">
              Your data is yours. Timbrality never sells personal data — we only use it to improve your recommendations. Learn more in our{' '}
              <a href="/privacy" className="text-blue-400 hover:text-blue-300 transition-colors underline">
                Privacy Policy
              </a>.
            </p>
          </div>
        </div>

        {/* Call to Action */}
        <div className="bg-gradient-to-br from-neutral-800/60 to-neutral-900/80 backdrop-blur-xl border border-white/20 rounded-3xl p-8 md:p-12 shadow-2xl border-glow-animation text-center">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-transparent to-blue-500/10 pointer-events-none rounded-3xl"></div>
          
          <div className="relative z-10">
            <h2 className="font-playfair text-3xl md:text-4xl font-bold text-white mb-6 tracking-tight">
              Ready to Discover?
            </h2>
            <p className="text-neutral-300 font-inter text-lg leading-relaxed mb-8 max-w-2xl mx-auto">
              Connect your Spotify and start discovering music that truly resonates with your unique taste.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                className="bg-white text-neutral-900 hover:bg-neutral-100 transition-all duration-300 hover:scale-105 px-8 py-3 text-lg font-inter"
                asChild
              >
                <a href="/auth">Get Started</a>
              </Button>
              <Button 
                variant="outline" 
                className="border-white text-white hover:bg-white hover:text-neutral-900 transition-all duration-300 hover:scale-105 px-8 py-3 text-lg font-inter"
                asChild
              >
                <a href="/pricing">View Pricing</a>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}