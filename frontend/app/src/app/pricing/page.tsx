'use client'

import React, { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Check } from "lucide-react"
import { LandingNavbar } from "@/components/landing/LandingNavbar"

export default function PricingPage() {
  const [isYearly, setIsYearly] = useState(false)

  const features = {
    free: [
      "Basic music recommendations",
      "Connect Spotify account", 
      "Browse album database",
      "Basic preference learning"
    ],
    premium: [
      "Advanced AI recommendations",
      "Unlimited recommendation requests",
      "Last.fm integration",
      "Custom playlist generation",
      "Detailed music analysis",
      "Priority support",
      "Export recommendations"
    ]
  }

  return (
    <div className="min-h-screen bg-neutral-900 relative">
      {/* Background overlay similar to landing page */}
      <div className="fixed inset-0 bg-neutral-900"></div>
      
      {/* Header */}
      <LandingNavbar />
      
      <div className="relative z-10 px-4 md:px-8 lg:px-12 py-16 max-w-7xl mx-auto pt-24">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="font-playfair text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6 tracking-tight">
            Pricing
          </h1>
          <p className="text-neutral-300 text-lg md:text-xl font-inter font-medium max-w-2xl mx-auto leading-relaxed">
            Choose the plan that works for you
          </p>
          
          {/* Toggle */}
          <div className="flex items-center justify-center gap-6 mt-12 mb-8">
            <span className={`font-inter font-medium text-lg ${!isYearly ? 'text-white' : 'text-neutral-400'}`}>
              Monthly
            </span>
            <button
              onClick={() => setIsYearly(!isYearly)}
              className={`relative inline-flex h-7 w-12 items-center rounded-full transition-all duration-300 ${
                isYearly ? 'bg-white' : 'bg-neutral-700'
              }`}
            >
              <span
                className={`inline-block h-5 w-5 transform rounded-full transition-transform duration-300 ${
                  isYearly ? 'translate-x-6 bg-neutral-900' : 'translate-x-1 bg-white'
                }`}
              />
            </button>
            <span className={`font-inter font-medium text-lg ${isYearly ? 'text-white' : 'text-neutral-400'}`}>
              Yearly
              <span className="ml-2 text-sm text-green-400 font-normal">(Save 20%)</span>
            </span>
          </div>
        </div>

        {/* Pricing Cards */}
        <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
          {/* Free Plan */}
          <div className="relative bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-2xl p-6 shadow-2xl transition-all duration-300 hover:border-neutral-600/50">
            <div className="mb-6">
              <div className="flex items-center gap-3 mb-3">
                <h2 className="font-playfair text-2xl font-bold text-white">Free</h2>
                <span className="inline-flex items-center rounded-full bg-neutral-700 px-2.5 py-0.5 text-xs font-medium text-neutral-300 font-inter">
                  Beta
                </span>
              </div>
              <div className="font-playfair text-4xl font-bold text-white mb-1">
                Free
              </div>
              <p className="text-neutral-400 font-inter text-sm font-medium">Includes</p>
            </div>

            <div className="space-y-4 mb-6">
              <ul className="space-y-3">
                {features.free.map((feature, index) => (
                  <li key={index} className="flex items-center gap-3">
                    <Check className="h-4 w-4 text-green-400 flex-shrink-0" />
                    <span className="text-neutral-200 font-inter text-sm font-medium">{feature}</span>
                  </li>
                ))}
              </ul>
            </div>

            <Button 
              variant="outline" 
              className="w-full py-3 font-inter border-neutral-600 text-neutral-300 hover:bg-neutral-700 hover:text-white transition-all duration-300"
            >
              Current Plan
            </Button>
          </div>

          {/* Premium Plan */}
          <div className="relative bg-gradient-to-br from-neutral-800/60 to-neutral-900/80 backdrop-blur-xl border border-white/20 rounded-2xl p-6 shadow-2xl border-glow-animation overflow-hidden">
            {/* Gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-transparent to-blue-500/10 pointer-events-none"></div>
            
            <div className="relative z-10">
              <div className="mb-6">
                <div className="flex items-center gap-2 mb-3 flex-wrap">
                  <h2 className="font-playfair text-2xl font-bold text-white">Premium</h2>
                  <span className="inline-flex items-center rounded-full bg-white/20 backdrop-blur-sm px-2.5 py-0.5 text-xs font-medium text-white font-inter">
                    Beta
                  </span>
                  <span className="inline-flex items-center rounded-full bg-purple-600/80 backdrop-blur-sm px-2.5 py-0.5 text-xs font-medium text-white font-inter">
                    Free Trial
                  </span>
                </div>
                <div className="font-playfair text-4xl font-bold text-white mb-1">
                  ${isYearly ? '10' : '1'}
                  <span className="text-xl font-normal text-neutral-300">
                    /{isYearly ? 'year' : 'mo'}
                  </span>
                </div>
                {isYearly && (
                  <p className="text-xs text-neutral-300 font-inter font-medium">
                    Free for 7 days • Credit card required
                  </p>
                )}
                <p className="text-neutral-300 font-inter text-sm font-medium mt-1">
                  Everything in Free, plus
                </p>
              </div>

              <div className="space-y-4 mb-6">
                <ul className="space-y-3">
                  {features.premium.map((feature, index) => (
                    <li key={index} className="flex items-center gap-3">
                      <Check className="h-4 w-4 text-green-400 flex-shrink-0" />
                      <span className="text-neutral-200 font-inter text-sm font-medium">{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <Button className="w-full py-3 font-inter bg-white text-neutral-900 hover:bg-neutral-100 transition-all duration-300 hover:scale-105">
                Start for Free
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}