'use client'

import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Home, ArrowLeft, Search, Music } from 'lucide-react'

export default function NotFound() {
  return (
    <div className="min-h-screen bg-neutral-900 flex items-center justify-center px-4">
      <div className="text-center max-w-md mx-auto">
        {/* 404 Number */}
        <div className="mb-8">
          <h1 className="text-9xl font-bold text-foreground font-playfair tracking-tight">
            404
          </h1>
        </div>
        
        {/* Main Message */}
        <div className="mb-8">
          <h2 className="text-2xl text-foreground font-playfair mb-3 font-inter tracking-tight">
            Page Not Found
          </h2>
          <p className="text-neutral-300 font-inter">
            The page you're looking for doesn't exist or has been moved.
          </p>
        </div>
        
        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3 justify-center mb-8">
          <Button asChild variant="default" className="font-inter">
            <Link href="/" className="flex items-center gap-2">
              <Home className="w-4 h-4" />
              Home
            </Link>
          </Button>
          <Button asChild variant="secondary" className="font-inter">
            <Link href="javascript:history.back()" className="flex items-center gap-2">
              <ArrowLeft className="w-4 h-4" />
              Go Back
            </Link>
          </Button>
        </div>

        {/* Subtle tagline */}
        <div className="opacity-60 mt-6">
            <p className="text-sm font-playfair text-neutral-400 italic">
            /ˈtambər/ · The quality of a sound that distinguishes different types of musical instruments or voices
            </p>
          </div>
      </div>
    </div>
  )
}








