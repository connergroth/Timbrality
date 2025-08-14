'use client'

import { Button } from '@/components/ui/button'
import { Home, RefreshCw, AlertTriangle } from 'lucide-react'
import Link from 'next/link'

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <html>
      <body>
        <div className="min-h-screen bg-background flex items-center justify-center px-4">
          <div className="text-center max-w-md mx-auto">
            {/* Error Icon */}
            <div className="mb-8">
              <div className="w-24 h-24 bg-destructive/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <AlertTriangle className="w-12 h-12 text-destructive" />
              </div>
            </div>
            
            {/* Main Message */}
            <div className="mb-8">
              <h2 className="text-2xl text-primary font-playfair mb-3 tracking-tight">
                Something Went Wrong
              </h2>
              <p className="text-muted-foreground font-inter mb-4">
                An unexpected error occurred. Please try again or contact support if the problem persists.
              </p>
              {error.digest && (
                <p className="text-xs text-muted-foreground font-mono bg-muted px-2 py-1 rounded">
                  Error ID: {error.digest}
                </p>
              )}
            </div>
            
            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-3 justify-center mb-8">
              <Button onClick={reset} variant="default" className="font-inter">
                <RefreshCw className="w-4 h-4 mr-2" />
                Try Again
              </Button>
              
              <Button asChild variant="secondary" className="font-inter">
                <Link href="/" className="flex items-center gap-2">
                  <Home className="w-4 h-4" />
                  Home
                </Link>
              </Button>
            </div>

            {/* Subtle tagline */}
            <div className="opacity-60 mt-6">
              <p className="text-sm font-playfair text-muted-foreground italic">
                /ˈtambər/ · The quality of a sound that distinguishes different types of musical instruments or voices
              </p>
            </div>
          </div>
        </div>
      </body>
    </html>
  )
}
