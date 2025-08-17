'use client'

import React from 'react'
import { LandingNavbar } from "@/components/landing/LandingNavbar"

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-neutral-900 relative">
      {/* Background overlay similar to landing page */}
      <div className="fixed inset-0 bg-neutral-900"></div>
      
      {/* Header */}
      <LandingNavbar />
      
      <div className="relative z-10 px-4 md:px-8 lg:px-12 py-16 max-w-4xl mx-auto pt-24">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="font-playfair text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6 tracking-tight">
            Terms of Service
          </h1>
          <p className="text-neutral-300 text-lg md:text-xl font-inter font-medium max-w-2xl mx-auto leading-relaxed">
            Your agreement to use Timbrality
          </p>
        </div>

        {/* Content */}
        <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-8 md:p-12 shadow-2xl">
          <div className="prose prose-invert max-w-none">
            
            <div className="mb-8">
              <p className="text-neutral-300 font-inter text-lg leading-relaxed mb-4">
                <strong className="text-white">Effective Date:</strong> August 17, 2025
              </p>
            </div>

            <p className="text-neutral-300 font-inter text-lg leading-relaxed mb-8">
              Welcome to Timbrality! These Terms of Service ("Terms") govern your use of the Timbrality app, website, and related services. By creating an account or using Timbrality, you agree to these Terms.
            </p>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                1. Eligibility
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed">
                You must be at least 13 years old to use Timbrality. By using the app, you confirm that you meet this requirement.
              </p>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                2. License to Use
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed">
                We grant you a limited, non-exclusive, non-transferable license to use Timbrality for personal, non-commercial purposes.
              </p>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                3. Accounts
              </h2>
              
              <ul className="space-y-2 text-neutral-300 font-inter leading-relaxed">
                <li>• You are responsible for maintaining the confidentiality of your account.</li>
                <li>• You are responsible for all activity under your account.</li>
                <li>• We may suspend or terminate accounts that violate these Terms.</li>
              </ul>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                4. Acceptable Use
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed mb-4">
                You agree not to:
              </p>
              
              <ul className="space-y-2 text-neutral-300 font-inter leading-relaxed">
                <li>• Use the app for unlawful purposes.</li>
                <li>• Scrape, reverse-engineer, or misuse the recommendation system.</li>
                <li>• Interfere with the security or integrity of Timbrality.</li>
              </ul>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                5. Intellectual Property
              </h2>
              
              <ul className="space-y-2 text-neutral-300 font-inter leading-relaxed">
                <li>• All code, design, algorithms, and branding are owned by Timbrality.</li>
                <li>• Spotify, Last.fm, and Album of the Year data remain the property of their respective owners.</li>
              </ul>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                6. Payments & Subscriptions
              </h2>
              
              <ul className="space-y-2 text-neutral-300 font-inter leading-relaxed">
                <li>• Premium features may be available via subscription.</li>
                <li>• Payments are processed by Stripe.</li>
                <li>• Subscriptions automatically renew unless canceled.</li>
                <li>• Refunds are handled in accordance with Stripe's policies and applicable law.</li>
              </ul>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                7. Disclaimers
              </h2>
              
              <ul className="space-y-2 text-neutral-300 font-inter leading-relaxed">
                <li>• Recommendations are provided "as is" without guarantees.</li>
                <li>• We do not guarantee continuous uptime or error-free service.</li>
              </ul>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                8. Limitation of Liability
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed">
                To the fullest extent permitted by law, Timbrality is not liable for indirect, incidental, or consequential damages, including lost playlists or data.
              </p>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                9. Termination
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed">
                We may suspend or terminate your account if you violate these Terms. You may also delete your account at any time.
              </p>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                10. Governing Law
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed">
                These Terms are governed by the laws of the State of Colorado, USA.
              </p>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                11. Changes to Terms
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed">
                We may update these Terms at any time. Continued use of the app after updates means you accept the revised Terms.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                12. Contact
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed">
                <a href="mailto:hello@timbrality.com" className="text-neutral-400 hover:text-white transition-colors">hello@timbrality.com</a>
              </p>
            </section>

          </div>
        </div>
      </div>
    </div>
  )
}