'use client'

import React from 'react'
import { LandingNavbar } from "@/components/landing/LandingNavbar"

export default function PrivacyPage() {
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
            Privacy Policy
          </h1>
          <p className="text-neutral-300 text-lg md:text-xl font-inter font-medium max-w-2xl mx-auto leading-relaxed">
            How we protect and handle your data
          </p>
        </div>

        {/* Content */}
        <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-3xl p-8 md:p-12 shadow-2xl">
          <div className="prose prose-invert max-w-none">
            
            <div className="mb-8">
              <p className="text-neutral-300 font-inter text-lg leading-relaxed mb-4">
                <strong className="text-white">Effective Date:</strong> August 17, 2025
              </p>
              <p className="text-neutral-300 font-inter text-lg leading-relaxed">
                <strong className="text-white">Last Updated:</strong> August 17, 2025
              </p>
            </div>

            <p className="text-neutral-300 font-inter text-lg leading-relaxed mb-8">
              Timbrality ("we," "our," or "us") values your privacy. This Privacy Policy explains what information we collect, how we use it, and your rights as a user of the Timbrality app and website.
            </p>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                1. Information We Collect
              </h2>
              
              <div className="space-y-4">
                <div>
                  <h3 className="text-xl font-inter font-semibold text-white mb-2">Account Information:</h3>
                  <p className="text-neutral-300 font-inter leading-relaxed">
                    When you sign up, we may collect your name, email address, and linked accounts (e.g., Spotify, Last.fm).
                  </p>
                </div>

                <div>
                  <h3 className="text-xl font-inter font-semibold text-white mb-2">OAuth Tokens:</h3>
                  <p className="text-neutral-300 font-inter leading-relaxed">
                    Access tokens provided by Spotify or Last.fm to connect your music accounts.
                  </p>
                </div>

                <div>
                  <h3 className="text-xl font-inter font-semibold text-white mb-2">Usage Data:</h3>
                  <p className="text-neutral-300 font-inter leading-relaxed">
                    Listening history, ratings, recommendations viewed, and app interactions.
                  </p>
                </div>

                <div>
                  <h3 className="text-xl font-inter font-semibold text-white mb-2">Device & Log Data:</h3>
                  <p className="text-neutral-300 font-inter leading-relaxed">
                    IP address, device type, operating system, crash reports, and analytics.
                  </p>
                </div>

                <div>
                  <h3 className="text-xl font-inter font-semibold text-white mb-2">Payment Information:</h3>
                  <p className="text-neutral-300 font-inter leading-relaxed">
                    If you subscribe to premium features, billing is handled securely through Stripe. We do not store your payment card details.
                  </p>
                </div>
              </div>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                2. How We Use Information
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed mb-4">
                We use your information to:
              </p>
              
              <ul className="space-y-2 text-neutral-300 font-inter leading-relaxed">
                <li>• Provide personalized music recommendations.</li>
                <li>• Sync your music data across services (Spotify, Last.fm, Album of the Year).</li>
                <li>• Improve app features, algorithms, and user experience.</li>
                <li>• Process payments and manage subscriptions.</li>
                <li>• Communicate updates, account info, or important service changes.</li>
              </ul>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                3. Sharing of Information
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed mb-4">
                <strong className="text-white">We do not sell your personal data.</strong>
              </p>
              
              <p className="text-neutral-300 font-inter leading-relaxed mb-4">
                We may share data with:
              </p>
              
              <div className="space-y-4">
                <div>
                  <h3 className="text-xl font-inter font-semibold text-white mb-2">Service Providers:</h3>
                  <p className="text-neutral-300 font-inter leading-relaxed">
                    Hosting (e.g., Supabase, Vercel), analytics, Stripe for billing.
                  </p>
                </div>

                <div>
                  <h3 className="text-xl font-inter font-semibold text-white mb-2">Third-Party APIs:</h3>
                  <p className="text-neutral-300 font-inter leading-relaxed">
                    Spotify, Last.fm, Album of the Year to retrieve or sync music data.
                  </p>
                </div>

                <div>
                  <h3 className="text-xl font-inter font-semibold text-white mb-2">Legal Authorities:</h3>
                  <p className="text-neutral-300 font-inter leading-relaxed">
                    If required by law.
                  </p>
                </div>
              </div>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                4. Data Security
              </h2>
              
              <ul className="space-y-2 text-neutral-300 font-inter leading-relaxed">
                <li>• Data is encrypted in transit and at rest.</li>
                <li>• OAuth tokens are stored securely.</li>
                <li>• We regularly monitor our systems for vulnerabilities.</li>
              </ul>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                5. Your Rights
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed mb-4">
                Depending on your location:
              </p>
              
              <div className="space-y-4">
                <div>
                  <h3 className="text-xl font-inter font-semibold text-white mb-2">GDPR (EU users):</h3>
                  <p className="text-neutral-300 font-inter leading-relaxed">
                    You have the right to access, correct, delete, and export your data.
                  </p>
                </div>

                <div>
                  <h3 className="text-xl font-inter font-semibold text-white mb-2">CCPA/CPRA (California users):</h3>
                  <p className="text-neutral-300 font-inter leading-relaxed">
                    You may opt out of "data selling" (we do not sell data, but provide this right as required).
                  </p>
                </div>

                <div>
                  <h3 className="text-xl font-inter font-semibold text-white mb-2">General Rights:</h3>
                  <p className="text-neutral-300 font-inter leading-relaxed">
                    You may request account deletion by contacting [your support email].
                  </p>
                </div>
              </div>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                6. Children's Privacy
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed">
                Timbrality is not intended for users under 13. We do not knowingly collect data from children under 13.
              </p>
            </section>

            <section className="mb-10">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                7. Changes to This Policy
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed">
                We may update this Privacy Policy from time to time. We will notify you of significant changes via email or in-app notice.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="font-playfair text-3xl font-bold text-white mb-6 tracking-tight">
                8. Contact Us
              </h2>
              
              <p className="text-neutral-300 font-inter leading-relaxed mb-4">
                If you have questions, contact us at:
              </p>
              
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