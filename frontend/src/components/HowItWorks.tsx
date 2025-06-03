import React from 'react';
import { BrainIcon, UsersIcon, StarIcon } from 'lucide-react';
import { useScrollAnimation } from '../hooks/useScrollAnimation';

export function HowItWorks() {
  const sectionRef = useScrollAnimation();
  const features = [{
    icon: <BrainIcon className="w-8 h-8" />,
    title: 'Content-Based Analysis',
    description: 'Our AI analyzes audio features, genres, and musical patterns to understand what makes each song unique.'
  }, {
    icon: <UsersIcon className="w-8 h-8" />,
    title: 'Collaborative Filtering',
    description: 'Learn from millions of listening patterns to discover music loved by people with similar tastes.'
  }, {
    icon: <StarIcon className="w-8 h-8" />,
    title: 'Personal Ratings',
    description: 'Your feedback continuously improves recommendations, creating a truly personalized experience.'
  }];

  return <section ref={sectionRef} id="how-it-works" className="py-20 px-6 bg-gray-800/50 min-h-[50vh]">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-gray-100 scroll-fade-up">
            How It Works
          </h2>
          <p className="text-lg text-gray-300 max-w-2xl mx-auto scroll-fade-up scroll-delay-100">
            Our hybrid recommendation engine combines collaborative filtering (NMF), content-based filtering (TF-IDF & audio features), and metadata from Last.fm, Spotify, and Albumoftheyear.org to create a hybrid recommendation model that truly resonates with you.
          </p>
        </div>
        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => <div key={index} className="text-center p-6 rounded-xl bg-gray-900/50 backdrop-blur-sm border border-gray-700/50 hover:border-blue-500/50 transition-all duration-300 scroll-fade-up" style={{ transitionDelay: `${(index + 2) * 100}ms` }}>
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-[#00C6FF] to-[#0099FF] mb-6">
                {feature.icon}
              </div>
              <h3 className="text-xl font-semibold mb-4 text-gray-100">
                {feature.title}
              </h3>
              <p className="text-gray-300 leading-relaxed">
                {feature.description}
              </p>
            </div>)}
        </div>
      </div>
    </section>;
}