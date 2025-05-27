import React from 'react';
import { GithubIcon, BookOpenIcon, ShieldIcon } from 'lucide-react';
export function Footer() {
  return <footer className="py-12 px-6 bg-gray-900 border-t border-gray-800">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <p className="text-lg font-medium text-gray-300 mb-6">
          Revolutionizing music discovery through artificial intelligence.
          </p>
          <div className="flex justify-center items-center gap-8">
            <a href="https://github.com/connergroth/Tensoe" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors duration-300">
              <GithubIcon className="w-5 h-5" />
              GitHub
            </a>
            <a href="#" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors duration-300">
              <ShieldIcon className="w-5 h-5" />
              Privacy
            </a>
            <a href="#" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors duration-300">
              <BookOpenIcon className="w-5 h-5" />
              Blog
            </a>
          </div>
        </div>
        <div className="text-center pt-8 border-t border-gray-800">
          <p className="text-gray-500 text-sm">
            © {new Date().getFullYear()} Tensoe. All rights reserved.
          </p>
          <p className="text-gray-500 text-sm">
            Tensoe (/ˈtɛn.soʊ/) — a blend of tensor and tone, uniting machine learning and music to shape sound that resonates.
          </p>
        </div>
      </div>
    </footer>;
}