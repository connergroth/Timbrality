import { useEffect, useRef } from 'react';

export function useScrollAnimation() {
  const elementRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            // Add visible class to all elements with scroll-fade-up within the section
            const section = entry.target as HTMLElement;
            const animatedElements = section.querySelectorAll('.scroll-fade-up');
            animatedElements.forEach((el) => {
              el.classList.add('visible');
            });
          }
        });
      },
      {
        threshold: 0.15, // Increased threshold for earlier triggering
        rootMargin: '-50px 0px' // Start animation slightly before element is fully in view
      }
    );

    const currentElement = elementRef.current;
    if (currentElement) {
      observer.observe(currentElement);
    }

    return () => {
      if (currentElement) {
        observer.unobserve(currentElement);
      }
    };
  }, []);

  return elementRef;
} 