import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
		"./index.html",
		"./src/**/*.{js,jsx,ts,tsx}",
		"./public/**/*.html",
		// More explicit patterns for production builds
		"./src/components/**/*.{js,ts,jsx,tsx}",
		"./src/pages/**/*.{js,ts,jsx,tsx}",
		"./**/*.{html,js,ts,jsx,tsx}",
	],
	safelist: [
		// Custom component classes defined in CSS
		"btn-primary",
		"btn-secondary", 
		"feature-card",
		"gradient-text",
		"shader-bg",
		
		// Custom animation classes
		"animate-fade-in",
		"animate-wave", 
		"animate-pulse-slow",
		"animate-float",
		
		// Custom tensoe brand colors
		"bg-tensoe-navy",
		"bg-tensoe-navy-light",
		"bg-tensoe-blue",
		"bg-tensoe-blue-light",
		"text-tensoe-navy",
		"text-tensoe-navy-light", 
		"text-tensoe-blue",
		"text-tensoe-blue-light",
		"border-tensoe-blue",
		"border-tensoe-navy",
		
		// Icon and text colors that might be dynamic
		"text-black",
		"text-white",
		"text-gray-300",
		"text-gray-400",
		"text-gray-100",
		
		// Background colors commonly used
		"bg-black",
		"bg-white", 
		"bg-gray-300",
		"bg-gray-400",
		"bg-gray-100",
		"bg-gray-800",
		"bg-gray-900",
		
		// Common utility classes
		"opacity-10",
		"opacity-20",
		"opacity-60",
		"w-1",
		"w-2",
		"h-16",
		"h-20",
		"rounded-full",
		
		// Hover states
		"hover:bg-gray-400",
		"hover:text-white",
		"hover:border-gray-500",
		
		// Animation delay patterns
		{
			pattern: /animate-.*/,
		},
		{
			pattern: /bg-tensoe-.*/,
		},
		{
			pattern: /text-tensoe-.*/,
		},
		{
			pattern: /border-tensoe-.*/,
		},
		// All possible animation delays and durations
		{
			pattern: /delay-.*/,
		},
		{
			pattern: /duration-.*/,
		}
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				sidebar: {
					DEFAULT: 'hsl(var(--sidebar-background))',
					foreground: 'hsl(var(--sidebar-foreground))',
					primary: 'hsl(var(--sidebar-primary))',
					'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
					accent: 'hsl(var(--sidebar-accent))',
					'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
					border: 'hsl(var(--sidebar-border))',
					ring: 'hsl(var(--sidebar-ring))'
				},
				tensoe: {
					navy: '#0B1426',
					'navy-light': '#1A2332',
					blue: '#60A5FA',
					'blue-light': '#93C5FD',
					white: '#FFFFFF'
				}
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: {
						height: '0'
					},
					to: {
						height: 'var(--radix-accordion-content-height)'
					}
				},
				'accordion-up': {
					from: {
						height: 'var(--radix-accordion-content-height)'
					},
					to: {
						height: '0'
					}
				},
				'wave': {
					'0%, 100%': { transform: 'scaleY(1)' },
					'50%': { transform: 'scaleY(1.5)' }
				},
				'pulse-slow': {
					'0%, 100%': { opacity: '0.4' },
					'50%': { opacity: '0.8' }
				},
				'float': {
					'0%, 100%': { transform: 'translateY(0px)' },
					'50%': { transform: 'translateY(-10px)' }
				},
				'fade-in': {
					from: {
						opacity: '0',
						transform: 'translateY(20px)'
					},
					to: {
						opacity: '1',
						transform: 'translateY(0)'
					}
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'wave': 'wave 2s ease-in-out infinite',
				'pulse-slow': 'pulse-slow 3s ease-in-out infinite',
				'float': 'float 6s ease-in-out infinite',
				'fade-in': 'fade-in 1s ease-out forwards'
			}
		}
	},
	plugins: [require("tailwindcss-animate")],
} satisfies Config;
