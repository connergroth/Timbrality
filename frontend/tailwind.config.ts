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
	// Disable purging in production to ensure identical dev/prod styles
	purge: false,
	// Include ALL classes to prevent any differences between dev and prod
	safelist: [
		// Ensure ALL classes are preserved
		{
			pattern: /.*/,
		},
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
		
		// ALL tensoe brand colors variations
		"bg-tensoe-navy",
		"bg-tensoe-navy-light",
		"bg-tensoe-blue",
		"bg-tensoe-blue-light",
		"bg-tensoe-white",
		"text-tensoe-navy",
		"text-tensoe-navy-light", 
		"text-tensoe-blue",
		"text-tensoe-blue-light",
		"text-tensoe-white",
		"border-tensoe-blue",
		"border-tensoe-navy",
		"border-tensoe-navy-light",
		"border-tensoe-blue-light",
		"border-tensoe-white",
		
		// ALL color variations to prevent any purging
		"text-black",
		"text-white",
		"text-gray-100",
		"text-gray-200",
		"text-gray-300",
		"text-gray-400",
		"text-gray-500",
		"text-gray-600",
		"text-gray-700",
		"text-gray-800",
		"text-gray-900",
		"bg-black",
		"bg-white", 
		"bg-gray-100",
		"bg-gray-200",
		"bg-gray-300",
		"bg-gray-400",
		"bg-gray-500",
		"bg-gray-600",
		"bg-gray-700",
		"bg-gray-800",
		"bg-gray-900",
		"bg-transparent",
		
		// ALL size variations
		"w-1", "w-2", "w-3", "w-4", "w-5", "w-6", "w-7", "w-8", "w-9", "w-10",
		"w-11", "w-12", "w-14", "w-16", "w-20", "w-24", "w-32", "w-36", "w-64", "w-72",
		"h-1", "h-2", "h-3", "h-4", "h-5", "h-6", "h-7", "h-8", "h-9", "h-10",
		"h-11", "h-12", "h-14", "h-16", "h-20", "h-24", "h-32", "h-36", "h-64", "h-72",
		
		// ALL opacity variations
		"opacity-0", "opacity-10", "opacity-20", "opacity-30", "opacity-40", "opacity-50",
		"opacity-60", "opacity-70", "opacity-80", "opacity-90", "opacity-100",
		
		// ALL border radius variations
		"rounded-none", "rounded-sm", "rounded", "rounded-md", "rounded-lg", "rounded-xl", "rounded-2xl", "rounded-3xl", "rounded-full",
		
		// ALL hover states
		"hover:bg-gray-100", "hover:bg-gray-200", "hover:bg-gray-300", "hover:bg-gray-400", "hover:bg-gray-500",
		"hover:text-white", "hover:text-black", "hover:text-gray-100", "hover:text-gray-300", "hover:text-gray-400",
		"hover:border-gray-400", "hover:border-gray-500", "hover:border-gray-600",
		"hover:scale-105", "hover:transform", "hover:opacity-100",
		
		// ALL animation and transition classes
		{
			pattern: /^animate-.*/,
		},
		{
			pattern: /^transition-.*/,
		},
		{
			pattern: /^duration-.*/,
		},
		{
			pattern: /^delay-.*/,
		},
		{
			pattern: /^ease-.*/,
		},
		// ALL color pattern variations
		{
			pattern: /^bg-tensoe-.*/,
		},
		{
			pattern: /^text-tensoe-.*/,
		},
		{
			pattern: /^border-tensoe-.*/,
		},
		{
			pattern: /^bg-.*/,
		},
		{
			pattern: /^text-.*/,
		},
		{
			pattern: /^border-.*/,
		},
		{
			pattern: /^hover:.*/,
		},
		{
			pattern: /^focus:.*/,
		},
		{
			pattern: /^active:.*/,
		},
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
