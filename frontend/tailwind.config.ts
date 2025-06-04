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
	],
	safelist: [
		// Core Tailwind utilities - be more specific to avoid warnings
		{
			pattern: /^(bg|text|border|rounded|px|py|mx|my|mt|mb|ml|mr|pt|pb|pl|pr|w|h|min-w|min-h|max-w|max-h|flex|grid|gap|space|items|justify|absolute|relative|fixed|static|top|bottom|left|right|inset|z|opacity|transform|transition|duration|ease|scale|translate|rotate|cursor|overflow|shadow|blur|backdrop)-.+/,
		},
		{
			pattern: /^animate-.+/,
		},
		// Ensure custom tensoe colors are always generated
		{
			pattern: /^(bg|text|border|ring|from|to|via)-tensoe-(navy|navy-light|blue|blue-light|white)$/,
		},
		{
			pattern: /^(bg|text|border|ring|from|to|via)-tensoe-(navy|navy-light|blue|blue-light|white)\/(10|20|30|40|50|60|70|80|90)$/,
		},
		// Animation classes
		'animate-fade-in',
		'animate-pulse',
		'animate-pulse-slow',
		'animate-wave',
		'animate-spin',
		// Custom CSS classes
		'gradient-text',
		'shader-bg',
		'btn-primary',
		'btn-secondary',
		'feature-card',
		// Common layout classes that might get purged
		'container',
		'mx-auto',
		'min-h-screen',
		'relative',
		'absolute',
		'inset-0',
		'z-10',
		'opacity-10',
		'opacity-60',
		'overflow-hidden',
		// Grid and flex classes
		'grid',
		'md:grid-cols-3',
		'gap-6',
		'gap-8',
		'flex',
		'flex-col',
		'sm:flex-row',
		'items-center',
		'justify-center',
		'text-center',
		// Typography classes
		'text-lg',
		'text-xl',
		'text-2xl',
		'text-4xl',
		'text-6xl',
		'md:text-3xl',
		'md:text-5xl',
		'md:text-8xl',
		'font-bold',
		'font-semibold',
		'leading-relaxed',
		// Colors that might get purged
		'text-white',
		'text-gray-300',
		'text-gray-400',
		'text-black',
		'bg-transparent',
		// Spacing classes
		'mb-4',
		'mb-6',
		'mb-12',
		'mb-16',
		'mt-8',
		'mt-16',
		'px-4',
		'px-6',
		'px-8',
		'py-2',
		'py-4',
		'py-20',
		'max-w-2xl',
		'max-w-4xl',
		// Border and radius
		'rounded-full',
		'rounded-lg',
		'border',
		'border-gray-600',
		'hover:border-gray-500',
		// Transform and transition
		'transform',
		'transition-all',
		'transition-colors',
		'transition-transform',
		'duration-300',
		'hover:text-white',
		'hover:translate-x-1',
		'group-hover:translate-x-1',
		'scale-150',
		'-translate-x-1/2',
		'-translate-y-1/2',
		// Position classes
		'top-1/4',
		'left-1/4',
		'right-1/4',
		'bottom-1/4',
		'left-1/3',
		'top-1/3',
		'top-1/2',
		'left-1/2',
		// Size classes
		'w-1',
		'w-2',
		'w-16',
		'w-24',
		'w-32',
		'w-36',
		'w-px',
		'h-16',
		'h-20',
		'h-32',
		'h-36',
		'h-px',
		'h-1.5',
		'h-4',
		'w-4',
		// Blur and effects
		'blur',
		'blur-sm',
		'backdrop-blur-sm',
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
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'wave': 'wave 2s ease-in-out infinite',
				'pulse-slow': 'pulse-slow 3s ease-in-out infinite',
				'float': 'float 6s ease-in-out infinite'
			}
		}
	},
	plugins: [require("tailwindcss-animate")],
} satisfies Config;
