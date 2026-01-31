/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
  	extend: {
  		fontFamily: {
  			sans: [
  				'Inter',
  				'system-ui',
  				'sans-serif'
  			]
  		},
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
			crypto: {
				green: '#00D4AA',
				red: '#FF6B6B',
				bitcoin: '#F7931A',
				ethereum: '#627EEA'
			},
			coinbase: {
				blue: 'hsl(var(--coinbase-blue))',
				'blue-light': 'hsl(var(--coinbase-blue-light))',
				'blue-dark': 'hsl(var(--coinbase-blue-dark))',
				green: 'hsl(var(--coinbase-green))',
				'green-light': 'hsl(var(--coinbase-green-light))',
				red: 'hsl(var(--coinbase-red))',
				'red-light': 'hsl(var(--coinbase-red-light))',
				gray: 'hsl(var(--coinbase-gray))',
				'gray-light': 'hsl(var(--coinbase-gray-light))',
				'gray-dark': 'hsl(var(--coinbase-gray-dark))',
				dark: 'hsl(var(--coinbase-dark))',
				'light-gray': 'hsl(var(--coinbase-light-gray))',
				surface: 'hsl(var(--coinbase-surface))',
				'surface-elevated': 'hsl(var(--coinbase-surface-elevated))',
				border: 'hsl(var(--coinbase-border))',
				'border-light': 'hsl(var(--coinbase-border-light))'
			},
			chart: {
				'1': 'hsl(var(--chart-1))',
				'2': 'hsl(var(--chart-2))',
				'3': 'hsl(var(--chart-3))',
				'4': 'hsl(var(--chart-4))',
				'5': 'hsl(var(--chart-5))'
			}
		},
  		borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		},
		boxShadow: {
			sm: 'var(--shadow-sm)',
			DEFAULT: 'var(--shadow)',
			md: 'var(--shadow-md)',
			lg: 'var(--shadow-lg)',
			xl: 'var(--shadow-xl)',
			'2xl': 'var(--shadow-2xl)',
			modern: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
			'modern-lg': '0 35px 60px -12px rgba(0, 0, 0, 0.3)',
			'modern-xl': '0 45px 80px -12px rgba(0, 0, 0, 0.35)'
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
  			'price-flash': {
  				'0%': {
  					backgroundColor: 'transparent'
  				},
  				'50%': {
  					backgroundColor: 'hsl(var(--accent))'
  				},
  				'100%': {
  					backgroundColor: 'transparent'
  				}
  			},
  			'float': {
  				'0%, 100%': {
  					transform: 'translateY(0px)'
  				},
  				'50%': {
  					transform: 'translateY(-20px)'
  				}
  			}
  		},
  		animation: {
  			'accordion-down': 'accordion-down 0.2s ease-out',
  			'accordion-up': 'accordion-up 0.2s ease-out',
  			'price-flash': 'price-flash 0.8s ease-in-out',
  			'float': 'float 6s ease-in-out infinite'
  		}
  	}
  },
  plugins: [require("tailwindcss-animate")],
  darkMode: ["class"],
}