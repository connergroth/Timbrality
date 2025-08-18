import type { Metadata } from 'next'
import { Inter, Playfair_Display } from 'next/font/google'
import './globals.css'
import { SupabaseProvider } from '@/components/SupabaseProvider'
import { ThemeProvider } from '@/components/ThemeProvider'
import { SidebarProvider } from '@/contexts/SidebarContext'

const inter = Inter({ subsets: ['latin'] })
const playfair = Playfair_Display({
  subsets: ['latin'],
  variable: '--font-playfair'
})

export const metadata: Metadata = {
  title: 'Timbrality',
  description: 'Your personal AI music agent',
  icons: {
    icon: [
      { url: '/favicon.ico', sizes: '32x32' },
      { url: '/soundwhite.png', sizes: '32x32', type: 'image/png' },
      { url: '/soundwhite.png', sizes: '16x16', type: 'image/png' },
    ],
    shortcut: '/favicon.ico',
    apple: '/soundwhite.png',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${inter.className} ${playfair.variable} bg-neutral-900 text-foreground`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <SupabaseProvider>
            <SidebarProvider>
              {children}
            </SidebarProvider>
          </SupabaseProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
