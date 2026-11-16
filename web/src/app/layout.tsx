import type { Metadata } from 'next'
import { Geist, Geist_Mono, Source_Serif_4 } from 'next/font/google'
import { Providers } from '@/lib/providers'
import { Sidebar } from '@/components/Sidebar'
import { PipelineStatus } from '@/components/PipelineStatus'
import { SearchStatusSync } from '@/components/SearchStatusSync'
import './globals.css'

// Typography: Geist for body and headings (--font-display is aliased to it
// in globals.css), a serif italic for small accents, Geist Mono for code.
const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
})

const accentSerif = Source_Serif_4({
  variable: '--font-accent',
  subsets: ['latin'],
  style: ['italic'],
  weight: ['400', '500'],
})

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})

export const metadata: Metadata = {
  title: 'AI Job Agent',
  description:
    'AI-powered job search assistant — scrape, filter, track, and apply with confidence.',
  icons: {
    icon: [
      { url: '/favicon.svg', type: 'image/svg+xml' },
    ],
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${accentSerif.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full bg-background text-foreground">
        <Providers>
          {/* Backend search-state sync (polls /api/search/status) */}
          <SearchStatusSync />

          {/* Fixed sidebar */}
          <Sidebar />

          {/* Main content area — offset by sidebar width */}
          <div className="flex flex-1 flex-col pl-64">
            {/* Floating pipeline status bar */}
            <PipelineStatus />

            {/* Page content */}
            <main className="flex-1 overflow-y-auto px-8 py-8">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  )
}
