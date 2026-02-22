import type { Metadata } from 'next';
import { Barlow_Condensed, IBM_Plex_Mono, IBM_Plex_Sans } from 'next/font/google';

import './globals.css';

const sans = IBM_Plex_Sans({
  variable: '--font-sans',
  weight: ['400', '500', '600', '700'],
  subsets: ['latin'],
});

const mono = IBM_Plex_Mono({
  variable: '--font-mono',
  weight: ['400', '500'],
  subsets: ['latin'],
});

const display = Barlow_Condensed({
  variable: '--font-display',
  weight: ['500', '600', '700'],
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: {
    default: 'Stock Prediction Interface',
    template: '%s | Stock Prediction Interface',
  },
  description:
    'Professional weekly signal intelligence with explainability, walk-forward metrics, and scenario analysis.',
  applicationName: 'Stock Prediction Interface',
  keywords: [
    'HSI',
    'signal intelligence',
    'quant trading',
    'walk-forward analysis',
    'explainable AI signals',
  ],
  openGraph: {
    title: 'Stock Prediction Interface',
    description:
      'Weekly stock prediction workflow with signal, confidence context, and validation analytics.',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${sans.variable} ${mono.variable} ${display.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
