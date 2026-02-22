import type { Metadata } from 'next';
import { JetBrains_Mono, Plus_Jakarta_Sans, Syne } from 'next/font/google';

import './globals.css';

const sans = Plus_Jakarta_Sans({
  variable: '--font-sans',
  weight: ['400', '500', '600', '700'],
  subsets: ['latin'],
});

const mono = JetBrains_Mono({
  variable: '--font-mono',
  weight: ['400', '500'],
  subsets: ['latin'],
});

const display = Syne({
  variable: '--font-display',
  weight: ['500', '600', '700'],
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: {
    default: 'Nell Signal Terminal',
    template: '%s | Nell Signal Terminal',
  },
  description:
    'High-conviction weekly signal intelligence for HSI traders: clean signal, risk context, and scenario proof.',
  applicationName: 'Nell Signal Terminal',
  keywords: [
    'HSI',
    'signal intelligence',
    'quant trading',
    'walk-forward analysis',
    'explainable AI signals',
  ],
  openGraph: {
    title: 'Nell Signal Terminal',
    description:
      'Signal clarity for HSI: weekly stance, confidence context, and walk-forward proof.',
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
