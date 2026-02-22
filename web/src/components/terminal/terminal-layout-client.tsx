'use client';

import { usePathname } from 'next/navigation';
import type { ReactNode } from 'react';

import { TerminalShell } from '@/components/terminal/terminal-shell';

export function TerminalLayoutClient({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  return <TerminalShell pathname={pathname}>{children}</TerminalShell>;
}
