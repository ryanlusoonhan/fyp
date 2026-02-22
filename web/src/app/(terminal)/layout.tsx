import type { ReactNode } from 'react';

import { TerminalLayoutClient } from '@/components/terminal/terminal-layout-client';

export default function TerminalLayout({ children }: { children: ReactNode }) {
  return <TerminalLayoutClient>{children}</TerminalLayoutClient>;
}
