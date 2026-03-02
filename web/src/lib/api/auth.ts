import { createClient } from '@supabase/supabase-js';
import { headers } from 'next/headers';

type AuthSource = 'bearer' | 'supabase' | 'default';

export interface AuthContext {
  userId: string | null;
  source: AuthSource;
}

function readEnv(name: string): string | null {
  const value = process.env[name];
  if (!value || value.trim().length === 0) {
    return null;
  }
  return value.trim();
}

export function parseBearerAuthHeader(rawAuthorization: string | null): string | null {
  if (!rawAuthorization) {
    return null;
  }

  const [scheme, token] = rawAuthorization.split(' ');
  if (!scheme || !token || scheme.toLowerCase() !== 'bearer') {
    return null;
  }
  return token.trim();
}

async function resolveUserIdFromSupabase(accessToken: string): Promise<string | null> {
  const url = readEnv('NEXT_PUBLIC_SUPABASE_URL');
  const anonKey = readEnv('NEXT_PUBLIC_SUPABASE_ANON_KEY');
  if (!url || !anonKey) {
    return null;
  }

  const authClient = createClient(url, anonKey, {
    auth: { persistSession: false, autoRefreshToken: false },
    global: {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    },
  });

  const { data, error } = await authClient.auth.getUser();
  if (error || !data.user) {
    return null;
  }
  return data.user.id;
}

export async function getRequestContext(): Promise<AuthContext> {
  const hdrs = await headers();
  const token = parseBearerAuthHeader(hdrs.get('authorization'));
  if (!token) {
    return { userId: null, source: 'default' };
  }

  const userId = await resolveUserIdFromSupabase(token);
  if (!userId) {
    return { userId: null, source: 'bearer' };
  }

  return { userId, source: 'supabase' };
}
