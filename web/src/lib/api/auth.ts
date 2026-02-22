import { createClient } from '@supabase/supabase-js';
import { headers } from 'next/headers';

import type { PlanId } from '@/lib/types';

type AuthSource = 'header' | 'bearer' | 'supabase' | 'default';

export interface AuthContext {
  plan: PlanId;
  userId: string | null;
  source: AuthSource;
}

function isPlanId(value: string): value is PlanId {
  return value === 'free' || value === 'pro' || value === 'elite';
}

function readEnv(name: string): string | null {
  const value = process.env[name];
  if (!value || value.trim().length === 0) {
    return null;
  }
  return value.trim();
}

function parseBearerToken(rawAuthorization: string | null): string | null {
  if (!rawAuthorization) {
    return null;
  }

  const [scheme, token] = rawAuthorization.split(' ');
  if (!scheme || !token || scheme.toLowerCase() !== 'bearer') {
    return null;
  }
  return token.trim();
}

export function parsePlanFromDevToken(token: string | null): PlanId | null {
  if (!token) {
    return null;
  }

  if (token.startsWith('plan:')) {
    const candidate = token.replace('plan:', '').trim();
    return isPlanId(candidate) ? candidate : null;
  }

  if (token.startsWith('demo_')) {
    const candidate = token.replace('demo_', '').trim();
    return isPlanId(candidate) ? candidate : null;
  }

  return null;
}

async function resolvePlanFromSupabase(accessToken: string): Promise<{ userId: string; plan: PlanId } | null> {
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

  const { data: userData, error: userError } = await authClient.auth.getUser();
  if (userError || !userData.user) {
    return null;
  }

  const userId = userData.user.id;
  const serviceRole = readEnv('SUPABASE_SERVICE_ROLE_KEY');
  if (!serviceRole) {
    return { userId, plan: 'free' };
  }

  const adminClient = createClient(url, serviceRole, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

  const subscriptionResult = await adminClient
    .from('subscriptions')
    .select('plan,status,current_period_end')
    .eq('user_id', userId)
    .eq('status', 'active')
    .order('current_period_end', { ascending: false })
    .limit(1)
    .maybeSingle();

  if (subscriptionResult.data && isPlanId(subscriptionResult.data.plan)) {
    return { userId, plan: subscriptionResult.data.plan };
  }

  const profileResult = await adminClient
    .from('profiles')
    .select('plan')
    .eq('id', userId)
    .maybeSingle();

  if (profileResult.data && isPlanId(profileResult.data.plan)) {
    return { userId, plan: profileResult.data.plan };
  }

  return { userId, plan: 'free' };
}

export async function getRequestContext(defaultPlan: PlanId = 'free'): Promise<AuthContext> {
  const hdrs = await headers();

  const planHeader = hdrs.get('x-plan-id');
  if (planHeader && isPlanId(planHeader)) {
    return { plan: planHeader, userId: null, source: 'header' };
  }

  const token = parseBearerToken(hdrs.get('authorization'));
  const planFromToken = parsePlanFromDevToken(token);
  if (planFromToken) {
    return { plan: planFromToken, userId: null, source: 'bearer' };
  }

  if (token) {
    const supabaseContext = await resolvePlanFromSupabase(token);
    if (supabaseContext) {
      return { plan: supabaseContext.plan, userId: supabaseContext.userId, source: 'supabase' };
    }
  }

  return { plan: defaultPlan, userId: null, source: 'default' };
}

export async function getRequestPlan(defaultPlan: PlanId = 'free'): Promise<PlanId> {
  const context = await getRequestContext(defaultPlan);
  return context.plan;
}
