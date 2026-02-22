-- Nell Signal Terminal MVP schema
-- Apply with: supabase db push

create extension if not exists pgcrypto;

do $$
begin
  if not exists (select 1 from pg_type where typname = 'plan_id') then
    create type public.plan_id as enum ('free', 'pro', 'elite');
  end if;
end $$;

do $$
begin
  if not exists (select 1 from pg_type where typname = 'subscription_status') then
    create type public.subscription_status as enum ('active', 'trialing', 'past_due', 'canceled');
  end if;
end $$;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  plan public.plan_id not null default 'free',
  timezone text not null default 'Asia/Hong_Kong',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger profiles_set_updated_at
before update on public.profiles
for each row
execute function public.set_updated_at();

create table if not exists public.subscriptions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  plan public.plan_id not null,
  status public.subscription_status not null,
  stripe_customer_id text unique,
  stripe_subscription_id text unique,
  current_period_start timestamptz,
  current_period_end timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists subscriptions_user_status_period_idx
  on public.subscriptions (user_id, status, current_period_end desc);

create trigger subscriptions_set_updated_at
before update on public.subscriptions
for each row
execute function public.set_updated_at();

create table if not exists public.signal_snapshots (
  id uuid primary key default gen_random_uuid(),
  as_of_date date not null,
  market text not null,
  objective text not null check (objective in ('f1', 'return')),
  threshold numeric(8, 6) not null,
  probability_buy numeric(8, 6) not null,
  probability_no_buy numeric(8, 6) not null,
  predicted_class smallint not null check (predicted_class in (0, 1)),
  model_version text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (as_of_date, market, objective)
);

create index if not exists signal_snapshots_date_idx
  on public.signal_snapshots (as_of_date desc);

create table if not exists public.scenario_runs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  objective text not null check (objective in ('f1', 'return')),
  threshold_min numeric(8, 6) not null,
  threshold_max numeric(8, 6) not null,
  step numeric(8, 6) not null,
  cost numeric(8, 6) not null,
  best_threshold numeric(8, 6) not null,
  best_score numeric(10, 6) not null,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists scenario_runs_user_created_idx
  on public.scenario_runs (user_id, created_at desc);

alter table public.profiles enable row level security;
alter table public.subscriptions enable row level security;
alter table public.scenario_runs enable row level security;

drop policy if exists "profiles_select_self" on public.profiles;
create policy "profiles_select_self"
on public.profiles
for select
to authenticated
using (auth.uid() = id);

drop policy if exists "profiles_update_self" on public.profiles;
create policy "profiles_update_self"
on public.profiles
for update
to authenticated
using (auth.uid() = id);

drop policy if exists "subscriptions_select_self" on public.subscriptions;
create policy "subscriptions_select_self"
on public.subscriptions
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "scenario_runs_select_self" on public.scenario_runs;
create policy "scenario_runs_select_self"
on public.scenario_runs
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "scenario_runs_insert_self" on public.scenario_runs;
create policy "scenario_runs_insert_self"
on public.scenario_runs
for insert
to authenticated
with check (auth.uid() = user_id);
