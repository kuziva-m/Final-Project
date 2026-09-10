-- Ledger scans table.
-- Each row is one independently-scanned ledger page for one business.
-- `tables` stores whatever column layout that business's ledger actually
-- uses (columns + rows as read from the page) — no fixed schema is
-- enforced, since different businesses format their records differently
-- and scans are not queried across businesses.
--
-- Run this once in the Supabase SQL editor (or via `supabase db push`)
-- before the API's SUPABASE_URL / SUPABASE_KEY point at a live project.

create table if not exists scans (
    id uuid primary key default gen_random_uuid(),
    business_id text not null,
    tables jsonb not null,
    overall_confidence real not null,
    scanned_at timestamptz not null default now()
);

create index if not exists scans_business_id_idx on scans (business_id);
