'use client'

import { useQuery } from '@tanstack/react-query'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
} from 'recharts'
import { api } from '@/lib/api'
import type { StatsEndings, StatsReachedInterview } from '@/lib/types'
import { Loader2, TrendingDown, TrendingUp } from 'lucide-react'
import { cn } from '@/lib/utils'

// "How processes ended": the last stage a closed interview process reached,
// in the order a process moves.
const ENDED_STAGES: { stage: string; label: string }[] = [
  { stage: 'Screening Call', label: 'After the screening call' },
  { stage: 'Interview', label: 'After an interview' },
  { stage: 'Final Interview', label: 'After the final interview' },
  { stage: 'Offer', label: 'After an offer' },
  { stage: 'Accepted', label: 'After accepting' },
]
const AFTER: Record<string, string> = {
  Applied: 'after applying',
  'Screening Call': 'after the screening call',
  Interview: 'after an interview',
  'Final Interview': 'after the final interview',
  Offer: 'after the offer',
  Accepted: 'after accepting the offer',
}

/** One "what happened" line's stage, in the words of how the card closed:
 *  'after an interview', or 'you withdrew after an interview' / 'you turned
 *  down the offer' when the user ended it. */
function endedHow(status: string, stage: string): string {
  if (status !== 'I Withdrew') return AFTER[stage] ?? stage
  return stage === 'Offer' ? 'you turned down the offer' : `you withdrew ${AFTER[stage] ?? stage}`
}

/** 'Mar 3' from a UTC 'YYYY-MM-DD HH:MM:SS' (local time); '' when unknown. */
function shortDate(at: string | null | undefined): string {
  if (!at) return ''
  const d = new Date(`${at.replace(' ', 'T')}Z`)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

/* Palette: the app's own accent language, muted for data.
   strong = emerald (HIGH MATCH badge), worth = violet (Worth-a-look pill),
   applied = the Kanban's Applied cyan. Everything else stays zinc. */
const C = {
  strong: '#10b981',
  worth: '#c4b5fd',
  applied: '#0891b2',
  axis: '#a1a1aa',
  grid: '#f4f4f5',
}

const tooltipStyle = {
  backgroundColor: '#ffffff',
  border: '1px solid #e4e4e7',
  borderRadius: 10,
  boxShadow: '0 4px 12px rgb(0 0 0 / 0.08)',
  fontSize: 12.5,
  padding: '8px 12px',
}

function Delta({ now, prev }: { now: number; prev: number }) {
  if (!prev && !now) return null
  const pct = prev ? Math.round(((now - prev) / prev) * 100) : 100
  const up = pct >= 0
  return (
    <span
      className={cn(
        'inline-flex items-center gap-0.5 text-xs font-medium tabular-nums',
        up ? 'text-emerald-600' : 'text-rose-500'
      )}
      title={`vs last week: ${prev}`}
    >
      {up ? <TrendingUp className="size-3" /> : <TrendingDown className="size-3" />}
      {Math.abs(pct)}%
    </span>
  )
}

function Kpi({
  label,
  value,
  sub,
  delta,
}: {
  label: string
  value: number | string
  sub?: string
  delta?: { now: number; prev: number }
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </span>
      <div className="flex items-baseline gap-2">
        <span className="text-4xl font-semibold tabular-nums tracking-tight">
          {value}
        </span>
        {delta && <Delta now={delta.now} prev={delta.prev} />}
      </div>
      {sub && <span className="text-xs text-muted-foreground">{sub}</span>}
    </div>
  )
}

/* The funnel — custom-drawn so widths, labels and conversions are exact.
   Width uses a sqrt scale so a drop of several orders of magnitude between
   the first and last stage stays legible; the exact numbers are always
   printed. */
function Funnel({ f }: { f: any }) {
  const stages = [
    { key: 'surfaced', label: 'Surfaced', n: f.surfaced, tone: 'bg-zinc-200 text-zinc-700' },
    { key: 'reviewed', label: 'Reviewed', n: f.reviewed, tone: 'bg-zinc-300 text-zinc-800' },
    { key: 'applied', label: 'Applied', n: f.applied, tone: 'bg-cyan-600/85 text-white' },
    { key: 'in_process', label: 'In process', n: f.in_process, tone: 'bg-emerald-500/85 text-white' },
    { key: 'offers', label: 'Offers', n: f.offers, tone: 'bg-emerald-600 text-white' },
  ]
  const max = Math.max(f.surfaced, 1)
  return (
    <div className="space-y-1">
      {stages.map((s, i) => {
        const prev = i > 0 ? stages[i - 1].n : null
        const conv = prev ? Math.round((s.n / Math.max(prev, 1)) * 100) : null
        const width = Math.max(6, Math.sqrt(s.n / max) * 100)
        return (
          <div key={s.key} className="group flex items-center gap-3">
            <span className="w-24 shrink-0 text-right text-[13px] text-muted-foreground">
              {s.label}
            </span>
            <div className="relative h-9 flex-1">
              <div
                className={cn(
                  'flex h-full items-center rounded-md px-3 transition-all',
                  s.tone
                )}
                style={{ width: `${width}%`, minWidth: '3.5rem' }}
              >
                <span className="text-sm font-semibold tabular-nums">
                  {s.n.toLocaleString()}
                </span>
              </div>
            </div>
            <span className="w-20 shrink-0 text-xs tabular-nums text-muted-foreground">
              {conv !== null ? `${conv}% of prev` : ' '}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function SectionTitle({
  children,
  hint,
}: {
  children: React.ReactNode
  hint?: string
}) {
  return (
    <div className="mb-3 flex items-baseline justify-between">
      <h2
        className="text-lg tracking-[0.06em] text-foreground/85"
        style={{ fontFamily: 'var(--font-display), sans-serif' }}
      >
        {children}
      </h2>
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </div>
  )
}

export default function StatsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['stats-overview'],
    queryFn: () => api.stats.overview(),
    refetchInterval: 60_000,
  })

  if (isLoading || !data) {
    return (
      <div className="flex items-center gap-2 p-8 text-muted-foreground">
        <Loader2 className="size-4 animate-spin" /> Loading statistics…
      </div>
    )
  }

  const f = data.funnel
  const reach: StatsReachedInterview = data
  // Optional: an API from before 2026-10-28 does not send it.
  const endings: StatsEndings | undefined = data.endings
  const endedRows = ENDED_STAGES.map((s) => ({
    ...s,
    n: endings?.ended_after?.[s.stage as keyof StatsEndings['ended_after']] ?? 0,
  })).filter((r) => r.n > 0)
  const maxEnded = Math.max(...endedRows.map((r) => r.n), 1)
  const appliedRate = f.reviewed ? Math.round((f.applied / f.reviewed) * 100) : 0
  const roleTypes = (data.role_types ?? []).slice(0, 8).map((r: any) => ({
    ...r,
    total: r.strong + r.worth_a_look,
    category: String(r.category).replace(/_/g, ' '),
  }))
  const maxRole = Math.max(...roleTypes.map((r: any) => r.total), 1)

  return (
    <div className="mx-auto max-w-5xl space-y-10 p-6 pb-16">
      {/* ── KPI strip ── */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Pipeline</h1>
        <p
          className="mb-6 mt-0.5 text-[15px] italic text-muted-foreground"
          style={{ fontFamily: 'var(--font-accent), serif' }}
        >
          Last {data.days} days · refreshes with every run
        </p>
        <div className="grid grid-cols-2 gap-x-6 gap-y-6 sm:grid-cols-4">
          <Kpi
            label="Roles this week"
            value={data.week.surfaced.now}
            delta={data.week.surfaced}
            sub={`${f.surfaced.toLocaleString()} in ${data.days} days`}
          />
          <Kpi
            label="Strong this week"
            value={data.week.strong.now}
            delta={data.week.strong}
            sub={`${f.strong} in ${data.days} days`}
          />
          <Kpi
            label="Applied this week"
            value={data.week.applied.now}
            delta={data.week.applied}
            sub={`${f.applied} in ${data.days} days`}
          />
          <Kpi
            label="Apply rate"
            value={`${appliedRate}%`}
            sub="of reviewed roles"
          />
        </div>
      </div>

      {/* ── Funnel ── */}
      <div>
        <SectionTitle hint={`${data.days}-day cohort · in-process & offers are current state`}>
          The funnel
        </SectionTitle>
        <Funnel f={f} />
      </div>

      {/* ── Daily rhythm: surfaced + applied, synced ── */}
      <div>
        <SectionTitle hint="hover to inspect a day">Daily rhythm</SectionTitle>
        <div className="rounded-xl border border-border/60 p-4">
          <div className="mb-1 flex items-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="size-2 rounded-sm" style={{ background: C.strong }} />
              strong
            </span>
            <span className="flex items-center gap-1.5">
              <span className="size-2 rounded-sm" style={{ background: C.worth }} />
              worth a look
            </span>
          </div>
          <div className="h-[210px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.daily} syncId="days" margin={{ top: 4, right: 4, left: -22, bottom: 0 }} barCategoryGap="22%">
                <XAxis dataKey="day" hide />
                <YAxis tick={{ fontSize: 10.5, fill: C.axis }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: C.grid }} />
                <Bar dataKey="strong" name="strong" stackId="t" fill={C.strong} />
                <Bar dataKey="worth_a_look" name="worth a look" stackId="t" fill={C.worth} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-1 flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className="size-2 rounded-sm" style={{ background: C.applied }} />
            applications sent
          </div>
          <div className="h-[86px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.daily} syncId="days" margin={{ top: 4, right: 4, left: -22, bottom: 0 }} barCategoryGap="22%">
                <XAxis dataKey="day" tick={{ fontSize: 10, fill: C.axis }} axisLine={false} tickLine={false}
                       tickFormatter={(d: string) => d.slice(5)} interval={4} />
                <YAxis tick={{ fontSize: 10.5, fill: C.axis }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: C.grid }} />
                <Bar dataKey="applied" name="applied" fill={C.applied} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid gap-10 lg:grid-cols-2">
        {/* ── Where applications stand ── */}
        <div>
          <SectionTitle hint="all time">Where applications stand</SectionTitle>
          {/* From the card history: counts a card that got to a screening call
              or later even if it now sits in a closed status below. The
              big number is confirmed only; cards whose stage was only inferred
              wait for the user's confirmation on the muted line. */}
          {typeof reach.reached_interview_total === 'number' && (
            <div className="mb-5">
              <Kpi
                label="Reached interview (all time)"
                value={reach.reached_interview_total}
                sub="screening call or later, whatever happened next"
              />
              {(reach.reached_interview_unconfirmed ?? 0) > 0 && (
                <p className="mt-1 text-xs text-muted-foreground">
                  +{reach.reached_interview_unconfirmed} more to confirm
                </p>
              )}
            </div>
          )}
          <div className="space-y-2.5">
            {['To Apply', 'Applied', 'Screening Call', 'Interview', 'Final Interview', 'Offer', 'Accepted', 'Rejected', 'Ghosted', 'I Withdrew']
              .filter((s) => data.statuses?.[s])
              .map((s) => {
                const n = data.statuses[s]
                const maxS = Math.max(...Object.values(data.statuses as Record<string, number>), 1)
                const active = !['Rejected', 'Ghosted', 'I Withdrew'].includes(s)
                return (
                  <div key={s} className="flex items-center gap-3 text-sm">
                    <span className="w-28 shrink-0 text-[13px] text-muted-foreground">{s}</span>
                    <div className="h-4 flex-1">
                      <div
                        className={cn('h-full rounded-sm', active ? 'bg-cyan-600/80' : 'bg-zinc-300')}
                        style={{ width: `${Math.max((n / maxS) * 100, 1.5)}%` }}
                      />
                    </div>
                    <span className="w-10 shrink-0 text-right text-sm font-medium tabular-nums">{n}</span>
                  </div>
                )
              })}
          </div>
        </div>

        {/* ── Role types ── */}
        <div>
          <SectionTitle hint={`categorized roles · ${data.uncategorized} ATS finds uncategorized`}>
            What kind of roles
          </SectionTitle>
          <div className="space-y-2.5">
            {roleTypes.map((r: any) => (
              <div key={r.category} className="flex items-center gap-3 text-sm">
                <span className="w-40 shrink-0 truncate text-[13px] text-muted-foreground">
                  {r.category}
                </span>
                <div className="flex h-4 flex-1 gap-px">
                  <div className="h-full rounded-l-sm" style={{ width: `${(r.strong / maxRole) * 100}%`, background: C.strong, minWidth: r.strong ? 3 : 0 }} />
                  <div className="h-full rounded-r-sm" style={{ width: `${(r.worth_a_look / maxRole) * 100}%`, background: C.worth }} />
                </div>
                <span className="w-10 shrink-0 text-right text-sm font-medium tabular-nums">
                  {r.total}
                </span>
              </div>
            ))}
            {roleTypes.length === 0 && (
              <p className="text-sm text-muted-foreground">No categorized roles in this window.</p>
            )}
          </div>
        </div>
      </div>

      {/* ── How processes ended: closed cards (Rejected / Ghosted / I
          Withdrew), from the card history and the user's answers on the Kanban
          ("How far did it get?") ── */}
      {endings && (
        <div>
          <SectionTitle hint="all time · Rejected, Ghosted and I Withdrew cards">How processes ended</SectionTitle>
          <div className="grid gap-10 lg:grid-cols-2">
            <div>
              <p className="mb-3 text-[13px] text-muted-foreground">
                Interview processes, by the last stage they reached
              </p>
              <div className="space-y-2.5">
                {endedRows.map((r) => (
                  <div key={r.stage} className="flex items-center gap-3 text-sm">
                    <span className="w-44 shrink-0 text-[13px] text-muted-foreground">{r.label}</span>
                    <div className="h-4 flex-1">
                      <div
                        className="h-full rounded-sm bg-amber-400/80"
                        style={{ width: `${Math.max((r.n / maxEnded) * 100, 1.5)}%` }}
                      />
                    </div>
                    <span className="w-10 shrink-0 text-right text-sm font-medium tabular-nums">{r.n}</span>
                  </div>
                ))}
                {endedRows.length === 0 && (
                  <p className="text-sm text-muted-foreground">None recorded yet.</p>
                )}
              </div>
              {endings.to_confirm > 0 && (
                <p className="mt-2 text-xs text-muted-foreground">
                  +{endings.to_confirm} more to confirm — open the card on the Kanban and pick
                  how far it got
                </p>
              )}
              {(endings.withdrew ?? 0) > 0 && (
                <p className="mt-2 text-xs text-muted-foreground">
                  You withdrew from {endings.withdrew}{' '}
                  {endings.withdrew === 1 ? 'process' : 'processes'} yourself
                </p>
              )}
            </div>
            <div>
              <p className="mb-3 text-[13px] text-muted-foreground">What happened, newest first</p>
              {endings.notes.length > 0 ? (
                <ul className="space-y-3">
                  {endings.notes.map((n) => (
                    <li key={n.app_id} className="text-sm">
                      <span className="font-medium text-foreground">{n.company ?? 'A company'}</span>
                      <span className="text-muted-foreground">
                        {' '}· {endedHow(n.status, n.stage)}
                        {shortDate(n.at) ? ` · ${shortDate(n.at)}` : ''}
                      </span>
                      <p className="mt-0.5 text-foreground/85">“{n.what_happened}”</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Nothing yet. When a card moves to Rejected, Ghosted or I Withdrew, the Kanban
                  asks how far it got and what happened — or add it any time with “Add details”
                  on a closed card.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
