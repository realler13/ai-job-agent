'use client'

import { Fragment, useState, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { Job, PostingRead } from '@/lib/types'
import { MatchBadge } from '@/components/MatchBadge'
import { ConfidenceBar } from '@/components/ConfidenceBar'
import { DismissModal } from '@/components/DismissModal'
import { CompanyIntelPanel } from '@/components/CompanyIntelPanel'
import { PostingCheckPanel } from '@/components/PostingCheckPanel'
import { DocumentPanel } from '@/components/DocumentPanel'
import { Card, CardContent } from '@/components/ui/card'
import { Button, buttonVariants } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import {
  Eye,
  Bookmark,
  XCircle,
  CheckSquare,
  Square,
  Loader2,
  Inbox,
  ChevronDown,
  ExternalLink,
} from 'lucide-react'

// Contract-pay note (2026-10-03; Phase 27 review fix): contract and
// agency roles are never penalized, but the row says the pay works
// differently. The note is built from what the row supports — "paid hourly"
// only on /hr pay, "paid a day rate" only on /day, "(W2 through an agency, or
// 1099)" only when the employment type names W2/1099/C2C or the company is a
// staffing agency. Mirrors comp_utils.is_contract_pay / contract_pay_note /
// STAFFING_AGENCY_RE (Python, digest cards) and the other feed page — keep
// the three in step.
const CONTRACT_PAY_TAIL =
  'taxes, benefits and paid time off work differently from a salaried role; compare take-home, not the annualized figure.'
const CONTRACT_TYPE_RE = /\b(?:contract(?:or)?|temp(?:orary)?|freelance|1099|c2c)\b/i
const CONTRACT_WORD_RE = /\b(?:contract(?:or)?|freelance|1099|c2c)\b/i
const EMPLOYEE_TYPE_RE = /\b(?:(?:full|part)-?\s?time|intern(?:ship)?)\b/i
const TAX_FORM_RE = /\b(?:w-?2|1099|c2c)\b/i
const STAFFING_AGENCY_RE = new RegExp(
  String.raw`\bstaffing\b|\(via\s|\brecruit(?:ing|ment)\b|\bworkforce solutions\b` +
    String.raw`|\b(?:aquent|robert half|russell tobin|creative circle|the planet group|workgenius` +
    String.raw`|pms for hire|synergis|mindsource|cypress hcm|insight global|inspyr solutions|cella` +
    String.raw`|adecco|randstad|kforce|24 seven talent|solomon page|pyramid consulting|intellipro` +
    String.raw`|intelliswift|infotree|epitec|atr international|ektello|us tech solutions` +
    String.raw`|the judge group|talentburst|syndicatebleu|pride global|horizontal talent` +
    String.raw`|hellokindred|harvey nash|experis|dexian|akkodis|yoh|varite|teksystems` +
    String.raw`|apex systems|kelly services|magnit|willhire|tcwglobal|the mom project` +
    String.raw`|saicon consultants|motion recruitment|mondo|yunojuno|ursus)\b`,
  'i'
)

const cleanField = (v?: string | null) => {
  const s = (v ?? '').trim()
  return ['nan', 'none'].includes(s.toLowerCase()) ? '' : s
}

function isContractPay(etype?: string | null, comp?: string | null): boolean {
  const e = cleanField(etype)
  if (CONTRACT_TYPE_RE.test(e)) return true
  return cleanField(comp).includes('/hr') && !EMPLOYEE_TYPE_RE.test(e)
}

function contractPayNote(
  etype?: string | null,
  comp?: string | null,
  company?: string | null
): string | null {
  if (!isContractPay(etype, comp)) return null
  const e = cleanField(etype)
  const c = cleanField(comp)
  const agency = STAFFING_AGENCY_RE.test(cleanField(company))
  const head =
    CONTRACT_WORD_RE.test(e) || agency
      ? 'Contract role'
      : CONTRACT_TYPE_RE.test(e)
        ? 'Temporary role'
        : 'Hourly role'
  let basis = ''
  if (c.includes('/hr') && head !== 'Hourly role') basis = ', paid hourly'
  else if (c.includes('/day')) basis = ', paid a day rate'
  const via = TAX_FORM_RE.test(e) || agency ? ' (W2 through an agency, or 1099)' : ''
  return `${head}${basis}${via} — ${CONTRACT_PAY_TAIL}`
}

// Posting-check chip (Phase 27c): at most one small muted chip, and only
// when the facts disagree or call for a check — nothing for "Looks active".
// A summary of the facts, not a score: it never moves the row.
function postingChipText(r?: PostingRead | null): string | null {
  if (!r) return null
  if (r.read === 'mixed') return 'Mixed signals'
  if (r.read === 'check_first') return 'Check first'
  return null
}
// end postingChipText

export default function ReviewPage() {
  const queryClient = useQueryClient()
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [dismissTarget, setDismissTarget] = useState<{
    id: number
    title: string
  } | null>(null)
  const [batchDismissOpen, setBatchDismissOpen] = useState(false)
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null)
  const [showFullDescription, setShowFullDescription] = useState<Set<string>>(
    new Set()
  )

  const {
    data: rawJobs = [],
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['unreviewed'],
    queryFn: api.jobs.unreviewed,
  })

  // Comp sort + filter (2026-07-26). Client-side: the queue is small.
  // Sorting by comp uses the yearly-normalized max (deterministic AND
  // "(est.)" LLM values both carry it); rows without comp sort last.
  const [sortByComp, setSortByComp] = useState(false)
  const [minComp, setMinComp] = useState(0) // 0 = no filter, else yearly USD
  const compOf = (j: Job) => j.comp_yearly_max ?? j.comp_yearly_min ?? null
  const jobs = (() => {
    let out = rawJobs as Job[]
    if (minComp > 0) out = out.filter((j) => (compOf(j) ?? 0) >= minComp)
    if (sortByComp)
      // Sort within tiers (strong block stays first) so the tier headers
      // don't interleave; comp descending inside each block, no-comp last.
      out = [...out].sort(
        (a, b) =>
          (a.tier === 'strong' ? 0 : 1) - (b.tier === 'strong' ? 0 : 1) ||
          (compOf(b) ?? -1) - (compOf(a) ?? -1),
      )
    return out
  })()

  const trackMutation = useMutation({
    mutationFn: (id: number) => api.jobs.track(id),
    onMutate: async (id: number) => {
      // Cancel outgoing refetches so they don't overwrite our optimistic update
      await queryClient.cancelQueries({ queryKey: ['unreviewed'] })
      const previous = queryClient.getQueryData<Job[]>(['unreviewed'])
      // Optimistically remove the tracked job from the list
      queryClient.setQueryData<Job[]>(['unreviewed'], (old) =>
        old ? old.filter((j) => j.id !== id) : []
      )
      return { previous }
    },
    onError: (_err, _id, context) => {
      // Roll back on error
      if (context?.previous) {
        queryClient.setQueryData(['unreviewed'], context.previous)
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['unreviewed'] })
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      queryClient.invalidateQueries({ queryKey: ['applications'] })
    },
  })

  const dismissMutation = useMutation({
    mutationFn: ({ id, reason }: { id: number; reason: string }) =>
      api.jobs.dismiss(id, reason),
    onMutate: async (vars: { id: number; reason: string }) => {
      await queryClient.cancelQueries({ queryKey: ['unreviewed'] })
      const previous = queryClient.getQueryData<Job[]>(['unreviewed'])
      queryClient.setQueryData<Job[]>(['unreviewed'], (old) =>
        old ? old.filter((j) => j.id !== vars.id) : []
      )
      return { previous }
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) {
        queryClient.setQueryData(['unreviewed'], context.previous)
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['unreviewed'] })
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })

  const toggleSelect = useCallback((id: number) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }, [])

  const toggleAll = useCallback(() => {
    if (selected.size === jobs.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(jobs.map((j: Job) => j.id)))
    }
  }, [jobs, selected.size])

  const handleTrackAll = useCallback(() => {
    const ids =
      selected.size > 0
        ? Array.from(selected)
        : jobs.map((j: Job) => j.id)
    ids.forEach((id) => trackMutation.mutate(id))
    setSelected(new Set())
  }, [selected, jobs, trackMutation])

  const handleDismissOne = useCallback(
    (reason: string) => {
      if (!dismissTarget) return
      dismissMutation.mutate({ id: dismissTarget.id, reason })
      setDismissTarget(null)
      setSelected((prev) => {
        const next = new Set(prev)
        next.delete(dismissTarget.id)
        return next
      })
    },
    [dismissTarget, dismissMutation]
  )

  const handleBatchDismiss = useCallback(
    (reason: string) => {
      const ids = Array.from(selected)
      ids.forEach((id) => dismissMutation.mutate({ id, reason }))
      setSelected(new Set())
      setBatchDismissOpen(false)
    },
    [selected, dismissMutation]
  )

  const toggleExpand = useCallback((jobId: string) => {
    setExpandedJobId((prev) => (prev === jobId ? null : jobId))
  }, [])

  const toggleFullDescription = useCallback((jobId: string) => {
    setShowFullDescription((prev) => {
      const next = new Set(prev)
      if (next.has(jobId)) {
        next.delete(jobId)
      } else {
        next.add(jobId)
      }
      return next
    })
  }, [])

  // Loading state
  if (isLoading) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="size-8 animate-spin text-muted-foreground" />
          <p className="text-base text-muted-foreground">
            Loading review queue...
          </p>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <div className="flex flex-col items-center gap-3 text-center">
          <div className="flex size-12 items-center justify-center rounded-full bg-red-500/10">
            <Eye className="size-6 text-red-400" />
          </div>
          <p className="text-base font-medium text-foreground">
            Failed to load review queue
          </p>
          <p className="max-w-xs text-sm text-muted-foreground">
            {error instanceof Error
              ? error.message
              : 'An unknown error occurred.'}
          </p>
          <p className="max-w-xs text-xs text-muted-foreground">
            This is usually a brief service restart — retrying typically fixes it.
          </p>
          <Button onClick={() => refetch()} className="mt-1">
            Try again
          </Button>
        </div>
      </div>
    )
  }

  const allSelected = jobs.length > 0 && selected.size === jobs.length
  const DESCRIPTION_TRUNCATE_LENGTH = 500

  // Tiered grouping. Backend returns rows tier-sorted (strong first); fall back
  // to deriving the tier from match_level for older rows that predate the column.
  const tierOf = (j: Job): 'strong' | 'worth_a_look' =>
    j.tier === 'strong' || j.tier === 'worth_a_look'
      ? j.tier
      : j.match_level === 'HIGH_MATCH' || j.match_level === 'USER_INPUT'
        ? 'strong'
        : 'worth_a_look'
  const strongCount = jobs.filter((j: Job) => tierOf(j) === 'strong').length
  const worthCount = jobs.length - strongCount

  // Posting age: early applications tend to get more recruiter attention.
  const agePill = (dp: string | null) => {
    if (!dp) return null
    const days = Math.floor((Date.now() - new Date(dp).getTime()) / 86400000)
    if (isNaN(days) || days < 0) return null
    const label = days === 0 ? 'today' : days < 7 ? `${days}d` : `${Math.floor(days / 7)}w`
    const cls =
      days >= 21
        ? 'bg-red-500/15 text-red-400 border-red-500/25'
        : days >= 14
          ? 'bg-amber-500/15 text-amber-400 border-amber-500/25'
          : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/25'
    return (
      <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>
        {label}
      </span>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            To Review
          </h1>
          <p className="mt-1 text-base text-muted-foreground">
            {jobs.length} job{jobs.length !== 1 ? 's' : ''} awaiting review
          </p>
        </div>
      </div>

      {/* Toolbar */}
      {jobs.length > 0 && (
        <div className="flex items-center gap-3 rounded-lg border border-border/50 bg-card p-3">
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5"
            onClick={toggleAll}
          >
            {allSelected ? (
              <CheckSquare className="size-4 text-primary" />
            ) : (
              <Square className="size-4" />
            )}
            {allSelected ? 'Deselect All' : 'Select All'}
          </Button>

          <div className="h-4 w-px bg-border" />

          <Button
            variant="default"
            size="sm"
            className="gap-1.5"
            onClick={handleTrackAll}
            disabled={trackMutation.isPending}
          >
            {trackMutation.isPending ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Bookmark className="size-3.5" />
            )}
            Track{selected.size > 0 ? ` (${selected.size})` : ' All'}
          </Button>

          <Button
            variant="destructive"
            size="sm"
            className="gap-1.5"
            onClick={() => {
              if (selected.size === 0) {
                // Select all first if none selected
                setSelected(new Set(jobs.map((j: Job) => j.id)))
              }
              setBatchDismissOpen(true)
            }}
            disabled={dismissMutation.isPending}
          >
            {dismissMutation.isPending ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <XCircle className="size-3.5" />
            )}
            Dismiss{selected.size > 0 ? ` (${selected.size})` : ' All'}
          </Button>

          <div className="h-4 w-px bg-border" />

          {/* Comp sort + filter (2026-07-26) */}
          <button
            className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
              sortByComp
                ? 'bg-green-600 text-white'
                : 'bg-muted text-muted-foreground hover:bg-muted/70'
            }`}
            onClick={() => setSortByComp((v) => !v)}
            title="Sort by comp (within tier), highest first"
          >
            💰 Sort by comp
          </button>
          <div className="flex items-center gap-1">
            {[
              [0, 'All'],
              [80000, '$80K+'],
              [100000, '$100K+'],
              [120000, '$120K+'],
              [150000, '$150K+'],
            ].map(([v, label]) => (
              <button
                key={String(v)}
                className={`rounded-full px-2 py-1 text-xs transition-colors ${
                  minComp === v
                    ? 'bg-indigo-600 text-white'
                    : 'bg-muted text-muted-foreground hover:bg-muted/70'
                }`}
                onClick={() => setMinComp(v as number)}
              >
                {label}
              </button>
            ))}
          </div>

          {selected.size > 0 && (
            <span className="ml-auto text-sm text-muted-foreground">
              {selected.size} selected
            </span>
          )}
        </div>
      )}

      {/* Empty state */}
      {jobs.length === 0 ? (
        <div className="flex h-[50vh] items-center justify-center">
          <div className="flex flex-col items-center gap-4 text-center">
            <div className="flex size-16 items-center justify-center rounded-2xl bg-muted">
              <Inbox className="size-8 text-muted-foreground" />
            </div>
            <div>
              <p className="text-base font-medium text-foreground">
                Review queue is empty
              </p>
              <p className="mt-1 max-w-xs text-sm text-muted-foreground">
                All jobs have been reviewed. Run a new search to discover more
                opportunities.
              </p>
            </div>
          </div>
        </div>
      ) : (
        /* Queue cards */
        <div className="flex flex-col gap-3">
          {jobs.map((job: Job, idx: number) => {
            const tier = tierOf(job)
            const showTierHeader = idx === 0 || tier !== tierOf(jobs[idx - 1])
            const isSelected = selected.has(job.id)
            const jobKey = String(job.id)
            const isExpanded = expandedJobId === jobKey
            const rawDesc = job.description || (job.raw_data?.description as string) || ''
            const descriptionText =
              rawDesc && rawDesc !== 'relevant' ? rawDesc : ''
            const isDescriptionLong =
              descriptionText.length > DESCRIPTION_TRUNCATE_LENGTH
            const showFull = showFullDescription.has(jobKey)

            return (
              <Fragment key={job.id}>
                {showTierHeader && (
                  <div className="mt-3 flex items-center gap-2.5 first:mt-0">
                    <h2 className="text-sm font-semibold uppercase tracking-wider text-foreground">
                      {tier === 'strong' ? 'Strong matches' : 'Worth a look'}
                    </h2>
                    <span className="rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
                      {tier === 'strong' ? strongCount : worthCount}
                    </span>
                    {tier === 'worth_a_look' && (
                      <span className="text-xs text-muted-foreground/70">
                        adjacent &amp; stretch roles
                      </span>
                    )}
                    <div className="h-px flex-1 bg-border" />
                  </div>
                )}
                <Card
                  className={`group border-border/50 transition-all ${
                    isSelected ? 'border-primary/50 bg-primary/5' : 'bg-card'
                  }`}
                >
                <CardContent className="p-4">
                  {/* Row header */}
                  <div className="flex items-center gap-4">
                    {/* Checkbox */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        toggleSelect(job.id)
                      }}
                      className="shrink-0"
                    >
                      {isSelected ? (
                        <CheckSquare className="size-5 text-primary" />
                      ) : (
                        <Square className="size-5 text-muted-foreground hover:text-foreground" />
                      )}
                    </button>

                    {/* Expand chevron */}
                    <button
                      onClick={() => toggleExpand(jobKey)}
                      className="shrink-0"
                    >
                      <ChevronDown
                        className={cn(
                          'size-4 text-muted-foreground transition-transform duration-200',
                          isExpanded && 'rotate-180'
                        )}
                      />
                    </button>

                    {/* Job info — clickable to expand */}
                    <div
                      className="flex min-w-0 flex-1 cursor-pointer flex-col gap-1"
                      onClick={() => toggleExpand(jobKey)}
                    >
                      <div className="flex items-center gap-2">
                        <h3 className="truncate text-base font-semibold text-foreground">
                          {job.title}
                        </h3>
                        <MatchBadge level={job.match_level} />
                        {agePill(job.date_posted)}
                        {postingChipText(job.posting_read) && (
                          <span
                            className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground"
                            title={(job.posting_read?.reasons ?? []).join('; ')}
                          >
                            {postingChipText(job.posting_read)}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {job.company}
                        {job.location ? ` \u00B7 ${job.location}` : ''}
                      </p>
                      <p className="mt-0.5 flex flex-wrap items-center gap-2 text-sm">
                        {job.comp_text ? (
                          <span className="font-semibold text-green-700 dark:text-green-500">
                            {'\uD83D\uDCB0'} {job.comp_text}
                          </span>
                        ) : (
                          <span className="text-muted-foreground/60">
                            {'\uD83D\uDCB0'} N/A
                          </span>
                        )}
                        {job.employment_type && (
                          <span className="rounded-full bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
                            {job.employment_type}
                          </span>
                        )}
                        {job.job_level && (
                          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                            Level: {job.job_level}
                          </span>
                        )}
                      </p>
                      {contractPayNote(job.employment_type, job.comp_text, job.company) && (
                        <p className="mt-0.5 text-xs leading-snug text-amber-800 dark:text-amber-400">
                          {contractPayNote(job.employment_type, job.comp_text, job.company)}
                        </p>
                      )}
                      {job.home_state_highlight && (
                        <p className="mt-1 text-xs font-medium text-green-700 dark:text-green-500">
                          {'\u2713'} Matching location: {job.home_state_highlight}
                        </p>
                      )}
                      {!isExpanded && job.reasoning && (
                        <p className="mt-1 line-clamp-2 text-sm leading-relaxed text-muted-foreground/80">
                          {job.reasoning}
                        </p>
                      )}
                    </div>

                    {/* Confidence */}
                    <div className="hidden shrink-0 sm:block">
                      <ConfidenceBar value={job.confidence} />
                    </div>

                    {/* Actions */}
                    <div className="flex shrink-0 items-center gap-2">
                      {job.link && (
                        <a
                          href={job.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-3 py-1.5 text-sm font-medium text-foreground shadow-sm transition-colors hover:bg-muted"
                          onClick={(e) => e.stopPropagation()}
                          title="Open the original posting"
                        >
                          Posting
                          <ExternalLink className="size-3.5" />
                        </a>
                      )}
                      <Button
                        variant="default"
                        size="sm"
                        className="gap-1.5"
                        onClick={(e) => {
                          e.stopPropagation()
                          trackMutation.mutate(job.id)
                          setSelected((prev) => {
                            const next = new Set(prev)
                            next.delete(job.id)
                            return next
                          })
                        }}
                        disabled={trackMutation.isPending}
                      >
                        <Bookmark className="size-3.5" />
                        Track
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="gap-1.5 text-muted-foreground hover:text-destructive"
                        onClick={(e) => {
                          e.stopPropagation()
                          setDismissTarget({
                            id: job.id,
                            title: job.title,
                          })
                        }}
                      >
                        <XCircle className="size-3.5" />
                        Dismiss
                      </Button>
                    </div>
                  </div>

                  {/* Expanded content */}
                  <div
                    className={cn(
                      'transition-all duration-200',
                      isExpanded
                        ? 'mt-2 max-h-[80vh] overflow-y-auto opacity-100'
                        : 'max-h-0 overflow-hidden opacity-0'
                    )}
                  >
                    <div className="space-y-4 border-t border-border pt-3">
                      {/* AI Reasoning */}
                      {job.reasoning && (
                        <div>
                          <div className="mb-1.5 text-sm font-medium uppercase tracking-wider text-muted-foreground">
                            AI Reasoning
                          </div>
                          <div className="rounded-lg border border-border bg-muted/50 p-3">
                            <p className="text-base leading-relaxed text-foreground">
                              {job.reasoning}
                            </p>
                          </div>
                        </div>
                      )}

                      {/* Posting check — fetched only once the row is open */}
                      <PostingCheckPanel jobId={job.id} enabled={isExpanded} />

                      {/* Company Intel */}
                      <div>
                        <div className="mb-1.5 text-sm font-medium uppercase tracking-wider text-muted-foreground">
                          Company Intel
                        </div>
                        <CompanyIntelPanel
                          company={job.company}
                          jobTitle={job.title}
                          matchLevel={job.match_level}
                        />
                      </div>

                      {/* Documents */}
                      <div>
                        <div className="mb-1.5 text-sm font-medium uppercase tracking-wider text-muted-foreground">
                          Documents
                        </div>
                        <DocumentPanel job={job} />
                      </div>

                      {/* View Original Posting */}
                      {job.link && (
                        <div>
                          <a
                            href={job.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className={cn(
                              buttonVariants({
                                variant: 'outline',
                                size: 'sm',
                              }),
                              'w-fit gap-1.5 no-underline'
                            )}
                          >
                            <ExternalLink className="size-3.5" />
                            View Original Posting
                          </a>
                        </div>
                      )}

                      {/* Full Description */}
                      {descriptionText && (
                        <div>
                          <div className="mb-1.5 text-sm font-medium uppercase tracking-wider text-muted-foreground">
                            Description
                          </div>
                          <div className="rounded-lg border border-border bg-muted/50 p-3">
                            <p className="whitespace-pre-wrap text-base leading-relaxed text-foreground">
                              {isDescriptionLong && !showFull
                                ? descriptionText.slice(
                                    0,
                                    DESCRIPTION_TRUNCATE_LENGTH
                                  ) + '...'
                                : descriptionText}
                            </p>
                            {isDescriptionLong && (
                              <button
                                onClick={() => toggleFullDescription(jobKey)}
                                className="mt-2 text-sm font-medium text-primary hover:underline"
                              >
                                {showFull ? 'Show less' : 'Show more'}
                              </button>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
                </Card>
              </Fragment>
            )
          })}
        </div>
      )}

      {/* Single dismiss modal */}
      <DismissModal
        open={!!dismissTarget}
        onOpenChange={(open) => {
          if (!open) setDismissTarget(null)
        }}
        onDismiss={handleDismissOne}
        jobTitle={dismissTarget?.title}
      />

      {/* Batch dismiss modal */}
      <DismissModal
        open={batchDismissOpen}
        onOpenChange={setBatchDismissOpen}
        onDismiss={handleBatchDismiss}
        jobTitle={`${selected.size} selected jobs`}
      />
    </div>
  )
}
