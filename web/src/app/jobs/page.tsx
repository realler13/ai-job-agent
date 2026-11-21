'use client'

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { useSearchStore } from '@/store/search'
import type { Job } from '@/lib/types'
import { MatchBadge } from '@/components/MatchBadge'
import { ConfidenceBar } from '@/components/ConfidenceBar'
import { JobDetailDrawer } from '@/components/JobDetailDrawer'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  Briefcase,
  Building2,
  MapPin,
  Calendar,
  Bookmark,
  Loader2,
  Inbox,
  ExternalLink,
  Eye,
  EyeOff,
  Tag,
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

export default function JobsPage() {
  const queryClient = useQueryClient()
  const searchActive = useSearchStore((s) => s.active)
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [showAll, setShowAll] = useState(false)

  const {
    data,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['jobs', 'latest-session', showAll],
    queryFn: () => api.jobs.latestSession(showAll),
    refetchInterval: searchActive ? 5000 : false,
  })

  const trackMutation = useMutation({
    mutationFn: (id: number) => api.jobs.track(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      queryClient.invalidateQueries({ queryKey: ['applications'] })
    },
  })

  const generateDocsMutation = useMutation({
    mutationFn: (id: number) => api.jobs.generateDocs(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })

  const jobs: Job[] = data?.jobs ?? []
  const sessionId = data?.session_id ?? null
  const totalCount = data?.total ?? 0

  const handleCardClick = (job: Job) => {
    setSelectedJob(job)
    setDrawerOpen(true)
  }

  const handleTrack = (id: number) => {
    trackMutation.mutate(id)
    setDrawerOpen(false)
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="size-8 animate-spin text-muted-foreground" />
          <p className="text-base text-muted-foreground">Loading jobs...</p>
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
            <Briefcase className="size-6 text-red-400" />
          </div>
          <p className="text-base font-medium text-foreground">Failed to load jobs</p>
          <p className="max-w-xs text-sm text-muted-foreground">
            {error instanceof Error ? error.message : 'An unknown error occurred.'}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            New Results
          </h1>
          <p className="mt-1 text-base text-muted-foreground">
            {showAll ? (
              <>Showing all {totalCount} jobs from latest session</>
            ) : (
              <>Showing {totalCount} relevant job{totalCount !== 1 ? 's' : ''} from latest session</>
            )}
            {sessionId && (
              <span className="ml-2 text-sm text-muted-foreground/70">
                ({sessionId})
              </span>
            )}
            {searchActive && (
              <span className="ml-2 inline-flex items-center gap-1 text-emerald-400">
                <span className="relative flex size-2">
                  <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex size-2 rounded-full bg-emerald-500" />
                </span>
                Live
              </span>
            )}
          </p>
        </div>

        {/* Show All toggle */}
        <Button
          variant={showAll ? 'default' : 'outline'}
          size="sm"
          className="gap-1.5"
          onClick={() => setShowAll(!showAll)}
        >
          {showAll ? (
            <EyeOff className="size-3.5" />
          ) : (
            <Eye className="size-3.5" />
          )}
          {showAll ? 'Show Relevant Only' : 'Show All'}
        </Button>
      </div>

      {/* Empty state */}
      {jobs.length === 0 ? (
        <div className="flex h-[50vh] items-center justify-center">
          <div className="flex flex-col items-center gap-4 text-center">
            <div className="flex size-16 items-center justify-center rounded-2xl bg-muted">
              <Inbox className="size-8 text-muted-foreground" />
            </div>
            <div>
              <p className="text-base font-medium text-foreground">
                {showAll ? 'No jobs in the latest session' : 'No relevant jobs found'}
              </p>
              <p className="mt-1 max-w-xs text-sm text-muted-foreground">
                {showAll
                  ? 'Start a search from the sidebar to discover new opportunities, or add a job manually from the Input page.'
                  : "No relevant jobs found in the latest session. Try 'Show All' to see filtered jobs, or start a new search."
                }
              </p>
            </div>
          </div>
        </div>
      ) : (
        /* Card grid */
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {jobs.map((job) => (
            <Card
              key={job.id}
              className="group cursor-pointer border-border/50 bg-card transition-all hover:border-border hover:shadow-lg hover:shadow-black/20"
              onClick={() => handleCardClick(job)}
            >
              <CardContent className="flex flex-col gap-3 p-4">
                {/* Top: match + confidence */}
                <div className="flex items-center justify-between">
                  <MatchBadge level={job.match_level} />
                  <ConfidenceBar value={job.confidence} />
                </div>

                {/* Title */}
                <h3 className="line-clamp-2 text-base font-semibold leading-snug text-foreground group-hover:text-primary transition-colors">
                  {job.title}
                </h3>

                {/* Company */}
                <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
                  <Building2 className="size-3.5 shrink-0" />
                  <span className="truncate">{job.company}</span>
                </div>

                {/* Location + Date + Category row */}
                <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
                  {job.location && (
                    <div className="flex items-center gap-1">
                      <MapPin className="size-3 shrink-0" />
                      <span className="truncate">{job.location}</span>
                    </div>
                  )}
                  {job.comp_text ? (
                    <span className="font-semibold text-green-700 dark:text-green-500">
                      💰 {job.comp_text}
                    </span>
                  ) : (
                    <span className="text-muted-foreground/60">💰 N/A</span>
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
                  {job.home_state_highlight && (
                    <div className="flex items-center gap-1 font-medium text-green-700 dark:text-green-500">
                      <MapPin className="size-3 shrink-0" />
                      <span className="truncate">✓ Matching location: {job.home_state_highlight}</span>
                    </div>
                  )}
                  {job.date_posted && (
                    <div className="flex items-center gap-1">
                      <Calendar className="size-3 shrink-0" />
                      <span>{job.date_posted}</span>
                    </div>
                  )}
                  {job.keyword_category && (
                    <div className="flex items-center gap-1">
                      <Tag className="size-3 shrink-0" />
                      <span className="truncate capitalize">
                        {job.keyword_category.replace(/_/g, ' ')}
                      </span>
                    </div>
                  )}
                </div>
                {contractPayNote(job.employment_type, job.comp_text, job.company) && (
                  <p className="text-xs leading-snug text-amber-800 dark:text-amber-400">
                    {contractPayNote(job.employment_type, job.comp_text, job.company)}
                  </p>
                )}

                {/* Actions */}
                <div className="flex items-center gap-2 border-t border-border/50 pt-3">
                  <Button
                    variant="default"
                    size="sm"
                    className="gap-1.5"
                    onClick={(e) => {
                      e.stopPropagation()
                      handleTrack(job.id)
                    }}
                    disabled={trackMutation.isPending}
                  >
                    {trackMutation.isPending ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      <Bookmark className="size-3.5" />
                    )}
                    Track
                  </Button>
                  {job.link && (
                    <a
                      href={job.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="ml-auto inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-3 py-1.5 text-sm font-medium text-foreground shadow-sm transition-colors hover:bg-muted"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Posting
                      <ExternalLink className="size-3.5" />
                    </a>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Job detail drawer */}
      <JobDetailDrawer
        job={selectedJob}
        open={drawerOpen}
        onOpenChange={setDrawerOpen}
        onTrack={handleTrack}
        onGenerateDocs={async (id) => {
          await generateDocsMutation.mutateAsync(id)
        }}
      />
    </div>
  )
}
