'use client'

import { useState, useCallback, useRef, useEffect, Fragment } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { cn } from '@/lib/utils'
import type { ApplicationOutcome, GeneratingJobs, KanbanStatus } from '@/lib/types'

// The API returns flat application rows (not nested job object). The
// ApplicationOutcome fields come from the outcome ledger; Partial because a
// not-yet-restarted API does not send them.
interface FlatApplication extends Partial<ApplicationOutcome> {
  app_id: number
  job_id: number
  status: KanbanStatus
  notes: string | null
  date_applied: string | null
  title: string
  company: string
  link: string | null
  location: string | null
  cv_path: string | null
  cl_path: string | null
  cv_drive_id: string | null
  cl_drive_id: string | null
  match_level: string | null
  confidence: number | null
  keyword_category: string | null
  cv_quality_flag: string | null
  cv_audit_summary: string | null
  cv_domain_used: string | null
  intel_status: string | null
  intel_path: string | null
  // The interview stage the prep sheet was built for. NULL when the sheet
  // has no recorded stage.
  prep_stage?: string | null
  // Phase 27c: 'gone' = the company's own job board stopped listing it (two
  // complete checks in a row). Cards are never hidden — they just say so.
  listing_status?: string | null
  listing_gone_since?: string | null
}
import { MatchBadge } from '@/components/MatchBadge'
import { CVQABadge } from '@/components/CVQABadge'
import { DocumentPanel } from '@/components/DocumentPanel'
import { CompanyIntelPanel } from '@/components/CompanyIntelPanel'
import { PostingCheckPanel } from '@/components/PostingCheckPanel'
import { OutreachIntelPanel } from '@/components/OutreachIntelPanel'
import { InterviewPrepPanel } from '@/components/InterviewPrepPanel'
import { WarmPathsBadge } from '@/components/WarmPathsBadge'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  Kanban,
  FileText,
  FileSignature,
  Loader2,
  Inbox,
  ChevronDown,
  ChevronUp,
  MessageSquare,
  Trash2,
  Sparkles,
} from 'lucide-react'

const KANBAN_STATUSES: KanbanStatus[] = [
  'To Apply',
  'Applied',
  'Screening Call',
  'Interview',
  'Final Interview',
  'Offer',
  'Accepted',
  'Rejected',
  'Ghosted',
  'I Withdrew',
]

const STATUS_COLORS: Record<string, string> = {
  'To Apply': 'text-blue-400',
  Applied: 'text-cyan-400',
  'Screening Call': 'text-amber-400',
  Interview: 'text-amber-300',
  'Final Interview': 'text-orange-400',
  Offer: 'text-emerald-400',
  Accepted: 'text-emerald-300',
  Rejected: 'text-red-400',
  Ghosted: 'text-slate-500',
  'I Withdrew': 'text-indigo-300',
}

const STATUS_DOT_COLORS: Record<string, string> = {
  'To Apply': 'bg-blue-400',
  Applied: 'bg-cyan-400',
  'Screening Call': 'bg-amber-400',
  Interview: 'bg-amber-300',
  'Final Interview': 'bg-orange-400',
  Offer: 'bg-emerald-400',
  Accepted: 'bg-emerald-300',
  Rejected: 'bg-red-400',
  Ghosted: 'bg-slate-500',
  'I Withdrew': 'bg-indigo-300',
}

// Stages where an interview is on the table (prep sheet is live there).
const INTERVIEW_STAGES: string[] = ['Screening Call', 'Interview', 'Final Interview', 'Offer']
// "Reached interview" = Screening Call or later.
const REACHED_INTERVIEW: string[] = [...INTERVIEW_STAGES, 'Accepted']
// Closed cards: the company said no, it went quiet, or the user ended it.
// On one of these the user says how far the process got and what happened —
// any time, not only when the card closes; that answer becomes the card's
// furthest stage (2026-10-28).
const WITHDREW = 'I Withdrew'
const CLOSED_STATUSES: string[] = ['Rejected', 'Ghosted', WITHDREW]

/** 'Mar 10' from a ledger timestamp — UTC 'YYYY-MM-DD HH:MM:SS' (shown in
 *  local time) or a bare 'YYYY-MM-DD'. An inferred date gets a leading '~'.
 *  Empty string when the date is unknown. */
function stageDate(at: string | null | undefined, inferred = false): string {
  if (!at) return ''
  const d = at.length <= 10 ? new Date(`${at}T12:00:00`) : new Date(`${at.replace(' ', 'T')}Z`)
  if (Number.isNaN(d.getTime())) return ''
  const label = d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    ...(d.getFullYear() !== new Date().getFullYear() ? { year: 'numeric' as const } : {}),
  })
  return inferred ? `~${label}` : label
}

// An inferred entry (a stage guessed from a prep sheet, not recorded as a
// move) proves only that SOME interview stage was reached — never which one —
// so it is shown as uncertain, not as the guessed stage.
const UNCONFIRMED_STAGE = 'an interview stage'

/** Muted caption under the status of a dead card that got interviews:
 *  'reached Interview · Mar 10'; when inferred, 'reached an interview stage ·
 *  ~Mar 10' (the stage and the date are both unconfirmed). */
function reachCaption(app: FlatApplication): string | null {
  if (!CLOSED_STATUSES.includes(app.status)) return null
  if (!app.highest_stage || !REACHED_INTERVIEW.includes(app.highest_stage)) return null
  const date = stageDate(app.reached_at, app.reached_inferred)
  const stage = app.reached_inferred ? UNCONFIRMED_STAGE : app.highest_stage
  return `reached ${stage}${date ? ` · ${date}` : ''}`
}

/** One-line card history: 'Applied Mar 3 → Interview Mar 10 → Rejected Apr 2'
 *  ('an interview stage ~Mar 10' for an inferred entry). Null when there is
 *  nothing worth a line (a lone undated status). */
function historyLine(app: FlatApplication): string | null {
  const history = app.history ?? []
  if (history.length === 0 || (history.length === 1 && !history[0].at)) return null
  return history
    .map((h) => {
      const date = stageDate(h.at, h.inferred)
      const stage = h.inferred ? UNCONFIRMED_STAGE : h.status
      return date ? `${stage} ${date}` : stage
    })
    .join(' → ')
}

/** The one posting-check fact a card shows: 'No longer on Acme's job board
 *  since Mar 3'. Null unless the company board stopped listing the job. */
function goneLine(app: FlatApplication): string | null {
  if (app.listing_status !== 'gone') return null
  const since = stageDate(app.listing_gone_since)
  return `No longer on ${app.company}'s job board${since ? ` since ${since}` : ''}`
}
// end goneLine

/** Which interview prep panel the expanded card shows: 'live' while an
 *  interview is on the table; 'readonly' for a card that has a sheet and whose
 *  history reached an interview, so it survives a later close (Rejected,
 *  Ghosted or I Withdrew); null otherwise.
 *  A card whose first recorded status is Rejected / Ghosted has no ranked
 *  stage (highest_stage null). Its sheet stays readable when the sheet has no
 *  stage either (prep_stage null) — Help: "the sheet stays on the card to
 *  read". A sheet built by a mis-click that the ledger corrected (prep_stage
 *  set, no reach recorded) stays hidden. */
function prepPanel(app: FlatApplication): 'live' | 'readonly' | null {
  if (INTERVIEW_STAGES.includes(app.status)) return 'live'
  if (!app.has_prep) return null
  if (app.highest_stage && REACHED_INTERVIEW.includes(app.highest_stage)) return 'readonly'
  if (!app.highest_stage && !app.prep_stage) return 'readonly'
  return null
}

const OUTCOME_STAGES: string[] = ['Applied', 'Screening Call', 'Interview', 'Final Interview', 'Offer']
// One tap adds a common answer; the user can still type their own.
const OUTCOME_HINTS: string[] = [
  'They chose another candidate',
  'Role cancelled or on hold',
  'Pay didn’t match',
  'No feedback given',
  'Never heard back',
]
// …and the user's own reasons, when they ended it.
const WITHDREW_HINTS: string[] = [
  'Pay didn’t match',
  'Took another offer',
  'Not the right fit',
  'Role wasn’t as described',
  'Location or schedule',
]
const ENDED_AFTER: Record<string, string> = {
  Applied: 'applying',
  'Screening Call': 'the screening call',
  Interview: 'an interview',
  'Final Interview': 'the final interview',
  Offer: 'the offer',
  Accepted: 'accepting the offer',
}

function isClosed(app: FlatApplication): boolean {
  return CLOSED_STATUSES.includes(app.status)
}

/** The question, in the words of how the card closed. */
function outcomeQuestion(app: FlatApplication): string {
  if (app.status === 'Ghosted') return 'How far did it get before it went quiet?'
  if (app.status === WITHDREW) return 'How far did it get before you withdrew?'
  return 'How far did it get?'
}

/** The ready answers to tap: the company's reasons, or the user's own. */
function outcomeHints(app: FlatApplication): string[] {
  return app.status === WITHDREW ? WITHDREW_HINTS : OUTCOME_HINTS
}

/** The stages the answer offers ('Accepted' only when the card got there —
 *  an offer withdrawn after it was accepted). */
function outcomeStages(app: FlatApplication): string[] {
  return app.highest_stage === 'Accepted' || app.outcome?.stage === 'Accepted'
    ? [...OUTCOME_STAGES, 'Accepted']
    : OUTCOME_STAGES
}

/** Where the question starts: the last answer, else the furthest stage the
 *  history knows (an inferred stage too — the user confirms or changes
 *  it), else Applied. */
function outcomePrefill(app: FlatApplication): string {
  if (app.outcome?.stage) return app.outcome.stage
  if (app.highest_stage && outcomeStages(app).includes(app.highest_stage)) return app.highest_stage
  return 'Applied'
}

/** The answer in one line: 'Ended after an interview — “They chose another
 *  candidate”' ('Went quiet after …' when Ghosted; 'You withdrew after …', or
 *  'You turned down the offer', when the user ended it). Null with no answer. */
function endedLine(app: FlatApplication): string | null {
  if (!isClosed(app) || !app.outcome) return null
  const after = ENDED_AFTER[app.outcome.stage] ?? app.outcome.stage
  const head =
    app.status === WITHDREW
      ? app.outcome.stage === 'Offer'
        ? 'You turned down the offer'
        : `You withdrew after ${after}`
      : `${app.status === 'Ghosted' ? 'Went quiet after' : 'Ended after'} ${after}`
  return app.outcome.what_happened ? `${head} — “${app.outcome.what_happened}”` : head
}

/** A tapped hint joins the typed text; it never replaces it. */
function withHint(text: string, hint: string): string {
  const typed = text.trim()
  if (!typed) return hint
  if (typed.toLowerCase().includes(hint.toLowerCase())) return typed
  return `${typed.replace(/[.;,]$/, '')}; ${hint}`
}
// end outcome helpers

/** Debounced notes input — compact inline version for the table cell. */
function NotesCell({
  application,
  onSave,
}: {
  application: FlatApplication
  onSave: (id: number, notes: string) => void
}) {
  const [value, setValue] = useState(application.notes ?? '')
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Sync external updates
  useEffect(() => {
    setValue(application.notes ?? '')
  }, [application.notes])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const next = e.target.value
    setValue(next)
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      onSave(application.app_id, next)
    }, 800)
  }

  return (
    <Input
      value={value}
      onChange={handleChange}
      placeholder="Add notes..."
      className="h-8 border-border/30 bg-muted/30 text-sm text-foreground placeholder:text-muted-foreground/50 hover:border-border focus:border-border focus:bg-muted/50"
    />
  )
}

/** Expanded notes textarea with auto-save on blur and debounce. */
function ExpandedNotes({
  application,
  onSave,
}: {
  application: FlatApplication
  onSave: (id: number, notes: string) => void
}) {
  const [value, setValue] = useState(application.notes ?? '')
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const savedRef = useRef(application.notes ?? '')

  useEffect(() => {
    setValue(application.notes ?? '')
    savedRef.current = application.notes ?? ''
  }, [application.notes])

  const flush = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    if (value !== savedRef.current) {
      onSave(application.app_id, value)
      savedRef.current = value
    }
  }, [application.app_id, onSave, value])

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const next = e.target.value
    setValue(next)
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      onSave(application.app_id, next)
      savedRef.current = next
    }, 800)
  }

  return (
    <Textarea
      value={value}
      onChange={handleChange}
      onBlur={flush}
      placeholder="Add notes, interview prep, follow-up reminders..."
      rows={5}
      className="min-h-[120px] resize-y border-border/30 bg-muted/30 text-base leading-relaxed text-foreground placeholder:text-muted-foreground/50 hover:border-border focus:border-border focus:bg-muted/50"
    />
  )
}

/** "How far did it get?" on a closed card — inside the open card, never a
 *  pop-up. It opens by itself when a card is moved to Rejected, Ghosted or
 *  I Withdrew, and from "Add details" on any closed card later; "Not now"
 *  leaves the card as it is (the move is already saved). */
function OutcomeForm({
  app,
  onDone,
}: {
  app: FlatApplication
  onDone: (saved: boolean) => void
}) {
  const queryClient = useQueryClient()
  const [stage, setStage] = useState(() => outcomePrefill(app))
  const [text, setText] = useState(app.outcome?.what_happened ?? '')
  // A zero-size mark at the form's top-left corner.
  const anchorRef = useRef<HTMLSpanElement>(null)

  // A moved card jumps to the top of the list: bring the question into view.
  // Scrolling to the corner (not the whole box) only ever moves the page up or
  // down — the open card spans the full table width, so scrolling the box
  // itself into view slid the table sideways and hid the question's start.
  useEffect(() => {
    anchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' })
  }, [])

  const save = useMutation({
    mutationFn: () =>
      api.applications.setOutcome(app.app_id, {
        stage,
        what_happened: text.trim() || null,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['applications'] })
      queryClient.invalidateQueries({ queryKey: ['stats-overview'] })
      onDone(true)
    },
  })

  return (
    <div className="relative max-w-3xl space-y-3 rounded-lg border border-border/60 bg-background/70 p-4">
      <span ref={anchorRef} aria-hidden className="absolute left-0 top-0 size-0" />
      <p className="text-base font-medium text-foreground">{outcomeQuestion(app)}</p>
      <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Furthest stage reached">
        {outcomeStages(app).map((s) => (
          <button
            key={s}
            type="button"
            role="radio"
            aria-checked={stage === s}
            onClick={() => setStage(s)}
            className={cn(
              'rounded-full border px-3 py-1 text-sm transition-colors',
              stage === s
                ? 'border-primary bg-primary text-primary-foreground'
                : 'border-border/60 bg-card text-foreground hover:border-border hover:bg-muted/50'
            )}
          >
            {s}
          </button>
        ))}
      </div>
      <div className="space-y-1.5">
        <label htmlFor={`outcome-${app.app_id}`} className="text-sm text-muted-foreground">
          What happened? <span className="text-muted-foreground/70">(optional)</span>
        </label>
        <Input
          id={`outcome-${app.app_id}`}
          value={text}
          maxLength={300}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !save.isPending) save.mutate()
          }}
          placeholder={
            app.status === WITHDREW
              ? 'e.g. The role turned out to be different from the posting'
              : 'e.g. They went with an internal candidate'
          }
          className="h-9 border-border/40 bg-muted/30 text-base text-foreground placeholder:text-muted-foreground/50"
        />
        <div className="flex flex-wrap gap-1.5 pt-0.5">
          {outcomeHints(app).map((h) => (
            <button
              key={h}
              type="button"
              onClick={() => setText((t) => withHint(t, h).slice(0, 300))}
              className="rounded-full border border-dashed border-border/70 px-2.5 py-0.5 text-xs text-muted-foreground hover:border-border hover:text-foreground"
            >
              {h}
            </button>
          ))}
        </div>
      </div>
      {save.isError && (
        <p className="text-sm text-red-500">Could not save it — please try again.</p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" onClick={() => save.mutate()} disabled={save.isPending}>
          {save.isPending && <Loader2 className="size-3.5 animate-spin" />}
          Save
        </Button>
        <Button size="sm" variant="ghost" onClick={() => onDone(false)} disabled={save.isPending}>
          Not now
        </Button>
        <span className="text-xs text-muted-foreground">
          Shows on the card and on the Stats page.
        </span>
      </div>
    </div>
  )
}

export default function KanbanPage() {
  const queryClient = useQueryClient()
  const [expandedRow, setExpandedRow] = useState<number | null>(null)
  // The closed card whose "How far did it get?" question is open.
  const [outcomeEdit, setOutcomeEdit] = useState<number | null>(null)
  // A short line under the filters (where a closed card went); clears itself.
  const [notice, setNotice] = useState<string | null>(null)
  useEffect(() => {
    if (!notice) return
    const t = setTimeout(() => setNotice(null), 6000)
    return () => clearTimeout(t)
  }, [notice])
  const [quickApplyOpen, setQuickApplyOpen] = useState(false)
  const [activeFilters, setActiveFilters] = useState<Set<string>>(new Set())
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [sortKey, setSortKey] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')

  const toggleSelect = useCallback((jobId: number) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(jobId)) next.delete(jobId)
      else next.add(jobId)
      return next
    })
  }, [])

  const toggleSort = useCallback((key: string) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }, [sortKey])

  const {
    data: applications = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ['applications'],
    queryFn: api.applications.list,
  })

  const { data: configData } = useQuery({
    queryKey: ['config'],
    queryFn: api.config,
    staleTime: 60_000,
  })

  const updateMutation = useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: number
      body: { status?: string; notes?: string }
    }) => api.applications.update(id, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['applications'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (appId: number) => api.applications.delete(appId),
    onMutate: async (appId: number) => {
      await queryClient.cancelQueries({ queryKey: ['applications'] })
      const previous = queryClient.getQueryData<FlatApplication[]>(['applications'])
      queryClient.setQueryData<FlatApplication[]>(['applications'], (old) =>
        old ? old.filter((a) => a.app_id !== appId) : []
      )
      return { previous }
    },
    onError: (_err, _id, context) => {
      if (context?.previous) {
        queryClient.setQueryData(['applications'], context.previous)
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['applications'] })
    },
  })

  // Which cards are generating their CV + cover letter. The server knows
  // every generation in flight (a click, a batch, or the background one that
  // a job added by hand starts); this page also marks a card the moment it is
  // clicked. A click while that job is already generating answers
  // 'in_progress', and the card keeps saying "Generating…" instead of
  // flicking back to Generate as if nothing happened.
  const [clickedGen, setClickedGen] = useState<Set<number>>(new Set())
  const { data: generatingData } = useQuery({
    queryKey: ['generating'],
    queryFn: api.jobs.generating,
    refetchInterval: (query) => (query.state.data?.job_ids.length ? 5000 : false),
  })
  const serverGenerating = generatingData?.job_ids
  // A generation that ended: refresh the board so its View button appears.
  const prevGenerating = useRef<number[]>([])
  useEffect(() => {
    const now = serverGenerating ?? []
    if (prevGenerating.current.some((id) => !now.includes(id))) {
      queryClient.invalidateQueries({ queryKey: ['applications'] })
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    }
    prevGenerating.current = now
  }, [serverGenerating, queryClient])
  const isGenerating = (jobId: number) =>
    clickedGen.has(jobId) || !!serverGenerating?.includes(jobId)

  const generateDocsMutation = useMutation({
    mutationFn: (jobId: number) => api.jobs.generateDocs(jobId),
    onMutate: (jobId) => {
      setClickedGen((prev) => new Set(prev).add(jobId))
    },
    onSettled: async (res, err, jobId) => {
      queryClient.invalidateQueries({ queryKey: ['applications'] })
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      // Before un-marking the card, learn whether the server is still at it
      // (an 'in_progress' answer, or a dropped request whose generation goes on).
      await queryClient.invalidateQueries({ queryKey: ['generating'] })
      const still = queryClient
        .getQueryData<GeneratingJobs>(['generating'])
        ?.job_ids.includes(jobId)
      if (!still && (err || res?.status === 'failed')) {
        const company = applications.find((a: FlatApplication) => a.job_id === jobId)?.company
        setNotice(`Couldn't make the CV${company ? ` for ${company}` : ''}. Please try again.`)
      }
      setClickedGen((prev) => {
        const next = new Set(prev)
        next.delete(jobId)
        return next
      })
    },
  })

  // Batch generation runs server-side in the background: one job can take
  // several minutes at the xhigh generation effort, so waiting on a single
  // request used to die at the proxy while the CVs kept arriving. Poll the
  // batch status only while one is running.
  const [batchProgress, setBatchProgress] = useState<string | null>(null)
  const { data: batch } = useQuery({
    queryKey: ['batch-status'],
    queryFn: api.jobs.batchStatus,
    refetchInterval: (query) => (query.state.data?.running ? 5000 : false),
  })
  const batchRunning = batch?.running ?? false

  // Refresh the board as each job finishes (CVs appear one by one), and
  // summarise once when a batch this page watched comes to an end.
  const seenDone = useRef<number | null>(null)
  const sawRunning = useRef(false)
  useEffect(() => {
    if (!batch) return
    if (seenDone.current !== null && batch.done > seenDone.current) {
      queryClient.invalidateQueries({ queryKey: ['applications'] })
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    }
    seenDone.current = batch.done
    if (batch.running) {
      sawRunning.current = true
      return
    }
    if (sawRunning.current) {
      sawRunning.current = false
      const cvGen = batch.results.filter(
        (r) => r.cv && r.cv !== 'exists' && r.cv !== 'failed' && !r.cv.startsWith('error') && !r.cv.startsWith('skipped'),
      ).length
      const intelGen = batch.results.filter((r) => r.intel === 'generated').length
      setBatchProgress(
        batch.error
          ? `Batch stopped: ${batch.error}`
          : `Done! ${cvGen} CVs generated, ${intelGen} companies researched.`,
      )
      setTimeout(() => setBatchProgress(null), 8000)
    }
  }, [batch, queryClient])

  const batchGenerateMutation = useMutation({
    mutationFn: (jobIds: number[]) => api.jobs.batchGenerate(jobIds),
    onSuccess: () => {
      setBatchProgress(null)
      queryClient.invalidateQueries({ queryKey: ['batch-status'] })   // starts the polling
    },
    onError: (e: Error) => {
      setBatchProgress(
        e.message.includes('409')
          ? 'A batch is already running — its progress is shown here.'
          : `Could not start the batch: ${e.message}`,
      )
      queryClient.invalidateQueries({ queryKey: ['batch-status'] })
      setTimeout(() => setBatchProgress(null), 6000)
    },
  })

  const handleBatchGenerate = useCallback(() => {
    if (selected.size === 0) {
      setBatchProgress('Select jobs first using the checkboxes, then click Batch Generate.')
      setTimeout(() => setBatchProgress(null), 4000)
      return
    }
    batchGenerateMutation.mutate(Array.from(selected))
  }, [selected, batchGenerateMutation])

  const batchLine = batchRunning && batch
    ? (() => {
        const cur = applications.find((a) => a.job_id === batch.current)
        const label = cur ? ` — ${cur.company}: ${cur.title}` : ''
        return `Generating ${Math.min(batch.done + 1, batch.total)} of ${batch.total}${label}. ` +
          'Each CV takes a few minutes; they appear on the board as they finish — you can keep working.'
      })()
    : null

  const handleStatusChange = useCallback(
    (id: number, status: string, from?: string, answered = false) => {
      updateMutation.mutate({ id, body: { status } })
      // Closing a card asks how far it got, right in the open card — and so
      // does switching how it closed (Rejected → I Withdrew) before the
      // question is answered. Moving it back to a live stage puts the question away.
      if (CLOSED_STATUSES.includes(status)) {
        if (!(from && CLOSED_STATUSES.includes(from) && answered)) {
          setExpandedRow(id)
          setOutcomeEdit(id)
        }
      } else {
        setOutcomeEdit((cur) => (cur === id ? null : cur))
      }
    },
    [updateMutation]
  )

  const handleNotesSave = useCallback(
    (id: number, notes: string) => {
      updateMutation.mutate({ id, body: { notes } })
    },
    [updateMutation]
  )

  // Under a filter the card no longer matches, a closed card leaves the list
  // once its question is put away: say where it went, so it does not just
  // vanish. True when it leaves.
  const leavesView = (id: number, saved: boolean): boolean => {
    const app = applications.find((a: FlatApplication) => a.app_id === id)
    if (!app || activeFilters.size === 0 || activeFilters.has(app.status)) return false
    setNotice(`${saved ? 'Saved. ' : ''}${app.company} is under ${app.status} now.`)
    return true
  }

  // Saved or "Not now".
  const closeOutcome = (id: number, saved: boolean) => {
    setOutcomeEdit(null)
    if (leavesView(id, saved)) setExpandedRow(null)
  }

  // "Add details" on a closed card in the list: open the card at the question.
  const openOutcome = (id: number) => {
    if (outcomeEdit !== null && outcomeEdit !== id) leavesView(outcomeEdit, false)
    setExpandedRow(id)
    setOutcomeEdit(id)
  }

  const toggleExpand = (id: number) => {
    // Closing the open card, or opening another, puts its question away.
    if (outcomeEdit !== null) {
      setOutcomeEdit(null)
      leavesView(outcomeEdit, false)
    }
    setExpandedRow((prev) => (prev === id ? null : id))
  }

  const toggleFilter = useCallback((status: string) => {
    setActiveFilters((prev) => {
      const next = new Set(prev)
      if (next.has(status)) {
        next.delete(status)
      } else {
        next.add(status)
      }
      return next
    })
  }, [])

  // Quick-apply answers from config
  const quickApplyAnswers = configData?.standard_answers ?? null  // API key is standard_answers

  // Loading state
  if (isLoading) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="size-8 animate-spin text-muted-foreground" />
          <p className="text-base text-muted-foreground">Loading pipeline...</p>
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
            <Kanban className="size-6 text-red-400" />
          </div>
          <p className="text-base font-medium text-foreground">
            Failed to load pipeline
          </p>
          <p className="max-w-xs text-sm text-muted-foreground">
            {error instanceof Error
              ? error.message
              : 'An unknown error occurred.'}
          </p>
        </div>
      </div>
    )
  }

  // Status summary counts
  const statusCounts: Record<string, number> = {}
  applications.forEach((app: FlatApplication) => {
    statusCounts[app.status] = (statusCounts[app.status] ?? 0) + 1
  })

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          Kanban
        </h1>
        <p className="mt-1 text-base text-muted-foreground">
          {applications.length} application
          {applications.length !== 1 ? 's' : ''} in the pipeline
        </p>
      </div>

      {/* Status filter chips — click to toggle filter */}
      {applications.length > 0 && (<>
        <div className="flex flex-wrap items-center gap-2">
          {KANBAN_STATUSES.filter((s) => statusCounts[s]).map((status) => {
            const isActive = activeFilters.has(status)
            return (
              <button
                key={status}
                onClick={() => toggleFilter(status)}
                className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm transition-all ${
                  isActive
                    ? 'border-primary bg-primary/10 ring-1 ring-primary/30'
                    : 'border-border/50 bg-card hover:border-border hover:bg-muted/50'
                }`}
              >
                <span
                  className={`font-medium ${STATUS_COLORS[status] ?? 'text-foreground'}`}
                >
                  {status}
                </span>
                <span className="tabular-nums text-muted-foreground">
                  {statusCounts[status]}
                </span>
              </button>
            )
          })}
          {activeFilters.size > 0 && (
            <button
              onClick={() => setActiveFilters(new Set())}
              className="flex items-center gap-1 rounded-full border border-border/50 bg-card px-3 py-1 text-sm text-muted-foreground hover:border-border hover:text-foreground"
            >
              Clear filters
            </button>
          )}
          <Button
            variant="outline"
            size="sm"
            className="ml-auto gap-1.5"
            onClick={handleBatchGenerate}
            disabled={batchGenerateMutation.isPending || batchRunning}
          >
            {batchGenerateMutation.isPending || batchRunning ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Sparkles className="size-3.5" />
            )}
            {batchGenerateMutation.isPending || batchRunning
              ? batchRunning && batch
                ? `Generating ${batch.done}/${batch.total}…`
                : 'Starting…'
              : selected.size > 0
                ? `Batch Generate (${selected.size} selected)`
                : 'Batch Generate CVs & Intel'}
          </Button>
        </div>
        {(batchLine || batchProgress || notice) && (
          <p className="text-sm text-muted-foreground">{batchLine ?? batchProgress ?? notice}</p>
        )}
      </>)}

      {/* Empty state */}
      {applications.length === 0 ? (
        <div className="flex h-[50vh] items-center justify-center">
          <div className="flex flex-col items-center gap-4 text-center">
            <div className="flex size-16 items-center justify-center rounded-2xl bg-muted">
              <Inbox className="size-8 text-muted-foreground" />
            </div>
            <div>
              <p className="text-base font-medium text-foreground">
                No applications yet
              </p>
              <p className="mt-1 max-w-xs text-sm text-muted-foreground">
                Track jobs from the Jobs or Review pages to start building your
                pipeline.
              </p>
            </div>
          </div>
        </div>
      ) : (
        /* Applications table */
        <div className="overflow-x-auto rounded-xl border border-border/50">
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/40 hover:bg-muted/40">
                <TableHead className="w-[40px]">
                  <input
                    type="checkbox"
                    className="size-4 rounded border-border"
                    checked={(() => {
                      const visible = (activeFilters.size > 0
                        ? applications.filter((a: FlatApplication) => activeFilters.has(a.status))
                        : applications
                      ).map((a: FlatApplication) => a.job_id)
                      return visible.length > 0 && visible.every((id: number) => selected.has(id))
                    })()}
                    onChange={() => {
                      const visible = (activeFilters.size > 0
                        ? applications.filter((a: FlatApplication) => activeFilters.has(a.status))
                        : applications
                      ).map((a: FlatApplication) => a.job_id)
                      const allSelected = visible.every((id: number) => selected.has(id))
                      if (allSelected) {
                        setSelected(new Set())
                      } else {
                        setSelected(new Set(visible))
                      }
                    }}
                  />
                </TableHead>
                {[
                  { key: 'status', label: 'Status', w: 'w-[160px]' },
                  { key: 'title', label: 'Title', w: '' },
                  { key: 'company', label: 'Company', w: '' },
                  { key: 'match_level', label: 'Match', w: 'w-[110px]' },
                  { key: 'cv_quality_flag', label: 'CV QA', w: 'w-[160px]' },
                  { key: 'date_applied', label: 'Date Applied', w: 'w-[110px]' },
                ].map((col) => (
                  <TableHead
                    key={col.key}
                    className={cn(col.w, 'cursor-pointer select-none hover:text-foreground')}
                    onClick={() => toggleSort(col.key)}
                  >
                    <span className="inline-flex items-center gap-1">
                      {col.label}
                      {sortKey === col.key && (
                        <span className="text-xs">{sortDir === 'asc' ? '▲' : '▼'}</span>
                      )}
                    </span>
                  </TableHead>
                ))}
                <TableHead className="w-[140px]">Docs</TableHead>
                <TableHead className="w-[50px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {(activeFilters.size > 0
                ? applications.filter(
                    // the card just closed stays in view while its question is
                    // open on screen — never once the card is folded away
                    (a: FlatApplication) =>
                      activeFilters.has(a.status) ||
                      (a.app_id === outcomeEdit && a.app_id === expandedRow)
                  )
                : applications
              ).slice().sort((a: FlatApplication, b: FlatApplication) => {
                if (!sortKey) return 0
                const key = sortKey as keyof FlatApplication
                const va = a[key] ?? ''
                const vb = b[key] ?? ''
                const cmp = String(va).localeCompare(String(vb), undefined, { sensitivity: 'base' })
                return sortDir === 'asc' ? cmp : -cmp
              }).map((app: FlatApplication, index: number) => {
                const isExpanded = expandedRow === app.app_id
                const genBusy = isGenerating(app.job_id)
                // Build a job-like object from flat fields for DocumentPanel
                const jobLike = {
                  id: app.job_id,
                  title: app.title,
                  company: app.company,
                  link: app.link ?? '',
                  location: app.location ?? '',
                  match_level: app.match_level as any,
                  cv_quality_flag: app.cv_quality_flag,
                  cv_path: app.cv_path,
                  cl_path: app.cl_path,
                  confidence: app.confidence,
                  keyword_category: app.keyword_category,
                  cv_audit_summary: app.cv_audit_summary,
                  cv_domain_used: app.cv_domain_used,
                }

                const isEven = index % 2 === 0

                return (
                  <Fragment key={app.app_id}>
                    <TableRow
                      className={`group cursor-pointer ${isEven ? 'bg-transparent' : 'bg-muted/20'} ${isExpanded ? 'border-b-0 bg-muted/10' : ''}`}
                      onClick={() => toggleExpand(app.app_id)}
                    >
                      {/* Checkbox */}
                      <TableCell className="w-[40px]" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          checked={selected.has(app.job_id)}
                          onChange={() => toggleSelect(app.job_id)}
                          className="size-4 rounded border-border"
                        />
                      </TableCell>

                      {/* Status selector + one-click Apply for To Apply rows */}
                      <TableCell onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center gap-1.5">
                        {app.status === 'To Apply' && (
                          <Button
                            size="sm"
                            className="h-8 shrink-0 bg-cyan-600 px-2.5 text-xs font-semibold text-white hover:bg-cyan-500"
                            title="Mark as Applied — triggers the outreach playbook"
                            onClick={() => handleStatusChange(app.app_id, 'Applied')}
                          >
                            ✓ Applied
                          </Button>
                        )}
                        <Select
                          value={app.status}
                          onValueChange={(v) =>
                            v && handleStatusChange(app.app_id, v, app.status, !!app.outcome)
                          }
                        >
                          <SelectTrigger className="h-8 w-[150px] border-border/30 bg-transparent text-sm hover:border-border">
                            <SelectValue>
                              <span className="flex items-center gap-2">
                                <span className={`size-2 shrink-0 rounded-full ${STATUS_DOT_COLORS[app.status] ?? 'bg-foreground'}`} />
                                <span className={STATUS_COLORS[app.status] ?? 'text-foreground'}>
                                  {app.status}
                                </span>
                              </span>
                            </SelectValue>
                          </SelectTrigger>
                          <SelectContent>
                            {KANBAN_STATUSES.map((status) => (
                              <SelectItem key={status} value={status}>
                                <span className="flex items-center gap-2">
                                  <span className={`size-2 shrink-0 rounded-full ${STATUS_DOT_COLORS[status] ?? 'bg-foreground'}`} />
                                  <span
                                    className={
                                      STATUS_COLORS[status] ?? 'text-foreground'
                                    }
                                  >
                                    {status}
                                  </span>
                                </span>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        </div>
                        {reachCaption(app) && (
                          <span className="mt-1 block max-w-[150px] whitespace-normal pl-1 text-xs leading-tight text-muted-foreground">
                            {reachCaption(app)}
                          </span>
                        )}
                        {/* A closed card: the user's line, or a way to add one —
                            any time, not only when the card closes. */}
                        {isClosed(app) &&
                          (app.outcome?.what_happened ? (
                            <button
                              type="button"
                              onClick={() => openOutcome(app.app_id)}
                              className="mt-0.5 line-clamp-2 max-w-[150px] whitespace-normal pl-1 text-left text-xs italic leading-tight text-muted-foreground hover:text-foreground"
                              title={`${app.outcome.what_happened} (click to change)`}
                            >
                              “{app.outcome.what_happened}”
                            </button>
                          ) : (
                            <button
                              type="button"
                              onClick={() => openOutcome(app.app_id)}
                              className="mt-0.5 block pl-1 text-left text-xs text-muted-foreground hover:text-primary hover:underline"
                            >
                              {app.outcome ? 'Edit details' : '+ Add details'}
                            </button>
                          ))}
                      </TableCell>

                      {/* Title */}
                      <TableCell className="max-w-[300px]">
                        {app.link && /^https?:\/\//i.test(app.link) ? (
                          <a
                            href={app.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block truncate text-base font-medium text-foreground hover:text-primary hover:underline"
                            title={app.title}
                            onClick={(e) => e.stopPropagation()}
                          >
                            {app.title}
                          </a>
                        ) : (
                          <span className="block truncate text-base font-medium text-foreground" title={app.title}>
                            {app.title}
                          </span>
                        )}
                        {goneLine(app) && (
                          <span className="block whitespace-normal text-xs text-muted-foreground">{goneLine(app)}</span>
                        )}
                      </TableCell>

                      {/* Company */}
                      <TableCell className="max-w-[180px] truncate text-base text-muted-foreground" title={app.company}>
                        {app.company}
                      </TableCell>

                      {/* Match */}
                      <TableCell>
                        <MatchBadge level={app.match_level as any} />
                      </TableCell>

                      {/* CV QA */}
                      <TableCell className="max-w-[170px]">
                        <CVQABadge flag={app.cv_quality_flag} />
                      </TableCell>

                      {/* Date Applied */}
                      <TableCell className="text-sm tabular-nums text-muted-foreground">
                        {app.date_applied ?? '--'}
                      </TableCell>

                      {/* Expand / Docs */}
                      <TableCell>
                        <div className="flex items-center gap-1">
                          {app.cv_path || app.cl_path ? (
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-7 gap-1 px-2 text-sm"
                              onClick={(e) => {
                                e.stopPropagation()
                                toggleExpand(app.app_id)
                              }}
                            >
                              <FileText className="size-3.5" />
                              View
                              {isExpanded ? (
                                <ChevronUp className="size-3" />
                              ) : (
                                <ChevronDown className="size-3" />
                              )}
                            </Button>
                          ) : (
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-7 gap-1 px-2 text-sm text-primary hover:bg-primary hover:text-primary-foreground"
                              onClick={(e) => {
                                e.stopPropagation()
                                if (!genBusy) generateDocsMutation.mutate(app.job_id)
                              }}
                              disabled={genBusy}
                              title={genBusy ? 'Writing the CV and cover letter, usually 5–10 minutes' : undefined}
                            >
                              {genBusy ? (
                                <Loader2 className="size-3.5 animate-spin" />
                              ) : (
                                <FileSignature className="size-3.5" />
                              )}
                              {genBusy ? 'Generating…' : 'Generate'}
                            </Button>
                          )}
                          {/* Expand indicator */}
                          <span className="ml-1 text-muted-foreground/50 transition-transform group-hover:text-muted-foreground">
                            {isExpanded ? (
                              <ChevronUp className="size-4" />
                            ) : (
                              <ChevronDown className="size-4" />
                            )}
                          </span>
                        </div>
                      </TableCell>

                      {/* Delete button */}
                      <TableCell onClick={(e) => e.stopPropagation()}>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 w-7 p-0 text-muted-foreground/50 hover:text-destructive"
                          onClick={() => deleteMutation.mutate(app.app_id)}
                          disabled={deleteMutation.isPending}
                          title="Remove from pipeline"
                        >
                          <Trash2 className="size-3.5" />
                        </Button>
                      </TableCell>
                    </TableRow>

                    {/* Expanded detail panel — notes + docs + company intel */}
                    {isExpanded && (
                      <TableRow className="hover:bg-transparent">
                        <TableCell
                          colSpan={9}
                          className="border-b border-border/50 bg-muted/30 p-0"
                        >
                          <div
                            // whitespace-normal: the parent <td> is whitespace-nowrap
                            // (shadcn default) and it INHERITS — without the reset,
                            // every paragraph in these panels renders on one giant
                            // unwrapped line and blows the layout out horizontally.
                            className="animate-in fade-in slide-in-from-top-1 grid min-w-0 gap-4 overflow-hidden whitespace-normal px-6 py-5 duration-200 lg:grid-cols-2"
                            onClick={(e) => e.stopPropagation()}
                          >
                            {/* Card history from the outcome ledger */}
                            {historyLine(app) && (
                              <p className="min-w-0 text-sm text-muted-foreground lg:col-span-2">
                                {historyLine(app)}
                              </p>
                            )}

                            {/* A closed card: how far it got + what happened */}
                            {isClosed(app) && (
                              <div className="min-w-0 lg:col-span-2">
                                {outcomeEdit === app.app_id ? (
                                  <OutcomeForm
                                    app={app}
                                    onDone={(saved) => closeOutcome(app.app_id, saved)}
                                  />
                                ) : app.outcome ? (
                                  <p className="text-sm text-foreground/85">
                                    {endedLine(app)}
                                    <button
                                      type="button"
                                      className="ml-2 text-sm text-primary hover:underline"
                                      onClick={() => setOutcomeEdit(app.app_id)}
                                    >
                                      Edit
                                    </button>
                                  </p>
                                ) : (
                                  <p className="text-sm text-muted-foreground">
                                    How far did it get?
                                    <button
                                      type="button"
                                      className="ml-2 text-sm font-medium text-primary hover:underline"
                                      onClick={() => setOutcomeEdit(app.app_id)}
                                    >
                                      Add details
                                    </button>
                                  </p>
                                )}
                              </div>
                            )}

                            {/* Left column: Notes + Documents */}
                            <div className="flex min-w-0 flex-col gap-4">
                              {/* Expanded notes */}
                              <div className="space-y-1.5">
                                <span className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
                                  Notes
                                </span>
                                <ExpandedNotes
                                  application={app}
                                  onSave={handleNotesSave}
                                />
                              </div>

                              {/* Documents */}
                              <DocumentPanel
                                job={jobLike}
                                busy={genBusy}
                                onGenerate={async () => {
                                  await generateDocsMutation.mutateAsync(
                                    app.job_id
                                  )
                                }}
                                applicationId={app.app_id}
                                applicationStatus={app.status}
                                sentCv={app.sent_cv ?? null}
                                onSentCvSaved={() =>
                                  queryClient.invalidateQueries({ queryKey: ['applications'] })
                                }
                              />
                            </div>

                            {/* Right column: Posting check + Company Intel */}
                            <div className="flex min-w-0 flex-col gap-4 overflow-hidden">
                              <PostingCheckPanel jobId={app.job_id} enabled />
                              <CompanyIntelPanel
                                company={app.company}
                                jobTitle={app.title}
                                matchLevel={app.match_level}
                              />
                            </div>

                            {/* Full width: Warm paths + Outreach playbook */}
                            <div className="min-w-0 overflow-hidden lg:col-span-2">
                              <WarmPathsBadge company={app.company} />
                            </div>
                            <div className="min-w-0 overflow-hidden lg:col-span-2">
                              <OutreachIntelPanel appId={app.app_id} />
                            </div>
                            {/* Interview prep — live in an interview stage, read-only
                                after it (see prepPanel). */}
                            {prepPanel(app) && (
                              <div className="min-w-0 overflow-hidden lg:col-span-2">
                                <InterviewPrepPanel
                                  appId={app.app_id}
                                  readOnly={prepPanel(app) === 'readonly'}
                                />
                              </div>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                  </Fragment>
                )
              })}
            </TableBody>
          </Table>
        </div>
      )}

      {/* Quick Apply Answers */}
      {quickApplyAnswers && (
        <Card className="border-border/50">
          <CardHeader
            className="cursor-pointer"
            onClick={() => setQuickApplyOpen((prev) => !prev)}
          >
            <CardTitle className="flex items-center gap-2 text-sm">
              <MessageSquare className="size-4" />
              Quick Apply Answers
              {quickApplyOpen ? (
                <ChevronUp className="ml-auto size-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="ml-auto size-4 text-muted-foreground" />
              )}
            </CardTitle>
          </CardHeader>
          {quickApplyOpen && (
            <CardContent>
              <div className="space-y-4">
                {typeof quickApplyAnswers === 'object' &&
                  Object.entries(
                    quickApplyAnswers as Record<string, string>
                  ).map(([question, answer]) => (
                    <div key={question} className="space-y-1">
                      <p className="text-sm font-medium text-muted-foreground">
                        {question}
                      </p>
                      <p className="rounded-lg border border-border/50 bg-muted/50 p-3 text-base text-foreground">
                        {answer}
                      </p>
                    </div>
                  ))}
              </div>
            </CardContent>
          )}
        </Card>
      )}
    </div>
  )
}
