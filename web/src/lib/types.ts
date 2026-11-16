export interface Job {
  id: number
  title: string
  company: string
  location: string
  link: string
  date_posted: string | null
  match_level: MatchLevel
  confidence: number | null
  reasoning: string | null
  keyword_category: string | null
  reviewed: boolean
  session_id: string | null
  cv_path: string | null
  cl_path: string | null
  cv_quality_flag: string | null
  cv_audit_summary: string | null
  cv_domain_used: string | null
  description: string | null
  relevance: string | null
  rejection_reason: string | null
  tier: Tier
  created_at: string | null
  date: string | null
  // For a multi-location posting, the listed location that matches the
  // profile's search area (server-computed) — shown so the candidate sees
  // which listed location is the right one.
  ca_highlight?: string | null
  // Per-job compensation + employment type (2026-07-26), extracted
  // deterministically at save time (comp_utils.extract_comp).
  comp_text?: string | null
  employment_type?: string | null
  comp_yearly_min?: number | null
  comp_yearly_max?: number | null
  // Employer's internal level when the posting publishes one, e.g. Apple's
  // "Machine Learning ICT4–ICT5" (2026-09-26).
  job_level?: string | null
  raw_data?: Record<string, unknown>
  // "Is this posting real?" — facts, never a score. posting_read is
  // sent by /api/jobs/unreviewed only (null when nothing could be read).
  posting_read?: PostingRead | null
  listing_status?: ListingStatus | null
  listing_gone_since?: string | null
}

// 'listed' / 'gone' on the company's own job board; null = we can't tell.
export type ListingStatus = 'listed' | 'gone'

// The overall read: a summary of the posting facts, not a score. It never
// changes a job's tier or its place in the feed.
export interface PostingRead {
  // not_enough_info: no posting date and no company-board check — no chip.
  read: 'looks_active' | 'mixed' | 'check_first' | 'not_enough_info'
  label: string
  reasons: string[]
}

export type PostingFactSource =
  | 'company_board' | 'our_scanner' | 'posting_text' | 'public_source' | 'your_history'

export interface PostingFact {
  key: string
  text: string
  source: PostingFactSource
  url?: string | null
  quotes?: string[]
  caution?: boolean
}

/** GET /api/jobs/{id}/posting-check */
export interface PostingCheck extends PostingRead {
  job_id: number
  company: string
  facts: PostingFact[]
  quality: { checked_at: string; model: string | null } | null
  quality_available: boolean
}

/** POST /api/jobs/{id}/posting-check/quality */
export interface PostingQualityResponse {
  status: 'fresh' | 'cached' | 'no_text' | 'failed'
  check: PostingCheck | null
}

// Tiered output: the feed groups on this. NULL = reject (not surfaced).
export type Tier = 'strong' | 'worth_a_look' | null

export type MatchLevel =
  | 'HIGH_MATCH' | 'POTENTIAL_MATCH' | 'WORTH_A_LOOK' | 'CONFIRMED_POTENTIAL'
  | 'IRRELEVANT' | 'FILTERED_OUT' | 'DUPLICATE' | 'USER_INPUT' | 'DOWNGRADED'

// 'I Withdrew' = the candidate ended the process.
export type KanbanStatus =
  | 'To Apply' | 'Applied' | 'Screening Call' | 'Interview'
  | 'Final Interview' | 'Offer' | 'Accepted' | 'Rejected' | 'Ghosted' | 'I Withdrew'

// The candidate's one-click answer to "Did you send this CV?" on a Kanban card:
// sent as generated | sent with edits | rewrote it / sent another CV.
export type SentCvAnswer = 'as_is' | 'edited' | 'own'

// One step of a card's history line, from the outcome ledger. `at` is a UTC
// 'YYYY-MM-DD HH:MM:SS' (null = date unknown); `inferred` = a proxy, shown "~".
export interface StageHistoryEntry {
  status: KanbanStatus
  at: string | null
  inferred: boolean
}

// The candidate's answer on a closed card — Rejected / Ghosted / I Withdrew:
// how far the process got and, optionally, what happened. Null until answered, and
// while the card is live. `at` = when it was answered (UTC).
export interface CardOutcome {
  stage: KanbanStatus
  what_happened: string | null
  at: string | null
}

// Ledger-derived fields on every /api/applications row (2026-10-03).
export interface ApplicationOutcome {
  highest_stage: KanbanStatus | null   // furthest stage ever reached
  reached_at: string | null
  reached_inferred: boolean
  history: StageHistoryEntry[]
  outcome?: CardOutcome | null         // optional: an API from before 2026-10-28
  sent_cv: SentCvAnswer | null
  has_prep: boolean                    // a prep sheet exists (kept after rejection)
}

// Flat row shape actually returned by /api/applications (a JOIN of
// applications + scrapes) — the old { id, job: Job } shape matched nothing.
export interface Application {
  app_id: number
  job_id: number
  status: KanbanStatus
  notes: string | null
  date_applied: string | null
  updated_at: string | null
  title: string
  company: string
  link: string
  location: string
  cv_path: string | null
  cl_path: string | null
  cv_drive_id: string | null
  cl_drive_id: string | null
  match_level: MatchLevel
  confidence: number | null
  keyword_category: string | null
  cv_quality_flag: string | null
  // Cards are never hidden; a 'gone' card shows one line saying so.
  listing_status?: ListingStatus | null
  listing_gone_since?: string | null
}

// "Reached interview (all time)" on /api/stats/overview.
// _total = confirmed: the card sits at Screening Call or later now, or a
// recorded move got it there. _unconfirmed = inferred from an existing prep
// sheet and not yet confirmed by the user (shown as "+M more to confirm").
// Optional: an older API sends neither / only _total.
export interface StatsReachedInterview {
  reached_interview_total?: number
  reached_interview_unconfirmed?: number
}

// "How processes ended" on /api/stats/overview: the closed
// cards (Rejected / Ghosted / I Withdrew); withdrew = the ones the candidate
// ended. ended_after = interview processes (Screening Call or later,
// confirmed) by the furthest stage reached; to_confirm = an interview stage
// inferred from a prep sheet and not yet confirmed; notes = the "what happened" lines.
export interface StatsEndings {
  closed: number
  answered: number
  withdrew?: number
  ended_after: Partial<Record<KanbanStatus, number>>
  to_confirm: number
  notes: {
    app_id: number
    company: string | null
    title: string | null
    status: KanbanStatus
    stage: KanbanStatus
    what_happened: string
    at: string | null
  }[]
}

export interface CascadeStats {
  stage1: { total: number; passed: number; filtered: number; tokens: number; cost: number }
  stage2: { total: number; high: number; potential: number; rejected: number; tokens: number; cost: number }
  stage3: { total: number; upgraded: number; confirmed: number; downgraded: number; tokens: number; cost: number }
}

export interface SearchStatus {
  active: boolean
  paused: boolean
  progress: number
  status_text: string
  session_id: string | null
}

export type RejectionReason =
  | 'too_junior' | 'social_media' | 'wrong_industry' | 'agency_vendor'
  | 'wrong_location' | 'bad_company_fit' | 'already_applied' | 'other'

export interface CompanyIntel {
  company_name: string
  company_profile: string | null
  salary_intel: string | null
  hiring_intel: string | null
  // "Is this company really hiring?" (markdown): our scanner's own listing
  // history first, then dated, sourced public facts. Optional — an API from
  // before 2026-10-03 does not send it.
  hiring_health?: string | null
  salary_recommendation: string | null
  careers_url: string | null
  last_checked: string | null
}

/** One job's outcome within a background batch generation. */
export interface BatchResult {
  job_id: number
  cv: string | null      // a file path, 'exists', 'failed', 'skipped: …' or 'error: …'
  intel: string | null   // 'generated', 'cached' or 'error: …'
  error: string | null
}

/** GET /api/batch-generate/status */
export interface BatchStatus {
  running: boolean
  total: number
  done: number
  current: number | null   // job_id being worked on
  results: BatchResult[]
  error: string | null
  started_at: string | null
  finished_at: string | null
}

/** POST /api/jobs/{id}/generate-docs. 'in_progress': that job's documents
 *  were already being generated (a batch, or a job added by hand). */
export interface GenerateDocsResult {
  status: 'completed' | 'partial' | 'failed' | 'in_progress'
  job_id: number
  cv_path?: string | null
  cl_path?: string | null
  cv_quality_flag?: string | null
  error?: string
}

/** GET /api/generating: the jobs whose documents are being generated now. */
export interface GeneratingJobs {
  job_ids: number[]
}
