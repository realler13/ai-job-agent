'use client'

import { Button, buttonVariants } from '@/components/ui/button'
import { CVQABadge } from '@/components/CVQABadge'
import { api } from '@/lib/api'
import type { Job, SentCvAnswer } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Check, FileText, Download, FileSignature, Loader2 } from 'lucide-react'
import { useId, useState } from 'react'

const SENT_CV_OPTIONS: { value: SentCvAnswer; label: string }[] = [
  { value: 'as_is', label: 'Yes, as is' },
  { value: 'edited', label: 'Yes, with my edits' },
  { value: 'own', label: 'No, I rewrote it' },
]

/**
 * Minimal shape that DocumentPanel needs. This allows both a full Job object
 * and the flat jobLike objects that Kanban constructs from FlatApplication rows.
 */
interface DocumentPanelJob {
  id: number
  title: string
  company: string
  cv_path?: string | null
  cl_path?: string | null
  cv_quality_flag?: string | null
  cv_audit_summary?: string | null
  cv_domain_used?: string | null
}

interface DocumentPanelProps {
  job: Job | DocumentPanelJob
  onGenerate?: () => Promise<void>
  /**
   * The server is generating this job's documents right now (a click, a
   * batch, or the background generation a job added by hand starts).
   */
  busy?: boolean
  className?: string
  /**
   * Kanban only: the application behind this card. When an id is passed and
   * the card is Applied or later with a CV, the panel asks "Did you send this
   * CV?". The Review page passes nothing, so the question never shows there.
   */
  applicationId?: number
  applicationStatus?: string
  sentCv?: SentCvAnswer | null
  onSentCvSaved?: () => void
}

export function DocumentPanel({
  job,
  onGenerate,
  busy = false,
  className,
  applicationId,
  applicationStatus,
  sentCv,
  onSentCvSaved,
}: DocumentPanelProps) {
  const [generating, setGenerating] = useState(false)
  const [sentAnswer, setSentAnswer] = useState<SentCvAnswer | null>(sentCv ?? null)
  const [savingSent, setSavingSent] = useState<SentCvAnswer | null>(null)
  const [sentError, setSentError] = useState(false)
  // Ties the "Did you send this CV?" question to its button group, so a
  // screen reader hears the question with each option.
  const sentQuestionId = useId()

  // Follow the board when it refetches (e.g. the answer was changed
  // elsewhere): adjust during render when the prop changes, not in an effect.
  const [boardSentCv, setBoardSentCv] = useState<SentCvAnswer | null>(sentCv ?? null)
  if ((sentCv ?? null) !== boardSentCv) {
    setBoardSentCv(sentCv ?? null)
    setSentAnswer(sentCv ?? null)
  }

  const handleSentCv = async (answer: SentCvAnswer) => {
    if (applicationId == null || savingSent) return
    setSavingSent(answer)
    setSentError(false)
    try {
      await api.applications.setSentCv(applicationId, answer)
      setSentAnswer(answer)
      onSentCvSaved?.()
    } catch {
      setSentError(true)
    } finally {
      setSavingSent(null)
    }
  }

  const hasCv = !!job.cv_path
  const askSentCv =
    applicationId != null &&
    hasCv &&
    !!applicationStatus &&
    applicationStatus !== 'To Apply'
  const hasCl = !!job.cl_path
  const cvQualityFlag = job.cv_quality_flag ?? null
  const cvDomainUsed = ('cv_domain_used' in job ? job.cv_domain_used : null) ?? null
  const cvAuditSummary = ('cv_audit_summary' in job ? job.cv_audit_summary : null) ?? null

  const handleGenerate = async () => {
    if (!onGenerate) return
    setGenerating(true)
    try {
      await onGenerate()
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div
      className={cn(
        'flex flex-col gap-3 rounded-lg border border-border bg-card p-3',
        className
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
          Documents
        </span>
        {cvQualityFlag && <CVQABadge flag={cvQualityFlag} />}
      </div>

      {hasCv || hasCl ? (
        <div className="flex flex-wrap gap-2">
          {hasCv && (
            <a
              href={api.jobs.cvUrl(job.id)}
              target="_blank"
              rel="noopener noreferrer"
              download
              className={cn(buttonVariants({ variant: 'outline', size: 'sm' }), 'gap-1.5 no-underline')}
            >
              <FileText className="size-3.5" />
              CV
              <Download className="size-3" />
            </a>
          )}
          {hasCl && (
            <a
              href={api.jobs.clUrl(job.id)}
              target="_blank"
              rel="noopener noreferrer"
              download
              className={cn(buttonVariants({ variant: 'outline', size: 'sm' }), 'gap-1.5 no-underline')}
            >
              <FileSignature className="size-3.5" />
              Cover Letter
              <Download className="size-3" />
            </a>
          )}
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          <p className="text-sm text-muted-foreground">
            No documents generated yet.
          </p>
          {onGenerate && (
            <Button
              variant="default"
              size="sm"
              className="w-fit gap-1.5"
              onClick={handleGenerate}
              disabled={generating || busy}
            >
              {generating || busy ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <FileText className="size-3.5" />
              )}
              {generating || busy ? 'Generating…' : 'Generate CV & Cover Letter'}
            </Button>
          )}
          {(generating || busy) && (
            <p role="status" className="text-sm text-muted-foreground">
              This usually takes 5–10 minutes. You can leave this page; it keeps going.
            </p>
          )}
        </div>
      )}

      {/* One click, no required step: which CV actually went out. */}
      {askSentCv && (
        <div className="flex flex-col gap-1.5">
          <div className="flex flex-wrap items-baseline gap-x-1.5">
            <span id={sentQuestionId} className="text-sm text-muted-foreground">
              Did you send this CV?
            </span>
            {/* Always mounted so the "saved" text is announced when it appears. */}
            <span role="status" className="text-xs text-muted-foreground/70">
              {sentAnswer && !savingSent ? 'saved — you can change it' : ''}
            </span>
          </div>
          <div
            role="group"
            aria-labelledby={sentQuestionId}
            className="flex flex-wrap gap-1.5"
          >
            {SENT_CV_OPTIONS.map((option) => {
              const chosen = sentAnswer === option.value
              const saving = savingSent === option.value
              return (
                <Button
                  key={option.value}
                  variant={chosen ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => handleSentCv(option.value)}
                  // Only the OTHER options are disabled while one saves: the
                  // clicked button keeps keyboard focus (handleSentCv ignores
                  // a second click while saving).
                  disabled={savingSent !== null && !saving}
                  aria-pressed={chosen}
                  aria-busy={saving}
                >
                  {saving ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    chosen && <Check className="size-3.5" />
                  )}
                  {option.label}
                </Button>
              )
            })}
          </div>
          {sentError && (
            <span role="alert" className="text-xs text-red-400">
              Could not save — please try again.
            </span>
          )}
        </div>
      )}

      {cvDomainUsed && (
        <div className="text-sm text-muted-foreground">
          Domain: <span className="font-medium text-foreground">{cvDomainUsed}</span>
        </div>
      )}

      {cvAuditSummary && !cvAuditSummary.startsWith('{') && (
        <details className="text-sm">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
            QA Audit Details
          </summary>
          <p className="mt-1 whitespace-pre-wrap text-muted-foreground">
            {cvAuditSummary}
          </p>
        </details>
      )}
    </div>
  )
}
