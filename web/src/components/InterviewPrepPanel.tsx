'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '@/lib/api'
import { Button } from '@/components/ui/button'
import {
  AlertCircle,
  GraduationCap,
  Loader2,
  RefreshCw,
  Sparkles,
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface PrepResponse {
  app_id: number
  prep_status: 'pending' | 'ready' | 'failed' | null
  prep_path: string | null
  prep_generated_at: string | null
  content: string | null
}

const markdownComponents = {
  a: ({ href, children, ...props }: any) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-primary underline hover:text-primary/80"
      {...props}
    >
      {children}
    </a>
  ),
  p: ({ children, ...props }: any) => (
    <p className="mb-2 last:mb-0" {...props}>
      {children}
    </p>
  ),
  h1: ({ children, ...props }: any) => (
    <h2 className="mt-3 mb-1 text-lg font-bold text-foreground" {...props}>
      {children}
    </h2>
  ),
  h2: ({ children, ...props }: any) => (
    <h3 className="mt-3 mb-1 text-base font-semibold text-foreground" {...props}>
      {children}
    </h3>
  ),
  h3: ({ children, ...props }: any) => (
    <h4 className="mt-2 mb-1 text-sm font-semibold text-foreground" {...props}>
      {children}
    </h4>
  ),
  ul: ({ children, ...props }: any) => (
    <ul className="ml-4 list-disc space-y-1" {...props}>
      {children}
    </ul>
  ),
  ol: ({ children, ...props }: any) => (
    <ol className="ml-4 list-decimal space-y-1" {...props}>
      {children}
    </ol>
  ),
  code: ({ children, ...props }: any) => (
    <code className="rounded bg-muted px-1 py-0.5 text-xs" {...props}>
      {children}
    </code>
  ),
}

const POLL_MS = 15000

/**
 * Interview-prep sheet for an application: likely questions, the candidate's strongest
 * STAR stories mapped to them, a company angle, and smart questions to ask.
 * Auto-generated when the card moves to an interview stage (Screening Call /
 * Interview / Final Interview / Offer); regenerable on demand.
 *
 * readOnly: the card is no longer in an interview stage (e.g. Rejected or
 * Ghosted after interviews) — the sheet stays readable, but nothing here can
 * start a new generation (no Regenerate / Prep me).
 */
export function InterviewPrepPanel({
  appId,
  readOnly = false,
}: {
  appId: number
  readOnly?: boolean
}) {
  const [prep, setPrep] = useState<PrepResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [spawning, setSpawning] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchPrep = useCallback(async () => {
    try {
      const data = await api.applications.prep(appId)
      setPrep(data)
      return data
    } catch {
      return null
    } finally {
      setLoading(false)
    }
  }, [appId])

  // Initial load + poll while a generation is pending.
  useEffect(() => {
    fetchPrep()
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [fetchPrep])

  useEffect(() => {
    if (prep?.prep_status === 'pending' && !pollRef.current) {
      pollRef.current = setInterval(async () => {
        const data = await fetchPrep()
        if (data && data.prep_status !== 'pending' && pollRef.current) {
          clearInterval(pollRef.current)
          pollRef.current = null
        }
      }, POLL_MS)
    }
  }, [prep?.prep_status, fetchPrep])

  const generate = useCallback(async () => {
    setSpawning(true)
    try {
      await api.applications.generatePrep(appId)
      await fetchPrep()
    } finally {
      setSpawning(false)
    }
  }, [appId, fetchPrep])

  const status = prep?.prep_status ?? null

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-sm font-medium uppercase tracking-wider text-muted-foreground">
          <GraduationCap className="size-3.5" />
          Interview Prep
          {status === 'ready' && (
            <span className="rounded bg-indigo-500/15 px-1.5 py-0.5 text-[10px] font-semibold normal-case text-indigo-400">
              ready
            </span>
          )}
        </span>
        {!readOnly && (status === 'ready' || status === 'failed') && (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1 text-xs"
            onClick={generate}
            disabled={spawning}
            title="Regenerate interview prep"
          >
            {spawning ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <RefreshCw className="size-3" />
            )}
            Regenerate
          </Button>
        )}
      </div>

      {loading ? (
        <div className="flex items-center gap-2 py-3 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Loading…
        </div>
      ) : status === 'pending' ? (
        <div className="flex items-center gap-2 rounded-md border border-border/50 bg-muted/30 px-3 py-3 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin text-primary" />
          Building your prep sheet… (a minute or two; refreshes automatically)
        </div>
      ) : status === 'failed' ? (
        <div className="flex items-center gap-2 rounded-md border border-red-500/30 bg-red-500/5 px-3 py-3 text-sm text-red-400">
          <AlertCircle className="size-4" />
          {readOnly
            ? 'The prep sheet could not be generated.'
            : 'Generation failed — hit Regenerate to retry.'}
        </div>
      ) : status === 'ready' && prep?.content ? (
        <div
          className={cn(
            'max-h-[480px] overflow-y-auto rounded-md border border-border/50',
            'bg-muted/20 px-4 py-3 text-sm leading-relaxed'
          )}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {prep.content}
          </ReactMarkdown>
        </div>
      ) : readOnly ? (
        <div className="rounded-md border border-dashed border-border/60 px-3 py-3 text-sm text-muted-foreground">
          No prep sheet to show.
        </div>
      ) : (
        <div className="flex items-center justify-between rounded-md border border-dashed border-border/60 px-3 py-3">
          <span className="text-sm text-muted-foreground">
            No prep yet — generated automatically when the card moves to an
            interview stage. Prep me now:
          </span>
          <Button
            variant="outline"
            size="sm"
            className="h-7 gap-1 text-xs"
            onClick={generate}
            disabled={spawning}
          >
            {spawning ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Sparkles className="size-3" />
            )}
            Prep me
          </Button>
        </div>
      )}
    </div>
  )
}
