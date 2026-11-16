'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '@/lib/api'
import { Button } from '@/components/ui/button'
import {
  AlertCircle,
  Loader2,
  RefreshCw,
  Send,
  Sparkles,
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface IntelResponse {
  app_id: number
  intel_status: 'pending' | 'ready' | 'failed' | null
  intel_path: string | null
  intel_generated_at: string | null
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
  table: ({ children, ...props }: any) => (
    <div className="my-2 overflow-x-auto">
      <table className="w-full border-collapse text-xs" {...props}>
        {children}
      </table>
    </div>
  ),
  th: ({ children, ...props }: any) => (
    <th className="border border-border/60 bg-muted/50 px-2 py-1 text-left font-semibold" {...props}>
      {children}
    </th>
  ),
  td: ({ children, ...props }: any) => (
    <td className="border border-border/60 px-2 py-1 align-top" {...props}>
      {children}
    </td>
  ),
  code: ({ children, ...props }: any) => (
    <code className="rounded bg-muted px-1 py-0.5 text-xs" {...props}>
      {children}
    </code>
  ),
}

const POLL_MS = 15000

/**
 * Outreach playbook for an application: company pulse, whom to contact,
 * follow-up cadence, message drafts. Auto-generated when the application
 * moves to "Applied"; regenerable on demand.
 */
export function OutreachIntelPanel({ appId }: { appId: number }) {
  const [intel, setIntel] = useState<IntelResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [spawning, setSpawning] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchIntel = useCallback(async () => {
    try {
      const data = await api.applications.intel(appId)
      setIntel(data)
      return data
    } catch {
      return null
    } finally {
      setLoading(false)
    }
  }, [appId])

  // Initial load + poll while a generation is pending.
  useEffect(() => {
    fetchIntel()
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [fetchIntel])

  useEffect(() => {
    if (intel?.intel_status === 'pending' && !pollRef.current) {
      pollRef.current = setInterval(async () => {
        const data = await fetchIntel()
        if (data && data.intel_status !== 'pending' && pollRef.current) {
          clearInterval(pollRef.current)
          pollRef.current = null
        }
      }, POLL_MS)
    }
  }, [intel?.intel_status, fetchIntel])

  const generate = useCallback(async () => {
    setSpawning(true)
    try {
      await api.applications.generateIntel(appId)
      await fetchIntel()
    } finally {
      setSpawning(false)
    }
  }, [appId, fetchIntel])

  const status = intel?.intel_status ?? null

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-sm font-medium uppercase tracking-wider text-muted-foreground">
          <Send className="size-3.5" />
          Outreach Intel
          {status === 'ready' && (
            <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-semibold normal-case text-emerald-500">
              ready
            </span>
          )}
        </span>
        {(status === 'ready' || status === 'failed') && (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1 text-xs"
            onClick={generate}
            disabled={spawning}
            title="Regenerate outreach intel"
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
          Researching company pulse + contacts… (a few minutes; refreshes
          automatically)
        </div>
      ) : status === 'failed' ? (
        <div className="flex items-center gap-2 rounded-md border border-red-500/30 bg-red-500/5 px-3 py-3 text-sm text-red-400">
          <AlertCircle className="size-4" />
          Generation failed — hit Regenerate to retry.
        </div>
      ) : status === 'ready' && intel?.content ? (
        <div
          className={cn(
            'max-h-[480px] overflow-y-auto rounded-md border border-border/50',
            'bg-muted/20 px-4 py-3 text-sm leading-relaxed'
          )}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {intel.content}
          </ReactMarkdown>
        </div>
      ) : (
        <div className="flex items-center justify-between rounded-md border border-dashed border-border/60 px-3 py-3">
          <span className="text-sm text-muted-foreground">
            No outreach intel yet — generated automatically when the card moves
            to Applied.
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
            Generate now
          </Button>
        </div>
      )}
    </div>
  )
}
