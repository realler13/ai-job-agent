'use client'

import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { PostingCheck, PostingFact, PostingFactSource } from '@/lib/types'
import { Button } from '@/components/ui/button'
import { ExternalLink, FileSearch, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

// "Is this posting real?". Facts about the posting, each with
// where it came from — never a score: nothing here changes the job's tier or
// its place in the feed. The read label is a plain summary of the facts.

const SOURCE_LABEL: Record<PostingFactSource, string> = {
  company_board: 'Company job board',
  our_scanner: 'Our scanner',
  posting_text: 'The posting',
  public_source: 'Public source',
  your_history: 'Your history',
}

// Links come from public research text: only plain web links are clickable.
const isWebLink = (url?: string | null): url is string =>
  !!url && /^https?:\/\//i.test(url)

const QUALITY_STATUS_NOTE: Record<string, string> = {
  no_text: 'The posting text we have is too short to read.',
  failed: "Couldn't read the posting just now — try again later.",
}

function SourceTag({ fact }: { fact: PostingFact }) {
  const label = SOURCE_LABEL[fact.source] ?? 'Source'
  const cls = 'shrink-0 rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground'
  // A research source, or the new link of a role re-posted on its board.
  if ((fact.source === 'public_source' || fact.source === 'company_board') && isWebLink(fact.url)) {
    return (
      <a
        href={fact.url}
        target="_blank"
        rel="noopener noreferrer"
        className={cn(cls, 'inline-flex items-center gap-1 hover:text-foreground')}
        onClick={(e) => e.stopPropagation()}
      >
        {label}
        <ExternalLink className="size-3" />
      </a>
    )
  }
  return <span className={cls}>{label}</span>
}

export function PostingCheckPanel({
  jobId,
  enabled,
}: {
  jobId: number
  // Fetch only when the panel is actually visible (the Review page mounts
  // every row's expanded content, hidden).
  enabled: boolean
}) {
  const queryClient = useQueryClient()
  const [note, setNote] = useState<string | null>(null)

  const { data, isLoading, isError } = useQuery<PostingCheck>({
    queryKey: ['posting-check', jobId],
    queryFn: () => api.postingCheck.get(jobId),
    enabled: enabled && !!jobId,
    staleTime: 5 * 60_000,
    retry: false,
  })

  const readMutation = useMutation({
    mutationFn: () => api.postingCheck.readPosting(jobId),
    onSuccess: (res) => {
      setNote(QUALITY_STATUS_NOTE[res.status] ?? null)
      queryClient.invalidateQueries({ queryKey: ['posting-check', jobId] })
    },
    onError: () => setNote(QUALITY_STATUS_NOTE.failed),
  })

  return (
    <div>
      <div className="mb-1.5 flex flex-wrap items-baseline gap-x-2 text-sm font-medium uppercase tracking-wider text-muted-foreground">
        Posting check
        {data?.label && (
          <span className="text-xs font-normal normal-case tracking-normal text-muted-foreground/80">
            {data.label}
          </span>
        )}
      </div>

      {!enabled && !data ? null : isLoading ? (
        <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin" />
          Checking…
        </p>
      ) : isError || !data ? (
        <p className="text-sm text-muted-foreground">
          The posting check isn&apos;t available right now.
        </p>
      ) : (
        <div className="space-y-2">
          {data.facts.length > 0 ? (
            <ul className="space-y-1.5">
              {data.facts.map((f, i) => (
                <li key={`${f.key}-${i}`} className="flex items-start gap-2 text-sm leading-snug">
                  <div className="min-w-0 flex-1">
                    <span
                      className={
                        f.caution
                          ? 'text-amber-700 dark:text-amber-400'
                          : 'text-foreground'
                      }
                    >
                      {f.text}
                    </span>
                    {(f.quotes ?? []).map((q, j) => (
                      <span
                        key={j}
                        className="mt-0.5 block text-xs italic text-muted-foreground"
                      >
                        &ldquo;{q}&rdquo;
                      </span>
                    ))}
                  </div>
                  <SourceTag fact={f} />
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">No facts to show yet.</p>
          )}

          {!data.quality_available && (
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <Button
                variant="outline"
                size="sm"
                className="gap-1.5"
                disabled={readMutation.isPending}
                onClick={(e) => {
                  e.stopPropagation()
                  setNote(null)
                  readMutation.mutate()
                }}
                title="Check how specific the posting is: tools, team, first-year work, pay — each quoted from the posting"
              >
                {readMutation.isPending ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <FileSearch className="size-3.5" />
                )}
                {readMutation.isPending ? 'Reading…' : 'Read the posting'}
              </Button>
              {note && <span className="text-xs text-muted-foreground">{note}</span>}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
