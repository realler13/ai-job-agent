'use client'

import { useCallback, useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '@/lib/api'
import type { CompanyIntel } from '@/lib/types'
import { Button } from '@/components/ui/button'
import {
  Activity,
  AlertCircle,
  Building2,
  CheckCircle2,
  DollarSign,
  ExternalLink,
  Loader2,
  Search,
  Target,
  Users,
  Sparkles,
} from 'lucide-react'
import { cn } from '@/lib/utils'

/** Strip Perplexity citation brackets like [1], [2][3], etc. */
function stripCitations(text: string): string {
  return text.replace(/\[\d+\](\[\d+\])*/g, '')
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
  h3: ({ children, ...props }: any) => (
    <h3
      className="mt-3 mb-1 text-base font-semibold text-foreground"
      {...props}
    >
      {children}
    </h3>
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
  li: ({ children, ...props }: any) => (
    <li className="text-base" {...props}>
      {children}
    </li>
  ),
  strong: ({ children, ...props }: any) => (
    <strong className="font-semibold text-foreground" {...props}>
      {children}
    </strong>
  ),
  table: ({ children, ...props }: any) => (
    <div className="my-2 overflow-x-auto rounded-md border border-border">
      <table className="w-full text-sm" {...props}>
        {children}
      </table>
    </div>
  ),
  thead: ({ children, ...props }: any) => (
    <thead className="border-b border-border bg-muted/50" {...props}>
      {children}
    </thead>
  ),
  tbody: ({ children, ...props }: any) => (
    <tbody className="divide-y divide-border" {...props}>
      {children}
    </tbody>
  ),
  tr: ({ children, ...props }: any) => (
    <tr className="hover:bg-muted/30" {...props}>
      {children}
    </tr>
  ),
  th: ({ children, ...props }: any) => (
    <th
      className="px-3 py-2 text-left text-sm font-semibold text-foreground"
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ children, ...props }: any) => (
    <td className="px-3 py-2 text-sm text-foreground/80" {...props}>
      {children}
    </td>
  ),
}

interface CompanyIntelPanelProps {
  company: string
  jobTitle: string
  matchLevel?: string | null
}

export function CompanyIntelPanel({
  company,
  jobTitle,
  matchLevel,
}: CompanyIntelPanelProps) {
  const [intel, setIntel] = useState<CompanyIntel | null>(null)
  const [loading, setLoading] = useState(false)
  const [enriching, setEnriching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [justLoaded, setJustLoaded] = useState(false)

  // Auto-fetch cached intel when company changes
  useEffect(() => {
    if (!company) return
    setLoading(true)
    setError(null)
    setJustLoaded(false)
    api.companyIntel
      .get(company, jobTitle)
      .then((data) => {
        setIntel(data)
      })
      .catch(() => {
        // No cached data is fine — user can trigger enrichment
        setIntel(null)
      })
      .finally(() => setLoading(false))
  }, [company, jobTitle])

  const handleEnrich = useCallback(async () => {
    setEnriching(true)
    setError(null)
    setJustLoaded(false)
    console.log(`[CompanyIntel] Enriching company: ${company}, job: ${jobTitle}`)
    try {
      const data = await api.companyIntel.enrich(company, jobTitle)
      console.log('[CompanyIntel] Enrichment result:', data)
      setIntel(data)
      setJustLoaded(true)
      // Clear the "just loaded" highlight after 5 seconds
      setTimeout(() => setJustLoaded(false), 5000)
    } catch (e) {
      const message =
        e instanceof Error ? e.message : 'Failed to fetch company intel'
      console.error('[CompanyIntel] Enrichment failed:', message, e)
      setError(message)
    } finally {
      setEnriching(false)
    }
  }, [company, jobTitle])

  const hasAnyIntel =
    intel && (intel.company_profile || intel.salary_intel || intel.hiring_intel || intel.salary_recommendation || intel.careers_url)
  // Hiring health can exist before any research — its first lines come free
  // from our own scanner history — so it shows without flipping the panel
  // into its "researched" state (the prompt and button label stay as they were).
  const hasHealth = Boolean(intel?.hiring_health)
  const isHighMatch = matchLevel === 'HIGH_MATCH'

  return (
    <div className="rounded-lg border border-border bg-muted/30 p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Search className="size-4 text-muted-foreground" />
          <span className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
            Company Intel
          </span>
        </div>
        {intel?.last_checked && (
          <span className="text-xs text-muted-foreground">
            Updated {intel.last_checked}
          </span>
        )}
      </div>

      {/* HIGH_MATCH prompt */}
      {isHighMatch && !hasAnyIntel && !loading && !enriching && (
        <div className="mb-3 flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-2.5">
          <Sparkles className="mt-0.5 size-4 shrink-0 text-amber-500" />
          <p className="text-sm text-amber-700 dark:text-amber-400">
            This is a high-confidence match. Research the company before applying
            to tailor your approach.
          </p>
        </div>
      )}

      {/* Enriching state — prominent banner */}
      {enriching && (
        <div className="mb-3 flex items-center gap-2.5 rounded-md border border-blue-500/30 bg-blue-500/10 p-3">
          <Loader2 className="size-5 animate-spin text-blue-500" />
          <span className="text-base font-medium text-blue-700 dark:text-blue-400">
            Researching {company}...
          </span>
        </div>
      )}

      {loading && (
        <div className="flex items-center gap-2 py-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          Loading cached intel...
        </div>
      )}

      {!loading && intel && (hasAnyIntel || hasHealth) && (
        <div className="flex flex-col gap-3">
          {/* Success indicator after fresh enrichment */}
          {justLoaded && (
            <div className="flex items-center gap-2 rounded-md border border-green-500/30 bg-green-500/10 p-2">
              <CheckCircle2 className="size-4 text-green-600 dark:text-green-400" />
              <span className="text-sm font-medium text-green-700 dark:text-green-400">
                Company intel loaded successfully
              </span>
            </div>
          )}

          {/* Company Profile */}
          {intel.company_profile && (
            <div className="rounded-md border border-border/50 bg-background/60 p-3">
              <div className="mb-1.5 flex items-center gap-1.5">
                <Building2 className="size-4 text-blue-500" />
                <span className="text-base font-semibold text-foreground">
                  Company Profile
                </span>
              </div>
              <div className="text-base leading-relaxed text-foreground/80">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                  {stripCitations(intel.company_profile)}
                </ReactMarkdown>
              </div>
            </div>
          )}

          {/* Hiring Health — our scanner's history first, then dated, sourced
              public facts. The server already turned citations into links. */}
          {intel.hiring_health && (
            <div className="rounded-md border border-border/50 bg-background/60 p-3">
              <div className="mb-1.5 flex items-center gap-1.5">
                <Activity className="size-4 text-rose-500" />
                <span className="text-base font-semibold text-foreground">
                  Hiring Health
                </span>
              </div>
              <div className="text-base leading-relaxed text-foreground/80">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                  {intel.hiring_health}
                </ReactMarkdown>
              </div>
            </div>
          )}

          {/* Salary & Compensation */}
          {intel.salary_intel && (
            <div className="rounded-md border border-border/50 bg-background/60 p-3">
              <div className="mb-1.5 flex items-center gap-1.5">
                <DollarSign className="size-4 text-green-500" />
                <span className="text-base font-semibold text-foreground">
                  Salary &amp; Compensation
                </span>
              </div>
              <div className="text-base leading-relaxed text-foreground/80">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                  {stripCitations(intel.salary_intel)}
                </ReactMarkdown>
              </div>
            </div>
          )}

          {/* Hiring Intel */}
          {intel.hiring_intel && (
            <div className="rounded-md border border-border/50 bg-background/60 p-3">
              <div className="mb-1.5 flex items-center gap-1.5">
                <Users className="size-4 text-purple-500" />
                <span className="text-base font-semibold text-foreground">
                  Hiring Intel
                </span>
              </div>
              <div className="text-base leading-relaxed text-foreground/80">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                  {stripCitations(intel.hiring_intel)}
                </ReactMarkdown>
              </div>
            </div>
          )}

          {/* Salary Strategy (Opus recommendation) */}
          {intel.salary_recommendation && (
            <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3">
              <div className="mb-1.5 flex items-center gap-1.5">
                <Target className="size-4 text-amber-600" />
                <span className="text-base font-semibold text-foreground">
                  Salary Strategy
                </span>
                <span className="ml-auto rounded-full bg-amber-500/15 px-2 py-0.5 text-xs font-medium text-amber-700 dark:text-amber-400">
                  Opus Advisor
                </span>
              </div>
              <div className="text-base leading-relaxed text-foreground/80">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                  {intel.salary_recommendation}
                </ReactMarkdown>
              </div>
            </div>
          )}

          {/* Careers URL */}
          {intel.careers_url && (
            <a
              href={intel.careers_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex w-fit items-center gap-1.5 text-sm font-medium text-primary hover:underline"
            >
              <ExternalLink className="size-3.5" />
              Careers Page
            </a>
          )}
        </div>
      )}

      {/* Error display — prominent red banner */}
      {error && (
        <div className="mb-2 flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 p-2.5">
          <AlertCircle className="mt-0.5 size-4 shrink-0 text-red-500" />
          <p className="text-sm font-medium text-red-700 dark:text-red-400">
            {error}
          </p>
        </div>
      )}

      {/* Enrich button */}
      {!loading && (
        <Button
          variant="outline"
          size="sm"
          className={cn('mt-2 gap-1.5', (hasAnyIntel || hasHealth) && 'mt-3')}
          onClick={handleEnrich}
          disabled={enriching}
        >
          {enriching ? (
            <>
              <Loader2 className="size-3.5 animate-spin" />
              Researching {company}...
            </>
          ) : hasAnyIntel ? (
            <>
              <Search className="size-3.5" />
              Refresh Intel
            </>
          ) : (
            <>
              <Search className="size-3.5" />
              Research Company
            </>
          )}
        </Button>
      )}
    </div>
  )
}
