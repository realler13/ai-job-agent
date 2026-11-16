'use client'

import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { Users } from 'lucide-react'

interface WarmMatch {
  owner: string
  name: string
  url: string
  company: string
  position: string
  matched_via: string
  score: number
}

interface WarmResponse {
  matches: WarmMatch[]
  total_matches?: number
  possible?: WarmMatch[]
}

/**
 * Warm paths for a company: people from the imported networks
 * (official LinkedIn CSV exports only). A match's owner is either
 * 'candidate' (the user's own export) or any other label, e.g. a friend or
 * mentor who offered introductions. Shows the top relevance-ranked
 * connections; a referral ask beats any cold outreach.
 */
export function WarmPathsBadge({ company }: { company: string }) {
  const [data, setData] = useState<WarmResponse | null>(null)

  useEffect(() => {
    let alive = true
    api.warmPaths
      .forCompany(company)
      .then((d) => {
        if (alive) setData(d)
      })
      .catch(() => {
        if (alive) setData({ matches: [] })
      })
    return () => {
      alive = false
    }
  }, [company])

  const matches = data?.matches ?? []
  const total = data?.total_matches ?? matches.length
  const possibleCount = data?.possible?.length ?? 0
  if (matches.length === 0 && possibleCount === 0) return null

  return (
    <div className="rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2">
      <div className="flex items-center gap-1.5 text-sm font-medium text-amber-500">
        <Users className="size-3.5" />
        {total} warm path{total !== 1 ? 's' : ''} — referral ask beats cold
        outreach
      </div>
      <ul className="mt-1 space-y-0.5 text-sm text-muted-foreground">
        {matches.slice(0, 4).map((m) => (
          <li key={`${m.owner}-${m.url}`} className="truncate">
            {m.owner === 'candidate' ? 'You know' : 'Someone you know knows'}{' '}
            <a
              href={m.url}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-foreground underline-offset-2 hover:underline"
            >
              {m.name}
            </a>
            {m.position ? ` — ${m.position}` : ''}
            {m.owner !== 'candidate' && (
              <span className="ml-1 font-medium text-amber-500">
                → a contact&apos;s network — ask them for the intro
              </span>
            )}
            {m.owner === 'candidate' && (
              <span className="ml-1 font-medium text-emerald-600">
                → your connection — message them directly
              </span>
            )}
          </li>
        ))}
        {total > 4 && <li className="text-xs">…and {total - 4} more</li>}
        {possibleCount > 0 && (
          <li className="text-xs italic">
            {possibleCount} fuzzy match{possibleCount !== 1 ? 'es' : ''} need a
            human look (similar company names — not auto-confirmed)
          </li>
        )}
      </ul>
    </div>
  )
}
