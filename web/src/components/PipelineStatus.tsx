'use client'

import { useSearchStore } from '@/store/search'
import { cn } from '@/lib/utils'
import { Loader2, Pause } from 'lucide-react'

export function PipelineStatus() {
  const { active, paused, progress, statusText } = useSearchStore()

  if (!active) return null

  const pct = Math.round(progress * 100)

  return (
    <div className="fixed top-0 right-0 left-0 z-50">
      {/* Progress track */}
      <div className="h-1 w-full bg-muted">
        <div
          className={cn(
            'h-full transition-all duration-500 ease-out',
            paused ? 'bg-amber-500' : 'bg-emerald-500'
          )}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Status bar */}
      <div className="flex items-center gap-2 border-b border-border/50 bg-card/80 px-4 py-1.5 text-sm backdrop-blur-sm">
        {paused ? (
          <Pause className="size-3.5 text-amber-500" />
        ) : (
          <Loader2 className="size-3.5 animate-spin text-emerald-500" />
        )}
        <span className="font-medium text-foreground">
          {paused ? 'Paused' : 'Searching'}
        </span>
        <span className="text-muted-foreground">{statusText}</span>
        <span className="ml-auto tabular-nums text-muted-foreground">
          {pct}%
        </span>
      </div>
    </div>
  )
}
