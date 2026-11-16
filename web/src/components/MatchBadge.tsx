import { Badge } from '@/components/ui/badge'
import type { MatchLevel } from '@/lib/types'
import { cn } from '@/lib/utils'

const matchConfig: Record<
  string,
  { label: string; className: string }
> = {
  HIGH_MATCH: {
    label: 'High Match',
    className: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/25',
  },
  POTENTIAL_MATCH: {
    label: 'Potential',
    className: 'bg-amber-500/15 text-amber-400 border-amber-500/25',
  },
  WORTH_A_LOOK: {
    label: 'Worth a Look',
    className: 'bg-sky-500/15 text-sky-400 border-sky-500/25',
  },
  CONFIRMED_POTENTIAL: {
    label: 'Confirmed',
    className: 'bg-amber-500/15 text-amber-300 border-amber-500/25',
  },
  IRRELEVANT: {
    label: 'Irrelevant',
    className: 'bg-slate-500/15 text-slate-400 border-slate-500/25',
  },
  FILTERED_OUT: {
    label: 'Filtered',
    className: 'bg-slate-500/15 text-slate-500 border-slate-500/25',
  },
  DUPLICATE: {
    label: 'Duplicate',
    className: 'bg-slate-500/15 text-slate-500 border-slate-500/25',
  },
  USER_INPUT: {
    label: 'User Input',
    className: 'bg-blue-500/15 text-blue-400 border-blue-500/25',
  },
  DOWNGRADED: {
    label: 'Downgraded',
    className: 'bg-red-500/15 text-red-400 border-red-500/25',
  },
}

interface MatchBadgeProps {
  level: MatchLevel
  className?: string
}

export function MatchBadge({ level, className }: MatchBadgeProps) {
  const config = matchConfig[level] ?? {
    label: level,
    className: 'bg-slate-500/15 text-slate-400 border-slate-500/25',
  }

  return (
    <Badge
      variant="outline"
      className={cn(
        'text-xs font-semibold tracking-wide uppercase',
        config.className,
        className
      )}
    >
      {config.label}
    </Badge>
  )
}
