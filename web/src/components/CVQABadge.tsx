import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { CheckCircle, AlertTriangle, XCircle } from 'lucide-react'

interface CVQABadgeProps {
  flag: string | null
  className?: string
}

// The flag grades the CV only — the cover letter has its own check. Every
// label says "CV QA" so a green badge in the Documents panel never reads as
// "both documents passed".
const flagConfig: Record<
  string,
  { label: string; icon: typeof CheckCircle; className: string }
> = {
  PASSED: {
    label: 'CV QA: Passed',
    icon: CheckCircle,
    className: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/25',
  },
  REVIEW_RECOMMENDED: {
    label: 'CV QA: Review',
    icon: AlertTriangle,
    className: 'bg-amber-500/15 text-amber-400 border-amber-500/25',
  },
  HALLUCINATION: {
    label: 'CV QA: Hallucination',
    icon: XCircle,
    className: 'bg-red-500/15 text-red-400 border-red-500/25',
  },
  HALLUCINATION_UNRESOLVED: {
    label: 'CV QA: Needs review',
    icon: AlertTriangle,
    className: 'bg-amber-500/15 text-amber-400 border-amber-500/25',
  },
}

const CV_ONLY_TITLE = 'Quality check of the CV only — the cover letter is checked separately'

export function CVQABadge({ flag, className }: CVQABadgeProps) {
  if (!flag) return null

  const config = flagConfig[flag]
  if (!config) {
    return (
      <Badge variant="outline" className={cn('text-sm', className)} title={CV_ONLY_TITLE}>
        CV QA: {flag}
      </Badge>
    )
  }

  const Icon = config.icon

  return (
    <Badge
      variant="outline"
      title={CV_ONLY_TITLE}
      className={cn(
        'gap-1 text-xs font-semibold',
        config.className,
        className
      )}
    >
      <Icon className="size-3" />
      {config.label}
    </Badge>
  )
}
