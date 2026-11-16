import { cn } from '@/lib/utils'

interface ConfidenceBarProps {
  value: number | null
  className?: string
  showLabel?: boolean
}

export function ConfidenceBar({
  value,
  className,
  showLabel = true,
}: ConfidenceBarProps) {
  if (value == null) {
    return (
      <span className="text-sm text-muted-foreground">--</span>
    )
  }

  const pct = Math.round(value * 100)
  const [barColor, trackColor] =
    value >= 0.8
      ? ['bg-emerald-500', 'bg-emerald-500/15']
      : value >= 0.5
        ? ['bg-amber-500', 'bg-amber-500/15']
        : ['bg-red-500', 'bg-red-500/15']

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div
        className={cn(
          'relative h-1.5 w-16 overflow-hidden rounded-full',
          trackColor
        )}
      >
        <div
          className={cn('absolute inset-y-0 left-0 rounded-full transition-all', barColor)}
          style={{ width: `${pct}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-sm tabular-nums text-muted-foreground">
          {pct}%
        </span>
      )}
    </div>
  )
}
