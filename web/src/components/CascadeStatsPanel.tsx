'use client'

import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '@/components/ui/card'
import type { CascadeStats } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Filter, Target, Award } from 'lucide-react'

interface CascadeStatsPanelProps {
  stats: CascadeStats
  className?: string
}

function StatRow({
  label,
  value,
  subtext,
  icon: Icon,
  color,
}: {
  label: string
  value: number
  subtext?: string
  icon: typeof Filter
  color: string
}) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={cn(
          'flex size-8 shrink-0 items-center justify-center rounded-lg',
          color
        )}
      >
        <Icon className="size-4" />
      </div>
      <div className="flex-1">
        <div className="text-sm font-medium text-foreground">{label}</div>
        {subtext && (
          <div className="text-sm text-muted-foreground">{subtext}</div>
        )}
      </div>
      <div className="text-right">
        <div className="text-sm font-semibold tabular-nums text-foreground">
          {value}
        </div>
      </div>
    </div>
  )
}

function formatCost(cost: number): string {
  if (cost < 0.01) return '<$0.01'
  return `$${cost.toFixed(2)}`
}

export function CascadeStatsPanel({
  stats,
  className,
}: CascadeStatsPanelProps) {
  const totalCost =
    (stats.stage1?.cost ?? 0) +
    (stats.stage2?.cost ?? 0) +
    (stats.stage3?.cost ?? 0)

  return (
    <Card className={cn('', className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Filter className="size-4" />
          Cascade Pipeline
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          {/* Stage 1 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
                Stage 1 — Pre-filter
              </span>
              <span className="text-sm text-muted-foreground">
                {formatCost(stats.stage1?.cost ?? 0)}
              </span>
            </div>
            <StatRow
              label={`${stats.stage1?.passed ?? 0} passed`}
              value={stats.stage1?.total ?? 0}
              subtext={`${stats.stage1?.filtered ?? 0} filtered out`}
              icon={Filter}
              color="bg-blue-500/15 text-blue-400"
            />
          </div>

          <div className="h-px bg-border" />

          {/* Stage 2 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
                Stage 2 — Grading
              </span>
              <span className="text-sm text-muted-foreground">
                {formatCost(stats.stage2?.cost ?? 0)}
              </span>
            </div>
            <StatRow
              label={`${stats.stage2?.high ?? 0} high / ${stats.stage2?.potential ?? 0} potential`}
              value={stats.stage2?.total ?? 0}
              subtext={`${stats.stage2?.rejected ?? 0} rejected`}
              icon={Target}
              color="bg-amber-500/15 text-amber-400"
            />
          </div>

          <div className="h-px bg-border" />

          {/* Stage 3 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
                Stage 3 — Judge
              </span>
              <span className="text-sm text-muted-foreground">
                {formatCost(stats.stage3?.cost ?? 0)}
              </span>
            </div>
            <StatRow
              label={`${stats.stage3?.upgraded ?? 0} upgraded / ${stats.stage3?.confirmed ?? 0} confirmed`}
              value={stats.stage3?.total ?? 0}
              subtext={`${stats.stage3?.downgraded ?? 0} downgraded`}
              icon={Award}
              color="bg-emerald-500/15 text-emerald-400"
            />
          </div>

          <div className="h-px bg-border" />

          {/* Total */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-foreground">
              Total cost
            </span>
            <span className="text-sm font-semibold tabular-nums text-foreground">
              {formatCost(totalCost)}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
