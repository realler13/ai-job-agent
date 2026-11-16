'use client'

import { useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { Terminal } from 'lucide-react'

interface LogViewerProps {
  lines: string[]
  className?: string
  maxHeight?: string
}

function lineColor(line: string): string {
  const lower = line.toLowerCase()
  if (lower.includes('error') || lower.includes('exception')) return 'text-red-400'
  if (lower.includes('warning') || lower.includes('warn')) return 'text-amber-400'
  if (lower.includes('success') || lower.includes('passed')) return 'text-emerald-400'
  if (lower.includes('info')) return 'text-blue-400'
  return 'text-zinc-300'
}

export function LogViewer({
  lines,
  className,
  maxHeight = '500px',
}: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const isAutoScrollRef = useRef(true)

  useEffect(() => {
    const el = containerRef.current
    if (el && isAutoScrollRef.current) {
      el.scrollTop = el.scrollHeight
    }
  }, [lines])

  const handleScroll = () => {
    const el = containerRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
    isAutoScrollRef.current = atBottom
  }

  return (
    <div
      className={cn(
        'dark overflow-hidden rounded-xl border border-border bg-zinc-950',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
        <Terminal className="size-3.5 text-zinc-500" />
        <span className="text-sm font-medium text-zinc-400">Logs</span>
        <span className="ml-auto text-sm tabular-nums text-zinc-600">
          {lines.length} lines
        </span>
      </div>

      {/* Log content */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="overflow-auto p-3 font-mono text-sm leading-relaxed"
        style={{ maxHeight }}
      >
        {lines.length === 0 ? (
          <div className="py-8 text-center text-zinc-600">
            No log output yet.
          </div>
        ) : (
          lines.map((line, i) => (
              <div key={i} className={cn('whitespace-pre-wrap', lineColor(line))}>
                {line}
              </div>
          ))
        )}
      </div>
    </div>
  )
}
