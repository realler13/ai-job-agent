'use client'

import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { useSearchStore } from '@/store/search'

/**
 * Keeps the Zustand search store in sync with the backend's real state.
 *
 * Before this, NOTHING consumed /api/search/status or the SSE stream: the
 * store was only mutated optimistically by the Sidebar buttons, so progress
 * sat at "Starting..." for the whole run, `active` latched true forever after
 * completion (jobs page refetching every 5s indefinitely), and a page reload
 * mid-run made the UI claim nothing was running.
 *
 * Polls fast (2.5s) while a search is active, slow (15s) when idle — the idle
 * poll is what recovers true state after a reload.
 */
export function SearchStatusSync() {
  const active = useSearchStore((s) => s.active)
  const setStatus = useSearchStore((s) => s.setStatus)

  const { data } = useQuery({
    queryKey: ['search-status'],
    queryFn: api.search.status,
    refetchInterval: active ? 2500 : 15000,
    refetchOnWindowFocus: true,
  })

  useEffect(() => {
    if (!data) return
    setStatus({
      active: !!data.active,
      paused: !!data.paused,
      progress: data.progress ?? 0,
      statusText: data.status_text ?? 'Idle',
      sessionId: data.session_id ?? null,
    })
  }, [data, setStatus])

  return null
}
