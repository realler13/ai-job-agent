import { create } from 'zustand'

interface SearchStore {
  active: boolean
  paused: boolean
  progress: number
  statusText: string
  sessionId: string | null
  setStatus: (s: Partial<Omit<SearchStore, 'setStatus'>>) => void
}

export const useSearchStore = create<SearchStore>((set) => ({
  active: false,
  paused: false,
  progress: 0,
  statusText: 'Idle',
  sessionId: null,
  setStatus: (s) => set(s),
}))
