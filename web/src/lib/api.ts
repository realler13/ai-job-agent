import type {
  CompanyIntel, BatchStatus, SentCvAnswer, PostingCheck, PostingQualityResponse,
  ApplicationOutcome, GenerateDocsResult, GeneratingJobs,
} from './types'

// Use relative URLs — Next.js rewrites /api/* to FastAPI on port 8502
// This also works from another device, because the browser only talks to port 8503
export function getBase() {
  return process.env.NEXT_PUBLIC_API_URL || ''
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  health: () => fetchJSON<{ status: string; db: string }>(`${getBase()}/api/health`),
  jobs: {
    list: (params?: Record<string, string | number | boolean>) => {
      const qs = params
        ? '?' +
          new URLSearchParams(
            Object.entries(params).map(([k, v]) => [k, String(v)])
          ).toString()
        : ''
      return fetchJSON<{ jobs: any[]; total: number }>(`${getBase()}/api/jobs${qs}`)
    },
    latestSession: (includeIrrelevant = false) =>
      fetchJSON<{ jobs: any[]; total: number; session_id: string | null }>(
        `${getBase()}/api/jobs/latest-session${includeIrrelevant ? '?include_irrelevant=true' : ''}`
      ),
    unreviewed: () => fetchJSON<any[]>(`${getBase()}/api/jobs/unreviewed`),
    get: (id: number) => fetchJSON<any>(`${getBase()}/api/jobs/${id}`),
    create: (body: {
      title: string
      company: string
      location?: string
      link?: string
      description?: string
    }) =>
      fetchJSON<{ job_id: number }>(`${getBase()}/api/jobs`, {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    track: (id: number) =>
      fetchJSON<any>(`${getBase()}/api/jobs/${id}/track`, { method: 'POST' }),
    dismiss: (id: number, reason?: string) =>
      fetchJSON<any>(`${getBase()}/api/jobs/${id}/dismiss`, {
        method: 'POST',
        body: JSON.stringify({ reason }),
      }),
    generateDocs: (id: number) =>
      fetchJSON<GenerateDocsResult>(`${getBase()}/api/jobs/${id}/generate-docs`, {
        method: 'POST',
      }),
    cvUrl: (id: number) => `${getBase()}/api/jobs/${id}/cv`,
    clUrl: (id: number) => `${getBase()}/api/jobs/${id}/cl`,
    // Starts a BACKGROUND batch and returns at once; poll batchStatus().
    batchGenerate: (jobIds: number[]) =>
      fetchJSON<{ status: string; total: number }>(`${getBase()}/api/batch-generate`, {
        method: 'POST',
        body: JSON.stringify({ job_ids: jobIds }),
      }),
    batchStatus: () => fetchJSON<BatchStatus>(`${getBase()}/api/batch-generate/status`),
    // The jobs whose CV + cover letter are being generated right now.
    generating: () => fetchJSON<GeneratingJobs>(`${getBase()}/api/generating`),
  },
  applications: {
    list: () => fetchJSON<any[]>(`${getBase()}/api/applications`),
    update: (id: number, body: { status?: string; notes?: string }) =>
      fetchJSON<any>(`${getBase()}/api/applications/${id}`, {
        method: 'PUT',
        body: JSON.stringify(body),
      }),
    delete: (id: number) =>
      fetchJSON<any>(`${getBase()}/api/applications/${id}`, {
        method: 'DELETE',
      }),
    intel: (id: number) =>
      fetchJSON<any>(`${getBase()}/api/applications/${id}/intel`),
    generateIntel: (id: number) =>
      fetchJSON<any>(`${getBase()}/api/applications/${id}/intel`, {
        method: 'POST',
      }),
    prep: (id: number) =>
      fetchJSON<any>(`${getBase()}/api/applications/${id}/prep`),
    generatePrep: (id: number) =>
      fetchJSON<any>(`${getBase()}/api/applications/${id}/prep`, {
        method: 'POST',
      }),
    // "Did you send this CV?" — one click on the Kanban card; changeable.
    setSentCv: (id: number, answer: SentCvAnswer) =>
      fetchJSON<{ app_id: number; sent_cv: SentCvAnswer }>(
        `${getBase()}/api/applications/${id}/sent-cv`,
        { method: 'PUT', body: JSON.stringify({ answer }) }
      ),
    // A closed card (Rejected / Ghosted / I Withdrew): how far it got + what
    // happened — answered or changed any time while the card is closed.
    setOutcome: (id: number, body: { stage: string; what_happened: string | null }) =>
      fetchJSON<{ app_id: number } & Omit<ApplicationOutcome, 'sent_cv' | 'has_prep'>>(
        `${getBase()}/api/applications/${id}/outcome`,
        { method: 'PUT', body: JSON.stringify(body) }
      ),
  },
  warmPaths: {
    forCompany: (company: string) =>
      fetchJSON<any>(`${getBase()}/api/warm-paths?company=${encodeURIComponent(company)}`),
    import: () =>
      fetchJSON<any>(`${getBase()}/api/warm-paths/import`, { method: 'POST' }),
    stats: () => fetchJSON<any>(`${getBase()}/api/warm-paths/stats`),
  },
  search: {
    status: () => fetchJSON<any>(`${getBase()}/api/search/status`),
    start: (params: {
      limit?: number
      cascade?: boolean
      resume_path?: string
      base_cv_path?: string
      model_filter?: string
      model_generation?: string
      search_sites?: string[]
      stage1_model?: string
      stage2_model?: string
      stage3_model?: string
      send_email?: boolean
      target_email?: string
      auto_generate_docs?: boolean
    } = {}) =>
      fetchJSON<any>(`${getBase()}/api/search/start`, {
        method: 'POST',
        body: JSON.stringify(params),
      }),
    stop: () => fetchJSON<any>(`${getBase()}/api/search/stop`, { method: 'POST' }),
    pause: () =>
      fetchJSON<any>(`${getBase()}/api/search/pause`, { method: 'POST' }),
    progressUrl: () => `${getBase()}/api/search/progress`,
  },
  logs: (lines?: number) =>
    fetchJSON<{ lines: string[]; file: string | null }>(
      `${getBase()}/api/logs${lines ? `?lines=${lines}` : ''}`
    ),
  stats: {
    overview: (days = 30) =>
      fetchJSON<any>(`${getBase()}/api/stats/overview?days=${days}`),
    cascade: () => fetchJSON<any>(`${getBase()}/api/stats/cascade`),
    rejections: () =>
      fetchJSON<Record<string, number>>(`${getBase()}/api/stats/rejections`),
  },
  companyIntel: {
    get: (company: string, jobTitle?: string) => {
      const params = new URLSearchParams()
      if (jobTitle) params.set('job_title', jobTitle)
      const qs = params.toString()
      return fetchJSON<CompanyIntel>(
        `${getBase()}/api/company-intel/${encodeURIComponent(company)}${qs ? `?${qs}` : ''}`
      )
    },
    enrich: (company: string, jobTitle: string, location?: string) =>
      fetchJSON<CompanyIntel>(`${getBase()}/api/company-intel/enrich`, {
        method: 'POST',
        body: JSON.stringify({ company_name: company, job_title: jobTitle, location }),
      }),
  },
  // "Is this posting real?" — get() is free (no network, no model);
  // readPosting() makes one cheap model read of the posting text.
  postingCheck: {
    get: (id: number) =>
      fetchJSON<PostingCheck>(`${getBase()}/api/jobs/${id}/posting-check`),
    readPosting: (id: number) =>
      fetchJSON<PostingQualityResponse>(`${getBase()}/api/jobs/${id}/posting-check/quality`, {
        method: 'POST',
      }),
  },
  config: () => fetchJSON<any>(`${getBase()}/api/config`),
}
