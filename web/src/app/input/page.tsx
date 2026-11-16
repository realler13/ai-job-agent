'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  PenLine,
  Send,
  Loader2,
  CheckCircle,
  Building2,
  MapPin,
  Link as LinkIcon,
  FileText,
  ExternalLink,
  Search,
  Sparkles,
} from 'lucide-react'
import Link from 'next/link'
import { CompanyIntelPanel } from '@/components/CompanyIntelPanel'

interface FormState {
  title: string
  company: string
  location: string
  link: string
  description: string
}

const INITIAL_FORM: FormState = {
  title: '',
  company: '',
  location: '',
  link: '',
  description: '',
}

export default function InputPage() {
  const [form, setForm] = useState<FormState>(INITIAL_FORM)
  const [createdJobId, setCreatedJobId] = useState<number | null>(null)
  const [submittedJob, setSubmittedJob] = useState<{
    company: string
    title: string
    location: string
  } | null>(null)
  const [enrichmentStatus, setEnrichmentStatus] = useState<
    'idle' | 'running' | 'done' | 'error'
  >('idle')

  const createMutation = useMutation({
    mutationFn: api.jobs.create,
    onSuccess: (data) => {
      const company = form.company.trim()
      const title = form.title.trim()
      const location = form.location.trim()
      setCreatedJobId(data.job_id)
      setSubmittedJob({ company, title, location })

      // Auto-trigger enrichment (Perplexity intel + Opus salary advisor) in background
      setEnrichmentStatus('running')
      api.companyIntel
        .enrich(company, title, location || undefined)
        .then(() => setEnrichmentStatus('done'))
        .catch(() => setEnrichmentStatus('error'))
    },
  })

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target
    setForm((prev) => ({ ...prev, [name]: value }))
    // Clear success state when user starts editing again
    if (createdJobId) setCreatedJobId(null)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!form.title.trim() || !form.company.trim()) return

    createMutation.mutate({
      title: form.title.trim(),
      company: form.company.trim(),
      location: form.location.trim() || undefined,
      link: form.link.trim() || undefined,
      description: form.description.trim() || undefined,
    })
  }

  const handleReset = () => {
    setForm(INITIAL_FORM)
    setCreatedJobId(null)
    setSubmittedJob(null)
    setEnrichmentStatus('idle')
    createMutation.reset()
  }

  // The description is optional: a company that reached out directly (an
  // inbound process) may have no posting at all.
  const isValid = form.title.trim() && form.company.trim()

  // Count words in description
  const wordCount = form.description.trim()
    ? form.description.trim().split(/\s+/).length
    : 0

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          Manual Job Input
        </h1>
        <p className="mt-1 text-base text-muted-foreground">
          Add a job with its title and company. The description is optional —
          leave it empty when the company reached out to you directly; with no
          description you get the general version of your CV.
        </p>
      </div>

      {/* Two-column layout */}
      <div className="grid gap-6 lg:grid-cols-5">
        {/* Form — left side (60%) */}
        <div className="lg:col-span-3">
          <Card className="border-border/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <PenLine className="size-4" />
                Job Details
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="flex flex-col gap-5">
                {/* Title */}
                <div className="space-y-1.5">
                  <label
                    htmlFor="title"
                    className="text-sm font-medium text-muted-foreground"
                  >
                    Job Title <span className="text-red-400">*</span>
                  </label>
                  <Input
                    id="title"
                    name="title"
                    value={form.title}
                    onChange={handleChange}
                    placeholder="e.g. Video Producer"
                    className="h-10"
                  />
                </div>

                {/* Company */}
                <div className="space-y-1.5">
                  <label
                    htmlFor="company"
                    className="text-sm font-medium text-muted-foreground"
                  >
                    Company <span className="text-red-400">*</span>
                  </label>
                  <Input
                    id="company"
                    name="company"
                    value={form.company}
                    onChange={handleChange}
                    placeholder="e.g. Acme Studios"
                    className="h-10"
                  />
                </div>

                {/* Location + URL side by side */}
                <div className="grid gap-5 sm:grid-cols-2">
                  <div className="space-y-1.5">
                    <label
                      htmlFor="location"
                      className="text-sm font-medium text-muted-foreground"
                    >
                      Location
                    </label>
                    <Input
                      id="location"
                      name="location"
                      value={form.location}
                      onChange={handleChange}
                      placeholder="e.g. Seattle, WA or Remote"
                      className="h-10"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label
                      htmlFor="link"
                      className="text-sm font-medium text-muted-foreground"
                    >
                      Job URL
                    </label>
                    <Input
                      id="link"
                      name="link"
                      value={form.link}
                      onChange={handleChange}
                      placeholder="https://..."
                      className="h-10"
                    />
                  </div>
                </div>

                {/* Description */}
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <label
                      htmlFor="description"
                      className="text-sm font-medium text-muted-foreground"
                    >
                      Job Description{' '}
                      <span className="text-muted-foreground/70">(optional)</span>
                    </label>
                    <span className="text-sm tabular-nums text-muted-foreground">
                      {wordCount} words
                    </span>
                  </div>
                  <Textarea
                    id="description"
                    name="description"
                    value={form.description}
                    onChange={handleChange}
                    placeholder="Paste the full job description here — or leave empty if the company reached out to you directly"
                    className="min-h-[240px] resize-y text-base leading-relaxed"
                  />
                </div>

                {/* Submit row */}
                <div className="flex items-center gap-3 border-t border-border/50 pt-4">
                  <Button
                    type="submit"
                    disabled={!isValid || createMutation.isPending}
                    className="gap-1.5"
                  >
                    {createMutation.isPending ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <Send className="size-4" />
                    )}
                    {createMutation.isPending ? 'Submitting...' : 'Submit Job'}
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleReset}
                    className="text-muted-foreground"
                  >
                    Clear
                  </Button>
                </div>

                {/* Error message */}
                {createMutation.isError && (
                  <div className="rounded-lg border border-red-500/25 bg-red-500/10 p-3 text-base text-red-400">
                    {createMutation.error instanceof Error
                      ? createMutation.error.message
                      : 'Failed to create job. Please try again.'}
                  </div>
                )}
              </form>
            </CardContent>
          </Card>
        </div>

        {/* Preview — right side (40%) */}
        <div className="lg:col-span-2">
          <Card className="sticky top-6 border-border/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <FileText className="size-4" />
                Preview
              </CardTitle>
            </CardHeader>
            <CardContent>
              {/* Success state */}
              {createdJobId ? (
                <div className="flex flex-col gap-5">
                  <div className="flex flex-col items-center gap-4 py-6 text-center">
                    <div className="flex size-14 items-center justify-center rounded-2xl bg-emerald-500/10">
                      <CheckCircle className="size-7 text-emerald-400" />
                    </div>
                    <div>
                      <p className="text-base font-medium text-foreground">
                        Job saved &amp; tracked to Kanban
                      </p>
                      <p className="mt-1 text-sm text-muted-foreground">
                        Job ID: {createdJobId}
                      </p>
                    </div>

                    {/* Background pipeline status */}
                    <div className="flex flex-col gap-1.5 text-sm">
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Sparkles className="size-3.5 text-primary" />
                        Writing your CV and cover letter: usually 5–10 minutes.
                      </div>
                      <div className="text-muted-foreground">
                        Its Kanban card says Generating… until they are ready.
                      </div>
                      {enrichmentStatus === 'running' && (
                        <div className="flex items-center gap-2 text-blue-500">
                          <Loader2 className="size-3.5 animate-spin" />
                          Generating company intel...
                        </div>
                      )}
                      {enrichmentStatus === 'done' && (
                        <div className="flex items-center gap-2 text-emerald-500">
                          <CheckCircle className="size-3.5" />
                          Company intel ready
                        </div>
                      )}
                      {enrichmentStatus === 'error' && (
                        <div className="flex items-center gap-2 text-amber-500">
                          <Search className="size-3.5" />
                          Intel enrichment failed — use button below to retry
                        </div>
                      )}
                    </div>

                    <div className="flex flex-col gap-2">
                      <Link href="/kanban">
                        <Button variant="default" size="sm" className="gap-1.5">
                          <ExternalLink className="size-3.5" />
                          View in Kanban
                        </Button>
                      </Link>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleReset}
                      >
                        Add Another Job
                      </Button>
                    </div>
                  </div>

                  {/* Company intel for the submitted job */}
                  {submittedJob && (
                    <CompanyIntelPanel
                      company={submittedJob.company}
                      jobTitle={submittedJob.title}
                      matchLevel="USER_INPUT"
                    />
                  )}
                </div>
              ) : /* Live preview */ form.title || form.company || form.description ? (
                <div className="flex flex-col gap-4">
                  {/* Title + company */}
                  {(form.title || form.company) && (
                    <div>
                      {form.title && (
                        <h3 className="text-base font-semibold text-foreground">
                          {form.title}
                        </h3>
                      )}
                      {form.company && (
                        <div className="mt-1 flex items-center gap-1.5 text-sm text-muted-foreground">
                          <Building2 className="size-3.5" />
                          {form.company}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Location + URL */}
                  {(form.location || form.link) && (
                    <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
                      {form.location && (
                        <div className="flex items-center gap-1">
                          <MapPin className="size-3" />
                          {form.location}
                        </div>
                      )}
                      {form.link && (
                        <div className="flex items-center gap-1">
                          <LinkIcon className="size-3" />
                          <a
                            href={form.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="truncate text-primary hover:underline"
                          >
                            {(() => {
                              try {
                                return new URL(form.link).hostname
                              } catch {
                                return form.link
                              }
                            })()}
                          </a>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Description preview */}
                  {form.description && (
                    <div className="space-y-1.5">
                      <span className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
                        Description
                      </span>
                      <div className="max-h-[300px] overflow-y-auto rounded-lg border border-border/50 bg-muted/50 p-3 text-sm leading-relaxed text-muted-foreground">
                        {form.description.split('\n').map((line, i) => (
                          <p key={i} className={line.trim() ? '' : 'h-3'}>
                            {line}
                          </p>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                /* Empty preview */
                <div className="flex flex-col items-center gap-3 py-8 text-center">
                  <PenLine className="size-8 text-muted-foreground/40" />
                  <p className="text-sm text-muted-foreground">
                    Start filling in the form to see a live preview of the job
                    listing.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
