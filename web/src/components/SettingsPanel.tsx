'use client'

import { useEffect, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  SelectGroup,
  SelectLabel,
  SelectSeparator,
} from '@/components/ui/select'
import { useSettingsStore, type ServerConfig } from '@/store/settings'
import { api } from '@/lib/api'
import { useSearchStore } from '@/store/search'
import {
  Settings,
  Cpu,
  Globe,
  Mail,
  FileText,
  ChevronDown,
  ChevronRight,
  Zap,
  Brain,
  Award,
  Sparkles,
  Hash,
  ScrollText,
} from 'lucide-react'
import { cn } from '@/lib/utils'

// Group models by provider for organized display
interface ModelGroup {
  label: string
  prefix: string
}

const MODEL_GROUPS: ModelGroup[] = [
  { label: 'Anthropic Claude', prefix: 'claude' },
  { label: 'Google Gemini', prefix: 'gemini' },
  { label: 'OpenAI', prefix: 'gpt' },
  { label: 'Ollama (Local)', prefix: 'ollama/' },
]

function groupModels(models: string[]): { label: string; models: string[] }[] {
  const groups: { label: string; models: string[] }[] = []
  for (const group of MODEL_GROUPS) {
    const matching = models.filter((m) => m.startsWith(group.prefix))
    if (matching.length > 0) {
      groups.push({ label: group.label, models: matching })
    }
  }
  // Catch any ungrouped models
  const allGrouped = groups.flatMap((g) => g.models)
  const ungrouped = models.filter((m) => !allGrouped.includes(m))
  if (ungrouped.length > 0) {
    groups.push({ label: 'Other', models: ungrouped })
  }
  return groups
}

// Friendly site labels
const SITE_LABELS: Record<string, string> = {
  linkedin: 'LinkedIn',
  indeed: 'Indeed',
  glassdoor: 'Glassdoor',
  zip_recruiter: 'ZipRecruiter',
  ats_boards: 'ATS Boards (company career sites)',
  industry_boards: 'Industry Boards (10 sites)',
}

function CollapsibleSection({
  title,
  icon: Icon,
  children,
  defaultOpen = false,
}: {
  title: string
  icon: typeof Settings
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-lg border border-border">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-base font-medium text-foreground transition-colors hover:bg-muted/50"
        onClick={() => setOpen(!open)}
      >
        <Icon className="size-4 shrink-0 text-muted-foreground" />
        <span className="flex-1">{title}</span>
        {open ? (
          <ChevronDown className="size-4 text-muted-foreground" />
        ) : (
          <ChevronRight className="size-4 text-muted-foreground" />
        )}
      </button>
      {open && (
        <div className="border-t border-border px-3 pb-3 pt-3">{children}</div>
      )}
    </div>
  )
}

function ModelSelector({
  label,
  value,
  onChange,
  models,
  disabled,
  helpText,
  icon: Icon,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  models: string[]
  disabled?: boolean
  helpText?: string
  icon?: typeof Cpu
}) {
  const grouped = groupModels(models)
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        {Icon && <Icon className="size-3.5 text-muted-foreground" />}
        <label className="text-sm font-medium text-muted-foreground">
          {label}
        </label>
      </div>
      <Select value={value} onValueChange={(v) => { if (v) onChange(v) }} disabled={disabled}>
        <SelectTrigger className="w-full text-sm">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {grouped.map((group, gi) => (
            <SelectGroup key={group.label}>
              <SelectLabel>{group.label}</SelectLabel>
              {group.models.map((model) => (
                <SelectItem key={model} value={model}>
                  {model}
                </SelectItem>
              ))}
              {gi < grouped.length - 1 && <SelectSeparator />}
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>
      {helpText && (
        <p className="text-xs text-muted-foreground/70">{helpText}</p>
      )}
    </div>
  )
}

export function SettingsPanel() {
  const [open, setOpen] = useState(false)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [allSites, setAllSites] = useState<string[]>([])
  const [rejectionStats, setRejectionStats] = useState<Record<string, number>>(
    {}
  )
  const [prompts, setPrompts] = useState<Record<string, string>>({})
  const [activePromptTab, setActivePromptTab] = useState('stage1')
  const { active: searchActive } = useSearchStore()

  const {
    resumePath,
    baseCvPath,
    cascadeEnabled,
    stage1Model,
    stage2Model,
    stage3Model,
    generationModel,
    filterModel,
    searchSites,
    maxJobs,
    emailRecipient,
    autoSendEmail,
    autoGenerateDocs,
    setResumePath,
    setBaseCvPath,
    setCascadeEnabled,
    setStage1Model,
    setStage2Model,
    setStage3Model,
    setGenerationModel,
    setFilterModel,
    setSearchSites,
    toggleSite,
    setMaxJobs,
    setEmailRecipient,
    setAutoSendEmail,
    setAutoGenerateDocs,
    hydrateFromServer,
  } = useSettingsStore()

  // Load config from server on mount
  useEffect(() => {
    api.config().then((cfg) => {
      if (cfg.available_models) {
        setAvailableModels(cfg.available_models)
      }
      if (cfg.search_sites) {
        setAllSites(cfg.search_sites)
      }
      // Hydrate settings store with server defaults (only if first load)
      if (cfg.available_models && cfg.cascade_models) {
        hydrateFromServer(cfg as ServerConfig)
      }
      if (cfg.prompts) {
        setPrompts(cfg.prompts)
      }
    }).catch(() => {
      // API not available, use hardcoded fallbacks
      setAvailableModels([
        'claude-fable-5-2',
        'claude-fable-5-1',
        'claude-opus-5-5',
        'claude-opus-5',
        'claude-sonnet-5',
        'claude-opus-4-8',
        'claude-sonnet-4-6',
        'claude-haiku-4-5-20251001',
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gpt-4.1',
        'gpt-4.1-mini',
      ])
      setAllSites([
        'linkedin',
        'indeed',
        'glassdoor',
        'zip_recruiter',
        'ats_boards',
        'industry_boards',
      ])
    })
  }, [hydrateFromServer])

  // Load rejection stats when panel opens
  useEffect(() => {
    if (open) {
      api.stats.rejections().then(setRejectionStats).catch(() => {})
    }
  }, [open])

  const models = availableModels.length > 0 ? availableModels : [
    'claude-opus-4-8',
    'claude-sonnet-4-6',
    'claude-haiku-4-5-20251001',
  ]

  const sites = allSites.length > 0 ? allSites : [
    'linkedin',
    'indeed',
    'glassdoor',
    'zip_recruiter',
    'ats_boards',
    'industry_boards',
  ]

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger
        render={
          <Button variant="ghost" size="icon-sm" className="text-muted-foreground hover:text-foreground" />
        }
      >
        <Settings className="size-4" />
        <span className="sr-only">Settings</span>
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Search Configuration</DialogTitle>
          <DialogDescription>
            Configure AI models, search sources, and pipeline settings.
            Changes persist across sessions.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          {/* File Paths */}
          <CollapsibleSection title="File Paths" icon={FileText}>
            <div className="space-y-3">
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-muted-foreground">
                  Resume Path
                </label>
                <Input
                  value={resumePath}
                  onChange={(e) => setResumePath(e.target.value)}
                  placeholder="resume.pdf"
                  disabled={searchActive}
                  className="text-sm"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-muted-foreground">
                  Base CV Path
                </label>
                <Input
                  value={baseCvPath}
                  onChange={(e) => setBaseCvPath(e.target.value)}
                  placeholder="base_cv.docx"
                  disabled={searchActive}
                  className="text-sm"
                />
                <p className="text-xs text-muted-foreground/70">
                  Domain-specific CVs (base_cv_*.docx) are auto-selected based
                  on keyword category.
                </p>
              </div>
            </div>
          </CollapsibleSection>

          {/* AI Models & Cascade */}
          <CollapsibleSection
            title="AI Models & Cascade"
            icon={Cpu}
            defaultOpen
          >
            <div className="space-y-4">
              {/* Cascade Toggle */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-base font-medium text-foreground">
                    3-Stage Cascade
                  </label>
                  <p className="text-xs text-muted-foreground/70">
                    S1: Binary filter &rarr; S2: Full grading &rarr; S3: Judge
                    borderline
                  </p>
                </div>
                <Switch
                  checked={cascadeEnabled}
                  onCheckedChange={setCascadeEnabled}
                  disabled={searchActive}
                />
              </div>

              {cascadeEnabled ? (
                <>
                  <div className="rounded-lg bg-muted/30 p-2.5">
                    <p className="text-xs text-muted-foreground">
                      Each stage uses a dedicated model for cost efficiency.
                      Stage 1 is the cheapest (binary YES/NO), Stage 3 is
                      premium (borderline decisions).
                    </p>
                  </div>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                    <ModelSelector
                      label="Stage 1 (Pre-filter)"
                      value={stage1Model}
                      onChange={setStage1Model}
                      models={models}
                      disabled={searchActive}
                      helpText="Fast binary filter"
                      icon={Zap}
                    />
                    <ModelSelector
                      label="Stage 2 (Grading)"
                      value={stage2Model}
                      onChange={setStage2Model}
                      models={models}
                      disabled={searchActive}
                      helpText="Full recruiter grading"
                      icon={Brain}
                    />
                    <ModelSelector
                      label="Stage 3 (Judge)"
                      value={stage3Model}
                      onChange={setStage3Model}
                      models={models}
                      disabled={searchActive}
                      helpText="Borderline decisions"
                      icon={Award}
                    />
                  </div>
                </>
              ) : (
                <ModelSelector
                  label="Filter Model"
                  value={filterModel}
                  onChange={setFilterModel}
                  models={models}
                  disabled={searchActive}
                  helpText="Single model for all job relevance filtering"
                  icon={Brain}
                />
              )}

              <div className="h-px bg-border" />

              <ModelSelector
                label="Generation Model"
                value={generationModel}
                onChange={setGenerationModel}
                models={models}
                disabled={searchActive}
                helpText="Used for tailoring CVs and Cover Letters"
                icon={Sparkles}
              />

              {/* Auto-generate toggle */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-base font-medium text-foreground">
                    Auto-generate CV/CL
                  </label>
                  <p className="text-xs text-muted-foreground/70">
                    Generate tailored docs for HIGH_MATCH jobs automatically
                  </p>
                </div>
                <Switch
                  checked={autoGenerateDocs}
                  onCheckedChange={setAutoGenerateDocs}
                  disabled={searchActive}
                />
              </div>
            </div>
          </CollapsibleSection>

          {/* Search Sites */}
          <CollapsibleSection title="Search Sources" icon={Globe} defaultOpen>
            <div className="space-y-2">
              {sites.map((site) => {
                const checked = searchSites.includes(site)
                return (
                  <label
                    key={site}
                    className={cn(
                      'flex cursor-pointer items-center gap-2.5 rounded-md px-2 py-1.5 text-base transition-colors hover:bg-muted/50',
                      searchActive && 'pointer-events-none opacity-50'
                    )}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleSite(site)}
                      disabled={searchActive}
                      className="size-3.5 rounded border-border accent-primary"
                    />
                    <span
                      className={cn(
                        'text-base',
                        checked
                          ? 'font-medium text-foreground'
                          : 'text-muted-foreground'
                      )}
                    >
                      {SITE_LABELS[site] ?? site}
                    </span>
                  </label>
                )
              })}
            </div>
          </CollapsibleSection>

          {/* Max Jobs */}
          <CollapsibleSection title="Limits" icon={Hash}>
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-muted-foreground">
                Max Jobs (0 = unlimited)
              </label>
              <Input
                type="number"
                min={0}
                value={maxJobs}
                onChange={(e) =>
                  setMaxJobs(Math.max(0, parseInt(e.target.value) || 0))
                }
                disabled={searchActive}
                className="w-32 text-sm"
              />
              <p className="text-xs text-muted-foreground/70">
                Limit total jobs scraped per search session.
              </p>
            </div>
          </CollapsibleSection>

          {/* Email Settings */}
          <CollapsibleSection title="Email Settings" icon={Mail}>
            <div className="space-y-3">
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-muted-foreground">
                  Recipient(s)
                </label>
                <Input
                  value={emailRecipient}
                  onChange={(e) => setEmailRecipient(e.target.value)}
                  placeholder="email@example.com"
                  className="text-sm"
                />
                <p className="text-xs text-muted-foreground/70">
                  Comma-separated email addresses for result delivery.
                </p>
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-base font-medium text-foreground">
                    Auto-send after search
                  </label>
                  <p className="text-xs text-muted-foreground/70">
                    Email results when a search completes
                  </p>
                </div>
                <Switch
                  checked={autoSendEmail}
                  onCheckedChange={setAutoSendEmail}
                />
              </div>
            </div>
          </CollapsibleSection>

          {/* Rejection Stats */}
          <CollapsibleSection title="Rejection Patterns" icon={FileText}>
            {Object.keys(rejectionStats).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(rejectionStats)
                  .sort(([, a], [, b]) => b - a)
                  .map(([reason, count]) => (
                    <div
                      key={reason}
                      className="flex items-center justify-between rounded-md bg-muted/30 px-2.5 py-1.5"
                    >
                      <span className="text-sm text-muted-foreground">
                        {reason.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                      </span>
                      <span className="text-sm font-semibold tabular-nums text-foreground">
                        {count}
                      </span>
                    </div>
                  ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No dismissals recorded yet.
              </p>
            )}
          </CollapsibleSection>

          {/* AI Prompts */}
          <CollapsibleSection title="AI Prompts" icon={ScrollText}>
            {Object.keys(prompts).length > 0 ? (
              <div className="space-y-3">
                {/* Prompt category tabs */}
                <div className="flex flex-wrap gap-1.5">
                  {([
                    { key: 'stage1', label: 'Stage 1' },
                    { key: 'stage2_claude', label: 'Stage 2' },
                    { key: 'stage3', label: 'Stage 3' },
                    { key: 'cv_summary_claude', label: 'CV Summary' },
                    { key: 'cv_experience_claude', label: 'CV Experience' },
                  ] as const).map(({ key, label }) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setActivePromptTab(key)}
                      className={cn(
                        'rounded-md px-2.5 py-1 text-xs font-medium transition-colors',
                        activePromptTab === key
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                      )}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                {/* Prompt description */}
                <p className="text-xs text-muted-foreground/70">
                  {activePromptTab === 'stage1' && 'Stage 1 — Fast Binary Filter (YES/NO relevance check)'}
                  {activePromptTab === 'stage2_claude' && 'Stage 2 — Batched Grading (full recruiter evaluation)'}
                  {activePromptTab === 'stage3' && 'Stage 3 — Premium Judge (borderline decision with web enrichment)'}
                  {activePromptTab === 'cv_summary_claude' && 'CV Summary Template (professional summary tailoring)'}
                  {activePromptTab === 'cv_experience_claude' && 'CV Experience Template (bullet point sharpening)'}
                </p>

                {/* Prompt content */}
                <textarea
                  readOnly
                  value={prompts[activePromptTab] ?? ''}
                  className="w-full rounded-md border border-border bg-muted/30 p-3 font-mono text-xs leading-relaxed text-foreground focus:outline-none"
                  style={{ maxHeight: '300px', minHeight: '150px', resize: 'vertical' }}
                />
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                Loading prompts from server...
              </p>
            )}
          </CollapsibleSection>
        </div>
      </DialogContent>
    </Dialog>
  )
}
