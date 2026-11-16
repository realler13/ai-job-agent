'use client'

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Button, buttonVariants } from '@/components/ui/button'
import { MatchBadge } from '@/components/MatchBadge'
import { ConfidenceBar } from '@/components/ConfidenceBar'
import { DocumentPanel } from '@/components/DocumentPanel'
import { CompanyIntelPanel } from '@/components/CompanyIntelPanel'
import { PostingCheckPanel } from '@/components/PostingCheckPanel'
import type { Job } from '@/lib/types'
import { cn } from '@/lib/utils'
import {
  Building2,
  MapPin,
  Calendar,
  ExternalLink,
  Tag,
  Bookmark,
  XCircle,
} from 'lucide-react'

interface JobDetailDrawerProps {
  job: Job | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onTrack?: (id: number) => void
  onDismiss?: (id: number) => void
  onGenerateDocs?: (id: number) => Promise<void>
}

export function JobDetailDrawer({
  job,
  open,
  onOpenChange,
  onTrack,
  onDismiss,
  onGenerateDocs,
}: JobDetailDrawerProps) {
  if (!job) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="pr-8 text-lg leading-tight">
            {job.title}
          </DialogTitle>
          <DialogDescription className="flex items-center gap-2 pt-1">
            <Building2 className="size-3.5 shrink-0" />
            {job.company}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4">
          {/* Metadata row */}
          <div className="flex flex-wrap items-center gap-2">
            <MatchBadge level={job.match_level} />
            <ConfidenceBar value={job.confidence} />
          </div>

          {/* Details grid */}
          <div className="grid grid-cols-2 gap-3 text-sm">
            {job.location && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <MapPin className="size-3.5 shrink-0" />
                <span>{job.location}</span>
              </div>
            )}
            {job.date_posted && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Calendar className="size-3.5 shrink-0" />
                <span>{job.date_posted}</span>
              </div>
            )}
            {job.keyword_category && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Tag className="size-3.5 shrink-0" />
                <span className="capitalize">
                  {job.keyword_category.replace(/_/g, ' ')}
                </span>
              </div>
            )}
            {job.session_id && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Calendar className="size-3.5 shrink-0" />
                <span className="text-sm">{job.session_id}</span>
              </div>
            )}
          </div>

          {/* Link */}
          {job.link && (
            <a
              href={job.link}
              target="_blank"
              rel="noopener noreferrer"
              className={cn(buttonVariants({ variant: 'outline', size: 'sm' }), 'w-fit gap-1.5 no-underline')}
            >
              <ExternalLink className="size-3.5" />
              View Original Posting
            </a>
          )}

          {/* Reasoning */}
          {job.reasoning && (
            <div className="rounded-lg border border-border bg-muted/50 p-3">
              <div className="mb-1 text-sm font-medium uppercase tracking-wider text-muted-foreground">
                AI Reasoning
              </div>
              <p className="text-sm leading-relaxed text-foreground">
                {job.reasoning}
              </p>
            </div>
          )}

          {/* Posting check — facts about the posting, fetched while open */}
          <PostingCheckPanel jobId={job.id} enabled={open} />

          {/* Company Intel */}
          <CompanyIntelPanel
            company={job.company}
            jobTitle={job.title}
            matchLevel={job.match_level}
          />

          {/* Documents */}
          <DocumentPanel
            job={job}
            onGenerate={
              onGenerateDocs ? () => onGenerateDocs(job.id) : undefined
            }
          />

          {/* Action buttons */}
          <div className="flex gap-2 border-t border-border pt-3">
            {onTrack && (
              <Button
                variant="default"
                size="sm"
                className="gap-1.5"
                onClick={() => onTrack(job.id)}
              >
                <Bookmark className="size-3.5" />
                Track
              </Button>
            )}
            {onDismiss && (
              <Button
                variant="destructive"
                size="sm"
                className="gap-1.5"
                onClick={() => onDismiss(job.id)}
              >
                <XCircle className="size-3.5" />
                Dismiss
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
