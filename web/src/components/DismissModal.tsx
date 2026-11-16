'use client'

import { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import type { RejectionReason } from '@/lib/types'

const REJECTION_REASONS: { value: RejectionReason; label: string }[] = [
  { value: 'too_junior', label: 'Wrong seniority' },
  { value: 'social_media', label: 'Wrong role type' },
  { value: 'wrong_industry', label: 'Wrong industry' },
  { value: 'agency_vendor', label: 'Wrong employer type' },
  { value: 'wrong_location', label: 'Wrong location' },
  { value: 'bad_company_fit', label: 'Bad company fit' },
  { value: 'already_applied', label: 'Already applied' },
  { value: 'other', label: 'Other' },
]

interface DismissModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onDismiss: (reason: string) => void
  jobTitle?: string
}

export function DismissModal({
  open,
  onOpenChange,
  onDismiss,
  jobTitle,
}: DismissModalProps) {
  const [selectedReason, setSelectedReason] = useState<string>('')
  const [customReason, setCustomReason] = useState('')

  const handleSubmit = () => {
    const reason =
      selectedReason === 'other' && customReason.trim()
        ? customReason.trim()
        : selectedReason
    if (reason) {
      onDismiss(reason)
      setSelectedReason('')
      setCustomReason('')
      onOpenChange(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) {
          setSelectedReason('')
          setCustomReason('')
        }
        onOpenChange(next)
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Dismiss Job</DialogTitle>
          <DialogDescription>
            {jobTitle
              ? `Why are you dismissing "${jobTitle}"?`
              : 'Select a reason for dismissing this job.'}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3 py-2">
          <Select value={selectedReason} onValueChange={(v) => setSelectedReason(v ?? '')}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a reason..." />
            </SelectTrigger>
            <SelectContent>
              {REJECTION_REASONS.map((r) => (
                <SelectItem key={r.value} value={r.value}>
                  {r.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {selectedReason === 'other' && (
            <Textarea
              placeholder="Describe the reason..."
              value={customReason}
              onChange={(e) => setCustomReason(e.target.value)}
              className="min-h-20 resize-none"
            />
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleSubmit}
            disabled={
              !selectedReason ||
              (selectedReason === 'other' && !customReason.trim())
            }
          >
            Dismiss
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
