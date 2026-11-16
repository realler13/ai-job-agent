import {
  Sparkles,
  Search,
  Filter,
  ClipboardCheck,
  FileText,
  Briefcase,
  Eye,
  Kanban,
  PenLine,
  ScrollText,
  Settings,
  Lightbulb,
  ArrowRight,
  CheckCircle,
  Star,
  Clock,
  Shield,
  Building2,
  DollarSign,
  ChevronRight,
  Brain,
  Mail,
  Users,
  TrendingUp,
  BadgeCheck,
  MessageCircle,
  GraduationCap,
  FileSearch,
} from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'

/* ------------------------------------------------------------------ */
/*  Reusable sub-components for the help page                         */
/* ------------------------------------------------------------------ */

function SectionAnchor({ id }: { id: string }) {
  return <div id={id} className="scroll-mt-8" />
}

function SectionHeading({
  icon: Icon,
  title,
  id,
}: {
  icon: React.ComponentType<{ className?: string }>
  title: string
  id: string
}) {
  return (
    <>
      <SectionAnchor id={id} />
      <h2 className="group flex items-center gap-2.5 text-2xl font-bold tracking-tight text-foreground">
        <Icon className="size-6 shrink-0 text-primary" />
        {title}
        <a
          href={`#${id}`}
          className="text-muted-foreground/0 transition-colors group-hover:text-muted-foreground"
          aria-label={`Link to ${title}`}
        >
          #
        </a>
      </h2>
    </>
  )
}

function Divider() {
  return <hr className="border-border/60" />
}

/* Four-step flow card */
function StepCard({
  step,
  icon: Icon,
  title,
  description,
}: {
  step: number
  icon: React.ComponentType<{ className?: string }>
  title: string
  description: string
}) {
  return (
    <div className="relative flex flex-col items-center gap-3 rounded-xl border border-border/50 bg-card p-5 text-center">
      {/* Step number badge */}
      <div className="flex size-10 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
        {step}
      </div>
      <Icon className="size-6 text-primary" />
      <h3 className="text-base font-semibold text-foreground">{title}</h3>
      <p className="text-sm leading-relaxed text-muted-foreground">
        {description}
      </p>
    </div>
  )
}

/* Table of contents link */
function TocLink({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      className="flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-primary"
    >
      <ChevronRight className="size-3.5 shrink-0" />
      {label}
    </a>
  )
}

/* ------------------------------------------------------------------ */
/*  Main Page                                                         */
/* ------------------------------------------------------------------ */

export default function HelpPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-10 pb-16">
      {/* ============================================================ */}
      {/*  Welcome                                                     */}
      {/* ============================================================ */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="flex size-12 items-center justify-center rounded-2xl bg-primary/10">
            <Sparkles className="size-6 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-foreground">
              Welcome to AI Job Agent
            </h1>
            <p className="text-base text-muted-foreground">
              Your personal guide to the system
            </p>
          </div>
        </div>

        <div className="space-y-3 text-base leading-relaxed text-muted-foreground">
          <p>
            AI Job Agent is your personal AI-powered job search assistant.
            It works around the clock to find roles that
            match your experience — scanning over 200 company career pages, four
            major job boards, and ten niche industry sites, then filtering
            everything through AI that understands what you actually do.
          </p>
          <p>
            No more scrolling through hundreds of irrelevant listings. The
            system does the heavy lifting so you can focus on what matters:
            preparing, interviewing, and landing the right role.
          </p>
        </div>
      </div>

      {/* Table of contents */}
      <Card className="border-border/50">
        <CardContent className="p-5">
          <p className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            On this page
          </p>
          <div className="grid gap-1.5 sm:grid-cols-2">
            <TocLink href="#how-it-works" label="How It Works" />
            <TocLink href="#opus-advisor" label="Opus Salary Advisor" />
            <TocLink href="#auto-cv-cl" label="Auto CV & Cover Letter" />
            <TocLink href="#company-intel" label="Company Intel" />
            <TocLink href="#posting-check" label="Is This Posting Real?" />
            <TocLink href="#jobs-tab" label="Jobs Tab" />
            <TocLink href="#review-tab" label="Review Tab" />
            <TocLink href="#kanban-tab" label="Kanban Tab" />
            <TocLink href="#outreach" label="Outreach Playbook" />
            <TocLink href="#interview-prep" label="Interview Prep" />
            <TocLink href="#warm-paths" label="Warm Paths" />
            <TocLink href="#stats-tab" label="Stats Tab" />
            <TocLink href="#input-tab" label="Input Tab" />
            <TocLink href="#email-digests" label="Email Digests" />
            <TocLink href="#logs-tab" label="Logs Tab" />
            <TocLink href="#settings" label="Settings" />
            <TocLink href="#tips" label="Tips" />
          </div>
        </CardContent>
      </Card>

      <Divider />

      {/* ============================================================ */}
      {/*  How It Works                                                */}
      {/* ============================================================ */}
      <section className="space-y-5">
        <SectionHeading icon={Sparkles} title="How It Works" id="how-it-works" />

        <p className="text-base leading-relaxed text-muted-foreground">
          On the schedule you set up —{' '}
          <strong className="text-foreground">twice a day</strong> in the
          default setup — the system runs
          automatically. Here is what happens behind the scenes:
        </p>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StepCard
            step={1}
            icon={Search}
            title="Search"
            description="Scans over 200 company career pages, LinkedIn, Indeed, Glassdoor, ZipRecruiter, Google Jobs, Aquent (creative staffing), and ten niche industry boards — automatically, on your schedule. Google Jobs results are kept only when they carry a direct employer application link — no paywall boards or redirect chains."
          />
          <StepCard
            step={2}
            icon={Filter}
            title="AI Filter"
            description="A 3-stage AI cascade reviews every listing. Quick screen, detailed grading, then a premium judge picks the real matches."
          />
          <StepCard
            step={3}
            icon={ClipboardCheck}
            title="Review"
            description="Matched jobs appear in your Review tab with AI reasoning explaining why each one fits your background."
          />
          <StepCard
            step={4}
            icon={FileText}
            title="Apply"
            description="A tailored CV and cover letter are generated automatically for each match, using your real experience and verified metrics."
          />
        </div>

        {/* Arrow connectors on large screens */}
        <div className="hidden lg:flex items-center justify-center gap-1 text-muted-foreground/40">
          <span className="text-sm">Search</span>
          <ArrowRight className="size-4" />
          <span className="text-sm">Filter</span>
          <ArrowRight className="size-4" />
          <span className="text-sm">Review</span>
          <ArrowRight className="size-4" />
          <span className="text-sm">Apply</span>
        </div>

        <p className="text-sm leading-relaxed text-muted-foreground">
          You will get an email summary after each scheduled search, sent to
          the recipients set in your configuration. Even on days with nothing
          new, you will hear from the system so you know it is working.
        </p>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Opus Salary Advisor                                         */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={Brain} title="Opus Salary Advisor" id="opus-advisor" />

        <p className="text-base leading-relaxed text-muted-foreground">
          After the AI finds matching jobs, the most capable model in the
          pipeline — <strong className="text-foreground">Claude Opus</strong> —
          goes a step further. It analyzes market data alongside your
          profile to produce a personalized salary recommendation for each role.
        </p>

        <div className="space-y-2 text-sm leading-relaxed text-muted-foreground">
          <h3 className="text-lg font-semibold text-foreground">
            What goes into the recommendation
          </h3>
          <ul className="ml-5 list-disc space-y-1">
            <li>
              <strong className="text-foreground">Market rates</strong> — Salary
              data pulled from sources like Glassdoor and Levels.fyi for similar
              roles in the same region.
            </li>
            <li>
              <strong className="text-foreground">
                Company&apos;s posted pay range
              </strong>{' '}
              — When the listing includes compensation info, Opus factors it in.
            </li>
            <li>
              <strong className="text-foreground">
                Your experience
              </strong>{' '}
              — The track record and domain expertise in your profile are
              weighted against the role requirements.
            </li>
            <li>
              <strong className="text-foreground">Seniority alignment</strong> —
              How the role&apos;s level compares to your current
              level, so the recommendation accounts for lateral moves vs.
              step-ups.
            </li>
          </ul>
        </div>

        <div className="flex items-start gap-2 text-sm leading-relaxed text-muted-foreground">
          <BadgeCheck className="mt-0.5 size-4 shrink-0 text-primary" />
          <p>
            The recommendation appears in the{' '}
            <strong className="text-foreground">Salary Strategy</strong> section
            under Company Intel, marked with an{' '}
            <strong className="text-foreground">Opus Advisor</strong> badge so
            you know it came from the premium model. Use it to understand your
            worth before applying or stepping into a negotiation.
          </p>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Auto CV & Cover Letter                                      */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={FileText} title="Auto CV & Cover Letter" id="auto-cv-cl" />

        <p className="text-base leading-relaxed text-muted-foreground">
          For every <strong className="text-foreground">HIGH MATCH</strong> job,
          the system automatically generates a tailored CV and cover letter —
          no manual work required.
        </p>

        <div className="space-y-2 text-sm leading-relaxed text-muted-foreground">
          <h3 className="text-lg font-semibold text-foreground">
            How it works
          </h3>
          <div className="space-y-1.5">
            <div className="flex items-start gap-2">
              <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
              <p>
                <strong className="text-foreground">Domain-specific base CV</strong>{' '}
                — The system picks from your domain-specific base CVs (one
                per field you target) based on which keyword category the job
                matched.
              </p>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
              <p>
                <strong className="text-foreground">AI-rewritten content</strong>{' '}
                — Claude Opus rewrites the professional summary to align with
                each specific job description, using only your verified metrics
                and real accomplishments. The experience bullets come from your
                own base CV.
              </p>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
              <p>
                <strong className="text-foreground">Quality judge</strong> — A
                separate AI audit checks the generated CV across five dimensions
                (hallucination, enrichment, keywords, formatting, and length)
                before saving. A green <strong className="text-foreground">CV QA</strong>{' '}
                badge means it passed; yellow means a quick review is recommended.
              </p>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
              <p>
                <strong className="text-foreground">Download anywhere</strong>{' '}
                — Generated documents are available from the Review tab (expand
                any job card), the Kanban board, and through the API. They
                download as Word (.docx) files, ready to edit.
              </p>
            </div>
          </div>
        </div>

        <p className="text-sm leading-relaxed text-muted-foreground">
          For <strong className="text-foreground">POTENTIAL</strong> jobs,
          documents are not generated automatically — you decide first whether
          the role is worth pursuing, then hit{' '}
          <strong className="text-foreground">Generate CV &amp; Cover Letter</strong>{' '}
          from the expanded job card or Kanban row when you are ready. Writing
          them takes about 5–10 minutes: the button says Generating… until
          they are ready, then View appears. You can leave the page meanwhile.
        </p>
        <p className="text-sm leading-relaxed text-muted-foreground">
          You can toggle auto-generation off in Settings if you prefer to
          generate documents selectively — for example, only for jobs you decide
          to track.
        </p>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Company Intel                                               */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={Building2} title="Company Intel" id="company-intel" />

        <p className="text-base leading-relaxed text-muted-foreground">
          Every matched job comes with a Company Intel panel — a dossier of
          background research that helps you decide whether to apply and how to
          position yourself. It has five sections:
        </p>

        <div className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          <div className="flex items-start gap-2">
            <Building2 className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Company Profile</strong> —
              What the company does, its industry, approximate size, funding
              stage, recent news, and culture notes. Gives you a snapshot so you
              know what you are walking into.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <Eye className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Hiring Health</strong> — Is
              this company really hiring? First, what our own scanner has seen:
              how many listings and distinct titles from this company in the
              last 90 days, and whether this exact title keeps being re-listed.
              Then dated, sourced public facts from the last 12 months: layoffs,
              hiring freezes or restructuring, funding news, the open-role
              trend, leadership changes, and layoffs or restructuring in the
              teams closest to the roles you target.
              Every line carries a date and a link; where nothing is found, it
              says so. Facts only — the call is yours.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <DollarSign className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Salary &amp; Compensation</strong>{' '}
              — Market rate data for the role pulled from job postings, Glassdoor,
              Levels.fyi, and other sources. Includes ranges, median figures, and
              any equity or bonus information when available.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <Users className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Hiring Intel</strong> — Who is
              hiring for this role: team structure, the likely hiring manager,
              recruiter names found on LinkedIn, and any recent hiring activity
              signals. Useful for personalized outreach.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <TrendingUp className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Salary Strategy</strong> — The{' '}
              <strong className="text-foreground">Opus Advisor</strong>{' '}
              recommendation tailored to this specific role. It synthesizes the
              market data, company pay range, and your profile into a
              concrete target number and negotiation guidance.
            </p>
          </div>
        </div>

        <div className="flex items-start gap-2 text-sm leading-relaxed text-muted-foreground">
          <Search className="mt-0.5 size-4 shrink-0 text-primary" />
          <p>
            Click <strong className="text-foreground">Research Company</strong>{' '}
            to fetch fresh intel from the web. Results are cached (30 days for
            company profiles, 14 days for salary data and hiring health) so
            repeat visits are instant.
          </p>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Is This Posting Real? (Phase 27c)                           */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={FileSearch} title="Is This Posting Real?" id="posting-check" />

        <p className="text-base leading-relaxed text-muted-foreground">
          Open a job and you&apos;ll see a short{' '}
          <strong className="text-foreground">Posting check</strong> above
          Company Intel: plain facts about the posting, each with where it came
          from. It is a summary of the facts, not a score — it never changes a
          job&apos;s match, its tier or its place in your feed.
        </p>

        <div className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          <div className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">The facts</strong> — whether
              the job is still on the company&apos;s own job board, how many days
              ago it was posted and when our scanner first saw it, whether the
              same title keeps being re-listed, a pay range that is unusually
              wide (the top is more than double the bottom), layoffs or hiring
              freezes reported in the last six months (with a link), and — just
              for your reference — your earlier applications at the same company
              and how far they got. The daily email shows the most telling of
              these in one short line under each job, plus how many people in
              your imported networks work there.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <Shield className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">&ldquo;No longer on the job
              board&rdquo;</strong> only comes from company job boards we can read
              in full (Greenhouse, Lever, Ashby and similar), and only after two
              complete checks at least six hours apart didn&apos;t find the job.
              If the company re-posted the same title under a new link, the
              job shows &ldquo;Still on the job board under a new link&rdquo;
              instead, with that link. Jobs from
              LinkedIn, Indeed and other job sites can&apos;t be checked this way,
              so they never show as gone. A gone job leaves your Review list and
              the email digest (nothing is deleted — it comes back if the board
              lists it again). Jobs on your Kanban board are never hidden; the
              card just says &ldquo;No longer on the company&apos;s job board
              since&hellip;&rdquo;.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <Eye className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">The overall read</strong> —{' '}
              <em>Looks active</em>, <em>Mixed signals</em> or{' '}
              <em>Check before investing time</em> — or <em>Not enough to go
              on</em> when the job has no posting date and came from a site we
              can&apos;t check (most LinkedIn finds). It is a summary of the
              facts about the posting itself (still listed, re-posted, how old),
              not a score. Company news such as layoffs is shown as its own
              dated fact but never changes the read: a big company can cut one
              team and still be hiring for yours. On the Review list, a small
              grey chip appears only for Mixed signals or Check first (hover it
              to see why); the email shows one short grey line under every job.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <FileSearch className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Read the posting</strong> —
              checks how specific the posting is: the tools it names, the team
              or who the role reports to, what the first months&apos; work looks
              like, the pay, anything that contradicts itself (say, an
              entry-level title asking for 8+ years), and how much of it is
              generic text. Every point is quoted word for word from the
              posting. Strong matches are read automatically after each search.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <Search className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              Companies are researched automatically (the Hiring Health part of
              Company Intel) for Strong matches after each search, and when you
              track a job — only the company&apos;s name is searched, nothing
              about you.
            </p>
          </div>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Jobs Tab                                                    */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={Briefcase} title="Jobs Tab" id="jobs-tab" />

        <p className="text-base leading-relaxed text-muted-foreground">
          This is where all new results from the latest search appear. Think of
          it as your inbox for job opportunities.
        </p>

        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-foreground">
            Match badges
          </h3>
          <div className="space-y-2 text-sm leading-relaxed text-muted-foreground">
            <div className="flex items-start gap-2">
              <Star className="mt-0.5 size-4 shrink-0 text-emerald-400" />
              <p>
                <strong className="text-foreground">HIGH MATCH</strong> — The AI
                is confident this role aligns strongly with your
                experience. These are your top picks.
              </p>
            </div>
            <div className="flex items-start gap-2">
              <Star className="mt-0.5 size-4 shrink-0 text-amber-400" />
              <p>
                <strong className="text-foreground">POTENTIAL</strong> — A
                promising match but maybe not a perfect fit in every dimension.
                Worth a look.
              </p>
            </div>
            <div className="flex items-start gap-2">
              <Star className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
              <p>
                <strong className="text-foreground">Other badges</strong> — The
                AI had lower confidence. These might still surface something
                interesting, especially from niche boards.
              </p>
            </div>
          </div>

          <h3 className="text-lg font-semibold text-foreground">
            Company Intel &amp; Documents
          </h3>
          <div className="space-y-2 text-sm leading-relaxed text-muted-foreground">
            <div className="flex items-start gap-2">
              <Building2 className="mt-0.5 size-4 shrink-0 text-primary" />
              <p>
                When you expand a job, you will see a{' '}
                <strong className="text-foreground">Company Intel</strong>{' '}
                panel with five sections — Company Profile, Hiring Health,
                Salary &amp; Compensation, Hiring Intel, and the Opus-powered Salary
                Strategy. See{' '}
                <a href="#company-intel" className="text-primary hover:underline">
                  Company Intel
                </a>{' '}
                and{' '}
                <a href="#opus-advisor" className="text-primary hover:underline">
                  Opus Salary Advisor
                </a>{' '}
                above for full details.
              </p>
            </div>
            <div className="flex items-start gap-2">
              <FileText className="mt-0.5 size-4 shrink-0 text-primary" />
              <p>
                HIGH MATCH jobs also get an auto-generated tailored CV and cover
                letter. See{' '}
                <a href="#auto-cv-cl" className="text-primary hover:underline">
                  Auto CV &amp; Cover Letter
                </a>{' '}
                for how the system picks templates, rewrites content, and
                quality-checks every document.
              </p>
            </div>
          </div>

          <h3 className="text-lg font-semibold text-foreground">
            What you can do here
          </h3>
          <ul className="space-y-1.5 text-sm leading-relaxed text-muted-foreground">
            <li className="flex items-start gap-2">
              <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
              Click any job card to see the full details, AI reasoning, and
              company intel.
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
              Hit <strong className="text-foreground">Track</strong> to add a
              job to your Kanban pipeline — it moves to &quot;To Apply&quot; automatically.
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
              Use the{' '}
              <strong className="text-foreground">Show All</strong> toggle to
              see everything, including jobs the AI filtered out. Handy if you
              want to double-check what was skipped.
            </li>
          </ul>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Review Tab                                                  */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={Eye} title="Review Tab" id="review-tab" />

        <p className="text-base leading-relaxed text-muted-foreground">
          This is your main workspace — every job that passed the AI filter
          lands here, waiting for your decision. Roles that the rules in your
          profile exclude are left out. Only fresh roles stay: Strong
          matches up to a week old and Worth a look up to three days old
          (counted from the posting date, or from the day we first saw the job).
          Early applicants get the most recruiter attention, so apply within a
          day when you can.
        </p>

        <div className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          <div className="flex items-start gap-2">
            <Shield className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">AI Reasoning</strong> — Click
              a job to expand it and see exactly <em>why</em> the AI thinks this
              role fits (or doesn&apos;t). It considers your experience,
              the role requirements, location, seniority, and domain
              match.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <Building2 className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Company Intel</strong> — See
              company background, hiring team details, and salary data.
              Click <strong className="text-foreground">Research Company</strong>{' '}
              to fetch fresh intel powered by web search.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Track</strong> moves the job
              into your application pipeline.{' '}
              <strong className="text-foreground">Dismiss</strong> archives it
              — you will be asked for a reason so the system can learn your
              preferences over time.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              You can select multiple jobs and batch Track or Dismiss them all
              at once. Efficient for busy mornings.
            </p>
          </div>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Kanban / ATS Tab                                            */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading
          icon={Kanban}
          title="Kanban Tab"
          id="kanban-tab"
        />

        <p className="text-base leading-relaxed text-muted-foreground">
          Your application pipeline, visualized. Every job you track appears
          here, and you can move it through the stages as you progress.
        </p>

        <h3 className="text-lg font-semibold text-foreground">
          Pipeline stages
        </h3>
        <div className="grid gap-2 sm:grid-cols-3">
          {[
            'To Apply',
            'Applied',
            'Screening Call',
            'Interview',
            'Final Interview',
            'Offer',
            'Accepted',
            'Rejected',
            'Ghosted',
            'I Withdrew',
          ].map((stage) => (
            <div
              key={stage}
              className="flex items-center gap-2 rounded-lg border border-border/50 bg-card px-3 py-2 text-sm font-medium text-foreground"
            >
              <ArrowRight className="size-3 text-primary" />
              {stage}
            </div>
          ))}
        </div>

        <div className="space-y-2 text-sm leading-relaxed text-muted-foreground">
          <p>
            Change a job&apos;s status with the dropdown on each card. Your
            changes save automatically.
          </p>
          <p>
            <strong className="text-foreground">Nothing is forgotten.</strong>{' '}
            Every move is kept as history. Open a card to see its line at the
            top — for example{' '}
            <em>Applied Mar 3 → Interview Mar 10 → Rejected Apr 2</em>. A
            card that got interviews and was later closed (Rejected, Ghosted
            or I Withdrew) keeps a small note under its status (
            <em>reached Interview · Mar 10</em>), and its interview prep sheet
            stays readable. A &quot;~&quot; marks a stage or date that is not
            confirmed yet — when a stage was inferred instead of recorded, the
            note says <em>reached an interview stage · ~Mar 10</em> until the
            real stage is confirmed. If you pick an interview stage by mistake,
            move the card straight back to where it was within ten minutes and
            it will not count. Moving it on instead — to Rejected, or to a
            later stage — keeps the interview on its record.
          </p>
          <p>
            <strong className="text-foreground">How far did it get?</strong>{' '}
            When you move a card to Rejected or Ghosted, the card opens and asks
            two things: the furthest stage the process reached (Applied,
            Screening Call, Interview, Final Interview or Offer) and, if you
            like, what happened — one line, or tap a ready answer such as{' '}
            <em>They chose another candidate</em>. If you ended it yourself,
            pick <strong className="text-foreground">I Withdrew</strong>{' '}
            instead: the same question, with your reasons to tap (
            <em>Pay didn&apos;t match</em>, <em>Took another offer</em>…).
            Your stage is what the card and the Stats page count, even if the
            card skipped a stage on the board, and it confirms any
            &quot;~&quot; stage. Both are optional: &quot;Not now&quot; keeps
            the card as it is. You can add or change it any time later — every
            closed card has <em>+ Add details</em> under its status (or click
            your line there to change it).
          </p>
          <p>
            <strong className="text-foreground">Did you send this CV?</strong>{' '}
            Once a card is Applied, the Documents panel asks which CV went out:
            ours as is, ours with your edits, or your own rewrite. It is one
            click, optional, and you can change the answer later. It helps
            interview prep talk about the CV the company actually saw.
          </p>
          <p>
            The <strong className="text-foreground">CV QA</strong> column tells
            you if the auto-generated CV passed the quality check. A green badge
            means it looks good; a yellow badge means you may want to review it
            before submitting.
          </p>
          <p>
            You will also find a section for{' '}
            <strong className="text-foreground">Quick Apply Answers</strong>{' '}
            — pre-filled responses for common application form fields (years of
            experience, work authorization, salary expectations, etc.) so you
            can copy-paste them quickly.
          </p>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Input Tab                                                   */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={PenLine} title="Input Tab" id="input-tab" />

        <p className="text-base leading-relaxed text-muted-foreground">
          Found a job on your own that is not from the automated search? This is
          where you add it manually.
        </p>

        <h3 className="text-lg font-semibold text-foreground">
          Full submission flow
        </h3>
        <ul className="space-y-1.5 text-sm leading-relaxed text-muted-foreground">
          <li className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <span>
              <strong className="text-foreground">1. Paste the job</strong> —
              Enter the title and company name. The job description is
              optional: paste it if there is one, and leave it empty when the
              company reached out to you directly. Optionally add location and
              a link to the original posting.
            </span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <span>
              <strong className="text-foreground">2. Hit Submit</strong> — The
              job is saved to the database and added to your Kanban board under
              &quot;To Apply&quot;.
            </span>
          </li>
          <li className="flex items-start gap-2">
            <Building2 className="mt-0.5 size-4 shrink-0 text-primary" />
            <span>
              <strong className="text-foreground">
                3. Intel auto-generated
              </strong>{' '}
              — The system immediately researches the company and produces the
              full Company Intel dossier (profile, salary data, hiring intel,
              and the Opus Salary Strategy recommendation). You will see the
              results right on the page.
            </span>
          </li>
          <li className="flex items-start gap-2">
            <FileText className="mt-0.5 size-4 shrink-0 text-primary" />
            <span>
              <strong className="text-foreground">
                4. CV &amp; CL auto-generated
              </strong>{' '}
              — A CV and cover letter are created using the right
              domain-specific base template, your real experience, and verified
              metrics, tailored to the job description you pasted. With no
              description, or only a few lines, you get the general version of
              your CV. The quality judge audits the output before saving. This
              takes about 5–10 minutes; the Kanban card says Generating… until
              the documents are ready.
            </span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <span>
              <strong className="text-foreground">5. Ready to apply</strong> —
              Everything is in place: intel for research, documents for
              submission, and the job is tracked on your Kanban. Just download
              and go.
            </span>
          </li>
        </ul>

        <p className="text-sm leading-relaxed text-muted-foreground">
          This is perfect for jobs shared by friends, found on company websites
          directly, or spotted on social media. You get the same full treatment
          (intel, documents, quality checks) as automated matches.
        </p>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Outreach Playbook                                            */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={Sparkles} title="Outreach Playbook" id="outreach" />
        <p className="text-base leading-relaxed text-muted-foreground">
          The moment you mark a card <strong className="text-foreground">Applied</strong>{' '}
          (use the <strong className="text-foreground">✓ Applied</strong> button on
          To&nbsp;Apply rows), the system researches the company in the background
          and builds a personal outreach playbook — it appears in the expanded
          card a few minutes later.
        </p>
        <div className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          <div className="flex items-start gap-2">
            <Building2 className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Company pulse</strong> — what
              actually happened at the company in the last 30 days (launches,
              news, community chatter), each item framed as a conversation hook.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <Shield className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Who to reach</strong> — likely
              hiring manager and recruiters from public sources only. Every name
              carries a citation and a confidence label; if nobody is public,
              you get the exact LinkedIn search to run instead. When a{' '}
              <strong className="text-foreground">verified email</strong> is
              found, it is marked as safe to use — unverified addresses are
              never shown as sendable.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Follow-up plan</strong> — dated
              checklist: day-2 intro note (never a status ask), day-7 follow-up,
              day-14 final value-add, day-21 close. Two unsolicited follow-ups is
              the ceiling — then let it go. Due follow-ups (up to five) also
              appear in the morning email digest, below the Strong matches.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Ready-to-edit drafts</strong> —
              a LinkedIn connection note and short follow-up messages, grounded
              only in your verified track record. Edit, then send from your own
              accounts — the system never sends anything itself.
            </p>
          </div>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Interview Prep                                               */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={GraduationCap} title="Interview Prep" id="interview-prep" />
        <p className="text-base leading-relaxed text-muted-foreground">
          When you move a card to{' '}
          <strong className="text-foreground">Screening Call</strong>,{' '}
          <strong className="text-foreground">Interview</strong>,{' '}
          <strong className="text-foreground">Final Interview</strong>, or{' '}
          <strong className="text-foreground">Offer</strong>, the system
          builds a prep sheet in the background — it appears in the expanded card
          within a minute. You can also hit{' '}
          <strong className="text-foreground">Prep me</strong> any time. When the
          card moves on to a later stage, the sheet is rebuilt for that stage;
          moving it back does not replace it. If the process ends (Rejected,
          Ghosted or I Withdrew), the sheet stays on the card to read.
        </p>
        <div className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          <div className="flex items-start gap-2">
            <MessageCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Likely questions</strong> — the
              5-7 things this specific posting will probe, each with a one-line
              angle on how to answer.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <Star className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Your real stories</strong> —
              your strongest STAR examples mapped to those questions, pulled only
              from your verified experience. Every number is real; nothing is
              invented, and honest gaps are named with how to bridge them.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <Building2 className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Company angle &amp; smart
              questions</strong> — why-this-company talking points from company
              intel, plus sharp questions to ask them back that show you
              understand the role.
            </p>
          </div>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Warm Paths                                                   */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={CheckCircle} title="Warm Paths" id="warm-paths" />
        <p className="text-base leading-relaxed text-muted-foreground">
          An amber panel on a Kanban card means someone in your imported
          networks works at that company — a referral ask beats any cold message.
        </p>
        <div className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          <div className="flex items-start gap-2">
            <Shield className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              Connections come from the official LinkedIn exports you import
              (one or more networks) — nothing is ever scraped from LinkedIn.
              When a connection comes from another person&apos;s export, the play is
              simple: <strong className="text-foreground">ask that person to
              make the intro</strong>. The playbook will say the same.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              The most relevant contacts and recruiters rank first, so at a big
              employer the useful contact tops the list. A note about &quot;fuzzy matches
              needing a look&quot; means similar-but-unconfirmed company names —
              treat those as maybes.
            </p>
          </div>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Stats Tab                                                    */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={Kanban} title="Stats Tab" id="stats-tab" />
        <p className="text-base leading-relaxed text-muted-foreground">
          The pipeline at a glance, refreshed after every run and every action.
        </p>
        <div className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          <div className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">This week&apos;s numbers</strong>{' '}
              with up/down arrows vs last week, and the{' '}
              <strong className="text-foreground">funnel</strong>: surfaced →
              reviewed → applied → in process → offers, with conversion between
              each stage. The funnel is where progress shows up as a number.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Daily rhythm</strong> — roles
              per day (green = strong) with your applications on the same
              timeline, plus where all applications currently stand and what
              kinds of roles are coming in.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Reached interview (all time)</strong>{' '}
              — how many applications ever got to a screening call or later.
              It comes from each card&apos;s history, so a process that has
              since closed still counts. The funnel&apos;s &quot;in
              process&quot; number only shows the ones that are live right
              now.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">How processes ended</strong>{' '}
              — the interview processes that closed (Rejected, Ghosted or I
              Withdrew), by the last stage they reached, how many you withdrew
              from yourself, and the &quot;what happened&quot; lines you added
              on those cards, newest first.
            </p>
          </div>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Email Digests                                               */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={Mail} title="Email Digests" id="email-digests" />

        <p className="text-base leading-relaxed text-muted-foreground">
          After each search run, the system sends an email digest to the
          recipients set in your configuration — one per scheduled run. The
          greeting adjusts to the time of day automatically.
        </p>

        <div className="space-y-2 text-sm leading-relaxed text-muted-foreground">
          <h3 className="text-lg font-semibold text-foreground">
            What is in the email
          </h3>
          <ul className="ml-5 list-disc space-y-1">
            <li>
              <strong className="text-foreground">Match badges</strong> — Each
              job is labeled HIGH MATCH, POTENTIAL, or lower, with a confidence
              score.
            </li>
            <li>
              <strong className="text-foreground">Intel previews</strong> — For
              every matched job, the email includes a snapshot of the company
              profile and salary data so you can triage without opening the
              dashboard.
            </li>
            <li>
              <strong className="text-foreground">AI reasoning</strong> — A
              brief summary of why each role matched your background.
            </li>
            <li>
              <strong className="text-foreground">Posting check</strong> — Jobs
              the company&apos;s own job board no longer lists are left out, and
              a Strong match that needs a check gets one short grey line saying
              why (see{' '}
              <a href="#posting-check" className="text-primary underline">
                Is This Posting Real?
              </a>
              ).
            </li>
            <li>
              <strong className="text-foreground">Zero-result days</strong> — If
              nothing new was found, you still receive a &quot;Nothing new today&quot;
              email so you know the system ran successfully.
            </li>
          </ul>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Logs Tab                                                    */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={ScrollText} title="Logs Tab" id="logs-tab" />

        <p className="text-base leading-relaxed text-muted-foreground">
          Curious about what happened during the last search? The Logs tab shows
          you.
        </p>

        <div className="space-y-2 text-sm leading-relaxed text-muted-foreground">
          <p>You can filter by stage to focus on what interests you:</p>
          <ul className="ml-5 list-disc space-y-1">
            <li>
              <strong className="text-foreground">Stage 1</strong> — The quick
              filter. Shows which jobs were instantly accepted or rejected based
              on title, location, and keywords.
            </li>
            <li>
              <strong className="text-foreground">Stage 2</strong> — Detailed
              grading. The AI reads the full job description and scores it
              against your background.
            </li>
            <li>
              <strong className="text-foreground">Stage 3</strong> — The final
              judge. A more powerful AI model makes the ultimate call, sometimes
              enriched with live company research.
            </li>
          </ul>
          <p>
            This is useful if you are wondering why a specific job was filtered
            out, or if you want to see how many listings were processed in a
            run.
          </p>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Settings                                                    */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading icon={Settings} title="Settings" id="settings" />

        <p className="text-base leading-relaxed text-muted-foreground">
          Click the gear icon in the top-left corner of the sidebar to open
          Settings. Here is what each option does:
        </p>

        <div className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          <div className="flex items-start gap-2">
            <Clock className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">AI Models</strong> — Which AI
              models are used at each filtering stage. The defaults are already
              tuned for the best balance of speed and accuracy. You generally do
              not need to change these.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <Search className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Search Sources</strong> —
              Toggle which job boards to include: LinkedIn, Indeed, Glassdoor,
              ZipRecruiter, ATS company boards, and industry niche boards.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <FileText className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">
                Auto-generate CV/CL
              </strong>{' '}
              — When ON (the default), the system creates a tailored CV and
              cover letter for every matched job automatically. Turn it off if
              you want to generate documents selectively.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <DollarSign className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">AI Prompts</strong> — View the
              actual instructions the AI uses to filter and grade jobs. These
              are read-only for transparency — so you can see exactly how your
              background is being represented.
            </p>
          </div>
        </div>
      </section>

      <Divider />

      {/* ============================================================ */}
      {/*  Tips                                                        */}
      {/* ============================================================ */}
      <section className="space-y-4">
        <SectionHeading
          icon={Lightbulb}
          title="Tips for Getting the Most Out of It"
          id="tips"
        />

        <div className="grid gap-3 sm:grid-cols-2">
          {[
            {
              title: 'Check Review daily',
              body: 'New jobs appear after each scheduled search run. Apply within a day when you can: a role marked "apply today" was posted in the last 24 hours, and older roles drop off after three days (Worth a look) or a week (Strong).',
            },
            {
              title: 'Research before applying',
              body: 'Use "Research Company" on any job to pull fresh company intel. This can help you tailor your outreach and prepare for interviews.',
            },
            {
              title: 'Use salary data in negotiations',
              body: 'When available, salary ranges are pulled from job postings and company intel. Knowing the range gives you leverage.',
            },
            {
              title: 'Review your tailored CV',
              body: 'Your CV is auto-customized for each role using your real metrics and experience. Download it, give it a quick read, and submit with confidence.',
            },
            {
              title: 'Add jobs you find yourself',
              body: 'The Input tab is there for roles you discover through networking, social media, or direct company websites. Paste the description when there is one and your documents are tailored to it; with none, you get the general version of your CV.',
            },
            {
              title: 'Dismiss with a reason',
              body: 'When you pass on a job, selecting a reason helps the system understand your preferences better over time.',
            },
          ].map((tip) => (
            <Card key={tip.title} className="border-border/50">
              <CardContent className="p-4">
                <h3 className="mb-1 text-sm font-semibold text-foreground">
                  {tip.title}
                </h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {tip.body}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* ============================================================ */}
      {/*  Footer                                                      */}
      {/* ============================================================ */}
      <Divider />
      <div className="text-center text-sm text-muted-foreground">
        <p>
          Built with care to help you find the perfect next role.
          <br />
          You&apos;ve got this.
        </p>
      </div>
    </div>
  )
}
