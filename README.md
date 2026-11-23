# AI Job Agent

An automated job-search pipeline. It scrapes four job boards, about 260
company ATS boards and a set of niche boards, puts every posting through a
three-stage LLM cascade, tiers what survives, writes a tailored CV and cover
letter for the strongest matches, and emails a digest. Everything it knows
about *who* is job hunting lives in a single YAML file, so it runs for anyone
who writes one.

Built originally for my wife's job search, then generalised. The parts that
were about her are now configuration; the parts that were about *job hunting*
stayed code.

```
resume + keywords
  → scrape (4 job boards, ~260 ATS boards, niche boards, RSS)
  → dedupe → pre-filter (location, exclusions, staleness)
  → Stage 1 plausibility gate   ·  lean-YES, removes most of the volume
  → Stage 2 recruiter grading   ·  batched, rejects are final
  → Stage 3 judge               ·  premium model, borderline cases only
  → tier: strong / worth a look / rejected
  → comp floor → CV + cover letter → email digest
  → review UI → kanban → post-apply outreach playbook
```

---

## Why it is built this way

Most of the interesting decisions came from things going wrong in production.

**A cascade, because filtering everything with a good model is unaffordable.**
A run scrapes thousands of postings. Sending each to a frontier model costs
more than the job search is worth. So: the first stage answers one binary
question ("could this *plausibly* be a role in the target field?") and
eliminates most of the volume; the second does real recruiter grading on the
survivors, in batches; only genuinely borderline cases reach the expensive
judge, and that stage is capped per run. Cost per run lands in single-digit
dollars rather than triple.

**Recall mode, because a rejected job is invisible forever.** Early versions
dropped anything that was not an obvious match, and good roles vanished
silently. Now the third stage cannot delete — it can only *tier*. Borderline
roles land in "worth a look" under an honest label. Only an explicit REJECT
removes something, and the tier column is the single axis the feed groups on.

**Verdict-binding, because judges hedge.** The recurring failure was a model
naming a disqualifier and then keeping the job anyway — "this is an
org-seniority Director role… worth a look." Every reject rule now ends with an
explicit clause saying the softer neighbouring rule does not apply, and the
prompts state that a verdict must follow its own reasoning. Before this, a
large share of the "worth a look" tier was hedged rejections.

**A verified-metrics bank, because generators invent numbers.** CV and cover
letter prompts may use *only* the figures in your profile's metrics block, and
the profile pins one canonical years-of-experience figure. Without the pin,
documents drift from one figure to another; with it, every document agrees.
The quality judge also looks for composition fraud — claims where every
individual fact is true but the combination is not.

**A comp floor that judges the ceiling.** A role whose pay *ceiling* falls
below the floor is removed, not demoted — a demoted role still shows up in
the feed, which defeats the point. Roles that state no pay are never
penalised. On-site roles in the secondary metro can be held to a higher
floor.

**Posting text is data, not instructions.** Job descriptions increasingly
carry sentences addressed to AI tools ("include the word X in your
application"). Every prompt that embeds a posting runs it through a sanitiser
first and tells the model that the posting is untrusted.

**Observability, because "why did it reject that?" needs an answer.** With
LangFuse configured, every posting produces a trace: one span per stage, each
model call logged with its prompt, completion, token usage and cost, and the
stage verdict recorded as a score carrying the model's stated reason. You can
open any rejected posting weeks later and read exactly what the model saw and
said.

---

## Quick start

```bash
git clone https://github.com/realler13/ai-job-agent.git
cd ai-job-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install

cp profiles/example.yaml profiles/profile.yaml   # gitignored
$EDITOR profiles/profile.yaml                    # make it yours

cp env.example .env                              # add API keys
```

Put your resume (`resume.pdf`) and a base CV (`base_cv.docx`) in the
repository root; both are gitignored. Then run a search:

```bash
PYTHONPATH=src python -m job_agent.main \
  --resume resume.pdf --base-cv base_cv.docx --cascade --limit 20
```

Or bring up the web UI — FastAPI on `:8502`, Next.js on `:8503`:

```bash
(cd web && npm install)
./start_dev.sh --prod
```

The resume path the web UI uses is in its Settings panel. Without `--prod`
the frontend runs as a dev server on `:3000`; add that origin
to `CORS_ALLOWED_ORIGINS` (see `env.example`), which defaults to
`http://localhost:8503`.

You need an `ANTHROPIC_API_KEY` for the default models (Gemini and OpenAI
models work as overrides and fallbacks — the router dispatches on the model
name). Everything else is optional: Perplexity adds company and role context,
LangFuse adds tracing (`ops/langfuse/`), SMTP credentials enable the email
digest.

No scheduler is shipped. The pipeline is one command, so cron, launchd or a
systemd timer can run it as often as you like.

---

## The profile is the whole interface

Everything about the person lives in `profiles/profile.yaml`. The shipped
`profiles/example.yaml` describes a fictional candidate (a video producer in
Seattle) and is what the agent falls back to when you have not written your
own.

| Section | What it controls |
|---|---|
| `candidate` | identity, headline, pronouns, the canonical years figure |
| `paths` | optional interview notes and career source bank used as fact sources |
| `search` | target job titles by category, exclusions, locations, staleness |
| `metrics_text` | the only numbers generators may use |
| `rules` | the judgement policy: target roles, seniority, craft, industry, rejects |
| `application_answers` | boilerplate for application forms |
| `domains` | role categories and the base CV each one routes to |

`rules` is the interesting one — it is the prompt tuning, in plain English,
versioned alongside your profile. The shipped example encodes a creative-
production search; rewriting it for a different field is a text edit, not a
code change.

**Location rules come from the profile.** The profile supplies your home
location and state, the primary and secondary metro terms, the
allowed-location list and the two pay floors; the example profile points
them at Seattle and Portland. A posting in your primary metro ranks first,
remote second, the rest of your home state goes to the judge, and other
states and countries are dropped. A town name that also exists elsewhere is
written with its state (`vancouver, wa`) and then counts only next to that
state.

See [docs/CONFIGURING.md](docs/CONFIGURING.md) for the field-by-field guide.

---

## What is in the box

- **Scrapers** — LinkedIn, Indeed, Glassdoor, ZipRecruiter via `python-jobspy`
  and browser-based fallbacks; about 260 company ATS boards, with scrapers
  for 13 platforms (Greenhouse, Lever, Ashby, SmartRecruiters, Workday,
  Recruitee, Workable, Teamtailor, Eightfold, BambooHR, Comeet, Phenom and a
  bespoke Apple scraper); niche industry boards, remote boards, RSS feeds,
  Reddit job subreddits and Google Jobs. The board list in `config.py` is
  tuned to media and entertainment employers — replace it with yours. Every
  network call is wrapped in a timeout, so one hung connection cannot stall a
  run, and one broken board is skipped instead of ending the ATS phase.
- **Circuit breakers for throttling sites** — LinkedIn throttles logged-out
  searches on some days: they hang to the timeout or come back empty. After a
  run of bad searches in a row (`SITE_BREAKER_TRIP` in `config.py`) the run
  skips LinkedIn's remaining searches, with one probe search per pass so a
  LinkedIn that recovers is picked up again; one summary line goes to the
  log. Glassdoor's posting pages and the Workday, Apple and Built In boards
  have their own stops and time budgets.
- **Posting text for list-only boards** — Glassdoor, ZipRecruiter and several
  niche boards list a job without its posting. For the jobs that make a tier,
  the text is looked up before the documents and the digest: another saved
  copy of the same job first, then the posting page (its schema.org
  JobPosting, or the largest block of text). A posting that still has too
  little text gets the general CV and the templated cover letter, with no
  model call, and the web UI says why.
- **Cross-source dedup** — the same job from LinkedIn and an aggregator has
  different URLs and slightly different company spellings. Company names are
  compared as normalised token tuples with corporate suffixes stripped, and a
  brand-prefix match needs corroborating location before it collapses two rows.
  LinkedIn's copy wins and inherits structured fields from the copy it replaces.
- **The cascade** — Stage 1 and Stage 2 run on `claude-sonnet-5`; the Stage 3
  judge is `claude-fable-5-2` with `claude-fable-5-1` as its first fallback.
  Each stage has a fallback chain, and a token-bucket limiter paces calls
  under the provider's rate limits. `scripts/replay_stage3_models.py` replays
  the judge's own recent cases against a new model before you switch.
- **Documents** — `claude-opus-5-5` by default. Domain-specific base CVs
  chosen by role category; the base's audited bullets ship as they are, and
  the summary is the only model-written text that can ship, through a
  deterministic fact gate. Cover letters get deterministic checks, a word cap
  and a templated fallback. A length gate estimates rendered height without
  rendering to keep the CV at two pages.
- **"Is this posting real?"** — facts, never a score: whether the posting is
  still on the company's own board, how old it is, how often it was
  re-listed, how wide the pay band is, and dated hiring-health notes for the
  company.
- **Review UI** — a Next.js app with a tiered review queue, a kanban board
  with per-application history and outcomes, and stats.
- **Post-apply** — when a role moves to Applied, the agent builds an outreach
  playbook: what is happening at the company, who to contact (cite-or-omit —
  a name appears only with a source URL), a follow-up cadence and grounded
  drafts. It never sends anything.

## Tests

About 2,550 tests, all written around a fictional candidate: Marisol
Vantongeren, a video producer in Seattle (`tests/fixtures/test_profile.yaml`).
`tests/conftest.py` pins that profile before anything else loads, switches
tracing off and nulls the Anthropic client, so a model call a test forgot to
mock returns nothing instead of making a paid request. Run the suite from a
scratch directory, so files the tests write stay out of the tree:

```bash
REPO="$PWD"                      # start in the repository root
mkdir -p /tmp/job-agent-tests && cd /tmp/job-agent-tests
PYTHONPATH="$REPO/src" python -m pytest --rootdir="$REPO" -p no:cacheprovider -q "$REPO/tests"
```

[tests/fixtures/README.md](tests/fixtures/README.md) describes the persona
and the one rule for new test data: fictional people and obviously fictional
companies only.

## Cost and honesty

Budget roughly a few dollars per run at this source count —
dominated by Stage 2, with Stage 3 capped per run. Scraping public job boards
is rate-limited and polite, and the agent uses only logged-out access; the
LinkedIn connection import expects an official CSV self-export.

This is a personal project shaped by one search in one field. It works well
for creative-production roles in a US metro. Pointed at a different field it
will need its `rules` block and its board list rewritten — that is the
intended way to change it, but it is real work, not a checkbox.

## License

MIT — see [LICENSE](LICENSE).
