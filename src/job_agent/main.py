
import click
import logging
import pandas as pd
import signal
import threading
import time
import os
import re
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load .env BEFORE importing .config — config.py reads PERPLEXITY_API_KEY (and
# other keys) at import time, so USE_PERPLEXITY would be False if .env loaded
# later. A scheduled CLI run usually does not export .env, so this is
# the only thing that activates Perplexity enrichment in the CLI run.
load_dotenv()

from jobspy import scrape_jobs
from .config import KEYWORDS, SEARCH_SITES, SITE_MIN_GAP_SEC, ALLOWED_LOCATIONS, EXCLUDE_PHRASES, KEYWORD_TO_CATEGORY, AUTO_GENERATE_CV_CL, AUTO_SEND_EMAIL, EMAIL_RECIPIENTS, REMOTE_SECOND_PASS, REMOTE_PASS_SITES, REMOTE_PASS_KEYWORDS, BROWSER_SCRAPE_KEYWORDS, SECONDARY_SECOND_PASS, SECONDARY_PASS_LOCATION, SECONDARY_PASS_KEYWORDS_EXTRA, DEFAULT_GENERATION_MODEL
from .config import SITE_BREAKER_TRIP
from .candidate_profile import SEARCH
from .utils import (setup_logging, load_proxies,
                    company_tokens, location_key, dedup_scraped_jobs)

from .gpt_filter import load_profile_text, batch_job_relevance_check
from .postprocess import send_email_with_results, send_error_alert
from .db_manager import DBManager
from .session import SessionManager

from job_agent.glassdoor_crawler import enrich_jobs_with_crawl4ai
import job_agent.ziprecruiter_scraper as ziprecruiter_scraper

# Search geography comes from the candidate profile (search.locations), never
# from literals in this module: the regular scrape pass searches the home
# location, and the primary-metro label is the fallback when a job row carries
# no location of its own.
PRIMARY_PASS_LOCATION = SEARCH["locations"]["home"]
PRIMARY_METRO_LABEL = SEARCH["locations"]["primary_metro_label"]


def _auto_enrich_matched_jobs(relevant_df, profile_text, db, progress_callback, logger):
    """Enrich HIGH_MATCH / POTENTIAL_MATCH jobs with Perplexity intel + Opus salary advisor.

    Non-blocking: any failure logs a warning and continues.
    Guarded by AUTO_ENRICH_MATCHES and USE_PERPLEXITY config flags.
    """
    from .config import AUTO_ENRICH_MATCHES, USE_PERPLEXITY
    if not AUTO_ENRICH_MATCHES or not USE_PERPLEXITY:
        return

    try:
        from .perplexity_utils import (
            get_company_profile,
            get_hiring_intel,
            get_salary_intel,
            generate_salary_recommendation,
        )
    except ImportError as e:
        logger.warning(f"Auto-enrich: import failed: {e}")
        return

    total = len(relevant_df)
    logger.info(f"Auto-enriching {total} matched jobs with company intel + salary strategy...")
    if progress_callback:
        progress_callback(f"Enriching {total} matched jobs with company intel...", 0.91)

    for idx, (_, row) in enumerate(relevant_df.iterrows(), 1):
        job = row.to_dict()
        company = job.get("company", "Unknown")
        title = job.get("title", "Unknown")
        location = job.get("location", PRIMARY_METRO_LABEL)

        try:
            # 1. Perplexity: company profile, hiring intel, salary intel
            profile = get_company_profile(company, db=db)
            hiring = get_hiring_intel(company, title, db=db)
            salary = get_salary_intel(title, company, location, db=db)

            # 2. Opus salary recommendation
            intel_dict = {
                "company_profile": profile,
                "salary_intel": salary,
                "hiring_intel": hiring,
            }
            generate_salary_recommendation(
                job=job,
                company_intel=intel_dict,
                profile_text=profile_text,
                db=db,
            )

            if progress_callback:
                pct = 0.91 + (idx / total) * 0.03
                progress_callback(f"Enriched {idx}/{total}: {company}", pct)
        except Exception as e:
            logger.warning(f"Auto-enrich failed for {title} at {company}: {e}")
            continue

    logger.info(f"Auto-enrichment complete for {total} jobs")


# Clock for the time budgets below (a module name so tests can move time).
_clock = time.monotonic


def _strong_rows_by_confidence(strong_df):
    """Strong rows, highest confidence first (NaN confidence last)."""
    if "confidence" in strong_df.columns:
        return strong_df.sort_values("confidence", ascending=False, na_position="last")
    return strong_df


def _auto_hiring_health(strong_df, db, logger, should_stop=None) -> dict:
    """Phase 27c: research each Strong job's company ("Hiring health":
    layoffs, freezes, funding, open-role trend, leadership and creative-team
    changes — dated, sourced facts) so the panel is ready when the candidate opens it.

    One call per distinct company not researched within its 14-day TTL, at
    most AUTO_HIRING_HEALTH_MAX_COMPANIES per run, and no new call once
    AUTO_HIRING_HEALTH_BUDGET_SEC have passed (each call already has its own
    hard deadline), nor once `should_stop()` is true (the Stop button). The
    search carries the company name only. Facts, never a score: nothing here
    touches tier, confidence or ordering. Never raises.
    """
    from .config import (AUTO_HIRING_HEALTH, AUTO_HIRING_HEALTH_BUDGET_SEC,
                         AUTO_HIRING_HEALTH_MAX_COMPANIES, USE_PERPLEXITY)
    counts = {"checked": 0, "cached": 0, "skipped": 0, "failed": 0}
    if not AUTO_HIRING_HEALTH or not USE_PERPLEXITY or strong_df is None or strong_df.empty:
        return counts
    from . import perplexity_utils

    started = _clock()
    seen: set = set()
    for _, row in _strong_rows_by_confidence(strong_df).iterrows():
        raw = row.get("company")
        company = str(raw).strip() if isinstance(raw, str) else ""
        key = company.lower()
        if key in seen:
            continue
        seen.add(key)
        if not company or key in ("unknown", "nan"):
            counts["skipped"] += 1
            continue
        try:
            if db.is_company_ignored(company):
                counts["skipped"] += 1
                continue
            if perplexity_utils.hiring_health_is_fresh(company, db):
                counts["cached"] += 1
                continue
        except Exception as e:
            logger.warning(f"Auto hiring health: cache check failed for {company}: {e}")
            counts["skipped"] += 1
            continue
        if counts["checked"] + counts["failed"] >= AUTO_HIRING_HEALTH_MAX_COMPANIES:
            counts["skipped"] += 1
            continue
        if (_clock() - started >= AUTO_HIRING_HEALTH_BUDGET_SEC
                or (should_stop is not None and should_stop())):
            counts["skipped"] += 1
            continue
        try:
            perplexity_utils.get_hiring_health(company, db=db)
            counts["checked"] += 1
        except Exception as e:
            counts["failed"] += 1
            logger.warning(f"Auto hiring health failed for {company}: {e}")
    logger.info(
        f"Auto hiring health: {counts['checked']} researched, {counts['cached']} already fresh, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )
    return counts


def _auto_posting_quality(strong_df, db, logger, should_stop=None) -> dict:
    """Phase 27c: one cheap "how specific is this posting?" read per Strong
    job (posting_check.run_quality_reads — cited facts only, cached 30 days,
    capped per run and in wall-clock time). Facts for the job panel; never
    feeds tier, confidence or ordering. Never raises."""
    from .config import POSTING_QUALITY_BUDGET_SEC, POSTING_QUALITY_MAX_PER_RUN
    empty = {"fresh": 0, "cached": 0, "skipped": 0, "failed": 0}
    if strong_df is None or strong_df.empty:
        return empty
    job_ids: list[int] = []
    for _, row in _strong_rows_by_confidence(strong_df).iterrows():
        raw_url = row.get("job_url")
        raw_link = row.get("link")
        link = ((raw_url if isinstance(raw_url, str) and raw_url else None)
                or (raw_link if isinstance(raw_link, str) and raw_link else None))
        job_id = db.get_job_id_by_link(link) if link else None
        if job_id and job_id not in job_ids:
            job_ids.append(job_id)
    if not job_ids:
        return empty
    from . import posting_check
    out = posting_check.run_quality_reads(
        db, job_ids, cap=POSTING_QUALITY_MAX_PER_RUN, budget_sec=POSTING_QUALITY_BUDGET_SEC,
        should_stop=should_stop,
    )
    logger.info(f"Posting quality reads: {out}")
    return out


def _refresh_listing_status(db, logger, where: str) -> None:
    """Recompute the "still on the company job board?" status (Phase 27c).
    Non-blocking: a failure is logged and the run goes on."""
    try:
        counts = db.refresh_listing_status()
        logger.info(f"Listing status ({where}): {counts}")
    except Exception as e:
        logger.warning(f"Listing status refresh failed ({where}): {e}")


class _ScrapeTimeout(Exception):
    """Raised when a scraping call exceeds its time budget."""


def _run_in_thread_with_timeout(func, args, kwargs, timeout_sec, label):
    """Thread timeout fallback for contexts where SIGALRM is unavailable
    (non-main threads: the FastAPI background search AND the parallel jobspy
    workers).

    Raw DAEMON thread, not ThreadPoolExecutor: executor threads are
    non-daemon and Python joins them at interpreter exit, so one permanently
    hung scrape would let the pipeline finish its work and then hang the
    process at shutdown. A daemon thread may linger after a timeout (Python
    can't kill threads) but never blocks exit — the pipeline advances, which
    is the property that matters.
    """
    result: dict = {}

    def _target():
        try:
            result["value"] = func(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 — must surface into the caller
            result["error"] = e

    t = threading.Thread(target=_target, daemon=True, name=f"scrape-{label[:30]}")
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        raise _ScrapeTimeout(f"{label} timed out after {timeout_sec}s (thread abandoned)")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _run_with_timeout(func, args=(), kwargs=None, timeout_sec=120, label="operation",
                      force_thread=False):
    """Run *func* with a hard time budget.

    Main thread: SIGALRM (interrupts even hung C calls). Non-main threads
    (the FastAPI background search) and platforms without SIGALRM: a
    thread-pool fallback — previously this case silently ran with NO timeout,
    so every UI-triggered search was one hung connection away from freezing
    until a service restart (the documented Phase 10e incident, unprotected
    on the primary interface).

    force_thread: use the daemon-thread timeout even on the main thread.
    REQUIRED for asyncio-heavy scrapers (Scrapling/ZipRecruiter, crawl4ai):
    a SIGALRM-raised exception inside an event-loop callback gets swallowed
    by asyncio's exception handling and the loop keeps spinning — one run
    spun at full CPU for hours inside solve_cloudflare with the 340s alarm
    long expired. A daemon thread is abandoned regardless of
    what the code inside catches.
    """
    kwargs = kwargs or {}
    if force_thread or not hasattr(signal, "SIGALRM"):
        return _run_in_thread_with_timeout(func, args, kwargs, timeout_sec, label)

    def _handler(signum, frame):
        raise _ScrapeTimeout(f"{label} timed out after {timeout_sec}s")

    try:
        old_handler = signal.signal(signal.SIGALRM, _handler)
    except ValueError:
        # Non-main thread (FastAPI background search) — SIGALRM unavailable.
        return _run_in_thread_with_timeout(func, args, kwargs, timeout_sec, label)
    signal.alarm(timeout_sec)
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Columns to include in CSV exports
EXPORT_COLUMNS = [
    'title', 'company', 'location', 'job_url', 'date_posted',
    'description', 'match_level', 'confidence', 'gpt_relevance', 'gpt_reasoning',
]


class _SiteBreaker:
    """Per-run circuit breaker for one jobspy site (LinkedIn).

    LinkedIn throttles logged-out clients on some runs: searches hang to the
    120 s limit or come back empty, and the LinkedIn loop can run for hours
    for almost nothing while the digest waits. After `trip_after` bad
    searches IN A ROW (timed out, or zero results) the breaker opens and the
    site's searches are skipped. A good search resets the count, so a
    slow-but-working day (scattered timeouts between good results) never
    trips it. One instance is shared by the primary, remote and secondary
    passes.

    Recovery: LinkedIn sometimes comes back after a long bad streak and then
    returns good searches for the rest of the run. An open breaker therefore
    lets ONE probe search through at the start of each later pass and after
    every `probe_every` skipped searches within a pass. A good probe closes it
    (count reset); a bad probe keeps it open. A dead day costs at most a few
    probes of 120 s each.

    A lock guards the state, so it stays correct if a site's searches ever run
    in more than one worker thread.
    """

    def __init__(self, site: str, trip_after: int, probe_every: int = 10):
        self.site = site
        self.trip_after = max(1, int(trip_after))
        self.probe_every = max(1, int(probe_every))
        self._lock = threading.Lock()
        self._streak = 0
        self.tripped = False        # open right now
        self.tripped_at = ""        # "label/'keyword'" of the search that first opened it
        self.skipped: dict = {}     # pass label -> searches skipped
        self.trips = 0
        self.recoveries = 0
        self.probes = 0
        self._label = None          # pass seen by the last allow()
        self._since_probe = 0       # skips in this pass since the trip / last probe
        self._skipped_open = 0      # skips since this opening (for the recovery line)
        self.recovered_after = 0    # skips before the last recovery
        self._probing = False
        self._reported = False

    def allow(self, label: str) -> bool:
        """True if the next search may run (normal, or a probe of an open
        breaker); counts it as skipped otherwise."""
        with self._lock:
            new_pass = label != self._label
            if new_pass:
                self._label = label
                self._since_probe = 0
            if not self.tripped:
                return True
            if self._probing:
                pass                        # one probe at a time
            elif new_pass or self._since_probe >= self.probe_every:
                self._since_probe = 0
                self._probing = True
                self.probes += 1
                return True
            self.skipped[label] = self.skipped.get(label, 0) + 1
            self._since_probe += 1
            self._skipped_open += 1
            return False

    def record(self, ok: bool, where: str = ""):
        """Record one finished search. Returns "tripped" on the call that opens
        the breaker, "recovered" on the good probe that closes it, else None."""
        with self._lock:
            if self.tripped:
                if not self._probing:
                    return None
                self._probing = False
                if not ok:
                    return None             # bad probe: stays open
                self.tripped = False
                self._streak = 0
                self.recoveries += 1
                self.recovered_after = self._skipped_open
                return "recovered"
            if ok:
                self._streak = 0
                return None
            self._streak += 1
            if self._streak >= self.trip_after:
                self.tripped = True
                self.trips += 1
                if not self.tripped_at:
                    self.tripped_at = where
                self._since_probe = 0
                self._skipped_open = 0
                return "tripped"
            return None

    def report(self, logger) -> None:
        """The ONE summary line per run, logged after the site's last pass so
        the counts are exact: a WARNING when the breaker tripped, one INFO
        "not tripped" line otherwise. Both are meant to be read by a log
        check: keep "<site> circuit breaker:", "tripped N time(s)", "N <site>
        searches skipped", "N recovery" and "not tripped" in them."""
        with self._lock:
            if self._reported:
                return
            self._reported = True
            if not self.trips:
                logger.info(f"{self.site} circuit breaker: not tripped this run")
                return
            total = sum(self.skipped.values())
            parts = ", ".join(f"{k} {v}" for k, v in self.skipped.items()) or "none"
            state = "still open at the end" if self.tripped else "closed at the end"
            trips, recoveries, probes = self.trips, self.recoveries, self.probes
        logger.warning(
            f"{self.site} circuit breaker: tripped {trips} time(s) after {self.trip_after} "
            f"searches in a row timed out or returned nothing (first at {self.tripped_at}); "
            f"{total} {self.site} searches skipped ({parts}); {probes} probe search(es), "
            f"{recoveries} recovery(ies), {state}. The site was throttling these searches; "
            f"the other sites and phases ran as usual."
        )


def _site_breakers() -> dict:
    """Fresh breakers for one run (state must never carry across runs: the
    FastAPI process runs many searches)."""
    return {site: _SiteBreaker(site, n) for site, n in SITE_BREAKER_TRIP.items()}


def _parallel_jobspy_pass(
    sites, keywords, kwargs_for, *,
    session, control_flags, progress_callback, db, all_jobs, limit,
    label, base_pct, logger, breakers=None,
):
    """Run jobspy sites CONCURRENTLY — one worker thread per site, each
    walking the keyword list sequentially for its own site.

    Why this shape: the sequential site-alternating loops took most of a
    whole run. Per-site workers make
    wall-clock ≈ max(per-site time) instead of the sum, while each site
    still sees a strictly sequential request stream. SITE_MIN_GAP_SEC is now
    ACTUALLY enforced per site (it was imported-but-dead config; the old
    loop's pacing came incidentally from alternation) — it is the tuning
    knob if a site starts throttling under the faster cadence.

    Timeouts: workers are non-main threads, so _run_with_timeout transparently
    uses the daemon-thread fallback (SIGALRM is main-thread-only). DB writes
    are safe from workers (DBManager uses thread-local connections; WAL).

    breakers: {site: _SiteBreaker} shared across the passes of one run. An
    open breaker's skipped keywords cost no request and no pacing sleep (its
    probes run as normal searches); the other sites' workers are untouched.
    """
    lock = threading.Lock()
    collected: list = []
    hit_limit = threading.Event()

    def _worker(site: str):
        min_gap = SITE_MIN_GAP_SEC.get(site, 0)
        breaker = (breakers or {}).get(site)
        last_start = 0.0
        for keyword in keywords:
            if hit_limit.is_set():
                return
            if session.check_stop() or (control_flags and control_flags.get('stop')):
                return
            # Before the pause wait and the pacing sleep: a skipped search
            # costs nothing. Silent here; the breaker logs one summary. A
            # probe of an open breaker runs like any search (pacing included).
            if breaker and not breaker.allow(label):
                continue
            while control_flags and control_flags.get('pause'):
                time.sleep(1)
            session.wait_if_paused()

            gap = min_gap - (time.monotonic() - last_start)
            if gap > 0:
                time.sleep(gap)
            last_start = time.monotonic()

            jobs = pd.DataFrame()
            try:
                jobs = _run_with_timeout(
                    scrape_jobs, kwargs=kwargs_for(site, keyword),
                    timeout_sec=120, label=f"{label}/{site}/{keyword}",
                )
                if jobs is None:
                    jobs = pd.DataFrame()
            except _ScrapeTimeout as e:
                logger.warning(f"⏱️ {e} — skipping")
            except Exception as e:
                logger.warning(f"Error scraping {label}/{site}/'{keyword}': {e}")

            # Heartbeat (the sequential loops used to run 70+ silent minutes)
            logger.info(f"jobspy[{label}] {site}/'{keyword}': {len(jobs)} jobs")

            # Timeouts and errors leave jobs empty, so "bad" = nothing came back.
            event = breaker.record(not jobs.empty, f"{label}/'{keyword}'") if breaker else None
            if event == "tripped":
                logger.info(
                    f"jobspy[{label}] {site}: {breaker.trip_after} bad searches in a row "
                    f"— skipping {site} (one probe search per pass and every "
                    f"{breaker.probe_every} skips; summary after the last pass)")
            elif event == "recovered":
                logger.info(f"{site} recovered after {breaker.recovered_after} skipped searches")

            if not jobs.empty:
                kw_cat = KEYWORD_TO_CATEGORY.get(keyword.lower(), "")
                recs = jobs.to_dict("records")
                for r in recs:
                    db.save_job(r, session_id=session.id, keyword_category=kw_cat)
                with lock:
                    collected.extend(recs)
                    if limit and len(all_jobs) + len(collected) >= limit:
                        hit_limit.set()

    threads = [
        threading.Thread(target=_worker, args=(s,), daemon=True,
                         name=f"jobspy-{label}-{s}")
        for s in sites
    ]
    for t in threads:
        t.start()
    # Main thread owns progress reporting (progress_callback / SSE state is
    # not assumed thread-safe) and just shepherds the workers.
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=5)
        with lock:
            n = len(collected)
        progress_callback(
            f"{label} pass: {'+'.join(sites)} in parallel — {n} jobs so far...",
            base_pct)

    all_jobs.extend(collected)
    return collected

def dedup_baseline(db) -> tuple:
    """(existing_links, existing_pairs) for dedup_scraped_jobs, snapshotted
    BEFORE scraping (save_job() during scraping would otherwise make every new
    job look "already in the database"). Every known link blocks a re-scrape,
    but only rows the AI GRADED (or the candidate added by hand) block a same-title copy
    from another source: a row dropped before grading (location / keywords /
    age) must not — otherwise a copy dropped on a location rule hides another
    source's copy of the same role that carries a usable location.
    Company keys are FUZZY (utils.company_tokens / tokens_match): exact
    title|company hashing let one board's short brand name and another board's
    long legal name for the same employer surface as two roles."""
    conn = db._get_thread_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT link FROM scrapes WHERE link IS NOT NULL")
    links = {row[0] for row in cursor.fetchall()}
    pairs: dict = {}
    cursor.execute("SELECT title, company, location FROM scrapes WHERE title IS NOT NULL "
                   "AND match_level IS NOT NULL AND match_level != ''")
    for row in cursor.fetchall():
        t = str(row[0]).strip().lower()
        c = company_tokens(row[1])
        if t and c:
            pairs.setdefault(t, []).append((c, location_key(row[2])))
    return links, pairs


def run_search_session(
    resume_path: str,
    base_cv_path: str,
    model_filter: str,
    model_generation: str,
    limit: int = None,
    site_list: list = None,
    keywords_list: list = None,
    control_flags: dict = None,
    progress_callback = None,
    logger = None,
    custom_prompt: str = None,
    send_email: bool = None,
    target_email: str = None,
    session: SessionManager = None,
    db: DBManager = None,
    cascade_config: dict = None,
):
    """
    Main Logic for Job Search Session.
    Accepts either a SessionManager or legacy control_flags/progress_callback.
    """
    if logger is None:
        logger = logging.getLogger("job_agent")

    if db is None:
        db = DBManager()

    # Create session if not provided
    if session is None:
        session = SessionManager()

    # If legacy control_flags/progress_callback provided, wire them up
    if progress_callback is None:
        progress_callback = session.as_progress_callback()

    # Defaults
    if send_email is None:
        send_email = AUTO_SEND_EMAIL
    if target_email is None:
        target_email = EMAIL_RECIPIENTS
    site_list = site_list or SEARCH_SITES
    keywords_list = keywords_list or KEYWORDS

    if cascade_config and cascade_config.get("enabled"):
        logger.info(
            f"SESSION START: 3-Stage Cascade | "
            f"S1={cascade_config.get('stage1_model')} → "
            f"S2={cascade_config.get('stage2_model')} → "
            f"S3={cascade_config.get('stage3_model')} | "
            f"Gen={model_generation} | Session={session.id}"
        )
    else:
        logger.info(f"SESSION START: Filter={model_filter}, Gen={model_generation}, Session={session.id}")

    # Load resources
    profile_text = load_profile_text(resume_path)
    if not profile_text:
        progress_callback("Failed to load resume.", 0)
        return []

    proxies = load_proxies()

    all_jobs = []
    results_wanted_per_site = 20 if not limit else min(limit, 20)


    # Snapshot BEFORE scraping (see dedup_baseline).
    existing_links, existing_pairs = set(), {}
    try:
        existing_links, existing_pairs = dedup_baseline(db)
        logger.info(f"Dedup baseline: {len(existing_links)} links, {len(existing_pairs)} titles with company keys")
    except Exception as e:
        logger.warning(f"Failed to load dedup baseline: {e}")

    progress_callback("Starting Job Search...", 0)

    # Separate special pseudo-sites AND the slow browser scrapers (ZipRecruiter,
    # Glassdoor) from the fast per-keyword jobspy loop. The browser scrapers run a
    # real browser (~30-90s/query), so they get bounded dedicated phases over a
    # small curated keyword set — never the full keyword loop.
    special_sites = ("ats_boards", "industry_boards", "remote_boards", "reddit_jobs",
                     "zip_recruiter", "glassdoor", "google_jobs")
    regular_sites = [s for s in site_list if s not in special_sites]
    scrape_ats = "ats_boards" in site_list
    scrape_industry = "industry_boards" in site_list
    scrape_remote = "remote_boards" in site_list
    scrape_reddit_jobs = "reddit_jobs" in site_list
    scrape_zip = "zip_recruiter" in site_list
    scrape_glassdoor = "glassdoor" in site_list
    scrape_google = "google_jobs" in site_list

    # One breaker per throttle-prone site, shared by the primary, remote and
    # secondary passes.
    breakers = _site_breakers()

    # === SCRAPING PHASE: fast jobspy sites (LinkedIn, Indeed) — PARALLEL per site ===
    # One worker thread per site walking the full keyword list (see
    # _parallel_jobspy_pass). Wall-clock ≈ slowest site instead of the sum —
    # this phase alone was over an hour of one measured run.
    if regular_sites:
        progress_callback(
            f"Scraping {'+'.join(regular_sites)} ({len(keywords_list)} keywords, parallel)...",
            0.05)
        _parallel_jobspy_pass(
            regular_sites, keywords_list,
            lambda site, keyword: dict(
                site_name=[site], search_term=keyword,
                location=PRIMARY_PASS_LOCATION,
                results_wanted=results_wanted_per_site,
                country_indeed='USA',
                linkedin_fetch_description=True,
                proxies=proxies, is_remote=False,
            ),
            session=session, control_flags=control_flags,
            progress_callback=progress_callback, db=db, all_jobs=all_jobs,
            limit=limit, label="primary", base_pct=0.30, logger=logger,
            breakers=breakers,
        )
        if session.check_stop() or (control_flags and control_flags.get('stop')):
            progress_callback("Search Stopped by User.", 1.0)
            return all_jobs

    # === ZIPRECRUITER PHASE (Scrapling solve_cloudflare — bounded keyword set) ===
    # Slow (~90s/query) browser scrape; runs a small curated keyword set and bails
    # after 2 consecutive failures (blocked this run). Bigger SIGALRM budget than
    # jobspy since the Cloudflare solve takes longer.
    if scrape_zip and not (limit and len(all_jobs) >= limit):
        progress_callback("ZipRecruiter (Cloudflare solve)...", 0.58)
        _zip_fails = 0
        for kw in BROWSER_SCRAPE_KEYWORDS:
            if session.check_stop() or (control_flags and control_flags.get('stop')):
                break
            if limit and len(all_jobs) >= limit:
                break
            if _zip_fails >= 2:
                logger.warning("ZipRecruiter: 2 consecutive failures — skipping rest this run")
                break
            try:
                zdf = _run_with_timeout(
                    ziprecruiter_scraper.scrape_ziprecruiter,
                    kwargs=dict(keyword=kw, location=PRIMARY_PASS_LOCATION,
                                results_wanted=results_wanted_per_site),
                    # 2 fetch attempts x 150s internal + Cloudflare-solve margin.
                    # force_thread: SIGALRM is swallowed by the solver's asyncio
                    # loop (it can hang for hours) — daemon-thread abandonment
                    # is the only timeout that holds here.
                    timeout_sec=340, label=f"ZipRecruiter/{kw}", force_thread=True,
                )
            except _ScrapeTimeout as e:
                logger.warning(f"⏱️ {e}"); zdf = None
            except Exception as e:
                logger.warning(f"ZipRecruiter '{kw}' failed: {e}"); zdf = None
            # Breaker contract: None = blocked/error/TIMEOUT (counts), empty
            # DataFrame = page rendered fine but 0 matches (legitimate — resets).
            # Timeouts count now: a hang costs the FULL 340s budget, so 8
            # browser keywords of "slow, not blocked" = 45 min of zero yield —
            # exactly the failure mode the breaker exists to cut short.
            if zdf is not None:
                _zip_fails = 0
                if not zdf.empty:
                    recs = zdf.to_dict("records")
                    all_jobs.extend(recs)
                    kw_cat = KEYWORD_TO_CATEGORY.get(kw.lower(), "")
                    for r in recs:
                        db.save_job(r, session_id=session.id, keyword_category=kw_cat)
            else:
                _zip_fails += 1

    # === GLASSDOOR PHASE (crawl4ai render — bounded keyword set) ===
    if scrape_glassdoor and not (limit and len(all_jobs) >= limit):
        from .glassdoor_crawler import scrape_glassdoor as _scrape_gd
        progress_callback("Glassdoor (crawl4ai)...", 0.60)
        _gd_fails = 0
        # Glassdoor ignores a typed location; only its own city id narrows a
        # search. With the home city's id in the profile
        # (search.glassdoor_city_id) each keyword runs in two lanes: the home
        # city, then nationwide — the lane that brings remote roles and roles
        # in other metros. Without the id, nationwide only. The location
        # pre-filter sorts the nationwide results.
        from .glassdoor_crawler import search_locations as _gd_locations
        _gd_searches = [(kw, loc) for kw in BROWSER_SCRAPE_KEYWORDS
                        for loc in _gd_locations()]
        for kw, gd_loc in _gd_searches:
            if session.check_stop() or (control_flags and control_flags.get('stop')):
                break
            if limit and len(all_jobs) >= limit:
                break
            if _gd_fails >= 2:
                logger.warning("Glassdoor: 2 consecutive failures — skipping rest this run")
                break
            try:
                gdf = _scrape_gd(kw, gd_loc, results_wanted=results_wanted_per_site)
            except Exception as e:
                logger.warning(f"Glassdoor '{kw}' failed: {e}"); gdf = None
            if gdf is not None:
                _gd_fails = 0
                if not gdf.empty:
                    recs = gdf.to_dict("records")
                    # Enrich descriptions by rendering each posting page (the
                    # search list ships description='' — without this most
                    # Glassdoor jobs hit Stage 2 with no text to judge).
                    # enrich_jobs_with_crawl4ai has 30s/page + 300s/batch caps and an
                    # anti-bot circuit breaker (3 blocked pages in a row -> 30-min hold).
                    try:
                        recs = asyncio.run(enrich_jobs_with_crawl4ai(recs))
                    except Exception as e:
                        logger.warning(f"Glassdoor description enrichment failed: {e} — using list data")
                    all_jobs.extend(recs)
                    kw_cat = KEYWORD_TO_CATEGORY.get(kw.lower(), "")
                    for r in recs:
                        db.save_job(r, session_id=session.id, keyword_category=kw_cat)
            else:
                _gd_fails += 1

    # === REMOTE SECOND-PASS (national remote roles — the profile allows remote) ===
    # The loop above searches the profile's home location only. This pass adds
    # NATIONAL remote roles via jobspy is_remote=True over a curated set of
    # high-signal keywords. Residual noise (e.g. manufacturing "production manager")
    # is removed by the NEGATIVE_KEYWORDS pre-filter downstream. Dedup is handled by
    # db.save_job.
    if REMOTE_SECOND_PASS and not (limit and len(all_jobs) >= limit):
        progress_callback("Remote pass: national remote roles (parallel)...", 0.62)
        _parallel_jobspy_pass(
            REMOTE_PASS_SITES, REMOTE_PASS_KEYWORDS,
            lambda site, keyword: dict(
                site_name=[site], search_term=keyword,
                location=None if site in ("linkedin", "glassdoor") else "United States",
                is_remote=True, results_wanted=results_wanted_per_site,
                country_indeed="USA", linkedin_fetch_description=True,
                proxies=proxies,
            ),
            session=session, control_flags=control_flags,
            progress_callback=progress_callback, db=db, all_jobs=all_jobs,
            limit=limit, label="remote", base_pct=0.62, logger=logger,
            breakers=breakers,
        )

    # === SECONDARY-METRO PASS (the profile's secondary metro: its cities are in
    # the allowed locations, but the regular loop searches the home location only) ===
    if SECONDARY_SECOND_PASS and not (limit and len(all_jobs) >= limit):
        progress_callback(
            f"Secondary-metro pass: {SECONDARY_PASS_LOCATION} roles (parallel)...", 0.63)
        secondary_keywords = REMOTE_PASS_KEYWORDS + SECONDARY_PASS_KEYWORDS_EXTRA
        _parallel_jobspy_pass(
            REMOTE_PASS_SITES, secondary_keywords,
            lambda site, keyword: dict(
                site_name=[site], search_term=keyword,
                location=SECONDARY_PASS_LOCATION, is_remote=False,
                results_wanted=results_wanted_per_site,
                country_indeed="USA", linkedin_fetch_description=True,
                proxies=proxies,
            ),
            session=session, control_flags=control_flags,
            progress_callback=progress_callback, db=db, all_jobs=all_jobs,
            limit=limit, label="secondary", base_pct=0.63, logger=logger,
            breakers=breakers,
        )

    # After the last jobspy pass, so the skipped count is the whole run's.
    for _b in breakers.values():
        _b.report(logger)

    # === ATS BOARDS PHASE ===
    if scrape_ats:
        from .ats_scraper import scrape_ats_boards
        from .config import ATS_COMPANIES

        progress_callback("Scanning ATS boards (Greenhouse/Lever)...", 0.65)
        # Two calls, each under its own budget: Workday boards page through
        # fuzzy server searches and can run long, and a timeout discards
        # everything its call collected. Run on their own, a Workday overrun
        # costs only Workday's results; Greenhouse, Lever, Ashby and the other
        # full-list boards (a few minutes) are already kept. The two budgets
        # add up to the single 2400 s the phase had before.
        ats_passes = [
            ("ATS phase (non-Workday boards)",
             [c for c in ATS_COMPANIES if c.get("ats") != "workday"], 900),
            ("ATS phase (Workday boards)",
             [c for c in ATS_COMPANIES if c.get("ats") == "workday"], 1500),
        ]
        ats_frames = []
        for ats_label, ats_boards, ats_budget in ats_passes:
            if not ats_boards:
                continue
            if session.check_stop() or (control_flags and control_flags.get('stop')):
                break
            try:
                # db/run_id: full-list boards record which postings they still
                # list (Phase 27c) — recording never changes the returned jobs.
                # deadline: the call stops starting boards 30 s before its
                # budget ends and returns what it has; the SIGALRM, which
                # discards everything, stays as the backstop.
                part = _run_with_timeout(
                    scrape_ats_boards, args=(ats_boards, keywords_list, progress_callback),
                    kwargs={"db": db, "run_id": session.id,
                            "deadline": time.monotonic() + ats_budget - 30},
                    timeout_sec=ats_budget, label=ats_label,
                )
            except _ScrapeTimeout as e:
                logger.warning(f"⏱️ {e} — no results from these {len(ats_boards)} boards")
                part = None
            if part is not None and not part.empty:
                ats_frames.append(part)
        ats_df = pd.concat(ats_frames, ignore_index=True) if ats_frames else pd.DataFrame()
        if not ats_df.empty:
            ats_records = ats_df.to_dict("records")
            all_jobs.extend(ats_records)
            for r in ats_records:
                db.save_job(r, session_id=session.id)
            logger.info(f"ATS boards: added {len(ats_records)} jobs")
        # Here, not at the end: runs that find nothing new return early.
        _refresh_listing_status(db, logger, "after ATS boards")

    # === INDUSTRY BOARDS PHASE ===
    if scrape_industry:
        from .scraper_industry_html import scrape_industry_html_boards
        from .scraper_rss import scrape_rss_feeds

        progress_callback("Scanning industry boards...", 0.75)
        try:
            ind_df = _run_with_timeout(
                scrape_industry_html_boards, args=(keywords_list, progress_callback),
                timeout_sec=1200, label="Industry phase",
            )
        except _ScrapeTimeout as e:
            logger.warning(f"⏱️ {e}"); ind_df = pd.DataFrame()
        try:
            rss_df = _run_with_timeout(
                scrape_rss_feeds, args=(keywords_list, progress_callback),
                timeout_sec=300, label="RSS phase",
            )
        except _ScrapeTimeout as e:
            logger.warning(f"⏱️ {e}"); rss_df = pd.DataFrame()
        for df in (ind_df, rss_df):
            if not df.empty:
                records = df.to_dict("records")
                all_jobs.extend(records)
                for r in records:
                    db.save_job(r, session_id=session.id)
        ind_count = len(ind_df) if not ind_df.empty else 0
        rss_count = len(rss_df) if not rss_df.empty else 0
        logger.info(f"Industry boards: {ind_count} HTML + {rss_count} RSS = {ind_count + rss_count} jobs")

    # === REMOTE-WORK BOARDS PHASE ===
    if scrape_remote:
        from .scraper_remote_boards import scrape_remote_boards

        progress_callback("Scanning remote-work boards...", 0.81)
        try:
            remote_df = _run_with_timeout(
                scrape_remote_boards, args=(keywords_list, progress_callback),
                timeout_sec=600, label="Remote-boards phase",
            )
        except _ScrapeTimeout as e:
            logger.warning(f"⏱️ {e}"); remote_df = pd.DataFrame()
        if not remote_df.empty:
            records = remote_df.to_dict("records")
            all_jobs.extend(records)
            for r in records:
                db.save_job(r, session_id=session.id)
            logger.info(f"Remote-work boards: added {len(records)} jobs")

    # === GOOGLE JOBS PHASE (SerpAPI, budget-guarded: ~4 credits/day) ===
    if scrape_google:
        from .scraper_google_jobs import scrape_google_jobs

        progress_callback("Scanning Google Jobs (SerpAPI)...", 0.82)
        try:
            gjobs = _run_with_timeout(
                scrape_google_jobs, args=(db,),
                timeout_sec=120, label="Google Jobs phase",
            )
        except _ScrapeTimeout as e:
            logger.warning(f"⏱️ {e}"); gjobs = []
        if gjobs:
            all_jobs.extend(gjobs)
            for r in gjobs:
                db.save_job(r, session_id=session.id)
            logger.info(f"Google Jobs: added {len(gjobs)} jobs")

    # === REDDIT PHASE ===
    if scrape_reddit_jobs:
        from .scraper_reddit import scrape_reddit

        progress_callback("Scanning Reddit job subreddits...", 0.84)
        try:
            reddit_jobs = _run_with_timeout(
                scrape_reddit, args=(keywords_list,),
                timeout_sec=300, label="Reddit phase",
            )
        except _ScrapeTimeout as e:
            logger.warning(f"⏱️ {e}"); reddit_jobs = []
        if reddit_jobs:
            all_jobs.extend(reddit_jobs)
            for r in reddit_jobs:
                db.save_job(r, session_id=session.id)
            logger.info(f"Reddit: added {len(reddit_jobs)} jobs")

    def _send_no_results_digest(total_scraped: int):
        """The 'Nothing new today' email — these early-return paths used to skip
        the digest entirely, which is the most common outcome of a mature-DB
        scheduled run (everything deduped); without it a quiet day looks like a
        failure."""
        if not (send_email and target_email):
            return
        try:
            profile_text = load_profile_text(resume_path)
        except Exception:
            profile_text = ""
        try:
            send_email_with_results(
                None, target_email, pd.DataFrame(),
                profile_text=profile_text,
                total_scraped=total_scraped,
                db=db,
            )
        except Exception as e:
            logger.warning(f"No-results digest failed: {e}")

    # Process Results
    if not all_jobs:
        progress_callback("No jobs found.", 1.0)
        _send_no_results_digest(0)
        return []

    # Dedup against pre-scraping snapshot (not live DB, which includes
    # jobs we just saved during this session's scraping phase).
    # Fuzzy two-layer dedup (utils.dedup_scraped_jobs): by link AND by
    # title + brand-prefix company key; the LinkedIn copy wins a dup group.
    before = len(all_jobs)
    new_jobs = dedup_scraped_jobs(all_jobs, existing_links, existing_pairs)
    if before - len(new_jobs):
        logger.info(f"Dedup: {before - len(new_jobs)} duplicate(s) dropped, {len(new_jobs)} unique jobs remain")

    if not new_jobs:
        progress_callback("All found jobs were already in database.", 1.0)
        _send_no_results_digest(len(all_jobs))
        return []

    return process_found_jobs(
        new_jobs,
        resume_path,
        base_cv_path,
        model_filter,
        model_generation,
        progress_callback=progress_callback,
        control_flags=control_flags,
        logger=logger,
        custom_prompt=custom_prompt,
        send_email=send_email,
        target_email=target_email,
        session=session,
        db=db,
        cascade_config=cascade_config,
    )

def process_pending_db_jobs(
    resume_path: str,
    base_cv_path: str,
    model_filter: str,
    model_generation: str,
    progress_callback = None,
    control_flags: dict = None,
    logger = None
):
    """
    Fetch unprocessed jobs from DB and run them through AI filtering/gen pipeline.
    """
    if logger is None: logger = logging.getLogger("job_agent")

    db = DBManager()
    pending_jobs = db.get_pending_jobs()

    if not pending_jobs:
        if progress_callback: progress_callback("No pending jobs found.", 1.0)
        return []

    if progress_callback: progress_callback(f"Found {len(pending_jobs)} pending jobs. Processing...", 0.0)

    return process_found_jobs(
        pending_jobs,
        resume_path,
        base_cv_path,
        model_filter,
        model_generation,
        progress_callback=progress_callback,
        control_flags=control_flags,
        logger=logger,
        is_retry=True,
        db=db,
    )


def process_found_jobs(
    jobs_list: list,
    resume_path: str,
    base_cv_path: str,
    model_filter: str,
    model_generation: str,
    progress_callback = None,
    control_flags: dict = None,
    logger = None,
    is_retry: bool = False,
    custom_prompt: str = None,
    send_email: bool = False,
    target_email: str = None,
    session: SessionManager = None,
    db: DBManager = None,
    cascade_config: dict = None,
):
    """
    Shared pipeline: Filter -> AI Check -> Docs -> Notify
    """
    if logger is None: logger = logging.getLogger("job_agent")
    if db is None: db = DBManager()
    if session is None: session = SessionManager()
    profile_text = load_profile_text(resume_path)

    df = pd.DataFrame(jobs_list)
    total_scraped = len(df)

    # 2. Process & Filter (Dedupe within session, location, exclusions).
    #    Feed in the candidate's suppressed no-go locations so the candidate's feedback drops
    #    future matches (closes the feedback loop).
    from .utils import process_and_filter_jobs as utils_filter
    suppressed_locs = db.get_suppressed_locations() if db else None
    df = utils_filter(df, suppressed_locations=suppressed_locs)

    if progress_callback: progress_callback(f"Candidates for AI Filtering: {len(df)}", 0.75 if not is_retry else 0.1)

    # GPT Filter & Generation
    if not df.empty:
        if progress_callback: progress_callback(f"AI Filtering ({model_filter})...", 0.80 if not is_retry else 0.2)

        # Check control flags
        if control_flags and control_flags.get('stop'): return []
        if session.check_stop(): return []

        try:
            df = batch_job_relevance_check(df, profile_text, model=model_filter, progress_callback=progress_callback, custom_prompt_template=custom_prompt, cascade_config=cascade_config, db=db)
        except Exception as e:
            import traceback
            logger.error(f"Cascade filter failed: {e}")
            send_error_alert(
                "Cascade filter failed",
                f"Error: {e}\n\n{traceback.format_exc()}",
                pipeline_stage="AI Filtering (cascade)",
            )
            raise

        # Derive tier BEFORE persisting — JobMatcher/legacy fallback paths have
        # no tier column, and saving first wrote tier=NULL for all their matches.
        if "tier" not in df.columns:
            from .cascade_filter import assign_tier
            df["tier"] = df["match_level"].apply(assign_tier) if "match_level" in df.columns else None

        # Save relevance to DB using single-transaction method
        for index, row in df.iterrows():
            try:
                job_id = db.save_job_with_relevance(
                    row.to_dict(),
                    relevance=row.get('gpt_relevance'),
                    reasoning=row.get('gpt_reasoning') or row.get('reason'),
                    match_level=row.get('match_level'),
                    confidence=row.get('confidence'),
                    session_id=session.id,
                    tier=row.get('tier'),
                )
            except Exception as e:
                logger.warning(f"Failed to update relevance for {row.get('title')}: {e}")

        # This run's newly tiered rows join the "still listed?" set.
        _refresh_listing_status(db, logger, "after grading")

        # Tiered output: surface everything with a tier (strong + worth_a_look).
        relevant_df = df[df["tier"].notna()].copy()
        # Sort strong tier first, then by confidence descending within each tier.
        if not relevant_df.empty:
            relevant_df["_tier_rank"] = relevant_df["tier"].map({"strong": 0, "worth_a_look": 1}).fillna(2)
            sort_cols = ["_tier_rank"] + (["confidence"] if "confidence" in relevant_df.columns else [])
            ascending = [True] + ([False] if "confidence" in relevant_df.columns else [])
            relevant_df = relevant_df.sort_values(sort_cols, ascending=ascending).drop(columns=["_tier_rank"])

        if progress_callback: progress_callback(f"Found {len(relevant_df)} relevant jobs!", 0.90 if not is_retry else 0.8)

        # === Posting text for tiered jobs listed without it ===
        # Glassdoor, ZipRecruiter, ShowbizJobs, Podnews and others list a job
        # without its posting; read it now, before the documents and the digest
        # (posting_text: another saved copy, then the posting page).
        if not relevant_df.empty:
            try:
                from .posting_text import fill_tiered_jobs
                fill_tiered_jobs(
                    relevant_df, db, logger,
                    should_stop=lambda: bool(control_flags and control_flags.get('stop')) or session.check_stop())
            except Exception as e:
                if type(e).__name__ == "_ScrapeTimeout":
                    raise
                logger.warning(f"Posting text lookup failed: {e}")

        # === Auto-enrich matched jobs with Perplexity intel + Opus salary advisor ===
        # STRONG tier only: enriching the recall-widened worth_a_look tier burned
        # 3 Perplexity + 1 Opus salary call per job on jobs the candidate may never open.
        # Worth-a-look jobs get enriched on demand from the UI instead.
        enrich_df = relevant_df[relevant_df["tier"] == "strong"] if "tier" in relevant_df.columns else relevant_df
        if not enrich_df.empty:
            try:
                _auto_enrich_matched_jobs(enrich_df, profile_text, db, progress_callback, logger)
            except Exception as e:
                logger.warning(f"Perplexity enrichment failed: {e}")
                send_error_alert(
                    "Perplexity enrichment failed",
                    f"Error: {e}\n\nEnrichment is non-blocking; pipeline continues.",
                    pipeline_stage="Auto-enrich (Perplexity + salary)",
                )

        # === Phase 27c "Is this posting real?" — facts for Strong jobs ===
        # Company research + one posting-specificity read each; both bounded
        # (caps + wall-clock budgets), both non-blocking, neither touches
        # tier / confidence / ordering.
        # The UI search passes no control_flags — its Stop goes through
        # session.check_stop(), re-checked before every call in both loops.
        def _stop_requested():
            return bool(control_flags and control_flags.get('stop')) or session.check_stop()

        if not enrich_df.empty and not _stop_requested():
            if progress_callback:
                progress_callback(f"Checking {len(enrich_df)} strong matches (company news, posting details)...", 0.92)
            try:
                _auto_hiring_health(enrich_df, db, logger, should_stop=_stop_requested)
            except Exception as e:
                logger.warning(f"Auto hiring health failed (non-fatal): {e}")
            try:
                _auto_posting_quality(enrich_df, db, logger, should_stop=_stop_requested)
            except Exception as e:
                logger.warning(f"Posting quality reads failed (non-fatal): {e}")

        if relevant_df.empty:
            if send_email and target_email:
                try:
                    send_email_with_results(
                        None, target_email, pd.DataFrame(),
                        profile_text=profile_text,
                        total_scraped=total_scraped,
                        cascade_stats=cascade_config.get("stats") if cascade_config else None,
                        db=db,
                    )
                except Exception as e:
                    logger.warning(f"Post-process error sending no-results email: {e}")
            return []

        if not relevant_df.empty:
            total_relevant = len(relevant_df)
            base_percent = 0.95 if not is_retry else 0.9

            for index, (i, row) in enumerate(relevant_df.iterrows(), 1):
                company = row.get('company', 'Unknown')
                if progress_callback:
                    step_prog = base_percent + ((index / total_relevant) * (1.0 - base_percent) * 0.9)
                    progress_callback(f"Saving {company} ({index}/{total_relevant})...", step_prog)

                if control_flags and control_flags.get('stop'): break
                if session.check_stop(): break
                try:
                    job_id = db.save_job(row.to_dict(), session_id=session.id)
                except Exception as e:
                    logger.error(f"Failed to save job {company}: {e}")

            # Auto-generate CV/CL for HIGH_MATCH jobs — HARD-CAPPED per run.
            # Each CV+CL is ~$0.22 of premium-model spend, so an uncapped run
            # with many HIGH matches adds up fast. Top-confidence jobs get
            # docs automatically; the rest are one click away in the UI or via
            # the assistant's generate_cv.
            if AUTO_GENERATE_CV_CL:
                from .config import AUTO_CV_MAX_PER_RUN
                # tier == strong (not just match_level): secondary-metro on-site
                # HIGHs are tier-demoted by policy and must not consume the doc budget.
                high_match_rows = relevant_df[
                    (relevant_df["match_level"] == "HIGH_MATCH")
                    & (relevant_df.get("tier") == "strong")
                ]
                if len(high_match_rows) > AUTO_CV_MAX_PER_RUN:
                    if "confidence" in high_match_rows.columns:
                        high_match_rows = high_match_rows.sort_values("confidence", ascending=False)
                    skipped = len(high_match_rows) - AUTO_CV_MAX_PER_RUN
                    high_match_rows = high_match_rows.head(AUTO_CV_MAX_PER_RUN)
                    logger.info(
                        f"CV auto-gen cap: generating top {AUTO_CV_MAX_PER_RUN} by confidence; "
                        f"{skipped} HIGH matches available on-demand (UI button / assistant)"
                    )
                if not high_match_rows.empty:
                    logger.info(f"Auto-generating CV/CL for {len(high_match_rows)} HIGH_MATCH jobs")
                    from .docs_builder import generate_package

                    gen_success = 0
                    for gen_i, (_, hm_row) in enumerate(high_match_rows.iterrows(), 1):
                        if control_flags and control_flags.get('stop'):
                            break
                        if session.check_stop():
                            break

                        hm_company = hm_row.get('company', 'Unknown')
                        hm_title = hm_row.get('title', 'Unknown')
                        if progress_callback:
                            progress_callback(
                                f"Auto-generating CV/CL for {hm_title} at {hm_company} ({gen_i}/{len(high_match_rows)})...",
                                0.95 + 0.04 * gen_i / len(high_match_rows),
                            )

                        try:
                            # Look up the job_id from DB by link
                            # Use pd.notna() to guard against Pandas NaN (which is truthy)
                            raw_url = hm_row.get('job_url')
                            raw_link = hm_row.get('link')
                            link = (raw_url if pd.notna(raw_url) else None) or (raw_link if pd.notna(raw_link) else None)

                            job_id = db.get_job_id_by_link(link) if link else None
                            job_details = db.get_job_details(job_id) if job_id else None

                            if not job_details:
                                logger.warning(f"Auto-gen: Could not find DB record for {hm_title} at {hm_company} (link={link!r}), skipping")
                                continue

                            # Skip if docs already exist
                            if job_details.get('cv_path'):
                                logger.debug(f"Auto-gen: CV already exists for {hm_title} at {hm_company}, skipping")
                                continue

                            # The shared path saves the version, the paths (only
                            # files that exist on disk) and the QA verdict.
                            out = generate_package(
                                job_details, db, profile_text, base_cv_path=base_cv_path,
                                model=model_generation, source="pipeline",
                            )
                            if out.get("error"):
                                raise RuntimeError(out["error"])

                            if not out["cv_path"] and not out["cl_path"]:
                                logger.warning(f"Auto-gen: Doc generation produced no files for {hm_title} at {hm_company} ({out['status']})")
                                continue
                            gen_success += 1
                            logger.info(f"Auto-gen: Generated CV/CL for {hm_title} at {hm_company} (job_id={job_details['id']})")
                        except Exception as e:
                            logger.error(f"Auto-gen failed for {hm_title} at {hm_company}: {e}")
                            send_error_alert(
                                "CV generation failed",
                                f"Job: {hm_title} at {hm_company}\nError: {e}",
                                pipeline_stage="CV/CL Auto-Generation",
                            )

                    logger.info(f"Auto-gen: Generated docs for {gen_success}/{len(high_match_rows)} HIGH_MATCH jobs")

            # LLM comp fallback for tiered rows the deterministic extractor
            # left empty (bounded, ~pennies) — merge results into the digest
            # df so today's email shows the "(est.)" figures too.
            try:
                from .comp_utils import (enrich_tiered_missing_comp, extract_comp,
                                         apply_comp_floor, floor_for_location)
                llm_comp = enrich_tiered_missing_comp(db, cap=20)
                if llm_comp and "job_url" in relevant_df.columns:
                    for col in ("comp_text", "employment_type"):
                        if col not in relevant_df.columns:
                            relevant_df[col] = None
                    for idx, r in relevant_df.iterrows():
                        c = llm_comp.get(r.get("job_url") or r.get("link"))
                        if c:
                            relevant_df.at[idx, "comp_text"] = c["comp_text"]
                            if c["employment_type"] and not r.get("employment_type"):
                                relevant_df.at[idx, "employment_type"] = c["employment_type"]
                # Comp floor (terminal): rows whose KNOWN comp
                # ceiling sits under their location's floor (the profile's
                # search.comp_floor_yearly; a higher floor applies to
                # secondary-metro on-site roles) are UNTIERED in the DB and
                # DROPPED from this digest df — a one-tier demote left
                # clearly below-floor roles visible.
                apply_comp_floor(db)
                if "tier" in relevant_df.columns:
                    drop_idx = []
                    for idx, r in relevant_df.iterrows():
                        if r.get("tier") not in ("strong", "worth_a_look"):
                            continue
                        c = llm_comp.get(r.get("job_url") or r.get("link")) or extract_comp(dict(r))
                        ymax = c.get("comp_yearly_max")
                        if ymax and ymax < floor_for_location(r.get("location")):
                            drop_idx.append(idx)
                    if drop_idx:
                        logger.info(f"Comp floor: dropping {len(drop_idx)} below-floor row(s) from digest")
                        relevant_df = relevant_df.drop(index=drop_idx)
            except Exception as e:
                logger.warning(f"LLM comp fallback failed (non-fatal): {e}")

            # Post-Process
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            os.makedirs("results", exist_ok=True)
            output_file = f"results/jobs_{timestamp}.csv"

            # Post-process steps are INDEPENDENT failure domains: a CSV
            # failure must never cost the candidate the digest email (a shared
            # try-block was silently skipping it).
            available_export_cols = [c for c in EXPORT_COLUMNS if c in relevant_df.columns]
            try:
                relevant_df[available_export_cols].to_csv(output_file, index=False)
            except Exception as e:
                logger.warning(f"CSV export failed: {e}")
                output_file = None  # digest tolerates a missing attachment

            try:
                if send_email and target_email:
                    send_email_with_results(
                        output_file, target_email, relevant_df,
                        profile_text=profile_text,
                        total_scraped=total_scraped,
                        cascade_stats=cascade_config.get("stats") if cascade_config else None,
                        db=db,
                    )
                else:
                    logger.info("Email sending skipped (not requested or no address provided).")
            except Exception as e:
                logger.warning(f"Digest email failed: {e}")

            if progress_callback: progress_callback(f"Search Complete. Saved to {output_file}", 1.0)
            return relevant_df.to_dict("records")
    else:
        if progress_callback: progress_callback("No jobs passed pre-filter.", 1.0)
        if send_email and target_email:
            try:
                send_email_with_results(
                    None, target_email, pd.DataFrame(),
                    profile_text=profile_text,
                    total_scraped=total_scraped,
                    cascade_stats=cascade_config.get("stats") if cascade_config else None,
                    db=db,
                )
            except Exception as e:
                logger.warning(f"Post-process error sending no-results email: {e}")
        return []

    return []

@click.command()
@click.option("--resume", required=True, type=click.Path(exists=True), help="Path to the PDF resume.")
@click.option("--base-cv", required=True, type=click.Path(exists=True), help="Path to the base DOCX CV.")
@click.option("--model-filter", default="claude-haiku-4-5-20251001", help="Model for filtering (default: claude-haiku-4-5).")
@click.option("--model-generation", default=DEFAULT_GENERATION_MODEL,
              help=f"Model for CV/CL generation (default: config DEFAULT_GENERATION_MODEL = {DEFAULT_GENERATION_MODEL}).")
@click.option("--limit", default=None, type=int, help="Limit number of jobs to process (for testing).")
@click.option("--site", default=None, help="Specific site to scrape (e.g. linkedin, glassdoor).")
@click.option("--email/--no-email", default=None, help="Send results via email (default: AUTO_SEND_EMAIL config).")
@click.option("--email-to", default=None, help="Email recipient address (default: EMAIL_RECIPIENTS config).")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--cascade", is_flag=True, help="Enable 3-stage cascade filter.")
@click.option("--cascade-s1", default=None, help="Stage 1 model override (default: config CASCADE_STAGE1_MODEL).")
@click.option("--cascade-s2", default=None, help="Stage 2 model override (default: config CASCADE_STAGE2_MODEL).")
@click.option("--cascade-s3", default=None, help="Stage 3 model override (default: config CASCADE_STAGE3_MODEL).")
def cli(resume, base_cv, model_filter, model_generation, limit, site, email, email_to, verbose, cascade, cascade_s1, cascade_s2, cascade_s3):
    """
    AI Job Agent: run one job search session from the command line.
    """
    logger = setup_logging(verbose)

    from rich.console import Console
    console = Console()

    target_sites = [site] if site else None

    session = SessionManager()
    db = DBManager()

    # Build cascade config if --cascade flag is set
    cascade_config = None
    if cascade:
        from .config import CASCADE_STAGE1_MODEL, CASCADE_STAGE2_MODEL, CASCADE_STAGE3_MODEL
        cascade_config = {
            "enabled": True,
            "stage1_model": cascade_s1 or CASCADE_STAGE1_MODEL,
            "stage2_model": cascade_s2 or CASCADE_STAGE2_MODEL,
            "stage3_model": cascade_s3 or CASCADE_STAGE3_MODEL,
            "stage1_prompt": None,
            "stage2_prompt": None,
            "stage3_prompt": None,
        }
        console.print(f"[bold green]Cascade enabled:[/bold green] S1={cascade_config['stage1_model']} → S2={cascade_config['stage2_model']} → S3={cascade_config['stage3_model']}")

    def cli_progress(text, percent):
        console.print(f"[bold]{text}[/bold]")

    run_search_session(
        resume_path=resume,
        base_cv_path=base_cv,
        model_filter=model_filter,
        model_generation=model_generation,
        limit=limit,
        site_list=target_sites,
        logger=logger,
        progress_callback=cli_progress,
        send_email=email,       # None → defaults to AUTO_SEND_EMAIL in run_search_session
        target_email=email_to,  # None → defaults to EMAIL_RECIPIENTS in run_search_session
        session=session,
        db=db,
        cascade_config=cascade_config,
    )

if __name__ == "__main__":
    cli()
