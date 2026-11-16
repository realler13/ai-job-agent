"""Google Jobs via SerpAPI — the coverage-gap source.

Google Jobs indexes any careers page carrying schema.org JobPosting markup —
which includes every ATS we CAN'T scrape natively (iCIMS, SuccessFactors,
Paylocity, JazzHR, Avature) and the unresolvable Workday tenants (PlayStation,
EA). This is the guaranteed-delivery, structured complement to our native
scrapers; overlap is harmless (dedup absorbs it).

BUDGET: free tier = 250 searches/month, 1 credit per request. Strategy:
QUERIES_PER_RUN queries per run; at two runs a day that is about 120 a
month, under the 230 hard stop enforced via the serpapi_usage table. A rotating keyword queue (persisted index) cycles the
top categories; query A targets the profile's home location on-site,
query B remote-US.
"""

import logging
import os
import re
from datetime import datetime

import requests

from .candidate_profile import SEARCH

logger = logging.getLogger(__name__)

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
MONTHLY_HARD_STOP = 230
QUERIES_PER_RUN = 2

# Rotation queue — weighted toward the highest-yield categories.
ROTATION_QUERIES = [
    "video producer", "content producer", "youtube producer",
    "creative producer", "podcast producer", "live events producer",
    # Add or remove queries to match your profile's target titles.
    "production manager video", "digital series producer",
    "multimedia producer", "creator operations", "event producer",
    "program manager content",
]

# EMPLOYER-GRADE LINK POLICY (2026-07-05):
# Google indexes real employer/ATS pages AND an ecosystem of aggregator chum
# (Hollylist premium paywalls, WhatJobs/JobLeads redirect chains, SEO listing
# pages). Blocklisting chum trails reality — new boards appear weekly — so we
# default-deny: a row is kept ONLY if some apply option is employer-grade,
# meaning a known ATS host or a domain that matches the company name. Rows
# whose only apply path is an aggregator are not actionable and are dropped
# at source. LinkedIn/Indeed/ZipRecruiter apply options don't count either:
# our native scrapers already cover those boards with better data.

# ATS hosts we trust — deliberately includes platforms we CAN'T scrape
# natively (iCIMS, SuccessFactors, Taleo, Paylocity...): reaching those is
# this source's whole reason to exist.
_ATS_HOSTS = (
    "greenhouse.io", "lever.co", "myworkdayjobs.com", "ashbyhq.com",
    "smartrecruiters.com", "icims.com", "successfactors.com", "sapsf.com",
    "workable.com", "recruitee.com", "teamtailor.com", "bamboohr.com",
    "eightfold.ai", "paylocity.com", "jazzhr.com", "avature.net",
    "oraclecloud.com", "taleo.net", "ultipro.com",
    "jobvite.com", "breezy.hr", "applytojob.com", "rippling-ats.com",
    "gem.com", "governmentjobs.com", "schooljobs.com", "workdayjobs.com",
    "dover.com",  # measured on live rows 2026-07-05
    # willhire.co / magnitglobal.com removed 2026-10-03: a contingent-workforce
    # marketplace is not an employer surface, and the native WillHire scraper
    # already covers it, with the contingent-marketplace cap applied.
)

# Known aggregator/board cores — hard-denied BEFORE any company matching.
# Default-deny already kills unknown boards; this list exists because a board
# whose name collides with a company token would otherwise self-certify
# ("Monster Energy" + monster.com, "24 Seven Talent" + talent.com — both
# confirmed leaks in the 2026-07-05 adversarial review).
_AGGREGATOR_CORES = frozenset((
    "monster", "talent", "tealhq", "builtin", "builtinchicago", "builtinnyc",
    "builtinla", "builtinsf", "dice", "career", "careers", "whatjobs",
    "jobleads", "mediabistro", "lensa", "care", "jooble", "bebee",
    "learn4good", "hollylist", "jobzmall", "localjobs", "remotejobs",
    "diversityjobs", "edtechjobs", "sonicjobs", "recruit", "linkedin",
    "indeed", "ziprecruiter", "glassdoor", "simplyhired", "snagajob",
    "jobrapido", "adzuna", "jobgether", "themuse", "virtualvocations",
    "ihirebroadcasting", "jobdekhi", "trabajo", "employbl",
    "welcometothejungle", "useparallel", "careervault", "sorce", "mindpal",
    "tallo", "getwork", "jobot", "salary", "jobcase", "neuvoo", "jobrific",
    "workopolis", "cosmoquick", "hiringcafe", "jora", "getjob",
))

# Residual labels that don't disprove a company↔domain match
# ('jobs.ebayinc.com' → core 'ebayinc' = 'ebay' + 'inc'). Corporate suffixes
# ONLY — job-flavored residues (jobs/careers/work/get) are exactly the
# aggregator naming pattern and were removed after the adversarial review.
_DOMAIN_STOP_SUFFIXES = frozenset(
    "inc llc co corp corporation hq holdings global".split())

# Company-name tokens too generic to prove a domain belongs to the employer.
# Includes job-market words: "Work & Co"/"Remote Technology"/"Local Media"
# must not self-certify getwork.com / remotejobs.org / localjobs.com.
_COMPANY_STOP_TOKENS = frozenset(
    "the and inc corp corporation company co llc ltd group holdings media "
    "jobs careers career global international studios entertainment "
    "work works remote local talent hire hiring job staffing recruiting "
    "agency".split())


def _registrable_core(url: str) -> str:
    """'https://jobs.ebayinc.com/x' -> 'ebayinc' (registrable label only)."""
    try:
        host = url.split("://", 1)[-1].split("/", 1)[0].split(":")[0].lower()
    except Exception:
        return ""
    parts = [p for p in host.split(".") if p and p != "www"]
    if len(parts) < 2:
        return ""
    # second-level label; handles co.uk-style suffixes crudely but safely
    core = parts[-2]
    if core in ("co", "com", "org", "net", "ac", "gov", "edu",
                "ltd", "plc") and len(parts) >= 3:
        core = parts[-3]
    return core


def _company_tokens(company: str, min_len: int = 4) -> list:
    toks = re.findall(r"[a-z0-9]+", (company or "").lower())
    return [t for t in toks if len(t) >= min_len and t not in _COMPANY_STOP_TOKENS]


def _is_ats_host(host: str, path: str) -> bool:
    """ATS-host check with path awareness: bare vendor roots / www marketing
    hosts don't count — real postings live on subdomains (boards.greenhouse.io,
    apply.workable.com, tenant.wd1.myworkdayjobs.com) or deep paths."""
    for a in _ATS_HOSTS:
        if host == a or host == "www." + a:
            # vendor root: only path-based ATSes qualify (governmentjobs.com/
            # careers/<agency>), and only with a real path
            if a in ("governmentjobs.com", "schooljobs.com") and len(path) > 1:
                return True
            continue
        if host.endswith("." + a):
            # cross-company browse/search pages on ATS domains aren't a
            # posting (jobs.workable.com/search?query=..., confirmed leak)
            if path.startswith("/search") or path in ("", "/"):
                return False
            return True
    return False


def is_employer_grade(url: str, company: str) -> bool:
    """True if the link plausibly lands on the employer's own application
    surface: a known ATS host, or a domain matching the company name
    (careers.google.com~Google, jobs.ebayinc.com~eBay). A 'jobs.' subdomain
    alone proves nothing — jobs.stevenagefc.com served staffing-agency chum."""
    if not url:
        return False
    rest = url.split("://", 1)[-1]
    host = rest.split("/", 1)[0].split(":")[0].lower()
    path = ("/" + rest.split("/", 1)[1].split("?")[0]) if "/" in rest else ""
    core = _registrable_core(url)
    # Known boards are denied BEFORE company matching: a name collision
    # ("Monster Energy" + monster.com) must never self-certify a board.
    if core in _AGGREGATOR_CORES:
        return False
    if _is_ats_host(host, path):
        return True
    if not core:
        return False
    core_flat = core.replace("-", "")  # coca-cola.com → cocacola
    toks = _company_tokens(company, min_len=3)
    # Token concatenation: epicgames.com~Epic Games, sonymusic.com~Sony Music,
    # coca-cola.com~Coca-Cola (adversarial review: single-token anchoring
    # rejected all of these — the 'other' token isn't a corporate suffix).
    if len(toks) >= 2:
        for i in range(len(toks) - 1):
            if toks[i] + toks[i + 1] == core_flat:
                return True
        if "".join(toks) == core_flat:
            return True
    # Anchored single-token match: the company token IS the core, or is its
    # prefix/suffix with a corporate-suffix residual ('ebayinc'='ebay'+'inc').
    # Bare containment is a trap: 'capital' ⊂ 'venturecapitalcareers'
    # (a job board) scored a false accept during calibration.
    for t in toks:
        if t == core_flat:
            return True
        if core_flat.startswith(t) and core_flat[len(t):] in _DOMAIN_STOP_SUFFIXES:
            return True
    # domain core embedded in a longer company token ('sony' in 'sonymusic')
    if len(core_flat) >= 4 and any(core_flat in t for t in _company_tokens(company)):
        return True
    return False


# Junk-row guards (source-agnostic red flags Google passes through verbatim).
_JUNK_COMPANIES = frozenset((
    "premium employer", "confidential", "employer", "not disclosed",
    "hiring company", "private", "private employer", "undisclosed",
    "anonymous", "recruiting agency", "staffing agency", ""))
# Token-level check for placeholder words that no real employer name carries.
# Deliberately narrow: "Anonymous Content" and "Private Division" are REAL
# production companies — only 'confidential'/'undisclosed' are safe as tokens.
_JUNK_COMPANY_TOKENS = frozenset(("confidential", "undisclosed", "confidencial"))
_LISTING_TITLE_RE = re.compile(
    r"\bjobs?\s+(?:in|near|at)\b|\bhiring\s+now\b|^\d+\s+best\b|\bjobs\s*!?\s*$",
    re.IGNORECASE)


def _is_junk_company(company: str) -> bool:
    c = (company or "").strip().lower()
    if c in _JUNK_COMPANIES:
        return True
    return any(t in _JUNK_COMPANY_TOKENS for t in re.findall(r"[a-z]+", c))


def _is_listing_page_title(title: str) -> bool:
    """'Contract Podcast Producer Jobs in Springfield', 'Producer Jobs
    (Hiring Now!)', '20 Best video producer jobs' = SEO listing pages, not
    jobs. ('Steve Jobs Theater' titles stay safe — no in/near/at follows.)"""
    return bool(_LISTING_TITLE_RE.search(title or ""))


def _budget_state(db):
    """(month_key, used) from the serpapi_usage table (created on demand)."""
    month = datetime.now().strftime("%Y-%m")
    conn = db._get_thread_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS serpapi_usage (
        month TEXT PRIMARY KEY, count INTEGER DEFAULT 0, rotation_idx INTEGER DEFAULT 0)""")
    conn.commit()
    row = conn.execute("SELECT count, rotation_idx FROM serpapi_usage WHERE month=?",
                       (month,)).fetchone()
    if not row:
        conn.execute("INSERT INTO serpapi_usage (month, count, rotation_idx) VALUES (?,0,0)", (month,))
        conn.commit()
        return month, 0, 0
    return month, row[0], row[1]


def _spend(db, month, n, new_idx):
    conn = db._get_thread_conn()
    conn.execute("UPDATE serpapi_usage SET count = count + ?, rotation_idx = ? WHERE month = ?",
                 (n, new_idx, month))
    conn.commit()


def _pick_apply_url(job: dict, company: str) -> str:
    """First employer-grade apply option, or '' when every option is an
    aggregator (callers drop the row — a chum-only posting is not actionable)."""
    options = job.get("apply_options") or []
    for o in options:
        link = o.get("link", "")
        if link and is_employer_grade(link, company):
            return link
    return ""


def _map_job(job: dict) -> dict | None:
    """SerpAPI job → standard dict, or None for rows the candidate can't act on
    (aggregator-only links, paywall-placeholder companies, SEO listing pages)."""
    from .utils import normalize_date_posted
    company = job.get("company_name", "") or ""
    title = job.get("title", "") or ""
    if _is_junk_company(company):
        return None
    if _is_listing_page_title(title):
        return None
    job_url = _pick_apply_url(job, company)
    if not job_url:
        return None
    loc = job.get("location", "") or ""
    if loc.strip().lower() == "anywhere":
        loc = "Remote"  # Google's remote label → our location gate's vocabulary
    highlights = []
    for h in job.get("job_highlights") or []:
        items = h.get("items") or []
        if items:
            highlights.append(f"{h.get('title', '')}: " + " ".join(items))
    desc = (job.get("description", "") or "") + ("\n" + "\n".join(highlights) if highlights else "")
    det = job.get("detected_extensions") or {}
    return {
        "title": title,
        "company": company,
        "location": loc,
        "job_url": job_url,
        "description": desc,
        "date_posted": normalize_date_posted(det.get("posted_at", "")),
        "site": "google_jobs",
        # coverage analytics: which board Google indexed this from
        "via": job.get("via", ""),
    }


def search_google_jobs(query: str, location: str = "", remote: bool = False,
                       api_key: str = None) -> list:
    """One SerpAPI google_jobs request (1 credit) → list of standard job dicts.
    No budget accounting here — callers guard the budget."""
    key = api_key or SERPAPI_KEY or os.getenv("SERPAPI_KEY", "")
    if not key:
        logger.warning("SERPAPI_KEY missing — google_jobs skipped")
        return []
    params = {"engine": "google_jobs", "q": query, "hl": "en", "gl": "us",
              "api_key": key}
    if remote:
        params["ltype"] = "1"
        params["location"] = location or "United States"
    elif location:
        params["location"] = location
        params["lrad"] = "50"
    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            logger.warning(f"google_jobs error for {query!r}: {data['error']}")
            return []
        raw = data.get("jobs_results") or []
        jobs, dropped = [], []
        for j in raw:
            m = _map_job(j)
            if m:
                jobs.append(m)
            else:
                dropped.append(f"{j.get('company_name', '?')} | {j.get('title', '?')}")
        if dropped:
            # Named log so dropped-but-interesting roles aren't silently lost —
            # the assistant's search_company_board can chase the employer surface.
            logger.info(f"google_jobs {query!r}: kept {len(jobs)}/{len(raw)}; "
                        f"dropped (no employer link): " + "; ".join(dropped[:10]))
        return jobs
    except Exception as e:
        logger.warning(f"google_jobs request failed for {query!r}: {e}")
        return []


def scrape_google_jobs(db) -> list:
    """Pipeline entry: 2 budget-guarded queries per run — one on-site around the
    profile's home location (search.locations.home, rotating keyword), one
    remote-US (next keyword). Silently skips when the monthly hard-stop is
    reached."""
    if not (SERPAPI_KEY or os.getenv("SERPAPI_KEY")):
        return []
    month, used, idx = _budget_state(db)
    if used + QUERIES_PER_RUN > MONTHLY_HARD_STOP:
        logger.info(f"google_jobs: monthly budget reached ({used}/{MONTHLY_HARD_STOP}) — skipping")
        return []
    q_local = ROTATION_QUERIES[idx % len(ROTATION_QUERIES)]
    q_remote = ROTATION_QUERIES[(idx + 1) % len(ROTATION_QUERIES)]
    results = []
    results += search_google_jobs(q_local, location=SEARCH["locations"]["home"])
    results += search_google_jobs(q_remote, remote=True)
    _spend(db, month, QUERIES_PER_RUN, (idx + 2) % len(ROTATION_QUERIES))
    logger.info(f"Google Jobs: {len(results)} jobs "
                f"({q_local!r} local + {q_remote!r} remote) — {used + 2}/{MONTHLY_HARD_STOP} this month")
    return results
