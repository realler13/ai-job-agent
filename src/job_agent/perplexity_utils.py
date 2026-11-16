"""
Perplexity Agent API integration for search-grounded company intelligence.
(Migrated from Sonar chat completions 2026-08-13 — /v1/agent, presets, typed output.)

Four use cases:
  1. careers_page_lookup  — resolve company careers URL + ATS type (7-day cache)
  2. company_profile      — 3-sentence company summary for cascade enrichment (30-day cache)
  3. job_enrichment       — clarify if borderline role is production vs social/marketing (14-day cache)
  4. hiring_health        — "is this company really hiring?": dated, sourced public facts
                            (14-day cache) + a free signal from our own scrape history

All calls are optional (USE_PERPLEXITY must be True) and fail gracefully.
Caching is via the company_intel SQLite table (db_manager.py).
"""

import json
import logging
import re
import time
import threading
from collections import defaultdict
from datetime import date, timedelta
from typing import Optional
from urllib.parse import urlparse

import requests

from .config import (
    PERPLEXITY_API_KEY,
    PERPLEXITY_PRESET,
    PERPLEXITY_CITATIONS_PRESET,
    PERPLEXITY_TIMEOUT,
    PERPLEXITY_RPM_LIMIT,
    PERPLEXITY_COMPANY_PROFILE_TTL_DAYS,
    PERPLEXITY_JOB_ENRICHMENT_TTL_DAYS,
    PERPLEXITY_CAREERS_TTL_DAYS,
    PERPLEXITY_HIRING_INTEL_TTL_DAYS,
    PERPLEXITY_SALARY_INTEL_TTL_DAYS,
    PERPLEXITY_SALARY_REC_TTL_DAYS,
    USE_PERPLEXITY,
    DEFAULT_GENERATION_MODEL,
    METRICS_BANK_TEXT,
)

logger = logging.getLogger(__name__)


class PerplexityRateLimiter:
    """Simple token-bucket rate limiter for Perplexity (50 RPM)."""

    def __init__(self, rpm: int = PERPLEXITY_RPM_LIMIT):
        self._rpm = rpm
        self._lock = threading.Lock()
        self._calls: list[float] = []

    def wait_if_needed(self):
        with self._lock:
            now = time.time()
            self._calls = [t for t in self._calls if now - t < 60]
            if len(self._calls) >= self._rpm:
                sleep_time = 60 - (now - self._calls[0]) + 0.1
                time.sleep(sleep_time)
            self._calls.append(time.time())


_rate_limiter = PerplexityRateLimiter()


def _within_ttl(last_checked_str: Optional[str], ttl_days: int) -> bool:
    """Return True if last_checked is within TTL days of today."""
    if not last_checked_str:
        return False
    try:
        checked = date.fromisoformat(last_checked_str)
        return (date.today() - checked).days < ttl_days
    except Exception:
        return False


_AGENT_API_URL = "https://api.perplexity.ai/v1/agent"


def _agent_search(query: str, preset: str = PERPLEXITY_PRESET) -> Optional[dict]:
    """Make a single Perplexity Agent API call (2026 migration from Sonar).

    Request is {"preset", "input"}; the response is a typed `output` array —
    `message` items carry the answer text (content parts of type
    "output_text"), `search_results` items carry the grounding sources.

    Returns {"content": str, "search_results": [ ... ]} or None on failure.
    """
    if not USE_PERPLEXITY:
        return None
    _rate_limiter.wait_if_needed()
    try:
        resp = requests.post(
            _AGENT_API_URL,
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"preset": preset, "input": query},
            timeout=PERPLEXITY_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        texts: list[str] = []
        results: list[dict] = []
        for item in data.get("output") or []:
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text" and part.get("text"):
                        texts.append(part["text"])
            elif item.get("type") == "search_results":
                results.extend(item.get("results") or [])
        content = "\n".join(texts).strip() or str(data.get("output_text") or "").strip()
        if not content:
            return None
        return {"content": content, "search_results": results}
    except Exception as e:
        logger.warning(f"Perplexity API failed: {e}")
        return None


def _raw_search(query: str) -> Optional[str]:
    """Text-only Perplexity search. Returns prose or None on failure."""
    out = _agent_search(query)
    return out["content"] if out else None


def get_company_profile(company_name: str, db=None) -> Optional[str]:
    """
    Return a 3-sentence company profile for cascade enrichment.
    Checks company_intel cache first (30-day TTL).
    """
    if not USE_PERPLEXITY:
        return None
    if db:
        cached = db.get_company_intel(company_name)
        if cached and cached.get("company_profile") and _within_ttl(
            cached.get("last_checked"), PERPLEXITY_COMPANY_PROFILE_TTL_DAYS
        ):
            return cached["company_profile"]
    query = (
        f"Brief company profile for {company_name}: What do they do, what industry, "
        f"approximate size/stage (startup/growth/enterprise), and headquarters location? "
        f"3 sentences max."
    )
    result = _raw_search(query)
    if result and db:
        existing = db.get_company_intel(company_name) or {}
        db.save_company_intel({
            **existing,
            "company_name": company_name,
            "company_profile": result,
        })
    return result


def resolve_careers_page(company_name: str, db=None) -> Optional[dict]:
    """
    Resolve careers URL and ATS type for a company.
    Returns {"careers_url": str, "ats_type": str} or None.
    Checks company_intel cache first (7-day TTL).
    """
    if not USE_PERPLEXITY:
        return None
    if db:
        cached = db.get_company_intel(company_name)
        if cached and cached.get("careers_url") and _within_ttl(
            cached.get("last_checked"), PERPLEXITY_CAREERS_TTL_DAYS
        ):
            return {"careers_url": cached["careers_url"], "ats_type": cached.get("ats_type")}
    query = (
        f"What is the careers or jobs page URL for {company_name}? "
        f"Also identify what applicant tracking system they use "
        f"(Greenhouse, Lever, Workday, Ashby, SmartRecruiters, or other). "
        f'Return ONLY a JSON object: {{"careers_url": "...", "ats_type": "..."}} '
        f'If no careers page exists, return {{"careers_url": null, "ats_type": "not_found"}}'
    )
    raw = _raw_search(query)
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group())
    except Exception:
        return None
    if db:
        existing = db.get_company_intel(company_name) or {}
        db.save_company_intel({
            **existing,
            "company_name": company_name,
            "careers_url": data.get("careers_url"),
            "ats_type": data.get("ats_type"),
        })
    return data


def enrich_borderline_job(title: str, company: str, db=None) -> Optional[str]:
    """
    Clarify if a borderline job is production vs social/marketing.
    Used only for Stage 3 POTENTIAL_MATCH jobs.
    Cache key = __enrichment__{company} (14-day TTL).
    """
    if not USE_PERPLEXITY:
        return None
    cache_key = f"__enrichment__{company}"
    if db:
        cached = db.get_company_intel(cache_key)
        if cached and cached.get("company_profile") and _within_ttl(
            cached.get("last_checked"), PERPLEXITY_JOB_ENRICHMENT_TTL_DAYS
        ):
            return cached["company_profile"]
    query = (
        f'For the role "{title}" at {company}: What does the team/department '
        f"that would hire this role actually do? Is this a content production role, "
        f"a social media management role, a marketing role, or a project management role? "
        f"1-2 sentences."
    )
    result = _raw_search(query)
    if result and db:
        db.save_company_intel({"company_name": cache_key, "company_profile": result})
    return result


def enrich_job_description(title: str, company: str, location: str = "", db=None) -> Optional[str]:
    """Fetch a concise role summary for a job whose SCRAPED description is too
    thin for Stage 2 to judge fairly (boards often return a one-line stub).

    Used at the Stage 1 -> Stage 2 boundary. Cache key = __jd__{company}__{title}
    (14-day TTL, shared with job-enrichment).
    """
    if not USE_PERPLEXITY:
        return None
    cache_key = f"__jd__{company}__{title}"
    if db:
        cached = db.get_company_intel(cache_key)
        if cached and cached.get("company_profile") and _within_ttl(
            cached.get("last_checked"), PERPLEXITY_JOB_ENRICHMENT_TTL_DAYS
        ):
            return cached["company_profile"]
    loc = f" in {location}" if location else ""
    query = (
        f'For the job posting "{title}" at {company}{loc}: summarize what this role '
        f"actually does day-to-day, its seniority level, and the team/department it sits in. "
        f"State clearly whether it is hands-on content/video/media PRODUCTION, social-media "
        f"management, marketing, or technical project management. 3-4 factual sentences; "
        f"if you cannot find the specific posting, describe what this title typically means "
        f"at a company like this."
    )
    result = _raw_search(query)
    if result and db:
        db.save_company_intel({"company_name": cache_key, "company_profile": result})
    return result


def get_hiring_intel(company_name: str, job_title: str, db=None) -> Optional[str]:
    """
    Return actionable hiring intelligence for a company+role combination.
    Surfaces hiring managers, LinkedIn activity, recent news, and team structure.
    Cache key = __hiring_intel__{company} (14-day TTL).
    """
    if not USE_PERPLEXITY:
        return None
    # Per company+ROLE cache: the old per-company key served one role's intel
    # to every other role at that company for 14 days.
    safe_title = re.sub(r"[^A-Za-z0-9 &/+-]+", "", str(job_title or ""))[:60]
    cache_key = f"__hiring_intel__{company_name}__{safe_title}"[:200]
    if db:
        cached = db.get_company_intel(cache_key)
        if cached and cached.get("company_profile") and _within_ttl(
            cached.get("last_checked"), PERPLEXITY_HIRING_INTEL_TTL_DAYS
        ):
            return cached["company_profile"]
    # Posting-anchored phrasing + anti-hallucination guard (empirically zero
    # invented names across live probes). The previous "Search LinkedIn for..."
    # prompt invited invented names/URLs — LinkedIn blocks crawlers, so its
    # grounding was thin.
    query = (
        f"{company_name} is hiring a '{job_title}'. Based on public information "
        f"(the job posting, LinkedIn, press, company pages), which team or "
        f"department does this role sit in, and who is the likely hiring "
        f"manager or team lead? Also: which recruiters at {company_name} handle "
        f"hiring for this kind of role? Name names ONLY if a public source "
        f"states them, and say which source each name comes from. If no names "
        f"are publicly documented, say so explicitly instead of guessing, and "
        f"describe the team or title to target instead."
    )
    result = _raw_search(query)
    if result and db:
        db.save_company_intel({"company_name": cache_key, "company_profile": result})
    return result


# === Hiring health ===
# "Is this company really hiring?" Two parts, composed fresh on every read:
#   1. a free internal signal from our own scrape history (never cached), and
#   2. dated, sourced public facts from Perplexity (cached 14 days under
#      __hiring_health__{company}). Cite-or-omit is enforced in CODE: a line
#      without a recent date and a source link never reaches the panel.

HIRING_HEALTH_TTL_DAYS = 14
HIRING_HEALTH_HISTORY_DAYS = 90
# Calibrated on a 90-day corpus of scraped listings: one job reaches other
# boards within 14 days most of the time, so a repeat title ACROSS
# boards is cross-posting, not re-listing. A re-listing is a new listing on the
# SAME board and SAME location, 14+ days after the previous one counted.
HIRING_HEALTH_RELIST_GAP_DAYS = 14
# Public facts must fall in the last 12 months. The slack lets a month-only
# date ("2025-10" reads as the 1st) through without admitting a fact from
# two years ago.
HIRING_HEALTH_LOOKBACK_DAYS = 365
_LOOKBACK_SLACK_DAYS = 31
# Hard wall-clock budget for the one search (rate-limit wait + request).
# requests' timeout bounds each socket wait, not the whole call, and SIGALRM
# does not work in API worker threads — so the call runs in a daemon thread.
HIRING_HEALTH_SEARCH_DEADLINE_SEC = PERPLEXITY_TIMEOUT * 2 + 15
_NO_SIGNAL = "No public signal found"
_HEALTH_UNAVAILABLE = (
    "_Public-source check did not complete — nothing is shown rather than "
    "unverified text. Research the company again to retry._"
)
_URL_RE = re.compile(r"https?://[^\s<>()\[\]]+")
_MARKER_RE = re.compile(r"[ \t]*\[(\d+)\](?!\()")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{1,2})(?:-(\d{1,2}))?\b")
_NAMED_DATE_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:(\d{1,2})(?:st|nd|rd|th)?,?\s+)?(20\d{2})\b",
    re.IGNORECASE,
)
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def _host(url) -> str:
    """Lower-cased host of a URL without the www. prefix ("" when unparseable)."""
    try:
        host = urlparse(str(url or "")).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _norm_text(value) -> str:
    """Case- and whitespace-insensitive form for exact title/location matching."""
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


# The Review page mounts a (hidden) company panel for EVERY row, so one page
# load asks for the scanner signal of every row at once. `scrapes` has no
# company index — a per-company read scans the whole 90-day window (tens of
# thousands of rows) — so one pass builds a company -> row-id map
# and every panel then reads its rows by primary key.
_SCANNER_INDEX_TTL_SECONDS = 600
_scanner_index_lock = threading.Lock()
_scanner_index: dict = {"db_path": None, "built_at": 0.0, "ids": {}}


def _company_rows(db, company: str) -> list:
    """This company's (title, link, location, created_at) rows in the window."""
    conn = db._get_thread_conn()
    db_path = getattr(db, "db_path", None)
    window = f"-{HIRING_HEALTH_HISTORY_DAYS} days"
    with _scanner_index_lock:
        fresh = (
            _scanner_index["db_path"] == db_path
            and time.time() - _scanner_index["built_at"] < _SCANNER_INDEX_TTL_SECONDS
        )
        if not fresh:
            ids = defaultdict(list)
            for row_id, name in conn.execute(
                "SELECT id, company FROM scrapes WHERE created_at >= datetime('now', ?)",
                (window,),
            ):
                key = _norm_text(name)
                if key:
                    ids[key].append(int(row_id))
            _scanner_index.update(db_path=db_path, built_at=time.time(), ids=dict(ids))
        row_ids = _scanner_index["ids"].get(_norm_text(company), [])
    rows: list = []
    for start in range(0, len(row_ids), 500):
        chunk = ",".join(str(i) for i in row_ids[start:start + 500])
        rows.extend(conn.execute(
            f"SELECT title, link, location, created_at FROM scrapes WHERE id IN ({chunk})"
        ).fetchall())
    return rows


def get_scanner_stats(company_name: str, db, job_title: Optional[str] = None) -> Optional[dict]:
    """
    Our own scrape history for a company (last 90 days), as numbers:
    {"listings", "distinct_titles", "window_days",
     "title": None (no job_title asked) | {"listings", "boards", "first_seen",
              "latest_seen", "relisted"}}  — dates as YYYY-MM-DD or None.
    `scrapes` keeps one row per link, so one job on four boards is four
    listings. A re-listing is a new listing of the same title on the SAME
    board and location, HIRING_HEALTH_RELIST_GAP_DAYS+ days after the previous
    one counted. None when the company has no history or the read fails.
    """
    company = str(company_name or "").strip()
    if not company or db is None:
        return None
    try:
        rows = _company_rows(db, company)
    except Exception as e:
        logger.warning(f"Scanner signal read failed for {company}: {e}")
        return None
    if not rows:
        return None

    titles = {_norm_text(r[0]) for r in rows} - {""}
    stats = {
        "listings": len(rows),
        "distinct_titles": len(titles),
        "window_days": HIRING_HEALTH_HISTORY_DAYS,
        "title": None,
    }
    wanted = _norm_text(job_title)
    if not wanted:
        return stats
    same = [r for r in rows if _norm_text(r[0]) == wanted]

    # (board, location) -> the days a new link for this title first appeared there
    sightings: dict = defaultdict(set)
    for _title, link, location, created in same:
        try:
            day = date.fromisoformat(str(created)[:10])
        except ValueError:
            continue
        sightings[(_host(link), _norm_text(location))].add(day)
    relisted = 0
    for days in sightings.values():
        ordered = sorted(days)
        anchor, n = ordered[0], 0
        for d in ordered[1:]:
            if (d - anchor).days >= HIRING_HEALTH_RELIST_GAP_DAYS:
                n, anchor = n + 1, d
        relisted = max(relisted, n)
    all_days = sorted(d for days in sightings.values() for d in days)
    boards = {board for board, _loc in sightings} - {""}
    stats["title"] = {
        "listings": len(same),
        "boards": len(boards),
        "first_seen": all_days[0].isoformat() if all_days else None,
        "latest_seen": all_days[-1].isoformat() if all_days else None,
        "relisted": relisted,
    }
    return stats


def get_scanner_signal(company_name: str, db, job_title: Optional[str] = None) -> Optional[str]:
    """
    Free internal hiring signal from our own scrape history (last 90 days).
    Returns markdown bullets ("Seen by our scanner: ...") or None when the
    company has no history or the read fails. Counts, not verdicts: `scrapes`
    keeps one row per link, so one job on four boards is four listings — the
    distinct-title count is the cleaner measure of open roles. New listings
    show up within 10 minutes (the company index TTL). The numbers come from
    get_scanner_stats.
    """
    stats = get_scanner_stats(company_name, db, job_title)
    if not stats:
        return None

    lines = [
        f"- **Seen by our scanner:** {_count(stats['listings'], 'listing')} from this company in "
        f"the last {stats['window_days']} days (one job can appear on several job "
        f"boards), {_count(stats['distinct_titles'], 'distinct title')}."
    ]
    same = stats["title"]
    if same is None:
        return "\n".join(lines)
    if not same["listings"]:
        lines.append(
            f"- **This exact title:** not seen in the last {HIRING_HEALTH_HISTORY_DAYS} days."
        )
        return "\n".join(lines)

    line = f"- **This exact title:** {_count(same['listings'], 'listing')}"
    if same["boards"]:
        line += f" on {_count(same['boards'], 'job board')}"
    if same["first_seen"]:
        line += f", first seen {same['first_seen']}, latest {same['latest_seen']}"
    if same["relisted"]:
        line += (
            f"; re-listed {_count(same['relisted'], 'time')} (a new listing on the same job board "
            f"and location, {HIRING_HEALTH_RELIST_GAP_DAYS}+ days after the previous one)."
        )
    else:
        line += "; no re-listing seen."
    lines.append(line)
    return "\n".join(lines)


def _is_recent(text: str) -> bool:
    """True when a line carries a date inside the last 12 months (URLs must be
    stripped first — their paths are full of years). Full and month dates are
    judged exactly; a bare year counts only when no fuller date is given and it
    is this year or last."""
    today = date.today()
    oldest = today - timedelta(days=HIRING_HEALTH_LOOKBACK_DAYS + _LOOKBACK_SLACK_DAYS)
    latest = today + timedelta(days=366)   # an announced, dated plan is still a fact
    found = []
    candidates = [(int(y), int(m), int(d or 1)) for y, m, d in _ISO_DATE_RE.findall(text)]
    candidates += [(int(y), _MONTHS[mon[:3].lower()], int(d or 1))
                   for mon, d, y in _NAMED_DATE_RE.findall(text)]
    for y, m, d in candidates:
        try:
            found.append(date(y, m, d))
        except ValueError:
            continue            # "2025-2026" is a span of years, not a month
    if found:
        return any(oldest <= f <= latest for f in found)
    return any(today.year - 1 <= int(y) <= today.year for y in _YEAR_RE.findall(text))


# Words a topic label may be built from (the six topics the query asks for).
# A "No public signal found" line passes without a source only when everything
# before the phrase is such a label — "Cut 500 roles, and on freezes: No public
# signal found" is a claim and needs its own date and source.
_TOPIC_WORDS = frozenset("""
    layoffs layoff job cuts hiring freezes freeze or and restructuring
    reorganization funding acquisition acquisitions m&a financial finances
    signals signal open roles role openings trend visibly visible activity is
    news leadership executive departures hires changes change in the of content
    production marketing organization org team teams creative cuts video events
""".split())


def _is_topic_label(prefix: str) -> bool:
    """True when `prefix` (the text before "No public signal found") is only a
    bullet / bold topic label such as "- **Layoffs:** " — or nothing at all."""
    label = re.sub(r"[*_#|>]", " ", prefix).strip().lstrip("-•").strip()
    if not label:
        return True
    if len(label) > 90 or label[-1] not in ":—–-" or re.search(r"\d", label):
        return False
    return all(w in _TOPIC_WORDS for w in re.findall(r"[a-z&]+", label.lower()))


def _search_with_deadline(query: str, preset: str) -> Optional[dict]:
    """_agent_search under HIRING_HEALTH_SEARCH_DEADLINE_SEC of wall clock.
    Raw daemon thread (as main._run_in_thread_with_timeout): a hung call is
    abandoned, never joined, and its late answer is discarded — nothing is
    cached from it. Returns None on timeout or failure."""
    box: dict = {}

    def _target():
        try:
            box["value"] = _agent_search(query, preset=preset)
        except Exception as e:  # _agent_search already logs; keep the thread quiet
            box["error"] = e

    thread = threading.Thread(target=_target, daemon=True, name="hiring-health-search")
    thread.start()
    thread.join(timeout=HIRING_HEALTH_SEARCH_DEADLINE_SEC)
    if thread.is_alive():
        logger.warning(
            f"Hiring health search exceeded {HIRING_HEALTH_SEARCH_DEADLINE_SEC}s (thread abandoned)")
        return None
    if "error" in box:
        logger.warning(f"Hiring health search failed: {box['error']}")
        return None
    return box.get("value")


def _hiring_health_query(company_name: str) -> str:
    """The search query. Company name only — no candidate details ever leave here."""
    today = date.today()
    since = today - timedelta(days=365)
    return (
        f"Research the employer {company_name}. Today is {today.isoformat()}. Report ONLY "
        f"dated, sourced facts from the last 12 months (since {since.isoformat()}) on six topics:\n"
        f"1. Layoffs — date, scale, teams affected.\n"
        f"2. Hiring freezes or restructuring.\n"
        f"3. Funding, acquisition or financial signals.\n"
        f"4. Open roles — whether the company is visibly hiring, and the open-role trend.\n"
        f"5. Leadership changes in the content, production or marketing organization.\n"
        f"6. Layoffs or restructuring in content, video, creative, production, marketing "
        f"or events teams — date, scale, teams affected.\n"
        f"Answer as a flat markdown bullet list, one fact per line, in this shape:\n"
        f"- **Layoffs:** YYYY-MM-DD — what happened — source\n"
        f"- **Creative team cuts:** YYYY-MM-DD — what happened — source\n"
        f"Every line must carry the date of the event and the source it comes from "
        f"(citation and full URL). When nothing public is found on a topic, write "
        f"exactly: - **<topic>:** {_NO_SIGNAL}\n"
        f"Facts, not verdicts: no advice, no summary, no speculation about what the "
        f"facts mean. Never guess — leave out anything a public source does not state."
    )


def _cite_or_omit(content: str, search_results: Optional[list]) -> Optional[str]:
    """
    Keep only the answer lines that carry BOTH a date inside the last 12
    months and a source link from the search's own grounding results; "[n]"
    citation markers become links to those results. A "No public signal
    found" line that is only a topic label plus the phrase is kept as is.
    Returns the surviving markdown, or None when nothing survives or the
    search returned no grounding results at all (an answer nothing grounded
    is not a check) — an incomplete check, which the caller must never cache
    or show as "no signal".
    """
    urls = [str(r.get("url") or "") for r in (search_results or []) if isinstance(r, dict)]
    grounded_hosts = {_host(u) for u in urls} - {""}
    if not grounded_hosts:
        return None

    def _link(m):
        i = int(m.group(1)) - 1
        url = urls[i] if 0 <= i < len(urls) else ""
        if not url.startswith("http"):
            return ""
        return f" ([source]({url.replace('(', '%28').replace(')', '%29')}))"

    # A block = one top-level line plus its indented continuation lines, so a
    # bullet whose source sits on a sub-bullet is judged as one fact.
    blocks: list[list[str]] = []
    for raw in str(content or "").splitlines():
        if not raw.strip():
            continue
        if raw[0] in " \t" and blocks:
            blocks[-1].append(raw.rstrip())
        else:
            blocks.append([raw.rstrip()])

    kept: list[str] = []
    heading = None
    dropped = 0
    for block in blocks:
        text = "\n".join(block)
        if text.lstrip().startswith("#"):
            heading = text.strip()
            continue
        if text.lstrip().startswith("|"):
            # A table row (the prompt asks for bullets): flatten its cells so a
            # surviving row still reads as one fact without its header.
            if not text.strip().strip("|:- "):
                continue
            cells = [c.strip() for c in text.strip().strip("|").split("|")]
            text = "- " + " — ".join(c for c in cells if c)
        at = text.lower().find(_NO_SIGNAL.lower())
        if at >= 0 and _is_topic_label(text[:at]):
            # Topic label + the phrase; anything said after it is dropped.
            text = _MARKER_RE.sub("", text[:at + len(_NO_SIGNAL)])
        else:
            text = _MARKER_RE.sub(_link, text)
            # A model-written URL counts only when its site is among the
            # grounding results.
            sourced = any(_host(u) in grounded_hosts for u in _URL_RE.findall(text))
            dated = _is_recent(_URL_RE.sub(" ", text))
            if not (sourced and dated):
                dropped += 1
                continue
        if heading:
            kept.append(heading)
            heading = None
        kept.append(text)
    if not kept:
        return None
    if dropped:
        kept.append(f"\n_{_count(dropped, 'line')} without a dated source omitted._")
    return "\n".join(kept)


def hiring_health_is_fresh(company_name: str, db) -> bool:
    """True when this company's public-source research is cached and inside
    HIRING_HEALTH_TTL_DAYS — get_hiring_health would make no network call.
    Never raises."""
    company = str(company_name or "").strip()
    if not company or db is None:
        return False
    try:
        cached = db.get_company_intel(f"__hiring_health__{company}"[:200])
    except Exception as e:
        logger.warning(f"Hiring health cache read failed for {company}: {e}")
        return False
    if not cached or not cached.get("company_profile"):
        return False
    return _within_ttl(cached.get("last_checked"), HIRING_HEALTH_TTL_DAYS)


def get_hiring_health(company_name: str, db=None, job_title: Optional[str] = None,
                      cached_only: bool = False) -> Optional[str]:
    """
    Return the "Hiring health" markdown for a company: our own scanner signal
    first, then dated, sourced public facts from the last 12 months (layoffs,
    freezes/restructuring, funding, open-role trend, leadership changes, cuts
    in content / video / creative / production / marketing / events teams).
    Cache key = __hiring_health__{company} (14-day TTL) — public facts only;
    the scanner signal is recomputed on every call. `job_title` feeds the
    scanner signal only and never the search query. `cached_only=True` never
    calls the network (panel GET). The search runs under a hard wall-clock
    deadline (HIRING_HEALTH_SEARCH_DEADLINE_SEC). A failed, timed-out or
    unsourced search is never cached and never shown as "no signal".
    """
    company = str(company_name or "").strip()
    if not company:
        return None
    signal = get_scanner_signal(company, db, job_title) if db else None

    cache_key = f"__hiring_health__{company}"[:200]
    cached = db.get_company_intel(cache_key) if db else None
    cached_text = cached.get("company_profile") if cached else None
    cached_at = cached.get("last_checked") if cached else None
    research = checked = None
    if cached_text and (cached_only or _within_ttl(cached_at, HIRING_HEALTH_TTL_DAYS)):
        research, checked = cached_text, cached_at

    attempted = False
    if research is None and not cached_only and USE_PERPLEXITY:
        attempted = True
        out = _search_with_deadline(_hiring_health_query(company), PERPLEXITY_CITATIONS_PRESET)
        fresh = _cite_or_omit(out["content"], out.get("search_results")) if out else None
        if fresh:
            research, checked = fresh, date.today().isoformat()
            if db:
                db.save_company_intel({"company_name": cache_key, "company_profile": fresh})
                # Show the stamp the cache carries, so a later cached read
                # prints the same check date.
                checked = (db.get_company_intel(cache_key) or {}).get("last_checked") or checked
        elif cached_text:
            # Search failed: the expired cache's dated facts are still facts —
            # show them under their own check date rather than nothing.
            research, checked = cached_text, cached_at

    parts = [signal] if signal else []
    if research:
        parts.append(f"**Public sources** (checked {checked}):\n\n{research}")
    elif attempted:
        parts.append(_HEALTH_UNAVAILABLE)
    return "\n\n".join(parts) if parts else None


def get_salary_intel(job_title: str, company: str, location: str, db=None) -> Optional[str]:
    """
    Return salary/compensation intelligence for a job title at a specific company.
    Queries Glassdoor, Levels.fyi, LinkedIn Salary, and Payscale data.
    Cache key = __salary_intel__{company}__{job_title} (14-day TTL).
    """
    if not USE_PERPLEXITY:
        return None
    cache_key = f"__salary_intel__{company}__{job_title}"
    if db:
        cached = db.get_company_intel(cache_key)
        if cached and cached.get("company_profile") and _within_ttl(
            cached.get("last_checked"), PERPLEXITY_SALARY_INTEL_TTL_DAYS
        ):
            return cached["company_profile"]
    query = (
        f"What is the typical salary range for {job_title} at {company} in {location}? "
        f"Include base salary, total compensation if available. "
        f"Also what do similar {job_title} roles pay at comparable companies in {location}? "
        f"Focus on specific numbers and ranges from Glassdoor, Levels.fyi, "
        f"LinkedIn Salary, or Payscale data."
    )
    result = _raw_search(query)
    if result and db:
        db.save_company_intel({"company_name": cache_key, "company_profile": result})
    return result


def generate_salary_recommendation(
    job: dict,
    company_intel: dict,
    profile_text: str = "",
    db=None,
) -> Optional[str]:
    """
    Generate a salary negotiation recommendation using Opus.

    Uses the candidate's profile, Perplexity market data, and job details
    to produce a personalized salary strategy.

    Args:
        job: Job dict with title, company, location, description.
        company_intel: Dict with company_profile, salary_intel, hiring_intel keys.
        profile_text: Candidate resume/profile text.
        db: DBManager for caching.

    Returns:
        Salary recommendation text, or None on failure.
    """
    title = job.get("title", "")
    company = job.get("company", "")
    location = job.get("location", "")

    cache_key = f"__salary_rec__{company}__{title}"

    # Check cache
    if db:
        cached = db.get_company_intel(cache_key)
        if cached and cached.get("company_profile") and _within_ttl(
            cached.get("last_checked"), PERPLEXITY_SALARY_REC_TTL_DAYS
        ):
            return cached["company_profile"]

    # Build context from Perplexity intel
    salary_intel = company_intel.get("salary_intel") or "No salary data available."
    company_profile = company_intel.get("company_profile") or "No company profile available."
    hiring_intel = company_intel.get("hiring_intel") or ""

    # Truncate description for prompt — untrusted data: AI-directed and
    # applicant-instruction sentences stripped first.
    from .jd_sanitizer import clean_jd, UNTRUSTED_JD_CLAUSE
    _desc = job.get("description")
    description = clean_jd(_desc if isinstance(_desc, str) else "")[:3000]

    # Truncate profile for prompt
    profile_summary = profile_text[:4000] if profile_text else "Profile not available."

    prompt = f"""You are a salary negotiation advisor for creative and media production professionals.
Based on the candidate's profile and the market data below, recommend a salary target and negotiation strategy.

<candidate_profile>
{profile_summary}
</candidate_profile>

<verified_metrics>
{METRICS_BANK_TEXT}
</verified_metrics>

<untrusted_data_notice>{UNTRUSTED_JD_CLAUSE}</untrusted_data_notice>
<job>
Title: {title}
Company: {company}
Location: {location}
Description: {description}
</job>

<market_data>
{salary_intel}
</market_data>

<company_context>
{company_profile}
</company_context>

{f'<hiring_context>{hiring_intel}</hiring_context>' if hiring_intel else ''}

<instructions>
Provide a concise salary strategy (3-4 paragraphs max):

1. **Recommended salary range**: Base salary + total comp estimate (equity, bonus, benefits) if data is available. If specific data is limited, provide a reasonable range based on title, location, and company stage/size.

2. **Candidate positioning**: How the candidate's experience and track record position them within this range, using ONLY facts from the candidate profile and the verified metrics above. Be specific about which verified metrics are most relevant leverage for THIS role.

3. **Key negotiation points**: 2-3 concrete talking points the candidate should use, based on gaps between their strengths and the role requirements.

4. **Compensation trajectory**: Whether this role is a step up, lateral, or step down — and whether the brand/opportunity justifies any trade-off.

Be direct and actionable. No filler.
</instructions>"""

    try:
        from .gpt_filter import call_llm
        result = call_llm(prompt, model=DEFAULT_GENERATION_MODEL, temperature=0.2)
        if result and db:
            db.save_company_intel({"company_name": cache_key, "company_profile": result})
        return result or None
    except Exception as e:
        logger.warning(f"Salary recommendation failed for {title} at {company}: {e}")
        return None
