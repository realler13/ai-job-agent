"""
ZipRecruiter scraper via Scrapling's StealthyFetcher with Cloudflare solving.

ZipRecruiter fronts its search with Cloudflare ("forbidden aa" 403 / Turnstile
challenge) — a plain request (and even jobspy's app-API) gets blocked. The fix
that works WITHOUT proxies: StealthyFetcher(solve_cloudflare=True) drives a
stealth Firefox (Camoufox) that solves the challenge and returns the rendered
results page. `wait_selector="article"` blocks until the job cards render
(otherwise the page sometimes returns before hydration → 0 cards).

This is SLOW (~60-120s per query: real browser + challenge solve), so callers
MUST bound the keyword set — see ZIP_SEARCH_KEYWORDS / the dedicated phase in
main.py. Do not run it across all ~148 keywords.

Card layout (after a redesign): each listing is
<article data-testid="job-card-{listing key}"> with an h2 title, the company in
a /co/ link and the location in a /jobs-search link. The old layout put the key
in id="job-card-..." with data-testid company/location anchors. Both are read:
during the switch some pages still came in the old layout, so the site served
both for a while and may again. The new page also embeds the word "forbidden"
in its Next.js payload, which a block check over the raw HTML mistakes for a
block page — that, not Cloudflare, makes every search come back empty.

Posting dates: the cards show none, but the page's Next.js payload holds one
JobCard object per card with its listingKey, status.postedAtUtc (Unix
seconds) and rollingPostedAt ("22 days ago"). The rolling date is the
original date moved forward in steps of 30 days, so a two-month-old posting
can read "4 days ago"; postedAtUtc is the real age and wins.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote_plus

import pandas as pd
from bs4 import BeautifulSoup

from .candidate_profile import SEARCH

try:
    from scrapling.fetchers import StealthyFetcher
    SCRAPLING_AVAILABLE = True
except Exception:  # ImportError or any env breakage — never sink main.py
    StealthyFetcher = None
    SCRAPLING_AVAILABLE = False

from .utils import normalize_date_posted

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.ziprecruiter.com/jobs-search"
_FETCH_TIMEOUT_MS = 150_000

# New layout first, old layout second (see module docstring).
_CARD_SELECTOR = "article[data-testid^='job-card-'], article[id^='job-card']"
_BLOCK_MARKERS = (
    "access denied", "forbidden", "captcha", "just a moment", "are you a human",
)
# The results page title states the hit count: "187 Video Producer Jobs (NOW HIRING) in ..."
_TITLE_COUNT_RE = re.compile(r"^\s*([\d,]+)\s.*\bjobs\b", re.IGNORECASE)

# Payload pieces (quotes unescaped first; see module docstring). The seconds
# value is written "$n1794940649" (the payload's big-number form).
_JOBCARD_MARK = '"$typeName":"job_services.job_card.public.v1.JobCard"'
_LISTING_KEY_RE = re.compile(r'"listingKey":"([^"]+)"')
_POSTED_AT_RE = re.compile(r'"postedAtUtc":\{[^{}]*?"seconds":"?(?:\$n)?(\d+)')
_ROLLING_RE = re.compile(r'"rollingPostedAt":"([^"]*)"')
_AGO_RE = re.compile(r"(\d+)\+?\s+(minute|hour|day|week|month)s?\s+ago|(today|just posted|yesterday)", re.I)


def _card_key(card) -> str:
    """Listing key from data-testid (new layout) or id (old layout)."""
    raw = card.get("data-testid") or card.get("id") or ""
    return raw.replace("job-card-", "", 1).strip() if raw.startswith("job-card-") else ""


def _ago_to_date(text: str) -> str:
    """Turn "22 days ago" or "today" into YYYY-MM-DD; "" when unreadable."""
    m = _AGO_RE.search(text or "")
    if not m:
        return ""
    if m.group(3):
        days = 1 if m.group(3).lower() == "yesterday" else 0
    else:
        n = int(m.group(1))
        days = {"minute": 0, "hour": 0, "day": n, "week": 7 * n, "month": 30 * n}[m.group(2).lower()]
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


def _posting_dates(body: str) -> dict[str, str]:
    """Listing key → date posted (YYYY-MM-DD) from the page payload:
    postedAtUtc when present, else the relative rollingPostedAt."""
    flat = body.replace('\\"', '"')
    dates: dict[str, str] = {}
    for chunk in flat.split(_JOBCARD_MARK)[1:]:
        key = _LISTING_KEY_RE.search(chunk)
        if not key or key.group(1) in dates:
            continue
        posted = _POSTED_AT_RE.search(chunk)
        date = normalize_date_posted(posted.group(1)) if posted else ""
        if not date:
            rolling = _ROLLING_RE.search(chunk)
            date = _ago_to_date(rolling.group(1)) if rolling else ""
        if date:
            dates[key.group(1)] = date
    return dates


def _looks_blocked(soup: BeautifulSoup) -> bool:
    """Block/challenge markers in what a person would SEE — the page title and
    visible text — never in script payloads, where words like "forbidden" are
    ordinary framework keys on a perfectly good results page."""
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    seen = soup.get_text(" ", strip=True).lower()
    return any(m in seen for m in _BLOCK_MARKERS)


def _fetch_html(url: str, attempts: int = 2) -> Optional[str]:
    """Fetch a ZipRecruiter page, solving Cloudflare; retry once if cards missing."""
    if not SCRAPLING_AVAILABLE:
        logger.warning("ZipRecruiter: scrapling not available — skipping")
        return None
    last_body: Optional[str] = None
    for i in range(attempts):
        try:
            page = StealthyFetcher.fetch(
                url,
                headless=True,
                solve_cloudflare=True,   # solves the Turnstile/Interstitial block
                network_idle=True,
                humanize=True,           # human-like cursor movement
                wait_selector="article", # wait for job cards to render
                timeout=_FETCH_TIMEOUT_MS,
            )
        except Exception as e:
            logger.warning(f"ZipRecruiter fetch error (attempt {i + 1}/{attempts}): {e}")
            continue
        body = getattr(page, "body", None) or getattr(page, "html_content", "") or ""
        if body and "job-card" in body:
            return body
        last_body = body
    return last_body


def scrape_ziprecruiter(
    keyword: str,
    location: Optional[str] = None,
    results_wanted: int = 20,
) -> Optional[pd.DataFrame]:
    """Scrape one ZipRecruiter search (keyword x location).

    ``location`` defaults to the profile's home location
    (search.locations.home).

    Returns a DataFrame of jobs, or None when the page is blocked/empty
    (so the caller's circuit-breaker can react).
    """
    location = location or SEARCH["locations"]["home"]
    url = f"{_SEARCH_URL}?search={quote_plus(keyword)}&location={quote_plus(location)}"
    logger.info(f"ZipRecruiter: searching '{keyword}' in {location} (Cloudflare solve)...")
    body = _fetch_html(url)
    if not body:
        logger.warning(f"ZipRecruiter: no content for '{keyword}' (likely blocked)")
        return None

    soup = BeautifulSoup(body, "html.parser")
    cards = soup.select(_CARD_SELECTOR)
    if not cards:
        # Zero cards is ambiguous: a genuine no-results page, a block page, or a
        # layout we no longer parse. Block/interstitial pages are small and show
        # telltale text; real ZipRecruiter result pages are several hundred KB.
        title_el = soup.find("title")
        page_title = title_el.get_text(strip=True) if title_el else ""
        if len(body) < 50_000 or _looks_blocked(soup):
            logger.warning(f"ZipRecruiter: blocked/empty shell for '{keyword}'")
            return None  # counts toward the circuit breaker
        m = _TITLE_COUNT_RE.match(page_title)
        if m and int(m.group(1).replace(",", "")) > 0:
            # The page says it has jobs but no card parsed: the layout changed.
            # Say so plainly (logging it as "blocked" hid the real cause) and
            # count it toward the breaker so the run does not burn every keyword.
            logger.warning(
                f"ZipRecruiter: page lists jobs ({page_title[:80]!r}) but 0 cards parsed "
                f"for '{keyword}' — card layout changed? Update _CARD_SELECTOR")
            return None
        logger.info(f"ZipRecruiter: page rendered, 0 cards for '{keyword}'")
        return pd.DataFrame()  # legitimate empty — resets the breaker

    dates = _posting_dates(body)
    jobs: list[dict] = []
    seen: set[str] = set()
    for card in cards:
        # The listing key (after "job-card-") dedups ZipRecruiter's doubled DOM cards
        key = _card_key(card)
        if not key or key in seen:
            continue
        seen.add(key)

        h2 = card.select_one("h2")
        title = h2.get_text(strip=True) if h2 else ""
        if not title:
            continue

        comp = card.select_one("a[data-testid='job-card-company'], a[href^='/co/']")
        loc = card.select_one("a[data-testid='job-card-location'], a[href^='/jobs-search']")
        jobs.append({
            "site": "zip_recruiter",
            "title": title,
            "company": comp.get_text(strip=True) if comp else "",
            "location": loc.get_text(strip=True) if loc else location,
            # ZipRecruiter's public listing URL format (lvk=listing key) —
            # the old synthetic /jobs/{key} path returned 404.
            "job_url": f"https://www.ziprecruiter.com/jobs//j?lvk={key}",
            "description": "",  # list view has no full description; cascade/enrich handle it
            "date_posted": dates.get(key),  # None when the payload has no date for it
        })
        if len(jobs) >= results_wanted:
            break

    if not jobs:
        logger.info(f"ZipRecruiter: 0 jobs matched for '{keyword}' (page fine)")
        return pd.DataFrame()
    logger.info(f"ZipRecruiter: {len(jobs)} jobs for '{keyword}'")
    return pd.DataFrame(jobs)
