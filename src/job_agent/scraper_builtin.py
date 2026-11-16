"""
Built In scraper — tech/media jobs from Built In's regional sites (see REGIONS).

Built In's regional subdomains serve server-rendered HTML. No auth wall,
no Cloudflare. Keyword search via `?search=X`, pagination via `?page=N`.

Card structure (verified April 2026 via live probe):
  div[data-id="job-card"] id="job-card-{id}"
    a[data-id="company-title"]                    → company name
    a[data-id="job-card-title"]                   → job title + URL path
    span ... "Reposted X Hours/Days Ago"          → posting age

Built In jobs are indexed by numeric ID embedded in the URL path
(/job/{slug}/{id}), which we use for deduplication.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from .ats_scraper import _matches_keywords

logger = logging.getLogger(__name__)

# Sample regions (the scraper's first test sites). Built In has a site per
# metro: add or swap in your own here.
REGIONS = {
    "la": "https://www.builtinla.com",
    "sf": "https://www.builtinsf.com",
}
REQUEST_TIMEOUT = 20
INTER_REQUEST_DELAY = 1.0

# Broad seed queries for Built In's fuzzy server search. Querying all ~148
# keywords (2 regions x 148 x 3 pages ≈ 888 requests) triggered HTTP 429; these
# few seeds catch the same roles and each result is title-gated against the full
# keyword list downstream, so precision is preserved at a fraction of the load.
BUILTIN_SEARCH_SEEDS = [
    "producer", "content", "video", "production",
    "podcast", "creative", "studio", "editorial",
]

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})


def _parse_posting_age(text: str) -> str:
    """Convert "Reposted 3 Hours Ago" → "YYYY-MM-DD" (today).
    Returns empty string if unparseable."""
    if not text:
        return ""
    m = re.search(
        r"(?:reposted|posted)\s+(\d+)\s+(hour|day|week|month)s?\s+ago",
        text, re.I,
    )
    if not m:
        # Some say "Today" or "Yesterday"
        if "today" in text.lower():
            return datetime.now().strftime("%Y-%m-%d")
        if "yesterday" in text.lower():
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        return ""
    n = int(m.group(1))
    unit = m.group(2).lower()
    now = datetime.now()
    if unit == "hour":
        dt = now
    elif unit == "day":
        dt = now - timedelta(days=n)
    elif unit == "week":
        dt = now - timedelta(weeks=n)
    elif unit == "month":
        dt = now - timedelta(days=30 * n)
    else:
        return ""
    return dt.strftime("%Y-%m-%d")


def _parse_card(card, base_url: str) -> dict | None:
    """Extract job fields from a single Built In job card."""
    title_link = card.select_one('a[data-id="job-card-title"]')
    company_link = card.select_one('a[data-id="company-title"]')
    if not title_link or not company_link:
        return None

    title = title_link.get_text(strip=True)
    href = (title_link.get("href") or "").strip()
    company = company_link.get_text(strip=True)
    if not title or not href or not company:
        return None
    job_url = href if href.startswith("http") else f"{base_url}{href}"

    # Full card text — pull posting-age line, location hints, salary hints
    card_text = card.get_text(" ", strip=True)
    date_posted = _parse_posting_age(card_text)

    # Location: Built In cards include city/state + work-arrangement tags. Our
    # pipeline already has a location pre-filter, so we stash whatever short
    # chunk mentions the city.
    location = ""
    # Simple heuristic: grab text near known city tokens for the sample regions
    # (extend the tuple when you add a region).
    for hint in ("Los Angeles", "Hollywood", "Santa Monica", "Culver City", "Burbank",
                 "Pasadena", "San Francisco", "Bay Area", "Oakland", "Berkeley",
                 "Palo Alto", "San Jose", "Mountain View", "Remote", "Hybrid"):
        if hint in card_text:
            location = hint
            break

    return {
        "title": title,
        "company": company,
        "location": location,
        "job_url": job_url,
        "description": "",  # Built In list view doesn't include a snippet
        "date_posted": date_posted,
        "site": "builtin",
        "department": "",
    }


def _get_with_backoff(url: str, max_retries: int = 3):
    """GET with exponential backoff on HTTP 429 (Built In rate-limits hard).

    Returns the Response, or None if 429 persisted through all retries.
    Raises on non-429 HTTP errors (caller handles).
    """
    for attempt in range(max_retries):
        r = _session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 429:
            wait = INTER_REQUEST_DELAY * (2 ** attempt) + 1.0
            logger.info(f"Built In 429 — backing off {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    return None


def _scrape_region(region: str, seed: str, match_keywords: list[str], max_pages: int = 3) -> list[dict]:
    """Scrape one Built In region (a REGIONS key) for one broad SEED query.

    The server search is fuzzy, so we query a few broad seeds and gate every
    result title against the FULL keyword list — same coverage, far fewer
    requests (which is what fixed the 429s).
    """
    base = REGIONS.get(region)
    if not base:
        logger.warning(f"Built In: unknown region {region!r}")
        return []

    results: list[dict] = []
    seen_urls: set[str] = set()

    for page in range(1, max_pages + 1):
        url = f"{base}/jobs?search={requests.utils.quote(seed)}&page={page}"
        try:
            r = _get_with_backoff(url)
            if r is None:
                logger.warning(f"Built In {region}/{seed!r}: still 429 after retries (page {page})")
                break
        except Exception as e:
            logger.warning(f"Built In {region} fetch error (page {page}): {e}")
            break

        soup = BeautifulSoup(r.text, "html.parser")
        cards = soup.select('div[data-id="job-card"]')
        if not cards:
            break

        new_on_page = 0
        for card in cards:
            job = _parse_card(card, base)
            if not job:
                continue
            if job["job_url"] in seen_urls:
                continue
            seen_urls.add(job["job_url"])
            new_on_page += 1  # new card PARSED — pagination continues even on a no-match page
            # Gate fuzzy server results against the FULL keyword list
            if not _matches_keywords(job["title"], match_keywords):
                continue
            results.append(job)

        if new_on_page == 0 and page > 1:
            break  # past the last relevant page
        time.sleep(INTER_REQUEST_DELAY)

    return results


def scrape_builtin(keywords: list[str], max_pages: int = 3) -> list[dict]:
    """Scrape every Built In region in REGIONS for all given keywords.

    Args:
        keywords: Terms to query Built In's search with.
        max_pages: Pages per region per keyword (default 3 → up to ~75 jobs).

    Returns:
        Deduplicated list of job dicts.
    """
    all_jobs: list[dict] = []
    seen_urls: set[str] = set()

    for region in REGIONS:
        for seed in BUILTIN_SEARCH_SEEDS:
            for job in _scrape_region(region, seed, keywords, max_pages=max_pages):
                if job["job_url"] in seen_urls:
                    continue
                seen_urls.add(job["job_url"])
                all_jobs.append(job)

    logger.info(
        f"Built In: {len(all_jobs)} unique jobs "
        f"({len(BUILTIN_SEARCH_SEEDS)} seeds x {len(REGIONS)} regions)"
    )
    return all_jobs
