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
"""
from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.ziprecruiter.com/jobs-search"
_FETCH_TIMEOUT_MS = 150_000


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
    cards = soup.select("article[id^='job-card']")
    if not cards:
        # Zero cards is ambiguous: a genuine no-results page or a block page.
        # Block/interstitial pages are small and carry telltale markers; real
        # ZipRecruiter result pages are several hundred KB.
        low = body.lower()
        blocked = any(m in low for m in (
            "access denied", "forbidden", "captcha", "just a moment", "are you a human",
        ))
        if blocked or len(body) < 50_000:
            logger.warning(f"ZipRecruiter: blocked/empty shell for '{keyword}'")
            return None  # counts toward the circuit breaker
        logger.info(f"ZipRecruiter: page rendered, 0 cards for '{keyword}'")
        return pd.DataFrame()  # legitimate empty — resets the breaker

    jobs: list[dict] = []
    seen: set[str] = set()
    for card in cards:
        # The listing id (after "job-card-") dedups ZipRecruiter's doubled DOM cards
        key = (card.get("id") or "").replace("job-card-", "").strip()
        if not key or key in seen:
            continue
        seen.add(key)

        h2 = card.select_one("h2")
        title = h2.get_text(strip=True) if h2 else ""
        if not title:
            continue

        comp = card.select_one("a[data-testid='job-card-company']")
        loc = card.select_one("a[data-testid='job-card-location']")
        jobs.append({
            "site": "zip_recruiter",
            "title": title,
            "company": comp.get_text(strip=True) if comp else "",
            "location": loc.get_text(strip=True) if loc else location,
            # ZipRecruiter's public listing URL format (lvk=listing key) —
            # the old synthetic /jobs/{key} path returned 404.
            "job_url": f"https://www.ziprecruiter.com/jobs//j?lvk={key}",
            "description": "",  # list view has no full description; cascade/enrich handle it
            "date_posted": None,
        })
        if len(jobs) >= results_wanted:
            break

    if not jobs:
        logger.info(f"ZipRecruiter: 0 jobs matched for '{keyword}' (page fine)")
        return pd.DataFrame()
    logger.info(f"ZipRecruiter: {len(jobs)} jobs for '{keyword}'")
    return pd.DataFrame(jobs)
