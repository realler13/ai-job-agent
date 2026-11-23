
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import pandas as pd
from bs4 import BeautifulSoup
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

from .candidate_profile import SEARCH, glassdoor_city_id

# Setup module logger
logger = logging.getLogger(__name__)

# Per-page crawl timeout (seconds). If a single page takes longer, skip it.
PER_PAGE_TIMEOUT = 30
# Total budget for the entire enrichment batch (seconds). Prevents runaway sessions.
TOTAL_ENRICHMENT_TIMEOUT = 300  # 5 minutes

# Glassdoor search-page render budget (seconds).
GLASSDOOR_SEARCH_TIMEOUT = 45

# Glassdoor ignores locKeyword: a search with a city name in it renders the
# nationwide results page ("... jobs in United States", canonical
# us-<keyword>-jobs-...). Only Glassdoor's own location id narrows a search
# (locT=C = city, locId = the id in its city links, ...-SRCH_IL.0,13_IC<id>_...).
# The profile supplies the home city's id (search.glassdoor_city_id); with no
# id there is nothing here and Glassdoor is searched nationwide only.
def _home_location_ids() -> dict:
    """{lower-case home location: (locT, locId)} from the profile, or {}."""
    city = glassdoor_city_id()
    home = str(SEARCH.get("locations", {}).get("home") or "").strip().lower()
    return {home: city} if city and home else {}


GLASSDOOR_LOCATION_IDS = _home_location_ids()


def search_locations() -> list:
    """The locations each Glassdoor keyword is searched in, in order: the home
    location when the profile gives its Glassdoor id, then nationwide (None).
    Without an id, nationwide only."""
    home = SEARCH.get("locations", {}).get("home")
    if home and str(home).strip().lower() in GLASSDOOR_LOCATION_IDS:
        return [home, None]
    return [None]


# Circuit breaker for posting pages. Glassdoor answers every posting page with
# a Cloudflare JS challenge ("Humans only") even when the search list renders
# fine, with a fresh browser as well as right after a successful search in the
# same browser. So after this many blocked pages in a row we stop, log one
# line, and hold off for BREAKER_HOLD_SECONDS: long enough to cover the rest of
# the run's Glassdoor batches, short enough that the next run (or a search
# from the app, same process) tries again.
BLOCKED_PAGES_TO_TRIP = 3
BREAKER_HOLD_SECONDS = 30 * 60
_breaker_open_until = 0.0  # time.monotonic() deadline; 0 = closed


def _search_url(keyword: str, location: str | None) -> str:
    """Glassdoor search URL for a keyword in a city it has an id for.

    No location (None or "") is the nationwide search, on purpose: it is the
    only lane that returns remote roles and roles outside the home city. A
    city without an id still gets the old locKeyword search (also nationwide
    in practice) and one warning per search.
    """
    base = f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={quote_plus(keyword)}"
    if not (location or "").strip():
        return base
    loc = GLASSDOOR_LOCATION_IDS.get((location or "").strip().lower())
    if loc:
        return f"{base}&locT={loc[0]}&locId={loc[1]}"
    logger.warning(f"Glassdoor: no location id for {location!r} — results will be nationwide")
    return f"{base}&locKeyword={quote_plus(location)}"


async def _scrape_glassdoor_search(keyword: str, location: str, results_wanted: int) -> List[Dict[str, Any]]:
    """Render a Glassdoor jobs search with a real browser and extract job cards.

    jobspy's Glassdoor API path is blocked (HTTP 400 — needs csrf/cookie flow);
    crawl4ai's Chromium renders the public search page, which bypasses the API
    block. Selectors verified live 2026-06: li[data-test=jobListing] cards.
    """
    url = _search_url(keyword, location)
    bc = BrowserConfig(headless=True, verbose=False)
    rc = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, page_timeout=GLASSDOOR_SEARCH_TIMEOUT * 1000)
    async with AsyncWebCrawler(config=bc) as crawler:
        result = await asyncio.wait_for(
            crawler.arun(url=url, config=rc), timeout=GLASSDOOR_SEARCH_TIMEOUT + 20
        )
    html = result.html or ""
    if len(html) < 50_000:
        # Real Glassdoor result pages are several hundred KB; a small response
        # is a block/interstitial shell — treat as an error (breaker counts it).
        raise RuntimeError(f"Glassdoor returned a {len(html)}-byte shell (likely blocked)")
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select('li[data-test="jobListing"]')

    jobs: List[Dict[str, Any]] = []
    seen: set = set()
    for c in cards:
        t = c.select_one('a[data-test="job-title"]')
        if not t:
            continue
        title = t.get_text(strip=True)
        href = (t.get("href") or "").strip()
        if not title or not href or href in seen:
            continue
        seen.add(href)
        comp = c.select_one('[class*="EmployerProfile_compactEmployerName"], [data-test="employer-short-name"]')
        loc = c.select_one('[data-test="emp-location"]')
        jobs.append({
            "site": "glassdoor",
            "title": title,
            "company": comp.get_text(strip=True) if comp else "",
            "location": loc.get_text(strip=True) if loc else (location or ""),
            "job_url": href if href.startswith("http") else f"https://www.glassdoor.com{href}",
            "description": "",  # list view; cascade/enrichment handle it
            "date_posted": None,
        })
        if len(jobs) >= results_wanted:
            break
    return jobs


def scrape_glassdoor(
    keyword: str,
    location: Optional[str] = None,
    results_wanted: int = 30,
) -> Optional["pd.DataFrame"]:
    """Scrape one Glassdoor search via crawl4ai. Returns a DataFrame or None.

    location=None (the default) searches nationwide. A location is narrowed
    only when GLASSDOOR_LOCATION_IDS has its id (the home location, from
    search.glassdoor_city_id); see search_locations().
    SLOW (real browser, ~30-45s/query) — callers must bound the keyword set.
    """
    if not CRAWL4AI_AVAILABLE:
        logger.warning("Glassdoor: crawl4ai not available — skipping")
        return None
    logger.info(f"Glassdoor: searching '{keyword}' in {location or 'the whole US'} (crawl4ai)...")
    try:
        jobs = asyncio.run(_scrape_glassdoor_search(keyword, location, results_wanted))
    except Exception as e:
        logger.warning(f"Glassdoor '{keyword}' failed: {e}")
        return None
    if not jobs:
        # Rendered fine, zero cards — legitimate empty result, NOT a block.
        logger.info(f"Glassdoor: 0 jobs for '{keyword}' (page rendered)")
        return pd.DataFrame()
    logger.info(f"Glassdoor: {len(jobs)} jobs for '{keyword}'")
    return pd.DataFrame(jobs)


async def _crawl_one(crawler, job, run_config) -> str:
    """Crawl a single job URL with a per-page timeout.

    Returns "ok", "blocked" (the anti-bot wall answered) or "failed".
    """
    url = job.get("job_url")
    if not url:
        return "failed"

    target_url = job.get("job_url_direct") or url
    logger.info(f"🕸️ Crawling: {job.get('title')}...")

    try:
        result = await asyncio.wait_for(
            crawler.arun(url=target_url, config=run_config),
            timeout=PER_PAGE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(f"⏱️ Timeout ({PER_PAGE_TIMEOUT}s) crawling {target_url} — skipping")
        return "failed"
    except Exception as e:
        logger.error(f"❌ Error crawling job {job.get('id', 'unknown')}: {e}")
        return "failed"

    if result.success and result.markdown and result.markdown.raw_markdown:
        job["description"] = result.markdown.raw_markdown
        job["description_source"] = "crawl4ai_enriched"
        return "ok"
    error = result.error_message or ""
    if "anti-bot" in error:
        # crawl4ai's own wording: "Blocked by anti-bot protection: <reason>".
        # The breaker logs these as one line, so no warning per page.
        logger.debug(f"Blocked crawling {target_url}: {error}")
        return "blocked"
    logger.warning(f"⚠️ Failed to crawl {target_url}: {error}")
    return "failed"


async def enrich_jobs_with_crawl4ai(jobs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich a list of job dictionaries by visiting their URLs with Crawl4AI
    and replacing the description with clean Markdown.

    Safeguards:
    - Each page gets PER_PAGE_TIMEOUT seconds before being skipped.
    - The entire batch gets TOTAL_ENRICHMENT_TIMEOUT seconds before aborting.
    - BLOCKED_PAGES_TO_TRIP blocked pages in a row stop the batch, and later
      batches skip for BREAKER_HOLD_SECONDS (see the breaker note above).
    """
    global _breaker_open_until
    if not jobs_list:
        return []

    if time.monotonic() < _breaker_open_until:
        logger.debug(f"Crawl4AI: posting pages blocked earlier this run — not enriching {len(jobs_list)} jobs")
        return jobs_list

    logger.info(f"🕷️ Crawl4AI: Enriching {len(jobs_list)} jobs...")

    if not CRAWL4AI_AVAILABLE:
        logger.warning("⚠️ Crawl4AI not available/installed. Skipping enrichment.")
        return jobs_list

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=50,
        exclude_external_links=True,
    )

    enriched_count = 0
    blocked_in_a_row = 0
    deadline = asyncio.get_event_loop().time() + TOTAL_ENRICHMENT_TIMEOUT

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for job in jobs_list:
                # Check total budget
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    logger.warning(
                        f"⏱️ Total enrichment budget ({TOTAL_ENRICHMENT_TIMEOUT}s) exhausted "
                        f"— skipping remaining {len(jobs_list) - enriched_count} jobs"
                    )
                    break

                status = await _crawl_one(crawler, job, run_config)
                if status == "ok":
                    enriched_count += 1
                blocked_in_a_row = blocked_in_a_row + 1 if status == "blocked" else 0
                if blocked_in_a_row >= BLOCKED_PAGES_TO_TRIP:
                    _breaker_open_until = time.monotonic() + BREAKER_HOLD_SECONDS
                    logger.warning(
                        f"Crawl4AI: {blocked_in_a_row} posting pages in a row blocked by anti-bot "
                        f"protection — no more posting pages for {BREAKER_HOLD_SECONDS // 60} min; "
                        f"these jobs keep the list data only"
                    )
                    break

    except Exception as e:
        logger.error(f"❌ Crawl4AI session error: {e} — returning partially enriched results")

    logger.info(f"✅ Crawl4AI: Enriched {enriched_count}/{len(jobs_list)} jobs")
    return jobs_list
