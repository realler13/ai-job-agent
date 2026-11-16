
import asyncio
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import pandas as pd
from bs4 import BeautifulSoup
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

from .candidate_profile import SEARCH

# Setup module logger
logger = logging.getLogger(__name__)

# Per-page crawl timeout (seconds). If a single page takes longer, skip it.
PER_PAGE_TIMEOUT = 30
# Total budget for the entire enrichment batch (seconds). Prevents runaway sessions.
TOTAL_ENRICHMENT_TIMEOUT = 300  # 5 minutes

# Glassdoor search-page render budget (seconds).
GLASSDOOR_SEARCH_TIMEOUT = 45


async def _scrape_glassdoor_search(keyword: str, location: str, results_wanted: int) -> List[Dict[str, Any]]:
    """Render a Glassdoor jobs search with a real browser and extract job cards.

    jobspy's Glassdoor API path is blocked (HTTP 400 — needs csrf/cookie flow);
    crawl4ai's Chromium renders the public search page, which bypasses the API
    block. Selectors verified live 2026-06: li[data-test=jobListing] cards.
    """
    url = (
        f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={quote_plus(keyword)}"
        f"&locKeyword={quote_plus(location)}"
    )
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
            "location": loc.get_text(strip=True) if loc else location,
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

    ``location`` defaults to the profile's home location
    (search.locations.home).
    SLOW (real browser, ~30-45s/query) — callers must bound the keyword set.
    """
    location = location or SEARCH["locations"]["home"]
    if not CRAWL4AI_AVAILABLE:
        logger.warning("Glassdoor: crawl4ai not available — skipping")
        return None
    logger.info(f"Glassdoor: searching '{keyword}' in {location} (crawl4ai)...")
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


async def _crawl_one(crawler, job, run_config):
    """Crawl a single job URL with a per-page timeout."""
    url = job.get("job_url")
    if not url:
        return False

    target_url = job.get("job_url_direct") or url
    logger.info(f"🕸️ Crawling: {job.get('title')}...")

    try:
        result = await asyncio.wait_for(
            crawler.arun(url=target_url, config=run_config),
            timeout=PER_PAGE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(f"⏱️ Timeout ({PER_PAGE_TIMEOUT}s) crawling {target_url} — skipping")
        return False
    except Exception as e:
        logger.error(f"❌ Error crawling job {job.get('id', 'unknown')}: {e}")
        return False

    if result.success and result.markdown and result.markdown.raw_markdown:
        job["description"] = result.markdown.raw_markdown
        job["description_source"] = "crawl4ai_enriched"
        return True
    else:
        logger.warning(f"⚠️ Failed to crawl {target_url}: {result.error_message}")
        return False


async def enrich_jobs_with_crawl4ai(jobs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich a list of job dictionaries by visiting their URLs with Crawl4AI
    and replacing the description with clean Markdown.

    Safeguards:
    - Each page gets PER_PAGE_TIMEOUT seconds before being skipped.
    - The entire batch gets TOTAL_ENRICHMENT_TIMEOUT seconds before aborting.
    """
    if not jobs_list:
        return []

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

                if await _crawl_one(crawler, job, run_config):
                    enriched_count += 1

    except Exception as e:
        logger.error(f"❌ Crawl4AI session error: {e} — returning partially enriched results")

    logger.info(f"✅ Crawl4AI: Enriched {enriched_count}/{len(jobs_list)} jobs")
    return jobs_list
