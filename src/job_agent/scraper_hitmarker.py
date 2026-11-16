"""
Hitmarker (hitmarker.net) scraper — gaming/esports jobs board.

Hitmarker is the dominant gaming-industry job board with ~5,000 active
listings. Plain HTTP works (no Cloudflare). Each detail page embeds a
schema.org `JobPosting` JSON-LD block, which is our cleanest extraction
target — far more reliable than scraping DOM selectors.

Strategy:
1. Pull the sitemap-jobs.xml index (p1, p2, ...).
2. Filter URLs by keyword substring match on the slug — cheap,
   no HTTP per URL.
3. For each matching URL, fetch the detail page and parse the JSON-LD
   JobPosting blob.
4. Cap at max_jobs to prevent runaway runs.

We deliberately don't use the site's JS-rendered search — the sitemap
is cleaner, stable, and doesn't need a headless browser.
"""
from __future__ import annotations

import json
import logging
import re
import time
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup

from .ats_scraper import _matches_keywords
from .utils import normalize_date_posted

logger = logging.getLogger(__name__)

SITEMAP_INDEX_URL = "https://hitmarker.net/sitemap.xml"
REQUEST_TIMEOUT = 20
PER_REQUEST_DELAY = 0.5  # seconds between detail fetches

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})


def _fetch_job_sitemap_urls() -> list[str]:
    """Walk the sitemap index and return all job detail URLs."""
    urls: list[str] = []

    try:
        resp = _session.get(SITEMAP_INDEX_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Hitmarker sitemap index fetch failed: {e}")
        return []

    # Strip the default namespace so ElementTree.iter() works without prefixes.
    idx_text = re.sub(r' xmlns="[^"]+"', "", resp.text, count=1)
    try:
        idx_root = ET.fromstring(idx_text)
    except ET.ParseError as e:
        logger.warning(f"Hitmarker sitemap index parse error: {e}")
        return []

    # Find all child sitemaps whose URL contains /sitemap-jobs.xml
    job_sitemap_urls = [
        loc.text for loc in idx_root.iter("loc")
        if loc.text and "sitemap-jobs.xml" in loc.text
    ]

    for sm_url in job_sitemap_urls:
        try:
            r = _session.get(sm_url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            logger.warning(f"Hitmarker sub-sitemap fetch failed ({sm_url}): {e}")
            continue

        sub_text = re.sub(r' xmlns="[^"]+"', "", r.text, count=1)
        try:
            sub_root = ET.fromstring(sub_text)
        except ET.ParseError as e:
            logger.warning(f"Hitmarker sub-sitemap parse error ({sm_url}): {e}")
            continue

        for loc in sub_root.iter("loc"):
            if loc.text and "/jobs/" in loc.text:
                urls.append(loc.text)

    return urls


def _extract_job_posting_ld(soup: BeautifulSoup) -> dict | None:
    """Find the schema.org JobPosting JSON-LD block on a detail page."""
    for script in soup.select('script[type="application/ld+json"]'):
        raw = (script.string or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        # JSON-LD may be a single object, a list, or a graph
        candidates = []
        if isinstance(data, list):
            candidates.extend(data)
        elif isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                candidates.extend(data["@graph"])
            else:
                candidates.append(data)
        for c in candidates:
            if isinstance(c, dict) and c.get("@type") == "JobPosting":
                return c
    return None


def _parse_jobposting(data: dict, url: str) -> dict:
    """Turn a schema.org JobPosting dict into our standardized job record."""
    title = str(data.get("title") or "").strip()

    # hiringOrganization may be a dict {"@type": "Organization", "name": "X"}
    # or a string
    hiring_org = data.get("hiringOrganization")
    company = ""
    if isinstance(hiring_org, dict):
        company = str(hiring_org.get("name") or "").strip()
    elif isinstance(hiring_org, str):
        company = hiring_org.strip()

    # jobLocation can be a dict, list, or missing. Prefer address.addressLocality
    # + addressRegion. For "TELECOMMUTE" (remote) → "Remote".
    location = ""
    jl = data.get("jobLocation")
    job_loc_type = str(data.get("jobLocationType") or "").upper()
    if isinstance(jl, list) and jl:
        jl = jl[0]
    if isinstance(jl, dict):
        addr = jl.get("address") or {}
        if isinstance(addr, dict):
            city = str(addr.get("addressLocality") or "").strip()
            region = str(addr.get("addressRegion") or "").strip()
            country = str(addr.get("addressCountry") or "").strip()
            parts = [p for p in (city, region, country) if p]
            location = ", ".join(parts[:2]) if parts else ""
    if not location and job_loc_type == "TELECOMMUTE":
        location = "Remote"

    # description is HTML
    desc_html = str(data.get("description") or "")
    description = BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True) if desc_html else ""

    # employmentType: common values "FULL_TIME", "CONTRACT", etc.
    employment_type = str(data.get("employmentType") or "")

    return {
        "title": title,
        "company": company,
        "location": location,
        "job_url": url,
        "description": description,
        "date_posted": normalize_date_posted(str(data.get("datePosted") or "")),
        "site": "hitmarker",
        "department": employment_type,  # repurpose the column for employment type
    }


def scrape_hitmarker(keywords: list[str], max_jobs: int = 100) -> list[dict]:
    """Scrape Hitmarker job listings via sitemap + JSON-LD on detail pages.

    Args:
        keywords: Keywords to match against URL slugs (cheap pre-filter).
        max_jobs: Cap on detail fetches to avoid runaway runs (default 100).

    Returns:
        List of standardized job dicts.
    """
    all_urls = _fetch_job_sitemap_urls()
    if not all_urls:
        logger.warning("Hitmarker: no sitemap URLs retrieved")
        return []

    # Cheap slug-based pre-filter. _matches_keywords is case-insensitive.
    # Slugs are hyphen-separated; normalize to spaces so "content producer"
    # matches "senior-content-producer-sky".
    def slug_text(url: str) -> str:
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        # Strip the trailing numeric job ID
        slug = re.sub(r"-\d{6,}$", "", slug)
        return slug.replace("-", " ")

    matches = [u for u in all_urls if _matches_keywords(slug_text(u), keywords)]
    logger.info(f"Hitmarker: {len(matches)} slug-matched URLs from {len(all_urls)} total")

    if not matches:
        return []

    # Cap to keep request volume reasonable
    # Sitemaps enumerate oldest->newest: capping the FIRST N re-fetched the
    # same old listings forever while new postings at the tail were never
    # detail-fetched. Take the LAST N, newest first.
    matches = list(reversed(matches[-max_jobs:]))

    jobs: list[dict] = []
    for url in matches:
        try:
            r = _session.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            logger.warning(f"Hitmarker detail fetch failed ({url}): {e}")
            time.sleep(PER_REQUEST_DELAY)
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        ld = _extract_job_posting_ld(soup)
        if not ld:
            logger.debug(f"Hitmarker: no JobPosting JSON-LD on {url}")
            time.sleep(PER_REQUEST_DELAY)
            continue

        job = _parse_jobposting(ld, url)
        if job["title"]:
            jobs.append(job)

        time.sleep(PER_REQUEST_DELAY)

    logger.info(f"Hitmarker: {len(jobs)} jobs parsed from {len(matches)} detail fetches")
    return jobs
