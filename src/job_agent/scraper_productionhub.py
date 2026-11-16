"""
ProductionHub (productionhub.com/jobs) scraper.

ProductionHub is a crew/production job board with ~80 active listings. It's
behind Cloudflare, so plain `requests` returns 403 — we use Scrapling's
StealthyFetcher (Firefox + Camoufox stealth) like we do for ZipRecruiter.

Page structure (verified April 2026 via live probe):
  div.media
  ├── a (image link)
  └── div.media-body
      ├── div.media-header
      │   ├── h5.media-title > a  (title + job URL)
      │   └── span.media-meta     (location — itemprop=addressLocality/Region)
      ├── div.media-desc > a      (description snippet)
      └── footer
          ├── span[glyphicons-th-list]    (category)
          ├── span[glyphicons-time]       (employment type)
          ├── span[glyphicons-briefcase]  (COMPANY)
          └── span[glyphicons-calendar]   (POSTED DATE)

Job URLs follow the pattern: /job/{numeric_id}/{slug} (prepend the domain).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime

from bs4 import BeautifulSoup

from .ats_scraper import _matches_keywords

logger = logging.getLogger(__name__)

BASE_URL = "https://www.productionhub.com"
JOBS_URL = f"{BASE_URL}/jobs"


def _fetch_stealth(url: str, timeout_ms: int = 60000) -> BeautifulSoup | None:
    """Fetch via Scrapling StealthyFetcher (Firefox + Camoufox), return soup."""
    try:
        from scrapling import StealthyFetcher
        page = StealthyFetcher.fetch(
            url,
            headless=True,
            block_images=True,
            timeout=timeout_ms,
        )
        if not page:
            return None
        html = getattr(page, "body", "") or page.text
        if not html:
            return None
        return BeautifulSoup(html, "html.parser")
    except Exception as e:
        logger.warning(f"ProductionHub stealth fetch failed for {url}: {e}")
        return None


def _extract_footer_field(footer, icon_class: str) -> str:
    """Find the footer span whose inner `<span class='glyphicons ...'>` matches
    the given icon class, then return the trailing text (company name or date)."""
    if footer is None:
        return ""
    for span in footer.find_all("span", recursive=False):
        icon = span.find("span", class_=lambda c: c and icon_class in c)
        if icon is None:
            continue
        # The field is the text AFTER the icon — either the span's text minus
        # icon text, or the trailing text node, or a link's text.
        link = span.find("a")
        if link:
            text = link.get_text(strip=True)
            if text:
                return text
        # Fallback: take span text with icon text stripped
        all_text = span.get_text(" ", strip=True)
        icon_text = icon.get_text(strip=True)
        if icon_text and all_text.startswith(icon_text):
            all_text = all_text[len(icon_text):].strip()
        return all_text
    return ""


def _parse_date(raw: str) -> str:
    """Accept "4/19/2026" or "04/19/2026" or "4-19-2026"; return YYYY-MM-DD or ''."""
    if not raw:
        return ""
    raw = raw.strip()
    for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def _parse_job_card(card) -> dict | None:
    """Extract a single job's fields from a `<div class="media">` card."""
    title_link = card.select_one("h5.media-title a")
    if not title_link:
        return None
    title = title_link.get_text(strip=True)
    href = (title_link.get("href") or "").strip()
    if not title or not href:
        return None
    job_url = href if href.startswith("http") else f"{BASE_URL}{href}"

    # Location: addressLocality + addressRegion
    city = card.select_one('span[itemprop="addressLocality"]')
    region = card.select_one('span[itemprop="addressRegion"]')
    loc_parts = [x.get_text(strip=True) for x in (city, region) if x]
    location = ", ".join(p for p in loc_parts if p)

    # Description snippet
    desc_el = card.select_one("div.media-desc a") or card.select_one("div.media-desc")
    description = desc_el.get_text(" ", strip=True) if desc_el else ""

    # Footer fields: company + date
    footer = card.find("footer")
    company = _extract_footer_field(footer, "glyphicons-briefcase")
    date_raw = _extract_footer_field(footer, "glyphicons-calendar")

    return {
        "title": title,
        "company": company,
        "location": location,
        "job_url": job_url,
        "description": description,
        "date_posted": _parse_date(date_raw),
        "site": "productionhub",
        "department": "",
    }


def scrape_productionhub(keywords: list[str], max_pages: int = 3) -> list[dict]:
    """Scrape ProductionHub job listings, filter by keyword match on title.

    ProductionHub's pagination isn't well-documented; we try up to `max_pages`
    with `?page=N` and stop early when a page returns no new cards (indicating
    we've passed the last page).
    """
    all_jobs: list[dict] = []
    seen_urls: set[str] = set()

    for page_num in range(1, max_pages + 1):
        url = JOBS_URL if page_num == 1 else f"{JOBS_URL}?page={page_num}"
        soup = _fetch_stealth(url)
        if soup is None:
            break

        cards = soup.select("div.media")
        # Filter to only cards that have a title link (some media blocks are
        # site chrome, not job cards).
        job_cards = [c for c in cards if c.select_one("h5.media-title a[href*='/job/']")]
        if not job_cards:
            break  # Past the last page

        new_on_page = 0
        for card in job_cards:
            job = _parse_job_card(card)
            if not job:
                continue
            if job["job_url"] in seen_urls:
                continue
            seen_urls.add(job["job_url"])
            new_on_page += 1  # new card PARSED — pagination continues even on a no-match page
            if not _matches_keywords(job["title"], keywords):
                continue
            all_jobs.append(job)

        # If we didn't see any *new* URLs on this page, we've wrapped — stop.
        if new_on_page == 0 and page_num > 1:
            break

    logger.info(f"ProductionHub: {len(all_jobs)} matching jobs from {len(seen_urls)} total scanned")
    return all_jobs
