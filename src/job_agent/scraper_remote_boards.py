"""
Remote-work job-board scrapers: We Work Remotely + Remote OK.

Both are remote-first boards with some signal for media / content / producer
roles. WWR exposes RSS feeds per category; Remote OK exposes a public JSON
API at /api. No auth needed for either.

Design notes:
- WWR titles are formatted as "Company: Role" — split on first ": ".
- Remote OK's first element is usually a legal/notice header — filter to dicts
  that actually carry `position`.
- Keyword filtering uses the shared `_matches_keywords` helper so behavior
  matches other scrapers (ATS, industry HTML, RSS).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd
import requests

from .ats_scraper import _matches_keywords, _strip_html
from .scraper_rss import _parse_rss_items, _parse_rss_date

logger = logging.getLogger(__name__)

_session = requests.Session()
_session.headers.update({
    "User-Agent": "AIJobAgent/1.0 (+job search assistant)",
    "Accept": "application/rss+xml, application/json, application/xml;q=0.9,*/*;q=0.8",
})

REQUEST_TIMEOUT = 20

# WWR categories likely to contain media/producer/content roles.
# Their category list doesn't include a dedicated media bucket, so we watch
# the catchall plus sales/marketing and product (producer crossover).
WEWORKREMOTELY_FEEDS = [
    "all-other-remote-jobs",
    "remote-sales-and-marketing-jobs",
    "remote-product-jobs",
]


def scrape_weworkremotely_rss(keywords: list[str]) -> list[dict]:
    """Scrape WWR category RSS feeds; return keyword-matched job dicts.

    WWR job titles in the RSS feed are formatted as "Company: Role" — we
    split on the first ": " so `company` and `title` land in the right
    columns downstream.
    """
    all_jobs: list[dict] = []
    seen_links: set[str] = set()

    for feed_slug in WEWORKREMOTELY_FEEDS:
        url = f"https://weworkremotely.com/categories/{feed_slug}.rss"
        try:
            resp = _session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"WeWorkRemotely RSS ({feed_slug}) error: {e}")
            time.sleep(0.3)
            continue

        items = _parse_rss_items(resp.text)

        for item in items:
            title = item["title"]
            link = item["link"]
            if not title or not link or link in seen_links:
                continue
            if not _matches_keywords(title, keywords):
                continue

            seen_links.add(link)

            # "Acme Corp: Senior Video Producer" -> company="Acme Corp", role="Senior Video Producer"
            company = ""
            role = title
            if ": " in title:
                company, role = title.split(": ", 1)

            all_jobs.append({
                "title": role.strip(),
                "company": company.strip(),
                "location": "Remote",
                "job_url": link,
                "description": item["description"],
                "date_posted": _parse_rss_date(item["pub_date"]),
                "site": "weworkremotely",
                "department": "",
            })

        time.sleep(0.3)  # be polite between feeds

    logger.info(
        f"WeWorkRemotely: {len(all_jobs)} matching jobs from {len(WEWORKREMOTELY_FEEDS)} feeds"
    )
    return all_jobs


def scrape_remoteok_api(keywords: list[str]) -> list[dict]:
    """Scrape Remote OK's public JSON API and filter by keyword.

    The API returns an array whose first element is typically a legal/notice
    object (no `position` field). We filter to real job dicts before matching.
    Matches against the position title first; falls back to tags.
    """
    url = "https://remoteok.com/api"
    try:
        resp = _session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Remote OK API error: {e}")
        return []

    if not isinstance(data, list):
        logger.warning(f"Remote OK API returned unexpected shape: {type(data).__name__}")
        return []

    jobs_raw = [j for j in data if isinstance(j, dict) and j.get("position")]

    results: list[dict] = []
    for j in jobs_raw:
        title = str(j.get("position") or "").strip()
        if not title:
            continue

        # Primary: match on position; fallback: match on tags
        if not _matches_keywords(title, keywords):
            tags = j.get("tags") or []
            tag_text = " ".join(str(t) for t in tags)
            if not _matches_keywords(tag_text, keywords):
                continue

        # Remote OK date is ISO 8601 with 'Z' suffix — convert to YYYY-MM-DD
        date_str = str(j.get("date") or "")
        date_posted = ""
        if date_str:
            try:
                date_posted = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00")
                ).strftime("%Y-%m-%d")
            except Exception:
                date_posted = ""

        location = str(j.get("location") or "").strip() or "Remote"

        job_url = str(j.get("url") or j.get("apply_url") or "").strip()
        if not job_url:
            continue  # Skip records with no usable URL (can't visit or dedup)

        results.append({
            "title": title,
            "company": str(j.get("company") or "").strip(),
            "location": location,
            "job_url": job_url,
            "description": _strip_html(str(j.get("description") or "")),  # API returns raw HTML
            "date_posted": date_posted,
            "site": "remoteok",
            "department": "",
        })

    logger.info(f"Remote OK: {len(results)} matching jobs from {len(jobs_raw)} total")
    return results


def scrape_remote_boards(
    keywords: list[str],
    progress_callback=None,
) -> pd.DataFrame:
    """Orchestrator: scrape all remote-work boards (WWR + Remote OK).

    Mirrors the shape of scrape_rss_feeds / scrape_industry_html_boards so
    main.py can plug it in the same way.
    """
    all_jobs: list[dict] = []

    if progress_callback:
        progress_callback("Remote: Scanning WeWorkRemotely...", 0.81)
    all_jobs.extend(scrape_weworkremotely_rss(keywords))

    if progress_callback:
        progress_callback("Remote: Scanning Remote OK...", 0.82)
    all_jobs.extend(scrape_remoteok_api(keywords))

    if progress_callback:
        progress_callback(f"Remote boards: found {len(all_jobs)} matching jobs", 0.83)

    if not all_jobs:
        return pd.DataFrame()
    return pd.DataFrame(all_jobs)
