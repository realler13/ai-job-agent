
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd
import requests

from .ats_scraper import _matches_keywords
from .utils import normalize_date_posted

logger = logging.getLogger(__name__)

# Shared HTTP session for connection reuse
_session = requests.Session()
_session.headers.update({
    "User-Agent": "AIJobAgent/1.0 (job search assistant)",
    "Accept": "application/rss+xml, application/xml, text/xml",
})

REQUEST_TIMEOUT = 15


def _parse_rss_items(xml_text: str) -> list[dict]:
    """Parse RSS 2.0 XML and return list of raw item dicts."""
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"RSS XML parse error: {e}")
        return []

    # RSS 2.0: /rss/channel/item
    channel = root.find("channel")
    if channel is None:
        # Try Atom or direct item list
        channel = root

    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()

        # Collect all <category> elements (used by EntertainmentCareers for city/state/type)
        categories = [c.text.strip() for c in item.findall("category") if c.text]

        items.append({
            "title": title,
            "link": link,
            "description": description,
            "pub_date": pub_date,
            "categories": categories,
        })

    return items


def _parse_rss_date(date_str: str) -> str:
    """Parse RSS pubDate (RFC 822) to YYYY-MM-DD."""
    if not date_str:
        return ""
    # RFC 822 formats commonly found in RSS
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",      # "Mon, 15 Jan 2026 10:30:00 +0000"
        "%a, %d %b %Y %H:%M:%S %Z",       # "Mon, 15 Jan 2026 10:30:00 GMT"
        "%a, %d %b %Y %H:%M:%S",          # No timezone
        "%Y-%m-%dT%H:%M:%S%z",            # ISO 8601
        "%Y-%m-%dT%H:%M:%S",              # ISO 8601 no tz
        "%Y-%m-%d",                        # Already formatted
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Fallback to normalize_date_posted
    return normalize_date_posted(date_str)


def scrape_entertainmentcareers_rss(
    keywords: list[str],
    feed_ids: list[str] | None = None,
) -> list[dict]:
    """
    Scrape EntertainmentCareers.net RSS feeds.

    Args:
        keywords: Keywords to filter job titles
        feed_ids: Category feed IDs (e.g. ["ecnjcat119", "ecnjcat168"]).
                  If None, uses ENTERTAINMENTCAREERS_FEEDS from config.

    Returns:
        List of standardized job dicts
    """
    if feed_ids is None:
        from .config import ENTERTAINMENTCAREERS_FEEDS
        feed_ids = ENTERTAINMENTCAREERS_FEEDS

    all_jobs = []
    seen_links = set()

    for feed_id in feed_ids:
        url = f"https://www.entertainmentcareers.net/{feed_id}"
        try:
            resp = _session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"EntertainmentCareers RSS error ({feed_id}): {e}")
            time.sleep(0.3)
            continue

        items = _parse_rss_items(resp.text)

        for item in items:
            title = item["title"]
            link = item["link"]

            if not title or not link:
                continue
            if link in seen_links:
                continue

            if not _matches_keywords(title, keywords):
                continue

            seen_links.add(link)

            # Extract location from categories (typically city, state, type)
            location = ""
            categories = item.get("categories", [])
            # Categories often contain: city, state, job type
            location_parts = [c for c in categories if c and c not in ("Full Time", "Part Time", "Freelance", "Contract", "Internship")]
            if location_parts:
                location = ", ".join(location_parts[:2])  # city, state

            all_jobs.append({
                "title": title,
                "company": "",  # Not provided in RSS feed
                "location": location,
                "job_url": link,
                "description": item["description"],
                "date_posted": _parse_rss_date(item["pub_date"]),
                "site": "entertainmentcareers",
                "department": "",
            })

        time.sleep(0.3)  # Rate limit between feeds

    logger.info(f"EntertainmentCareers RSS: {len(all_jobs)} matching jobs from {len(feed_ids)} feeds")
    return all_jobs


def scrape_gamescareer_rss(keywords: list[str]) -> list[dict]:
    """
    Scrape Games-Career.com RSS feed for producing/production roles.

    Args:
        keywords: Keywords to filter job titles

    Returns:
        List of standardized job dicts
    """
    url = "https://www.games-career.com/rss/Joboffer/Productmanagement_and_Producing"
    try:
        resp = _session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Games-Career RSS error: {e}")
        return []

    items = _parse_rss_items(resp.text)
    results = []

    for item in items:
        title = item["title"]
        link = item["link"]

        if not title or not link:
            continue

        if not _matches_keywords(title, keywords):
            continue

        results.append({
            "title": title,
            "company": "",
            "location": "",
            "job_url": link,
            "description": item["description"],
            "date_posted": _parse_rss_date(item["pub_date"]),
            "site": "gamescareer",
            "department": "",
        })

    logger.info(f"Games-Career RSS: {len(results)} matching jobs from {len(items)} total")
    return results


AQUENT_FEED_URL = "https://aquent.com/feeds/jobs.xml"


def scrape_aquent_feed(keywords: list[str]) -> list[dict]:
    """Aquent — global creative/marketing staffing agency. First-party XML
    feed, no auth, refreshed every ~15 min; a contract/temp lane the ATS
    boards don't cover.

    Filter: US items in the profile's home state (search.locations
    state_code / state_name) OR fully remote, then
    keyword match on title+segment. Titles carry a "[211286]" job-id suffix — strip it.
    """
    import re
    from .config import HOME_STATE_CODE, HOME_STATE_NAME
    # Aquent writes the state as a code ("WA") or, now and then, a name.
    home_state = {HOME_STATE_CODE, HOME_STATE_NAME} - {""}
    try:
        resp = requests.get(AQUENT_FEED_URL, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as e:
        logger.warning(f"Aquent feed failed: {e}")
        return []

    jobs = []
    for item in root.findall(".//item"):
        country = (item.findtext("location/country") or "").strip()
        if country != "US":
            continue
        state = (item.findtext("location/state") or "").strip()
        city = (item.findtext("location/city") or "").strip()
        # ONLY "Fully remote" counts as remote — hybrid/partial remote in an
        # out-of-market city still means an on-site commute.
        remote = (item.findtext("remotetype") or "").strip().lower() == "fully remote"
        in_home = state.lower() in home_state
        if not (in_home or remote):
            continue

        title = re.sub(r"\s*\[\d+\]\s*$", "", (item.findtext("title") or "").strip())
        segment = (item.findtext("segment") or "").strip()
        if not _matches_keywords(f"{title} {segment}", keywords):
            continue

        desc_html = item.findtext("description") or ""
        desc = re.sub(r"<[^>]+>", " ", desc_html)
        desc = re.sub(r"\s+", " ", desc).strip()
        # Fully-remote items must read "Remote" for our location gate even
        # when Aquent lists a base city — keep the city as context in desc.
        if remote and not in_home:
            location = "Remote"
            if city:
                desc_note = f"[Team base: {city}, {state}] "
            else:
                desc_note = ""
        else:
            location = f"{city}, {state}" if city else (state or "Remote")
            desc_note = ""
        placement = (item.findtext("placement_type") or "").strip()
        salary = (item.findtext("salary") or "").strip()
        if placement:
            desc = f"[{placement} via Aquent] " + desc_note + desc
        if salary:
            desc = f"[Salary: {salary}] " + desc

        jobs.append({
            "title": title,
            "company": "Aquent (staffing)",
            "location": location,
            "job_url": (item.findtext("url") or "").strip(),
            "description": desc[:8000],
            "date_posted": normalize_date_posted(item.findtext("pubDate") or ""),
            "site": "aquent",
        })
    logger.info(f"Aquent feed: {len(jobs)} matching CA/remote US jobs")
    return jobs


def scrape_rss_feeds(
    keywords: list[str],
    progress_callback=None,
) -> pd.DataFrame:
    """
    Orchestrator: scrape all RSS-based industry feeds.

    Args:
        keywords: Keywords to filter job titles
        progress_callback: Optional callback(message, progress_float)

    Returns:
        DataFrame of all matching jobs
    """
    all_jobs = []

    if progress_callback:
        progress_callback("RSS: Scanning EntertainmentCareers.net...", 0.78)

    jobs = scrape_entertainmentcareers_rss(keywords)
    all_jobs.extend(jobs)

    if progress_callback:
        progress_callback("RSS: Scanning Games-Career.com...", 0.79)

    jobs = scrape_gamescareer_rss(keywords)
    all_jobs.extend(jobs)

    if progress_callback:
        progress_callback("RSS: Scanning Aquent (creative staffing)...", 0.795)

    jobs = scrape_aquent_feed(keywords)
    all_jobs.extend(jobs)

    if progress_callback:
        progress_callback(f"RSS feeds: found {len(all_jobs)} matching jobs", 0.80)

    if not all_jobs:
        return pd.DataFrame()

    return pd.DataFrame(all_jobs)
