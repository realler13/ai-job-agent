"""
Recruitee, Workable, and Teamtailor ATS scrapers.

Follows the same pattern as ats_scraper.py (Greenhouse/Lever):
each function takes (token, company, keywords) and returns list[dict].

Recruitee API: GET https://{token}.recruitee.com/api/offers/
Workable API: GET https://apply.workable.com/api/v3/accounts/{token}/jobs
Teamtailor API: GET https://{token}.teamtailor.com/jobs (HTML) — but most
    Teamtailor sites expose an undocumented JSON feed at /api/v1/jobs.
"""

import logging
import re

import requests
from bs4 import BeautifulSoup

from .ats_scraper import _matches_keywords, _strip_html, report_board_fetch
from .utils import normalize_date_posted

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recruitee
# ---------------------------------------------------------------------------

def _recruitee_job_url(token: str, offer: dict) -> str:
    """Job URL: prefer careers_url, fall back to the constructed URL."""
    job_url = offer.get("careers_url", "")
    if not job_url:
        slug = offer.get("slug", "")
        if slug:
            job_url = f"https://{token}.recruitee.com/o/{slug}"
    return job_url


def scrape_recruitee(token: str, company: str, keywords: list[str]) -> list[dict]:
    """
    Scrape jobs from a Recruitee public job board.

    Endpoint: GET https://{token}.recruitee.com/api/offers/
    Returns JSON with ``offers`` array. Each offer has title, location,
    department, description (HTML), created_at, careers_url, etc.

    Args:
        token: The Recruitee subdomain (e.g. "fly-io")
        company: Company display name
        keywords: List of keyword phrases to filter by title

    Returns:
        List of standardized job dicts
    """
    url = f"https://{token}.recruitee.com/api/offers/"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Recruitee API error for {company} ({token}): {e}")
        report_board_fetch("recruitee", token, ok=False, complete=False, total_count=0, links=[])
        return []

    offers = data.get("offers", [])
    if not isinstance(offers, list):
        logger.warning(f"Recruitee ({company}): unexpected response format")
        report_board_fetch("recruitee", token, ok=False, complete=False, total_count=0, links=[])
        return []

    # Still-listed record (Phase 27c): closed/archived offers are not listed.
    _open = [o for o in offers if isinstance(o, dict)
             and str(o.get("status", "") or "").lower() not in ("closed", "archived")]
    report_board_fetch("recruitee", token, ok=True, complete=True, total_count=len(_open),
                       links=[(_recruitee_job_url(token, o), o.get("title", "")) for o in _open])

    results = []

    for offer in offers:
        title = offer.get("title", "")
        if not _matches_keywords(title, keywords):
            continue

        # Skip closed/archived offers
        status = offer.get("status", "").lower()
        if status in ("closed", "archived"):
            continue

        location = offer.get("location", "")
        department = offer.get("department", "")

        # Description is HTML
        description = _strip_html(offer.get("description", ""))

        job_url = _recruitee_job_url(token, offer)

        results.append({
            "title": title,
            "company": company,
            "location": location,
            "job_url": job_url,
            "description": description,
            "date_posted": normalize_date_posted(offer.get("created_at", "")),
            "site": "recruitee",
            "department": department,
        })

    logger.info(f"Recruitee ({company}): {len(results)} matching jobs from {len(offers)} total")
    return results


# ---------------------------------------------------------------------------
# Workable
# ---------------------------------------------------------------------------

_WORKABLE_API_URL = "https://apply.workable.com/api/v3/accounts/{token}/jobs"


def _workable_job_url(token: str, job: dict) -> str:
    shortcode = job.get("shortcode", "")
    job_url = job.get("url", "")
    if not job_url and shortcode:
        job_url = f"https://apply.workable.com/{token}/j/{shortcode}/"
    return job_url


def scrape_workable(token: str, company: str, keywords: list[str]) -> list[dict]:
    """
    Scrape jobs from a Workable public job board.

    Endpoint: POST https://apply.workable.com/api/v3/accounts/{token}/jobs
    Body: {"query": "", "location": "", "department": "", "worktype": "", "remote": ""}
    Response: {"results": [...], "nextToken": "..."} — paginated via token.

    Each result has: title, shortcode, department, shortDescription,
    location (city/region/country/remote), published (ISO date), url, etc.

    Full description requires a second API call:
    GET https://apply.workable.com/api/v1/widget/accounts/{token}/jobs/{shortcode}

    Args:
        token: The Workable account slug (e.g. "traveloka")
        company: Company display name
        keywords: List of keyword phrases to filter by title

    Returns:
        List of standardized job dicts
    """
    url = _WORKABLE_API_URL.format(token=token)
    all_jobs = []
    next_token = None
    max_pages = 20
    # Still-listed record (Phase 27c): complete only when the pages ran out
    # (no nextToken) — never after an error or the max_pages cap.
    fetch_error = False
    reached_end = False

    for _ in range(max_pages):
        # Workable API schema change (2026): filter fields must be ARRAYS —
        # string values now 400 ('"location" must be an array'). This silently
        # zeroed every Workable board until the payload was fixed.
        payload = {"query": "", "location": [], "department": [], "worktype": [], "remote": []}
        if next_token:
            payload["token"] = next_token

        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Workable API error for {company} ({token}): {e}")
            fetch_error = True
            break

        page_results = data.get("results", [])
        if not isinstance(page_results, list):
            fetch_error = True
            break

        all_jobs.extend(page_results)

        next_token = data.get("nextToken")
        if not next_token or not page_results:
            reached_end = not next_token
            break

    report_board_fetch(
        "workable", token,
        ok=not (fetch_error and not all_jobs),
        complete=reached_end and not fetch_error,
        total_count=len(all_jobs),
        links=[(_workable_job_url(token, j), j.get("title", "")) for j in all_jobs if isinstance(j, dict)],
    )

    if not all_jobs:
        logger.info(f"Workable ({company}): 0 jobs found")
        return []

    results = []

    for job in all_jobs:
        title = job.get("title", "")
        if not _matches_keywords(title, keywords):
            continue

        # Build location string
        loc = job.get("location", {})
        location_parts = []
        if isinstance(loc, dict):
            city = loc.get("city", "")
            region = loc.get("region", "")
            country = loc.get("country", "")
            if city:
                location_parts.append(city)
            if region:
                location_parts.append(region)
            if country and country.lower() not in ("us", "usa", "united states"):
                location_parts.append(country)
            if loc.get("remote", False):
                location_parts.append("Remote")
        elif isinstance(loc, str):
            location_parts.append(loc)
        location_name = ", ".join(location_parts) if location_parts else ""

        department = job.get("department", "")
        if isinstance(department, list):
            department = department[0] if department else ""

        job_url = _workable_job_url(token, job)

        # Use shortDescription from list endpoint (avoids N+1 detail calls)
        description = job.get("shortDescription", "")

        results.append({
            "title": title,
            "company": company,
            "location": location_name,
            "job_url": job_url,
            "description": description,
            "date_posted": normalize_date_posted(job.get("published", "")),
            "site": "workable",
            "department": department,
        })

    logger.info(f"Workable ({company}): {len(results)} matching jobs from {len(all_jobs)} total")
    return results


# ---------------------------------------------------------------------------
# Teamtailor
# ---------------------------------------------------------------------------

_TEAMTAILOR_JOBS_URL = "https://{token}.teamtailor.com/jobs"


def scrape_teamtailor(token: str, company: str, keywords: list[str]) -> list[dict]:
    """
    Scrape jobs from a Teamtailor public job board.

    Teamtailor doesn't have a documented public JSON API. We fetch the
    HTML jobs page and parse job listings from the structured HTML.

    The jobs page uses a consistent HTML structure with job cards
    containing title, location, and links.

    Args:
        token: The Teamtailor subdomain (e.g. "northvolt")
        company: Company display name
        keywords: List of keyword phrases to filter by title

    Returns:
        List of standardized job dicts
    """
    url = _TEAMTAILOR_JOBS_URL.format(token=token)
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        })
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Teamtailor fetch error for {company} ({token}): {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Teamtailor uses structured data (JSON-LD) for job postings
    results = []

    # Try JSON-LD first (most reliable)
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            import json
            ld_data = json.loads(script.string or "")
            # Could be a single JobPosting or a list
            postings = []
            if isinstance(ld_data, dict):
                if ld_data.get("@type") == "JobPosting":
                    postings = [ld_data]
                elif ld_data.get("@type") == "ItemList":
                    postings = ld_data.get("itemListElement", [])
            elif isinstance(ld_data, list):
                postings = ld_data

            for posting in postings:
                # ItemList wraps in {"item": {...}}
                if "item" in posting:
                    posting = posting["item"]
                if posting.get("@type") != "JobPosting":
                    continue

                title = posting.get("title", "")
                if not _matches_keywords(title, keywords):
                    continue

                # Location from JSON-LD
                job_location = posting.get("jobLocation", {})
                if isinstance(job_location, list):
                    job_location = job_location[0] if job_location else {}
                address = job_location.get("address", {}) if isinstance(job_location, dict) else {}
                location_parts = []
                if isinstance(address, dict):
                    city = address.get("addressLocality", "")
                    region = address.get("addressRegion", "")
                    if city:
                        location_parts.append(city)
                    if region:
                        location_parts.append(region)
                elif isinstance(address, str):
                    location_parts.append(address)

                if posting.get("jobLocationType") == "TELECOMMUTE":
                    location_parts.append("Remote")

                location_name = ", ".join(location_parts) if location_parts else ""

                job_url = posting.get("url", "")
                description = posting.get("description", "")
                if "<" in description:
                    description = _strip_html(description)

                date_posted = normalize_date_posted(posting.get("datePosted", ""))

                dept = ""
                if posting.get("department"):
                    dept = posting["department"]

                results.append({
                    "title": title,
                    "company": company,
                    "location": location_name,
                    "job_url": job_url,
                    "description": description,
                    "date_posted": date_posted,
                    "site": "teamtailor",
                    "department": dept,
                })
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug(f"Teamtailor JSON-LD parse skipped ({type(e).__name__}: {e}); falling back to HTML")
            continue

    # Fallback: parse HTML job cards if JSON-LD yielded nothing
    if not results:
        # Teamtailor uses various HTML structures; common patterns:
        # <a> tags with data attributes or structured classes containing job info
        for link in soup.find_all("a", href=re.compile(r"/jobs/\d+")):
            title_el = link.find(["h2", "h3", "span", "div"], class_=re.compile(r"title|name", re.I))
            if not title_el:
                # Sometimes the link text itself is the title
                title = link.get_text(strip=True)
            else:
                title = title_el.get_text(strip=True)

            if not title or not _matches_keywords(title, keywords):
                continue

            href = link.get("href", "")
            if href.startswith("/"):
                job_url = f"https://{token}.teamtailor.com{href}"
            else:
                job_url = href

            # Location is often in a sibling or child element
            loc_el = link.find(["span", "div"], class_=re.compile(r"location|city", re.I))
            location = loc_el.get_text(strip=True) if loc_el else ""

            results.append({
                "title": title,
                "company": company,
                "location": location,
                "job_url": job_url,
                "description": "",
                "date_posted": "",
                "site": "teamtailor",
                "department": "",
            })

    logger.info(f"Teamtailor ({company}): {len(results)} matching jobs")
    return results
