"""
Workday ATS scraper for public career sites.

Uses the Workday CXS (Customer Experience Services) API, which is the
public-facing JSON API that powers Workday career pages. This is a
reverse-engineered, undocumented API that has remained stable across
Workday instances since at least 2023.

API pattern (listings):
    POST https://{host}/wday/cxs/{company_slug}/{board}/jobs
    Body: {"limit": 20, "offset": 0, "searchText": "producer"}
    Response: {"jobPostings": [...], "total": 100}

    Each posting has: title, externalPath, locationsText, postedOn,
    timeType, bulletFields (usually contains the requisition ID).

API pattern (job detail):
    GET https://{host}/wday/cxs/{company_slug}/{board}{externalPath}
    Response: {"jobPostingInfo": {"jobDescription": "<html>...", ...},
               "hiringOrganization": {"name": "..."}}

    jobPostingInfo fields include: title, jobDescription (HTML),
    location, postedOn, startDate (YYYY-MM-DD), externalUrl,
    timeType, remoteType, country, etc.

Token format: full base URL, e.g.
    "https://disney.wd5.myworkdayjobs.com/disneycareer"

Verified live against Disney (wd5), Warner Bros (wd5), Sony (wd1)
on 2026-02-07.
"""

import logging
import re
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse

import pandas as pd
import requests

from .ats_scraper import _strip_html, _matches_keywords
from .utils import normalize_date_posted

logger = logging.getLogger(__name__)

# Broad search seeds for Workday's fuzzy server search — results are gated
# against the full keyword list, so coverage holds at ~1/18th the requests.
WORKDAY_SEARCH_SEEDS = [
    "producer", "production", "content", "video",
    "podcast", "creative", "editorial", "program manager",
]

# Workday API defaults
_DEFAULT_LIMIT = 20
_MAX_PAGES = 25  # Safety cap: 25 pages * 20 = 500 jobs max per keyword
_REQUEST_TIMEOUT = 20  # seconds
_INTER_REQUEST_DELAY = 0.5  # seconds between paginated requests
_DETAIL_FETCH_DELAY = 0.3  # seconds between detail fetches


def _parse_workday_relative_date(text: str) -> str:
    """
    Parse Workday's relative date strings into YYYY-MM-DD format.

    Workday's postedOn field returns human-readable relative dates like:
      "Posted Today", "Posted Yesterday", "Posted 4 Days Ago",
      "Posted 30+ Days Ago"

    If the text is already a date or can't be parsed, falls back to
    normalize_date_posted() which handles ISO dates, timestamps, etc.

    Args:
        text: The postedOn string from the Workday API

    Returns:
        Date string in YYYY-MM-DD format, or "" if unparseable
    """
    if not text:
        return ""

    text_lower = text.lower().strip()
    today = datetime.now()

    if "today" in text_lower:
        return today.strftime("%Y-%m-%d")

    if "yesterday" in text_lower:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # Match "Posted 4 Days Ago", "Posted 30+ Days Ago"
    match = re.search(r"(\d+)\+?\s*days?\s*ago", text_lower)
    if match:
        days = int(match.group(1))
        return (today - timedelta(days=days)).strftime("%Y-%m-%d")

    # Match "Posted 2 Weeks Ago"
    match = re.search(r"(\d+)\+?\s*weeks?\s*ago", text_lower)
    if match:
        weeks = int(match.group(1))
        return (today - timedelta(weeks=weeks)).strftime("%Y-%m-%d")

    # Match "Posted 1 Month Ago", "Posted 2 Months Ago"
    match = re.search(r"(\d+)\+?\s*months?\s*ago", text_lower)
    if match:
        months = int(match.group(1))
        return (today - timedelta(days=months * 30)).strftime("%Y-%m-%d")

    # Fall back to generic date normalization (handles ISO dates, timestamps)
    return normalize_date_posted(text)


def _parse_base_url(base_url: str) -> tuple[str, str, str]:
    """
    Parse a Workday career site base URL into its API components.

    Args:
        base_url: Full URL like "https://disney.wd5.myworkdayjobs.com/disneycareer"

    Returns:
        Tuple of (host_url, company_slug, board_token)
        e.g. ("https://disney.wd5.myworkdayjobs.com", "disney", "disneycareer")

    Raises:
        ValueError: If the URL doesn't match the expected Workday pattern
    """
    parsed = urlparse(base_url.rstrip("/"))

    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid Workday URL (no scheme/host): {base_url}")

    hostname = parsed.hostname  # e.g. "disney.wd5.myworkdayjobs.com"
    host_url = f"{parsed.scheme}://{hostname}"

    # Extract company slug from hostname: first segment before ".wd"
    # e.g. "disney.wd5.myworkdayjobs.com" -> "disney"
    parts = hostname.split(".")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse company slug from hostname: {hostname}")
    company_slug = parts[0]

    # Extract board token from path: last non-empty segment
    # e.g. "/disneycareer" -> "disneycareer"
    # e.g. "/en-US/disneycareer" -> "disneycareer" (handle locale prefixes)
    path_segments = [s for s in parsed.path.split("/") if s]
    if not path_segments:
        raise ValueError(f"Cannot parse board token from URL path: {base_url}")
    board_token = path_segments[-1]

    return host_url, company_slug, board_token


def _build_jobs_url(host_url: str, company_slug: str, board_token: str) -> str:
    """Build the Workday CXS jobs listing endpoint URL."""
    return f"{host_url}/wday/cxs/{company_slug}/{board_token}/jobs"


def _build_detail_url(
    host_url: str, company_slug: str, board_token: str, external_path: str
) -> str:
    """Build the Workday CXS job detail endpoint URL."""
    return f"{host_url}/wday/cxs/{company_slug}/{board_token}{external_path}"


def _fetch_job_detail(
    host_url: str,
    company_slug: str,
    board_token: str,
    external_path: str,
    session: requests.Session,
) -> dict:
    """
    Fetch full job details from the Workday CXS detail endpoint.

    The detail endpoint uses GET (not POST) and returns a response with
    jobPostingInfo (description, location, dates, etc.) and
    hiringOrganization (company division name).

    Returns:
        Dict with keys "jobPostingInfo" and "hiringOrganization",
        or empty dict on failure.
    """
    url = _build_detail_url(host_url, company_slug, board_token, external_path)
    try:
        resp = session.get(
            url,
            headers={"Accept": "application/json"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "jobPostingInfo": data.get("jobPostingInfo", {}),
            "hiringOrganization": data.get("hiringOrganization", {}),
        }
    except Exception as e:
        logger.debug(f"Workday detail fetch failed for {external_path}: {e}")
        return {}


def _base_url_path(base_url: str) -> str:
    """Extract the path portion of the base URL for building job links.

    e.g. "https://disney.wd5.myworkdayjobs.com/disneycareer" -> "/disneycareer"
    """
    parsed = urlparse(base_url.rstrip("/"))
    return parsed.path if parsed.path else ""


def scrape_workday(
    base_url: str,
    company: str,
    keywords: list[str],
    fetch_details: bool = True,
) -> list[dict]:
    """
    Scrape jobs from a Workday public career site.

    For each keyword, searches the Workday CXS API, paginates through
    results, filters by keyword match in the title, and optionally
    fetches individual job details for the full description.

    Args:
        base_url: Full Workday career site URL
                  (e.g. "https://disney.wd5.myworkdayjobs.com/disneycareer")
        company: Company display name (e.g. "Disney")
        keywords: List of keyword phrases to filter by title
        fetch_details: Whether to fetch individual job details for descriptions.
                       Set False to skip detail fetches (faster but no descriptions).

    Returns:
        List of standardized job dicts with keys:
        title, company, location, job_url, description, date_posted, site, department
    """
    try:
        host_url, company_slug, board_token = _parse_base_url(base_url)
    except ValueError as e:
        logger.warning(f"Workday URL parse error for {company}: {e}")
        return []

    jobs_url = _build_jobs_url(host_url, company_slug, board_token)
    url_path = _base_url_path(base_url)

    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json",
    })

    # Track seen external paths to avoid duplicate jobs across keyword searches
    seen_paths: set[str] = set()
    results: list[dict] = []
    consecutive_errors = 0  # Track consecutive HTTP errors for early bailout
    _MAX_CONSECUTIVE_ERRORS = 3  # Bail after 3 consecutive 404s (dead token)

    # Broad SEEDS drive the server search; every result title is still gated
    # against the FULL keyword list below. Searching all ~148 keywords per
    # company made the 23-company Workday phase a ~1-hour serial chain
    # (23 x 148 POSTs) for the same coverage these seeds provide.
    search_terms = WORKDAY_SEARCH_SEEDS

    logger.info(f"Workday ({company}): starting scan ({len(search_terms)} seed searches)...")

    for kw_idx, keyword in enumerate(search_terms):
        if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
            logger.warning(
                f"Workday ({company}): {consecutive_errors} consecutive errors "
                f"— token appears dead, skipping remaining keywords"
            )
            break

        offset = 0
        pages_fetched = 0
        keyword_had_error = False

        while pages_fetched < _MAX_PAGES:
            payload = {
                "limit": _DEFAULT_LIMIT,
                "offset": offset,
                "searchText": keyword,
            }

            try:
                resp = session.post(jobs_url, json=payload, timeout=_REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                consecutive_errors = 0  # Reset on success
            except requests.exceptions.HTTPError as e:
                if not keyword_had_error:
                    logger.warning(
                        f"Workday API HTTP error for {company} "
                        f"(keyword='{keyword}', offset={offset}): {e}"
                    )
                consecutive_errors += 1
                keyword_had_error = True
                break
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Workday API request error for {company} "
                    f"(keyword='{keyword}'): {e}"
                )
                consecutive_errors += 1
                break
            except ValueError as e:
                logger.warning(
                    f"Workday API JSON decode error for {company} "
                    f"(keyword='{keyword}'): {e}"
                )
                consecutive_errors += 1
                break

            postings = data.get("jobPostings", [])
            total = data.get("total", 0)

            if not postings:
                break

            for posting in postings:
                title = posting.get("title", "")
                external_path = posting.get("externalPath", "")

                # Deduplicate across keyword searches
                if external_path and external_path in seen_paths:
                    continue
                if external_path:
                    seen_paths.add(external_path)

                # Keyword filter on title
                if not _matches_keywords(title, keywords):
                    continue

                # Extract location and date from listing response
                location = posting.get("locationsText", "")
                posted_on = posting.get("postedOn", "")

                # Build the public-facing job URL from the base URL + externalPath
                job_url = (
                    f"{host_url}{url_path}{external_path}"
                    if external_path
                    else ""
                )

                # Fetch full description from the detail endpoint if enabled
                description = ""
                department = ""
                detail_date = ""

                if fetch_details and external_path:
                    time.sleep(_DETAIL_FETCH_DELAY)
                    detail_data = _fetch_job_detail(
                        host_url, company_slug, board_token, external_path, session
                    )
                    info = detail_data.get("jobPostingInfo", {})
                    org = detail_data.get("hiringOrganization", {})

                    # Description is HTML — strip it
                    raw_desc = info.get("jobDescription", "")
                    description = _strip_html(raw_desc)

                    # Use the detail's externalUrl if available (canonical URL)
                    detail_url = info.get("externalUrl", "")
                    if detail_url:
                        job_url = detail_url

                    # Department: use hiringOrganization name (e.g. "5070 Twentieth Century Fox")
                    department = org.get("name", "")

                    # startDate from detail is YYYY-MM-DD — use as fallback date
                    detail_date = info.get("startDate", "")

                # Parse the relative date from listing ("Posted Yesterday", etc.)
                date_posted = _parse_workday_relative_date(posted_on)

                # If relative date parsing failed, use startDate from detail
                if not date_posted and detail_date:
                    date_posted = normalize_date_posted(detail_date)

                results.append({
                    "title": title,
                    "company": company,
                    "location": location,
                    "job_url": job_url,
                    "description": description,
                    "date_posted": date_posted,
                    "site": "workday",
                    "department": department,
                })

            # Check if we've fetched all available postings
            offset += _DEFAULT_LIMIT
            pages_fetched += 1

            if offset >= total:
                break

            # Be polite between paginated requests
            time.sleep(_INTER_REQUEST_DELAY)

        # Progress logging every 25 keywords
        if (kw_idx + 1) % 25 == 0:
            logger.info(
                f"Workday ({company}): {kw_idx + 1}/{len(keywords)} keywords, "
                f"{len(results)} matches so far ({len(seen_paths)} unique postings)"
            )

    logger.info(
        f"Workday ({company}): {len(results)} matching jobs "
        f"(scanned {len(seen_paths)} unique postings)"
    )
    return results


def scrape_workday_boards(
    companies: list[dict],
    keywords: list[str],
    progress_callback=None,
) -> pd.DataFrame:
    """
    Scrape all configured Workday career boards.

    Iterates over a list of Workday company entries and scrapes each
    career board, returning a combined DataFrame of all matching jobs.

    Args:
        companies: List of dicts with keys: company, ats, token
                   where token is the full Workday career site URL
        keywords: Keywords to filter job titles
        progress_callback: Optional callback(message, progress_float)

    Returns:
        DataFrame of all matching jobs
    """
    all_jobs: list[dict] = []
    total = len(companies)

    for i, entry in enumerate(companies):
        company_name = entry["company"]
        base_url = entry["token"]  # For Workday, token is the full URL

        if progress_callback:
            progress_callback(
                f"Workday: Scanning {company_name}...",
                i / max(total, 1),
            )

        jobs = scrape_workday(base_url, company_name, keywords)
        all_jobs.extend(jobs)

    if progress_callback:
        progress_callback(
            f"Workday: Found {len(all_jobs)} jobs from {total} companies",
            1.0,
        )

    if not all_jobs:
        return pd.DataFrame()

    return pd.DataFrame(all_jobs)
