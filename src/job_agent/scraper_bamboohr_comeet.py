"""
BambooHR and Comeet ATS scrapers.

Follows the same pattern as ats_scraper.py (Greenhouse/Lever):
each function takes (token, company, keywords) and returns list[dict].

BambooHR API: GET https://{token}.bamboohr.com/careers/list
    Returns JSON: {meta: {totalCount}, result: [{id, jobOpeningName, departmentLabel, location, ...}]}
    No auth required for public job boards. Job detail only via HTML.
    Token = company subdomain (e.g. "examplecompany").

Comeet API: GET https://www.comeet.co/careers-api/2.0/company/{company_uid}/positions
    ?token={api_token}&details=true
    Returns JSON array of position objects.
    Token format = "company_uid:api_token" (e.g. "12.345:0123456789ABCDEF").
"""

import logging

import requests
from bs4 import BeautifulSoup

from .ats_scraper import _matches_keywords, _strip_html, report_board_fetch
from .utils import normalize_date_posted

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BambooHR
# ---------------------------------------------------------------------------

_BAMBOOHR_LIST_URL = "https://{subdomain}.bamboohr.com/careers/list"


def _bamboohr_job_url(token: str, job: dict) -> str:
    job_id = job.get("id", "")
    return f"https://{token}.bamboohr.com/careers/{job_id}" if job_id else ""


def scrape_bamboohr(token: str, company: str, keywords: list[str]) -> list[dict]:
    """
    Scrape jobs from a BambooHR public job board.

    Endpoint: GET https://{token}.bamboohr.com/careers/list
    Returns JSON with meta.totalCount and result[] array.

    Args:
        token: The BambooHR company subdomain (e.g. "examplecompany")
        company: Company display name
        keywords: List of keyword phrases to filter by title

    Returns:
        List of standardized job dicts
    """
    url = _BAMBOOHR_LIST_URL.format(subdomain=token)
    try:
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        logger.warning(f"BambooHR timeout for {company} ({token})")
        report_board_fetch("bamboohr", token, ok=False, complete=False, total_count=0, links=[])
        return []
    except requests.exceptions.HTTPError as e:
        logger.warning(f"BambooHR HTTP error for {company} ({token}): {e}")
        report_board_fetch("bamboohr", token, ok=False, complete=False, total_count=0, links=[])
        return []
    except Exception as e:
        logger.warning(f"BambooHR API error for {company} ({token}): {e}")
        report_board_fetch("bamboohr", token, ok=False, complete=False, total_count=0, links=[])
        return []

    jobs_data = data.get("result", [])
    if not isinstance(jobs_data, list):
        logger.warning(f"BambooHR ({company}): unexpected response — 'result' is not a list")
        report_board_fetch("bamboohr", token, ok=False, complete=False, total_count=0, links=[])
        return []

    # Still-listed record (Phase 27c): complete when the list length matches
    # the board's own meta.totalCount (when the payload carries one).
    _meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    _total = _meta.get("totalCount")
    report_board_fetch(
        "bamboohr", token, ok=True,
        complete=_total is None or _total == len(jobs_data),
        total_count=len(jobs_data),
        links=[(_bamboohr_job_url(token, j), j.get("jobOpeningName", "")) for j in jobs_data
               if isinstance(j, dict)],
    )

    results = []

    for job in jobs_data:
        title = job.get("jobOpeningName", "")
        if not _matches_keywords(title, keywords):
            continue

        # Build location string
        loc = job.get("location", {}) or {}
        location_parts = []
        city = loc.get("city", "")
        state = loc.get("state", "")
        if city:
            location_parts.append(city)
        if state:
            location_parts.append(state)
        if job.get("isRemote"):
            location_parts.append("Remote")
        location_name = ", ".join(location_parts) if location_parts else ""

        job_url = _bamboohr_job_url(token, job)

        results.append({
            "title": title,
            "company": company,
            "location": location_name,
            "job_url": job_url,
            "description": "",  # BambooHR list endpoint doesn't include descriptions
            "date_posted": "",  # Not provided in list response
            "site": "bamboohr",
            "department": job.get("departmentLabel", ""),
        })

    logger.info(f"BambooHR ({company}): {len(results)} matching jobs from {len(jobs_data)} total")
    return results


# ---------------------------------------------------------------------------
# Comeet (Spark Hire Recruit)
# ---------------------------------------------------------------------------

_COMEET_API_URL = "https://www.comeet.co/careers-api/2.0/company/{company_uid}/positions"


def scrape_comeet(token: str, company: str, keywords: list[str]) -> list[dict]:
    """
    Scrape jobs from a Comeet (Spark Hire Recruit) public job board.

    Endpoint: GET https://www.comeet.co/careers-api/2.0/company/{uid}/positions
    Returns JSON array of position objects.

    Args:
        token: Format "company_uid:api_token" (e.g. "12.345:0123456789ABCDEF")
        company: Company display name
        keywords: List of keyword phrases to filter by title

    Returns:
        List of standardized job dicts
    """
    # Parse compound token
    if ":" not in token:
        logger.warning(f"Comeet ({company}): invalid token format, expected 'uid:api_token'")
        report_board_fetch("comeet", token, ok=False, complete=False, total_count=0, links=[])
        return []

    company_uid, api_token = token.split(":", 1)

    url = _COMEET_API_URL.format(company_uid=company_uid)
    try:
        resp = requests.get(
            url,
            params={"token": api_token, "details": "true"},
            timeout=60,  # Comeet docs recommend 60s timeout
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        logger.warning(f"Comeet timeout for {company} ({company_uid})")
        report_board_fetch("comeet", token, ok=False, complete=False, total_count=0, links=[])
        return []
    except requests.exceptions.HTTPError as e:
        logger.warning(f"Comeet HTTP error for {company} ({company_uid}): {e}")
        report_board_fetch("comeet", token, ok=False, complete=False, total_count=0, links=[])
        return []
    except Exception as e:
        logger.warning(f"Comeet API error for {company} ({company_uid}): {e}")
        report_board_fetch("comeet", token, ok=False, complete=False, total_count=0, links=[])
        return []

    if not isinstance(data, list):
        logger.warning(f"Comeet ({company}): unexpected response — expected list, got {type(data).__name__}")
        report_board_fetch("comeet", token, ok=False, complete=False, total_count=0, links=[])
        return []

    # Still-listed record (Phase 27c): internal postings are not public.
    _public = [p for p in data if isinstance(p, dict) and not p.get("is_internal", False)]
    report_board_fetch(
        "comeet", token, ok=True, complete=True, total_count=len(_public),
        links=[(p.get("url_active_page", "") or p.get("url_comeet_hosted_page", ""), p.get("name", ""))
               for p in _public],
    )

    results = []

    for pos in data:
        title = pos.get("name", "")
        if not _matches_keywords(title, keywords):
            continue

        # Skip internal postings
        if pos.get("is_internal", False):
            continue

        # Build location string
        loc = pos.get("location", {}) or {}
        location_parts = []
        city = loc.get("city", "")
        state = loc.get("state", "")
        country = loc.get("country", "")
        if city:
            location_parts.append(city)
        if state:
            location_parts.append(state)
        if country and country.upper() not in ("US", "USA"):
            location_parts.append(country)
        if loc.get("is_remote"):
            location_parts.append("Remote")
        location_name = ", ".join(location_parts) if location_parts else loc.get("name", "")

        # Extract description from details array
        description = ""
        details = pos.get("details", [])
        if isinstance(details, list):
            parts = []
            for detail in details:
                value = detail.get("value", "")
                if value:
                    parts.append(_strip_html(value))
            description = "\n\n".join(parts)

        job_url = pos.get("url_active_page", "") or pos.get("url_comeet_hosted_page", "")

        results.append({
            "title": title,
            "company": company,
            "location": location_name,
            "job_url": job_url,
            "description": description,
            "date_posted": normalize_date_posted(pos.get("time_updated", "")),
            "site": "comeet",
            "department": pos.get("department", ""),
        })

    logger.info(f"Comeet ({company}): {len(results)} matching jobs from {len(data)} total")
    return results
