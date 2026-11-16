"""
Ashby and SmartRecruiters ATS scrapers.

Follows the same pattern as ats_scraper.py (Greenhouse/Lever):
each function takes (token, company, keywords) and returns list[dict].

Ashby API docs: https://developers.ashbyhq.com/docs/public-job-posting-api
SmartRecruiters API docs: https://developers.smartrecruiters.com/docs/posting-api

# ============================================================
# VERIFIED ASHBY COMPANIES (tokens for api.ashbyhq.com)
# Verified via jobs.ashbyhq.com/{token} — all active as of 2026-02-07
# ============================================================
# {"company": "OpenAI",    "ats": "ashby", "token": "openai"},
# {"company": "Notion",    "ats": "ashby", "token": "notion"},
# {"company": "Ramp",      "ats": "ashby", "token": "ramp"},
# {"company": "Deel",      "ats": "ashby", "token": "deel"},
# {"company": "Cursor",    "ats": "ashby", "token": "cursor"},
# {"company": "Cohere",    "ats": "ashby", "token": "cohere"},
# {"company": "Quora",     "ats": "ashby", "token": "quora"},
# {"company": "Buffer",    "ats": "ashby", "token": "buffer"},
#
# ============================================================
# VERIFIED SMARTRECRUITERS COMPANIES (slugs for api.smartrecruiters.com)
# Verified via careers.smartrecruiters.com/{slug} — all active as of 2026-02-07
# ============================================================
# {"company": "Bosch",                "ats": "smartrecruiters", "token": "BoschGroup"},
# {"company": "Visa",                 "ats": "smartrecruiters", "token": "Visa"},
# {"company": "Skechers",             "ats": "smartrecruiters", "token": "Skechers1"},
# {"company": "Equinox",              "ats": "smartrecruiters", "token": "Equinox"},
# {"company": "McDonald's",           "ats": "smartrecruiters", "token": "McDonaldsCorporation"},
# {"company": "IKEA (Inter Group)",   "ats": "smartrecruiters", "token": "InterIKEAGroup"},
# {"company": "LinkedIn",             "ats": "smartrecruiters", "token": "LinkedIn3"},
# {"company": "SmartRecruiters",      "ats": "smartrecruiters", "token": "SmartRecruiters"},
"""

import logging

import requests
from bs4 import BeautifulSoup

from .ats_scraper import _matches_keywords, _strip_html, report_board_fetch
from .utils import normalize_date_posted

logger = logging.getLogger(__name__)

# Ashby's API sometimes responds with brotli encoding which causes decoding
# errors in the requests library unless the brotli Python package is installed.
# Explicitly request gzip/deflate only to avoid this.
_SAFE_HEADERS = {"Accept-Encoding": "gzip, deflate"}


# ---------------------------------------------------------------------------
# Ashby
# ---------------------------------------------------------------------------

def scrape_ashby(token: str, company: str, keywords: list[str]) -> list[dict]:
    """
    Scrape jobs from an Ashby public job board.

    Endpoint: GET https://api.ashbyhq.com/posting-api/job-board/{token}
    Returns JSON with ``jobs`` array. Each job has title, location, department,
    team, descriptionPlain/descriptionHtml, publishedAt, jobUrl, etc.

    Args:
        token: The Ashby board token (e.g. "ramp", "openai")
        company: Company display name
        keywords: List of keyword phrases to filter by title

    Returns:
        List of standardized job dicts
    """
    url = f"https://api.ashbyhq.com/posting-api/job-board/{token}"
    try:
        resp = requests.get(url, timeout=15, headers=_SAFE_HEADERS)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        logger.warning(f"Ashby timeout for {company} ({token})")
        report_board_fetch("ashby", token, ok=False, complete=False, total_count=0, links=[])
        return []
    except requests.exceptions.HTTPError as e:
        logger.warning(f"Ashby HTTP error for {company} ({token}): {e}")
        report_board_fetch("ashby", token, ok=False, complete=False, total_count=0, links=[])
        return []
    except Exception as e:
        logger.warning(f"Ashby API error for {company} ({token}): {e}")
        report_board_fetch("ashby", token, ok=False, complete=False, total_count=0, links=[])
        return []

    jobs_data = data.get("jobs", [])
    if not isinstance(jobs_data, list):
        logger.warning(f"Ashby ({company}): unexpected response format — 'jobs' is not a list")
        report_board_fetch("ashby", token, ok=False, complete=False, total_count=0, links=[])
        return []

    # Still-listed record (Phase 27c): listed postings only — an unlisted
    # Ashby job is not public, so it is not "on the board".
    _listed = [j for j in jobs_data if isinstance(j, dict) and j.get("isListed", True)]
    report_board_fetch("ashby", token, ok=True, complete=True, total_count=len(_listed),
                       links=[(j.get("jobUrl", ""), j.get("title", "")) for j in _listed])

    results = []

    for job in jobs_data:
        title = job.get("title", "")
        if not _matches_keywords(title, keywords):
            continue

        # Skip unlisted postings
        if not job.get("isListed", True):
            continue

        # Build location string.
        # Primary location is a plain string; secondaryLocations is a list of
        # objects each with a "location" field.
        location_parts = []
        primary_loc = job.get("location", "")
        if primary_loc:
            location_parts.append(primary_loc)

        # Ashby flags remote jobs explicitly
        if job.get("isRemote"):
            if not any("remote" in p.lower() for p in location_parts):
                location_parts.append("Remote")

        location_name = ", ".join(location_parts) if location_parts else ""

        # Description: prefer plain text, fall back to stripping HTML
        description = job.get("descriptionPlain", "")
        if not description:
            description = _strip_html(job.get("descriptionHtml", ""))

        department = job.get("department", "")
        team = job.get("team", "")
        dept_str = department
        if team and team != department:
            dept_str = f"{department} — {team}" if department else team

        results.append({
            "title": title,
            "company": company,
            "location": location_name,
            "job_url": job.get("jobUrl", ""),
            "description": description,
            "date_posted": normalize_date_posted(job.get("publishedAt", "")),
            "site": "ashby",
            "department": dept_str,
        })

    logger.info(f"Ashby ({company}): {len(results)} matching jobs from {len(jobs_data)} total")
    return results


# ---------------------------------------------------------------------------
# SmartRecruiters
# ---------------------------------------------------------------------------

_SR_LIST_URL = "https://api.smartrecruiters.com/v1/companies/{slug}/postings"
_SR_DETAIL_URL = "https://api.smartrecruiters.com/v1/companies/{slug}/postings/{posting_id}"
_SR_PAGE_LIMIT = 100  # max items per page (API default is 10)


def _sr_fetch_posting_description(slug: str, posting_id: str) -> str:
    """
    Fetch the full description for a single SmartRecruiters posting.

    The list endpoint does NOT include job descriptions. We must hit the
    individual posting detail endpoint which returns jobAd.sections with
    jobDescription, qualifications, additionalInformation, etc.

    Returns concatenated plain text of all sections, or "" on error.
    """
    url = _SR_DETAIL_URL.format(slug=slug, posting_id=posting_id)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        detail = resp.json()
    except Exception as e:
        # Silent '' hid systematic failures (auth/schema change) that stripped
        # descriptions from every SR job — at least leave a trace.
        logger.debug(f"SR detail fetch failed for {slug}/{posting_id}: {e}")
        return ""

    # jobAd → sections → {companyDescription, jobDescription, qualifications, additionalInformation}
    job_ad = detail.get("jobAd", {})
    sections = job_ad.get("sections", {})

    parts = []
    for section_key in ("jobDescription", "qualifications", "additionalInformation", "companyDescription"):
        section = sections.get(section_key, {})
        text = section.get("text", "")
        if text:
            # Section text may contain HTML — strip it
            parts.append(_strip_html(text))

    return "\n\n".join(parts)


def _sr_job_url(token: str, posting: dict) -> str:
    """The human-readable posting URL. The 'ref' field is the API URL; the
    public page follows https://jobs.smartrecruiters.com/{slug}/{posting_id}."""
    posting_id = posting.get("id", "") or posting.get("uuid", "")
    ref_url = posting.get("ref", "")
    return f"https://jobs.smartrecruiters.com/{token}/{posting_id}" if posting_id else ref_url


def scrape_smartrecruiters(token: str, company: str, keywords: list[str]) -> list[dict]:
    """
    Scrape jobs from a SmartRecruiters public job board.

    Endpoint: GET https://api.smartrecruiters.com/v1/companies/{slug}/postings
    Returns paginated JSON with ``content`` array. Each posting has name,
    releasedDate, location, department, ref, company, etc.

    Individual posting descriptions require a second API call per posting.

    Args:
        token: The SmartRecruiters company slug (e.g. "BoschGroup", "Visa")
        company: Company display name
        keywords: List of keyword phrases to filter by title

    Returns:
        List of standardized job dicts
    """
    all_postings = []
    offset = 0
    # Still-listed record (Phase 27c): the board was read completely only when
    # pagination reached the board's own totalFound with no error. SR answers
    # 200 with totalFound 0 for dead slugs — that reads as an empty board,
    # which is never evidence that a posting closed.
    fetch_error = False
    reached_end = False
    total_found = 0

    # Paginate through all postings
    while True:
        url = _SR_LIST_URL.format(slug=token)
        params = {"limit": _SR_PAGE_LIMIT, "offset": offset}
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            logger.warning(f"SmartRecruiters timeout for {company} ({token}) at offset={offset}")
            fetch_error = True
            break
        except requests.exceptions.HTTPError as e:
            logger.warning(f"SmartRecruiters HTTP error for {company} ({token}): {e}")
            fetch_error = True
            break
        except Exception as e:
            logger.warning(f"SmartRecruiters API error for {company} ({token}): {e}")
            fetch_error = True
            break

        content = data.get("content", [])
        if not isinstance(content, list):
            logger.warning(f"SmartRecruiters ({company}): unexpected response — 'content' is not a list")
            fetch_error = True
            break

        all_postings.extend(content)

        # Check if we have more pages
        total_found = data.get("totalFound", 0)
        offset += len(content)
        if not content or offset >= total_found:
            reached_end = offset >= total_found
            break

    _complete = (not fetch_error and reached_end
                 and isinstance(total_found, int) and total_found == len(all_postings))
    report_board_fetch(
        "smartrecruiters", token,
        # A failure on the FIRST page read nothing at all; a later failure
        # still proved the postings it did read are listed.
        ok=not (fetch_error and not all_postings),
        complete=_complete,
        total_count=len(all_postings),
        links=[(_sr_job_url(token, p), p.get("name", "")) for p in all_postings if isinstance(p, dict)],
    )

    if not all_postings:
        logger.info(f"SmartRecruiters ({company}): 0 postings found")
        return []

    # Filter by keywords first, then fetch descriptions only for matches
    results = []

    for posting in all_postings:
        title = posting.get("name", "")
        if not _matches_keywords(title, keywords):
            continue

        # Build location string from nested location object
        loc = posting.get("location", {})
        location_parts = []
        city = loc.get("city", "")
        region = loc.get("region", "")
        country = loc.get("country", "")
        if city:
            location_parts.append(city)
        if region:
            location_parts.append(region)
        if country and country.lower() not in ("us", "usa", "united states"):
            # Only append country if it's not US (to avoid redundancy)
            location_parts.append(country)
        if loc.get("remote", False):
            location_parts.append("Remote")

        location_name = ", ".join(location_parts) if location_parts else ""

        # Department
        dept_obj = posting.get("department", {})
        department = ""
        if isinstance(dept_obj, dict):
            department = dept_obj.get("label", "") or dept_obj.get("name", "")
        elif isinstance(dept_obj, str):
            department = dept_obj

        # Build the posting URL.
        # The 'ref' field is the API URL; the human-readable URL follows the
        # pattern: https://jobs.smartrecruiters.com/{slug}/{posting_id}
        posting_id = posting.get("id", "") or posting.get("uuid", "")
        job_url = _sr_job_url(token, posting)

        # Fetch full description (individual API call per matched posting)
        description = ""
        if posting_id:
            description = _sr_fetch_posting_description(token, posting_id)

        results.append({
            "title": title,
            "company": company,
            "location": location_name,
            "job_url": job_url,
            "description": description,
            "date_posted": normalize_date_posted(posting.get("releasedDate", "")),
            "site": "smartrecruiters",
            "department": department,
        })

    logger.info(
        f"SmartRecruiters ({company}): {len(results)} matching jobs "
        f"from {len(all_postings)} total postings"
    )
    return results
