"""
Phenom career-site scraper (e.g. Disney's join.disneycareers.com).

Follows the same pattern as ats_scraper.py (Greenhouse/Lever):
scrape_phenom(token, company, keywords) returns list[dict].

API (the search page's own call):
    POST https://{host}/widgets
    Body: {"ddoKey": "refineSearch", "from": <offset>, "size": <n>,
           "keywords": "", "lang": "en_global", "country": "global", ...}
    Response: {"refineSearch": {"totalHits": <n>,
               "data": {"jobs": [{jobId, title, location, multi_location,
                                  postedDate (ISO), companyName, ...}]}}}

    An empty "keywords" lists the whole board, so the board is fetched in a
    couple of requests and titles are matched here, like Greenhouse/Lever.
    That also makes the fetch a complete board list for the "still listed?"
    check (report_board_fetch).

Posting page: https://{host}/{country}/{lang}/job/{jobId}/{Title-Slug}
    (the page's own canonical link). Its schema.org JobPosting
    (application/ld+json) carries the posting text. Its datePosted is the
    day the page was rendered, not the posting date, so the date comes from
    the list's postedDate.

Token format: the site's base URL with its country and language path, e.g.
    "https://join.disneycareers.com/global/en"
"""

import html
import json
import logging
import re
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .ats_scraper import _matches_keywords, report_board_fetch
from .utils import normalize_date_posted

logger = logging.getLogger(__name__)

# 500 is the most the widgets API returns per call. Ties in its default order
# shift between calls, so every page boundary can repeat a few postings and
# skip as many others: pages of 100 or 200 lost a few of a board's ~550
# postings, pages of 500 (one boundary) lost none. Fewer, larger pages keep
# the list complete.
_PAGE_SIZE = 500
_MAX_PAGES = 10
_REQUEST_TIMEOUT = 40  # a 500-posting page is about 1.5 MB
_INTER_REQUEST_DELAY = 1.5
_USER_AGENT = "JobAgentBot/1.0 (+job search assistant)"

# Posting pages fetched per run for their text, newest posting first. A board
# of several hundred postings matches a few dozen titles at most; the cap
# bounds a run if a keyword change ever matches far more.
DETAIL_FETCH_CAP = 40
DETAIL_TIMEOUT = 20
# Consecutive posting pages without text before the rest are skipped this run.
DETAIL_MAX_FAILURES = 3


def _is_phase_timeout(e: BaseException) -> bool:
    """main._ScrapeTimeout (the pipeline's SIGALRM phase budget), matched by
    name as in ats_scraper: it must end the phase, never be swallowed here."""
    return type(e).__name__ == "_ScrapeTimeout"


def _parse_token(token: str) -> tuple[str, str, str]:
    """"https://join.disneycareers.com/global/en" ->
    ("https://join.disneycareers.com", "global", "en")."""
    parsed = urlparse(token.rstrip("/"))
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid Phenom URL (no scheme/host): {token}")
    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) < 2:
        raise ValueError(f"Phenom URL needs /<country>/<lang>: {token}")
    return f"{parsed.scheme}://{parsed.hostname}", segments[0], segments[1]


def _search_body(country: str, lang: str, offset: int, size: int) -> dict:
    """The search page's own refineSearch request, with no keywords (the
    whole board)."""
    return {
        "lang": f"{lang}_{country}",
        "deviceType": "desktop",
        "country": country,
        "pageName": "search-results",
        "ddoKey": "refineSearch",
        "from": offset,
        "size": size,
        "jobs": True,
        "counts": True,
        "keywords": "",
        "global": True,
        "selected_fields": {},
        "locationData": {},
        "all_fields": ["category"],
        "clearAll": False,
        "jdsource": "facets",
        "isSliderEnable": False,
        "sortBy": "",
        "subsearch": "",
    }


def _fetch_board(session: requests.Session, host: str, country: str, lang: str):
    """Every posting on the board, deduplicated by jobId.

    Returns (jobs, total_hits, complete); raises on a failed or malformed
    first page. A later page that fails ends the fetch as incomplete.
    """
    url = f"{host}/widgets"
    by_id: dict[str, dict] = {}
    total = 0
    offset = 0
    complete = False
    for page in range(_MAX_PAGES):
        if page:
            time.sleep(_INTER_REQUEST_DELAY)
        try:
            resp = session.post(url, json=_search_body(country, lang, offset, _PAGE_SIZE),
                                timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            result = resp.json()["refineSearch"]
            jobs = result["data"]["jobs"]
            total = int(result["totalHits"])
        except Exception as e:
            if page == 0 or _is_phase_timeout(e):
                raise
            logger.warning(f"Phenom {host}: page at offset {offset} failed — board list incomplete")
            break
        for job in jobs:
            job_id = str(job.get("jobId") or "")
            if job_id:
                by_id.setdefault(job_id, job)
        offset += _PAGE_SIZE
        if not jobs or offset >= total:
            # Complete only when every posting the board counts came back.
            complete = total > 0 and len(by_id) >= total
            break
    return list(by_id.values()), total, complete


def _slug(title: str) -> str:
    """The site's own link slug: "Producer – Example Studio (As-Needed)" ->
    "Producer-Example-Studio-As-Needed"."""
    return re.sub(r"[^A-Za-z0-9]+", "-", title or "").strip("-")


def _title(job: dict) -> str:
    """The posting's title as text; the API can send null."""
    return str(job.get("title") or "")


def _job_link(host: str, country: str, lang: str, job: dict) -> str:
    job_id = str(job.get("jobId") or "")
    if not job_id:
        return ""
    slug = _slug(_title(job))
    base = f"{host}/{country}/{lang}/job/{job_id}"
    return f"{base}/{slug}" if slug else base


def _location(job: dict) -> str:
    """All of a posting's locations, "; "-joined like other multi-location
    postings, so a home-metro option on a New York-first posting still counts."""
    multi = [str(loc) for loc in (job.get("multi_location") or []) if loc]
    if len(multi) > 1:
        return "; ".join(multi)
    return job.get("location") or (multi[0] if multi else "")


def _parse_description(page_html: str) -> str:
    """Posting text from the page's JobPosting JSON-LD (plain text, one line
    per block). The HTML inside may arrive entity-encoded, so it is unescaped
    first."""
    soup = BeautifulSoup(page_html, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or script.get_text() or "")
        except ValueError:
            continue
        for node in data if isinstance(data, list) else [data]:
            if isinstance(node, dict) and node.get("@type") == "JobPosting" and node.get("description"):
                text = BeautifulSoup(html.unescape(node["description"]), "html.parser").get_text("\n", strip=True)
                if text:
                    return text
    return ""


def _add_descriptions(session: requests.Session, rows: list[dict], cap: int | None = None) -> int:
    """Fill row["description"] in place from each posting page, newest first.

    At most `cap` pages; DETAIL_MAX_FAILURES pages in a row without text
    stop the fetching (those rows keep the list data). Returns how many got
    text.
    """
    cap = DETAIL_FETCH_CAP if cap is None else cap
    picked = sorted(rows, key=lambda r: r.get("date_posted") or "", reverse=True)[:cap]
    filled = failures = 0
    for i, row in enumerate(picked):
        if i:
            time.sleep(_INTER_REQUEST_DELAY)
        text = ""
        try:
            resp = session.get(row["job_url"], headers={"Accept": "text/html"}, timeout=DETAIL_TIMEOUT)
            resp.raise_for_status()
            text = _parse_description(resp.text)
        except Exception as e:
            if _is_phase_timeout(e):
                raise
            logger.debug(f"Phenom posting page {row['job_url']}: {e}")
        if text:
            row["description"] = text
            filled += 1
            failures = 0
        else:
            failures += 1
            if failures >= DETAIL_MAX_FAILURES:
                logger.warning(f"Phenom: {failures} posting pages in a row gave no text — stopping description fetch")
                break
    return filled


def scrape_phenom(token: str, company: str, keywords: list[str], fetch_details: bool = True) -> list[dict]:
    """
    Scrape a Phenom career site: the whole board, titles matched here.

    Args:
        token: Site base URL with country/lang, e.g. "https://join.disneycareers.com/global/en"
        company: Company display name
        keywords: List of keyword phrases to filter by title
        fetch_details: Fetch posting pages for the description (capped)

    Returns:
        List of standardized job dicts
    """
    try:
        host, country, lang = _parse_token(token)
    except ValueError as e:
        logger.warning(f"Phenom URL parse error for {company}: {e}")
        return []

    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": _USER_AGENT})

    try:
        jobs, total, complete = _fetch_board(session, host, country, lang)
    except Exception as e:
        if _is_phase_timeout(e):
            raise
        logger.warning(f"Phenom API error for {company} ({token}): {e}")
        report_board_fetch("phenom", token, ok=False, complete=False, total_count=0, links=[])
        return []

    report_board_fetch(
        "phenom", token, ok=True, complete=complete, total_count=len(jobs),
        links=[(_job_link(host, country, lang, j), _title(j)) for j in jobs],
    )
    if not complete:
        logger.warning(f"Phenom ({company}): got {len(jobs)} of {total} postings — list incomplete")

    results = []
    for job in jobs:
        # A null title used to raise in _matches_keywords and end the scan.
        title = _title(job)
        if not _matches_keywords(title, keywords):
            continue
        results.append({
            "title": title,
            "company": company,
            "location": _location(job),
            "job_url": _job_link(host, country, lang, job),
            "description": "",  # list has only a teaser; _add_descriptions fills it
            "date_posted": normalize_date_posted(job.get("postedDate", "")),
            "site": "phenom",
            # The hiring entity ("Walt Disney Pictures", "Pixar") — the same
            # role hiringOrganization played on Disney's Workday rows.
            "department": job.get("companyName") or "",
        })

    filled = _add_descriptions(session, results) if fetch_details and results else 0
    logger.info(
        f"Phenom ({company}): {len(results)} matching jobs from {len(jobs)} total "
        f"(board says {total}; description for {filled})"
    )
    return results
