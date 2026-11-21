"""
Scrapers for Eightfold.ai (Netflix) and Apple careers (SSR hydration).

Eightfold API: GET https://explore.jobs.netflix.net/api/apply/v2/jobs
    ?domain={token}&start={offset}&num={limit}&query={search_term}
    Returns JSON: {count, positions: [{name, location, locations, department, ...}]}

Apple Careers: GET https://jobs.apple.com/{locale}/search?search={query}&page={page}
    Returns HTML with SSR hydration: JSON.parse("...") containing searchResults[].
"""

import json
import logging
import re

import requests

from .ats_scraper import _matches_keywords, _strip_html, report_board_fetch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eightfold.ai (Netflix)
# ---------------------------------------------------------------------------

_EIGHTFOLD_API = "https://explore.jobs.netflix.net/api/apply/v2/jobs"


def scrape_eightfold(token: str, company: str, keywords: list[str]) -> list[dict]:
    """
    Scrape jobs from an Eightfold.ai public job board.

    Args:
        token: The Eightfold domain (e.g. "netflix.com")
        company: Company display name
        keywords: List of keyword phrases to filter by title

    Returns:
        List of standardized job dicts
    """
    results = []
    # The API returns at most 10 positions per page whatever `num` asks for
    # (probed 2026-10-04: num=100 -> 10 of count=471), so the offset steps by
    # what came back, not by page_size — stepping by 100 read 50 of 471 jobs.
    page_size = 100
    offset = 0
    max_pages = 80  # Safety limit: 800 positions at 10 per page
    # Still-listed record (Phase 27c): complete only when paging reached the
    # board's own count with no error and every counted position was read
    # (never after the max_pages cap or a short page). Counted by DISTINCT
    # id: the API's ordering shifts between pages, so a sweep can return one
    # job twice and skip another — that sweep is not complete.
    fetch_error = False
    reached_end = False
    board_links: list = []
    seen_positions = 0
    seen_ids: set = set()

    for _ in range(max_pages):
        try:
            resp = requests.get(
                _EIGHTFOLD_API,
                params={"domain": token, "start": offset, "num": page_size, "query": ""},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Eightfold API error for {company}: {e}")
            fetch_error = True
            break

        positions = data.get("positions", [])
        if not positions:
            break

        seen_positions += len(positions)
        for pos in positions:
            _pid = pos.get("id", "") if isinstance(pos, dict) else ""
            if _pid:
                seen_ids.add(str(_pid))
            board_links.append((f"https://explore.jobs.netflix.net/careers/job/{_pid}" if _pid else "",
                                pos.get("name", "") if isinstance(pos, dict) else ""))

        for pos in positions:
            title = pos.get("name", "")
            if not _matches_keywords(title, keywords):
                continue

            location = pos.get("location", "")
            job_id = pos.get("id", "")
            job_url = f"https://explore.jobs.netflix.net/careers/job/{job_id}" if job_id else ""

            results.append({
                "title": title,
                "company": company,
                "location": location,
                "link": job_url,
                "job_url": job_url,
                "description": "",  # Available via detail endpoint, skip for speed
                "date_posted": pos.get("t_update", "") or pos.get("t_create", ""),
                "site": "eightfold",
                "department": pos.get("department", ""),
            })

        total = data.get("count", 0)
        offset += len(positions)
        if not isinstance(total, int) or offset >= total:
            reached_end = isinstance(total, int) and len(seen_ids) >= total
            break

    report_board_fetch(
        "eightfold", token,
        ok=not (fetch_error and not seen_positions),
        complete=reached_end and not fetch_error,
        total_count=len(seen_ids),
        links=board_links,
    )

    logger.info(f"Eightfold ({company}): {len(results)} matching jobs")
    return results


# ---------------------------------------------------------------------------
# Apple Careers (SSR hydration)
# ---------------------------------------------------------------------------

_APPLE_SEARCH_URL = "https://jobs.apple.com/{locale}/search"


def scrape_apple(token: str, company: str, keywords: list[str]) -> list[dict]:
    """
    Scrape jobs from Apple's careers page via SSR hydration data.

    Args:
        token: The locale path (e.g. "en-us")
        company: Company display name (usually "Apple")
        keywords: List of keyword phrases to filter by title

    Returns:
        List of standardized job dicts
    """
    results = []
    seen_ids: set[str] = set()
    max_pages = 5
    # A single 'producer' seed missed podcast/program/content roles entirely.
    # Titles are still gated against the FULL keyword list below, so broad
    # seeds add recall without noise.
    _APPLE_SEEDS = ["producer", "production", "podcast", "content"]

    for seed in _APPLE_SEEDS:
      for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                _APPLE_SEARCH_URL.format(locale=token),
                params={"search": seed, "page": page},
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Apple careers fetch error page {page}: {e}")
            break

        # Extract hydration JSON from HTML
        search_results = _extract_apple_hydration(resp.text)
        if not search_results:
            break

        for job in search_results:
            title = job.get("postingTitle", "")
            job_id = str(job.get("id", ""))

            if job_id in seen_ids:
                continue
            seen_ids.add(job_id)

            if not _matches_keywords(title, keywords):
                continue

            # Parse locations
            locations = job.get("locations", [])
            location_parts = []
            for loc in locations:
                city = loc.get("city", "")
                state = loc.get("stateProvince", "")
                country = loc.get("countryName", "")
                parts = [p for p in [city, state, country] if p]
                if parts:
                    location_parts.append(", ".join(parts))
            location_str = "; ".join(location_parts) if location_parts else ""

            position_id = job.get("positionId", "")
            job_url = f"https://jobs.apple.com/{token}/details/{position_id}" if position_id else ""

            # Append the detail page's Pay & Benefits section — the search
            # hydration's jobSummary LOSES it (2026-07-26: a posting's
            # hourly pay range never reached the DB).
            # Matched Apple jobs are few, so one detail fetch each is cheap.
            # The same fetch carries Apple's internal level (see
            # format_apple_level), which only the detail page has.
            detail = _fetch_apple_detail_fields(job_url) if job_url else {}
            pay_section = detail.get("pay", "")
            level = detail.get("level", "")
            desc = job.get("jobSummary", "")
            if pay_section:
                desc = (desc + "\n\n" + pay_section).strip()
            if level:
                # First line, so a truncated description never loses it and
                # the cascade's seniority judgment sees it.
                desc = f"{apple_level_line(level)}\n\n{desc}".strip()
            results.append({
                "title": title,
                "company": company,
                "location": location_str,
                "link": job_url,
                "job_url": job_url,
                "description": desc,
                "date_posted": job.get("postDateInGMT", "") or job.get("postingDate", ""),
                "site": "apple",
                "department": "",
                "job_level": level,
            })

    logger.info(f"Apple: {len(results)} matching jobs")
    return results


# Apple's internal level for a posting, from the DETAIL page's hydration
# (the search results never carry it). A posting may give a range
# (lowJobTitle "Machine Learning ICT4", highJobTitle "Machine Learning ICT5"),
# one end only ("Engineering Project/Program ICT3"), or nothing (common for
# producer roles). ICT = technical ladder, IC = non-technical.
_APPLE_LEVEL_FIELD = re.compile(r'\\?"(low|high)JobTitle\\?"\s*:\s*\\?"([^"\\]+)')


def extract_apple_levels(html: str) -> tuple[str, str]:
    """(low, high) level titles from an Apple detail page; '' when absent."""
    found = {}
    for which, value in _APPLE_LEVEL_FIELD.findall(html or ""):
        found.setdefault(which, value.strip())
    return found.get("low", ""), found.get("high", "")


def format_apple_level(low: str, high: str) -> str:
    """One display string: 'Machine Learning ICT4–ICT5' for a range within one
    job family, 'A – B' across families, the single title otherwise."""
    low, high = (low or "").strip(), (high or "").strip()
    if not low or not high or low == high:
        return low or high
    lf, _, lcode = low.rpartition(" ")
    hf, _, hcode = high.rpartition(" ")
    if lf and lf == hf:
        return f"{lf} {lcode}\u2013{hcode}"
    return f"{low} \u2013 {high}"


def apple_level_line(level: str) -> str:
    """The description line a judge reads. Apple's IC ladder is internal
    jargon, so the line carries a short key to it — and the configured target
    band (cascade_filter.APPLE_TARGET_BAND, the single source), so every judge
    reads the same rule the cascade applies in code
    (cascade_filter.apple_level_verdict). A profile with no band turns that
    rule off; the line then carries the level and the key, and no band."""
    # lazy: avoids an import cycle
    from .cascade_filter import APPLE_TARGET_BAND, APPLE_TARGET_HIGH, APPLE_TARGET_LOW
    key = "Apple IC ladder: 2 entry, 3 mid, 4 senior, 5 staff, 6 principal"
    if None in (APPLE_TARGET_BAND, APPLE_TARGET_LOW, APPLE_TARGET_HIGH):
        return f"Apple job level: {level} ({key}; MGR is the people-manager ladder)"
    band = APPLE_TARGET_BAND
    below = f"IC{APPLE_TARGET_LOW - 1} and below and " if APPLE_TARGET_LOW > 1 else ""
    return (f"Apple job level: {level} ({key}; the candidate's target band is "
            f"{band} / {band.replace('IC', 'ICT')} — IC{APPLE_TARGET_HIGH + 1} is above it, "
            f"{below}the MGR people-manager ladder are outside it)")


def _fetch_apple_detail_fields(url: str, timeout: int = 15) -> dict:
    """Fetch an Apple job detail page ONCE and return {'pay': ..., 'level': ...}."""
    try:
        resp = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logger.debug(f"Apple detail fetch failed for {url}: {e}")
        return {}
    import time as _time
    _time.sleep(0.5)  # be polite between detail fetches
    return {"pay": _apple_pay_section(html),
            "level": format_apple_level(*extract_apple_levels(html))}


def _fetch_apple_job_detail(url: str, timeout: int = 15) -> str:
    """Pay & Benefits text only (kept for existing callers)."""
    return _fetch_apple_detail_fields(url, timeout).get("pay", "")


def _apple_pay_section(html: str) -> str:
    """The Pay & Benefits section text of an Apple detail page (the search
    hydration's jobSummary loses it — the 2026-07-26 case). The
    detail SSR interleaves i18n blobs with content, so whole-page slicing is
    unreliable; a targeted window around the pay sentence is robust."""
    import html as _html
    try:
        raw = html.encode("utf-8", "ignore").decode("unicode_escape", "ignore")
        text = re.sub(r"<[^>]+>", " ", raw)
        text = _html.unescape(text)
        text = re.sub(r"\s+", " ", text)
        # Anchor on the pay SENTENCE, not the section header — "Pay &
        # Benefits" also appears in the page's i18n string table (that decoy
        # burned three debugging rounds on 2026-07-26).
        pay = text.find("base pay range")
        if pay == -1:
            pay = text.rfind("Pay & Benefits")  # real section comes AFTER the i18n table
        if pay == -1:
            return ""
        window = text[max(0, pay - 200):pay + 2300]
        # Only return it if it actually contains money — some markets omit pay
        return window if "$" in window else ""
    except Exception as e:
        logger.debug(f"Apple pay-section parse failed: {e}")
    return ""


def _extract_apple_hydration(html: str) -> list[dict]:
    """Extract searchResults from Apple's SSR hydration data in HTML."""
    match = re.search(r'JSON\.parse\("(.+?)"\)', html)
    if not match:
        return []

    try:
        raw = match.group(1).encode().decode("unicode_escape")
        data = json.loads(raw)
        loader_data = data.get("loaderData", {})
        # The search route contains searchResults
        for route_data in loader_data.values():
            if isinstance(route_data, dict) and "searchResults" in route_data:
                return route_data["searchResults"]
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
        logger.debug(f"Apple hydration parse error: {e}")

    return []
