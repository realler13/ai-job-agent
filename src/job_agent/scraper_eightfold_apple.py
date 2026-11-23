"""
Scrapers for Eightfold.ai (Netflix) and Apple careers (SSR hydration).

Eightfold API: GET https://explore.jobs.netflix.net/api/apply/v2/jobs
    ?domain={token}&start={offset}&num={limit}&query={search_term}
    Returns JSON: {count, positions: [{name, location, locations, department, ...}]}

Apple Careers: GET https://jobs.apple.com/{locale}/search?search={query}&location=united-states-USA&page={page}
    Returns HTML with SSR hydration: JSON.parse("...") containing searchResults[].
    Per posting: GET https://jobs.apple.com/api/v1/jobDetails/{positionId} (JSON: level, pay).
"""

import json
import logging
import re
import time

import requests

from .ats_scraper import _matches_keywords, _strip_html, report_board_fetch
from .config import STALE_POSTING_DAYS
from .utils import is_posting_old

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eightfold.ai (Netflix)
# ---------------------------------------------------------------------------

_EIGHTFOLD_API = "https://explore.jobs.netflix.net/api/apply/v2/jobs"


_RETRY_429_WAITS = (5, 15, 30)   # seconds


def _get_eightfold_page(token: str, offset: int, page_size: int):
    """One page of the positions API. A 429 is retried after a short wait:
    one rate-limit reply used to end the whole fetch, and Netflix was skipped
    on a large share of runs. The waits total under a minute so the fetch
    stays inside the per-board time limit."""
    params = {"domain": token, "start": offset, "num": page_size, "query": ""}
    resp = requests.get(_EIGHTFOLD_API, params=params, timeout=15)
    for wait in _RETRY_429_WAITS:
        if resp.status_code != 429:
            break
        logger.info(f"Eightfold rate limit (429) — retrying in {wait}s")
        time.sleep(wait)
        resp = requests.get(_EIGHTFOLD_API, params=params, timeout=15)
    return resp


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
            resp = _get_eightfold_page(token, offset, page_size)
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
# The JSON the job page itself is built from: about a tenth of the page's size
# and half its time, and it carries the level and the pay too.
_APPLE_DETAIL_API = "https://jobs.apple.com/api/v1/jobDetails/{position_id}"
_APPLE_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
# A single 'producer' seed missed podcast/program/content roles entirely.
# Titles are still gated against the FULL keyword list below, so broad seeds
# add recall without noise.
_APPLE_SEEDS = ["producer", "production", "podcast", "content"]
_APPLE_MAX_PAGES = 5
# US postings only: the worldwide search filled the five pages with India /
# Saudi / China roles the location gate drops anyway, and only the US-filtered
# search names the city instead of just the country.
_APPLE_LOCATION = "united-states-USA"

# jobs.apple.com can answer slowly for days at a time: 15s read timeouts,
# first on the job pages, then on the search pages too, until no seed gets
# past page 1 and a run finds no Apple jobs at all, while the same requests
# take 0.4s (search) and 4-6s (job page) at a quieter hour. So: a longer read
# timeout, a retry with backoff after a timeout, a pause between requests,
# and two stops so a slow Apple can never eat the ATS phase's time budget.
_APPLE_TIMEOUT = (5, 25)              # (connect, read) seconds
_APPLE_RETRY_WAITS = (5, 15)          # seconds before the 2nd and 3rd try
_APPLE_PAUSE_SEC = 1.0                # between any two requests
_APPLE_MAX_FAILURES_IN_ROW = 2        # requests that failed after all retries
_APPLE_BUDGET_SEC = 300               # the whole Apple scrape
_APPLE_MAX_DETAILS = 25               # job pages read per run
# The search spends the shared budget first. When Apple was slow but still
# answering, paging used it all and every row came back without its level and
# pay, so the level rule and the comp floor could not act. Paging stops once
# this share of the budget is gone; the rest is for the job reads.
_APPLE_SEARCH_SHARE = 0.5


class _AppleFetcher:
    """Every jobs.apple.com request of one scrape goes through here: one pause
    between requests, retries after a timeout / 429 / 5xx, and a stop for the
    rest of the scrape once Apple stops answering or the time budget is spent."""

    def __init__(self):
        self.started = time.monotonic()
        self.requests = 0
        self.failures_in_row = 0
        self.stopped = ""
        self.answered_with = 0   # HTTP status of a refused (4xx) last request

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def _out_of_time(self) -> bool:
        if time.monotonic() - self.started > _APPLE_BUDGET_SEC:
            if not self.stopped:
                self.stopped = f"time budget of {_APPLE_BUDGET_SEC}s spent"
                logger.warning(f"Apple: {self.stopped} — stopping for this run")
            return True
        return False

    def get(self, url: str, params: dict | None = None):
        """The response, or None when Apple did not answer usefully."""
        self.answered_with = 0
        if self.stopped or self._out_of_time():
            return None
        last_error = None
        for wait in (0,) + _APPLE_RETRY_WAITS:
            if wait:
                time.sleep(wait)
                if self._out_of_time():
                    return None
            elif self.requests:
                time.sleep(_APPLE_PAUSE_SEC)
            self.requests += 1
            try:
                resp = requests.get(url, params=params, headers=_APPLE_HEADERS,
                                    timeout=_APPLE_TIMEOUT)
                resp.raise_for_status()
                self.failures_in_row = 0
                return resp
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e                       # slow or busy: try again
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code == 429 or code >= 500:
                    last_error = e                   # busy: try again
                    continue
                # Apple answered: a 404 & co is about this URL, not a sign
                # that Apple is slow, so it never counts toward the stop.
                logger.debug(f"Apple {code} for {url} {params or ''}")
                self.failures_in_row = 0
                self.answered_with = code
                return None
            except Exception as e:
                # main.py's phase budget (SIGALRM) arrives as _ScrapeTimeout
                # inside requests.get: it must end the ATS phase, not count
                # as one failed Apple request.
                if type(e).__name__ == "_ScrapeTimeout":
                    raise
                last_error = e
                break
        self.failures_in_row += 1
        logger.warning(f"Apple fetch failed ({url} {params or ''}): {last_error}")
        if self.failures_in_row >= _APPLE_MAX_FAILURES_IN_ROW:
            self.stopped = f"{self.failures_in_row} requests in a row failed"
            logger.warning(f"Apple: {self.stopped} — stopping for this run")
        return None


def _apple_location(loc: dict) -> str:
    # The US-filtered search often leaves 'city' blank and names the place in
    # 'name' ("Orlando").
    city = loc.get("city", "") or loc.get("name", "")
    parts = [p for p in (city, loc.get("stateProvince", ""), loc.get("countryName", "")) if p]
    return ", ".join(dict.fromkeys(parts))


def _apple_is_stale(job: dict) -> bool:
    """Older than the pre-filter's stale gate, which drops it whatever its
    level is. Unknown dates are not stale (the gate lets them through too)."""
    date = str(job.get("postDateInGMT", "") or "")[:10]
    return is_posting_old(date, days=STALE_POSTING_DAYS)


def _apple_post_date(job: dict) -> str:
    """The posting's ISO date-time, for newest-first ordering ('' sorts last)."""
    return str(job.get("postDateInGMT", "") or "")


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
    fetcher = _AppleFetcher()
    # One posting is listed once per location (id "200000001-0101",
    # "200000001-0102", ...); positionId is the posting. Keyed by it, the
    # locations merge into one row and its job page is read once.
    postings: dict[str, dict] = {}
    locations: dict[str, list[str]] = {}
    pages = 0
    search_cut = ""

    for seed in _APPLE_SEEDS:
        for page in range(1, _APPLE_MAX_PAGES + 1):
            if fetcher.elapsed() > _APPLE_BUDGET_SEC * _APPLE_SEARCH_SHARE:
                search_cut = (f"search stopped at {fetcher.elapsed():.0f}s to "
                              f"leave time for the job reads")
                logger.warning(f"Apple: {search_cut}")
                break
            resp = fetcher.get(_APPLE_SEARCH_URL.format(locale=token),
                               {"search": seed, "location": _APPLE_LOCATION, "page": page})
            if resp is None:
                break
            pages += 1
            # Only postings in the shape we know: one odd entry must not end
            # the scrape (and the ATS phase with it).
            search_results = [j for j in _extract_apple_hydration(resp.text)
                              if isinstance(j, dict)]
            if not search_results:
                break
            for job in search_results:
                position_id = str(job.get("positionId", "") or job.get("id", ""))
                if not position_id:
                    continue
                if position_id not in postings:
                    postings[position_id] = job
                    locations[position_id] = []
                for loc in job.get("locations", []) or []:
                    if not isinstance(loc, dict):
                        continue
                    text = _apple_location(loc)
                    if text and text not in locations[position_id]:
                        locations[position_id].append(text)
            # Newest first: once a whole page is past the stale gate, every
            # later page is too.
            if all(_apple_is_stale(j) for j in search_results):
                break
        if fetcher.stopped or search_cut:
            break

    matching = [(pid, job) for pid, job in postings.items()
                if _matches_keywords(job.get("postingTitle", ""), keywords)]

    # The job page adds what the search leaves out: the Pay & Benefits
    # section (without it, a posting's hourly pay range never reached the DB)
    # and Apple's internal level (see
    # format_apple_level). Not read for a posting the stale gate drops
    # anyway, and not past the cap. Newest first: the cap and the time
    # budget then cut the oldest postings, the ones least worth applying to.
    to_read = sorted((pid for pid, job in matching if not _apple_is_stale(job)),
                     key=lambda pid: _apple_post_date(postings[pid]), reverse=True)
    details: dict[str, dict] = {}
    for position_id in to_read[:_APPLE_MAX_DETAILS]:
        if fetcher.stopped:
            break
        details[position_id] = _fetch_apple_detail_fields(position_id, token, fetcher)
    details_read = len(details)

    results = []
    for position_id, job in matching:
        title = job.get("postingTitle", "")
        job_url = f"https://jobs.apple.com/{token}/details/{position_id}"
        detail = details.get(position_id, {})
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
            "location": "; ".join(locations.get(position_id, [])),
            "link": job_url,
            "job_url": job_url,
            "description": desc,
            "date_posted": job.get("postDateInGMT", "") or job.get("postingDate", ""),
            "site": "apple",
            "department": "",
            "job_level": level,
        })

    logger.info(
        f"Apple: {len(results)} matching jobs (scanned {len(postings)} postings on "
        f"{pages} search pages, read {details_read} job pages, "
        f"{fetcher.requests} requests in {time.monotonic() - fetcher.started:.0f}s"
        + (f"; {search_cut}" if search_cut else "")
        + (f"; stopped early: {fetcher.stopped}" if fetcher.stopped else "") + ")")
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
        return f"{lf} {lcode}–{hcode}"
    return f"{low} – {high}"


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


def _fetch_apple_detail_fields(position_id: str, locale: str = "en-us",
                               fetcher: "_AppleFetcher | None" = None) -> dict:
    """Read one posting's job data ONCE: {'pay': ..., 'level': ...}, or {}.

    The JSON first. The job page is read only when the JSON answered but not
    in the shape we know (Apple changed it): after a timeout the heavier page
    would only add load to a server that is already slow."""
    fetcher = fetcher or _AppleFetcher()
    resp = fetcher.get(_APPLE_DETAIL_API.format(position_id=position_id))
    refused = fetcher.answered_with
    res = None
    if resp is not None:
        try:
            res = resp.json().get("res")
        except Exception as e:
            if type(e).__name__ == "_ScrapeTimeout":
                raise
            res = None
    if isinstance(res, dict) and (res.get("postingTitle") or res.get("jobSummary")):
        # The level first: str-checked titles, so it cannot raise. Only the
        # top level is checked above; a nested pay value of another type (a
        # null footer, localizations as a list, selectedLocation as a string)
        # raised AttributeError, which left scrape_apple and ended the whole
        # ATS phase. Now that posting just has no pay; it keeps its level,
        # and no page is read, since the JSON answered and the server may be
        # slow.
        low, high = res.get("lowJobTitle"), res.get("highJobTitle")
        level = format_apple_level(low if isinstance(low, str) else "",
                                   high if isinstance(high, str) else "")
        try:
            pay = _apple_pay_from_json(res)
        except Exception as e:
            if type(e).__name__ == "_ScrapeTimeout":
                raise
            logger.debug(f"Apple pay for {position_id} in an unexpected shape: {e!r}")
            pay = ""
        return {"pay": pay, "level": level}
    if resp is None and not refused:
        return {}
    page = fetcher.get(f"https://jobs.apple.com/{locale}/details/{position_id}")
    if page is None:
        return {}
    html = page.text
    return {"pay": _apple_pay_section(html),
            "level": format_apple_level(*extract_apple_levels(html))}


def _apple_pay_from_json(res: dict) -> str:
    """The Pay & Benefits footer of the posting's job data, as plain text.
    Pay differs by location; the location the job page opens on comes first,
    as on the page."""
    import html as _html
    footers = res.get("postingFooters") or []
    selected = (res.get("selectedLocation") or {}).get("id")
    footers = sorted(footers, key=lambda f: f.get("postLocationId") != selected)
    for footer in footers:
        for items in (footer.get("localizations") or {}).values():
            for item in items or []:
                content = item.get("content") or ""
                if "$" not in content:
                    continue  # some markets omit pay
                text = re.sub(r"<[^>]+>", " ", content)
                text = re.sub(r"\s+", " ", _html.unescape(text)).strip()
                return f"{item.get('name') or 'Pay & Benefits'}: {text}"
    return ""


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
    # The argument is a JS string literal: read it to its real closing quote,
    # skipping escaped characters. The lazy '(.+?)")' stopped at the first
    # '")' in a posting text ('the \"Studio\")') and the page parsed to 0 rows.
    match = re.search(r'JSON\.parse\("((?:[^"\\]|\\.)*)"\)', html or "", re.S)
    if not match:
        return []

    try:
        # The argument is a JS string literal; reading it as JSON keeps
        # non-ASCII intact. unicode_escape read the UTF-8 bytes as Latin-1:
        # an en dash in a title ("Producer – Video") was stored as "Producer â".
        try:
            raw = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            raw = match.group(1).encode().decode("unicode_escape")
        data = json.loads(raw)
        loader_data = data.get("loaderData", {})
        # The search route contains searchResults
        for route_data in loader_data.values():
            if isinstance(route_data, dict) and "searchResults" in route_data:
                found = route_data["searchResults"]
                return found if isinstance(found, list) else []
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
        logger.debug(f"Apple hydration parse error: {e}")

    return []
