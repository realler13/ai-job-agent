"""
Built In scraper — tech/media jobs from Built In's regional sites.

Built In runs one site per metro (builtinseattle.com, builtinnyc.com, ...).
Which ones to scrape is the candidate's market, so it comes from the profile
(``search.builtin_sites``, see candidate_profile.builtin_sites); a profile
without the key scrapes no Built In site at all.

Built In's regional subdomains serve server-rendered HTML. No auth wall,
no Cloudflare. Keyword search via `?search=X`, pagination via `?page=N`.

Card structure (verified April 2026 via live probe):
  div[data-id="job-card"] id="job-card-{id}"
    a[data-id="company-title"]                    → company name
    a[data-id="job-card-title"]                   → job title + URL path
    span ... "9 Days Ago" / "Reposted An Hour Ago" → posting age (own span)

Built In jobs are indexed by numeric ID embedded in the URL path
(/job/{slug}/{id}), which we use for deduplication.

The list view carries no posting text. Each posting page does, in a
schema.org JobPosting (application/ld+json) and in div#job-post-body-{id},
so after the search the newest matched postings get their page fetched for
the description (capped per run, see DETAIL_FETCH_CAP). The same JobPosting
carries datePosted, a plain date that matches the card's "N Days Ago" (for a
reposted card it is the repost date). It replaces the card's relative age on
the postings whose page is fetched; the rest keep the card's age.
"""
from __future__ import annotations

import html
import json
import logging
import re
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from .ats_scraper import _matches_keywords
from .candidate_profile import builtin_sites
from .config import PRIMARY_METRO_TERMS, SECONDARY_METRO_TERMS, _parse_metro_term
from .utils import normalize_date_posted

logger = logging.getLogger(__name__)



def regions_from_sites(sites) -> dict[str, str]:
    """{short key: base URL} for the given Built In base URLs.

    The key is the metro part of the host ("https://www.builtinseattle.com"
    -> "seattle"; the national "https://builtin.com" -> "builtin"); a key
    already taken falls back to the full host.
    """
    regions: dict[str, str] = {}
    for base in sites:
        host = base.split("://", 1)[-1].split("/", 1)[0]
        label = host[4:] if host.startswith("www.") else host
        key = label.split(".", 1)[0][len("builtin"):] or "builtin"
        if key in regions:
            key = host
        regions[key] = base
    return regions


def location_hints(primary_terms, secondary_terms) -> tuple[str, ...]:
    """The place names a card's text is scanned for, in priority order.

    The profile's primary then secondary metro terms, title-cased (a term
    written with its state, "vancouver, wa", contributes the town:
    "Vancouver"), then "Remote" and "Hybrid". Each name appears once.
    """
    hints: list[str] = []
    for term in list(primary_terms or ()) + list(secondary_terms or ()):
        if not str(term or "").strip():
            continue
        town, _state = _parse_metro_term(term)
        hint = town.title()
        if hint and hint not in hints:
            hints.append(hint)
    for tag in ("Remote", "Hybrid"):
        if tag not in hints:
            hints.append(tag)
    return tuple(hints)


#: {short key: base URL} — the profile's search.builtin_sites (empty = off).
REGIONS = regions_from_sites(builtin_sites())
#: Place names stashed as a card's location (first match wins).
LOCATION_HINTS = location_hints(PRIMARY_METRO_TERMS, SECONDARY_METRO_TERMS)
REQUEST_TIMEOUT = 20
INTER_REQUEST_DELAY = 1.0

# Posting pages fetched per run for their text. Most matched postings were
# saved by earlier runs (save_job keeps the first copy). Built In ids grow with
# time, so the newest ids are fetched first and this cap covers the new ones
# without hitting every page on every run.
DETAIL_FETCH_CAP = 40
DETAIL_TIMEOUT = 15
# Consecutive failed posting pages before the rest are skipped this run.
DETAIL_MAX_FAILURES = 3

# Broad seed queries for Built In's fuzzy server search. Querying all ~148
# keywords (2 regions x 148 x 3 pages ≈ 888 requests) triggered HTTP 429; these
# few seeds catch the same roles and each result is title-gated against the full
# keyword list downstream, so precision is preserved at a fraction of the load.
BUILTIN_SEARCH_SEEDS = [
    "producer", "content", "video", "production",
    "podcast", "creative", "studio", "editorial",
]

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})


# The card's age line. Cards dropped the "Posted" prefix ("9 Days Ago", "An
# Hour Ago", "One Month Ago", "Yesterday"); a reposted card still says
# "Reposted 7 Days Ago". A pattern that requires the prefix leaves most Built
# In rows with no date.
_AGE_RE = re.compile(
    r"(?:(?:re)?posted\s+)?"
    r"(?:(\d+|an?|one)\+?\s+(minute|hour|day|week|month|year)s?\s+ago|(today|yesterday))",
    re.I,
)


def _parse_posting_age(text: str) -> str:
    """Convert "9 Days Ago" / "Reposted 3 Hours Ago" / "Yesterday" →
    "YYYY-MM-DD". Returns empty string if unparseable."""
    m = _AGE_RE.search(text or "")
    if not m:
        return ""
    now = datetime.now()
    if m.group(3):
        days = 0 if m.group(3).lower() == "today" else 1
        return (now - timedelta(days=days)).strftime("%Y-%m-%d")
    n = int(m.group(1)) if m.group(1).isdigit() else 1  # "A", "An", "One"
    days = {"minute": 0, "hour": 0, "day": n, "week": 7 * n,
            "month": 30 * n, "year": 365 * n}[m.group(2).lower()]
    return (now - timedelta(days=days)).strftime("%Y-%m-%d")


def _card_age(card) -> str:
    """The card's age as YYYY-MM-DD, read from the text node that is only the
    age line. The card also holds a description snippet, and a word like
    "today" in it must not date the posting."""
    for node in card.find_all(string=True):
        if _AGE_RE.fullmatch(node.strip()):
            return _parse_posting_age(node.strip())
    return ""


def _parse_card(card, base_url: str) -> dict | None:
    """Extract job fields from a single Built In job card."""
    title_link = card.select_one('a[data-id="job-card-title"]')
    company_link = card.select_one('a[data-id="company-title"]')
    if not title_link or not company_link:
        return None

    title = title_link.get_text(strip=True)
    href = (title_link.get("href") or "").strip()
    company = company_link.get_text(strip=True)
    if not title or not href or not company:
        return None
    job_url = href if href.startswith("http") else f"{base_url}{href}"

    # Full card text — pull location hints
    card_text = card.get_text(" ", strip=True)
    date_posted = _card_age(card)

    # Location: Built In cards include city/state + work-arrangement tags. Our
    # pipeline already has a location pre-filter, so we stash whatever short
    # chunk mentions the city.
    location = ""
    # Simple heuristic: the first of the profile's metro place names (then
    # Remote / Hybrid) that the card text mentions — see LOCATION_HINTS.
    for hint in LOCATION_HINTS:
        if hint in card_text:
            location = hint
            break

    return {
        "title": title,
        "company": company,
        "location": location,
        "job_url": job_url,
        "description": "",  # list view has none; _add_descriptions fills it
        "date_posted": date_posted,  # _add_descriptions swaps in the page's date
        "site": "builtin",
        "department": "",
    }


def _get_with_backoff(url: str, max_retries: int = 3, timeout: float = REQUEST_TIMEOUT):
    """GET with exponential backoff on HTTP 429 (Built In rate-limits hard).

    Returns the Response, or None if 429 persisted through all retries.
    Raises on non-429 HTTP errors (caller handles).
    """
    for attempt in range(max_retries):
        r = _session.get(url, timeout=timeout)
        if r.status_code == 429:
            wait = INTER_REQUEST_DELAY * (2 ** attempt) + 1.0
            logger.info(f"Built In 429 — backing off {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    return None


def _job_id(job_url: str) -> int:
    """Built In's numeric posting id from /job/{slug}/{id} (0 if absent)."""
    m = re.search(r"/(\d+)/?(?:[?#].*)?$", job_url or "")
    return int(m.group(1)) if m else 0


def _html_to_text(fragment: str) -> str:
    """Posting HTML → plain text, one line per block, like the page reads."""
    return BeautifulSoup(fragment, "html.parser").get_text("\n", strip=True)


def _parse_description(page_html: str) -> str:
    """Pull the posting text out of a Built In posting page."""
    return _parse_posting_page(page_html)[0]


def _parse_posting_page(page_html: str) -> tuple[str, str]:
    """(posting text, date posted as YYYY-MM-DD) from a Built In posting page;
    either is "" when the page lacks it.

    The JobPosting JSON-LD is the stable source (it is what search engines
    read); the rendered body div is the fallback for the text if that block
    is missing.
    """
    soup = BeautifulSoup(page_html, "html.parser")
    date_posted = ""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or script.get_text() or "")
        except ValueError:
            continue
        # Built In wraps the JobPosting in an @graph; plain top-level also counts.
        nodes = []
        for n in data if isinstance(data, list) else [data]:
            if isinstance(n, dict):
                nodes.append(n)
                nodes.extend(n.get("@graph") or [])
        for node in nodes:
            if not (isinstance(node, dict) and node.get("@type") == "JobPosting"):
                continue
            date_posted = date_posted or normalize_date_posted(node.get("datePosted"))
            if node.get("description"):
                # The JSON string holds HTML; unescape first in case the
                # markup itself arrives entity-encoded.
                text = _html_to_text(html.unescape(node["description"]))
                if text:
                    return text, date_posted
    body = soup.select_one('div[id^="job-post-body-"]')
    return (body.get_text("\n", strip=True) if body else ""), date_posted


def _add_descriptions(jobs: list[dict], cap: int = DETAIL_FETCH_CAP) -> int:
    """Fill job["description"] in place from each posting page, newest first,
    and job["date_posted"] from the page's datePosted when it has one (a
    plain date beats the card's "N Days Ago").

    At most `cap` pages; each gets DETAIL_TIMEOUT seconds. A persistent 429
    or DETAIL_MAX_FAILURES failed pages in a row stop the fetching — those
    jobs keep the list data, as before. Returns how many got text.
    """
    picked = sorted(jobs, key=lambda j: _job_id(j["job_url"]), reverse=True)[:cap]
    by_id: dict[int, tuple[str, str]] = {}  # the same posting can come from several Built In sites
    filled = failures = 0
    for job in picked:
        jid = _job_id(job["job_url"])
        if jid and jid in by_id:
            text, date_posted = by_id[jid]
        else:
            text = date_posted = ""
            try:
                r = _get_with_backoff(job["job_url"], timeout=DETAIL_TIMEOUT)
                if r is None:
                    logger.warning("Built In: posting pages still 429 after retries — stopping description fetch")
                    break
                text, date_posted = _parse_posting_page(r.text)
            except Exception as e:
                logger.debug(f"Built In posting page {job['job_url']}: {e}")
            if jid:
                by_id[jid] = (text, date_posted)
            failures = 0 if text else failures + 1
            if failures >= DETAIL_MAX_FAILURES:
                logger.warning(
                    f"Built In: {failures} posting pages in a row gave no text — stopping description fetch"
                )
                break
            time.sleep(INTER_REQUEST_DELAY)
        if date_posted:
            job["date_posted"] = date_posted
        if text:
            job["description"] = text
            filled += 1
    logger.info(
        f"Built In: description for {filled} of the {len(picked)} newest postings "
        f"({len(jobs) - len(picked)} older ones keep list data)"
    )
    return filled


def _scrape_region(region: str, seed: str, match_keywords: list[str], max_pages: int = 3) -> list[dict]:
    """Scrape one Built In region (a REGIONS key) for one broad SEED query.

    The server search is fuzzy, so we query a few broad seeds and gate every
    result title against the FULL keyword list — same coverage, far fewer
    requests (which is what fixed the 429s).
    """
    base = REGIONS.get(region)
    if not base:
        logger.warning(f"Built In: unknown region {region!r}")
        return []

    results: list[dict] = []
    seen_urls: set[str] = set()

    for page in range(1, max_pages + 1):
        url = f"{base}/jobs?search={requests.utils.quote(seed)}&page={page}"
        try:
            r = _get_with_backoff(url)
            if r is None:
                logger.warning(f"Built In {region}/{seed!r}: still 429 after retries (page {page})")
                break
        except Exception as e:
            logger.warning(f"Built In {region} fetch error (page {page}): {e}")
            break

        soup = BeautifulSoup(r.text, "html.parser")
        cards = soup.select('div[data-id="job-card"]')
        if not cards:
            break

        new_on_page = 0
        for card in cards:
            job = _parse_card(card, base)
            if not job:
                continue
            if job["job_url"] in seen_urls:
                continue
            seen_urls.add(job["job_url"])
            new_on_page += 1  # new card PARSED — pagination continues even on a no-match page
            # Gate fuzzy server results against the FULL keyword list
            if not _matches_keywords(job["title"], match_keywords):
                continue
            results.append(job)

        if new_on_page == 0 and page > 1:
            break  # past the last relevant page
        time.sleep(INTER_REQUEST_DELAY)

    return results


def scrape_builtin(keywords: list[str], max_pages: int = 3) -> list[dict]:
    """Scrape every Built In region in REGIONS for all given keywords.

    Args:
        keywords: Terms to query Built In's search with.
        max_pages: Pages per region per keyword (default 3 → up to ~75 jobs).

    Returns:
        Deduplicated list of job dicts; [] (and no request) when the profile
        names no Built In site.
    """
    if not REGIONS:
        logger.info("Built In: skipped (no search.builtin_sites in the profile)")
        return []
    all_jobs: list[dict] = []
    seen_urls: set[str] = set()

    for region in REGIONS:
        for seed in BUILTIN_SEARCH_SEEDS:
            for job in _scrape_region(region, seed, keywords, max_pages=max_pages):
                if job["job_url"] in seen_urls:
                    continue
                seen_urls.add(job["job_url"])
                all_jobs.append(job)

    if all_jobs:
        _add_descriptions(all_jobs)

    logger.info(
        f"Built In: {len(all_jobs)} unique jobs "
        f"({len(BUILTIN_SEARCH_SEEDS)} seeds x {len(REGIONS)} regions)"
    )
    return all_jobs
