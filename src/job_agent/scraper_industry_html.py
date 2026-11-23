
import logging
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .ats_scraper import _matches_keywords
from .scraper_productionhub import _fetch_stealth, scrape_productionhub
from .scraper_hitmarker import scrape_hitmarker
from .scraper_builtin import scrape_builtin

logger = logging.getLogger(__name__)

# Shared HTTP session for connection reuse
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})

REQUEST_TIMEOUT = 15
INTER_REQUEST_DELAY = 0.5  # seconds between requests to same domain


def _fetch_page(url: str, stealth: bool = False, solve_cloudflare: bool = True,
                wait_selector: str | None = None) -> BeautifulSoup | None:
    """Fetch a URL and return parsed BeautifulSoup, or None on error.

    stealth=True goes through the stealth browser instead: MBW and DoorsOpen
    answer a plain request with 403, so trying requests
    first would only add a wasted round trip. solve_cloudflare and
    wait_selector are passed to scraper_productionhub._fetch_stealth.
    """
    if stealth:
        return _fetch_stealth(url, solve_cloudflare=solve_cloudflare,
                              wait_selector=wait_selector)
    try:
        resp = _session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        logger.warning(f"HTTP error fetching {url}: {e}")
        return None


# =============================================================================
# 1. ShowbizJobs
# =============================================================================

def scrape_showbizjobs(
    keywords: list[str],
    categories: list[str] | None = None,
    max_pages: int = 3,
) -> list[dict]:
    """
    Scrape ShowbizJobs.com job listings.

    Args:
        keywords: Keywords to filter job titles
        categories: ShowbizJobs category slugs. If None, uses config.
        max_pages: Pages per category to scrape (1-indexed)

    Returns:
        List of standardized job dicts
    """
    if categories is None:
        from .config import SHOWBIZJOBS_CATEGORIES
        categories = SHOWBIZJOBS_CATEGORIES

    results = []
    seen_urls = set()

    for cat in categories:
        for page in range(1, max_pages + 1):
            url = f"https://www.showbizjobs.com/jobs/category/{cat}?p={page}"
            soup = _fetch_page(url)
            if soup is None:
                break

            # Current markup (after a redesign): each posting is an
            # <a class="jl-list-item"> with div.jl-list-title / -company and
            # jl-list-col-location / -date columns. The old table#postings
            # markup is kept as a fallback; on its own it finds nothing.
            rows = soup.select("a.jl-list-item")
            if not rows:
                table = soup.find("table", id="postings")
                rows = table.find_all("tr") if table else soup.select(
                    "div.job-listing, tr.job-row, div.listing")

            if not rows:
                break

            found_any = False
            for row in rows:
                try:
                    if row.name == "a":
                        title_el = row.select_one("div.jl-list-title")
                        title = title_el.get_text(strip=True) if title_el else ""
                        href = row.get("href", "")
                    else:
                        # Legacy table row: title from h5 > a
                        title_el = row.find("h5")
                        title_link = title_el.find("a") if title_el else None
                        if not title_link:
                            continue
                        title = title_link.get_text(strip=True)
                        href = title_link.get("href", "")
                    if href and not href.startswith("http"):
                        href = f"https://www.showbizjobs.com{href}"

                    if not title or href in seen_urls:
                        continue

                    # A new card was PARSED — pagination must continue even if
                    # this page happens to contain no keyword matches (the old
                    # post-gate flag stopped whole categories at page 1).
                    found_any = True

                    if not _matches_keywords(title, keywords):
                        continue

                    company = ""
                    location = ""
                    date_posted = ""
                    if row.name == "a":
                        company_el = row.select_one("div.jl-list-company")
                        company = company_el.get_text(strip=True) if company_el else ""
                        loc_el = row.select_one("div.jl-list-col-location")
                        location = loc_el.get_text(" ", strip=True) if loc_el else ""
                        date_el = row.select_one("div.jl-list-col-date")
                        if date_el:
                            # "Today", "2d ago", "1w ago" or "Oct 16"
                            date_posted = parse_posted_date(date_el.get_text(strip=True))
                    else:
                        # Legacy: company from the first h6, location from a td
                        h6_tags = row.find_all("h6")
                        if h6_tags:
                            company_link = h6_tags[0].find("a")
                            company = (company_link or h6_tags[0]).get_text(strip=True)
                        for td in row.find_all("td"):
                            text = td.get_text(strip=True)
                            if any(s in text for s in [", ", "Remote", "CA", "NY", "TX"]):
                                if text != title and text != company:
                                    location = text
                                    break

                    seen_urls.add(href)

                    results.append({
                        "title": title,
                        "company": company,
                        "location": location,
                        "job_url": href,
                        "description": "",
                        "date_posted": date_posted,
                        "site": "showbizjobs",
                        "department": cat,
                    })

                except Exception as e:
                    logger.debug(f"ShowbizJobs row parse error: {e}")
                    continue

            time.sleep(INTER_REQUEST_DELAY)

            # If no jobs found on this page, stop paginating
            if not found_any:
                break

    logger.info(f"ShowbizJobs: {len(results)} matching jobs")
    return results


# =============================================================================
# 2. Podnews Jobs
# =============================================================================

def scrape_podnews(keywords: list[str]) -> list[dict]:
    """
    Scrape Podnews.net/jobs for podcast industry jobs.

    Single page, no pagination needed.
    """
    url = "https://podnews.net/jobs"
    soup = _fetch_page(url)
    if soup is None:
        return []

    results = []
    seen_urls: set[str] = set()

    # Each posting is a text link to /job/<company>/<slug>, followed by
    # ", Company", a <br> and a flag + location line. The old div.artblock
    # wrappers are gone: the page has no per-job container
    # any more, so the job link itself is the anchor point. The logo link to
    # the same URL carries no text and is skipped.
    for link_el in soup.find_all("a", href=lambda h: h and "/job/" in h):
        try:
            title = link_el.get_text(strip=True)
            href = link_el.get("href", "")
            if href and not href.startswith("http"):
                href = f"https://podnews.net{href}"

            if not title or href in seen_urls:
                continue
            seen_urls.add(href)

            if not _matches_keywords(title, keywords):
                continue

            company = ""
            location = ""
            sib = link_el.next_sibling
            if isinstance(sib, str):
                company = sib.strip().strip(",").strip()
            if not company:
                nxt = link_el.find_next_sibling()
                if nxt is not None and nxt.name in ("span", "div"):
                    company = nxt.get_text(strip=True)
            br = link_el.find_next_sibling("br")
            if br is not None:
                parts = []
                for node in br.next_siblings:
                    if getattr(node, "name", None) in ("a", "br", "div"):
                        break  # the next posting starts
                    text = node.get_text(" ", strip=True) if hasattr(node, "get_text") else str(node).strip()
                    if text:
                        parts.append(text)
                location = " ".join(parts)

            results.append({
                "title": title,
                "company": company,
                "location": location,
                "job_url": href,
                "description": "",
                "date_posted": "",
                "site": "podnews",
                "department": "",
            })

        except Exception as e:
            logger.debug(f"Podnews parse error: {e}")
            continue

    logger.info(f"Podnews: {len(results)} matching jobs")
    return results


# =============================================================================
# 3. Music Business Worldwide
# =============================================================================

def scrape_mbw(keywords: list[str], max_pages: int = 1) -> list[dict]:
    """
    Scrape MusicBusinessWorldwide.com/jobs for music industry jobs.

    Behind a Cloudflare challenge (plain requests get 403), so every page is a
    ~10-20s stealth-browser fetch. The jobs page shows its ~40 current
    postings with no pagination links, hence one page by default;
    /jobs/page/N/ is the old WordPress pattern.
    """
    results = []
    seen_urls = set()

    for page in range(1, max_pages + 1):
        if page == 1:
            url = "https://www.musicbusinessworldwide.com/jobs/"
        else:
            url = f"https://www.musicbusinessworldwide.com/jobs/page/{page}/"

        soup = _fetch_page(url, stealth=True, wait_selector="li.post-card")
        if soup is None:
            break

        # Current markup (Avada theme): li.post-card, title / company /
        # location in div.fusion-text-1 / -2 / -3. Legacy WordPress
        # job_listing items kept as a fallback.
        job_items = soup.select("li.post-card")
        if not job_items:
            job_items = soup.select("li.job_listing, li.type-job_listing, div.job_listing")
        if not job_items:
            # Broader fallback
            job_items = soup.select("article, li.listing")

        if not job_items:
            break

        found_any = False
        for item in job_items:
            try:
                card_title = item.select_one("div.fusion-text-1")
                if card_title is not None:
                    link_el = item.find("a", href=lambda h: h and "/jobs/job/" in h)
                    title = card_title.get_text(strip=True)
                    href = link_el.get("href", "") if link_el else ""
                    company_el = item.select_one("div.fusion-text-2")
                    loc_el = item.select_one("div.fusion-text-3")
                    company = company_el.get_text(strip=True) if company_el else ""
                    location = loc_el.get_text(strip=True) if loc_el else ""
                else:
                    # Legacy: title from a > strong, company/location from
                    # the following links
                    title_el = item.find("a")
                    if not title_el:
                        continue
                    strong = title_el.find("strong")
                    title = (strong or title_el).get_text(strip=True)
                    href = title_el.get("href", "")
                    all_links = item.find_all("a")
                    company = all_links[1].get_text(strip=True) if len(all_links) >= 2 else ""
                    location = all_links[2].get_text(strip=True) if len(all_links) >= 3 else ""

                if not title or not href or href in seen_urls:
                    continue

                found_any = True  # new card parsed — drives pagination, not matching

                if not _matches_keywords(title, keywords):
                    continue

                seen_urls.add(href)

                results.append({
                    "title": title,
                    "company": company,
                    "location": location,
                    "job_url": href,
                    "description": "",
                    "date_posted": "",
                    "site": "mbw",
                    "department": "",
                })

            except Exception as e:
                logger.debug(f"MBW parse error: {e}")
                continue

        time.sleep(INTER_REQUEST_DELAY)

        if not found_any:
            break

    logger.info(f"MBW: {len(results)} matching jobs")
    return results


# =============================================================================
# 4. Doors Open
# =============================================================================

def scrape_doorsopen(keywords: list[str], max_pages: int = 3) -> list[dict]:
    """
    Scrape DoorsOpen.co/jobs for entertainment industry jobs.

    Uses article.listing-item structure with pagination. The site's nginx
    answers plain requests with 403; the stealth browser
    gets the page (no Cloudflare challenge, ~4s). Its postings are almost all
    UK / EU, so the location gate drops most of them downstream.
    """
    results = []
    seen_urls = set()

    for page in range(1, max_pages + 1):
        if page == 1:
            url = "https://www.doorsopen.co/jobs/"
        else:
            url = f"https://www.doorsopen.co/jobs/?page={page}"

        # No Cloudflare challenge here — only an nginx rule against plain
        # clients — so the solver is skipped (it logs an ERROR when it finds
        # no challenge).
        soup = _fetch_page(url, stealth=True, solve_cloudflare=False)
        if soup is None:
            break

        articles = soup.select("article.listing-item, div.listing-item, div.job-card")
        if not articles:
            # Broader fallback
            articles = soup.select("article, div.job-listing")

        if not articles:
            break

        found_any = False
        for article in articles:
            try:
                # Current markup: div.listing-item__title > a (legacy: a
                # heading link). The card's first bare anchor is the logo
                # link, which has no text — falling back to it is what had
                # this source silently at 0 rows.
                title_el = article.select_one(
                    ".listing-item__title a, h3 a, h4 a, a.job-title")
                if not title_el:
                    title_el = article.find("a", href=True)
                if not title_el:
                    continue

                title = title_el.get_text(strip=True)
                href = title_el.get("href", "")
                if href and not href.startswith("http"):
                    href = f"https://www.doorsopen.co{href}"

                if not title or href in seen_urls:
                    continue

                seen_urls.add(href)
                found_any = True  # new card parsed — drives pagination, not matching

                if not _matches_keywords(title, keywords):
                    continue

                # Company + Location: span (current) or li (legacy)
                # .listing-item__info--item
                info_items = article.select(".listing-item__info--item, span.company, span.location")
                company = info_items[0].get_text(strip=True) if len(info_items) >= 1 else ""
                location = info_items[1].get_text(strip=True) if len(info_items) >= 2 else ""

                # Date ("Mar 4, 2026" on the current page)
                date_el = article.select_one(".listing-item__date, time, span.date")
                date_posted = parse_posted_date(date_el.get_text(strip=True)) if date_el else ""

                desc_el = article.select_one(".listing-item__desc")
                description = desc_el.get_text(" ", strip=True) if desc_el else ""

                results.append({
                    "title": title,
                    "company": company,
                    "location": location,
                    "job_url": href,
                    "description": description,
                    "date_posted": date_posted,
                    "site": "doorsopen",
                    "department": "",
                })

            except Exception as e:
                logger.debug(f"DoorsOpen parse error: {e}")
                continue

        time.sleep(INTER_REQUEST_DELAY)

        if not found_any:
            break

    logger.info(f"DoorsOpen: {len(results)} matching jobs")
    return results


# =============================================================================
# 5. The Digilogue
# =============================================================================

def scrape_digilogue(keywords: list[str]) -> list[dict]:
    """
    Scrape TheDigilogue.com/career-listings for creative/media jobs.

    Webflow-based, single page (curated weekly).
    """
    url = "https://www.thedigilogue.com/career-listings"
    soup = _fetch_page(url)
    if soup is None:
        return []

    results = []

    # Webflow uses various div structures; look for job cards
    cards = soup.select("div.job-card, div.w-dyn-item, div.collection-item")
    if not cards:
        # Broader fallback
        cards = soup.select("div.job-item, article, div.listing")

    for card in cards:
        try:
            # Title from h3 or heading element
            title_el = card.find(["h3", "h4", "h2"])
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            if not title:
                continue

            if not _matches_keywords(title, keywords):
                continue

            # Link: look for "apply" link or any job-related link
            link_el = card.find("a", href=True)
            href = link_el.get("href", "") if link_el else ""
            if href and not href.startswith("http"):
                href = f"https://www.thedigilogue.com{href}"

            # Company + Location from icon-labeled divs or text blocks.
            # Hardened 2026-09-01: the loose heuristics made "APPLY" (a button
            # label) the company and a whole-JD text div the location (a
            # corrupted row reached the feed).
            company = ""
            location = ""
            longest_text = ""
            _BUTTON_NOISE = {"apply", "apply now", "show more", "show less",
                             "learn more", "view job", "see more"}
            text_blocks = card.find_all(["div", "span", "p"])
            for block in text_blocks:
                text = block.get_text(strip=True)
                if not text or text == title:
                    continue
                if text.lower() in _BUTTON_NOISE:
                    continue
                if len(text) > len(longest_text):
                    longest_text = text
                # Heuristic: company is usually first non-title text
                if not company and len(text) < 100:
                    company = text
                # A real location is short — a huge "location" is the JD text
                # of a mis-parsed card.
                elif not location and len(text) <= 80 and any(
                        s in text.lower() for s in ["remote", "ca", "ny", "tx", ","]):
                    location = text

            results.append({
                "title": title,
                "company": company,
                "location": location,
                "job_url": href,
                # The card's longest text block IS the JD on the current page
                # layout — without it every digilogue job graded evidence-free.
                "description": longest_text if len(longest_text) >= 300 else "",
                "date_posted": "",
                "site": "digilogue",
                "department": "",
            })

        except Exception as e:
            logger.debug(f"Digilogue parse error: {e}")
            continue

    logger.info(f"Digilogue: {len(results)} matching jobs")
    return results


# =============================================================================
# 6. Remote Game Jobs
# =============================================================================

def scrape_remotegamejobs(keywords: list[str]) -> list[dict]:
    """
    Scrape RemoteGameJobs.com for remote gaming industry jobs.

    Single page (~33 newest postings). Since the redesign each posting is an
    <a class="card"> with the title in div.title > h3, the company in div.sub
    and the region in span.scope; the old div.job-box markup (kept as a
    fallback) finds nothing on the current page.
    """
    url = "https://remotegamejobs.com/"
    soup = _fetch_page(url)
    if soup is None:
        return []

    results = []

    boxes = soup.select("a.card")
    if not boxes:
        # Legacy markup
        boxes = soup.select("div.job-box, div.job-card, article.job, div.job-listing")

    for box in boxes:
        try:
            title_el = (box.select_one("div.title h3") or box.find(["h2", "h3", "h4"])
                        or box.select_one("a strong"))
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            if not title:
                continue

            if not _matches_keywords(title, keywords):
                continue

            # The card itself is the link on the current markup
            if box.name == "a":
                href = box.get("href", "")
            else:
                link_el = box.find("a", href=True)
                href = link_el.get("href", "") if link_el else ""
            if href and not href.startswith("http"):
                href = f"https://remotegamejobs.com{href}"

            company = ""
            company_el = box.select_one("div.sub, span.company, div.company, p.company")
            if company_el:
                company = company_el.get_text(strip=True)

            # "Remote" alone says nothing about where the candidate may work from; the
            # card names the allowed region ("Worldwide", "USA", "Europe").
            scope_el = box.select_one("span.scope")
            scope = scope_el.get_text(" ", strip=True) if scope_el else ""
            location = "Remote" if not scope or scope.lower() == "remote" else f"Remote ({scope})"

            results.append({
                "title": title,
                "company": company,
                "location": location,
                "job_url": href,
                "description": "",
                "date_posted": "",
                "site": "remotegamejobs",
                "department": "",
            })

        except Exception as e:
            logger.debug(f"RemoteGameJobs parse error: {e}")
            continue

    logger.info(f"RemoteGameJobs: {len(results)} matching jobs")
    return results


# =============================================================================
# 7. WillHire (Apple Contingent Workforce)
# =============================================================================

WILLHIRE_BASE = "https://applecontingentworkforce.willhire.co"

# WillHire cards say "Yesterday", "Today", "3 hours ago", "An hour ago" or
# "Jan 22, 2026". Stored raw, the relative ones never parsed downstream (no
# age pill, no stale gate) — convert them at scrape time, when "today" is known.
_REL_AGO_RE = re.compile(
    r"^(an?|\d+)\s+(minute|min|hour|hr|day|week|month)s?\s+ago$", re.I)
_REL_UNIT_DAYS = {"day": 1, "week": 7, "month": 30}
# ShowbizJobs' short forms: "2d ago", "1w ago", "3h ago"
_SHORT_AGO_RE = re.compile(r"^(\d+)\s*([hdw])\s+ago$", re.I)
_SHORT_UNIT = {"h": "hour", "d": "day", "w": "week"}


def parse_posted_date(text: str, now: datetime | None = None) -> str:
    """ISO date (YYYY-MM-DD) for a WillHire 'Posted:' value; the stripped
    input when it is not a recognisable date ('' stays '')."""
    raw = (text or "").strip()
    if not raw:
        return ""
    now = now or datetime.now()
    low = raw.lower().rstrip(".")
    if low in ("today", "just now", "just posted", "moments ago"):
        return now.strftime("%Y-%m-%d")
    if low == "yesterday":
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    m = _REL_AGO_RE.match(low)
    short = _SHORT_AGO_RE.match(low)
    if m or short:
        if m:
            n = 1 if m.group(1) in ("a", "an") else int(m.group(1))
            unit = m.group(2).lower()
        else:
            n, unit = int(short.group(1)), _SHORT_UNIT[short.group(2).lower()]
        if unit in ("minute", "min", "hour", "hr"):
            delta = timedelta(minutes=n) if unit in ("minute", "min") else timedelta(hours=n)
        else:
            delta = timedelta(days=n * _REL_UNIT_DAYS[unit])
        return (now - delta).strftime("%Y-%m-%d")
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # "Oct 16" (ShowbizJobs, older postings): this year, or last year when
    # that date is still ahead of today.
    try:
        d = datetime.strptime(f"{raw} {now.year}", "%b %d %Y")
        if d.date() > now.date():
            d = d.replace(year=now.year - 1)
        return d.strftime("%Y-%m-%d")
    except ValueError:
        pass
    return raw


def scrape_willhire(keywords: list[str], max_pages: int = 4) -> list[dict]:
    """
    Scrape Apple contingent workforce jobs from WillHire.

    Server-rendered HTML with pagination via ?page=N&size=100.
    Job cards: div.job-card with a.job-title, span.info-text for location/date.
    """
    results = []
    seen_urls: set[str] = set()

    for page in range(1, max_pages + 1):
        url = f"{WILLHIRE_BASE}/jobs?page={page}&size=100"
        soup = _fetch_page(url)
        if soup is None:
            break

        cards = soup.select("div.job-card")
        if not cards:
            break

        for card in cards:
            try:
                title_el = card.select_one("a.job-title")
                if not title_el:
                    continue

                title = title_el.get_text(strip=True)
                href = title_el.get("href", "")
                if href and not href.startswith("http"):
                    href = f"{WILLHIRE_BASE}{href}"

                if not title or href in seen_urls:
                    continue

                if not _matches_keywords(title, keywords):
                    continue

                # Location: first span.info-text inside the info row
                info_texts = card.select("span.info-text")
                location = ""
                date_posted = ""
                for span in info_texts:
                    text = span.get_text(strip=True)
                    if text.startswith("Posted:"):
                        date_posted = parse_posted_date(text.replace("Posted:", ""))
                    elif not location:
                        location = text

                seen_urls.add(href)
                results.append({
                    "title": title,
                    "company": "Apple (via WillHire)",
                    "location": location,
                    "link": href,
                    "job_url": href,
                    "description": "",
                    "date_posted": date_posted,
                    "site": "willhire",
                    "department": "",
                })

            except Exception as e:
                logger.debug(f"WillHire parse error: {e}")
                continue

        time.sleep(INTER_REQUEST_DELAY)

    logger.info(f"WillHire: {len(results)} matching jobs")
    return results


# =============================================================================
# 8. APMG Careers (American Public Media Group — UltiPro)
# =============================================================================

APMG_API_URL = (
    "https://recruiting2.ultipro.com/AME1098APMG/JobBoard/"
    "4b7ae4eb-a67b-4318-80fc-6d9467f9c542/JobBoardView/LoadSearchResults"
)
APMG_JOB_URL = (
    "https://recruiting2.ultipro.com/AME1098APMG/JobBoard/"
    "4b7ae4eb-a67b-4318-80fc-6d9467f9c542/OpportunityDetail?opportunityId={}"
)
_APMG_LOCATION_TYPES = {0: "On-site", 1: "Hybrid", 2: "Remote"}


def scrape_apmg(keywords: list[str]) -> list[dict]:
    """
    Scrape APMG Careers (American Public Media Group) via UltiPro REST API.
    No authentication required.
    """
    results = []
    try:
        resp = _session.post(
            APMG_API_URL,
            json={"opportunitySearch": {"Top": 100, "Skip": 0}},
            headers={"X-Requested-With": "XMLHttpRequest"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"APMG API error: {e}")
        return []

    for opp in data.get("opportunities", []):
        try:
            title = opp.get("Title", "").strip()
            if not title or not _matches_keywords(title, keywords):
                continue

            opp_id = opp.get("Id", "")
            job_url = APMG_JOB_URL.format(opp_id)

            # Build location string
            loc_type = opp.get("JobLocationType", 0)
            locations = opp.get("Locations", [])
            if loc_type == 2:
                location = "Remote"
            elif locations:
                addr = locations[0].get("Address", {})
                city = addr.get("City", "")
                state = (addr.get("State") or {}).get("Code", "")
                location = f"{city}, {state}".strip(", ")
                if loc_type == 1:
                    location = f"{location} (Hybrid)" if location else "Hybrid"
            else:
                location = _APMG_LOCATION_TYPES.get(loc_type, "")

            results.append({
                "title": title,
                "company": "American Public Media Group",
                "location": location,
                "job_url": job_url,
                "description": opp.get("BriefDescription", ""),
                "date_posted": opp.get("PostedDate", ""),
                "site": "apmg",
                "department": opp.get("JobCategoryName", ""),
            })
        except Exception as e:
            logger.debug(f"APMG parse error: {e}")
            continue

    logger.info(f"APMG: {len(results)} matching jobs")
    return results


# =============================================================================
# Orchestrator
# =============================================================================

def scrape_industry_html_boards(
    keywords: list[str],
    progress_callback=None,
) -> pd.DataFrame:
    """
    Orchestrator: scrape all HTML-based industry job boards.

    Args:
        keywords: Keywords to filter job titles
        progress_callback: Optional callback(message, progress_float)

    Returns:
        DataFrame of all matching jobs
    """
    all_jobs = []

    scrapers = [
        ("ShowbizJobs", lambda: scrape_showbizjobs(keywords)),
        ("Podnews", lambda: scrape_podnews(keywords)),
        ("MBW", lambda: scrape_mbw(keywords)),
        ("DoorsOpen", lambda: scrape_doorsopen(keywords)),
        ("Digilogue", lambda: scrape_digilogue(keywords)),
        ("RemoteGameJobs", lambda: scrape_remotegamejobs(keywords)),
        ("WillHire", lambda: scrape_willhire(keywords)),
        ("APMG", lambda: scrape_apmg(keywords)),
        ("ProductionHub", lambda: scrape_productionhub(keywords)),
        ("Hitmarker", lambda: scrape_hitmarker(keywords)),
        ("BuiltIn", lambda: scrape_builtin(keywords)),
    ]

    for i, (name, scraper_fn) in enumerate(scrapers):
        if progress_callback:
            progress_callback(
                f"Industry: Scanning {name}...",
                0.75 + (i / len(scrapers)) * 0.03,
            )

        try:
            jobs = scraper_fn()
            all_jobs.extend(jobs)
            if not jobs:
                # Zero yield is the signature of silent selector rot (MBW,
                # DoorsOpen, RemoteGameJobs all sat at 0 rows for months
                # unnoticed). WARN so it shows in logs/digests, not just INFO.
                logger.warning(f"Industry scraper {name}: 0 jobs — possible selector rot or dry source")
        except Exception as e:
            # main.py's phase budget (SIGALRM) arrives as _ScrapeTimeout: let
            # it end the phase instead of moving on to the next board without
            # a timer. Matched by name: importing main here would be circular.
            if type(e).__name__ == "_ScrapeTimeout":
                raise
            logger.warning(f"Industry scraper {name} failed: {e}")

    if progress_callback:
        progress_callback(f"Industry boards: found {len(all_jobs)} matching jobs", 0.78)

    if not all_jobs:
        return pd.DataFrame()

    return pd.DataFrame(all_jobs)
