
import logging
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .ats_scraper import _matches_keywords
from .scraper_productionhub import scrape_productionhub
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


def _fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch a URL and return parsed BeautifulSoup, or None on error."""
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

            # ShowbizJobs uses table#postings with tr rows
            table = soup.find("table", id="postings")
            if not table:
                # Try alternate: div-based listing
                rows = soup.select("div.job-listing, tr.job-row, div.listing")
                if not rows:
                    break
            else:
                rows = table.find_all("tr")

            if not rows:
                break

            found_any = False
            for row in rows:
                try:
                    # Title from h5 > a
                    title_el = row.find("h5")
                    if not title_el:
                        continue
                    title_link = title_el.find("a")
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

                    # Company from h6 > a (first h6)
                    company = ""
                    h6_tags = row.find_all("h6")
                    if h6_tags:
                        company_link = h6_tags[0].find("a")
                        if company_link:
                            company = company_link.get_text(strip=True)
                        else:
                            company = h6_tags[0].get_text(strip=True)

                    # Location: text content in td elements
                    location = ""
                    tds = row.find_all("td")
                    for td in tds:
                        text = td.get_text(strip=True)
                        # Location patterns: "City, State" or state abbreviations
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
                        "date_posted": "",
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

    # Podnews uses div.artblock for each job listing
    blocks = soup.select("div.artblock")
    if not blocks:
        # Fallback: look for article or link-based patterns
        blocks = soup.select("article, div.job-item, div.post")

    for block in blocks:
        try:
            # Title + link from a[href^="/job/"]
            link_el = block.find("a", href=lambda h: h and "/job/" in h)
            if not link_el:
                continue

            title = link_el.get_text(strip=True)
            href = link_el.get("href", "")
            if href and not href.startswith("http"):
                href = f"https://podnews.net{href}"

            if not title:
                continue

            if not _matches_keywords(title, keywords):
                continue

            # Company: text after title, often comma-separated
            company = ""
            block_text = block.get_text(separator="\n", strip=True)
            lines = [l.strip() for l in block_text.split("\n") if l.strip()]
            # Company is typically the line after the title
            for i, line in enumerate(lines):
                if line == title and i + 1 < len(lines):
                    company = lines[i + 1].strip(" ,")
                    break

            # Location: look for flag emoji or location text
            location = ""
            for line in lines:
                if any(flag in line for flag in ["\U0001f1fa\U0001f1f8", "US", "UK", "Remote", "CA", "NY"]):
                    location = line.strip()
                    break

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

def scrape_mbw(keywords: list[str], max_pages: int = 5) -> list[dict]:
    """
    Scrape MusicBusinessWorldwide.com/jobs for music industry jobs.

    WordPress job listing with pagination via /jobs/page/N/.
    """
    results = []
    seen_urls = set()

    for page in range(1, max_pages + 1):
        if page == 1:
            url = "https://www.musicbusinessworldwide.com/jobs/"
        else:
            url = f"https://www.musicbusinessworldwide.com/jobs/page/{page}/"

        soup = _fetch_page(url)
        if soup is None:
            break

        # WordPress job listings: li items with job links
        job_items = soup.select("li.job_listing, li.type-job_listing, div.job_listing")
        if not job_items:
            # Broader fallback
            job_items = soup.select("article, li.listing")

        if not job_items:
            break

        found_any = False
        for item in job_items:
            try:
                # Title from a > strong or a with job link
                title_el = item.find("a")
                if not title_el:
                    continue

                # Check for strong tag inside (common WP pattern)
                strong = title_el.find("strong")
                title = strong.get_text(strip=True) if strong else title_el.get_text(strip=True)

                href = title_el.get("href", "")
                if not title or not href or href in seen_urls:
                    continue

                found_any = True  # new card parsed — drives pagination, not matching

                if not _matches_keywords(title, keywords):
                    continue

                # Company and location from subsequent <a> tags or text
                company = ""
                location = ""
                all_links = item.find_all("a")
                if len(all_links) >= 2:
                    company = all_links[1].get_text(strip=True)
                if len(all_links) >= 3:
                    location = all_links[2].get_text(strip=True)

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

    Uses article.listing-item structure with pagination.
    """
    results = []
    seen_urls = set()

    for page in range(1, max_pages + 1):
        if page == 1:
            url = "https://www.doorsopen.co/jobs/"
        else:
            url = f"https://www.doorsopen.co/jobs/?page={page}"

        soup = _fetch_page(url)
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
                # Title from heading link (legacy) or the card's primary
                # anchor (current markup: title is plain link text — selector
                # rot had this source silently at 0 rows).
                title_el = article.select_one("h4.listing-item__title a, h3 a, h4 a, a.job-title")
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

                if not _matches_keywords(title, keywords):
                    continue

                # Company + Location from listing-item__info--item
                info_items = article.select("li.listing-item__info--item, span.company, span.location")
                company = info_items[0].get_text(strip=True) if len(info_items) >= 1 else ""
                location = info_items[1].get_text(strip=True) if len(info_items) >= 2 else ""

                # Date
                date_el = article.select_one(".listing-item__date, time, span.date")
                date_posted = date_el.get_text(strip=True) if date_el else ""

                seen_urls.add(href)
                found_any = True

                results.append({
                    "title": title,
                    "company": company,
                    "location": location,
                    "job_url": href,
                    "description": "",
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

    Single page with div.job-box items.
    """
    url = "https://remotegamejobs.com/"
    soup = _fetch_page(url)
    if soup is None:
        return []

    results = []

    # Job listings in div.job-box or similar containers
    boxes = soup.select("div.job-box, div.job-card, article.job, div.job-listing")
    if not boxes:
        # Broader fallback
        boxes = soup.select("div.job, li.job, div.listing")

    for box in boxes:
        try:
            # Title from heading (legacy markup) or <strong> inside the card
            # link (current markup: cards carry no headings — selector rot had
            # this source silently at 0 rows).
            title_el = box.find(["h2", "h3", "h4"]) or box.select_one("a strong")
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            if not title:
                continue

            if not _matches_keywords(title, keywords):
                continue

            # Link
            link_el = box.find("a", href=True)
            href = link_el.get("href", "") if link_el else ""
            if href and not href.startswith("http"):
                href = f"https://remotegamejobs.com{href}"

            # Company: current markup lists (title, company, worktype) as the
            # info-column strings; legacy selectors kept as fallback.
            company = ""
            col = box.select_one("div.column")
            if col:
                strings = [t.strip() for t in col.stripped_strings if t.strip()]
                if len(strings) >= 2 and strings[0] == title:
                    company = strings[1]
            if not company:
                company_el = box.select_one("span.company, div.company, p.company")
                if company_el:
                    company = company_el.get_text(strip=True)
                else:
                    for el in box.find_all(["span", "div", "p"]):
                        text = el.get_text(strip=True)
                        if text and text != title and len(text) < 80:
                            company = text
                            break

            results.append({
                "title": title,
                "company": company,
                "location": "Remote",
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
    if m:
        n = 1 if m.group(1) in ("a", "an") else int(m.group(1))
        unit = m.group(2).lower()
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
            logger.warning(f"Industry scraper {name} failed: {e}")

    if progress_callback:
        progress_callback(f"Industry boards: found {len(all_jobs)} matching jobs", 0.78)

    if not all_jobs:
        return pd.DataFrame()

    return pd.DataFrame(all_jobs)
