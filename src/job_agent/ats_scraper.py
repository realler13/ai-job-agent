
import logging
import re
import threading
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import KEYWORD_TO_CATEGORY
from .utils import normalize_date_posted

logger = logging.getLogger(__name__)


def _strip_html(html: str) -> str:
    """Strip HTML tags and return plain text."""
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def _matches_keywords(title: str, keywords: list[str]) -> bool:
    """Check if any keyword appears in the job title (case-insensitive)."""
    title_lower = title.lower()
    for kw in keywords:
        # Match if all words in the keyword phrase appear in the title
        kw_words = kw.lower().split()
        if all(w in title_lower for w in kw_words):
            return True
    return False


# === Phase 27c: "is this posting still listed?" ===
# A full-list ATS scraper reports, once per call, how its board fetch went and
# EVERY listed posting link on the board — before the keyword filter, so the
# record does not depend on the keywords of the search that ran it. The report
# sits in a per-thread slot that scrape_ats_boards reads after each board and
# hands to db.record_board_fetch. Workday / Apple (seed keyword searches) and
# Teamtailor (paginated HTML) never report: absence there proves nothing.
_board_report_slot = threading.local()


def report_board_fetch(platform: str, token: str, *, ok: bool, complete: bool,
                       total_count: int, links: list) -> None:
    """Called by an ELIGIBLE scraper once per call, after its list fetch ends
    (also on failure: ok=False). links = every listed posting on the board
    BEFORE the keyword filter, as (link, title) pairs (a bare link is also
    accepted), the link built with the same expression the scraper uses for
    job_url. The title lets a role re-posted under a new link read as "still
    listed", not "gone". Never raises."""
    try:
        pairs = []
        for item in links or []:
            link, title = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item, "")
            if link:
                pairs.append((str(link), str(title or "")))
        _board_report_slot.value = {
            "platform": platform,
            "token": token,
            "ok": bool(ok),
            "complete": bool(ok) and bool(complete),
            "total_count": int(total_count or 0),
            "links": pairs,
        }
    except Exception as e:  # a report must never cost a scrape
        logger.debug(f"Board fetch report skipped for {platform}:{token}: {e}")


def _clear_board_report() -> None:
    _board_report_slot.value = None


def _take_board_report():
    """The report the last scraper left in this thread's slot (then cleared)."""
    report = getattr(_board_report_slot, "value", None)
    _board_report_slot.value = None
    return report


def _resolve_keyword_category(title: str, keywords: list[str]) -> str:
    """Return the category of the LONGEST matching keyword for a title.

    Longest-match-wins mirrors resolve_cv_domain: "Senior Podcast Producer"
    must resolve via "Podcast Producer" (podcast_and_audio), not the bare
    "Producer" (producer_core) that happens to come first in KEYWORDS order.
    This stamp feeds the cascade's domain context AND the CV-variant choice,
    so a first-match stamp was silently selecting the wrong CV.
    """
    title_lower = title.lower()
    best_kw = None
    best_len = -1
    for kw in keywords:
        kw_lower = kw.lower()
        if all(w in title_lower for w in kw_lower.split()) and len(kw_lower) > best_len:
            best_kw = kw_lower
            best_len = len(kw_lower)
    return KEYWORD_TO_CATEGORY.get(best_kw, "") if best_kw else ""


def scrape_greenhouse(token: str, company: str, keywords: list[str]) -> list[dict]:
    """
    Scrape jobs from a Greenhouse public job board.

    Args:
        token: The Greenhouse board token (e.g. "airbnb")
        company: Company display name
        keywords: List of keyword phrases to filter by title

    Returns:
        List of standardized job dicts
    """
    url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Greenhouse API error for {company} ({token}): {e}")
        report_board_fetch("greenhouse", token, ok=False, complete=False, total_count=0, links=[])
        return []

    jobs_data = data.get("jobs", [])
    # Complete = a jobs list whose length matches the board's own total (when
    # the payload carries one).
    _listed = jobs_data if isinstance(jobs_data, list) else []
    _meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    _total = _meta.get("total")
    report_board_fetch(
        "greenhouse", token,
        ok=isinstance(data.get("jobs"), list),
        complete=_total is None or _total == len(_listed),
        total_count=len(_listed),
        links=[(j.get("absolute_url", ""), j.get("title", "")) for j in _listed if isinstance(j, dict)],
    )
    results = []

    for job in jobs_data:
        title = job.get("title", "")
        if not _matches_keywords(title, keywords):
            continue

        location_name = ""
        if job.get("location"):
            location_name = job["location"].get("name", "")

        departments = job.get("departments", [])
        department = (
            departments[0].get("name", "")
            if departments and isinstance(departments[0], dict)
            else ""
        )

        # Brand-name override for multi-brand boards (e.g. PMC hosts 8 brands under
        # one Greenhouse token: Hollywood Reporter, Deadline, Variety, Rolling Stone,
        # Billboard, IndieWire, Artforum, WWD). If Greenhouse's metadata contains a
        # "Brand Name" field, prefer that over the configured company name so each
        # posting shows its real brand in the Kanban.
        resolved_company = _extract_brand_from_metadata(job) or company

        row = {
            "title": title,
            "company": resolved_company,
            "location": location_name,
            "job_url": job.get("absolute_url", ""),
            "description": _strip_html(job.get("content", "")),
            # first_published = the real posting date; updated_at bumps on ANY edit,
            # making stale postings look fresh and bypass is_posting_old().
            "date_posted": normalize_date_posted(job.get("first_published") or job.get("updated_at", "")),
            "site": "greenhouse",
            "department": department,
        }
        # Hiring-team capture (People Intel spec M1, source #1 — the posting
        # itself): some boards expose "Hiring Manager"/"Recruiter" custom
        # metadata (e.g. DoorDash names them on ~98% of jobs). Free — it's in
        # the payload we already fetch; flows into raw_data via save_job and
        # feeds the outreach playbook as its most-reliable contact source.
        contact = _extract_hiring_contact_from_metadata(job)
        if contact:
            row["hiring_contact"] = contact
        results.append(row)

    logger.info(f"Greenhouse ({company}): {len(results)} matching jobs from {len(jobs_data)} total")
    return results


def _extract_brand_from_metadata(job: dict) -> str:
    """Pull the 'Brand Name' from a Greenhouse job's metadata array, if present.

    Greenhouse's API exposes custom fields via a list of
    {id, name, type, value} dicts at job["metadata"]. For multi-brand parent
    companies (PMC, Condé Nast, etc.), the brand of each posting lives there.
    Returns empty string if no usable brand is found.
    """
    metadata = job.get("metadata") or []
    if not isinstance(metadata, list):
        return ""
    for field in metadata:
        if not isinstance(field, dict):
            continue
        name = (field.get("name") or "").strip().lower()
        if name in ("brand name", "brand"):
            value = field.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


_HIRING_META_RE = re.compile(r"hiring\s*manager|recruiter|hiring\s*team", re.IGNORECASE)


def _extract_hiring_contact_from_metadata(job: dict) -> str:
    """Pull hiring-manager/recruiter names from Greenhouse custom metadata.

    Returns e.g. "Hiring Manager with Job Title: Jane Doe - Team Lead"
    or "" when the board doesn't expose people fields (most don't — this is
    company-specific custom metadata, verified populated on e.g. DoorDash)."""
    metadata = job.get("metadata") or []
    if not isinstance(metadata, list):
        return ""
    parts = []
    for field in metadata:
        if not isinstance(field, dict):
            continue
        name = (field.get("name") or "").strip()
        value = field.get("value")
        if name and _HIRING_META_RE.search(name) and isinstance(value, str) and value.strip():
            parts.append(f"{name}: {value.strip()}")
    return "; ".join(parts)[:300]


def scrape_lever(token: str, company: str, keywords: list[str]) -> list[dict]:
    """
    Scrape jobs from a Lever public job board.

    Args:
        token: The Lever company token (e.g. "twitch")
        company: Company display name
        keywords: List of keyword phrases to filter by title

    Returns:
        List of standardized job dicts
    """
    url = f"https://api.lever.co/v0/postings/{token}?mode=json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Lever API error for {company} ({token}): {e}")
        report_board_fetch("lever", token, ok=False, complete=False, total_count=0, links=[])
        return []

    if not isinstance(data, list):
        logger.warning(f"Lever ({company}): unexpected response format")
        report_board_fetch("lever", token, ok=False, complete=False, total_count=0, links=[])
        return []

    report_board_fetch(
        "lever", token, ok=True, complete=True, total_count=len(data),
        links=[(j.get("hostedUrl", ""), j.get("text", "")) for j in data if isinstance(j, dict)],
    )

    results = []

    for job in data:
        title = job.get("text", "")  # Lever uses 'text' not 'title'
        if not _matches_keywords(title, keywords):
            continue

        categories = job.get("categories", {})

        results.append({
            "title": title,
            "company": company,
            "location": categories.get("location", ""),
            "job_url": job.get("hostedUrl", ""),
            "description": _strip_html(job.get("descriptionPlain", "") or job.get("description", "")),
            "date_posted": normalize_date_posted(job.get("createdAt", "")),
            "site": "lever",
            "department": categories.get("department", ""),
        })

    logger.info(f"Lever ({company}): {len(results)} matching jobs from {len(data)} total")
    return results


def scrape_ats_boards(
    companies: list[dict],
    keywords: list[str],
    progress_callback=None,
    db=None,
    run_id=None,
    deadline: float | None = None,
) -> pd.DataFrame:
    """
    Scrape all configured ATS boards.

    Supports: greenhouse, lever, ashby, smartrecruiters, workday,
    recruitee, workable, teamtailor, eightfold, apple, bamboohr, comeet, phenom.

    Args:
        companies: List of dicts with keys: company, ats, token
        keywords: Keywords to filter job titles
        progress_callback: Optional callback(message, progress_float)
        db: Optional DBManager — when given, each full-list board's fetch
            report is recorded (Phase 27c "still listed?"). Recording never
            changes which jobs are returned.
        run_id: Session id stamped on the recorded fetches.
        deadline: Optional time.monotonic() value. A board is started only
            while one Workday board's time budget (plus 20 s) is left before
            it; otherwise the scan stops and returns what it collected. The
            caller's SIGALRM would throw all of it away.

    Returns:
        DataFrame of all matching jobs
    """
    # Lazy imports to avoid circular deps
    from .scraper_ashby_sr import scrape_ashby, scrape_smartrecruiters
    from .scraper_bamboohr_comeet import scrape_bamboohr, scrape_comeet
    from .scraper_eightfold_apple import scrape_apple, scrape_eightfold
    from .scraper_phenom import scrape_phenom
    from .scraper_rwt import scrape_recruitee, scrape_teamtailor, scrape_workable
    from .scraper_workday import _BOARD_TIME_BUDGET, scrape_workday

    all_jobs = []
    total = len(companies)

    for i, entry in enumerate(companies):
        if deadline is not None and deadline - time.monotonic() < _BOARD_TIME_BUDGET + 20:
            logger.warning(
                f"ATS: time nearly up — {total - i} of {total} boards skipped this run "
                f"(from {entry['company']}); keeping the {len(all_jobs)} jobs already found"
            )
            break
        company = entry["company"]
        ats = entry["ats"]
        token = entry["token"]

        if progress_callback:
            progress_callback(f"ATS: Scanning {company} ({ats})...", i / max(total, 1))

        _clear_board_report()
        try:
            if ats == "greenhouse":
                jobs = scrape_greenhouse(token, company, keywords)
            elif ats == "lever":
                jobs = scrape_lever(token, company, keywords)
            elif ats == "ashby":
                jobs = scrape_ashby(token, company, keywords)
            elif ats == "smartrecruiters":
                jobs = scrape_smartrecruiters(token, company, keywords)
            elif ats == "workday":
                jobs = scrape_workday(token, company, keywords)
            elif ats == "recruitee":
                jobs = scrape_recruitee(token, company, keywords)
            elif ats == "workable":
                jobs = scrape_workable(token, company, keywords)
            elif ats == "teamtailor":
                jobs = scrape_teamtailor(token, company, keywords)
            elif ats == "eightfold":
                jobs = scrape_eightfold(token, company, keywords)
            elif ats == "apple":
                jobs = scrape_apple(token, company, keywords)
            elif ats == "bamboohr":
                jobs = scrape_bamboohr(token, company, keywords)
            elif ats == "comeet":
                jobs = scrape_comeet(token, company, keywords)
            elif ats == "phenom":
                jobs = scrape_phenom(token, company, keywords)
            else:
                logger.warning(f"Unknown ATS type '{ats}' for {company}, skipping")
                continue
        except Exception as e:
            # One broken board (a malformed posting, an unexpected API shape)
            # must not stop every board after it. The pipeline's phase budget
            # (main._ScrapeTimeout, raised by SIGALRM) still has to end the
            # phase, so it passes through.
            if type(e).__name__ == "_ScrapeTimeout":
                raise
            logger.warning(f"ATS board {company} ({ats}) failed, skipping: "
                           f"{type(e).__name__}: {e}")
            _clear_board_report()  # a half-finished fetch is not a board list
            continue

        if db is not None:
            _record_board_report(db, company, run_id)

        # Annotate each job with keyword_category based on title match
        for job in jobs:
            if "keyword_category" not in job or not job["keyword_category"]:
                job["keyword_category"] = _resolve_keyword_category(
                    job.get("title", ""), keywords
                )
        all_jobs.extend(jobs)

    if progress_callback:
        progress_callback(f"ATS: Found {len(all_jobs)} jobs from {total} companies", 1.0)

    if not all_jobs:
        return pd.DataFrame()

    return pd.DataFrame(all_jobs)


def _record_board_report(db, company: str, run_id) -> None:
    """Hand this board's fetch report (if its scraper left one) to the DB.
    Non-blocking: any failure is logged and the scan goes on — except the
    pipeline's own phase timeout (main._ScrapeTimeout), which must still stop
    the ATS phase."""
    report = _take_board_report()
    if not report:
        return
    try:
        db.record_board_fetch(
            report["platform"], report["token"], company,
            ok=report["ok"], complete=report["complete"],
            total_count=report["total_count"], links=report["links"],
            run_id=run_id,
        )
    except Exception as e:
        if type(e).__name__ == "_ScrapeTimeout":
            raise
        logger.warning(f"Board fetch record failed for {company} "
                       f"({report['platform']}:{report['token']}): {e}")
