"""Posting check: "Is this posting real?" — facts, never a score.

Everything here describes a posting; nothing feeds tier, confidence,
match_level, reviewed or ordering. The facts:

  - listing status from the company's own job board (db.get_listing_status:
    "gone" only after two complete board fetches without it),
  - posted N days ago, first seen by our scanner, re-listing count,
  - an unusually wide stated pay range,
  - dated layoffs / freezes / cuts from the CACHED hiring-health research,
  - the candidate's earlier applications at the same company (history, not a
    legitimacy signal),
  - one cheap LLM read of how specific the posting is (tools, team, first
    months' scope, pay, contradictions, boilerplate) with verbatim quotes —
    cite-or-omit is enforced here in code, not trusted to the model.

The overall read ("Looks active" / "Mixed signals" / "Check before investing
time") is a deterministic summary of the free facts (see overall_read). The
quality read never feeds it. build_posting_check / feed_reads / caution_line
make no network or LLM call; only run_quality_read(s) calls the model.
"""
import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from .config import (
    POSTING_ACTIVE_POSTED_DAYS,
    POSTING_CUTS_WINDOW_DAYS,
    POSTING_OLD_POSTED_DAYS,
    POSTING_PAY_WIDTH_RATIO,
    POSTING_QUALITY_CALL_TIMEOUT_SEC,
    POSTING_QUALITY_EFFORT,
    POSTING_QUALITY_MAX_JD_CHARS,
    POSTING_QUALITY_MIN_JD_CHARS,
    POSTING_QUALITY_MODEL,
    POSTING_QUALITY_TTL_DAYS,
    POSTING_RELIST_CAUTION,
)
from .jd_sanitizer import UNTRUSTED_JD_CLAUSE, clean_jd

logger = logging.getLogger(__name__)
_clock = time.monotonic     # run_quality_reads' wall clock (tests replace it)

READ_LOOKS_ACTIVE, READ_MIXED, READ_CHECK_FIRST = "looks_active", "mixed", "check_first"
# No posting date and no company-board check: there are no facts to weigh, so
# nothing disagrees — no feed chip (most LinkedIn / Built In / Glassdoor rows).
READ_NOT_ENOUGH = "not_enough_info"
READ_LABELS = {"looks_active": "Looks active", "mixed": "Mixed signals",
               "check_first": "Check before investing time",
               "not_enough_info": "Not enough to go on"}
# The hiring-health research cache key (perplexity_utils.get_hiring_health);
# read as db.get_company_intel(key[:200])["company_profile"].
HEALTH_KEY = "__hiring_health__{company}"

# Stages worth naming in "got as far as ..." (db_manager.INTERVIEW_STAGES).
_NAMED_STAGES = ("Screening Call", "Interview", "Final Interview", "Offer")

# Source sites shown in the "we can only check company job boards" fact.
_SITE_NAMES = {
    "linkedin": "LinkedIn", "indeed": "Indeed", "glassdoor": "Glassdoor",
    "zip_recruiter": "ZipRecruiter", "ziprecruiter": "ZipRecruiter",
    "google": "Google Jobs", "google_jobs": "Google Jobs", "builtin": "Built In",
    "reddit": "Reddit",
}
# Company sites we only reach through keyword searches — absence proves nothing.
_SEARCHED_BOARDS = ("workday", "apple", "teamtailor")


# ── small helpers ────────────────────────────────────────────────────────────

def _today() -> date:
    return date.today()


def _as_date(value) -> Optional[date]:
    """A date from 'YYYY-MM-DD…' text; a UTC timestamp ('YYYY-MM-DD HH:MM:SS',
    what db_manager._utc_now writes) is converted to the local day first."""
    s = str(value or "").strip()
    if len(s) < 10:
        return None
    if len(s) >= 19 and s[10] in " T":
        try:
            dt = datetime.strptime(s[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc).astimezone().date()
        except ValueError:
            pass
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _short_date(value, today: Optional[date] = None) -> str:
    """'Oct 3' (this year) / 'Oct 3, 2025' (another year); '' when unparseable."""
    d = _as_date(value)
    if d is None:
        return ""
    today = today or _today()
    text = f"{d.strftime('%b')} {d.day}"
    return text if d.year == today.year else f"{text}, {d.year}"


def posted_days(date_posted, today: Optional[date] = None) -> Optional[int]:
    """Days since the posting date (scrapes.date_posted[:10]); None when the
    date is missing or unparseable ("Yesterday", NaN). created_at is NOT a
    substitute — that is when we found it, not when it was posted."""
    s = str(date_posted or "").strip()
    if len(s) < 10:
        return None
    try:
        d = date.fromisoformat(s[:10])
    except ValueError:
        return None
    return max(0, ((today or _today()) - d).days)


def _times(n: int) -> str:
    return "once" if n == 1 else f"{n} times"


def _money(v: float) -> str:
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M".replace(".0M", "M")
    return f"${round(v / 1000)}K"


def _num(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f > 0 else None      # NaN != NaN


def _fact(key, text, source, *, url=None, quotes=None, caution=False) -> dict:
    return {"key": key, "text": text, "source": source, "url": url,
            "quotes": list(quotes or []), "caution": bool(caution)}


# ── free facts ───────────────────────────────────────────────────────────────

def pay_width_fact(comp_min, comp_max, comp_source) -> Optional[dict]:
    """The stated yearly pay range is unusually wide: both bounds present,
    taken from the posting itself (comp_source 'structured' or 'text' — an
    'llm' estimate is never judged), and (max - min) > RATIO * min."""
    if str(comp_source or "") not in ("structured", "text"):
        return None
    lo, hi = _num(comp_min), _num(comp_max)
    if lo is None or hi is None or hi <= lo:
        return None
    if hi - lo <= POSTING_PAY_WIDTH_RATIO * lo:
        return None
    return _fact("pay_width", f"Pay range is unusually wide: {_money(lo)}–{_money(hi)}",
                 "posting_text", caution=True)


_BOLD_LABEL_RE = re.compile(r"^\s*(?:[-•*]\s+)?\*\*(?P<label>[^*]{1,80}?)\s*:?\s*\*\*\s*:?\s*")
_PLAIN_LABEL_RE = re.compile(r"^\s*[-•]\s*(?P<label>[A-Za-z][A-Za-z &/,-]{1,60}?)\s*(?::|—|–)\s*")
_SOURCE_LINK_RE = re.compile(r"\(?\[source\]\([^)]*\)\)?")
_LEADING_DATE_RE = re.compile(r"^\s*20\d{2}-\d{1,2}(?:-\d{1,2})?\s*(?:—|–|-|:)?\s*")
_CUT_WORDS = ("layoff", "freeze", "restructur", "cuts")
# A research line under a cuts label counts only when its body says a cut
# HAPPENED (or is announced). "No layoffs reported", "no plans to cut the
# content team", "the company lifted its hiring freeze" are answers to the
# topic, not cuts.
_CUT_VERB_RE = re.compile(
    r"\b(?:laid off|lay(?:s|ing)? off|layoffs?|cut|cuts|cutting|eliminat\w*|froze|freez\w*"
    r"|paus\w* hiring|hiring pause|restructur\w*|downsiz\w*|furlough\w*|reduc\w*|let go"
    r"|shut\w* down|clos\w*)\b")
_NEGATION_RE = re.compile(
    r"\b(?:no|not|none|never|nothing|without|den(?:y|ies|ied)|unconfirmed|rumou?r\w*)\b|n't\b")
_ENDED_RE = re.compile(
    r"\b(?:lift\w*|ended|ending|ends|revers\w*|resum\w*|unfro\w*|rescind\w*|called off"
    r"|no longer)\b")


def _says_cut_happened(body: str) -> bool:
    """True when a research line's body reports a cut: no "freeze lifted /
    hiring resumed" wording anywhere, and no negation before the first cut
    word (or anywhere, when there is no cut word — "150 employees" counts,
    "None reported" does not)."""
    b = str(body or "").lower()
    if "no public signal" in b or _ENDED_RE.search(b):
        return False
    m = _CUT_VERB_RE.search(b)
    return not _NEGATION_RE.search(b[:m.start()] if m else b)


def _cut_topic(label: str, text: str) -> str:
    """Plain topic name for a research line already known to be about cuts."""
    label, text = label.lower(), text.lower()
    if "layoff" in label:
        return "layoffs"
    if "cuts" in label:
        return "creative team cuts" if "creative" in label else "job cuts"
    if "freeze" in label and "restructur" in label:
        # The query's combined topic "Hiring freezes or restructuring":
        # the line itself says which one happened.
        return "restructuring" if "restructur" in text and "freeze" not in text else "hiring freeze"
    return "hiring freeze" if "freeze" in label else "restructuring"


def recent_cut_facts(health_md: Optional[str], today: Optional[date] = None) -> list:
    """Layoffs / hiring freezes / restructuring / team cuts from the CACHED
    hiring-health research (already cite-or-omit filtered when it was saved),
    dated within POSTING_CUTS_WINDOW_DAYS. One dict per line:
    {topic, date: 'YYYY-MM-DD', text, url}. "No public signal found" lines,
    lines without a usable date, and lines saying nothing was cut or a cut
    ended (_says_cut_happened) are skipped. Never a network call."""
    if not health_md:
        return []
    from .perplexity_utils import _ISO_DATE_RE, _MONTHS, _NAMED_DATE_RE, _URL_RE

    today = today or _today()
    oldest = today - timedelta(days=POSTING_CUTS_WINDOW_DAYS)
    latest = today + timedelta(days=366)   # an announced, dated cut is still a fact
    out = []
    for line in str(health_md).splitlines():
        m = _BOLD_LABEL_RE.match(line) or _PLAIN_LABEL_RE.match(line)
        if not m:
            continue
        label = m.group("label").strip()
        if not any(w in label.lower() for w in _CUT_WORDS):
            continue
        if "no public signal found" in line.lower():
            continue
        urls = _URL_RE.findall(line)
        body = _SOURCE_LINK_RE.sub(" ", line[m.end():])
        body_no_urls = _URL_RE.sub(" ", body)
        candidates = [(int(y), int(mo), int(d or 1)) for y, mo, d in _ISO_DATE_RE.findall(body_no_urls)]
        candidates += [(int(y), _MONTHS[mon[:3].lower()], int(d or 1))
                       for mon, d, y in _NAMED_DATE_RE.findall(body_no_urls)]
        dates = []
        for y, mo, d in candidates:
            try:
                dates.append(date(y, mo, d))
            except ValueError:
                continue
        in_window = [d for d in dates if oldest <= d <= latest]
        if not in_window:
            continue
        text = re.sub(r"\s+", " ", _LEADING_DATE_RE.sub("", body_no_urls)).strip(" —–-:;,")
        if not _says_cut_happened(text):
            continue
        out.append({
            "topic": _cut_topic(label, text),
            "date": max(in_window).isoformat(),
            "text": text,
            "url": urls[0].rstrip(".,;") if urls else None,
        })
    return out


def overall_read(*, listing: str, posted_days: Optional[int], relisted: int,
                 recent_cuts: int, cut_facts: Optional[list] = None) -> tuple:
    """The overall read — a summary of the facts, never a score, never fed
    into ranking. THE RULE:

      Check first ("check_first") when ANY of:
        - listing == 'gone' (missing from two complete company-board fetches);
        - re-listed >= POSTING_RELIST_CAUTION times on the same board;
        - posted more than POSTING_OLD_POSTED_DAYS ago AND listing 'unknown'.
      Company news (layoffs / freezes) is shown as its own dated fact but
      never drives the read (cuts in unrelated divisions of a large employer
      would flag every one of its roles "check first" while it has thousands
      of open roles; the research cannot tell whether a cut hit the role's
      own team). `recent_cuts` and
      `cut_facts` are accepted and ignored.
      Looks active ("looks_active") when no Check-first trigger AND
        (listing == 'listed' OR posted <= POSTING_ACTIVE_POSTED_DAYS days ago).
      Not enough to go on ("not_enough_info") when no posting date and the
        listing is not 'listed' — there are no facts to disagree (no chip).
      Otherwise Mixed ("mixed").

    Returns (read, reasons): short plain phrases, one per trigger met.
    The posting-quality read never feeds this either."""
    listing = listing if listing in ("listed", "gone") else "unknown"
    relisted = int(relisted or 0)
    recent_cuts = int(recent_cuts or 0)

    reasons = []
    if listing == "gone":
        reasons.append("no longer on the company's job board")
    if relisted >= POSTING_RELIST_CAUTION:
        reasons.append(f"re-listed {_times(relisted)}")
    if posted_days is not None and posted_days > POSTING_OLD_POSTED_DAYS and listing == "unknown":
        reasons.append(f"posted {posted_days} days ago and we can't check the company board")
    if reasons:
        return READ_CHECK_FIRST, reasons

    fresh = posted_days is not None and posted_days <= POSTING_ACTIVE_POSTED_DAYS
    if listing == "listed" or fresh:
        if listing == "listed":
            reasons.append("still on the company's job board")
        if fresh:
            reasons.append(_posted_phrase(posted_days))
        return READ_LOOKS_ACTIVE, reasons

    if posted_days is None:
        return READ_NOT_ENOUGH, ["no posting date, and we can't check the company board"]
    reasons.append(_posted_phrase(posted_days))
    return READ_MIXED, reasons


def _posted_phrase(days: int) -> str:
    if days == 0:
        return "posted today"
    if days == 1:
        return "posted yesterday"
    return f"posted {days} days ago"


# ── inputs shared by build / feed / caution ──────────────────────────────────

def _cached_health(db, company: str) -> Optional[str]:
    company = str(company or "").strip()
    if not company or db is None:
        return None
    try:
        row = db.get_company_intel(HEALTH_KEY.format(company=company)[:200])
    except Exception as e:
        logger.debug(f"Posting check: health cache read failed for {company}: {e}")
        return None
    return (row or {}).get("company_profile") or None


def _scanner_title_stats(db, company: str, title: str) -> Optional[dict]:
    """perplexity_utils.get_scanner_stats(...)["title"], None when unavailable."""
    if not str(company or "").strip():
        return None
    try:
        from . import perplexity_utils
        stats = perplexity_utils.get_scanner_stats(company, db, title)
    except Exception as e:
        logger.debug(f"Posting check: scanner stats failed for {company}: {e}")
        return None
    return (stats or {}).get("title") or None


def _listing(db, job_id: int) -> dict:
    try:
        out = db.get_listing_status(job_id)
    except Exception as e:
        logger.debug(f"Posting check: listing status failed for {job_id}: {e}")
        out = None
    if not isinstance(out, dict) or out.get("status") not in ("listed", "gone", "unknown"):
        return {"status": "unknown", "reason": None}
    return out


def _read_for_job(db, job: dict, listing_status: str, today: date) -> tuple:
    """(read, reasons, inputs) for one job dict; no network, no LLM."""
    company = str(job.get("company") or "").strip()
    days = posted_days(job.get("date_posted"), today)
    tstats = _scanner_title_stats(db, company, str(job.get("title") or ""))
    relisted = int((tstats or {}).get("relisted") or 0)
    cuts = recent_cut_facts(_cached_health(db, company), today)
    read, reasons = overall_read(listing=listing_status, posted_days=days,
                                 relisted=relisted, recent_cuts=len(cuts), cut_facts=cuts)
    return read, reasons, {"posted_days": days, "title_stats": tstats,
                           "relisted": relisted, "cuts": cuts}


# ── the posting check ────────────────────────────────────────────────────────

def _listing_fact(listing: dict, job: dict, company: str, today: date) -> dict:
    status, reason = listing.get("status"), listing.get("reason")
    owner = f"{company}'s" if company else "the company's"
    checked = _short_date(listing.get("checked_at"), today)
    if status == "listed" and reason == "relisted":
        text = (f"Still on {owner} job board under a new link"
                + (f" (checked {checked})" if checked else ""))
        return _fact("listing", text, "company_board", url=listing.get("relisted_link"))
    if status == "listed":
        text = f"Still on {owner} job board" + (f" (checked {checked})" if checked else "")
        return _fact("listing", text, "company_board")
    if status == "gone":
        since = _short_date(listing.get("gone_since"), today)
        text = f"No longer on {owner} job board" + (f" since {since}" if since else "")
        return _fact("listing", text, "company_board", caution=True)
    if reason == "not_board":
        site = str(job.get("site") or "").strip().lower()
        if site in _SEARCHED_BOARDS:
            text = (f"Came from a keyword search of {owner} job board — that search can't "
                    "show whether the posting is still up")
        else:
            name = _SITE_NAMES.get(site) or (site.replace("_", " ").title() if site else "a job search site")
            text = f"Came from {name} — we can only check postings from company job boards"
        return _fact("listing", text, "our_scanner")
    if reason == "missing_once":
        text = ("Missing from the company job board once" + (f" ({checked})" if checked else "")
                + " — we wait for a second check")
        return _fact("listing", text, "company_board")
    if reason == "stale_check":
        text = "Company board last checked " + (checked or "a while ago")
        return _fact("listing", text, "company_board")
    return _fact("listing", "Not checked against the company job board yet", "company_board")


# Earlier applications shown one by one; the rest are counted in one line.
_MAX_APPLICATION_FACTS = 3


def _application_fact(app: dict, today: date) -> Optional[dict]:
    """One earlier card at the same company, named with THAT card's company.
    Only a To Apply card is "saved"; any later status means the candidate applied (the
    card history's dated Applied entry when there is one, else the date the
    card was added — most cards only have an undated snapshot)."""
    title = str(app.get("title") or "").strip() or "a role"
    status = str(app.get("status") or "").strip()
    company = str(app.get("company") or "").strip() or "this company"
    applied = _short_date(app.get("applied_at"), today)
    tracked = _short_date(app.get("tracked_at"), today)
    if applied:
        head = f"You applied to {company} before: {title} — applied {applied}"
    elif status in ("", "To Apply"):
        head = f"You saved a role at {company} before: {title}" + (f" — saved {tracked}" if tracked else "")
    else:
        head = f"You applied to {company} before: {title}" + (f" (added {tracked})" if tracked else "")
    parts = [head]
    if status:
        parts.append(f"now {status}")
    stage = app.get("highest_stage")
    if stage in _NAMED_STAGES and stage != status:
        parts.append(f"got as far as {stage}")
    updated = _as_date(app.get("updated_at"))
    if updated is not None:
        ago = max(0, (today - updated).days)
        parts.append("last update today" if ago == 0 else
                     "last update yesterday" if ago == 1 else f"last update {ago} days ago")
    return _fact("applied_before", ", ".join(parts), "your_history")


def _clip(text, n: int = 300) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _quality_facts(result: dict, comp_source=None) -> list:
    """The stored quality read as facts. Team / first months / pay are shown
    as the posting's own words (the verified quote), never the model's
    paraphrase. "The posting doesn't …" appears only when the model said so
    explicitly over the WHOLE posting and nothing for that point was dropped
    by cite-or-omit (result["absent"]) — and never for pay when the job
    already shows pay taken from the posting (comp_source structured/text)."""
    facts = []
    absent = set(result.get("absent") or ())
    if str(comp_source or "") in ("structured", "text"):
        absent.discard("pay")
    tools = result.get("tools") or []
    if tools:
        names = ", ".join(t["name"] for t in tools)
        facts.append(_fact("tools", f"Names specific tools: {names}", "posting_text",
                           quotes=[t["quote"] for t in tools]))
    elif "tools" in absent:
        facts.append(_fact("tools", "The posting names no specific tools", "posting_text"))
    for key, field, present, missing in (
        ("team", "team", "Team or reporting line: “{}”",
         "The posting doesn't name a team or who the role reports to"),
        ("scope", "first_year_scope", "First months' work: “{}”",
         "The posting doesn't describe the first months' work"),
        ("pay_stated", "pay", "Pay stated: “{}”",
         "The posting doesn't state pay"),
    ):
        item = result.get(field)
        if item:
            facts.append(_fact(key, present.format(_clip(item.get("quote"))), "posting_text"))
        elif field in absent:
            facts.append(_fact(key, missing, "posting_text"))
    for c in result.get("contradictions") or []:
        facts.append(_fact("contradiction", f"The posting disagrees with itself: {_clip(c.get('text'))}",
                           "posting_text", quotes=[c.get("quote"), c.get("quote2")], caution=True))
    level = result.get("boilerplate")
    if level == "low":
        facts.append(_fact("boilerplate", "The posting is mostly about this role itself",
                           "posting_text"))
    elif level == "medium":
        facts.append(_fact("boilerplate", "About a quarter to half of the posting is generic company text (benefits, legal, culture)",
                           "posting_text"))
    elif level == "high":
        facts.append(_fact("boilerplate", "Most of the posting is generic company text (benefits, legal, culture) — little detail about the role itself",
                           "posting_text", caution=True))
    return facts


def build_posting_check(db, job_id: int) -> Optional[dict]:
    """The Posting check for one job (also the API body). None when the job
    does not exist. No network, no LLM: listing status, posting date, scanner
    history, pay width, CACHED research cuts, the candidate's earlier applications, and
    the stored quality read when one exists."""
    try:
        job = db.get_job_details(job_id)
    except Exception as e:
        logger.warning(f"Posting check: job {job_id} read failed: {e}")
        return None
    if not job:
        return None
    today = _today()
    company = str(job.get("company") or "").strip()
    listing = _listing(db, job_id)
    read, reasons, inputs = _read_for_job(db, job, listing["status"], today)

    facts = [_listing_fact(listing, job, company, today)]
    days = inputs["posted_days"]
    if days is not None:
        facts.append(_fact("posted", _posted_phrase(days).capitalize(), "posting_text",
                           caution=days > POSTING_OLD_POSTED_DAYS))
    else:
        facts.append(_fact("posted", "Posting date not available", "our_scanner"))
    tstats = inputs["title_stats"] or {}
    first_seen = _short_date(tstats.get("first_seen") or job.get("created_at"), today)
    if first_seen:
        facts.append(_fact("first_seen", f"First seen by our scanner {first_seen}", "our_scanner"))
    relisted = inputs["relisted"]
    if relisted >= 1:
        facts.append(_fact("relisted", f"Re-listed {_times(relisted)} on the same board", "our_scanner",
                           caution=relisted >= POSTING_RELIST_CAUTION))
    pay = pay_width_fact(job.get("comp_yearly_min"), job.get("comp_yearly_max"), job.get("comp_source"))
    if pay:
        facts.append(pay)
    for cut in inputs["cuts"]:
        facts.append(_fact("cuts", f"{cut['topic'].capitalize()} reported {cut['date']}: {_clip(cut['text'])}",
                           "public_source", url=cut.get("url"), caution=True))
    try:
        apps = db.get_company_applications(company, exclude_job_id=job_id) if company else []
    except Exception as e:
        logger.debug(f"Posting check: company applications failed for {company}: {e}")
        apps = []
    apps = list(apps or [])
    for app in apps[:_MAX_APPLICATION_FACTS]:
        fact = _application_fact(app, today)
        if fact:
            facts.append(fact)
    if len(apps) > _MAX_APPLICATION_FACTS:
        more = len(apps) - _MAX_APPLICATION_FACTS
        facts.append(_fact("applied_before",
                           f"…and {more} more earlier {'card' if more == 1 else 'cards'} at {company}",
                           "your_history"))

    quality = None
    try:
        stored = db.get_posting_quality(job_id)
    except Exception as e:
        logger.debug(f"Posting check: quality read failed for {job_id}: {e}")
        stored = None
    if stored and isinstance(stored.get("result"), dict):
        facts.extend(_quality_facts(stored["result"], job.get("comp_source")))
        quality = {"checked_at": stored.get("checked_at"), "model": stored.get("model")}

    return {"job_id": job_id, "company": company, "read": read, "label": READ_LABELS[read],
            "reasons": reasons, "facts": facts, "quality": quality,
            "quality_available": quality is not None}


def feed_reads(db, jobs: list) -> dict:
    """{job_id: {"read", "label", "reasons"}} for Review feed rows. Cheap: the
    listing comes from each row's own `listing_status` column (never a
    per-row get_listing_status), the cached research is read once per
    company, and scanner stats once per (company, title) — re-listing is a
    per-title count. A row that fails is left out."""
    today = _today()
    health: dict = {}
    stats: dict = {}
    out = {}
    for row in jobs or []:
        try:
            job_id = row.get("id")
            if job_id is None:
                continue
            company = str(row.get("company") or "").strip()
            title = str(row.get("title") or "")
            ckey = company.lower()
            if ckey not in health:
                health[ckey] = recent_cut_facts(_cached_health(db, company), today)
            skey = (ckey, re.sub(r"\s+", " ", title).strip().lower())
            if skey not in stats:
                stats[skey] = _scanner_title_stats(db, company, title)
            relisted = int((stats[skey] or {}).get("relisted") or 0)
            cuts = health[ckey]
            read, reasons = overall_read(
                listing=row.get("listing_status") or "unknown",
                posted_days=posted_days(row.get("date_posted"), today),
                relisted=relisted, recent_cuts=len(cuts), cut_facts=cuts)
            out[int(job_id)] = {"read": read, "label": READ_LABELS[read], "reasons": reasons}
        except Exception as e:
            logger.debug(f"Posting check: feed read failed for row {row.get('id')}: {e}")
    return out


def caution_line(db, job_id: int) -> Optional[str]:
    """One short line for the email digest, only when the read is Check first
    ("Check before investing time: re-listed 3 times; …"). None otherwise or
    on any failure."""
    try:
        job = db.get_job_details(job_id)
        if not job:
            return None
        read, reasons, _ = _read_for_job(db, job, _listing(db, job_id)["status"], _today())
    except Exception as e:
        logger.debug(f"Posting check: caution line failed for {job_id}: {e}")
        return None
    if read != READ_CHECK_FIRST:
        return None
    return f"{READ_LABELS[READ_CHECK_FIRST]}: " + "; ".join(reasons)


# ── the email digest's one-line summary ──────────────────────────────────────
# The enrichment also shows in the email digest, readable at a glance
# One short line per card, the most telling facts first; the
# full list stays in the app's Posting check panel.
GLANCE_MAX_ITEMS = 6
_GLANCE_LABEL_MAX = 110


def _glance_item(text, caution=False, url=None) -> dict:
    return {"text": _clip(text, _GLANCE_LABEL_MAX), "caution": bool(caution), "url": url}


def _health_checked_on(db, company: str, today: date) -> Optional[str]:
    """The date the company's public-source research was cached, when it holds
    real research (not the 'did not complete' marker). None otherwise."""
    try:
        row = db.get_company_intel(HEALTH_KEY.format(company=company)[:200])
    except Exception:
        return None
    text = (row or {}).get("company_profile") or ""
    if not text.strip() or "did not complete" in text:
        return None
    return _short_date((row or {}).get("last_checked"), today) or "recently"


def _warm_path_counts(db, company: str) -> dict:
    """{owner: matches} from the warm-paths matcher (auto-accepted matches
    only). {} on any failure — the email never breaks over it."""
    try:
        from .warm_paths import find_warm_paths
        found = find_warm_paths(db, company) or {}
    except Exception as e:
        logger.debug(f"Posting check: warm paths failed for {company}: {e}")
        return {}
    counts: dict = {}
    for m in found.get("matches") or []:
        owner = str(m.get("owner") or "").strip().lower() or "network"
        counts[owner] = counts.get(owner, 0) + 1
    return counts


def _applied_before_short(apps: list, today: date) -> str:
    """'Applied here before: <title> (<date>) — <status>, reached <stage>'
    (+ '; 2 more'). The panel keeps the full wording."""
    app = apps[0]
    title = str(app.get("title") or "").strip() or "a role"
    status = str(app.get("status") or "").strip()
    when = _short_date(app.get("applied_at") or app.get("tracked_at"), today)
    verb = "Saved here before" if status in ("", "To Apply") and not app.get("applied_at") else "Applied here before"
    text = f"{verb}: {_clip(title, 45)}" + (f" ({when})" if when else "")
    tail = [status] if status and status != "To Apply" else []
    stage = app.get("highest_stage")
    if stage in _NAMED_STAGES and stage != status:
        tail.append(f"reached {stage}")
    if tail:
        text += " — " + ", ".join(tail)
    if len(apps) > 1:
        text += f"; {len(apps) - 1} more"
    return text


def digest_glance(db, job_id: int) -> Optional[dict]:
    """{"read", "label", "reasons", "items": [{"text", "caution", "url"}]} for
    one digest card, or None when the job is unknown. Built from
    build_posting_check plus the research date and warm paths. No network, no
    LLM. Items, in order: company news (cuts, or "none found" when research is
    cached), still on the company board, re-listing, wide pay, the posting
    contradicting itself / mostly boilerplate, the candidate's earlier card at the company,
    people in the imported contact network who work there — at most GLANCE_MAX_ITEMS."""
    check = build_posting_check(db, job_id)
    if not check:
        return None
    today = _today()
    company = check.get("company") or ""
    facts = check.get("facts") or []
    by_key: dict = {}
    for f in facts:
        by_key.setdefault(f.get("key"), []).append(f)

    items = []
    cuts = by_key.get("cuts") or []
    for f in cuts[:2]:
        items.append(_glance_item(f["text"].split(":", 1)[0], caution=True, url=f.get("url")))
    if not cuts and company:
        checked = _health_checked_on(db, company, today)
        if checked:
            items.append(_glance_item(f"No layoffs or hiring freeze found (checked {checked})"))

    listing = _listing(db, job_id)
    if listing.get("status") == "listed":
        if listing.get("reason") == "relisted":
            items.append(_glance_item("Re-posted on the company board under a new link",
                                      url=listing.get("relisted_link")))
        else:
            items.append(_glance_item(f"On {company or 'the company'}'s job board"))
    for f in by_key.get("relisted") or []:
        items.append(_glance_item(f["text"], caution=f.get("caution")))
    for f in by_key.get("pay_width") or []:
        items.append(_glance_item(f["text"], caution=True))
    for f in (by_key.get("contradiction") or [])[:1]:
        items.append(_glance_item(f["text"], caution=True))
    for f in by_key.get("boilerplate") or []:
        if f.get("caution"):
            items.append(_glance_item("Posting is mostly generic company text, little about the role", caution=True))
    if company and by_key.get("applied_before"):
        try:
            apps = db.get_company_applications(company, exclude_job_id=job_id) or []
        except Exception:
            apps = []
        if apps:
            items.append(_glance_item(_applied_before_short(apps, today)))

    counts = _warm_path_counts(db, company) if company else {}
    if counts:
        names = {"candidate": "your network"}   # any additional imported network shows its own label
        bits = [f"{names.get(k, k.capitalize())} {n}" for k, n in
                sorted(counts.items(), key=lambda kv: (kv[0] != "candidate", -kv[1]))]
        items.append(_glance_item("People you know here: " + ", ".join(bits)))

    return {"read": check["read"], "label": check["label"], "reasons": check.get("reasons") or [],
            "items": items[:GLANCE_MAX_ITEMS]}


# ── posting quality read (one cheap LLM call) ────────────────────────────────

_QUALITY_PROMPT = """You read one job posting and report FACTS about how specific it is. You do not judge the job, the company or the candidate, and you give no advice.

{UNTRUSTED_JD_CLAUSE}

Job title: {title}
Company: {company}

<posting>
{jd_clean}
</posting>

Report only what the posting itself says. Every fact needs a "quote": an exact, word-for-word span copied from the posting (or from the job title), 3 to 25 words, no ellipses, no paraphrase. If you cannot quote it, leave the fact out.

Return ONLY a JSON object, no prose, no code fences:
{"tools": [{"name": "...", "quote": "..."}],
 "team": {"text": "...", "quote": "..."} or null,
 "first_year_scope": {"text": "...", "quote": "..."} or null,
 "pay": {"text": "...", "quote": "..."} or null,
 "contradictions": [{"text": "...", "quote": "...", "quote2": "..."}],
 "boilerplate": "low" | "medium" | "high"}

Definitions:
- tools: named software, platforms or equipment (for example Premiere Pro, Figma, Asana). At most 10. Generic words ("editing software", "social platforms") are not tools.
- team: a named team or department, or who the role reports to ("reports to the Head of Video"). The company name alone is not a team.
- first_year_scope: concrete work the person will own or deliver in the first 6 to 12 months, in one sentence. Slogans ("drive impact") are not scope.
- pay: the pay as the posting states it.
- contradictions: the posting disagrees with itself, for example a coordinator, assistant or entry-level title with senior requirements (8+ years, managing a team), or years required that do not fit the scope described. Pay is never a contradiction: a wide or high range is normal and is reported separately. Only clear cases; quote one side in "quote" and the other in "quote2". Empty list when none.
- boilerplate: how much of the posting is generic text that could appear in any company's posting (culture slogans, benefits lists, legal equal-opportunity text) rather than this role: "low" under a quarter, "medium" a quarter to a half, "high" over half."""

_PLACEHOLDER_RE = re.compile(r"\{(UNTRUSTED_JD_CLAUSE|title|company|jd_clean)\}")


def quality_prompt(title: str, company: str, jd_clean: str) -> str:
    """The quality-read prompt. Placeholders are filled in ONE pass, so a
    title or posting that itself contains "{jd_clean}" is never re-expanded."""
    values = {"UNTRUSTED_JD_CLAUSE": UNTRUSTED_JD_CLAUSE, "title": str(title or ""),
              "company": str(company or ""), "jd_clean": str(jd_clean or "")}
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], _QUALITY_PROMPT)


_QUOTE_TRANS = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-",
})


def _norm(s) -> str:
    """Comparison form for cite-or-omit: NFKC, curly quotes straightened,
    en/em dashes as '-', whitespace collapsed, lower-case."""
    s = unicodedata.normalize("NFKC", str(s or "")).translate(_QUOTE_TRANS)
    return re.sub(r"\s+", " ", s).strip().lower()


_BOILERPLATE_LEVELS = ("low", "medium", "high")
_MAX_TOOLS = 10
_PAY_AMOUNT_RE = re.compile(r"[$€£]\s?\d|\d\s?(?:k|K)\b|\d[\d,]*\s?(?:per hour|/hr|/hour|an hour|annually|per year|/yr)")
_MIN_QUOTE_WORDS = 3      # team / scope / pay / contradiction quotes
_MIN_QUOTE_CHARS = 12


def parse_quality_response(text: str, jd_clean: str, title: str,
                           truncated: bool = False) -> Optional[dict]:
    """Parse the model's JSON and enforce cite-or-omit IN CODE: a quote counts
    only when 3 <= len <= 400 (normalized) and it is a substring of the
    normalized title + posting. Dropped: a tool whose quote fails or does not
    contain the tool's name (at most 10 kept); team / scope / pay whose quote
    fails or is shorter than 3 words / 12 characters (a pay quote must also
    carry a digit); a contradiction unless BOTH quotes count at that length,
    differ, and are not both from the title; a boilerplate level outside
    low/medium/high (-> None). Team / scope / pay keep the QUOTE as their
    text — the model's paraphrase is never shown.
    "absent" lists the points the model explicitly reported as missing
    (null / empty list) with nothing dropped for them — the only points the
    panel may call missing. Empty when `truncated` (the model did not see the
    whole posting). Returns None when the text is not a JSON object with any
    expected key — the caller treats that as "not checked", never as "no
    issues"."""
    if not isinstance(text, str) or not text.strip():
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    expected = ("tools", "team", "first_year_scope", "pay", "contradictions", "boilerplate")
    if not isinstance(data, dict) or not any(k in data for k in expected):
        return None

    haystack = _norm(f"{title or ''}\n{jd_clean or ''}")

    title_norm = _norm(title)

    def quote_ok(q) -> bool:
        if not isinstance(q, str):
            return False
        n = _norm(q)
        return 3 <= len(n) <= 400 and n in haystack

    def long_quote_ok(q) -> bool:
        return (quote_ok(q) and len(_norm(q)) >= _MIN_QUOTE_CHARS
                and len(_norm(q).split()) >= _MIN_QUOTE_WORDS)

    dropped = 0
    dropped_keys: set = set()
    tools, seen = [], set()
    raw_tools = data.get("tools")
    for t in raw_tools if isinstance(raw_tools, list) else []:
        name = t.get("name") if isinstance(t, dict) else None
        quote = t.get("quote") if isinstance(t, dict) else None
        if not (isinstance(name, str) and _norm(name) and quote_ok(quote)
                and _norm(name) in _norm(quote)):
            dropped += 1
            dropped_keys.add("tools")
            continue
        if _norm(name) in seen:
            continue
        if len(tools) >= _MAX_TOOLS:
            dropped += 1
            continue
        seen.add(_norm(name))
        tools.append({"name": _clip(name, 80), "quote": quote.strip()})

    def single(key: str, need_digit: bool = False):
        nonlocal dropped
        item = data.get(key)
        if item is None:
            return None
        quote = item.get("quote") if isinstance(item, dict) else None
        if not long_quote_ok(quote) or (need_digit and not re.search(r"\d", quote)):
            dropped += 1
            dropped_keys.add(key)
            return None
        return {"text": _clip(quote.strip()), "quote": quote.strip()}

    team = single("team")
    scope = single("first_year_scope")
    pay = single("pay", need_digit=True)

    contradictions = []
    raw_c = data.get("contradictions")
    for c in raw_c if isinstance(raw_c, list) else []:
        if not (isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip()
                and long_quote_ok(c.get("quote")) and long_quote_ok(c.get("quote2"))):
            dropped += 1
            continue
        q1, q2 = _norm(c["quote"]), _norm(c["quote2"])
        # Two sides of one disagreement: different spans, not one inside the
        # other, and not both from the title.
        if q1 in q2 or q2 in q1 or (q1 in title_norm and q2 in title_norm):
            dropped += 1
            continue
        # Pay is never one side of a contradiction: a wide or high range is
        # normal and pay_width_fact reports it (a model that still argues
        # "3-5 years but the range tops out senior" is giving an opinion).
        if _PAY_AMOUNT_RE.search(c["quote"]) or _PAY_AMOUNT_RE.search(c["quote2"]):
            dropped += 1
            continue
        contradictions.append({"text": _clip(c["text"]), "quote": c["quote"].strip(),
                               "quote2": c["quote2"].strip()})

    level = data.get("boilerplate")
    level = level.strip().lower() if isinstance(level, str) else None
    if level not in _BOILERPLATE_LEVELS:
        level = None

    absent = []
    if not truncated:
        if isinstance(raw_tools, list) and not tools and "tools" not in dropped_keys:
            absent.append("tools")
        for key in ("team", "first_year_scope", "pay"):
            if key in data and data[key] is None:
                absent.append(key)

    return {"tools": tools, "team": team, "first_year_scope": scope, "pay": pay,
            "contradictions": contradictions, "boilerplate": level, "dropped": dropped,
            "absent": absent, "truncated": bool(truncated)}


def _call_with_deadline(func, timeout_sec: float):
    """Run *func* in a daemon thread with a hard time budget (SIGALRM does not
    work in API worker threads). A timed-out thread is abandoned, never joined
    — the pipeline step and the API request move on."""
    box = {}

    def _target():
        try:
            box["value"] = func()
        except BaseException as e:  # noqa: BLE001 — surfaced into the caller
            box["error"] = e

    thread = threading.Thread(target=_target, daemon=True, name="posting-quality")
    thread.start()
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        raise TimeoutError(f"posting quality call exceeded {timeout_sec}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _within_days(stamp, days: int) -> bool:
    d = _as_date(stamp)
    return d is not None and (_today() - d).days <= days


def _quality_read(db, job_id: int) -> tuple:
    """(result, called_llm). The worker behind run_quality_read."""
    try:
        job = db.get_job_details(job_id)
    except Exception as e:
        logger.warning(f"Posting quality: job {job_id} read failed: {e}")
        return {"status": "failed", "quality": None}, False
    if not job:
        return {"status": "no_text", "quality": None}, False
    jd_full = clean_jd(job.get("description")).strip()
    jd = jd_full[:POSTING_QUALITY_MAX_JD_CHARS].strip()
    truncated = len(jd_full) > POSTING_QUALITY_MAX_JD_CHARS
    if len(jd) < POSTING_QUALITY_MIN_JD_CHARS:
        return {"status": "no_text", "quality": None}, False
    jd_hash = hashlib.sha256(jd.encode("utf-8")).hexdigest()
    try:
        stored = db.get_posting_quality(job_id)
    except Exception:
        stored = None
    if (stored and stored.get("jd_hash") == jd_hash and isinstance(stored.get("result"), dict)
            and _within_days(stored.get("checked_at"), POSTING_QUALITY_TTL_DAYS)):
        return {"status": "cached", "quality": stored["result"]}, False

    title = str(job.get("title") or "")
    prompt = quality_prompt(title, str(job.get("company") or ""), jd)
    try:
        from . import gpt_filter
        # Own deadline (the shared client retries a stalled call for up to
        # ~15 min) and no refusal fallback (it would escalate this cheap read
        # to the generation model at xhigh effort).
        reply = _call_with_deadline(
            lambda: gpt_filter.call_llm(prompt, model=POSTING_QUALITY_MODEL,
                                        effort=POSTING_QUALITY_EFFORT, _allow_fallback=False),
            POSTING_QUALITY_CALL_TIMEOUT_SEC)
    except Exception as e:
        logger.warning(f"Posting quality: model call failed for job {job_id}: {e}")
        return {"status": "failed", "quality": None}, True
    result = parse_quality_response(reply, jd, title, truncated=truncated)
    if result is None:
        logger.warning(f"Posting quality: unparseable reply for job {job_id} — nothing stored")
        return {"status": "failed", "quality": None}, True
    try:
        saved = db.save_posting_quality(job_id, jd_hash, POSTING_QUALITY_MODEL, result)
    except Exception as e:
        logger.warning(f"Posting quality: save failed for job {job_id}: {e}")
        saved = False
    if not saved:
        return {"status": "failed", "quality": None}, True
    return {"status": "fresh", "quality": result}, True


def run_quality_read(db, job_id: int) -> dict:
    """{"status": "fresh"|"cached"|"no_text"|"failed", "quality": dict|None}.
    "cached" = a read within POSTING_QUALITY_TTL_DAYS of the same cleaned
    posting exists (no call). A failed or unparseable call stores nothing —
    "not checked", never "no issues". Never raises."""
    try:
        return _quality_read(db, job_id)[0]
    except Exception as e:
        logger.warning(f"Posting quality: job {job_id} failed: {e}")
        return {"status": "failed", "quality": None}


def run_quality_reads(db, job_ids: list, *, cap: int, budget_sec: float,
                      should_stop=None) -> dict:
    """Quality reads for the given jobs, in order. Stops STARTING reads after
    `cap` model calls, `budget_sec` of wall clock, or once `should_stop()` is
    true (the Stop button); the rest count as skipped (so do jobs with too
    little posting text). Each call has its own deadline
    (POSTING_QUALITY_CALL_TIMEOUT_SEC), so the step ends within about
    budget_sec + that. A per-job exception counts as failed. Returns
    {"fresh", "cached", "skipped", "failed"}."""
    counts = {"fresh": 0, "cached": 0, "skipped": 0, "failed": 0}
    started = _clock()
    calls = 0
    seen = set()
    for job_id in job_ids or []:
        if job_id is None or job_id in seen:
            continue
        seen.add(job_id)
        if (calls >= cap or _clock() - started >= budget_sec
                or (should_stop is not None and should_stop())):
            counts["skipped"] += 1
            continue
        try:
            result, called = _quality_read(db, job_id)
        except Exception as e:
            logger.warning(f"Posting quality: job {job_id} failed: {e}")
            counts["failed"] += 1
            continue
        calls += int(called)
        status = result.get("status")
        counts["skipped" if status == "no_text" else status] += 1
    return counts
