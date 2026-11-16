"""
Company-name derivation for jobs where the scraper left it blank.

JobSpy's Indeed scraper returns NaN for `company` when the posting is an ATS
passthrough (e.g. Indeed listing → Ashby/Greenhouse/Workday). The EntertainmentCareers
RSS scraper also drops company entirely — but encodes it in the title suffix.

This module recovers the company from two signals:
  1. `job_url_direct` — the canonical ATS URL, which embeds the company slug
  2. The title itself — EntertainmentCareers pattern "Role - Company"

display_company() is the separate, display-only view used in employer-facing
documents (letters, file names): see the section at the end of this module.

Intentional non-goals:
  - We do not follow HTTP redirects (grnh.se shortlinks stay unresolved).
  - We do not fix truncated titles from the source feed.
  - When no signal is usable, we return None — the caller decides how to handle it.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# --- Public API ---------------------------------------------------------------

def derive_company(job_data: dict) -> Optional[str]:
    """Return a best-effort company name for a scraped job.

    Priority:
      1. `company` already set in the job dict (returned cleaned).
      2. ATS slug from `job_url_direct`.
      3. Title suffix " - CompanyName" (EntertainmentCareers pattern).
      4. None.
    """
    existing = _clean_existing_company(job_data.get("company"))
    if existing:
        return existing

    direct = job_data.get("job_url_direct") or ""
    if direct:
        from_url = _company_from_ats_url(direct)
        if from_url:
            return from_url

    title = job_data.get("title") or ""
    from_title = _company_from_title_suffix(title)
    if from_title:
        return from_title

    return None


# --- Cleanup ------------------------------------------------------------------

def _clean_existing_company(val) -> Optional[str]:
    """Return a non-empty trimmed string, or None for NaN/None/empty/'nan'/'none'."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    return s


# --- ATS URL patterns ---------------------------------------------------------
# (compiled_regex, extraction_kind)
#   extraction_kind = "path"      → company is group 1 (first path segment)
#   extraction_kind = "subdomain" → company is group 1 (subdomain)

_ATS_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Subdomain-as-tenant
    (re.compile(r"https?://([a-z0-9][a-z0-9\-]*)\.myworkdayjobs\.com", re.I), "subdomain"),
    (re.compile(r"https?://([a-z0-9][a-z0-9\-]*)\.wd1\.myworkdayjobs\.com", re.I), "subdomain"),
    (re.compile(r"https?://([a-z0-9][a-z0-9\-]*)\.recruitee\.com", re.I), "subdomain"),
    (re.compile(r"https?://([a-z0-9][a-z0-9\-]*)\.bamboohr\.com", re.I), "subdomain"),
    (re.compile(r"https?://([a-z0-9][a-z0-9\-]*)\.teamtailor\.com", re.I), "subdomain"),

    # Path-as-tenant
    (re.compile(r"https?://jobs\.ashbyhq\.com/([^/?#]+)", re.I), "path"),
    (re.compile(r"https?://boards\.greenhouse\.io/([^/?#]+)", re.I), "path"),
    (re.compile(r"https?://job-boards\.greenhouse\.io/([^/?#]+)", re.I), "path"),
    (re.compile(r"https?://jobs\.lever\.co/([^/?#]+)", re.I), "path"),
    (re.compile(r"https?://apply\.workable\.com/([^/?#]+)", re.I), "path"),
    (re.compile(r"https?://careers\.smartrecruiters\.com/([^/?#]+)", re.I), "path"),
]

_GENERIC_SUBDOMAINS = frozenset({"jobs", "careers", "apply", "www", "app", "api"})


def _company_from_ats_url(url: str) -> Optional[str]:
    """Extract and prettify the company slug from a known ATS URL."""
    if not url:
        return None
    url = url.strip()
    for pattern, _kind in _ATS_PATTERNS:
        m = pattern.search(url)
        if not m:
            continue
        slug = m.group(1)
        if slug.lower() in _GENERIC_SUBDOMAINS:
            continue
        return _prettify_slug(slug)
    return None


def _prettify_slug(slug: str) -> str:
    """Turn 'acme3d-new-media' → 'acme3d New Media'. Preserves brand words containing digits
    (3m, 23andme), title-cases the rest."""
    words = slug.replace("-", " ").replace("_", " ").split()
    out = []
    for w in words:
        has_digit = any(c.isdigit() for c in w)
        has_alpha = any(c.isalpha() for c in w)
        if has_digit and has_alpha:
            out.append(w)  # brand with digits — keep original case
        else:
            out.append(w.capitalize())
    return " ".join(out)


# --- Title-suffix pattern -----------------------------------------------------
# EntertainmentCareers.net RSS titles look like:
#   "Podcast Producer - Example Media Group"
#   "Creative Producer - Example Studios"
# But sometimes the suffix isn't a company:
#   "YouTube Channel Lead - Hybrid"    (work mode)
#   "Senior Editor - $140k"            (salary)
#   "Producer - Full Time"             (employment type)

_NON_COMPANY_SUFFIXES = frozenset({
    # Work modes
    "hybrid", "remote", "on-site", "onsite", "in-office", "flexible", "wfh",
    # Employment types
    "full time", "part time", "full-time", "part-time", "contract", "freelance",
    "temp", "temporary", "permanent", "internship", "intern",
    # Common location shorthands
    "la", "nyc", "ny", "sf", "usa", "us",
})


def _company_from_title_suffix(title: str) -> Optional[str]:
    """Extract trailing ' - CompanyName' from titles like 'Podcast Producer - Example Media Group'.
    Returns None when the suffix is a work mode, salary, or location shorthand."""
    if not title or " - " not in title:
        return None
    candidate = title.rsplit(" - ", 1)[1].strip()
    if not candidate:
        return None
    # Filter out known non-company suffixes
    if candidate.lower() in _NON_COMPANY_SUFFIXES:
        return None
    # Filter out obvious salary/location patterns
    if candidate.startswith("$") or candidate.lower().startswith("up to "):
        return None
    # Too short to be meaningful (1-2 chars)
    if len(candidate) < 2:
        return None
    return candidate


# --- Display name (Phase 27) --------------------------------------------------
# What an EMPLOYER reads: the letter's recipient block, its RE: line, the
# company named in the letter prompt, and the CV/CL file names. The stored
# string is decorated by scrapers and boards ("Acme (via StaffCo)",
# "Acme Holdings | ACME", "Northwind, Inc.") and is sometimes misspelled.
# The DB value is never changed — this is a view.
#
# Conservative on purpose:
#   - qualifiers are a CLOSED list (same spirit as warm_paths): "(via X)",
#     "(Contractor)", "(staffing)", "(formerly X)". Other parentheticals are
#     part of the name as written ("Acme Motors (AM)", "Northwind
#     (Northwind Labs)");
#   - a bare " via X" tail is kept ("Jobs via Dice" would become "Jobs");
#   - legal suffixes: Inc / LLC / Ltd always, "Corp." only with its period or
#     after a comma ("Globex Corp" is a name, "Initech Corp." is a suffix);
#     "Co." and "Corporation" are never stripped ("Smith & Co.");
#   - a name is CORRECTED only when a job link's host is in the closed map
#     below (the employer's own job site), never from a generic domain slug
#     (that would turn "Example.Labs" into "Examplelabs").

_DISPLAY_QUALIFIER_RE = re.compile(
    r"\s*\(\s*(?:"
    r"via\b[^()]*"                                                  # (via StaffCo)
    r"|contract(?:ors?)?|contingent(?:\s+workforce)?|staffing(?:\s+agency)?"  # (Contractor)
    r"|(?:formerly|fka|f/k/a)\s[^()]*"                              # (formerly X)
    r")\s*\)\s*$", re.IGNORECASE)
_DISPLAY_TAGLINE_RE = re.compile(r"\s+\|.*$", re.DOTALL)            # "Name | tagline"
_LEGAL_SUFFIX_RE = re.compile(
    r"(?:,\s*|\s+)(?:inc\.?|llc\.?|l\.l\.c\.|ltd\.?)$"               # Inc / Inc. / LLC / Ltd.
    r"|(?:,\s*corp\.?|\s+corp\.)$",                                 # "X Corp." / "X, Corp"
    re.IGNORECASE)

# Job-link hosts that identify the employer beyond doubt (its own job site).
_EMPLOYER_BY_LINK_HOST = {
    "jobs.apple.com": "Apple",
    "applecontingentworkforce.willhire.co": "Apple",   # Apple's contingent marketplace
}
_LINK_FIELDS = ("link", "job_url", "job_url_direct", "url")


def _link_host(url) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    url = url.strip()
    try:
        host = urlparse(url if "://" in url else "https://" + url).hostname or ""
    except ValueError:
        return ""
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def _employer_from_links(job: dict) -> tuple:
    """(employer, host) when one of the job's links is on a host in
    _EMPLOYER_BY_LINK_HOST, else ("", "")."""
    for field in _LINK_FIELDS:
        host = _link_host((job or {}).get(field))
        if host in _EMPLOYER_BY_LINK_HOST:
            return _EMPLOYER_BY_LINK_HOST[host], host
    return "", ""


def _strip_display_decorations(name: str) -> str:
    """Drop trailing qualifiers and taglines (repeatedly), then one trailing
    legal suffix. Never strips a name down to nothing."""
    s = name.strip()
    for _ in range(4):  # "Apple, Inc. (via X) | tagline" nests 2-3 deep at most
        prev = s
        for rx in (_DISPLAY_TAGLINE_RE, _DISPLAY_QUALIFIER_RE):
            cut = rx.sub("", s).strip()
            if cut:
                s = cut
        if s == prev:
            break
    cut = _LEGAL_SUFFIX_RE.sub("", s).strip().rstrip(",").strip()
    if len(cut) >= 2 and re.search(r"\w", cut) and not cut.endswith(("&", "+")):
        s = cut
    return s


def display_company(job: dict) -> str:
    """The employer's name as a letter should print it ("" when unknown).

    Strips the decorations above from job["company"]; corrects the name only
    when a job link is on the employer's own job site (jobs.apple.com, Apple's
    WillHire marketplace -> "Apple"). Every change is logged. The job dict and
    the stored value are not modified."""
    original = _clean_existing_company((job or {}).get("company")) or ""
    shown = _strip_display_decorations(original) if original else ""
    employer, host = _employer_from_links(job)
    reason = ""
    if employer and shown.lower() != employer.lower():
        reason = f" (link domain {host} identifies {employer})"
        shown = employer
    if shown != original:
        logger.info(f"display_company: {original!r} -> {shown!r}{reason}")
    return shown
