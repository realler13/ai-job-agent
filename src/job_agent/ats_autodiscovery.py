"""
ATS Autodiscovery Module
========================

Given a company's careers URL, detect which Applicant Tracking System (ATS) they
use and extract the board token automatically.

Supported ATS families:
    greenhouse, lever, ashby, smartrecruiters, workable,
    recruitee, teamtailor, workday

Results are cached to `.cache/ats_map.json` so repeated runs skip the network.
"""

import json
import logging
import os
import re
import requests
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ATS pattern definitions
# ---------------------------------------------------------------------------

# Path-based ATS patterns: the token is the first meaningful path segment.
_PATH_ATS_PATTERNS: list[tuple[str, re.Pattern]] = [
    # boards.greenhouse.io/{token}  or  boards-api.greenhouse.io/v1/boards/{token}
    ("greenhouse", re.compile(
        r"^(?:boards(?:-api)?\.greenhouse\.io)$"
    )),
    ("lever", re.compile(r"^jobs\.lever\.co$")),
    ("ashby", re.compile(r"^jobs\.ashbyhq\.com$")),
    ("smartrecruiters", re.compile(r"^jobs\.smartrecruiters\.com$")),
    ("workable", re.compile(r"^apply\.workable\.com$")),
]

# Subdomain-based ATS patterns: the token is the subdomain.
_SUBDOMAIN_ATS_PATTERNS: list[tuple[str, str]] = [
    ("recruitee", "recruitee.com"),
    ("teamtailor", "teamtailor.com"),
]

# Workday is special: *.myworkdayjobs.com/* — return full base URL as "token".
_WORKDAY_RE = re.compile(r"\.myworkdayjobs\.com$", re.IGNORECASE)

# Cache location (relative to project root / cwd)
_CACHE_DIR = ".cache"
_CACHE_FILE = os.path.join(_CACHE_DIR, "ats_map.json")

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------------------------
# 1. Direct URL detection
# ---------------------------------------------------------------------------

def detect_ats_from_url(url: str) -> tuple[str, str] | None:
    """Detect ATS type and token from a known ATS URL pattern.

    Args:
        url: Any URL that might point to an ATS job board.

    Returns:
        ``(ats_name, token)`` or ``None`` if the URL doesn't match any
        recognised ATS pattern.  For workday, the "token" is the full
        base URL (scheme + host) since individual tokens are unreliable.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    host = (parsed.hostname or "").lower()
    if not host:
        return None

    # --- Workday (must check before generic patterns) ---
    if _WORKDAY_RE.search(host):
        base = f"{parsed.scheme or 'https'}://{host}"
        return ("workday", base)

    # --- Path-based ATS (greenhouse, lever, ashby, smartrecruiters, workable) ---
    for ats_name, host_re in _PATH_ATS_PATTERNS:
        if host_re.fullmatch(host):
            token = _extract_path_token(parsed.path, ats_name)
            if token:
                return (ats_name, token)

    # --- Subdomain-based ATS (recruitee, teamtailor) ---
    for ats_name, base_domain in _SUBDOMAIN_ATS_PATTERNS:
        if host.endswith(f".{base_domain}"):
            subdomain = host[: -(len(base_domain) + 1)]  # strip ".recruitee.com"
            if subdomain and subdomain not in ("www",):
                return (ats_name, subdomain)

    return None


def _extract_path_token(path: str, ats_name: str) -> str | None:
    """Pull the meaningful token out of the URL path.

    For ``boards-api.greenhouse.io/v1/boards/{token}/jobs`` the interesting
    segment is *after* ``/v1/boards/``.  For everything else it's simply the
    first non-empty path segment.
    """
    segments = [s for s in path.split("/") if s]

    if ats_name == "greenhouse":
        # boards-api form: /v1/boards/{token}/…
        if len(segments) >= 3 and segments[0] == "v1" and segments[1] == "boards":
            return segments[2]
        # boards.greenhouse.io/{token}/…
        if segments:
            return segments[0]
        return None

    # Generic: first path segment
    return segments[0] if segments else None


# ---------------------------------------------------------------------------
# 2. Discover ATS from a generic careers page
# ---------------------------------------------------------------------------

class _LinkExtractor(HTMLParser):
    """Lightweight HTML parser that collects href and src attributes."""

    def __init__(self):
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attr_dict = dict(attrs)
        if tag == "a" and attr_dict.get("href"):
            self.urls.append(attr_dict["href"])
        elif tag == "iframe" and attr_dict.get("src"):
            self.urls.append(attr_dict["src"])
        # Also pick up embedded script/link references that sometimes carry
        # ATS board URLs (e.g. <script src="https://boards.greenhouse.io/…">)
        elif tag == "script" and attr_dict.get("src"):
            self.urls.append(attr_dict["src"])


def discover_ats_from_careers_page(careers_url: str) -> tuple[str, str] | None:
    """Fetch a company's careers page and scan embedded links for ATS patterns.

    Args:
        careers_url: URL of the company's public careers/jobs page.

    Returns:
        ``(ats_name, token)`` or ``None`` if no recognisable ATS link is found.
    """
    # First, try the URL itself — maybe it *is* an ATS URL.
    direct = detect_ats_from_url(careers_url)
    if direct is not None:
        return direct

    # Fetch the page.
    try:
        resp = requests.get(careers_url, headers=_DEFAULT_HEADERS, timeout=10,
                            allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch careers page %s: %s", careers_url, exc)
        return None

    # Check if we were redirected to an ATS URL.
    redirect_match = detect_ats_from_url(resp.url)
    if redirect_match is not None:
        return redirect_match

    # Parse embedded links.
    html = resp.text
    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception as exc:
        logger.warning("HTML parse error for %s: %s", careers_url, exc)
        return None

    for link in parser.urls:
        match = detect_ats_from_url(link)
        if match is not None:
            return match

    # Last resort: regex scan for ATS URLs inside the full HTML (catches
    # URLs embedded in JavaScript variables, JSON-LD, etc.).
    for candidate in re.findall(r'https?://[^\s"\'<>]+', html):
        match = detect_ats_from_url(candidate)
        if match is not None:
            return match

    logger.info("No ATS detected on %s", careers_url)
    return None


# ---------------------------------------------------------------------------
# 3. Disk cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    """Load the ATS discovery cache from disk.  Returns ``{}`` on any error."""
    if not os.path.exists(_CACHE_FILE):
        return {}
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("cache root is not a dict")
            return data
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        logger.warning("ATS cache corrupted (%s), starting fresh: %s", _CACHE_FILE, exc)
        return {}


def _save_cache(cache: dict) -> None:
    """Persist the cache dict to disk, creating the directory if needed."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.warning("Failed to write ATS cache: %s", exc)


# ---------------------------------------------------------------------------
# 4. Company-level resolver
# ---------------------------------------------------------------------------

def resolve_company_ats(entry: dict, cache: dict | None = None) -> dict:
    """Resolve a single company config entry, filling in ``ats`` and ``token``
    via autodiscovery if they are missing.

    Args:
        entry: Dict that *may* contain ``ats``, ``token``, and/or
               ``careers_url``.  If ``ats`` *and* ``token`` are already
               present, the entry is returned as-is.
        cache: Optional shared cache dict.  When provided the caller is
               responsible for loading/saving; otherwise we load/save per
               call.

    Returns:
        A *new* dict (never mutates the input) with ``ats`` and ``token``
        filled in when possible.
    """
    result = dict(entry)  # shallow copy

    # Already fully specified — nothing to do.
    if result.get("ats") and result.get("token"):
        return result

    careers_url = result.get("careers_url")
    if not careers_url:
        return result

    # Check cache.
    own_cache = cache is None
    if own_cache:
        cache = _load_cache()

    cached = cache.get(careers_url)
    if cached and cached.get("ats") and cached.get("token"):
        result["ats"] = cached["ats"]
        result["token"] = cached["token"]
        return result

    # Run autodiscovery.
    discovery = discover_ats_from_careers_page(careers_url)
    if discovery is not None:
        ats_name, token = discovery
        result["ats"] = ats_name
        result["token"] = token

        cache[careers_url] = {
            "ats": ats_name,
            "token": token,
            "discovered_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        if own_cache:
            _save_cache(cache)
    else:
        logger.info(
            "Could not autodiscover ATS for %s (%s)",
            result.get("company", "unknown"),
            careers_url,
        )

    return result


# ---------------------------------------------------------------------------
# 5. Batch resolver
# ---------------------------------------------------------------------------

def resolve_all_companies(companies: list[dict]) -> list[dict]:
    """Resolve ATS info for a list of company config entries.

    Uses the disk cache so only truly new URLs hit the network.

    Args:
        companies: List of company config dicts.

    Returns:
        New list with ``ats``/``token`` filled in where possible.
    """
    cache = _load_cache()
    resolved = []
    new_discoveries = 0

    for entry in companies:
        before_size = len(cache)
        resolved_entry = resolve_company_ats(entry, cache=cache)
        if len(cache) > before_size:
            new_discoveries += 1
        resolved.append(resolved_entry)

    if new_discoveries > 0:
        _save_cache(cache)
        logger.info("Autodiscovered ATS for %d new companies", new_discoveries)

    return resolved


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- URL detection tests ---
    assert detect_ats_from_url("https://boards.greenhouse.io/airbnb") == ("greenhouse", "airbnb")
    assert detect_ats_from_url("https://boards-api.greenhouse.io/v1/boards/stripe/jobs") == ("greenhouse", "stripe")
    assert detect_ats_from_url("https://jobs.lever.co/spotify") == ("lever", "spotify")
    assert detect_ats_from_url("https://jobs.ashbyhq.com/ramp") == ("ashby", "ramp")
    assert detect_ats_from_url("https://jobs.smartrecruiters.com/some-company") == ("smartrecruiters", "some-company")
    assert detect_ats_from_url("https://apply.workable.com/traveloka/") == ("workable", "traveloka")
    assert detect_ats_from_url("https://fly-io.recruitee.com/") == ("recruitee", "fly-io")
    assert detect_ats_from_url("https://northvolt.teamtailor.com/jobs") == ("teamtailor", "northvolt")

    wd = detect_ats_from_url("https://disney.wd5.myworkdayjobs.com/disneycareer")
    assert wd is not None and wd[0] == "workday"
    assert wd[1] == "https://disney.wd5.myworkdayjobs.com"

    assert detect_ats_from_url("https://example.com/careers") is None
    assert detect_ats_from_url("https://www.google.com") is None
    assert detect_ats_from_url("not-a-url") is None

    # --- Edge cases ---
    # Trailing slashes, extra path segments
    assert detect_ats_from_url("https://boards.greenhouse.io/airbnb/jobs/123") == ("greenhouse", "airbnb")
    assert detect_ats_from_url("https://jobs.lever.co/spotify/some-job-id") == ("lever", "spotify")
    # Bare host with no token should return None (no token to extract)
    assert detect_ats_from_url("https://boards.greenhouse.io/") is None
    assert detect_ats_from_url("https://boards.greenhouse.io") is None

    print("All tests passed!")
