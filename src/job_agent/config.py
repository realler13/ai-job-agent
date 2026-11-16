
import os
import pandas as pd
import re

from .candidate_profile import (  # noqa: E402
    PROFILE, CANDIDATE, SEARCH, RULES, fill, metrics_text,
)

# ─────────────────────────────────────────────────────────────────────────────
# Everything below that describes a PERSON comes from the candidate profile
# (profiles/*.yaml). Everything that describes the JOB MARKET — ATS boards,
# scraper pacing, model routing — is impersonal and lives here as code.
# ─────────────────────────────────────────────────────────────────────────────

# === Keywords & Exclusions ===
# Structured by category for readability; flattened into KEYWORDS for use everywhere.
KEYWORDS_BY_CATEGORY = {
    category: list(terms)
    for category, terms in SEARCH["keywords_by_category"].items()
}

# Flat list used by main.py, gpt_filter.py, ats_scraper.py
KEYWORDS = [kw for group in KEYWORDS_BY_CATEGORY.values() for kw in group]

# === Domain-Based CV Variant System ===
# Maps keyword category → domain-specific base CV file.
DOMAIN_CV_MAP = {
    domain: spec["cv"] for domain, spec in PROFILE["domains"].items()
}

# Human-readable domain descriptions — injected into CV tailoring prompts
# so the LLM understands the framing of the chosen base CV variant.
DOMAIN_DESCRIPTIONS = {
    domain: spec["description"] for domain, spec in PROFILE["domains"].items()
}

# Reverse lookup: keyword string → category name.
# Built once at import time for O(1) lookups.
KEYWORD_TO_CATEGORY = {
    kw.lower(): cat
    for cat, keywords in KEYWORDS_BY_CATEGORY.items()
    for kw in keywords
}


def resolve_cv_domain(job_title: str = "", matched_keyword: str = None) -> str:
    """Determine the best CV domain for a job.

    Resolution order:
    1. If *matched_keyword* maps to a known category, use it.
    2. Fuzzy-match *job_title* against all keywords (case-insensitive substring).
    3. Fall back to ``"producer_core"`` (the broadest variant).

    Returns a key from DOMAIN_CV_MAP (e.g. ``"games_production"``).
    """
    # 1. Direct keyword lookup
    if matched_keyword:
        cat = KEYWORD_TO_CATEGORY.get(matched_keyword.lower())
        if cat:
            return cat

    # 2. Title-based fuzzy match — check longest keywords first so
    #    "Senior Podcast Producer" matches "Podcast Producer" category
    #    before the generic "Producer" in producer_core.
    if job_title:
        title_lower = job_title.lower()
        best_match = None
        best_len = 0
        for kw, cat in KEYWORD_TO_CATEGORY.items():
            if kw in title_lower and len(kw) > best_len:
                best_match = cat
                best_len = len(kw)
        if best_match:
            return best_match

    # 3. Default fallback
    return "producer_core"

# How fresh a role must be to show in the Review list and the email: a
# posting more than a week old is usually close to filled. Age = the posting date, or the day we first saw the
# job when it has none (utils.role_age_days). Older roles are hidden, never
# deleted. A tier not listed here is not limited.
REVIEW_MAX_AGE_DAYS = {"strong": 7, "worth_a_look": 3}

# Postings older than this many days are dropped in the pre-filter — applying
# to a weeks-old posting is usually applying to a filled role. Blank dates pass.
# The value comes from the profile (search.stale_posting_days). The shipped
# example uses 7: nothing older can ever be shown
# (see REVIEW_MAX_AGE_DAYS), so grading it only costs money.
STALE_POSTING_DAYS = SEARCH["stale_posting_days"]

EXCLUDE_PHRASES = list(SEARCH["exclude_phrases"])

# Applied to "Production Manager" / "Production Operations" / "Project Manager" /
# "Program Manager" matches to filter out manufacturing/industrial/tech roles.
# Split by check strictness (raw substring matching was killing good roles:
# "civil" in legal boilerplate, "aws" inside "laws"). All matching is
# now word-boundary based in utils.process_and_filter_jobs.
#
# INDUSTRIAL terms: one hit in title OR description kills the job.
NEGATIVE_KEYWORDS_INDUSTRIAL = list(SEARCH["negative_keywords_industrial"])
# TECH-STACK terms: a hit in the TITLE kills; in the description it takes 2+
# DISTINCT terms (tech-company JDs routinely mention one tool in passing —
# that alone must not kill a real production role).
NEGATIVE_KEYWORDS_TECH = list(SEARCH["negative_keywords_tech"])
# Back-compat combined list (some callers/tests import the old name).
NEGATIVE_KEYWORDS = NEGATIVE_KEYWORDS_INDUSTRIAL + NEGATIVE_KEYWORDS_TECH

ALLOWED_LOCATIONS = list(SEARCH["locations"]["allowed"])

# === Seniority Policy (fully judge-arbitrated) ===
# The target seniority band is described in the candidate profile
# (rules.candidate_summary / rules.target_roles / rules.seniority_rule) and
# spelled out in the cascade prompts.
#
# There is deliberately NO deterministic title regex. Seniority is decided by
# the cascade judge, which reads the DESCRIPTION rather than the title and
# applies a two-bucket rule:
#   - confirmed people-management (direct reports / hiring / headcount) or an
#     org-level executive title (VP/SVP/Chief/Head-of)  -> REJECT
#   - an over-leveled / ambiguous leadership TITLE with no report evidence
#     (an inflated title on a hands-on job)             -> lower tier
# A title-only regex cannot tell a hands-on "Video Production Manager" from a
# people-leading "Manager, Music Production"; only the description can.
# (History: a regex backstop lived here until it kept springing leaks — e.g.
# plural "Directors" slipping a \bdirector\b boundary — and was removed in
# favour of the agentic test.)


# === Location Priority ===
# The home location and both metro term lists come from the profile
# (search.locations). Primary-metro roles are fully actionable; remote works;
# on-site roles in the secondary metro are treated as a long commute — they
# are still surfaced, but capped at the worth_a_look tier, excluded from
# premium Stage-3 judging and from automatic CV generation: visible, but they
# don't consume the reviewer's attention or the model budget.
# The class names used below ("bay" = primary metro, "la" = secondary metro,
# "ca" = rest of the home state) and the California-specific helpers are
# functional identifiers kept from the example market the tool was first
# tuned for.
PRIMARY_METRO_TERMS = list(SEARCH["locations"]["primary_metro_terms"])
# Example-market helper: Bay Area cities whose names also exist elsewhere
# ("Dublin, Ireland", "Newark, NJ", "Brisbane, Australia") count as Bay Area
# only when the location also says California.
BAY_AREA_CA_TERMS = [
    "dublin", "newark", "union city", "alameda", "belmont", "brisbane",
    "danville", "saratoga",
]
SECONDARY_METRO_TERMS = list(SEARCH["locations"]["secondary_metro_terms"])
# Example-market helper: LA County places whose names also exist elsewhere
# count as LA only with California.
LA_METRO_CA_TERMS = ["hawthorne", "carson", "downey", "gardena"]
# Remote is decided from the LOCATION field only — NOT the title. "Supervisor,
# Remote Operations" in Bristol CT is an on-site broadcast job, not WFH. Bare
# country labels ("United States"/"USA"/"nationwide") are NOT remote either — a
# US-wide posting is often on-site-TBD — so they route to the judge as unknown.
# (Remote detection is inlined in classify_location below.)

# Non-CA US states (CA excluded on purpose). Abbreviations are matched only as a
# comma-delimited segment ("Jay, OK") to avoid colliding with words like "in" /
# "or"; full state names are matched as substrings (low collision risk).
_US_STATE_ABBR_NON_CA = {
    "al", "ak", "az", "ar", "co", "ct", "de", "fl", "ga", "hi", "id", "il",
    "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo",
    "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or",
    "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi",
    "wy", "dc",
}
_US_STATE_NAMES_NON_CA = {
    "alabama", "alaska", "arizona", "arkansas", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana",
    "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
    "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west virginia", "wisconsin", "wyoming",
}
# Non-US regions/cities — a "remote" role restricted to one of these is NOT
# actionable (the candidate needs US-based remote). High-precision, kept short.
_FOREIGN_MARKERS = {
    "canada", "toronto", "vancouver", "united kingdom", "london", "emea",
    "apac", "latam", "europe", "european", "india", "bangalore", "bengaluru",
    "germany", "berlin", "france", "spain", "brazil", "mexico city",
    "australia", "singapore", "ireland", "dublin", "netherlands", "poland",
    "philippines", "manila", "argentina",
}


def _has_non_ca_us_state(loc: str, parts: list) -> bool:
    """True if the location names a concrete non-CA US state (name or abbrev)."""
    if any(name in loc for name in _US_STATE_NAMES_NON_CA):
        return True
    return any(p in _US_STATE_ABBR_NON_CA for p in parts)


# Location feedback loop guard: a single "wrong location"
# dismissal must not block a whole area. A location string is blocked only
# when it names a specific place outside the candidate's area (these
# classify_location classes — never 'bay', 'remote' or 'unknown': a bare
# "United States" or "Hybrid" label would block every nationwide and every
# hybrid posting) AND it was dismissed for location at least this many times.
# Never "la" either: secondary-metro on-site roles are a lower priority but
# must stay visible — they are already capped to Worth a look, never hidden.
LOCATION_SUPPRESSIBLE_CLASSES = ("ca", "reject")
# Votes are counted per DAY: a batch dismissal, or the same job seen on two
# sites and dismissed together, is still saying it once.
LOCATION_SUPPRESS_MIN_VOTES = 2
# A bare state / country label is never a place to block ("California",
# "CA, US", "United States") — it would drop every statewide posting.
_REGION_ONLY_LABELS = {"ca", "california", "us", "usa", "u.s.", "united states",
                       "united states of america"}


def location_is_suppressible(location: str) -> bool:
    """True when a location string may be blocked by the feedback loop: a
    specific place outside the candidate's area (LOCATION_SUPPRESSIBLE_CLASSES), not a bare
    state / country label."""
    loc = str(location or "").strip().lower()
    segments = [seg.strip() for seg in loc.split(",") if seg.strip()]
    if not segments or all(seg in _REGION_ONLY_LABELS for seg in segments):
        return False
    return classify_location(loc) in LOCATION_SUPPRESSIBLE_CLASSES


def classify_location(location: str = "", title: str = "") -> str:
    """Classify a job location relative to the profile's home area
    (search.locations). Returns one of:

    - ``bay``     the profile's primary metro — top priority
    - ``remote``  remote, US or unspecified region — actionable
    - ``la``      the profile's secondary metro, on-site — capped, NOT rejected
    - ``ca``      other California on-site — actionable-ish, passes to the judge
    - ``unknown`` empty, or a bare national label — no signal, passes to judge
    - ``reject``  concrete non-CA place, or remote restricted to a non-US region

    Remote is read from the LOCATION only, never the title.
    """
    import re as _re
    loc = str(location or "").lower().strip()
    if not loc or loc == "none":
        return "unknown"
    tokens = set(_re.split(r"[^a-z.]+", loc))
    parts = [p.strip() for p in loc.split(",")]

    def _hit(terms):
        return any((t in loc) if len(t) > 3 else (t in tokens) for t in terms)

    # The primary metro wins outright (on-site near home is the top preference).
    if _hit(PRIMARY_METRO_TERMS):
        return "bay"
    if ("ca" in tokens or "california" in loc) and any(t in loc for t in BAY_AREA_CA_TERMS):
        return "bay"

    remote = ("remote" in tokens) or ("work from home" in loc) or ("anywhere" in tokens)
    foreign = any(f in loc for f in _FOREIGN_MARKERS)
    non_ca_state = _has_non_ca_us_state(loc, parts)
    us_signal = (any(lbl in loc for lbl in ("united states", "usa", "u.s."))
                 or "us" in tokens or non_ca_state)

    if remote:
        # Remote is fine UNLESS pinned to a non-US region with no US signal
        # ("Remote - India", "Remote, EMEA").
        if foreign and not us_signal:
            return "reject"
        return "remote"

    # On-site from here down. The secondary metro is capped elsewhere, never rejected.
    # Bare "la" == Los Angeles; "Lake Charles, LA" (Louisiana) is caught below.
    if _hit(SECONDARY_METRO_TERMS) or loc == "la":
        return "la"
    if ("ca" in tokens or "california" in loc) and any(t in loc for t in LA_METRO_CA_TERMS):
        return "la"

    # A California option ANYWHERE makes the role actionable, even in a multi-
    # location posting ("Austin, TX; Novato, California; Seattle, WA") — so check
    # CA before rejecting on a co-listed non-CA state. "california" (unambiguous)
    # is checked before a foreign locale; a bare "ca" token (could be Canada) is
    # checked after, so "Vancouver, CA" still rejects as foreign.
    if "california" in loc:
        return "ca"
    # A foreign locale hard-rejects only when there's NO US option — a
    # multi-country posting that also offers the US ("CA - Canada; US") should
    # pass to the judge, not be silently hidden. A concrete non-CA US state
    # still rejects even with a US label (on-site Denver is not actionable).
    if foreign and not us_signal:
        return "reject"
    if "ca" in tokens:
        return "ca"
    if non_ca_state:
        return "reject"

    # Bare national label ("United States") or an unrecognized string — no firm
    # location signal, so pass to the judge rather than hard-reject.
    return "unknown"


# Multi-location separators — how a posting joins several distinct locations.
# Deliberately NOT the comma (that's the "City, ST" delimiter). Whitespace-bounded
# " or " so it can't split a single token like "Portland, Oregon".
_LOCATION_SEP_RE = None


def _location_segments(location: str):
    """Split a location string into its distinct location options. A single
    location returns a one-element list. Segments are stripped, lowercased-safe
    handled by the caller."""
    import re as _re
    global _LOCATION_SEP_RE
    if _LOCATION_SEP_RE is None:
        _LOCATION_SEP_RE = _re.compile(r"[;\n]|\s+/\s+|\s+\|\s+|\s+or\s+", _re.IGNORECASE)
    loc = str(location or "").strip()
    if not loc:
        return []
    return [s.strip() for s in _LOCATION_SEP_RE.split(loc) if s.strip()]


def extract_actionable_location(location: str = ""):
    """For a MULTI-location posting where one in-area option is what makes
    the role actionable, return that segment so the reader can see WHICH location
    is the right one (e.g. "Austin, TX; Novato, California; Seattle, WA" →
    "Novato, California"). Returns None for single-location strings or when
    there is nothing to disambiguate (all in-area, or no in-area option). The
    primary metro ("bay") is preferred over other CA, which is preferred over
    the secondary metro ("la")."""
    segments = _location_segments(location)
    if len(segments) < 2:
        return None
    has_non_ca = any(
        classify_location(s) not in ("bay", "la", "ca", "remote") for s in segments)
    if not has_non_ca:
        return None  # all options are already CA/remote — nothing to flag
    for pref in ("bay", "ca", "la"):
        for s in segments:
            # Guard against a JD blob that happens to contain "california": a real
            # location segment is short. Skip anything implausibly long.
            if classify_location(s) == pref and len(s) <= 60:
                return s
    return None


def _normalize_location_segment(seg: str) -> str:
    """Canonicalize one location segment so the SAME place written differently
    compares equal: "Los Angeles, CA" == "Los Angeles, CA, US" ==
    "Los Angeles, California" == "Los Angeles, CA (Hybrid)". Drops parentheticals
    and trailing US country labels, canonicalizes "California"→"ca", normalizes
    comma/space. (Without this, exact-segment matching under-suppresses — a stored
    key would leak the ~half of postings that carry a ", US" suffix.)"""
    import re as _re
    s = str(seg or "").strip().lower()
    if not s:
        return ""
    s = _re.sub(r"\s*\([^)]*\)", "", s)              # drop "(hybrid)", "(remote)"
    s = _re.sub(r"\s*,\s*", ", ", s)                  # normalize comma spacing
    s = _re.sub(r"\bcalifornia\b", "ca", s)           # canonicalize CA
    # strip a trailing US country label (", US" / ", USA" / ", United States[ of America]")
    s = _re.sub(r",\s*(?:united states of america|united states|u\.?s\.?a?\.?|usa|us)\s*$", "", s)
    s = _re.sub(r"\s+", " ", s).strip().strip(",").strip()
    return s


def is_location_suppressed(location: str, suppressed) -> bool:
    """True if a job at ``location`` should be dropped given ``suppressed`` — the
    set of no-go location strings the candidate dismissed. Compares NORMALIZED segments
    (so "Los Angeles, CA" == "Los Angeles, CA, US"), never bare substrings (a
    stored "la" must not drop "Atlanta"), and only drops when EVERY actionable
    segment is suppressed, so a multi-location posting that still offers a
    non-suppressed CA/remote option is kept."""
    if not suppressed:
        return False
    supp = {_normalize_location_segment(s) for s in suppressed if s and str(s).strip()}
    supp.discard("")
    if not supp:
        return False
    loc = str(location or "").strip()
    if not loc or loc.lower() == "none":
        return False
    if _normalize_location_segment(loc) in supp:   # whole-location match (single-loc)
        return True
    segments = _location_segments(loc)
    if not segments:
        return False
    norm = [_normalize_location_segment(s) for s in segments]
    actionable = [n for n, s in zip(norm, segments)
                  if classify_location(s) in ("bay", "remote", "la", "ca")]
    candidates = actionable or norm
    return all(n in supp for n in candidates)


def location_priority(location: str = "", title: str = "") -> int:
    """0 = primary metro, 1 = remote, 2 = unknown/other/CA, 3 = secondary-metro on-site.

    Thin wrapper over ``classify_location`` (the single source of truth) for the
    call sites that tier by integer. A ``reject`` maps to 2 here — the pre-filter
    already drops rejects before the cascade, so this only matters for exempt
    USER_INPUT rows, which should stay visible.
    """
    return {"bay": 0, "remote": 1, "la": 3}.get(
        classify_location(location, title), 2)


MIN_DATE = pd.Timestamp("2024-01-01")

# --- Verified metrics ---
# The only numbers a generator may use. They come from the candidate profile:
# a ready-made `metrics_text` block, or a structured `metrics` list that
# candidate_profile.metrics_text() renders. Nothing about a person is encoded
# in this module.
METRICS_BANK = PROFILE.get("metrics") or []

METRICS_BANK_TEXT = metrics_text()

# --- Application Standard Answers (Phase 15) ---
APPLICATION_STANDARD_ANSWERS = dict(PROFILE["application_answers"].raw())

# --- LangFuse + Perplexity API Config ---
# LangFuse (self-hosted at ops/langfuse/, cascade observability). Keys live in
# ops/langfuse/langfuse-keys.conf, a file of its own next to the stack rather
# than the main dotenv file. Values already present in the environment win
# (load_dotenv never overrides).
from dotenv import load_dotenv as _load_langfuse_keys
_load_langfuse_keys(os.path.join(os.path.dirname(__file__), "..", "..",
                                 "ops", "langfuse", "langfuse-keys.conf"))
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
USE_LANGFUSE = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
USE_PERPLEXITY = bool(PERPLEXITY_API_KEY)
# Agent API presets (2026 migration from Sonar chat completions): fast/low/medium/high/xhigh.
# "fast" = old "sonar" model; the search_context_size knob is gone, consolidated into presets.
PERPLEXITY_PRESET = "fast"
# Contact discovery empirically needs deeper grounding than the cascade default
# (was search_context=medium on sonar) — one preset up.
PERPLEXITY_CITATIONS_PRESET = "low"
PERPLEXITY_TIMEOUT = 30
PERPLEXITY_RPM_LIMIT = 50
PERPLEXITY_COMPANY_PROFILE_TTL_DAYS = 30
PERPLEXITY_JOB_ENRICHMENT_TTL_DAYS = 14
PERPLEXITY_CAREERS_TTL_DAYS = 7
PERPLEXITY_HIRING_INTEL_TTL_DAYS = 14
PERPLEXITY_SALARY_INTEL_TTL_DAYS = 14
PERPLEXITY_SALARY_REC_TTL_DAYS = 30

# Auto-enrich matched jobs with Perplexity intel + Opus salary advisor after cascade.
# OFF by default (cost): enriching every match up front pays for jobs nobody
# may open. The UI CompanyIntelPanel and the
# assistant fetch the same intel ON DEMAND (cached 14-30d) — cost follows
# attention. Flip to True to restore pre-warming.
AUTO_ENRICH_MATCHES = False

# Thin-description rescue (Stage 1->2): a job whose SCRAPED description is shorter
# than this gets a Perplexity role-summary appended before Stage 2, so a good role
# isn't rejected as "INSUFFICIENT DATA" just because the board returned a stub.
# Bounded per run to control Perplexity cost. Inert unless USE_PERPLEXITY.
THIN_DESCRIPTION_THRESHOLD = 200
PERPLEXITY_JD_RESCUE_MAX = 60
# Comp extraction fallback is a trivial JSON pull — never the premium judge model
# (it borrowed CASCADE_STAGE3_MODEL and would have run on fable-5-1 at $10/$50).
COMP_EXTRACT_MODEL = "claude-sonnet-5"

# === Email Config ===
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "")
AUTO_SEND_EMAIL = True   # Email digest sent automatically after each search run
ERROR_ALERT_EMAIL = os.getenv("ERROR_ALERT_EMAIL", "")  # Separate recipient for backend error notifications

# === Site Config ===
SITE_MIN_GAP_SEC = {
    "glassdoor": 45,
    "zip_recruiter": 90,
    "linkedin": 10,
    "indeed": 10,
}

SEARCH_SITES = ["linkedin", "indeed", "glassdoor", "zip_recruiter", "ats_boards", "industry_boards", "remote_boards", "reddit_jobs", "google_jobs"]

# Keywords for the SLOW browser-based scrapers — ZipRecruiter (Scrapling
# solve_cloudflare, ~90s/query) and Glassdoor (crawl4ai render, ~40s/query).
# These run a real browser, so the set is kept SMALL; both fuzzy-search and
# expand broad terms well, and the cascade does the real filtering downstream.
BROWSER_SCRAPE_KEYWORDS = [
    "Content Producer", "Video Producer", "Creative Producer",
    "Production Manager", "Senior Producer", "Podcast Producer",
    "Post Production Producer",
    "Event Manager",  # Phase 25: live/experiential lane had zero browser coverage; biggest verified volume gap
]

# === Remote Second-Pass (Phase 16) ===
# The regular loop scrapes the profile's home location (search.locations.home).
# Remote roles are in scope too, so this adds a second jobspy pass with
# is_remote=True + national location over a
# CURATED set of high-signal producer keywords. (Generic titles like bare
# "Production Manager" pull manufacturing noise nationally; we use specific producer
# titles, and the NEGATIVE_KEYWORDS pre-filter removes residue downstream.)
# Bounded to a few sites/keywords to limit scrape load and block risk. Toggle off
# by setting REMOTE_SECOND_PASS = False.
REMOTE_SECOND_PASS = True
REMOTE_PASS_SITES = ["linkedin", "indeed"]   # jobspy sites that handle national remote well

# === Secondary-Metro Pass ===
# A profile can whitelist a second metro (search.locations.secondary_metro_terms
# and the matching entries in search.locations.allowed), yet the
# LinkedIn/Indeed loop only searches
# the home location. This pass runs the same curated keyword set against the
# secondary metro (search.locations.secondary_metro_label). Toggle off with
# SECONDARY_SECOND_PASS = False.
SECONDARY_SECOND_PASS = True
SECONDARY_PASS_LOCATION = SEARCH["locations"]["secondary_metro_label"]
SECONDARY_PASS_KEYWORDS_EXTRA = [
    # Extra keywords for the secondary-metro pass, on top of REMOTE_PASS_KEYWORDS
    "Entertainment Producer", "Post-Production Supervisor", "TV Producer",
]
REMOTE_PASS_KEYWORDS = [
    "Content Producer", "Video Producer", "Senior Producer", "Digital Producer",
    "Creative Producer", "Branded Content Producer", "Integrated Producer",
    "Social Producer", "Multimedia Producer", "Podcast Producer", "Audio Producer",
    "Content Production Manager",   # coordinator titles are left out: the title pre-filter rejects them
    "Creative Operations", "Content Operations", "Localization Producer",
    "Localization Project Manager", "Live Producer", "Livestream Producer",
    "Event Producer", "YouTube Producer", "Creative Project Manager",
    "Brand Producer", "Documentary Producer", "Lead Producer",
]

# === Industry Board Config ===
# ShowbizJobs category slugs for HTML scraping
SHOWBIZJOBS_CATEGORIES = ["production-crew", "podcast", "post-production", "digital", "music"]

# EntertainmentCareers.net RSS feed category IDs
ENTERTAINMENTCAREERS_FEEDS = [
    "ecnjcat119",   # Music
    "ecnjcat168",   # Podcast
    "ecnjcat121",   # Post Production
    "ecnjcat112",   # Film/TV Production
    "ecnjcat157",   # Content Creator
    # "ecnjcat166" (Coordinator) left out: coordinator titles are rejected by the title pre-filter
    "ecnjcat124",   # Theater/Live Events
    "ecnjcat138",   # Video Game
    "ecnjcat161",   # YouTube
]

# === AI Model Config ===
DEFAULT_FILTER_MODEL = "claude-haiku-4-5-20251001"
# Generation → opus-5 (2026-07-25 budget-guarded bake-off, $4.75): 4 real
# postings, fable-5 vs opus-5 head-to-head. Mean HM 78.0 vs 77.0 =
# statistical tie; BOTH zero hallucinations (opus-4-8 truth judge); opus-5
# 27s vs 42s mean gen. Tie on quality -> price decides: opus-5 $5/$25 vs
# fable's $10/$50 (the 2026-07-25 pricing check showed fable at 2x the
# assumed rate). Halves the cost of the generation stack.
# (History: fable-5 won the 2026-07-17 bake-off vs opus-4-8/sonnets.)
# Drives CV + CL + outreach + prep + salary synthesis.
# 2026-09-01: Fable 5.1 considered for generation and NOT adopted — same $10/$50
# rate, and the 07-25 bake-off showed no quality gain over opus-5 for CVs.
# Re-run a CV model bake-off against fable-5-1 before changing this.
# 2026-09-26: opus-5 -> opus-5-5. Opus 5 is legacy; 5.5 is the current Opus,
# CHEAPER ($4/$20 vs $5/$25) with a Jun-2026 knowledge cutoff. Its thinking is
# always on and its DEFAULT EFFORT DROPPED to "medium" (Opus 5: "high"). Pinned
# to "xhigh" in MODEL_EFFORT below (a CV reaches an employer, so it gets the extra care).
DEFAULT_GENERATION_MODEL = "claude-opus-5-5"

# Per-model effort pins. Effort is the only thinking control on the 5.5
# generation, and a request that omits it runs at the MODEL's default — which
# changed under us: Opus 5 defaulted to "high", Opus 5.5 defaults to "medium".
# Pin wherever the default differs from what we tuned against, so a model
# upgrade can never silently lower quality. call_llm(effort=...) overrides.
# Opus 5.5 runs above its default: generation output reaches an employer, and
# volume is low, so the extra thinking is cheap per
# use. At xhigh/max, call_llm raises max_tokens to 64K and streams — see there.
#
# Evidence, 2026-09-26 (a CV model bake-off; 4 real postings,
# fully grounded, sonnet-5 production grader as judge, $4.05 metered):
#   mean hiring-manager score  Opus 5 67.0 | 5.5@high 64.3 | 5.5@medium 64.7 (3 roles)
#   paired 5.5@high - Opus 5 = -2.75, SE 3.4 -> a TIE (July's fable-vs-opus
#   verdict too); medium >= high on 3/3 paired roles; 0 hallucinations in any
#   arm; generation 26s / 44s / 30s. One grounded xhigh CV: 125s, 14,350
#   output tokens (would have truncated at the old 16K cap), $0.42; its cover
#   letter 79s, $0.29. Effort did NOT buy score on this grader — xhigh is a
#   deliberate choice for CV care, not a measured win. Re-run the bake-off
#   with an xhigh arm before treating it as settled.
MODEL_EFFORT = {
    "claude-opus-5-5": "xhigh",
    # The Stage 3 judge. "high" is Fable 5.2's API default today; it is pinned
    # so the judge keeps the level the 2026-11-16 replay measured if that
    # default ever moves (Opus 5 -> 5.5 dropped a level without a word).
    # Anthropic's 5.2 notes: most of the gain arrives by "high"; above it is
    # uneven and costs many more tokens.
    "claude-fable-5-2": "high",
}

# Available Models for UI — grouped by provider, defaults listed first
AVAILABLE_MODELS = [
    # --- Anthropic Claude (primary — best for structured analysis) ---
    "claude-fable-5-2",           # Most capable; the Stage 3 judge ($8/$40 MTok)
    "claude-fable-5-1",           # Previous Fable; the judge's first fallback ($10/$50 MTok)
    "claude-opus-5-5",            # Current Opus — generation default ($4/$20 MTok)
    "claude-opus-5",              # Legacy Opus ($5/$25 MTok)
    "claude-sonnet-5",            # Speed/intelligence balance ($2/$10 MTok)
    "claude-opus-4-8",            # Previous-generation Opus ($5/$25 MTok)
    "claude-sonnet-4-6",          # Best speed/intelligence balance ($3/$15 MTok)
    "claude-haiku-4-5-20251001",  # Fast + cheap, great for classification ($1/$5 MTok)
    # --- Google Gemini (free tier available) ---
    "gemini-3-pro-preview",       # Flagship, best reasoning
    "gemini-3-flash-preview",     # Fast + capable
    "gemini-2.5-pro",             # Production-stable, advanced reasoning
    "gemini-2.5-flash",           # Best price/performance ratio
    "gemini-2.5-flash-lite",      # Budget option
    # --- OpenAI ---
    "gpt-5.2",                    # Latest flagship, broad intelligence
    "gpt-5.2-pro",                # Extended thinking, hardest tasks
    "gpt-5",                      # Previous flagship
    "gpt-5-mini",                 # Cost-efficient
    "gpt-4.1",                    # Excellent instruction following, 1M context
    "gpt-4.1-mini",               # Great balance, cheap
    "gpt-4o",                     # Legacy but still works
    # NOTE: ollama/* local models removed 2026-07-17 — deprecated; the
    # call_llm ollama/ routing still exists but nothing configures it.
]

# Fallback Configuration
# If the key model fails, try the models in the list in order
FALLBACK_CHAIN = {
    "gemini-3-pro-preview": ["gemini-2.5-pro", "gemini-2.5-flash", "claude-sonnet-4-6"],
    "gemini-3-flash-preview": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "claude-haiku-4-5-20251001"],
    "gemini-2.5-pro": ["gemini-3-pro-preview", "gemini-2.5-flash", "claude-sonnet-4-6"],
    "gemini-2.5-flash": ["gemini-3-flash-preview", "gemini-2.5-flash-lite", "claude-haiku-4-5-20251001"],
    "gpt-5.2": ["gpt-5", "gpt-4.1", "gpt-4o"],
    "gpt-5.2-pro": ["gpt-5.2", "gpt-5", "gpt-4.1"],
    "gpt-5": ["gpt-5.2", "gpt-4.1", "gpt-4o"],
    "gpt-5-mini": ["gpt-4.1-mini", "gpt-4o"],
    "gpt-4.1": ["gpt-4.1-mini", "gpt-4o"],
    "claude-fable-5-2": ["claude-opus-5-5", "claude-opus-5", "claude-sonnet-5"],
    "claude-fable-5-1": ["claude-opus-5-5", "claude-opus-5", "claude-sonnet-5"],
    "claude-opus-5-5": ["claude-opus-5", "claude-sonnet-5"],
    "claude-opus-5": ["claude-opus-4-8", "claude-sonnet-5"],
    "claude-sonnet-5": ["claude-sonnet-4-6", "claude-opus-5-5"],
    "claude-opus-4-8": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "claude-sonnet-4-6": ["claude-haiku-4-5-20251001", "claude-opus-4-8"],
}

# === Truncation Limits ===
TRUNCATION_JOB_DESC_FILTER = 6000    # For Stage 2/3 grading — full context where verdicts are made
TRUNCATION_JOB_DESC_STAGE1 = 2500    # Stage 1 is a lean-YES plausibility gate; 6K chars
                                     # per job made it the most expensive filter line
TRUNCATION_JOB_DESC_CV = 6000        # For CV tailoring prompts (2000 cut off the
                                     # Requirements/Qualifications sections — the
                                     # exact text the CV must mirror)
TRUNCATION_PROFILE_SUMMARY = 4000    # For summary generation
TRUNCATION_PROFILE_EXPERIENCE = 6000 # For experience bullet rewrite
TRUNCATION_TRANSCRIPT_CONTEXT = 44000 # Interview-notes budget. Sized to carry a
                                      # whole notes file of up to ~40K characters
                                      # with headroom (~9K tokens, sent ~3x per CV
                                      # since the consolidated single call —
                                      # ~$0.12/CV). An earlier 8000-character head
                                      # truncation silently dropped every later
                                      # section, so whole parts of the career and
                                      # the cover-letter stories never reached ANY
                                      # prompt. If the file outgrows the budget,
                                      # domain-priority section selection degrades
                                      # gracefully.

# === Interview Transcript Config ===
TRANSCRIPT_CONTEXT_PATH = PROFILE["paths"]["transcript"]

# === Career Source Bank ===
# The candidate's own career documents, condensed into one local text file. This is the AUTHORITATIVE fact
# source for CV/CL generation and the quality judges. Precedence: SOURCE BANK >
# transcript for facts/numbers/dates/titles; transcript stays authoritative for
# stories and voice. Gitignored (career PII stays local).
SOURCE_BANK_PATH = PROFILE["paths"]["source_bank"]
TRUNCATION_SOURCE_BANK = 42000  # chars; sized to inject a bank of up to ~40K
# whole (a truncated bank silently drops the material at its tail, such as
# project case studies and education)

# === Outreach Intel (Phase 24) ===
# When the candidate moves an application to "Applied", the system auto-generates an
# outreach playbook: 30-day company pulse (last30days engine), likely hiring
# team (Perplexity, public sources with citations ONLY — cite or omit), how to
# reach them, a dated follow-up cadence, and grounded message drafts.
OUTREACH_INTEL_ENABLED = True
# (No standalone playbook email — the app is the surface; the daily digest
#  carries a one-line 'intel ready' pointer instead.)
OUTREACH_INTEL_DIR = "results/outreach"
# Cadence anchors: days after date_applied. Research consensus (recruiter
# surveys: first touch 5-10 business days, max 2 unsolicited follow-ups):
#   day 2  — proactive intro (LinkedIn note, NOT a status ask)
#   day 7  — first follow-up (one proof point, ask about timeline)
#   day 14 — second and FINAL follow-up (must add something new)
#   day 21 — close loop: deprioritize, re-engage only on triggers
OUTREACH_FOLLOWUP_DAYS = [2, 7, 14, 21]
# last30days engine (installed as a Claude Code marketplace plugin).
LAST30DAYS_ENGINE_DIR = os.path.expanduser(
    "~/.claude/plugins/marketplaces/last30days-skill/skills/last30days")
OUTREACH_ENGINE_TIMEOUT_SEC = 480    # company-pulse subprocess hard timeout
OUTREACH_CONTACT_CACHE_DAYS = 14     # Perplexity contact-discovery cache TTL
# LinkedIn connection-note ceiling, enforced IN CODE on every playbook (the
# old "<=250 chars" prompt budget was never checked in code). 200 = what a
# free LinkedIn account can send; raise to 300 only on Premium.
LINKEDIN_NOTE_MAX = 200

# === Interview Prep (per-role) ===
# When the candidate moves a card to an interview stage (Screening Call / Interview /
# Final Interview), the system auto-generates a prep sheet: likely questions
# from the JD's competencies, the candidate's strongest STAR stories mapped to them (from
# the candidate's own interview notes — grounded, never invented), a company angle
# from cached intel, and smart questions to ask. Also available on demand via
# the Kanban "Prep me" button and the chat assistant's interview_prep tool.
INTERVIEW_PREP_ENABLED = True
INTERVIEW_PREP_DIR = "results/interview_prep"
# Optional fixed answers for a prep sheet. Set a value (the user's own wording)
# and the sheet uses it verbatim as the basis; None = a neutral answer is
# written and nothing is invented (enforced in code by interview_prep's
# post-checks).
CANDIDATE_NARRATIVE = {
    "why_left": None,           # answer to "why are you looking?"
    "comp_target": None,        # answer to the pay-expectations question
    "people_management": None,  # answer to the people-management question
}
# Stages at/after which prep auto-spawns on the transition into them. Offer is
# included (Phase 27): the offer conversation is prepped like any other round.
# An existing sheet is rebuilt only when the card reaches a HIGHER stage than
# the one it was built for (applications.prep_stage).
INTERVIEW_PREP_TRIGGER_STAGES = ("Screening Call", "Interview", "Final Interview", "Offer")

# === Auto-Generate CV/CL Config ===
AUTO_GENERATE_CV_CL = True  # Auto-generate CV/CL for HIGH_MATCH jobs after cascade filtering
# HARD CAP per run. CV+CL on the premium model costs ~$0.22/job — an uncapped
# high-volume run (dozens of HIGH matches at once) spends dollars on documents alone.
# The top-confidence jobs get docs automatically; the rest are one click away
# (Review page button / assistant generate_cv).
AUTO_CV_MAX_PER_RUN = 5

# Truth judge = opus-4-8 (2026-07-17, post-bake-off): generation moved to
# fable-5, and the fabrication gate must be DECORRELATED from the generator —
# a model is weakest at spotting its own characteristic errors. Opus is the
# strongest non-fable judge; one call per CV, so cost is negligible.
# (Earlier same-day this was fable-5, chosen while opus was the generator —
# the decorrelation principle is the constant, the model follows from it.)
# Truth judge = sonnet-5 (2026-07-25: all models moved to the 5 family).
# DECORRELATION INVARIANT holds: sonnet-5 != opus-5-5 (the generator) — the
# fabrication gate must run on a DIFFERENT model than the generator. Note:
# sonnet-5 is UNPROVEN at fabrication-catching (opus-4-8 caught the 07-17
# sonnet-4-6 JD-noun fabrication); the judge prompts carry explicit
# tense-fraud + JD-noun-injection checks — WATCH the HALLUCINATION flag rate
# for the first week and revert to opus-4-8 if catches drop suspiciously.
CV_QUALITY_MODEL = "claude-sonnet-5"
CV_QUALITY_MAX_RETRIES = 2
CV_QUALITY_FLAGS = ["PASSED", "HALLUCINATION", "HALLUCINATION_UNRESOLVED"]

# Generate a whole CV (summary + ALL role bullets) in ONE structured LLM call
# instead of 1 summary call + 1 call per role. Sends the shared context (profile,
# JD, rules, transcript) once instead of ~6x — cheaper AND more coherent. Falls
# back to the per-section path automatically if the single call fails/parses bad.
CV_SINGLE_CALL_GENERATION = True

# === CV Length Gate (2-page rule, no rendering) ===
# The .docx is the deliverable (the candidate edits it directly) — we deliberately do
# NOT render to PDF. Instead a deterministic height model estimates page count
# from the docx content against the fixed template geometry (US Letter, 0.5in
# margins → 540x720pt text box, 11pt default, 1.15x line, 10pt after each para).
# Calibrated on real generated files: genuine 2-page CVs land at ~1.5-1.8 pages of
# content; content over ~2.0 pages spills to a 3rd. Over the ceiling → a
# relevance-weighted trim (cut lowest-value bullets first), capped retries.
CV_LENGTH_GATE_ENABLED = True
CV_MAX_PAGES = 2.0           # a 2-page box; content over this spills to page 3
# Trim trigger. Calibrated against RENDERED ground truth (soffice) over the
# generated CVs + all base templates: every one renders to <=2 pages and the
# contextualSpacing-corrected estimator places them at 1.42-1.95 (max 1.946).
# A genuine 3-page CV reads ~2.5+. So 2.1 sits above the observed 2-page max
# with margin for estimator noise, and well below real 3-page overflow.
# (An earlier 2.02 ceiling + a bullet-spacing over-count false-tripped ~15% of
# genuine 2-page CVs — fixed by honouring List Bullet's w:contextualSpacing.)
CV_PAGE_CEILING = 2.1
CV_LENGTH_MAX_RETRIES = 2    # trim attempts before flagging CV_TOO_LONG
# Height-model constants (see scout calibration; do not change without re-measuring)
_CV_LINE_MULT = 1.15         # docDefault line rule 276/240
_CV_SPACE_AFTER_PT = 10.0    # docDefault w:after=200 → 10pt
_CV_CHAR_WIDTH_FACTOR = 0.48 # avg glyph advance ≈ 0.48 * font size (Helvetica 11)

# === Hiring-Manager Grader (2026-07-16) ===
# A second, COMPETITIVENESS judge (the truth judge stays grade_cv_quality):
# a senior-hiring-manager persona builds the ideal-candidate profile from THIS
# JD, then scores the CV against it (relevance, evidence, ATS alignment,
# differentiation, clarity) and returns concrete fixes. Below the threshold →
# ONE revision pass applying the fixes (angle/reframe only — never invention),
# re-checked by the fabrication gate before acceptance. Informational: the CV
# always ships; the grade lands in cv_audit_summary.
# Phase 27: a DIAGNOSTIC on the final CV text only (told today's date, grounded
# on the source bank, shown SKILLS). The revision pass no longer runs on the
# shipped path, so CV_HM_REVISE_THRESHOLD is unused by save_documents_for_job.
CV_HM_GRADER_ENABLED = True
# Grader model: fable-5 -> sonnet-5 (2026-07-25 cost audit). Grading = read
# JD + CV, score 5 dimensions — it does not need the $10/$50 flagship, and
# fable-5 grading fable-5's own output was partial self-grading anyway.
# sonnet-5 at $3/$15 cuts the two grader calls per package ~70%.
CV_HM_GRADER_MODEL = "claude-sonnet-5"
CV_HM_GRADE_THRESHOLD = 75   # informational bar: below this the grade is flagged in the audit
# Revision fires ONLY below this. 2026-07-25 audit of generated packages: with the
# revise trigger at 75 the loop fired 66% of the time and post-revision avg
# (70) was LOWER than pre (71) — two extra flagship calls per firing buying
# nothing. Reserve revision for genuinely broken CVs.
CV_HM_REVISE_THRESHOLD = 60

# === ATS Keyword Coverage ===
# Score the posting's required / preferred keywords against what the FINAL CV
# says (summary + shipped bullets + skills), classified as covered /
# synonym-only / missing-but-have-it / missing-gap, grounded on the same source
# bank as the generator. Since Phase 27 a DIAGNOSTIC only: nothing is woven
# into a shipped CV (the audit keeps the coverage; gaps still feed the letter).
CV_KEYWORD_COVERAGE_ENABLED = True
CV_KEYWORD_MODEL = CV_QUALITY_MODEL   # follows the truth judge (was a stale sonnet-4-6 pin)

# === CV content modes + truth gates (Phase 27) ===
# What the CV's EXPERIENCE section ships (the summary is gated the same way in
# every mode — the only LLM text that can reach an employer in a CV):
#   B0 = the routed base variant's own truth-audited bullets, verbatim (default)
#   B2 = the candidate's truth-audited bullets selected + ordered per job (cv_pool.py),
#        verbatim; title-first variant routing. Flip only after the offline gate.
# The generator's reworded bullets (B1) never ship: they are kept in the audit
# as a shadow. Env override: CV_CONTENT_MODE=B2 (anything else falls back to B0).
CV_CONTENT_MODES = ("B0", "B2")
CV_CONTENT_MODE = os.getenv("CV_CONTENT_MODE", "B0").strip().upper()
if CV_CONTENT_MODE not in CV_CONTENT_MODES:
    CV_CONTENT_MODE = "B0"
# Cohort tag stored with every saved document version: every employer-visible
# switch (B0 release, B2 flip, ...) gets its own tag and >= 3 days' separation.
CV_COHORT_TAG = os.getenv("CV_COHORT_TAG", "doc_b0")
# A posting shorter than this (after untrusted-text stripping) is "thin": the
# general version ships (base summary + base bullets), no tailoring.
CV_THIN_POSTING_CHARS = 300
# Judge the generator's (unshipped) B1 bullets too, for the shadow record —
# one extra truth-judge call per CV. False keeps the bullets without a verdict.
CV_B1_SHADOW_JUDGE = True
# ...but only on the background paths (generate_package source): the
# synchronous Generate click ('api') and manual add ('user_input') skip the
# diagnostic-only call (~24K tokens, 30-60 s) that the candidate would wait on.
CV_B1_SHADOW_SOURCES = ("pipeline", "batch")

# === B2 Bullet Pool (Phase 27 — engine only; the content-mode flag keeps it off) ===
# B2 = the candidate's truth-audited base-CV bullets, selected and ordered per job and
# written word for word (cv_pool.py). No bullet text is ever generated. The pool
# is the union of the 8 domain bases, built by scripts/build_cv_pool.py into a
# gitignored file (it holds the candidate's CV text) and refused when the bases change.
CV_POOL_PATH = "cv_pool.json"
# Selection is a SEPARATE small call: read the posting, return requirement
# quotes + bullet ids. The grader tier is enough; every id and quote is checked
# in code, so a weak answer costs a fallback, never a wrong bullet.
CV_B2_SELECTION_MODEL = CV_HM_GRADER_MODEL
CV_B2_SELECTION_EFFORT = "low"
CV_B2_SELECTION_TIMEOUT_SEC = 120
# Minimum bullets per role (most recent role, every other role) — the same
# floor the trim step keeps — capped by what the routed base itself has.
CV_B2_ROLE_FLOOR = (3, 2)
CV_B2_MAX_CHARS_RATIO = 1.05   # total bullet characters <= routed base's total + 5%
# When the selection call fails (timeout, unparseable, no verifiable quote):
# True = keyword-ranked selection from the same pool, False = no B2 for this job.
CV_B2_DETERMINISTIC_FALLBACK = True

# === Gemini API Config ===
GEMINI_RPM_LIMIT = 10         # Requests Per Minute (Free Tier is ~15)
GEMINI_MAX_RETRIES = 10       # Increased retries for robustness
GEMINI_INITIAL_BACKOFF = 2.0  # Seconds

# === Cascade Filter Config ===
# 3-Stage Cascade: Stage 1 (cheap binary YES/NO) → Stage 2 (full grading) → Stage 3 (judge borderline)
CASCADE_STAGE1_MODEL = "claude-sonnet-5"               # Plausibility gate — upgraded haiku→sonnet-5 2026-08-13 (a more capable model; ~3x stage-1 cost)
# S2 haiku→sonnet-4-6→sonnet-5 (2026-07-17). Stage 2 makes nuanced FINAL
# rejections (rejects never reach Stage 3). In a labeled A/B on real postings
# with known good and bad fits, sonnet-5 false-rejected clearly fewer of the
# good ones than sonnet-4-6, and lets a few more junk rows through to S3 —
# the RIGHT error shape (a false-keep costs one judge call; a false-reject
# loses the job forever).
CASCADE_STAGE2_MODEL = "claude-sonnet-5"               # Full recruiter grading — rejects are FINAL
# S3 fable-5-1 (2026-09-01) -> fable-5-2 (2026-11-16). Same request surface,
# $8/$40 per MTok instead of $10/$50, its own rate-limit pool. The judge is the
# one judgment-heavy, low-volume stage.
# Replayed first on 24 stored judge calls made under the current prompt
# (scripts/replay_stage3_models.py, $10 metered): 24/24 answers parsed; $0.148
# and 16 s per call against $0.187 and 21 s. The two models are NOT
# interchangeable: exact verdicts matched the stored ones on 15/24, while a
# same-day re-run of 5.1 matched 18/24 (the tool's gate G2 failed on that).
# Read at the visible tier, 5.2 kept 3 more jobs visible and hid 1.
# Re-run on the 7 contested cases, 5.2 changed its answer on 1 and 5.1 on 3;
# a blind panel of three reviewers (rules, candidate interest, hiring realism) sided
# with 5.2's tier in 16 of 20 samples and with 5.1's in 7 of 14. Net: a
# steadier judge that leans slightly toward "worth a look". To go back,
# swap this line and the first fallback below.
CASCADE_STAGE3_MODEL = "claude-fable-5-2"              # Premium judge (recall-mode); generation deliberately stays on Opus (see DEFAULT_GENERATION_MODEL)
CASCADE_STAGE2_BATCH_SIZE = 5                    # Jobs per API call in Stage 2
# HARD CAP on premium Stage-3 judging per run. The judge's only power is
# upgrading POTENTIAL->strong; an unjudged POTENTIAL still surfaces in the
# worth_a_look tier, so overflow loses nothing visible. 69 single Opus calls
# in one high-volume run cost $1.78 by themselves.
CASCADE_STAGE3_MAX_JOBS = 24

# Stage-level fallbacks: tried in order if the primary model fails.
# NOTE: for claude-* models call_llm() does NOT retry in-family — its Anthropic
# catch-all returns "" on error — so any Claude-to-Claude fallback must be listed
# HERE at the stage level. S2 makes FINAL rejections; keep a Claude grader first
# so a sonnet-5 outage degrades to sonnet-4-6 (A/B runner-up), not straight to
# a weaker cross-provider model.
# Ollama/local models REMOVED 2026-07-17 (long deprecated) — they
# survived here as zombie emergency fallbacks and failed their one real
# audition (a provider outage: Claude down -> Ollama also failed, 24 errors
# x 5-min timeouts ≈ 25 wasted minutes, zero rows rescued). Stage 1 is a
# binary YES/NO gate — gemini-2.5-flash handles it trivially and cross-provider
# fallbacks already exist at S2/S3. On total failure Stage 1 fails OPEN
# (job passes to Stage 2), so no job is ever lost to a fallback outage.
CASCADE_STAGE1_FALLBACKS = ["gemini-2.5-flash"]
CASCADE_STAGE2_FALLBACKS = ["claude-sonnet-4-6", "gemini-2.5-flash", "gpt-4.1-mini"]
# Fable 5.1 was the judge until 2026-11-16 and draws from a separate
# rate-limit pool, so it answers first when Fable 5.2 is down or throttled:
# an outage of the new model costs nothing in judgment. Then Opus 5.5 (current
# Opus) and legacy Opus 5, proven on this exact judging task.
CASCADE_STAGE3_FALLBACKS = ["claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-4-6", "gemini-2.5-pro"]

# Stage 1 is a cheap PLAUSIBILITY gate, not a fit judge. The tiered output lets
# Stage 2/3 catch and label adjacent/stretch roles, so Stage 1 leans hard toward
# YES and only drops the obviously-unrelated. Backtested: vs the old prompt this
# recovers real adjacent roles (3D Production Manager, PR/Media Lead, Content
# Personalization) from the cull with ZERO regression on existing matches; the
# noise it lets through is re-rejected by the untouched Stage 2 grader.
CASCADE_STAGE1_PROMPT = """Could this role PLAUSIBLY belong to the world of creative / media / content / brand / video / audio / live-event production — even loosely, as a stretch, or as an adjacent hybrid? A later AI stage judges the real fit and sorts it into a tier; your only job here is to drop the OBVIOUSLY unrelated.

Lean YES. Pass anything with any plausible production / content / creative / brand / studio / editorial / video / audio / podcast / events angle, including hybrid editor-producer roles, coordinator/associate levels, and roles where you are unsure.

Answer NO only if the role is CLEARLY and PRIMARILY one of these, with no content/creative/production component:
- Hardware or software engineering / IT / DevOps / data science / ML (requires AWS, Kubernetes, CI/CD, SQL, Python, Java, microservices)
- Technical Program/Project Manager with engineering oversight; Product Manager for a software product
- Manufacturing / industrial / construction / warehouse / logistics
- Clinical / medical / nursing / pharmacy
- Finance / accounting / legal / HR / recruiting
- Pure sales / business development / ad sales / account management / client services / partner management / customer success

Answer only YES or NO.

Job Title: {title}
Location: {location}
Description: {description_snippet}"""

# Claude-optimized Stage 1 variant (selected by detect_model_family, like S2/S3).
# Structure follows Claude guidance: data block first, instructions after, and an
# explicit single-word output contract — the parser reads the verdict token, and
# a thinking model given no format contract will preamble.
CASCADE_STAGE1_PROMPT_CLAUDE = """<job>
Title: {title}
Location: {location}
Description: {description_snippet}
</job>

<task>
Could this role PLAUSIBLY belong to the world of creative / media / content / brand / video / audio / live-event production — even loosely, as a stretch, or as an adjacent hybrid? A later AI stage judges the real fit and sorts it into a tier; your only job here is to drop the OBVIOUSLY unrelated.
</task>

<rules>
Lean YES. Pass anything with any plausible production / content / creative / brand / studio / editorial / video / audio / podcast / events angle, including hybrid editor-producer roles, coordinator/associate levels, and roles where you are unsure. Do NOT judge seniority, location, industry prestige, or fit quality — later stages own those calls. Asymmetric stakes: passing a bad job costs one cheap downstream review; dropping a good one loses it forever.

Answer NO only if the role is CLEARLY and PRIMARILY one of these, with no content/creative/production component:
- Hardware or software engineering / IT / DevOps / data science / ML (requires AWS, Kubernetes, CI/CD, SQL, Python, Java, microservices)
- Technical Program/Project Manager with engineering oversight; Product Manager for a software product
- Manufacturing / industrial / construction / warehouse / logistics
- Clinical / medical / nursing / pharmacy
- Finance / accounting / legal / HR / recruiting
- Pure sales / business development / ad sales / account management / client services / partner management / customer success
</rules>

<output_format>
Reply with exactly one word: YES or NO. No preamble, no reasoning, no punctuation.
</output_format>"""

CASCADE_STAGE3_PROMPT = """You are a senior executive recruiter reviewing a borderline job match.
A junior reviewer flagged this as POTENTIAL MATCH. Your job is to make the final call.

=== CANDIDATE PROFILE ===
{profile_text}

=== JOB LISTING ===
Title: {title}
Company: {company}
Location: {location}
{domain_hint}
Description: {description}

=== JUNIOR REVIEWER'S ASSESSMENT ===
{stage2_reason}

=== SENIORITY NOTE ===
Associate titles (Associate Producer) are a legitimate seniority level — they are NOT "junior" and should NOT be downgraded. Coordinator-level roles and other entry-level roles (assistant roles other than Assistant Producer, trainee, a JD whose experience requirement tops out at 0-2 years) are BELOW the target level and its pay band: final_level = REJECT per the BELOW-LEVEL WALL, and the "overqualified but strategic fit" exception below does NOT apply to them. If Stage 2 flagged "overqualified but strategic fit," weigh company brand and domain alignment heavily. A junior title at a recognisable organisation in the target field with a strong domain match should NOT be downgraded. A step-down can be acceptable for the right organisation. This exception works DOWNWARD only — it never lifts Director/Head/VP-level or business roles above WORTH_A_LOOK.

=== ADDITIONAL CONTEXT (from web search, if available) ===
{perplexity_context}

=== YOUR TASK ===
Sort this role into the RIGHT TIER for the candidate — do NOT decide whether the candidate is allowed to see it. The goal is recall with honest labeling, so DEFAULT TO KEEPING THE JOB VISIBLE.

LEVEL REALITY (overrides everything): the target level is the one stated in the candidate profile: hands-on individual-contributor (IC) production, with no direct reports. The STRONG zone is IC production roles at that level. DO NOT treat over-leveled roles as "strategic fit" — a Director role that demands many years of leadership experience is not a realistic target. DO NOT penalize under-leveled roles as "a step down" — an Associate/mid role at a good company IS the target.

PEOPLE-MANAGEMENT TEST (run FIRST, before any tier decision): a benign-looking title can hide a people-leadership role. Read the DESCRIPTION for direct-report language — "manage a team of", "lead/grow/build the team", "hire", "headcount", "direct reports", "people leadership", "manage managers", "performance reviews". If the role's core is people management, final_level = REJECT — the target level is individual-contributor with no direct reports, so a role built around running/hiring/reviewing a team is not viable for the candidate, however production-relevant it reads and however the posting frames it. An org-level executive TITLE (VP / SVP / EVP / Chief X Officer / Head of <function>) is people management by definition → also REJECT. "This manager-level title is really strategic planning, not people management" is the exact rationalization to refuse — if there are reports, REJECT. The hands-on exception (this is a real match, NOT a reject) applies ONLY when the role owns schedules / budgets / vendors / deliverables with NO team to manage — a hands-on Production Manager / Line Producer is fine. VERDICT-BINDING (hard override — applies to EVERY test in this prompt): your verdict MUST follow your own reasoning. If your analysis identifies ANY enumerated REJECT condition — confirmed people-management (explicit hiring/performance-reviews/succession/direct-reports), a craft-executor or creative-craft/directing-ladder role, a game-development producer, CMS/web-build work, a role with NO production component, a different profession, or a HARD JD requirement the candidate cannot meet (e.g. "15+ years in X", a mandatory platform/domain the candidate lacks) — final_level = REJECT. Do NOT soften a disqualifier you yourself identified into WORTH_A_LOOK. WORTH_A_LOOK exists for exactly two cases: (1) genuinely adjacent-plausible — a production-transferable core with a SOFT gap; (2) genuinely ambiguous — the description is insufficient to decide (e.g. a leadership title with no direct-report evidence). Writing "outside the candidate's domain" / "not production-focused" / "critical gap" and then choosing WORTH_A_LOOK is a contradiction — resolve it as REJECT.

CRAFT-SPECIALIST TEST (run SECOND): the target role is a PROJECT-MANAGER-style PRODUCER — planning, budgeting, scheduling, coordinating crews/vendors, and delivering productions. It is NOT a hands-on creative specialist role; editing/animation/design tools on a profile count only as supervision-support skills. If the role's core deliverable is PERSONALLY EXECUTING a craft — Motion Designer / Motion Graphics / Animator / 3D Artist, Videographer / Camera Operator / Photographer, staff Video Editor, Graphic / Visual Designer / Illustrator, Sound / Audio Engineer, Copywriter, AI Artist / Prompt Artist / AI Content Creator (personally generating AI images/video/assets as the core deliverable), on-camera talent (Host / Presenter / Streamer / Livestream Host — the core is BEING the talent, not producing the show), "Content Creator" / "Shorts & Reels Creator" / TikTok-creator roles whose core is personally shooting/editing/posting the content, physical event labor (stagehand / set construction / strike crew / fabrication), non-production operations dressed in content language, or AI-output RATING / data-annotation / rubric-grading gigs (reviewing AI-generated content against rubrics — grading model output for a rating platform is a different profession, however media-adjacent the domain reads) — final_level = REJECT. The creative-craft LEADERSHIP ladder (Creative Director / Associate Creative Director / Art Director / Design Director) is equally not the candidate's → REJECT. "The candidate's skills map to this craft" is the exact rationalization to refuse — capable-of is not the candidate's job. HYBRID EXCEPTION (EVIDENCE-GATED): hybrid status requires the DESCRIPTION to show real coordination duties — schedules / budgets / vendors / crew booking / client management — as a substantial half; a slash-title or the word "producer"/"production"/"post-production" alone is NOT evidence, and if the day-to-day is operating the craft (edit suite, camera, OBS/streaming rig, being on camera) it is a craft REJECT. Genuine evidenced hybrids stay WORTH_A_LOOK — never upgrade a hybrid to HIGH_MATCH. AI-content roles centered on DIRECTING/QC of AI output or building AI content pipelines (not personally generating assets) are hybrids too — WORTH_A_LOOK at most.

MARKETING-EVENTS WALL (run THIRD): corporate-marketing event roles are a DIFFERENT PROFESSION, not a live-events production adjacency — final_level = REJECT. The family: Field Marketing (Manager/Specialist/Marketer), Marketing Events Specialist/Coordinator/Manager, tradeshow/conference-booth and sponsorship-activation roles, demand-gen event programs (webinars, executive dinners, lead capture/MQLs, booth-and-swag logistics), corporate Meeting Planner, fundraising/development-events roles, advertising coordination, and Design Program Manager / design-ops (running a design org's workflow). Their deliverable is leads, brand presence, or design throughput — not a produced show or content. A live-events/experiential background does NOT rescue these: that adjacency applies ONLY when the JD shows produced-show/content duties (run-of-show, staging, AV/broadcast production, filming, editorial output) — "it's an event role, and events are events" is the rationalization to refuse, and the WORTH_A_LOOK adjacency bullet does NOT apply to this family.

GOVERNMENT / DEFENSE WALL (run FOURTH): roles supporting government, defense, military, or homeland-security programs (contractor-staffed federal seminars/symposia, military education centers, defense IT firms) — final_level = REJECT even when the FUNCTION reads as event/production coordination — federal-contract environments are a different market from media production and are commonly clearance-gated. "Event coordination is event coordination" is the rationalization to refuse, and the WORTH_A_LOOK adjacency bullet does NOT apply to this family.

BELOW-LEVEL WALL (run FIFTH): coordinator-level roles (Production / Event / Content / Post-Production / Studio Coordinator and the like) and other entry-level roles (assistant roles other than Assistant Producer, trainee, or a JD whose experience requirement tops out at 0-2 years) are BELOW the target level and its pay band — final_level = REJECT. The "overqualified but strategic fit" exception in the seniority note does NOT apply to them, "DO NOT penalize under-leveled roles" does NOT apply to them, and the WORTH_A_LOOK adjacency bullet does NOT apply to them — a coordinator role at a well-known brand is still a coordinator role. Associate Producer is NOT entry-level and stays a legitimate target. A title that co-names a producer role ("Production Coordinator / Producer") is judged on the DESCRIPTION: if the day-to-day is supporting producers (scheduling, logistics, paperwork), it is coordinator-level. Assistant Producer titles (Assistant Producer, Assistant Video Producer) are judged on the DESCRIPTION the same way: if the day-to-day is supporting producers (scheduling, logistics, paperwork, errands, call sheets, travel, releases, set or backstage support), it is coordinator-level and final_level = REJECT; if the role owns production work itself (producing segments or shoots, booking talent or guests, running edits or post, managing a budget or vendors), it is NOT below the candidate's level and is graded like any producer role (the pay floor still applies).

Choose final_level:
- HIGH_MATCH: clear fit on role AND at the target level (hands-on IC: Producer / Senior Producer IC / Associate Producer / hands-on Production Manager). NEVER available to Head/Director/VP/Principal/Executive-Producer/people-management or account/client business titles.
- CONFIRMED_POTENTIAL: solid fit but unclear on one axis (exact medium or scope) — still a real match. Also not available to leadership/business titles.
- WORTH_A_LOOK: adjacent or a stretch — content-heavy but lighter on video, a hybrid editor/producer role, an over-leveled or AMBIGUOUS leadership TITLE (Principal / Lead / Executive Producer) whose DESCRIPTION does NOT establish direct reports (it may be an inflated title or a hands-on senior role) — for "Director" titles apply DIRECTOR DISAMBIGUATION: (a) org-seniority Director (strategy/teams/org budgets, 10+ yrs leadership demanded) → REJECT; (b) craft Director (film/video/art/creative — directs the creative work) → REJECT; (c) inflated title on evidenced hands-on schedules/budgets/vendors IC work → WORTH_A_LOOK; name which reading you concluded — (c) must be evidenced, never the default; a Director title with an EMPTY or stub description has NO evidence and CANNOT reach (c), and the general "insufficient description → ambiguous" branch does NOT apply to Director titles: no JD evidence → REJECT (readings (a) and (b) are REJECT, never 'capped': the leadership-title cap does NOT apply to Director titles, and naming reading (a)/(b) then choosing WORTH_A_LOOK is the contradiction VERDICT-BINDING forbids), account/client business hybrids with real production work, or a producer role at a games studio that is more game-development than content. Keep it visible in the lower tier; do NOT reject it. (A leadership title WITH confirmed reports is a REJECT, not a WORTH_A_LOOK — see the PEOPLE-MANAGEMENT TEST.)
- REJECT: a genuinely DIFFERENT job, OR a confirmed people-management / org-executive role (per the PEOPLE-MANAGEMENT TEST), OR a craft-specialist role (per the CRAFT-SPECIALIST TEST: motion design, videography, staff editing, graphic/visual design, audio engineering, copywriting — and the Creative Director / Art Director ladder). Reject if the role is fundamentally a video-GAME development producer (shipping the game itself), a technical/engineering producer or TPM, a music producer (recording artist), an insurance/financial/agricultural "producer," a pure social-media/demand-gen/business-development role with no production component, a role supporting government/defense/military/homeland-security programs (contractor-staffed federal seminars, military education centers, defense IT firms — a different market from media production and commonly clearance-gated, even when the function reads as event coordination), a corporate/field-marketing events, meeting-planning, fundraising-events, or design-program/design-ops role (per the MARKETING-EVENTS WALL — "event" in the title does not make it live-events production), a coordinator-level or other entry-level role (per the BELOW-LEVEL WALL), or its core is running / hiring / reviewing a team. A mere over-leveled TITLE with no direct-report evidence is NOT a reject — that is WORTH_A_LOOK.

Disambiguate on the DESCRIPTION, not the title: a real match shows production signal (shoots, edits, post, campaigns, content calendars, briefs, video/audio deliverables). Seniority is a TIER signal for over-leveled TITLES — an ambiguous Director / Principal / Lead with real hands-on production work and no direct reports goes to WORTH_A_LOOK, not REJECT; seniority has TWO hard-gates: confirmed people management (direct reports / hiring / headcount / org-executive title) is a REJECT, because the target level is individual-contributor with no direct reports; and a coordinator-level or other entry-level role is a REJECT per the BELOW-LEVEL WALL — the overqualified-at-a-great-brand exception does NOT lift it to WORTH_A_LOOK.

Output JSON:
{{"final_level": "HIGH_MATCH or CONFIRMED_POTENTIAL or WORTH_A_LOOK or REJECT", "reason": "...", "apply_priority": <1-10>}}"""


# === Claude-Optimized Prompt Variants ===
# Claude models perform best with XML-tagged structured input.
# These variants preserve the same logic but use <tags> instead of === HEADERS ===.

CASCADE_STAGE3_PROMPT_CLAUDE = """You are a senior executive recruiter reviewing a borderline job match.
A junior reviewer flagged this as POTENTIAL MATCH. Your job is to make the final call.

<candidate_profile>
{profile_text}
</candidate_profile>

<job_listing>
Title: {title}
Company: {company}
Location: {location}
{domain_hint}
Description: {description}
</job_listing>

<junior_assessment>
{stage2_reason}
</junior_assessment>

<seniority_note>
Associate titles (Associate Producer) are a legitimate seniority level — they are NOT "junior" and should NOT be downgraded. Coordinator-level roles and other entry-level roles (assistant roles other than Assistant Producer, trainee, a JD whose experience requirement tops out at 0-2 years) are BELOW the target level and its pay band: final_level = REJECT per the BELOW-LEVEL WALL, and the "overqualified but strategic fit" exception below does NOT apply to them. If Stage 2 flagged "overqualified but strategic fit," weigh company brand and domain alignment heavily. A junior title at a recognisable organisation in the target field with a strong domain match should NOT be downgraded. A step-down can be acceptable for the right organisation. This exception works DOWNWARD only — it never lifts Director/Head/VP-level or business roles above WORTH_A_LOOK.
</seniority_note>

<additional_context>
{perplexity_context}
</additional_context>

<instructions>
Sort this role into the RIGHT TIER for the candidate — do NOT decide whether the candidate is allowed to see it. The goal is recall with honest labeling, so DEFAULT TO KEEPING THE JOB VISIBLE.

LEVEL REALITY (overrides everything): the target level is the one stated in the candidate profile: hands-on individual-contributor (IC) production, with no direct reports. The STRONG zone is IC production roles at that level. DO NOT treat over-leveled roles as "strategic fit" — a Director role that demands many years of leadership experience is not a realistic target. DO NOT penalize under-leveled roles as "a step down" — an Associate/mid role at a good company IS the target.

PEOPLE-MANAGEMENT TEST (run FIRST, before any tier decision): a benign-looking title can hide a people-leadership role. Read the DESCRIPTION for direct-report language — "manage a team of", "lead/grow/build the team", "hire", "headcount", "direct reports", "people leadership", "manage managers", "performance reviews". If the role's core is people management, final_level = REJECT — the target level is individual-contributor with no direct reports, so a role built around running/hiring/reviewing a team is not viable for the candidate, however production-relevant it reads and however the posting frames it. An org-level executive TITLE (VP / SVP / EVP / Chief X Officer / Head of <function>) is people management by definition → also REJECT. "This manager-level title is really strategic planning, not people management" is the exact rationalization to refuse — if there are reports, REJECT. The hands-on exception (this is a real match, NOT a reject) applies ONLY when the role owns schedules / budgets / vendors / deliverables with NO team to manage — a hands-on Production Manager / Line Producer is fine. VERDICT-BINDING (hard override — applies to EVERY test in this prompt): your verdict MUST follow your own reasoning. If your analysis identifies ANY enumerated REJECT condition — confirmed people-management (explicit hiring/performance-reviews/succession/direct-reports), a craft-executor or creative-craft/directing-ladder role, a game-development producer, CMS/web-build work, a role with NO production component, a different profession, or a HARD JD requirement the candidate cannot meet (e.g. "15+ years in X", a mandatory platform/domain the candidate lacks) — final_level = REJECT. Do NOT soften a disqualifier you yourself identified into WORTH_A_LOOK. WORTH_A_LOOK exists for exactly two cases: (1) genuinely adjacent-plausible — a production-transferable core with a SOFT gap; (2) genuinely ambiguous — the description is insufficient to decide (e.g. a leadership title with no direct-report evidence). Writing "outside the candidate's domain" / "not production-focused" / "critical gap" and then choosing WORTH_A_LOOK is a contradiction — resolve it as REJECT.

CRAFT-SPECIALIST TEST (run SECOND): the target role is a PROJECT-MANAGER-style PRODUCER — planning, budgeting, scheduling, coordinating crews/vendors, and delivering productions. It is NOT a hands-on creative specialist role; editing/animation/design tools on a profile count only as supervision-support skills. If the role's core deliverable is PERSONALLY EXECUTING a craft — Motion Designer / Motion Graphics / Animator / 3D Artist, Videographer / Camera Operator / Photographer, staff Video Editor, Graphic / Visual Designer / Illustrator, Sound / Audio Engineer, Copywriter, AI Artist / Prompt Artist / AI Content Creator (personally generating AI images/video/assets as the core deliverable), on-camera talent (Host / Presenter / Streamer / Livestream Host — the core is BEING the talent, not producing the show), "Content Creator" / "Shorts & Reels Creator" / TikTok-creator roles whose core is personally shooting/editing/posting the content, physical event labor (stagehand / set construction / strike crew / fabrication), non-production operations dressed in content language, or AI-output RATING / data-annotation / rubric-grading gigs (reviewing AI-generated content against rubrics — grading model output for a rating platform is a different profession, however media-adjacent the domain reads) — final_level = REJECT. The creative-craft LEADERSHIP ladder (Creative Director / Associate Creative Director / Art Director / Design Director) is equally not the candidate's → REJECT. "The candidate's skills map to this craft" is the exact rationalization to refuse — capable-of is not the candidate's job. HYBRID EXCEPTION (EVIDENCE-GATED): hybrid status requires the DESCRIPTION to show real coordination duties — schedules / budgets / vendors / crew booking / client management — as a substantial half; a slash-title or the word "producer"/"production"/"post-production" alone is NOT evidence, and if the day-to-day is operating the craft (edit suite, camera, OBS/streaming rig, being on camera) it is a craft REJECT. Genuine evidenced hybrids stay WORTH_A_LOOK — never upgrade a hybrid to HIGH_MATCH. AI-content roles centered on DIRECTING/QC of AI output or building AI content pipelines (not personally generating assets) are hybrids too — WORTH_A_LOOK at most.

MARKETING-EVENTS WALL (run THIRD): corporate-marketing event roles are a DIFFERENT PROFESSION, not a live-events production adjacency — final_level = REJECT. The family: Field Marketing (Manager/Specialist/Marketer), Marketing Events Specialist/Coordinator/Manager, tradeshow/conference-booth and sponsorship-activation roles, demand-gen event programs (webinars, executive dinners, lead capture/MQLs, booth-and-swag logistics), corporate Meeting Planner, fundraising/development-events roles, advertising coordination, and Design Program Manager / design-ops (running a design org's workflow). Their deliverable is leads, brand presence, or design throughput — not a produced show or content. A live-events/experiential background does NOT rescue these: that adjacency applies ONLY when the JD shows produced-show/content duties (run-of-show, staging, AV/broadcast production, filming, editorial output) — "it's an event role, and events are events" is the rationalization to refuse, and the WORTH_A_LOOK adjacency bullet does NOT apply to this family.

GOVERNMENT / DEFENSE WALL (run FOURTH): roles supporting government, defense, military, or homeland-security programs (contractor-staffed federal seminars/symposia, military education centers, defense IT firms) — final_level = REJECT even when the FUNCTION reads as event/production coordination — federal-contract environments are a different market from media production and are commonly clearance-gated. "Event coordination is event coordination" is the rationalization to refuse, and the WORTH_A_LOOK adjacency bullet does NOT apply to this family.

BELOW-LEVEL WALL (run FIFTH): coordinator-level roles (Production / Event / Content / Post-Production / Studio Coordinator and the like) and other entry-level roles (assistant roles other than Assistant Producer, trainee, or a JD whose experience requirement tops out at 0-2 years) are BELOW the target level and its pay band — final_level = REJECT. The "overqualified but strategic fit" exception in the seniority note does NOT apply to them, "DO NOT penalize under-leveled roles" does NOT apply to them, and the WORTH_A_LOOK adjacency bullet does NOT apply to them — a coordinator role at a well-known brand is still a coordinator role. Associate Producer is NOT entry-level and stays a legitimate target. A title that co-names a producer role ("Production Coordinator / Producer") is judged on the DESCRIPTION: if the day-to-day is supporting producers (scheduling, logistics, paperwork), it is coordinator-level. Assistant Producer titles (Assistant Producer, Assistant Video Producer) are judged on the DESCRIPTION the same way: if the day-to-day is supporting producers (scheduling, logistics, paperwork, errands, call sheets, travel, releases, set or backstage support), it is coordinator-level and final_level = REJECT; if the role owns production work itself (producing segments or shoots, booking talent or guests, running edits or post, managing a budget or vendors), it is NOT below the candidate's level and is graded like any producer role (the pay floor still applies).

Choose final_level:
- HIGH_MATCH: clear fit on role AND at the target level (hands-on IC: Producer / Senior Producer IC / Associate Producer / hands-on Production Manager). NEVER available to Head/Director/VP/Principal/Executive-Producer/people-management or account/client business titles.
- CONFIRMED_POTENTIAL: solid fit but unclear on one axis (exact medium or scope) — still a real match. Also not available to leadership/business titles.
- WORTH_A_LOOK: adjacent or a stretch — content-heavy but lighter on video, a hybrid editor/producer role, an over-leveled or AMBIGUOUS leadership TITLE (Principal / Lead / Executive Producer) whose DESCRIPTION does NOT establish direct reports (it may be an inflated title or a hands-on senior role) — for "Director" titles apply DIRECTOR DISAMBIGUATION: (a) org-seniority Director (strategy/teams/org budgets, 10+ yrs leadership demanded) → REJECT; (b) craft Director (film/video/art/creative — directs the creative work) → REJECT; (c) inflated title on evidenced hands-on schedules/budgets/vendors IC work → WORTH_A_LOOK; name which reading you concluded — (c) must be evidenced, never the default; a Director title with an EMPTY or stub description has NO evidence and CANNOT reach (c), and the general "insufficient description → ambiguous" branch does NOT apply to Director titles: no JD evidence → REJECT (readings (a) and (b) are REJECT, never 'capped': the leadership-title cap does NOT apply to Director titles, and naming reading (a)/(b) then choosing WORTH_A_LOOK is the contradiction VERDICT-BINDING forbids), account/client business hybrids with real production work, or a producer role at a games studio that is more game-development than content. Keep it visible in the lower tier; do NOT reject it. (A leadership title WITH confirmed reports is a REJECT, not a WORTH_A_LOOK — see the PEOPLE-MANAGEMENT TEST.)
- REJECT: a genuinely DIFFERENT job, OR a confirmed people-management / org-executive role (per the PEOPLE-MANAGEMENT TEST), OR a craft-specialist role (per the CRAFT-SPECIALIST TEST: motion design, videography, staff editing, graphic/visual design, audio engineering, copywriting — and the Creative Director / Art Director ladder). Reject if the role is fundamentally a video-GAME development producer (shipping the game itself), a technical/engineering producer or TPM, a music producer (recording artist), an insurance/financial/agricultural "producer," a pure social-media/demand-gen/business-development role with no production component, a role supporting government/defense/military/homeland-security programs (contractor-staffed federal seminars, military education centers, defense IT firms — a different market from media production and commonly clearance-gated, even when the function reads as event coordination), a corporate/field-marketing events, meeting-planning, fundraising-events, or design-program/design-ops role (per the MARKETING-EVENTS WALL — "event" in the title does not make it live-events production), a coordinator-level or other entry-level role (per the BELOW-LEVEL WALL), or its core is running / hiring / reviewing a team. A mere over-leveled TITLE with no direct-report evidence is NOT a reject — that is WORTH_A_LOOK.

Disambiguate on the DESCRIPTION, not the title: a real match shows production signal (shoots, edits, post, campaigns, content calendars, briefs, video/audio deliverables). Seniority is a TIER signal for over-leveled TITLES — an ambiguous Director / Principal / Lead with real hands-on production work and no direct reports goes to WORTH_A_LOOK, not REJECT; seniority has TWO hard-gates: confirmed people management (direct reports / hiring / headcount / org-executive title) is a REJECT, because the target level is individual-contributor with no direct reports; and a coordinator-level or other entry-level role is a REJECT per the BELOW-LEVEL WALL — the overqualified-at-a-great-brand exception does NOT lift it to WORTH_A_LOOK.

Output JSON:
{{"final_level": "HIGH_MATCH or CONFIRMED_POTENTIAL or WORTH_A_LOOK or REJECT", "reason": "...", "apply_priority": <1-10>}}
</instructions>"""


# Resolve any {{profile.*}} markers once, at import.
CASCADE_STAGE1_PROMPT = fill(CASCADE_STAGE1_PROMPT)
CASCADE_STAGE1_PROMPT_CLAUDE = fill(CASCADE_STAGE1_PROMPT_CLAUDE)
CASCADE_STAGE3_PROMPT = fill(CASCADE_STAGE3_PROMPT)
CASCADE_STAGE3_PROMPT_CLAUDE = fill(CASCADE_STAGE3_PROMPT_CLAUDE)

# === Model-Family Prompt Selection ===
# Maps (stage, model_family) → prompt template.
# Claude models use XML-tagged prompts; others use === HEADER === style.

def detect_model_family(model: str) -> str:
    """Detect model family from model name string."""
    lower = model.lower()
    if lower.startswith("claude"):
        return "claude"
    elif lower.startswith("gemini"):
        return "gemini"
    elif lower.startswith("ollama/"):
        return "ollama"
    else:
        return "openai"

# Stage 2 prompt (full recruiter grading) — defined in gpt_filter.py as DEFAULT_PROMPT_TEMPLATE.
# Claude-optimized variant uses XML tags for structured input sections.
# The actual prompt text is stored in gpt_filter.py (STAGE2_PROMPT_CLAUDE, DEFAULT_PROMPT_TEMPLATE).
# This dict maps family → the prompt import name for auto-selection.

def get_stage3_prompt(model: str) -> str:
    """Return the best Stage 3 prompt for the given model."""
    family = detect_model_family(model)
    if family == "claude":
        return CASCADE_STAGE3_PROMPT_CLAUDE
    return CASCADE_STAGE3_PROMPT


# === ATS Company Boards ===
# All tokens verified live 2026-02-19. 153 companies across 12 ATS platforms.
# Not on any supported ATS: Google (proprietary), Meta (GraphQL+auth),
# BBC (Workday board unknown), Starbucks (Eightfold+CSRF), Bloomberg LP (Avature),
# Vice (Workday but 0 jobs), SiriusXM/Pandora (iCIMS), Audacy (iCIMS),
# Niantic (Greenhouse board API disabled), Wondery (Amazon Jobs proprietary),
# Studio71 (Lever board discontinued), Pushkin Industries (no supported ATS),
# Lemonada Media (no supported ATS), Brat TV (no supported ATS),
# Pereira O'Dell (custom form), Duncan Channon (LinkedIn only),
# Group Nine Media (merged into Vox Media, already listed).
# Hearst uses Oracle HCM Cloud (clean API, future scraper).
ATS_COMPANIES = [
    # ==================== GREENHOUSE (boards-api.greenhouse.io) ====================
    # --- Big Tech & Platforms ---
    {"company": "Airbnb", "ats": "greenhouse", "token": "airbnb"},
    {"company": "Stripe", "ats": "greenhouse", "token": "stripe"},
    {"company": "DoorDash", "ats": "greenhouse", "token": "doordashusa"},
    {"company": "Databricks", "ats": "greenhouse", "token": "databricks"},
    {"company": "Cloudflare", "ats": "greenhouse", "token": "cloudflare"},
    {"company": "Pinterest", "ats": "greenhouse", "token": "pinterest"},
    {"company": "Block", "ats": "greenhouse", "token": "block"},
    {"company": "Coinbase", "ats": "greenhouse", "token": "coinbase"},
    {"company": "Roblox", "ats": "greenhouse", "token": "roblox"},
    {"company": "Roku", "ats": "greenhouse", "token": "roku"},
    {"company": "Discord", "ats": "greenhouse", "token": "discord"},
    {"company": "Figma", "ats": "greenhouse", "token": "figma"},
    {"company": "Reddit", "ats": "greenhouse", "token": "reddit"},
    {"company": "Dropbox", "ats": "greenhouse", "token": "dropbox"},
    {"company": "Lyft", "ats": "greenhouse", "token": "lyft"},
    {"company": "Instacart", "ats": "greenhouse", "token": "instacart"},
    {"company": "Twitch", "ats": "greenhouse", "token": "twitch"},
    {"company": "LG", "ats": "greenhouse", "token": "lgelectronics"},
    # --- Growth-Stage Tech ---
    {"company": "Duolingo", "ats": "greenhouse", "token": "duolingo"},
    {"company": "Ripple", "ats": "greenhouse", "token": "ripple"},
    {"company": "Asana", "ats": "greenhouse", "token": "asana"},
    {"company": "Vercel", "ats": "greenhouse", "token": "vercel"},
    {"company": "Crunchyroll", "ats": "greenhouse", "token": "crunchyroll"},
    {"company": "Webflow", "ats": "greenhouse", "token": "webflow"},
    {"company": "Airtable", "ats": "greenhouse", "token": "airtable"},
    {"company": "Nextdoor", "ats": "greenhouse", "token": "nextdoor"},
    {"company": "Kickstarter", "ats": "greenhouse", "token": "kickstarter"},
    {"company": "Squarespace", "ats": "greenhouse", "token": "squarespace"},
    {"company": "Sprout Social", "ats": "greenhouse", "token": "sproutsocial"},
    {"company": "Hootsuite", "ats": "greenhouse", "token": "hootsuite"},
    {"company": "Braze", "ats": "greenhouse", "token": "braze"},
    {"company": "Beautiful.ai", "ats": "greenhouse", "token": "beautifulai"},
    {"company": "Descript", "ats": "greenhouse", "token": "descript"},
    {"company": "Medium", "ats": "greenhouse", "token": "medium"},
    {"company": "Cameo", "ats": "greenhouse", "token": "cameo"},
    # --- Media & Publishing ---
    {"company": "Vox Media", "ats": "greenhouse", "token": "voxmedia"},
    {"company": "The New York Times", "ats": "greenhouse", "token": "thenewyorktimes"},
    {"company": "Axios", "ats": "greenhouse", "token": "axios"},
    {"company": "BuzzFeed", "ats": "greenhouse", "token": "buzzfeed"},
    # --- Streaming & Studios ---
    {"company": "Universal Pictures", "ats": "greenhouse", "token": "universal"},
    {"company": "A24", "ats": "greenhouse", "token": "a24"},
    {"company": "Sony Pictures Animation", "ats": "greenhouse", "token": "sonypicturesanimation"},
    {"company": "Laika", "ats": "greenhouse", "token": "laika"},
    {"company": "Blur Studio", "ats": "greenhouse", "token": "blurstudio"},
    {"company": "Tubi", "ats": "greenhouse", "token": "tubitv"},
    {"company": "MasterClass", "ats": "greenhouse", "token": "masterclass"},
    {"company": "Penske Media", "ats": "greenhouse", "token": "pmc"},
    # --- Podcasts & Audio ---
    {"company": "Crooked Media", "ats": "greenhouse", "token": "crookedmedia"},
    {"company": "Sony Music", "ats": "greenhouse", "token": "sonymusicentertainment"},
    # --- Gaming ---
    {"company": "Riot Games", "ats": "greenhouse", "token": "riotgames"},
    {"company": "Epic Games", "ats": "greenhouse", "token": "epicgames"},
    {"company": "Take-Two Interactive", "ats": "greenhouse", "token": "taketwo"},
    {"company": "Rockstar Games", "ats": "greenhouse", "token": "rockstargames"},
    {"company": "2K Games", "ats": "greenhouse", "token": "2k"},
    {"company": "Nintendo", "ats": "greenhouse", "token": "nintendo"},
    {"company": "Bungie", "ats": "greenhouse", "token": "bungie"},
    {"company": "Scopely", "ats": "greenhouse", "token": "scopely"},
    {"company": "Pocket Gems", "ats": "greenhouse", "token": "pocketgems"},
    {"company": "Zynga", "ats": "greenhouse", "token": "zyngacareers"},
    {"company": "Manticore Games", "ats": "greenhouse", "token": "manticoregames"},
    {"company": "Mythical Games", "ats": "greenhouse", "token": "mythicalgames"},
    {"company": "Naughty Dog", "ats": "greenhouse", "token": "naughtydog"},
    {"company": "Unity Technologies", "ats": "greenhouse", "token": "unity3d"},
    {"company": "AppLovin", "ats": "greenhouse", "token": "applovin"},
    # --- Ad Agencies & Creative ---
    {"company": "VaynerMedia", "ats": "greenhouse", "token": "vaynermedia"},
    {"company": "Power Digital Marketing", "ats": "greenhouse", "token": "powerdigitalmarketing"},
    {"company": "WPP", "ats": "greenhouse", "token": "wpp"},
    {"company": "Ogilvy", "ats": "greenhouse", "token": "ogilvy"},
    {"company": "AKQA", "ats": "greenhouse", "token": "akqa"},
    {"company": "GSD&M", "ats": "greenhouse", "token": "gsdm"},
    {"company": "Weber Shandwick", "ats": "greenhouse", "token": "webershandwick"},
    {"company": "Huge", "ats": "greenhouse", "token": "hugeinc"},
    {"company": "Wieden+Kennedy", "ats": "greenhouse", "token": "wk"},
    {"company": "R/GA", "ats": "greenhouse", "token": "rga"},
    # --- Events & Live Entertainment ---
    {"company": "AEG", "ats": "greenhouse", "token": "aegworldwide"},
    {"company": "NVE Experience Agency", "ats": "greenhouse", "token": "nve"},
    {"company": "Golden State Warriors", "ats": "greenhouse", "token": "goldenstate"},
    # --- AI/ML & Tech In-House ---
    {"company": "Anthropic", "ats": "greenhouse", "token": "anthropic"},
    # Runway ML (greenhouse:runwayml) removed 2026-06-01 — dead token (404);
    # Runway is covered by the ashby:runway entry below.
    {"company": "Scale AI", "ats": "greenhouse", "token": "scaleai"},
    {"company": "Brex", "ats": "greenhouse", "token": "brex"},
    {"company": "Harness", "ats": "greenhouse", "token": "harnessinc"},
    {"company": "Domino Data Lab", "ats": "greenhouse", "token": "dominodatalab"},
    # --- MCN & Creator Studios ---
    {"company": "Spotter", "ats": "greenhouse", "token": "spotter"},
    # --- Digital Media & Production Studios ---
    {"company": "Fandom", "ats": "greenhouse", "token": "fandom"},
    {"company": "ATTN:", "ats": "greenhouse", "token": "attn"},
    {"company": "Media.Monks", "ats": "greenhouse", "token": "monks"},
    # --- Audio/Video Content ---
    {"company": "Calm", "ats": "greenhouse", "token": "calm"},
    {"company": "Headspace", "ats": "greenhouse", "token": "hs"},
    # --- Tech Content ---
    {"company": "Twilio", "ats": "greenhouse", "token": "twilio"},
    # --- Sports & Entertainment ---
    {"company": "MLB", "ats": "greenhouse", "token": "majorleaguebaseball"},
    {"company": "California Academy of Sciences", "ats": "greenhouse", "token": "californiaacademyofsciences"},
    # --- Other ---
    {"company": "Zipline", "ats": "greenhouse", "token": "FlyZipline"},
    {"company": "Nuro", "ats": "greenhouse", "token": "nuro"},
    {"company": "Watershed", "ats": "greenhouse", "token": "watershed"},
    {"company": "Glossier", "ats": "greenhouse", "token": "glossier"},

    # ==================== LEVER (api.lever.co) ====================
    {"company": "Spotify", "ats": "lever", "token": "spotify"},
    {"company": "Plaid", "ats": "lever", "token": "plaid"},
    {"company": "Illumination", "ats": "lever", "token": "illumination"},
    {"company": "Skydance", "ats": "lever", "token": "skydance"},
    {"company": "Warner Music Group", "ats": "lever", "token": "wmg"},
    # Jellysmack REMOVED 2026-07-18: Lever board discontinued (persistent 404;
    # was flagged in Phase 12c already and should have been dropped then).
    {"company": "Jam City", "ats": "lever", "token": "jamcity"},
    {"company": "The Athletic", "ats": "lever", "token": "theathletic"},
    {"company": "Instrument", "ats": "lever", "token": "instrument"},
    {"company": "Snappr", "ats": "lever", "token": "snappr"},
    {"company": "Superside", "ats": "lever", "token": "superside"},
    {"company": "Meeting Tomorrow", "ats": "lever", "token": "meetingtomorrow"},
    {"company": "System1", "ats": "lever", "token": "system1"},
    {"company": "Wpromote", "ats": "lever", "token": "wpromote"},

    # ==================== ASHBY (api.ashbyhq.com) ====================
    {"company": "OpenAI", "ats": "ashby", "token": "openai"},
    {"company": "Notion", "ats": "ashby", "token": "notion"},
    {"company": "Ramp", "ats": "ashby", "token": "ramp"},
    {"company": "Deel", "ats": "ashby", "token": "deel"},
    {"company": "Cursor", "ats": "ashby", "token": "cursor"},
    {"company": "Cohere", "ats": "ashby", "token": "cohere"},
    {"company": "Quora", "ats": "ashby", "token": "quora"},
    {"company": "Buffer", "ats": "ashby", "token": "buffer"},
    {"company": "Perplexity", "ats": "ashby", "token": "perplexity"},
    {"company": "Suno", "ats": "ashby", "token": "suno"},
    {"company": "Runway", "ats": "ashby", "token": "runway"},
    {"company": "ElevenLabs", "ats": "ashby", "token": "elevenlabs"},
    {"company": "Replit", "ats": "ashby", "token": "replit"},
    {"company": "Gimlet Media", "ats": "ashby", "token": "gimlet"},
    {"company": "Substack", "ats": "ashby", "token": "substack"},
    {"company": "Gamma", "ats": "ashby", "token": "gamma"},
    {"company": "Pika Labs", "ats": "ashby", "token": "pika"},
    {"company": "Patreon", "ats": "ashby", "token": "patreon"},

    # ==================== SMARTRECRUITERS (api.smartrecruiters.com) ====================
    {"company": "Bosch", "ats": "smartrecruiters", "token": "BoschGroup"},
    {"company": "Visa", "ats": "smartrecruiters", "token": "Visa"},
    {"company": "Skechers", "ats": "smartrecruiters", "token": "Skechers1"},
    {"company": "Equinox", "ats": "smartrecruiters", "token": "Equinox"},
    {"company": "McDonald's", "ats": "smartrecruiters", "token": "McDonaldsCorporation"},
    {"company": "IKEA", "ats": "smartrecruiters", "token": "InterIKEAGroup"},
    {"company": "LinkedIn", "ats": "smartrecruiters", "token": "LinkedIn3"},
    {"company": "SmartRecruiters", "ats": "smartrecruiters", "token": "SmartRecruiters"},
    {"company": "iHeartMedia", "ats": "smartrecruiters", "token": "iHeartMedia"},
    {"company": "Live Nation", "ats": "smartrecruiters", "token": "LiveNationEntertainment"},
    {"company": "Red Bull", "ats": "smartrecruiters", "token": "RedBull"},
    {"company": "LVMH", "ats": "smartrecruiters", "token": "LVMH"},
    {"company": "Omnicom", "ats": "smartrecruiters", "token": "OmnicomGroup"},
    {"company": "ServiceNow", "ats": "smartrecruiters", "token": "ServiceNow"},
    {"company": "Sportradar", "ats": "smartrecruiters", "token": "Sportradar"},

    # ==================== WORKDAY (*.myworkdayjobs.com POST API) ====================
    {"company": "Disney", "ats": "workday", "token": "https://disney.wd5.myworkdayjobs.com/disneycareer"},
    {"company": "Warner Bros. Discovery", "ats": "workday", "token": "https://warnerbros.wd5.myworkdayjobs.com/global"},
    {"company": "Sony Pictures", "ats": "workday", "token": "https://sonyglobal.wd1.myworkdayjobs.com/SonyPicturesEntertainmentCareers"},
    # Walmart REMOVED 2026-07-18: Workday tenant returns HTTP 410 Gone
    # (permanently retired path) on consecutive runs.
    {"company": "Salesforce", "ats": "workday", "token": "https://salesforce.wd12.myworkdayjobs.com/External_Career_Site"},
    {"company": "Adobe", "ats": "workday", "token": "https://adobe.wd5.myworkdayjobs.com/external_experienced"},
    {"company": "Target", "ats": "workday", "token": "https://target.wd5.myworkdayjobs.com/targetcareers"},
    {"company": "NVIDIA", "ats": "workday", "token": "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite"},
    {"company": "Intel", "ats": "workday", "token": "https://intel.wd1.myworkdayjobs.com/External"},
    {"company": "Activision Blizzard", "ats": "workday", "token": "https://activision.wd1.myworkdayjobs.com/External"},
    {"company": "NBCUniversal", "ats": "workday", "token": "https://comcast.wd5.myworkdayjobs.com/Comcast_Careers"},
    {"company": "Samsung", "ats": "workday", "token": "https://sec.wd3.myworkdayjobs.com/Samsung_Careers"},
    {"company": "Snap", "ats": "workday", "token": "https://snapchat.wd1.myworkdayjobs.com/snap"},
    {"company": "Nike", "ats": "workday", "token": "https://nike.wd1.myworkdayjobs.com/nke"},
    {"company": "Bloomberg Industry Group", "ats": "workday", "token": "https://bloomberg.wd1.myworkdayjobs.com/Bloombergindustrygroup_External_Career_Site"},
    {"company": "Condé Nast", "ats": "workday", "token": "https://condenast.wd5.myworkdayjobs.com/CondeCareers"},
    {"company": "Fox TV Stations (East)", "ats": "workday", "token": "https://fox.wd1.myworkdayjobs.com/FOXTVST_EAST"},
    {"company": "Fox TV Stations (West)", "ats": "workday", "token": "https://fox.wd1.myworkdayjobs.com/FOXTVST_WEST"},
    {"company": "Tastemade (Wonder Group)", "ats": "workday", "token": "https://wonder.wd1.myworkdayjobs.com/WG"},
    {"company": "Cisco", "ats": "workday", "token": "https://cisco.wd5.myworkdayjobs.com/Cisco_Careers"},
    {"company": "Autodesk", "ats": "workday", "token": "https://autodesk.wd1.myworkdayjobs.com/Ext"},
    {"company": "Universal Music Group", "ats": "workday", "token": "https://umusic.wd5.myworkdayjobs.com/UMGUS"},
    {"company": "Wasserman", "ats": "workday", "token": "https://teamwass.wd5.myworkdayjobs.com/WassermanCareers"},

    # ==================== TEAMTAILOR (*.teamtailor.com HTML) ====================
    # Klarna + Paradox Interactive removed 2026-06-09 — both tokens 404
    # (live-verified; Paradox moved to career.paradoxinteractive.com). The
    # scraper stays ready for future Teamtailor companies (page-1 only).

    # ==================== EIGHTFOLD.AI (explore.jobs.*.net REST API) ====================
    {"company": "Netflix", "ats": "eightfold", "token": "netflix.com"},

    # ==================== APPLE CAREERS (jobs.apple.com SSR hydration) ====================
    {"company": "Apple", "ats": "apple", "token": "en-us"},

    # ==================== BAMBOOHR ({subdomain}.bamboohr.com/careers/list) ====================
    # --- Creator Studios ---
    {"company": "Mythical Entertainment", "ats": "bamboohr", "token": "mythicalaccount"},
    {"company": "Smosh", "ats": "bamboohr", "token": "smosh"},
    # --- Gaming ---
    {"company": "Offworld Industries", "ats": "bamboohr", "token": "owi"},
    {"company": "Studio Wildcard", "ats": "bamboohr", "token": "studiowildcard"},
    {"company": "Wolcen Studio", "ats": "bamboohr", "token": "wolcenstudio"},
    # Lemon Sky Studios (bamboohr:lemonskystudios) removed 2026-06-01 — dead token
    # (HTTP 302 redirect away; company appears to have left BambooHR).
    # --- Animation & VFX ---
    {"company": "Flying Bark Productions", "ats": "bamboohr", "token": "flyingbark"},
    {"company": "Image Engine", "ats": "bamboohr", "token": "imageengine"},
    {"company": "Cinesite Vancouver", "ats": "bamboohr", "token": "cinesitevancouver"},

    # ==================== RECRUITEE ({token}.recruitee.com/api/offers) ====================
    {"company": "SOFTGAMES", "ats": "recruitee", "token": "softgames"},
    {"company": "11 bit studios", "ats": "recruitee", "token": "11bitstudios"},
    {"company": "Lucid Games", "ats": "recruitee", "token": "lucidgames"},
    {"company": "RocketWerkz", "ats": "recruitee", "token": "rocketwerkz"},

    # ==================== WORKABLE (apply.workable.com/api/v3/accounts/{token}/jobs) ====================
    {"company": "Moonbug Entertainment", "ats": "workable", "token": "moonbug-entertainment"},
    {"company": "TheSoul Publishing", "ats": "workable", "token": "thesoul-publishing-1"},
    {"company": "PRX", "ats": "workable", "token": "prx"},
    {"company": "Impact Theory", "ats": "workable", "token": "impact-theory"},
    {"company": "PikPok", "ats": "workable", "token": "pikpok"},
    {"company": "Zinc Network", "ats": "workable", "token": "zincnetwork"},
    {"company": "Founders Forum Group", "ats": "workable", "token": "foundersforum"},
    # Phase 16 — audit adds (agencies + audio/podcast; tokens verified live 2026-06-01)
    {"company": "NPR", "ats": "greenhouse", "token": "nationalpublicradioinc"},
    {"company": "Complex Networks", "ats": "lever", "token": "Complex"},
    {"company": "Goodby Silverstein & Partners", "ats": "greenhouse", "token": "goodbysilversteinpartners"},
    {"company": "TBWA\\Chiat\\Day", "ats": "greenhouse", "token": "tbwachiatday"},
    {"company": "Mekanism", "ats": "smartrecruiters", "token": "mekanism"},

    # ==================== Phase 17 — Widen-the-net source research ====================
    # 74 companies, all tokens LIVE-verified (HTTP 200 + >=1 posting) 2026-06-01 across
    # agencies/branded-content, big-tech & creator platforms, gaming/DTC, universities/
    # foundations/fintech/biotech, and AI startups. Off-target/dead candidates were
    # dropped at verification (see widen-net research log).
    # --- Agencies & branded-content production shops (Greenhouse) ---
    {"company": "MrBeast", "ats": "greenhouse", "token": "mrbeastyoutube"},
    {"company": "Forbes", "ats": "greenhouse", "token": "forbes"},
    {"company": "WITHIN", "ats": "greenhouse", "token": "agencywithin"},
    {"company": "Something Special Studios", "ats": "greenhouse", "token": "somethingspecialstudios"},
    {"company": "Code and Theory", "ats": "greenhouse", "token": "codeandtheory"},
    {"company": "Movers+Shakers", "ats": "greenhouse", "token": "moversshakers"},
    {"company": "Saylor Agency", "ats": "greenhouse", "token": "saylor"},
    {"company": "Hovercraft", "ats": "greenhouse", "token": "hovercraft"},
    {"company": "143 Studios", "ats": "greenhouse", "token": "143studiosinc"},
    {"company": "Whalar", "ats": "greenhouse", "token": "whalarinc"},
    {"company": "WPP Media (GroupM)", "ats": "greenhouse", "token": "wppmedia"},
    {"company": "Townsquare Media", "ats": "greenhouse", "token": "townsquaremedia"},
    {"company": "Low Battery (RapTV)", "ats": "lever", "token": "raptv"},
    {"company": "Framestore", "ats": "recruitee", "token": "framestore"},
    {"company": "Big Spaceship", "ats": "workable", "token": "bigspaceship"},
    # --- Big-tech / creator-economy / video platforms ---
    {"company": "Captions", "ats": "ashby", "token": "mirage"},
    {"company": "HeyGen", "ats": "greenhouse", "token": "heygen"},
    {"company": "Opus Clip", "ats": "ashby", "token": "opusclip"},
    {"company": "Kapwing", "ats": "lever", "token": "kapwing"},
    {"company": "Rumble", "ats": "greenhouse", "token": "rumble"},
    {"company": "Teachable", "ats": "greenhouse", "token": "teachablecareers"},
    {"company": "Thinkific", "ats": "greenhouse", "token": "thinkific"},
    {"company": "Udemy", "ats": "greenhouse", "token": "udemy"},
    {"company": "Coursera", "ats": "greenhouse", "token": "coursera"},
    {"company": "Upwork", "ats": "greenhouse", "token": "upwork"},
    {"company": "Whop", "ats": "greenhouse", "token": "whop"},
    {"company": "Beehiiv", "ats": "bamboohr", "token": "beehiiv"},
    {"company": "Mux", "ats": "ashby", "token": "mux"},
    # --- Gaming studios (content) + DTC/e-commerce content studios ---
    {"company": "Singularity6", "ats": "greenhouse", "token": "singularity6"},
    {"company": "Second Dinner", "ats": "ashby", "token": "seconddinner"},
    {"company": "Rothy's", "ats": "ashby", "token": "rothys"},
    {"company": "Brooklinen", "ats": "greenhouse", "token": "brooklinen"},
    {"company": "Faire", "ats": "greenhouse", "token": "faire"},
    {"company": "Thrive Market", "ats": "greenhouse", "token": "thrivemarket"},
    {"company": "Olipop", "ats": "greenhouse", "token": "olipop"},
    # --- Universities / foundations / corp-comms / fintech / biotech brand teams ---
    {"company": "Chan Zuckerberg Initiative", "ats": "greenhouse", "token": "chanzuckerberginitiative"},
    {"company": "Wikimedia Foundation", "ats": "greenhouse", "token": "wikimedia"},
    {"company": "Mozilla", "ats": "greenhouse", "token": "mozilla"},
    {"company": "Khan Academy", "ats": "greenhouse", "token": "khanacademy"},
    {"company": "Gates Ventures", "ats": "greenhouse", "token": "gatesventures"},
    {"company": "ProPublica", "ats": "greenhouse", "token": "propublica"},
    {"company": "Omidyar Network", "ats": "greenhouse", "token": "omidyarnetwork"},
    {"company": "Carta", "ats": "greenhouse", "token": "carta"},
    {"company": "Gusto", "ats": "greenhouse", "token": "gusto"},
    {"company": "Chime", "ats": "greenhouse", "token": "chime"},
    {"company": "Mercury", "ats": "greenhouse", "token": "mercury"},
    {"company": "Robinhood", "ats": "greenhouse", "token": "robinhood"},
    {"company": "Affirm", "ats": "greenhouse", "token": "affirm"},
    {"company": "SoFi", "ats": "greenhouse", "token": "sofi"},
    {"company": "Oscar Health", "ats": "greenhouse", "token": "oscar"},
    {"company": "Oura", "ats": "greenhouse", "token": "oura"},
    {"company": "Headway", "ats": "greenhouse", "token": "headway"},
    {"company": "Ginkgo Bioworks", "ats": "greenhouse", "token": "ginkgobioworks"},
    # --- AI startups + well-funded outlets hiring content/video/brand producers ---
    {"company": "Harvey", "ats": "ashby", "token": "harvey"},
    {"company": "Sierra", "ats": "ashby", "token": "sierra"},
    {"company": "Synthesia", "ats": "ashby", "token": "synthesia"},
    {"company": "Canva", "ats": "smartrecruiters", "token": "Canva"},
    {"company": "Cresta", "ats": "greenhouse", "token": "cresta"},
    {"company": "Mistral AI", "ats": "lever", "token": "mistral"},
    {"company": "Abridge", "ats": "ashby", "token": "abridge"},
    {"company": "Glean", "ats": "greenhouse", "token": "gleanwork"},
    {"company": "Together AI", "ats": "greenhouse", "token": "togetherai"},
    {"company": "Linear", "ats": "ashby", "token": "linear"},
    {"company": "Lovable", "ats": "ashby", "token": "lovable"},
    {"company": "Decagon", "ats": "ashby", "token": "decagon"},
    {"company": "Mercor", "ats": "ashby", "token": "mercor"},
    {"company": "Baseten", "ats": "ashby", "token": "baseten"},
    {"company": "Modal", "ats": "ashby", "token": "modal"},
    {"company": "Cognition", "ats": "ashby", "token": "cognition"},
    {"company": "Cartesia", "ats": "ashby", "token": "cartesia"},
    {"company": "Lambda", "ats": "ashby", "token": "lambda"},
    {"company": "Anyscale", "ats": "ashby", "token": "anyscale"},
    {"company": "Crusoe", "ats": "ashby", "token": "crusoe"},
    {"company": "Stability AI", "ats": "greenhouse", "token": "stabilityai"},
]
# Added 2026-02-19: Spotter (Greenhouse), Scopely (Greenhouse), Pocket Gems (Greenhouse),
# Jellysmack (Lever), Jam City (Lever), The Athletic (Lever), Instrument (Lever),
# Tastemade/Wonder Group (Workday)
# Added 2026-02-19 (batch 2): Zynga (Greenhouse), Fandom (Greenhouse), ATTN: (Greenhouse),
# Calm (Greenhouse), Headspace (Greenhouse), Twilio (Greenhouse), Patreon (Ashby),
# ServiceNow (SmartRecruiters), Cisco (Workday), Autodesk (Workday)
# Added 2026-02-22: Market research expansion — 27 new companies.
# Greenhouse +19: Tubi, MasterClass, Penske Media, Manticore Games, Mythical Games,
#   Naughty Dog, Unity Technologies, AppLovin, Power Digital Marketing, NVE Experience Agency,
#   Golden State Warriors, Runway ML, Scale AI, Brex, Harness, Domino Data Lab, Media.Monks,
#   MLB, California Academy of Sciences
# Lever +5: Snappr, Superside, Meeting Tomorrow, System1, Wpromote
# SmartRecruiters +1: Sportradar
# Workday +2: Universal Music Group, Wasserman

# === Phase 27c: "Is this posting real?" (facts, never a score) ===
# ATS platforms whose per-company fetch returns the WHOLE board. Only these
# record board fetches; workday/apple are seed keyword searches, teamtailor is
# a paginated HTML page — absence there proves nothing.
POSTING_CHECK_FULL_LIST_PLATFORMS = ("greenhouse", "lever", "ashby", "smartrecruiters",
                                     "recruitee", "workable", "eightfold", "bamboohr", "comeet")
POSTING_GONE_AFTER_FETCHES = 2        # complete fetches in a row without the posting
POSTING_FETCH_DROP_RATIO = 0.5        # a board shrinking below half of its previous ok fetch = no evidence once
POSTING_GONE_MIN_SPAN_HOURS = 6       # the missing fetches must span this long (a search right after the pipeline is not a 2nd look)
POSTING_CHURN_MIN_REFS = 5            # churn guard: only boards that listed at least this many postings last time
POSTING_CHURN_KEEP_RATIO = 0.5        # fewer than this share of last time's postings still there = new links, no evidence once
POSTING_LISTED_FRESH_DAYS = 7         # "still listed" only when the board's latest complete fetch is this recent
POSTING_ACTIVE_POSTED_DAYS = 14
POSTING_OLD_POSTED_DAYS = 30
POSTING_RELIST_CAUTION = 2
POSTING_CUTS_WINDOW_DAYS = 183        # layoffs / freeze / creative-team cuts dated within ~6 months
POSTING_PAY_WIDTH_RATIO = 1.0         # (max - min) > ratio * min: the top is more than double the bottom (0.5 flagged ordinary 1.5x corporate bands)
POSTING_QUALITY_MODEL = "claude-sonnet-5"
POSTING_QUALITY_EFFORT = "low"
POSTING_QUALITY_TTL_DAYS = 30
POSTING_QUALITY_MAX_PER_RUN = 25
POSTING_QUALITY_BUDGET_SEC = 300
POSTING_QUALITY_CALL_TIMEOUT_SEC = 60  # per read; the step ends within ~budget + this
POSTING_QUALITY_MIN_JD_CHARS = 300
POSTING_QUALITY_MAX_JD_CHARS = 12000
AUTO_HIRING_HEALTH = True             # independent of AUTO_ENRICH_MATCHES (False)
AUTO_HIRING_HEALTH_MAX_COMPANIES = 20
AUTO_HIRING_HEALTH_BUDGET_SEC = 480
