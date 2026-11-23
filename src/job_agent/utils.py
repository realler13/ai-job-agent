
import logging
import os
import random
from datetime import datetime
from rich.logging import RichHandler

def setup_logging(verbose=False):
    """
    Configures logging:
    - Console: INFO (or DEBUG if verbose) with Rich formatting.
    - File: DEBUG (always) to logs/run_TIMESTAMP.log.
    - Suppresses common noisy libraries.
    """

    # 1. Create logs directory
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_file = f"logs/run_{timestamp}.log"

    # 2. Define Formatters
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # 3. Create Handlers

    # Console Handler (Rich)
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    # Default to ERROR for ultra-clean CLI (only progress bars + critical errors)
    # Usage of --verbose can lower it to DEBUG
    console_handler.setLevel(logging.DEBUG if verbose else logging.ERROR)

    # File Handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # 4. Configure Root Logger
    logging.basicConfig(
        level=logging.NOTSET, # Allow all logical levels to pass to handlers
        handlers=[console_handler, file_handler],
        force=True
    )

    # 5. Suppress Noisy Libraries
    noisy_libs = [
        "httpcore", "httpx", "openai", "googleapiclient",
        "google_auth_httplib2", "urllib3", "oauth2client",
        "anthropic", "anthropic._base_client",
        "scrapling", "jobspy", "JobSpy", "camoufox", "tls_client", "websockets", "playwright",
        "JobSpy:Glassdoor", "JobSpy:Indeed", "JobSpy:Linkedin", "JobSpy:ZipRecruiter", "JobSpy:Google"
    ]
    for lib in noisy_libs:
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(logging.CRITICAL)
        lib_logger.propagate = True

    # 6. Return Project Logger
    logger = logging.getLogger("job_agent")
    logger.setLevel(logging.DEBUG) # Project logs captured fully

    return logger

def load_proxies(path="secrets/proxies.txt"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return []

def load_user_agents(path="secrets/user_agents.txt"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]

def get_job_hash(job):
    """
    Generates a unique hash string for a job dict based on Title, Company, Location.
    Normalized to lowercase.
    """
    t = str(job.get("title", "")).strip().lower()
    c = str(job.get("company", "")).strip().lower()
    l = str(job.get("location", "")).strip().lower()
    return f"{t}|{c}|{l}"


# ── Cross-source fuzzy dedup ────────────────────────────────────────────────
# Boards name the same employer differently: one lists "Acmes", another
# "Acme's Premium Goods & Supplies", a third "Acme's" (illustrative names).
# Exact title|company hashing sees several distinct jobs. Matching policy (shaped
# by a 3-lens adversarial review that refuted the first bare-prefix draft with
# Apple/Applebee's, Delta/Delta Dental, Co-Star->'star'):
#   * company names -> TOKEN tuples (apostrophes folded: Acme's == Acmes);
#     corporate suffixes stripped only at the END of the name (Co-Star and
#     Company 3 keep their identity).
#   * exact tuple equality = same employer (location-free, as the old hash).
#   * a whole-token prefix relation ('acmes' vs 'acmes premium goods
#     supplies') is only a match when LOCATIONS corroborate (equal
#     normalized, or one side blank) — Epic vs Epic Games in different cities
#     stays two employers; Applebee's never token-prefixes Apple at all.
# The LinkedIn copy wins a duplicate group (richer listing), but only
# location-compatibly, and it inherits fields the aggregator copy had that
# LinkedIn lacks (structured comp, date, location) so the comp floor keeps
# its deterministic numbers.

import re as _re

_COMPANY_SUFFIX_RE = _re.compile(
    r"(?:[\s.,&-]+(?:llc|inc|incorporated|corp|corporation|ltd|limited|co|company|gmbh)\.?)+\s*$",
    _re.IGNORECASE)


def company_key(name) -> str:
    """Company squashed for exact comparison: lowercase, apostrophes folded,
    TRAILING corporate suffixes stripped, alphanumerics only.
    "Acme's" -> 'acmes'; "Co-Star" -> 'costar' (mid-name 'co' survives)."""
    s = str(name or "").strip().lower().replace("'", "")
    s = _COMPANY_SUFFIX_RE.sub("", s)
    return _re.sub(r"[^a-z0-9]", "", s)


def company_tokens(name) -> tuple:
    """Company as a tuple of alphanumeric tokens (same normalization as
    company_key): "Acme's Premium Goods & Supplies" ->
    ('acmes', 'premium', 'goods', 'supplies')."""
    s = str(name or "").strip().lower().replace("'", "")
    s = _COMPANY_SUFFIX_RE.sub("", s)
    return tuple(_re.findall(r"[a-z0-9]+", s))


def location_key(location) -> str:
    """Location squashed for corroboration: lowercase, trailing country label
    stripped, alphanumerics only. 'Tacoma, WA, US' == 'Tacoma, WA'."""
    s = str(location or "").strip().lower()
    s = _re.sub(r",\s*(?:us|usa|united states)\s*$", "", s)
    return _re.sub(r"[^a-z0-9]", "", s)


def tokens_match(a: tuple, b: tuple):
    """'exact' when the token tuples are equal; 'fuzzy' when one is a
    WHOLE-TOKEN prefix of the other (brand vs full legal name) and the brand
    is >= 4 chars; None otherwise. ('apple',) vs ('applebees',) is None —
    prefix relations inside a token never match."""
    if not a or not b:
        return None
    if a == b:
        return "exact"
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if longer[:len(shorter)] == shorter and len("".join(shorter)) >= 4:
        return "fuzzy"
    return None


def companies_match(name_a, name_b, loc_a="", loc_b="") -> bool:
    """Same-employer test for two raw company names: exact token equality, or
    a token-prefix brand relation corroborated by compatible locations."""
    m = tokens_match(company_tokens(name_a), company_tokens(name_b))
    if m == "exact":
        return True
    if m == "fuzzy":
        la, lb = location_key(loc_a), location_key(loc_b)
        return not la or not lb or la == lb
    return False


def _is_linkedin_job(job) -> bool:
    link = str(job.get("job_url") or job.get("link") or "")
    return "linkedin.com" in link or str(job.get("site", "")).lower() == "linkedin"


_MERGE_FIELDS = ("location", "min_amount", "max_amount", "interval",
                 "currency", "job_type", "date_posted", "description")


def _blankish(v) -> bool:
    return v is None or str(v).strip().lower() in ("", "nan", "none")


def dedup_scraped_jobs(all_jobs, existing_links, existing_pairs):
    """Cross-source dedup of a scraped batch BEFORE the AI cascade.

    ``existing_links``: set of links already in the DB (pre-scrape snapshot).
    ``existing_pairs``: {title_lower: [(company_tokens, location_key), ...]}
    from the same snapshot. A job is a duplicate when its link is known, or a
    same-title row (DB or earlier in this batch) matches per tokens_match —
    'fuzzy' additionally demands location corroboration so a brand-prefix
    collision across cities keeps both jobs. Within the batch the LinkedIn
    copy REPLACES an already-kept aggregator copy (location-compatible only),
    inheriting any fields it lacks. Empty-company records (RSS feeds) rely on
    link dedup only — a bare title must not collapse distinct jobs."""
    new_jobs = []
    seen: dict = {}  # title_lower -> [{"tokens": [...], "locs": [...], "idx": int}]

    def _loc_ok(lkey, locs):
        return not lkey or not locs or lkey in locs

    for job in all_jobs:
        link = job.get("job_url") or job.get("link")
        title = str(job.get("title", "")).strip().lower()
        toks = company_tokens(job.get("company"))
        lkey = location_key(job.get("location"))

        if link and link in existing_links:
            continue
        if title and toks:
            dup_db = False
            for db_toks, db_loc in existing_pairs.get(title, ()):
                m = tokens_match(toks, db_toks)
                if m == "exact" or (m == "fuzzy" and (not lkey or not db_loc or lkey == db_loc)):
                    dup_db = True
                    break
            if dup_db:
                logging.debug(f"Cross-source dedup vs DB: '{job.get('title')}' at '{job.get('company')}'")
                continue
        group = None
        if title and toks:
            for g in seen.get(title, ()):
                for g_toks in g["tokens"]:
                    m = tokens_match(toks, g_toks)
                    if m == "exact" or (m == "fuzzy" and _loc_ok(lkey, g["locs"])):
                        group = g
                        break
                if group:
                    break
        if group is not None:
            kept = new_jobs[group["idx"]]
            kept_loc = location_key(kept.get("location"))
            if (_is_linkedin_job(job) and not _is_linkedin_job(kept)
                    and (not lkey or not kept_loc or lkey == kept_loc)):
                # LinkedIn wins — but keeps whatever the aggregator copy knew
                # that LinkedIn doesn't (structured comp, date, location).
                merged = dict(job)
                for f in _MERGE_FIELDS:
                    if _blankish(merged.get(f)) and not _blankish(kept.get(f)):
                        merged[f] = kept[f]
                logging.debug(f"Dedup: LinkedIn copy wins for '{job.get('title')}' at '{job.get('company')}'")
                new_jobs[group["idx"]] = merged
            group["tokens"].append(toks)
            if lkey:
                group["locs"].append(lkey)
            continue
        if title and toks:
            seen.setdefault(title, []).append(
                {"tokens": [toks], "locs": [lkey] if lkey else [], "idx": len(new_jobs)})
        new_jobs.append(job)
    return new_jobs

import pandas as pd
import re
from typing import Optional
from .config import (EXCLUDE_PHRASES, NEGATIVE_KEYWORDS,
                     NEGATIVE_KEYWORDS_INDUSTRIAL, NEGATIVE_KEYWORDS_TECH,
                     STALE_POSTING_DAYS)


def normalize_date_posted(raw_value) -> str:
    """Normalize date_posted from various sources to YYYY-MM-DD string.

    Handles:
    - Greenhouse: ISO datetime (e.g. "2026-01-15T10:30:00-05:00")
    - Lever: Unix timestamp in milliseconds as string (e.g. "1706745600000")
    - JobSpy: pandas Timestamp, date string, or NaT
    - None / empty → ""
    """
    if raw_value is None:
        return ""
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return ""
    if hasattr(raw_value, 'strftime'):
        # pandas Timestamp or datetime object
        try:
            if pd.isna(raw_value):
                return ""
            return raw_value.strftime("%Y-%m-%d")
        except Exception:
            return ""

    s = str(raw_value).strip()
    if not s or s.lower() in ("none", "nat", "nan", ""):
        return ""

    # Lever: pure digits, 10+ chars = Unix ms timestamp
    if s.isdigit() and len(s) >= 10:
        try:
            ts = int(s)
            if ts > 1e12:  # milliseconds
                ts = ts / 1000
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            return ""

    # ISO datetime or date string
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:26].rstrip("Z"), fmt.replace("%z", "")).strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            continue

    # Try pandas as last resort
    try:
        parsed = pd.to_datetime(s, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
    except Exception:
        pass

    return ""


def role_age_days(date_posted, first_seen=None, today=None) -> Optional[int]:
    """Days since the posting date (normalized), or since `first_seen` (the
    row's created_at, i.e. when we first saw the job) when the posting has no
    usable date. None when neither parses."""
    today = today or datetime.now().date()
    for raw in (normalize_date_posted(date_posted) if date_posted else "", str(first_seen or "")[:10]):
        try:
            return max(0, (today - datetime.strptime(raw, "%Y-%m-%d").date()).days)
        except (ValueError, TypeError):
            continue
    return None


def is_fresh_for_review(tier, date_posted, first_seen=None, today=None) -> bool:
    """True when a role is fresh enough for the Review list and the email:
    config.REVIEW_MAX_AGE_DAYS by tier. An unknown age passes (nothing to
    judge), and a tier with no limit always passes."""
    from .config import REVIEW_MAX_AGE_DAYS
    limit = REVIEW_MAX_AGE_DAYS.get(str(tier or ""))
    if limit is None:
        return True
    age = role_age_days(date_posted, first_seen, today)
    return age is None or age <= limit


def is_posting_old(date_str: str, days: int = 30) -> bool:
    """Check if a normalized YYYY-MM-DD date string is older than `days`."""
    if not date_str:
        return False
    try:
        posted = datetime.strptime(date_str, "%Y-%m-%d")
        return (datetime.now() - posted).days > days
    except (ValueError, TypeError):
        return False


# Exclude phrases that should NOT hard-kill a title when it also carries a
# producer/creative signal — they become editor/producer-hybrid or creative-PM
# "worth a look" candidates instead of being dropped before any AI sees them.
_RESCUABLE_EXCLUDES = {"video editor", "camera operator", "product manager"}


# ── Below-level titles ───────────────────────────────────────────────────────
# Coordinator / assistant / trainee titles are entry-level: they sit below the
# level the profile targets, so they are rejected. Decided on the TITLE alone,
# before any LLM cost: a title is below-level when its HEAD role is a
# coordinator / assistant / trainee. A slash/&/and hybrid that co-names a real
# producer / manager / supervisor / lead half ("Production Coordinator /
# Producer") is NOT decided here — the level is ambiguous, so the judges read
# the JD. "Coordinating Producer" (a senior TV title) never matches.
BELOW_LEVEL_REASON = "coordinator-level role — below the profile's target level"
BELOW_LEVEL_RULE = "COORDINATOR_LEVEL"

_HYBRID_SPLIT_RE = re.compile(r"\s*(?:/|&|\||\+|\band\b|\bor\b)\s*", re.I)
_PAREN_RE = re.compile(r"\([^)]*\)?")
_COORDINATOR_RE = re.compile(r"\bco-?ordinators?\b", re.I)
_TRAINEE_RE = re.compile(r"\btrainees?\b", re.I)
_ASSISTANT_RE = re.compile(r"\b(?:assistants?|asst)\b", re.I)
# "Assistant <rank>" is a management / academic rank, not an entry-level
# assistant ("Assistant Production Manager", "Assistant Director"). An
# "Assistant TO the Director" is still an assistant.
_ASSISTANT_RANK_RE = re.compile(
    r"\b(?:assistant|asst)\.?\s+(?:(?!to\b|for\b|of\b)[\w.'-]+\s+){0,2}"
    r"(?:manager|director|supervisor|lead|vice\s+president|vp|professor|dean|"
    r"superintendent|controller|principal|curator|counsel|attorney|coach)\b", re.I)
_ASSISTANT_TO_RE = re.compile(r"\b(?:assistants?|asst)\.?\s+to\b", re.I)
# "Assistant (Video) Producer" can be real production work or support chores.
# It is judged on the duties, not the title — so the title gate
# passes it to the judges, whose BELOW-LEVEL WALL rejects the support-only
# versions.
_ASSISTANT_PRODUCER_RE = re.compile(
    r"\b(?:assistant|asst)\.?\s+(?:(?!to\b|for\b|of\b)[\w.'-]+\s+){0,2}producers?\b", re.I)
_JUNIOR_RE = re.compile(r"\b(?:junior|jr)\b", re.I)
_ROLE_NOUN_RE = re.compile(
    r"\b(?:co-?ordinators?|producers?|managers?|supervisors?|leads?|directors?)\b", re.I)
_LEVEL_WORD_RE = re.compile(
    r"\b(?:co-?ordinators?|assistants?|asst|trainees?|producers?|managers?|"
    r"supervisors?|leads?|directors?)\b", re.I)
# "Coordinator, Production" / "Producer - Google Assistant": the chunk before
# the first comma or spaced dash carries the role; the rest is team / product.
_HEAD_CHUNK_SPLIT_RE = re.compile(r",|\s[-–—]\s")
# Dash / parenthesis producer hybrids are hybrids too (Phase 27 review): a
# spaced-dash tail or a parenthetical that names a PRODUCER role is a second
# half, judged exactly like the half after a slash — "Production Coordinator
# – Producer" and "Production Coordinator (Associate Producer)" go to the
# judges like "Production Coordinator / Producer" does. The half must NAME a
# producer role — its head noun (before any comma) is "Producer", optionally
# with a level ("Associate Producer II"); "(Producer Track)" or "- Executive
# Producer's Office" is a coordinator's team, not a second role. A producer
# half that is itself entry-level ("(Assistant Producer)", "- Junior
# Producer") adds nothing, so those titles stay rejected.
_DASH_TAIL_SPLIT_RE = re.compile(r"\s[-–—]\s")
_PRODUCER_HEAD_RE = re.compile(r"\bproducers?(?:\s+(?:[IVX]{1,4}|\d{1,2}))?\W*$", re.I)
# "Producer-Coordinator" / "Coordinator-Producer" (unspaced hyphen) reads as
# "Producer / Coordinator". Only the producer↔coordinator compound: a
# "Producer-Director" is one public-media title, not a hybrid.
_PRODUCER_COMPOUND_RE = re.compile(
    r"\b(producers?)-(?=co-?ordinators?\b)|\b(co-?ordinators?)-(?=producers?\b)", re.I)


def _names_producer_role(half: str) -> bool:
    return bool(_PRODUCER_HEAD_RE.search(half.split(",", 1)[0]))


def _producer_hybrid_halves(segment: str) -> list[str]:
    """Extra hybrid halves inside one slash-segment: each parenthetical and each
    spaced-dash tail that names a producer role."""
    halves = [m.group(0).strip("() ") for m in _PAREN_RE.finditer(segment)]
    tails = _DASH_TAIL_SPLIT_RE.split(_PAREN_RE.sub(" ", segment))[1:]
    return [h for h in halves + tails if _names_producer_role(h)]


def _segment_level(segment: str) -> Optional[str]:
    """'below' / 'target' / None for one half of a (possibly hybrid) title."""
    seg = _PAREN_RE.sub(" ", segment)
    first = _HEAD_CHUNK_SPLIT_RE.split(seg, maxsplit=1)[0]
    head = first if _LEVEL_WORD_RE.search(first) else seg
    if _TRAINEE_RE.search(head):
        return "below"
    if _ASSISTANT_PRODUCER_RE.search(head):
        return "target"
    if _ASSISTANT_RE.search(head) and not _ASSISTANT_RANK_RE.search(head):
        return "below"
    if _COORDINATOR_RE.search(head):
        # Head noun decides: "Lead Production Coordinator" is a coordinator,
        # "Project Coordinator Lead" leads them.
        nouns = _ROLE_NOUN_RE.findall(head)
        if _COORDINATOR_RE.fullmatch(nouns[-1]):
            return "below"
        return "target"
    if _ROLE_NOUN_RE.search(seg) and not _JUNIOR_RE.search(seg):
        return "target"
    return None


def below_level_title_reason(title) -> Optional[str]:
    """BELOW_LEVEL_REASON when the title is coordinator / assistant / trainee
    level, else None. Title-only and deterministic (no LLM)."""
    if not isinstance(title, str) or not title.strip():
        return None
    # "Assistant to Producer/Director": the slash sits inside what the candidate would
    # assist — no half of that title is a producer role.
    if _ASSISTANT_TO_RE.search(title):
        return BELOW_LEVEL_REASON
    title = _PRODUCER_COMPOUND_RE.sub(lambda m: (m.group(1) or m.group(2)) + " / ", title)
    halves = []
    for seg in _HYBRID_SPLIT_RE.split(title):
        if seg.strip():
            halves.append(seg)
            halves.extend(_producer_hybrid_halves(seg))
    levels = [_segment_level(h) for h in halves]
    if "below" in levels and "target" not in levels:
        return BELOW_LEVEL_REASON
    return None


def split_below_level_titles(df: pd.DataFrame):
    """(kept_df, rejected_df): pull coordinator-level titles out BEFORE any LLM
    call. Rejected rows carry the standard reject representation so the DB row
    says why (JUNIOR_WEAK, no tier, the reason as reasoning)."""
    if df is None or df.empty or "title" not in df.columns:
        return df, pd.DataFrame()
    mask = df["title"].apply(lambda t: below_level_title_reason(t) is not None)
    if not mask.any():
        return df, pd.DataFrame()
    rejected = df[mask].copy()
    rejected["match_level"] = "JUNIOR_WEAK"
    rejected["confidence"] = 0.9
    rejected["reason"] = f"Pre-filter: {BELOW_LEVEL_REASON}"
    rejected["rejection_rule"] = BELOW_LEVEL_RULE
    rejected["gpt_relevance"] = "irrelevant"
    rejected["gpt_reasoning"] = rejected["reason"]
    rejected["cascade_stage"] = 0
    logging.info(f"Pre-filter: {len(rejected)} coordinator-level title(s) rejected before AI grading")
    return df[~mask].copy(), rejected


# ── Above-level titles ───────────────────────────────────────────────────────
# Director / Head-of / VP / Chief titles sit above the level the profile
# targets. LLM judges tend to blend a Director REJECT rule with the softer
# leadership cap, so the TITLE decides, before any LLM cost, hybrids
# included: a title with a director half ("Producer/Director") is rejected
# like any other leadership title. Executive Producer (no director half)
# stays with the judges' leadership cap.
ABOVE_LEVEL_REASON = "director / head-of level role — above the profile's target level"
ABOVE_LEVEL_RULE = "LEADERSHIP_TITLE"
_LEADERSHIP_HEAD_RE = re.compile(
    r"\b(?:directors?|head\s+of|vice[\s-]+presidents?|s?vp|evp|chief)\b|^\s*head\b", re.I)
# A later comma / dash chunk that IS a leadership title is a second role, not a
# team name: "Executive Producer, Head of Production" (reading only the first
# chunk let it through), "Product Management, Director", "VP, Head of
# People". The whole chunk must be the title, so a team or program named after
# one stays: "Directors Program", "Director's Office", "Chief Marketing Office".
_LEADERSHIP_TAIL_RE = re.compile(
    r"^(?:(?:senior|sr\.?|executive|associate|deputy|managing)\s+)?"
    r"(?:director(?:\s+of\b.*)?|head\s+of\b.*|vice[\s-]+president\b.*|s?vp|evp|"
    r"chief\s+(?:[\w&-]+\s+){0,3}officer|chief\s+of\s+staff\b.*)\W*$", re.I)


def above_level_title_reason(title) -> Optional[str]:
    """ABOVE_LEVEL_REASON when the title's role is Director / Head of / VP /
    Chief (as the head chunk, or as a later comma / dash chunk that is a whole
    leadership title), else None."""
    if not isinstance(title, str) or not title.strip():
        return None
    for h in (seg for seg in _HYBRID_SPLIT_RE.split(title) if seg.strip()):
        head, *tails = _HEAD_CHUNK_SPLIT_RE.split(_PAREN_RE.sub(" ", h).strip())
        if _LEADERSHIP_HEAD_RE.search(head) or any(
                _LEADERSHIP_TAIL_RE.match(t.strip()) for t in tails):
            return ABOVE_LEVEL_REASON
    return None


# ── Producer + craft / coordinator hybrids ───────────────────────────────────
# A title that pairs the target role with a craft or coordinator job
# ("Producer / Camera Operator", "Motion Designer & Producer") is one the
# profile does not target. A producer half next to a craft or coordinator half
# is rejected before any LLM cost. Producer + Strategist / Manager / Lead stays.
HYBRID_CRAFT_REASON = "producer + craft / coordinator hybrid — not the role the profile targets"
HYBRID_CRAFT_RULE = "HYBRID_CRAFT_TITLE"
# The craft / coordinator nouns below are a generic list for a producer-type
# example profile; edit them for a profile with another target role.
_CRAFT_ROLE_RE = re.compile(
    r"\b(?:editors?|videographers?|shooters?|camera\s*(?:operators?|ops?)|photographers?|"
    r"co-?ordinators?|animators?|designers?|writers?|copywriters?|directors?|"
    r"cinematographers?|dps?)\W*$", re.I)
_NON_ROLE_WORDS_RE = re.compile(r"\b(?:of|for|to|the|at|with)\b", re.I)


def _names_craft_role(chunk: str) -> bool:
    """A short role phrase that ENDS in a craft / coordinator noun ("Video
    Editor", "Guest Coordinator", "Director") — not "Office of the Director"."""
    chunk = _PAREN_RE.sub(" ", chunk).strip()
    if not chunk or len(chunk.split()) > 4 or _NON_ROLE_WORDS_RE.search(chunk):
        return False
    return bool(_CRAFT_ROLE_RE.search(chunk))


def hybrid_craft_title_reason(title) -> Optional[str]:
    """HYBRID_CRAFT_REASON when the title co-names a producer role and a
    craft / coordinator / director role (slash, &, and, comma or dash list),
    else None."""
    if not isinstance(title, str) or not title.strip():
        return None
    chunks = []
    for seg in _HYBRID_SPLIT_RE.split(_PAREN_RE.sub(" ", title)):
        chunks.extend(c.strip() for c in _HEAD_CHUNK_SPLIT_RE.split(seg) if c.strip())
    has_producer = any(_names_producer_role(c) for c in chunks)
    has_craft = any(_names_craft_role(c) for c in chunks)
    return HYBRID_CRAFT_REASON if has_producer and has_craft else None


def split_above_level_titles(df: pd.DataFrame):
    """(kept_df, rejected_df): pull Director / Head-of / VP / Chief titles and
    producer + craft / coordinator hybrids out BEFORE any LLM call, with the
    standard reject representation."""
    if df is None or df.empty or "title" not in df.columns:
        return df, pd.DataFrame()
    above = df["title"].apply(lambda t: above_level_title_reason(t) is not None)
    hybrid = df["title"].apply(lambda t: hybrid_craft_title_reason(t) is not None) & ~above
    mask = above | hybrid
    if not mask.any():
        return df, pd.DataFrame()
    rejected = df[mask].copy()
    rejected["match_level"] = "IRRELEVANT"
    rejected["confidence"] = 0.9
    rejected["reason"] = [f"Pre-filter: {ABOVE_LEVEL_REASON}" if a else f"Pre-filter: {HYBRID_CRAFT_REASON}"
                          for a in above[mask]]
    rejected["rejection_rule"] = [ABOVE_LEVEL_RULE if a else HYBRID_CRAFT_RULE for a in above[mask]]
    rejected["gpt_relevance"] = "irrelevant"
    rejected["gpt_reasoning"] = rejected["reason"]
    rejected["cascade_stage"] = 0
    logging.info(f"Pre-filter: {int(above.sum())} director / head-of and {int(hybrid.sum())} "
                 f"producer-craft hybrid title(s) rejected before AI grading")
    return df[~mask].copy(), rejected


def process_and_filter_jobs(df: pd.DataFrame, suppressed_locations=None) -> pd.DataFrame:
    """
    Standardized pipeline for processing job dataframes:
    1. Deduplication (Smart)
    2. Exclusion Filtering
    3. Location/Remote Analysis (drops hard rejects + the candidate's suppressed locations)
    4. Drops internal columns before returning.

    ``suppressed_locations``: optional iterable of location strings the candidate has
    dismissed as no-go (from the candidate's feedback); jobs whose location matches are
    dropped. See db_manager.get_suppressed_locations().
    """
    if df.empty:
        return df

    # 1. Deduplication (Smart)
    df['norm_title'] = df['title'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    # Corporate suffixes differ per source for the SAME employer — one board's
    # "Example Technologies" vs another's "Example Technologies LLC" would
    # surface as two roles with one JD. Strip them BEFORE squashing to
    # alphanumerics.
    # Trailing-anchored suffix strip — the unanchored version turned
    # "Co-Star" into 'star' and "Company 3" into '3' (adversarial review).
    df['norm_company'] = (df['company'].astype(str).str.lower()
                          .str.replace(r'(?:[\s.,&-]+(?:llc|inc|incorporated|corp|corporation|ltd|limited|co|company|gmbh)\.?)+\s*$', '', regex=True)
                          .str.replace(r'[^a-z0-9]', '', regex=True))
    # Location joins the key normalized too: "Tacoma, WA, US" vs
    # "Tacoma, WA" is the same place, not two postings.
    df['norm_location'] = (df['location'].astype(str).str.lower()
                           .str.replace(r',\s*(?:us|usa|united states)\s*$', '', regex=True)
                           .str.replace(r'[^a-z0-9]', '', regex=True))

    # Dedupe by normalized title+company AND normalized location
    df.drop_duplicates(subset=["norm_title", "norm_company", "norm_location"], keep="first", inplace=True)

    # Also dedupe by Link if possible
    if "job_url" in df.columns:
         df.drop_duplicates(subset=["job_url"], keep="first", inplace=True)
    elif "link" in df.columns:
         df.drop_duplicates(subset=["link"], keep="first", inplace=True)

    # 1b. Normalize date_posted across all sources
    if "date_posted" in df.columns:
        df["date_posted"] = df["date_posted"].apply(normalize_date_posted)

    # 1c. Filter out closed/expired listings (LinkedIn, Indeed, etc.)
    if "description" in df.columns:
        closed_patterns = r"(?:no longer accepting applications|this job has expired)"
        closed_mask = df["description"].astype(str).str.contains(
            closed_patterns, case=False, na=False
        )
        df = df[~closed_mask].copy()

    # 1d. Drop STALE postings. is_posting_old() existed since Phase 6 but had
    # lost its call site, so months-old rows could reach the digest
    # (since fixed). Unknown dates pass (blank != stale).
    if "date_posted" in df.columns:
        stale_mask = df["date_posted"].apply(
            lambda d: is_posting_old(d, days=STALE_POSTING_DAYS))
        if stale_mask.any():
            logging.info(f"Pre-filter: dropping {int(stale_mask.sum())} stale "
                         f"posting(s) older than {STALE_POSTING_DAYS} days")
        df = df[~stale_mask].copy()

    # 2. Exclusions (with editor/producer-hybrid + creative-PM rescue)
    # A craft/adjacent title that ALSO shows a producer/creative signal is not
    # hard-killed here — it's let through so the cascade can tier it (usually
    # "worth a look") rather than dropping it pre-AI. Pure "Video Editor" still
    # drops; "Video Editor / Producer" and "Creative Product Manager" survive.
    title_s = df["title"].astype(str)
    exclude_pattern = "|".join([re.escape(p) for p in EXCLUDE_PHRASES])
    mask_title = title_s.str.contains(exclude_pattern, case=False, na=False)

    hard_excludes = [p for p in EXCLUDE_PHRASES if p not in _RESCUABLE_EXCLUDES]
    mask_hard = title_s.str.contains(
        "|".join([re.escape(p) for p in hard_excludes]), case=False, na=False
    )
    craft_hybrid = title_s.str.contains(r"video editor|camera operator", case=False, na=False) & \
        title_s.str.contains(r"(?i)\b(?:producer|producing|production)\b", regex=True, na=False)
    creative_pm = title_s.str.contains(r"product manager", case=False, na=False) & \
        title_s.str.contains(r"(?i)\b(?:content|creative|brand|studio|media|editorial)\b", regex=True, na=False)
    rescued = (~mask_hard) & (craft_hybrid | creative_pm)
    df = df[~(mask_title & ~rescued)].copy()

    # 2b. Negative keyword filter for production/ops/PM titles
    # "Production Manager", "Project Manager", "Program Manager" can match
    # manufacturing/industrial/tech roles. Matching is WORD-BOUNDARY based —
    # raw substrings were killing flagship roles ("civil" inside legal
    # boilerplate, "aws" inside "laws"). Rules:
    #   industrial term: one hit in title OR description kills
    #   tech term: a hit in the TITLE kills; the description needs 2+ DISTINCT
    #   tech terms (one passing tool mention must not kill a production role)
    neg_trigger_pattern = r"(?i)\b(?:production|operations|project\s+manager|program\s+manager)\b"
    is_neg_trigger_title = df["title"].astype(str).str.contains(neg_trigger_pattern, regex=True, na=False)
    if is_neg_trigger_title.any():
        ind_pattern = r"(?i)\b(?:" + "|".join(re.escape(nk) for nk in NEGATIVE_KEYWORDS_INDUSTRIAL) + r")\b"
        tech_pattern = r"(?i)\b(?:" + "|".join(re.escape(nk) for nk in NEGATIVE_KEYWORDS_TECH) + r")\b"
        title_s2 = df["title"].astype(str)
        desc_s = df["description"].astype(str) if "description" in df.columns else pd.Series("", index=df.index)

        has_ind = title_s2.str.contains(ind_pattern, regex=True, na=False) | \
            desc_s.str.contains(ind_pattern, regex=True, na=False)
        has_tech_title = title_s2.str.contains(tech_pattern, regex=True, na=False)
        tech_desc_2plus = desc_s.str.findall(tech_pattern).apply(
            lambda hits: len({h.lower() for h in hits}) >= 2
        )
        neg_mask = is_neg_trigger_title & (has_ind | has_tech_title | tech_desc_2plus)
        df = df[~neg_mask].copy()

    # 3. Location & Remote Logic — classify_location is the single gate.
    #    It drops ONLY hard rejects: a concrete on-site place outside the
    #    target area, or a remote role pinned to a region the candidate cannot
    #    work from. Target-area, remote and unknown/empty locations are kept
    #    (unknown/empty pass through for the AI to judge). Remote is read from
    #    the location, NOT the title, so a "Remote Operations" title at an
    #    on-site address is correctly treated as on-site. Also drops the
    #    candidate's suppressed locations.
    from .config import classify_location, is_location_suppressed
    if not df.empty:
        df['loc_status'] = df.apply(
            lambda r: classify_location(r.get('location', ''), r.get('title', '')),
            axis=1)
        df['is_remote'] = df['loc_status'] == 'remote'
        keep = df['loc_status'] != 'reject'
        if suppressed_locations:
            # EXACT segment match (not substring) so a stored place can't drop
            # a different place that merely shares text with it, and a
            # multi-location posting with a non-suppressed option is kept.
            # See is_location_suppressed.
            supp = {str(s).lower().strip() for s in suppressed_locations if s}
            keep &= ~df['location'].apply(lambda loc: is_location_suppressed(loc, supp))
        df = df[keep]

    # 4. Drop internal columns so they don't leak into exports
    internal_cols = ['norm_title', 'norm_company', 'norm_location', 'loc_status']
    df = df.drop(columns=[c for c in internal_cols if c in df.columns])

    return df
