import sqlite3
import json
import logging
import math
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any
from urllib.parse import parse_qs, urlparse

from .company_deriver import derive_company

# Setup logger
logger = logging.getLogger(__name__)

# ── Outcome ledger (Phase 27) ────────────────────────────────────────────────
# Forward Kanban stages, in order. Rejected / Ghosted / I Withdrew are
# outcomes, not stages a card "reached", so they carry no rank.
STAGE_RANK = {
    "To Apply": 0, "Applied": 1, "Screening Call": 2, "Interview": 3,
    "Final Interview": 4, "Offer": 5, "Accepted": 6,
}
# Stages where an interview is on the table (prep sheet, read-only rules).
INTERVIEW_STAGES = ("Screening Call", "Interview", "Final Interview", "Offer")
# "Reached interview" = Screening Call or later.
REACHED_INTERVIEW_STAGES = tuple(
    s for s, r in STAGE_RANK.items() if r >= STAGE_RANK["Screening Call"])
# The candidate's one-click answer to "Did you send this CV?"
SENT_CV_ANSWERS = ("as_is", "edited", "own")
# Event sources whose stage/date is a proxy, not something the candidate or the app
# recorded when it happened — the UI marks these with "~".
INFERRED_EVENT_SOURCES = ("backfill_prep_proxy",)
# Live card moves. Only these can be auto-corrected as an accidental drag —
# never a backfilled snapshot or a manually confirmed history entry.
LIVE_EVENT_TYPES = ("tracked", "status_change")
# Rows that set the card's status (live moves + the backfilled snapshot).
STATUS_EVENT_TYPES = ("tracked", "status_change", "snapshot")
# Closed cards. On one of these the candidate can say how far the process got and
# what happened (the outcome form on the Kanban card). "I Withdrew" = the candidate
# ended the process (2026-10-28).
TERMINAL_STATUSES = ("Rejected", "Ghosted", "I Withdrew")
WITHDREW_STATUS = "I Withdrew"
# What the candidate can name as the furthest stage of a closed card.
OUTCOME_STAGES = ("Applied", "Screening Call", "Interview", "Final Interview",
                  "Offer", "Accepted")
# Source of every ledger row the outcome form writes. Its own corrections are
# the only ones a later answer may undo.
OUTCOME_SOURCE = "outcome_form"
# "What happened" is one line in the candidate's words.
OUTCOME_NOTE_MAX = 500
# Dismissal reasons that name a location problem (feedback loop: the job's
# location is learned as a no-go). Shared by the HTTP and chat dismiss paths.
LOCATION_REASON_HINTS = (
    "location", "remote", "commute", "on-site", "on site", "onsite",
    "relocat", "too far", "far away", "distance", "not remote", "wrong location",
)


def _utc_now() -> str:
    """UTC 'YYYY-MM-DD HH:MM:SS' — the shape SQLite's CURRENT_TIMESTAMP writes,
    so ledger times sort and compare against the existing timestamp columns."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _placed_before(event: Dict) -> Optional[int]:
    """For an outcome-form stage entry: the id of the move that closed the
    card (Rejected / Ghosted / I Withdrew) it explains. The candidate's answer carries no
    date, so the card line shows it right before that move instead of at the
    time the candidate answered."""
    if event.get("source") != OUTCOME_SOURCE or event.get("event_type") != "stage_reach":
        return None
    try:
        target = json.loads(event.get("note") or "{}").get("placed_before")
    except (ValueError, AttributeError):
        return None
    return target if isinstance(target, int) else None


def _clean_text(value) -> Optional[str]:
    """A scraper string field as stored: None for missing / NaN (rows pass
    through a DataFrame, so a key only some scrapers set arrives as float NaN
    everywhere else) / blank."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and value.lower() != "nan" else None


def _clean_date_posted(value) -> Optional[str]:
    """Normalize a scraped date_posted for storage. str(NaN)=='nan' and
    str(None)=='None' are truthy — without this, rows were stored with those literal
    strings and rendered as 'nan' in the UI."""
    if value is None or (isinstance(value, float)):
        return None
    text = str(value).strip()
    if text.lower() in ("nan", "none", "nat", ""):
        return None
    return text


# ── "Is this posting real?" (Phase 27c) ──────────────────────────────────────
_GH_JOB_PATH_RE = re.compile(r"/jobs/(\d+)(?:/|$)")
# When one posting sits on two boards, the most reassuring answer wins:
# unavailable is not closed.
_LISTING_PRECEDENCE = ("listed", "relisted", "stale_check", "missing_once", "no_complete_fetch",
                       "gone")
_TITLE_KEY_RE = re.compile(r"[^0-9a-z]+")


def title_key(title) -> str:
    """Comparison form of a posting title: lower-case, every run of
    non-alphanumerics as one space. '' for empty. Pure, no I/O."""
    if not isinstance(title, str):
        return ""
    return _TITLE_KEY_RE.sub(" ", title.lower()).strip()


def posting_ref(link) -> str:
    """Stable identity of a posting link, used on BOTH sides (sightings and
    scrapes rows). Greenhouse: 'gh:<digits>' from a gh_jid= query param, else
    from a *.greenhouse.io /jobs/<digits> path — the same posting is linked as
    boards.greenhouse.io, job-boards.greenhouse.io or the company's own careers
    domain. Otherwise: host lower-cased without 'www.' + path without the
    trailing '/'; scheme, query and fragment dropped. '' for empty or
    unparseable. Pure, no I/O."""
    if not isinstance(link, str):
        return ""
    link = link.strip()
    if not link:
        return ""
    try:
        parts = urlparse(link if "://" in link else "//" + link)
        host = (parts.hostname or "").lower()
        query = parse_qs(parts.query)
    except ValueError:
        return ""
    if "." not in host:
        return ""
    jid = (query.get("gh_jid") or [""])[0].strip()
    if jid.isdigit():
        return f"gh:{jid}"
    if host == "greenhouse.io" or host.endswith(".greenhouse.io"):
        m = _GH_JOB_PATH_RE.search(parts.path)
        if m:
            return f"gh:{m.group(1)}"
    if host.startswith("www."):
        host = host[4:]
    return host + parts.path.rstrip("/")


def _parse_utc(ts: Optional[str]) -> Optional[datetime]:
    """A _utc_now()-shaped timestamp as an aware UTC datetime (None if not)."""
    if not ts:
        return None
    try:
        return datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _unknown_listing(reason: str) -> Dict[str, Any]:
    return {"status": "unknown", "reason": reason, "board_key": None, "checked_at": None,
            "gone_since": None, "first_seen_at": None, "last_seen_at": None,
            "relisted_link": None}


class DBManager:
    def __init__(self, db_path: str = "job_search_data.db"):
        self.db_path = db_path
        self._local = threading.local()
        self.init_db()

    def _get_thread_conn(self) -> sqlite3.Connection:
        """Get or create a connection for the current thread.

        Validates cached connections with a lightweight probe query.
        If the connection is stale (e.g. WAL files deleted while open),
        it is closed and replaced with a fresh one.
        """
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
            except sqlite3.OperationalError:
                logger.warning("Stale DB connection detected, reconnecting...")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                self._local.conn = None
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        # Always hand back a connection with the default (tuple) row factory.
        # Methods that want sqlite3.Row set it themselves; this guarantees a
        # prior method that raised mid-query before resetting can't leak its
        # row_factory onto the next caller on the same thread.
        conn.row_factory = None
        return conn

    def get_connection(self):
        """Legacy compatibility — returns thread-local connection."""
        return self._get_thread_conn()

    def close(self):
        """Close the connection for the current thread."""
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @contextmanager
    def transaction(self):
        """Context manager for atomic transactions.

        On I/O errors, invalidates the cached connection so the next
        call to _get_thread_conn() creates a fresh one.
        """
        conn = self._get_thread_conn()
        try:
            yield conn
            conn.commit()
        except sqlite3.OperationalError as e:
            if "disk I/O error" in str(e):
                logger.warning("Disk I/O error during transaction, invalidating connection")
                try:
                    conn.close()
                except Exception:
                    pass
                self._local.conn = None
            else:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        except Exception:
            conn.rollback()
            raise

    def init_db(self):
        """Initialize the database tables."""
        try:
            conn = self._get_thread_conn()
            cursor = conn.cursor()

            # Scrapes table: Stores raw scraped jobs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scrapes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    title TEXT,
                    company TEXT,
                    location TEXT,
                    link TEXT UNIQUE,
                    raw_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Applications table: Stores tracking info
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    status TEXT DEFAULT 'To Apply',
                    notes TEXT,
                    date_applied TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(job_id) REFERENCES scrapes(id)
                )
            """)

            # Company intelligence cache (Phase 15 — Perplexity enrichment)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS company_intel (
                    company_name TEXT PRIMARY KEY,
                    careers_url TEXT,
                    ats_type TEXT,
                    company_profile TEXT,
                    last_checked DATE,
                    last_found_roles INTEGER DEFAULT 0,
                    ignored BOOLEAN DEFAULT FALSE
                )
            """)

            # Chat history for the in-app Assistant (Phase 19)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT 'candidate',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Locations the candidate dismissed as no-go (feedback loop): future jobs
            # whose location matches are dropped by the pre-filter.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS location_suppressions (
                    location TEXT PRIMARY KEY,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_job_id INTEGER
                )
            """)
            # One row per location-driven dismissal: a location is blocked
            # only after LOCATION_SUPPRESS_MIN_VOTES of them (one is not enough).
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS location_votes (
                    location TEXT NOT NULL,
                    job_id INTEGER NOT NULL,
                    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (location, job_id)
                )
            """)
            # Warm-path layer (Spec M3): connections from the OFFICIAL LinkedIn
            # CSV self-exports (the only LinkedIn data this project touches).
            # owner = whose network (the CSV file's prefix; 'candidate' = the
            # candidate's own). company_norm feeds
            # the exact/fuzzy company matcher in warm_paths.py.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS warm_connections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner TEXT NOT NULL,
                    name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    email TEXT,
                    company TEXT,
                    company_norm TEXT,
                    position TEXT,
                    connected_on TEXT,
                    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(owner, url)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_warm_company ON warm_connections(company_norm)")
            # People intel (Spec M2): named contacts per job, with email
            # resolution state. Only email_status='verified' is ever sendable.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    company TEXT,
                    name TEXT NOT NULL,
                    title TEXT,
                    role_type TEXT,          -- recruiter | hiring_manager | peer
                    email TEXT,
                    email_status TEXT,       -- verified | pattern_guess | unknown
                    source TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(job_id, name)
                )
            """)

            # Outcome ledger (Phase 27): append-only history of every card.
            # A status change used to overwrite the old status, so a card that
            # reached interviews and was later Rejected left no trace. NO
            # foreign keys on purpose — events must survive the deletion of
            # their application or job (title/company are copied in).
            # effective_at = when it happened, NULL when unknown (never
            # guessed); a 'correction' event invalidates corrects_event_id
            # without deleting it.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS application_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_id INTEGER NOT NULL,
                    job_id INTEGER,
                    event_type TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT,
                    effective_at TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    note TEXT,
                    corrects_event_id INTEGER,
                    title TEXT,
                    company TEXT
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_app_events_app ON application_events(app_id, id)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_app_events_to_status ON application_events(to_status)")
            # Which CV the candidate actually sent. provenance: 'inferred' (the active
            # version at the Applied move) | 'confirmed' (the candidate answered the
            # one-click question) | 'unknown'. Append-only; the latest row wins.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS application_artifact_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_id INTEGER NOT NULL,
                    job_id INTEGER,
                    version_id INTEGER,
                    cv_path TEXT,
                    answer TEXT,
                    provenance TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifact_links_app ON application_artifact_links(app_id, id)")
            # Saved document versions: one row per generation, never overwritten.
            # scrapes.cv_path / cl_path stay as the "active version" projection.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    generation_id TEXT,
                    cv_path TEXT,
                    cl_path TEXT,
                    meta_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_versions_job ON document_versions(job_id, id)")

            # "Is this posting real?" (Phase 27c). One row per fetch of a
            # full-list ATS board. evidence=1 only for a complete, successful,
            # non-empty fetch that did not suddenly shrink — only those may
            # prove a posting is GONE. note: failed|partial|empty|drop|churn|NULL.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS board_fetches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    board_key TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    token TEXT NOT NULL,
                    company TEXT,
                    run_id TEXT,
                    fetched_at TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    complete INTEGER NOT NULL,
                    total_count INTEGER NOT NULL,
                    link_count INTEGER NOT NULL,
                    evidence INTEGER NOT NULL,
                    note TEXT
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_board_fetches_board "
                "ON board_fetches(board_key, evidence, id)")
            # Which postings each board listed, keyed by posting_ref(link) —
            # recorded from the FULL board list, before any keyword filter.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS board_sightings (
                    board_key TEXT NOT NULL,
                    ref TEXT NOT NULL,
                    link TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    last_seen_fetch_id INTEGER NOT NULL,
                    title_key TEXT,
                    PRIMARY KEY (board_key, ref)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_board_sightings_ref ON board_sightings(ref)")
            # title_key (db_manager.title_key of the board's own title): a
            # role re-posted under a new link is "still listed", not gone.
            _bs_cols = {r[1] for r in cursor.execute("PRAGMA table_info(board_sightings)")}
            if "title_key" not in _bs_cols:
                cursor.execute("ALTER TABLE board_sightings ADD COLUMN title_key TEXT")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_board_sightings_title "
                "ON board_sightings(board_key, title_key)")
            # Posting quality read (cite-or-omit facts from one cheap LLM call).
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS posting_quality (
                    job_id INTEGER PRIMARY KEY,
                    jd_hash TEXT NOT NULL,
                    model TEXT,
                    checked_at TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_link ON scrapes(link)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id, id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON applications(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_company_intel_ignored ON company_intel(ignored)")

            # Migrations
            self._migrate_scrapes_table(cursor)
            self._migrate_applications_table(cursor)

            # Stats-page aggregates: partial index over the tiny tiered subset
            # (a small fraction of all rows) makes the funnel/daily queries index-only.
            # MUST run after migrations — it references migrated columns (tier),
            # which don't exist yet on a fresh DB at table-creation time.
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scrapes_tiered
                ON scrapes(created_at, tier, reviewed, keyword_category)
                WHERE tier IN ('strong','worth_a_look')
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_app_date_applied ON applications(date_applied)")
            # /api/jobs?reviewed=... ORDER BY created_at scanned all rows with a
            # temp sort (seconds on a large table) — cover the filter+sort.
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_scrapes_reviewed_created ON scrapes(reviewed, created_at)")
            # Fresh planner statistics — without sqlite_stat1 the planner chose a
            # slow MULTI-INDEX OR plan for the Review feed over the fast
            # partial-index plan. Cheap even on a large DB.
            cursor.execute("ANALYZE scrapes")

            conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _migrate_applications_table(self, cursor):
        """Ensure outreach-intel columns exist on applications for existing DBs."""
        try:
            cursor.execute("PRAGMA table_info(applications)")
            columns = [info[1] for info in cursor.fetchall()]
            new_cols = {
                # pending | ready | failed | NULL (never attempted)
                "intel_status": "TEXT",
                "intel_path": "TEXT",
                "intel_generated_at": "TIMESTAMP",
                # Interview prep artifact (same lifecycle as intel), spawned on
                # the transition into an interview stage. pending|ready|failed|NULL
                "prep_status": "TEXT",
                "prep_path": "TEXT",
                "prep_generated_at": "TIMESTAMP",
                # The interview stage the current prep sheet was built for;
                # prep regenerates only when the card reaches a higher one.
                "prep_stage": "TEXT",
            }
            for col, dtype in new_cols.items():
                if col not in columns:
                    try:
                        cursor.execute(f"ALTER TABLE applications ADD COLUMN {col} {dtype}")
                        logger.info(f"Migrated applications table: Added {col}")
                    except Exception as e:
                        logger.warning(f"Applications migration failed for {col}: {e}")
        except Exception as e:
            logger.error(f"Applications migration check failed: {e}")

    def _migrate_scrapes_table(self, cursor):
        """Ensure new columns exist in scrapes table for existing DBs."""
        try:
            cursor.execute("PRAGMA table_info(scrapes)")
            columns = [info[1] for info in cursor.fetchall()]

            new_cols = {
                "cv_path": "TEXT",
                "cl_path": "TEXT",
                "cv_drive_id": "TEXT",
                "cl_drive_id": "TEXT",
                "relevance": "TEXT",
                "reasoning": "TEXT",
                "session_id": "TEXT",
                "confidence": "REAL",
                "match_level": "TEXT",
                "reviewed": "INTEGER DEFAULT 0",
                "date_posted": "TEXT",
                "keyword_category": "TEXT",
                "cv_quality_flag": "TEXT",
                "cv_audit_summary": "TEXT",
                "cv_generated_at": "TEXT",
                "cv_domain_used": "TEXT",
                "rejection_reason": "TEXT",
                "reviewed_at": "TIMESTAMP",
                # Tiered output (recall pass): strong | worth_a_look | NULL(reject).
                # Orthogonal to match_level; the web feed groups on this.
                "tier": "TEXT",
                # Per-job comp + employment type (2026-07-26): extracted
                # deterministically from jobspy structured fields / JD text
                # (comp_utils.extract_comp) at save time. Yearly-normalized
                # numbers enable sorting/filtering; comp_text is the display.
                "comp_text": "TEXT",
                "employment_type": "TEXT",
                "comp_yearly_min": "INTEGER",
                "comp_yearly_max": "INTEGER",
                # 'structured' (scraper fields) | 'text' (JD regex) | 'llm'
                # (opus fallback, displayed with an "(est.)" marker)
                "comp_source": "TEXT",
                # Employer's internal level when the posting publishes one
                # (2026-09-26: Apple detail pages carry "Machine Learning
                # ICT4"-style low/high titles). Display text, set by scrapers.
                "job_level": "TEXT",
                # Phase 27c listing status, denormalized by
                # refresh_listing_status(): 'listed' | 'gone' | NULL (unknown).
                # No backfill — pre-27c rows stay unknown until a board fetch
                # sees them.
                "listing_status": "TEXT",
                "listing_checked_at": "TEXT",
                "listing_gone_since": "TEXT",
            }

            added_tier = False
            for col, dtype in new_cols.items():
                if col not in columns:
                    try:
                        cursor.execute(f"ALTER TABLE scrapes ADD COLUMN {col} {dtype}")
                        logger.info(f"Migrated scrapes table: Added {col}")
                        if col == "tier":
                            added_tier = True
                    except Exception as e:
                        logger.warning(f"Migration failed for {col}: {e}")

            # One-time backfill so existing HIGH/POTENTIAL/USER_INPUT matches
            # surface in the new tiered feed the moment the column lands.
            if added_tier:
                try:
                    cursor.execute(
                        "UPDATE scrapes SET tier='strong' "
                        "WHERE tier IS NULL AND match_level IN ('HIGH_MATCH','USER_INPUT')"
                    )
                    cursor.execute(
                        "UPDATE scrapes SET tier='worth_a_look' "
                        "WHERE tier IS NULL AND match_level='POTENTIAL_MATCH'"
                    )
                    logger.info("Backfilled tier for existing HIGH/POTENTIAL/USER_INPUT rows")
                except Exception as e:
                    logger.warning(f"Tier backfill failed: {e}")
        except Exception as e:
            logger.error(f"Migration check failed: {e}")

    def get_job_id_by_link(self, link: str) -> Optional[int]:
        """Get the job ID for a given link, or None if not found."""
        if not link:
            return None
        try:
            conn = self._get_thread_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM scrapes WHERE link = ?", (link,))
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Error looking up job by link: {e}")
            return None

    def get_job_location(self, job_id: int) -> Optional[str]:
        """Return the stored location string for a job id, or None."""
        try:
            conn = self._get_thread_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT location FROM scrapes WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Error fetching location for job {job_id}: {e}")
            return None

    def job_exists(self, link: str) -> bool:
        """Check if a job with this link already exists in the database."""
        if not link:
            return False
        try:
            conn = self._get_thread_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM scrapes WHERE link = ?", (link,))
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking job existence: {e}")
            return False

    def update_job_relevance(self, job_id: int, relevance: str, reasoning: str = None):
        """Update job relevance status and optional reasoning."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                if reasoning:
                    cursor.execute("UPDATE scrapes SET relevance = ?, reasoning = ? WHERE id = ?",
                                   (relevance, reasoning, job_id))
                else:
                    cursor.execute("UPDATE scrapes SET relevance = ? WHERE id = ?",
                                   (relevance, job_id))
                return True
        except Exception as e:
            logger.error(f"Error updating relevance for job {job_id}: {e}")
            return False

    def update_job_match_result(self, job_id: int, match_level: str, confidence: float,
                                reason: str = None, rejection_rule: str = None):
        """Update job with structured match result from JobMatcher."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                # Map match_level to legacy relevance
                relevance = "relevant" if match_level in ("HIGH_MATCH", "POTENTIAL_MATCH") else "irrelevant"
                cursor.execute("""
                    UPDATE scrapes
                    SET match_level = ?, confidence = ?, relevance = ?, reasoning = ?
                    WHERE id = ?
                """, (match_level, confidence, relevance, reason, job_id))
                return True
        except Exception as e:
            logger.error(f"Error updating match result for job {job_id}: {e}")
            return False

    def _fill_merged_fields(self, cursor, job_id: int, job_data: dict) -> None:
        """Fill pay and posting date the stored row LACKS from the graded job.
        The graded job can be a cross-source merge: the LinkedIn copy inherits
        pay / date / location from an aggregator copy in memory
        (utils._MERGE_FIELDS), but its DB row was saved during scraping, before
        the merge. Without this the pay floor and the freshness rule never saw
        what the digest saw (a role under the pay floor left the email but
        stayed in the Review list). Never overwrites stored values."""
        try:
            from .comp_utils import extract_comp
            comp = extract_comp(job_data)
            if comp.get("comp_text"):
                cursor.execute("""
                    UPDATE scrapes
                    SET comp_text = ?, employment_type = COALESCE(employment_type, ?),
                        comp_yearly_min = ?, comp_yearly_max = ?, comp_source = ?
                    WHERE id = ? AND (comp_text IS NULL OR comp_text = '')
                """, (comp["comp_text"], comp["employment_type"], comp["comp_yearly_min"],
                      comp["comp_yearly_max"], comp["comp_source"], job_id))
            date_posted = _clean_date_posted(job_data.get("date_posted"))
            if date_posted:
                cursor.execute("""
                    UPDATE scrapes SET date_posted = ?
                    WHERE id = ? AND (date_posted IS NULL OR date_posted IN ('', 'nan', 'None', 'NaT'))
                """, (date_posted, job_id))
        except Exception as e:
            logger.warning(f"Could not fill merged pay / date for job {job_id}: {e}")

    def save_job_with_relevance(self, job_data: Dict[str, Any], relevance: str = None,
                                reasoning: str = None, match_level: str = None,
                                confidence: float = None, session_id: str = None,
                                keyword_category: str = None, tier: str = None) -> Optional[int]:
        """Save a job and its relevance data in a single transaction."""
        try:
            link = job_data.get("job_url") or job_data.get("link")
            if not link:
                return None

            # pandas rows leak float('nan') into these fields (row.to_dict()).
            # NaN is truthy, passes the "is not None" guard below, and lands as
            # a NaN REAL in SQLite (breaks =/IS NULL filters and JSON export) —
            # normalize to None up front.
            def _denan(v):
                return None if isinstance(v, float) and math.isnan(v) else v
            relevance = _denan(relevance)
            reasoning = _denan(reasoning)
            match_level = _denan(match_level)
            confidence = _denan(confidence)
            tier = _denan(tier)

            # Allow keyword_category / tier from job_data dict as fallback
            kw_cat = _denan(keyword_category or job_data.get("keyword_category"))
            tier_val = tier or _denan(job_data.get("tier"))

            with self.transaction() as conn:
                cursor = conn.cursor()

                # Check if exists
                cursor.execute("SELECT id FROM scrapes WHERE link = ?", (link,))
                row = cursor.fetchone()
                if row:
                    job_id = row[0]
                    # Update relevance on existing job — guard on match_level
                    # (more reliable than relevance, which is a legacy compat field).
                    # tier uses COALESCE so a None never wipes an existing tier.
                    if match_level is not None or relevance is not None:
                        if kw_cat:
                            cursor.execute("""
                                UPDATE scrapes
                                SET relevance = ?, reasoning = ?, match_level = ?, confidence = ?,
                                    keyword_category = ?, tier = COALESCE(?, tier)
                                WHERE id = ?
                            """, (relevance, reasoning, match_level, confidence, kw_cat, tier_val, job_id))
                        else:
                            cursor.execute("""
                                UPDATE scrapes
                                SET relevance = ?, reasoning = ?, match_level = ?, confidence = ?,
                                    tier = COALESCE(?, tier)
                                WHERE id = ?
                            """, (relevance, reasoning, match_level, confidence, tier_val, job_id))
                    self._fill_merged_fields(cursor, job_id, job_data)
                    return job_id

                # Insert
                # Recover company from ATS URL / title suffix when scraper returned NaN/empty
                # (common for Indeed-fronted Ashby postings, EntertainmentCareers RSS).
                derived_company = derive_company(job_data)
                raw_json = json.dumps(job_data, default=str)
                date_posted = _clean_date_posted(job_data.get("date_posted"))
                from .comp_utils import extract_comp
                comp = extract_comp(job_data)
                cursor.execute("""
                    INSERT INTO scrapes (date, title, company, location, link, raw_data,
                                         session_id, relevance, reasoning, match_level, confidence,
                                         date_posted, keyword_category, tier,
                                         comp_text, employment_type,
                                         comp_yearly_min, comp_yearly_max, comp_source, job_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().strftime("%Y-%m-%d"),
                    job_data.get("title"),
                    derived_company,
                    job_data.get("location"),
                    link,
                    raw_json,
                    session_id,
                    relevance,
                    reasoning,
                    match_level,
                    confidence,
                    date_posted,
                    kw_cat,
                    tier_val,
                    comp["comp_text"],
                    comp["employment_type"],
                    comp["comp_yearly_min"],
                    comp["comp_yearly_max"],
                    comp["comp_source"],
                    _clean_text(job_data.get("job_level")),
                ))
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving job with relevance {job_data.get('title')}: {e}")
            return None

    def update_job_docs(self, job_id: int, cv_path: str = None, cl_path: str = None,
                        cv_drive_id: str = None, cl_drive_id: str = None):
        """Update job with generated document paths and IDs."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                updates = []
                params = []

                if cv_path:
                    updates.append("cv_path = ?")
                    params.append(cv_path)
                if cl_path:
                    updates.append("cl_path = ?")
                    params.append(cl_path)
                if cv_drive_id:
                    updates.append("cv_drive_id = ?")
                    params.append(cv_drive_id)
                if cl_drive_id:
                    updates.append("cl_drive_id = ?")
                    params.append(cl_drive_id)

                if not updates:
                    logger.warning(f"update_job_docs({job_id}): no paths to update (cv={cv_path!r}, cl={cl_path!r})")
                    return True

                params.append(job_id)
                query = f"UPDATE scrapes SET {', '.join(updates)} WHERE id = ?"

                cursor.execute(query, tuple(params))
                if cursor.rowcount == 0:
                    logger.warning(f"update_job_docs({job_id}): UPDATE matched 0 rows — job_id may not exist")
                else:
                    logger.debug(f"update_job_docs({job_id}): stored cv={cv_path!r}, cl={cl_path!r}")
                return True
        except Exception as e:
            logger.error(f"Error updating doc paths for job {job_id}: {e}")
            return False

    def update_job_cv_quality(self, job_id: int, flag: str, audit_summary: str = None,
                              domain_used: str = None) -> bool:
        """Update job with CV quality audit results."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE scrapes
                    SET cv_quality_flag = ?, cv_audit_summary = ?,
                        cv_generated_at = ?, cv_domain_used = ?
                    WHERE id = ?
                """, (flag, audit_summary, datetime.now().isoformat(), domain_used, job_id))
                return True
        except Exception as e:
            logger.error(f"Error updating CV quality for job {job_id}: {e}")
            return False

    def save_job(self, job_data: Dict[str, Any], session_id: str = None,
                 keyword_category: str = None) -> Optional[int]:
        """
        Save a scraped job to the database.
        Returns the job ID if successful (or existing ID), or None on error.
        """
        try:
            link = job_data.get("job_url") or job_data.get("link")
            if not link:
                return None

            # Allow keyword_category from job_data dict as fallback
            kw_cat = keyword_category or job_data.get("keyword_category")

            with self.transaction() as conn:
                cursor = conn.cursor()

                # Check if exists
                cursor.execute("SELECT id FROM scrapes WHERE link = ?", (link,))
                row = cursor.fetchone()
                if row:
                    return row[0]

                # Insert
                # Recover company from ATS URL / title suffix when scraper returned NaN/empty.
                derived_company = derive_company(job_data)
                raw_json = json.dumps(job_data, default=str)
                date_posted = _clean_date_posted(job_data.get("date_posted"))
                # Per-job comp + employment type (deterministic, no LLM)
                from .comp_utils import extract_comp
                comp = extract_comp(job_data)
                cursor.execute("""
                    INSERT INTO scrapes (date, title, company, location, link, raw_data,
                                         session_id, date_posted, keyword_category,
                                         comp_text, employment_type,
                                         comp_yearly_min, comp_yearly_max, comp_source, job_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().strftime("%Y-%m-%d"),
                    job_data.get("title"),
                    derived_company,
                    job_data.get("location"),
                    link,
                    raw_json,
                    session_id,
                    date_posted,
                    kw_cat,
                    comp["comp_text"],
                    comp["employment_type"],
                    comp["comp_yearly_min"],
                    comp["comp_yearly_max"],
                    comp["comp_source"],
                    _clean_text(job_data.get("job_level")),
                ))
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving job {job_data.get('title')}: {e}")
            return None

    def track_job(self, job_id: int, status: str = "To Apply", notes: str = "",
                  source: str = "app") -> bool:
        """Add a job to the tracking (applications) table. The initial ledger
        event is written in the same transaction."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()

                # Check if already tracked
                cursor.execute("SELECT id FROM applications WHERE job_id = ?", (job_id,))
                if cursor.fetchone():
                    return True  # Already tracked

                cursor.execute("""
                    INSERT INTO applications (job_id, status, notes, date_applied)
                    VALUES (?, ?, ?, ?)
                """, (job_id, status, notes, datetime.now().strftime("%Y-%m-%d")))
                self._write_status_event(cursor, cursor.lastrowid, job_id, None, status,
                                         source, event_type="tracked")
                return True
        except Exception as e:
            logger.error(f"Error tracking job {job_id}: {e}")
            return False

    def get_history(self, limit: int = 1000) -> List[Dict]:
        """Get scraped job history."""
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, date, title, company, location, link, created_at,
                       relevance, reasoning, match_level, confidence, date_posted,
                       keyword_category
                FROM scrapes
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            conn.row_factory = None
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching history: {e}")
            return []

    def get_session_jobs(self, session_id: str) -> List[Dict]:
        """Get all jobs from a specific session."""
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM scrapes
                WHERE session_id = ?
                ORDER BY created_at DESC
            """, (session_id,))
            rows = cursor.fetchall()
            conn.row_factory = None
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching session jobs: {e}")
            return []

    def get_session_results(self, session_id: str) -> List[Dict]:
        """Get display-ready results for a session (no raw_data)."""
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, date, title, company, location, link, created_at,
                       relevance, reasoning, match_level, confidence, date_posted,
                       keyword_category, session_id, reviewed, cv_path, cl_path,
                       cv_quality_flag, cv_audit_summary, cv_domain_used, rejection_reason, tier,
                       comp_text, employment_type, comp_yearly_min, comp_yearly_max, job_level
                FROM scrapes
                WHERE session_id = ?
                ORDER BY confidence DESC NULLS LAST, created_at DESC
            """, (session_id,))
            rows = cursor.fetchall()
            conn.row_factory = None
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching session results: {e}")
            return []

    def get_latest_session_id(self) -> Optional[str]:
        """Get the most recent session_id from the database."""
        try:
            conn = self._get_thread_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id FROM scrapes
                WHERE session_id IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Error fetching latest session_id: {e}")
            return None

    def get_unreviewed_jobs(self) -> List[Dict]:
        """Get relevant jobs that haven't been reviewed yet.

        Includes a server-side parsed ``description`` (the JD lives inside the
        raw_data JSON — the review page rendered an empty Description panel
        because the SELECT never shipped it). Within each tier, fresher
        postings sort first.

        The WHERE is deliberately index-friendly: the former
        ``OR match_level IN (...)`` legacy clause (pre-tiering rows) defeated
        the partial index idx_scrapes_tiered and forced a full-table scan +
        temp sort (seconds per Review-page load). `COALESCE(reviewed,0) = 0` instead of `reviewed IS NULL OR
        reviewed = 0`: the OR form let the planner pick a MULTI-INDEX OR walk over
        idx_scrapes_reviewed_created (nearly every row is unreviewed) whenever
        sqlite_stat1 was stale; the expression form always lands on the partial
        tiered index. It was also a feed LEAK: rows the comp floor / prompt
        walls deliberately untiered keep their HIGH/POTENTIAL match_level, so
        the clause re-admitted them (a row the pay floor had removed was
        still on the Review page). tier is the single feed axis — nothing
        else decides visibility.

        Phase 27c: a posting its company job board no longer lists
        (listing_status 'gone', see refresh_listing_status) leaves the feed.
        2026-10-04: so does a role past config.REVIEW_MAX_AGE_DAYS (Strong 7
        days, Worth a look 3) — filtered in Python after the indexed query.
        Reversible — nothing is deleted, and a later sighting brings it back.
        AND-only on purpose (no OR), so the partial tiered index still serves.
        """
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, date, title, company, location, link, created_at,
                       relevance, reasoning, match_level, confidence, session_id, date_posted,
                       keyword_category, reviewed, cv_path, cl_path,
                       cv_quality_flag, cv_audit_summary, cv_domain_used, rejection_reason, tier,
                       comp_text, employment_type, comp_yearly_min, comp_yearly_max, job_level, comp_source,
                       listing_status, listing_checked_at, listing_gone_since,
                       raw_data
                FROM scrapes
                WHERE tier IN ('strong', 'worth_a_look')
                  AND COALESCE(reviewed, 0) = 0
                  AND COALESCE(listing_status, '') <> 'gone'
                ORDER BY CASE
                            WHEN tier = 'strong'
                                 OR (tier IS NULL AND match_level IN ('HIGH_MATCH', 'USER_INPUT')) THEN 0
                            WHEN tier = 'worth_a_look'
                                 OR (tier IS NULL AND match_level IN ('POTENTIAL_MATCH', 'WORTH_A_LOOK')) THEN 1
                            ELSE 2 END,
                         CASE WHEN date_posted IS NULL OR date_posted IN ('nan', 'None', '')
                            THEN 1 ELSE 0 END,
                         date_posted DESC,
                         confidence DESC NULLS LAST, created_at DESC
            """)
            rows = cursor.fetchall()
            conn.row_factory = None
            results = []
            for row in rows:
                data = dict(row)
                raw = data.pop("raw_data", None)  # parse, don't ship the blob
                if raw:
                    try:
                        desc = json.loads(raw).get("description")
                        if desc and not isinstance(desc, float):
                            data["description"] = str(desc)
                    except Exception:
                        pass
                results.append(data)

            # Freshness (2026-10-04): Strong up to 7 days,
            # Worth a look up to 3 — by posting date, or first seen when the
            # posting has none. Hidden, never deleted (utils.is_fresh_for_review).
            from .utils import is_fresh_for_review
            results = [r for r in results
                       if is_fresh_for_review(r.get("tier"), r.get("date_posted"), r.get("created_at"))]

            # Location priority within each tier: primary metro (0) -> remote (1)
            # -> unknown (2) -> secondary-metro on-site (3). Stable sort preserves the
            # SQL's tier/freshness/confidence ordering inside each bucket.
            from .config import location_priority
            tier_rank = {"strong": 0, "worth_a_look": 1}
            results.sort(key=lambda r: (
                tier_rank.get(r.get("tier"), 0 if r.get("match_level") in ("HIGH_MATCH", "USER_INPUT") else 1),
                location_priority(r.get("location"), r.get("title")),
            ))
            return results
        except Exception as e:
            logger.error(f"Error fetching unreviewed jobs: {e}")
            return []

    def mark_reviewed(self, job_ids: List[int]):
        """Mark jobs as reviewed."""
        if not job_ids:
            return
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                placeholders = ",".join("?" for _ in job_ids)
                cursor.execute(
                    f"UPDATE scrapes SET reviewed = 1, "
                    f"reviewed_at = COALESCE(reviewed_at, CURRENT_TIMESTAMP) "
                    f"WHERE id IN ({placeholders})",
                    job_ids,
                )
        except Exception as e:
            logger.error(f"Error marking jobs as reviewed: {e}")

    def get_applications(self) -> List[Dict]:
        """Get tracked applications with job details."""
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    a.id as app_id,
                    a.job_id,
                    a.status,
                    a.notes,
                    a.date_applied,
                    a.updated_at,
                    a.intel_status,
                    a.intel_path,
                    a.prep_status,
                    a.prep_stage,
                    s.title,
                    s.company,
                    s.link,
                    s.location,
                    s.cv_drive_id,
                    s.cl_drive_id,
                    s.cv_path,
                    s.cl_path,
                    s.match_level,
                    s.confidence,
                    s.keyword_category,
                    s.cv_quality_flag,
                    -- Phase 27c: a tracked card is never hidden; it only
                    -- shows "No longer on <company>'s job board since …".
                    s.listing_status,
                    s.listing_gone_since
                FROM applications a
                JOIN scrapes s ON a.job_id = s.id
                ORDER BY a.updated_at DESC
            """)
            rows = cursor.fetchall()
            conn.row_factory = None
            apps = [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching applications: {e}")
            return []
        self._attach_outcome_fields(apps)
        return apps

    def _attach_outcome_fields(self, apps: List[Dict]) -> None:
        """Add the ledger-derived card fields in place: highest_stage,
        reached_at, reached_inferred, history, outcome, sent_cv, has_prep.
        Two bulk queries for the whole board. Never raises — a ledger read
        problem must not take the Kanban down, the cards just show no
        history."""
        for a in apps:
            a.update(highest_stage=None, reached_at=None, reached_inferred=False,
                     history=[], outcome=None, sent_cv=None,
                     has_prep=(a.get("prep_status") == "ready"))
        if not apps:
            return
        try:
            by_app: Dict[int, List[Dict]] = {}
            for ev in self._fetch_events():
                by_app.setdefault(ev["app_id"], []).append(ev)
            conn = self._get_thread_conn()
            # Latest link per application = the candidate's current answer (NULL while
            # the link is only inferred).
            sent = dict(conn.execute("""
                SELECT app_id, answer FROM application_artifact_links
                WHERE id IN (SELECT MAX(id) FROM application_artifact_links GROUP BY app_id)
            """).fetchall())
            for a in apps:
                events = by_app.get(a["app_id"], [])
                reach = self._stage_reach(events)
                a.update(highest_stage=reach["highest_stage"],
                         reached_at=reach["reached_at"],
                         reached_inferred=reach["reached_inferred"],
                         history=reach["history"],
                         outcome=self._current_outcome(events, a.get("status")),
                         sent_cv=sent.get(a["app_id"]))
        except Exception as e:
            logger.error(f"Error attaching outcome history to applications: {e}")

    def update_application(self, app_id: int, status: str, notes: str,
                           source: str = "app"):
        """Update application status and notes.

        When the status TRANSITIONS to 'Applied', re-stamp date_applied to today:
        the INSERT-time value is really the date the job was *tracked* ("To
        Apply"), and the outreach follow-up cadence must count from the real
        application day, not the tracking day.

        A REAL status change also appends an application_events row in this
        same transaction (the Kanban PUT resends the status on every notes
        save, so an unchanged status writes nothing). `source` says who moved
        the card: 'app' (Kanban) | 'chat'.
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                old = cursor.execute(
                    "SELECT status, job_id FROM applications WHERE id = ?", (app_id,)).fetchone()
                old_status = old[0] if old else None
                # COALESCE = partial update: a status-only save must not
                # clobber notes mid-typing (and vice versa).
                cursor.execute("""
                    UPDATE applications
                    SET status = COALESCE(?, status),
                        notes = COALESCE(?, notes),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, notes, app_id))
                # Stamp the real application day ONLY on the first forward
                # transition (from the tracking stage). A round-trip like
                # Applied→Ghosted→Applied must NOT reset the follow-up clock.
                if status == "Applied" and old_status in ("To Apply", None):
                    cursor.execute(
                        "UPDATE applications SET date_applied = ? WHERE id = ?",
                        (datetime.now().strftime("%Y-%m-%d"), app_id))
                # Outcome ledger — same transaction, so the status and its
                # history entry commit (or roll back) together.
                if old is not None and status is not None and status != old_status:
                    self._write_status_event(cursor, app_id, old[1], old_status, status, source)
                return True
        except Exception as e:
            logger.error(f"Error updating application {app_id}: {e}")
            return False

    # === Outcome ledger: application_events (Phase 27) ===

    def _insert_application_event(self, cursor, app_id: int, job_id: Optional[int],
                                  event_type: str, from_status: str = None,
                                  to_status: str = None, effective_at: str = None,
                                  source: str = "app", note: str = None,
                                  corrects_event_id: int = None) -> int:
        """INSERT one ledger row on the caller's cursor (no commit — the caller
        owns the transaction). Title and company are copied in so the event
        still reads after its application or job is deleted."""
        if job_id is None:
            row = cursor.execute(
                "SELECT job_id FROM applications WHERE id = ?", (app_id,)).fetchone()
            job_id = row[0] if row else None
        title = company = None
        if job_id is not None:
            row = cursor.execute(
                "SELECT title, company FROM scrapes WHERE id = ?", (job_id,)).fetchone()
            if row:
                title, company = row
        cursor.execute("""
            INSERT INTO application_events
                (app_id, job_id, event_type, from_status, to_status, effective_at,
                 recorded_at, source, note, corrects_event_id, title, company)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (app_id, job_id, event_type, from_status, to_status, effective_at,
              _utc_now(), source, note, corrects_event_id, title, company))
        return cursor.lastrowid

    # An interview-stage entry the card is moved BACK out of this fast is
    # treated as an accidental move, not a real stage reach. Back = a lower
    # pipeline stage, or exactly the status the card had before it entered the
    # stage (Rejected -> Interview -> Rejected is a mis-select undone, even
    # though Rejected is an outcome with no rank). Any other move keeps the
    # reach: Applied -> Interview -> Rejected in two quick moves is how an old
    # card's real interview-then-rejection is caught up.
    ACCIDENTAL_MOVE_WINDOW_SEC = 600

    def _write_status_event(self, cursor, app_id: int, job_id: Optional[int],
                            from_status: Optional[str], to_status: str,
                            source: str = "app", event_type: str = "status_change") -> int:
        """Ledger entry for a live card move (the caller's transaction).

        - A move to Applied that is a new application act (see below): the
          note carries the apply-time snapshot (JSON) and the active CV is
          linked as the inferred sent version (a new link each time; the
          latest link is the one the "Did you send this CV?" answer uses).
        - Moving a card back out of a Screening-Call-or-later stage within
          ACCIDENTAL_MOVE_WINDOW_SEC of entering it — to a lower pipeline
          stage, or to the very status it came from: the entry is invalidated
          by a 'correction' event (the mis-click stays in the audit trail but
          no longer counts as reaching that stage). Any other move (a later
          stage, or an outcome the card did not come from) keeps the reach.
        """
        now = _utc_now()
        note = None
        snapshot: Dict[str, Any] = {}
        first_applied = False
        if to_status == "Applied":
            # A new application act = a move FROM the tracking stage, the same
            # rule that re-stamps date_applied: a mis-select into Applied that
            # was undone does not freeze the snapshot, the real move later
            # takes a fresh one with the CV active then. A card whose whole
            # history is in the ledger (it has its 'tracked' event) and never
            # reached Applied also counts — e.g. a hand-added card first placed
            # at Interview and then moved to Applied. A card from before the
            # ledger coming back to Applied from an outcome never does: that is
            # not the day the candidate applied, and today's CV is not the one
            # that was sent.
            first_applied = from_status in ("To Apply", None)
            if not first_applied:
                born_here, applied_before = cursor.execute("""
                    SELECT EXISTS(SELECT 1 FROM application_events
                                  WHERE app_id = ? AND event_type = 'tracked'),
                           EXISTS(SELECT 1 FROM application_events
                                  WHERE app_id = ? AND to_status = 'Applied')
                """, (app_id, app_id)).fetchone()
                first_applied = bool(born_here) and not applied_before
            if first_applied:
                try:
                    snapshot = self._apply_snapshot(cursor, job_id)
                except Exception as e:
                    # The snapshot is analysis data: a read problem must not
                    # roll back the candidate's move (a failed SELECT leaves the
                    # transaction usable). The event still lands, marked.
                    logger.error(f"Apply-time snapshot failed for application {app_id}: {e}")
                    snapshot = {"snapshot": "apply_time", "error": "snapshot read failed"}
                note = json.dumps(snapshot, default=str)

        # Find the stage entry BEFORE inserting the leave event.
        accidental_entry = None
        if from_status in REACHED_INTERVIEW_STAGES:
            placeholders = ",".join("?" for _ in LIVE_EVENT_TYPES)
            row = cursor.execute(f"""
                SELECT id, from_status,
                       (julianday('now') - julianday(recorded_at)) * 86400.0
                FROM application_events
                WHERE app_id = ? AND to_status = ? AND event_type IN ({placeholders})
                ORDER BY id DESC LIMIT 1
            """, (app_id, from_status, *LIVE_EVENT_TYPES)).fetchone()
            # Back = a lower pipeline stage (an overshoot fixed downward), or
            # straight back to the status the card entered the stage from.
            # The second rule covers outcomes, which have no rank: Rejected ->
            # Interview -> Rejected is undone; Applied -> Interview ->
            # Rejected (the entry came from Applied) is a real catch-up.
            entered_from = row[1] if row else None
            moved_back = ((to_status in STAGE_RANK
                           and STAGE_RANK[to_status] < STAGE_RANK.get(from_status, -1))
                          or (entered_from is not None and to_status == entered_from))
            if moved_back and row and row[2] is not None and row[2] <= self.ACCIDENTAL_MOVE_WINDOW_SEC:
                already = cursor.execute(
                    "SELECT 1 FROM application_events WHERE event_type = 'correction' "
                    "AND corrects_event_id = ? LIMIT 1", (row[0],)).fetchone()
                if not already:
                    accidental_entry = row[0]

        event_id = self._insert_application_event(
            cursor, app_id, job_id, event_type, from_status=from_status,
            to_status=to_status, effective_at=now, source=source, note=note)

        if accidental_entry is not None:
            minutes = self.ACCIDENTAL_MOVE_WINDOW_SEC // 60
            self._insert_application_event(
                cursor, app_id, job_id, "correction", from_status=from_status,
                to_status=None, effective_at=now, source="auto",
                note=f"left {from_status} within {minutes} minutes of entering it; "
                     f"treated as an accidental move",
                corrects_event_id=accidental_entry)

        if first_applied:
            cv_path = snapshot.get("cv_path")
            self._insert_artifact_link(
                cursor, app_id, job_id, snapshot.get("document_version_id"),
                cv_path, None, "inferred" if cv_path else "unknown")
        return event_id

    def _apply_snapshot(self, cursor, job_id: Optional[int]) -> Dict[str, Any]:
        """What the job and its documents looked like when the candidate applied — the
        facts later analysis needs, frozen before anything is regenerated."""
        snap: Dict[str, Any] = {"snapshot": "apply_time"}
        if job_id is None:
            return snap
        cols = ("tier", "match_level", "confidence", "keyword_category",
                "cv_quality_flag", "cv_path", "cl_path", "comp_text",
                "employment_type", "comp_yearly_min", "comp_yearly_max",
                "comp_source", "job_level", "location")
        row = cursor.execute(
            f"SELECT {', '.join(cols)}, link, raw_data FROM scrapes WHERE id = ?",
            (job_id,)).fetchone()
        if not row:
            return snap
        snap.update(dict(zip(cols, row)))
        link, raw = row[len(cols)], row[len(cols) + 1]
        host = None
        if link:
            parsed = urlparse(str(link))
            # A manual add with no URL is stored under a synthetic manual:// link.
            host = "manual" if parsed.scheme == "manual" else (parsed.netloc or None)
        snap["source_host"] = host
        posting_length = 0
        if raw:
            try:
                desc = json.loads(raw).get("description")
                if isinstance(desc, str):
                    posting_length = len(desc)
            except Exception:
                pass
        snap["posting_length"] = posting_length
        snap["document_version_id"] = self._latest_document_version_id(cursor, job_id)
        return snap

    def record_application_event(self, app_id: int, job_id: Optional[int], event_type: str,
                                 from_status: str = None, to_status: str = None,
                                 effective_at: str = None, source: str = "app",
                                 note: str = None, corrects_event_id: int = None) -> int:
        """Append one event to an application's ledger; returns its id (0 on
        failure). effective_at = when it happened (UTC 'YYYY-MM-DD HH:MM:SS' or
        a bare date); leave it None when unknown — never guess a date.

        event_type is free text; the ones the app itself writes are 'tracked',
        'status_change', 'snapshot', 'stage_reach', 'correction', 'outcome'
        (the candidate's answer on a closed card; no to_status) and 'deleted'.
        Any valid event with a to_status counts toward the stage history; a
        'correction' needs corrects_event_id and invalidates that event
        (correcting a correction restores the original)."""
        if event_type == "correction" and not corrects_event_id:
            logger.error(f"record_application_event({app_id}): a correction needs corrects_event_id")
            return 0
        try:
            with self.transaction() as conn:
                return self._insert_application_event(
                    conn.cursor(), int(app_id), job_id, event_type,
                    from_status=from_status, to_status=to_status,
                    effective_at=effective_at, source=source, note=note,
                    corrects_event_id=corrects_event_id)
        except Exception as e:
            logger.error(f"Error recording {event_type} event for application {app_id}: {e}")
            return 0

    def _fetch_events(self, app_id: int = None) -> List[Dict]:
        """Ledger rows (one application, or all) oldest-first, each with a
        computed `valid` flag: False once a valid correction targets it.
        Corrections always point at EARLIER events, so walking newest-first
        resolves a correction-of-a-correction before its target is reached."""
        conn = self._get_thread_conn()
        cols = ("id", "app_id", "job_id", "event_type", "from_status", "to_status",
                "effective_at", "recorded_at", "source", "note", "corrects_event_id",
                "title", "company")
        sql = f"SELECT {', '.join(cols)} FROM application_events"
        params: tuple = ()
        if app_id is not None:
            sql += " WHERE app_id = ?"
            params = (int(app_id),)
        events = [dict(zip(cols, r)) for r in conn.execute(sql + " ORDER BY id", params).fetchall()]
        invalid = set()
        for ev in reversed(events):
            if (ev["event_type"] == "correction" and ev["id"] not in invalid
                    and ev["corrects_event_id"]):
                invalid.add(ev["corrects_event_id"])
        for ev in events:
            ev["valid"] = ev["id"] not in invalid
        return events

    def get_application_events(self, app_id: int) -> List[Dict]:
        """Every ledger event for an application, oldest-first, corrected ones
        included (`valid` False) — the full audit trail."""
        try:
            return self._fetch_events(app_id)
        except Exception as e:
            logger.error(f"Error fetching events for application {app_id}: {e}")
            return []

    @staticmethod
    def _stage_reach(events: List[Dict]) -> Dict[str, Any]:
        """Stage history from one application's events. Only valid,
        non-correction events with a to_status count. `history` is the compact
        card line: chronological, 'To Apply' dropped, repeats collapsed (a
        recorded entry replaces a guessed one for the same stage; switching
        how a closed card ended — Rejected -> I Withdrew — stays one close,
        under the latest status and the date it first closed); `at` is None
        when the date is unknown and `inferred` marks proxy entries."""
        moves = [e for e in events
                 if e.get("valid", True) and e["event_type"] != "correction" and e.get("to_status")]
        by_id = {e["id"]: e for e in moves}
        # Undated entries sort by when they were recorded — except a backfilled
        # snapshot: it is the status the card had when its history BEGAN, so
        # it goes before the first live move even when the backfill ran after
        # that move (the API is live a moment before the backfill); and the candidate's
        # outcome-form stage, which goes right before the move that closed the
        # card it explains.
        live_times = [e.get("effective_at") or e.get("recorded_at") or ""
                      for e in events if e.get("event_type") in LIVE_EVENT_TYPES]
        first_live = min(live_times) if live_times else None

        def _order(e):
            target = None if e.get("effective_at") else by_id.get(_placed_before(e))
            if target is not None and _placed_before(target) is None:
                at, kind, target_id = _order(target)
                return (at, kind, target_id - 0.5)
            at = e.get("effective_at") or e.get("recorded_at") or ""
            if (e["event_type"] == "snapshot" and not e.get("effective_at")
                    and first_live is not None and first_live <= at):
                return (first_live, 0, e["id"])
            return (at, 1, e["id"])

        moves.sort(key=_order)

        history: List[Dict] = []
        for e in moves:
            if e["to_status"] == "To Apply":
                continue
            entry = {"status": e["to_status"], "at": e.get("effective_at"),
                     "inferred": e.get("source") in INFERRED_EVENT_SOURCES}
            if history and history[-1]["status"] == e["to_status"]:
                # The stage is known for sure now; the guessed date goes too.
                if history[-1]["inferred"] and not entry["inferred"]:
                    history[-1] = entry
                continue
            if (history and e["to_status"] in TERMINAL_STATUSES
                    and history[-1]["status"] in TERMINAL_STATUSES):
                # A card marked Rejected that the candidate then marks as withdrawn
                # closed once: the latest word on how, the date it closed.
                history[-1] = dict(history[-1], status=e["to_status"])
                continue
            history.append(entry)

        highest, reached_at, reached_inferred = None, None, False
        ranked = [e for e in moves if e["to_status"] in STAGE_RANK]
        if ranked:
            highest = max(ranked, key=lambda e: STAGE_RANK[e["to_status"]])["to_status"]
            at_stage = [e for e in ranked if e["to_status"] == highest]
            # One recorded entry confirms the stage, whatever a proxy guessed
            # (count_reached_interview counts it as confirmed too).
            at_stage = [e for e in at_stage
                        if e.get("source") not in INFERRED_EVENT_SOURCES] or at_stage
            # Prefer the first entry that carries a date.
            first = next((e for e in at_stage if e.get("effective_at")), at_stage[0])
            reached_at = first.get("effective_at")
            reached_inferred = first.get("source") in INFERRED_EVENT_SOURCES
        return {"highest_stage": highest, "reached_at": reached_at,
                "reached_inferred": reached_inferred, "history": history}

    def get_stage_reach(self, app_id: int) -> Dict[str, Any]:
        """{'highest_stage', 'reached_at', 'reached_inferred', 'history':
        [{'status', 'at', 'inferred'}]} from the application's valid events.
        A card moved Interview → Rejected keeps highest_stage 'Interview'."""
        try:
            return self._stage_reach(self._fetch_events(app_id))
        except Exception as e:
            logger.error(f"Error computing stage reach for application {app_id}: {e}")
            return {"highest_stage": None, "reached_at": None,
                    "reached_inferred": False, "history": []}

    # === Outcome answers on closed cards (Rejected / Ghosted / I Withdrew) ===

    @staticmethod
    def _closing_event(events: List[Dict]) -> Optional[Dict]:
        """The status row that closed the card's current round: the first valid
        move into a closed status after its last valid move to a live stage
        (None when the ledger has none). `events` oldest-first."""
        closing = None
        for e in events:
            if (e["event_type"] not in STATUS_EVENT_TYPES or not e.get("valid", True)
                    or not e.get("to_status")):
                continue
            if e["to_status"] in TERMINAL_STATUSES:
                closing = closing or e
            else:
                closing = None
        return closing

    @staticmethod
    def _current_outcome(events: List[Dict], status: Optional[str]) -> Optional[Dict]:
        """The candidate's latest answer for the card's current closed round:
        {'stage', 'what_happened', 'at'}. None while the card is live, and
        when the candidate has not answered since it last left a live stage (an answer
        about an earlier rejection stays in the ledger only)."""
        if status not in TERMINAL_STATUSES:
            return None
        start = 0
        for e in events:
            if (e["event_type"] in STATUS_EVENT_TYPES and e.get("valid", True)
                    and e.get("to_status") and e["to_status"] not in TERMINAL_STATUSES):
                start = e["id"]
        latest = None
        for e in events:
            if e["event_type"] == "outcome" and e["id"] > start and e.get("valid", True):
                latest = e
        if latest is None:
            return None
        try:
            data = json.loads(latest.get("note") or "{}")
        except ValueError:
            return None
        if not isinstance(data, dict) or data.get("stage") not in OUTCOME_STAGES:
            return None
        return {"stage": data["stage"], "what_happened": data.get("what_happened") or None,
                "at": latest.get("effective_at") or latest.get("recorded_at")}

    @staticmethod
    def _outcome_plan(events: List[Dict], stage: str) -> Dict[str, Any]:
        """The ledger rows that make `stage` the card's furthest stage. Pure:
        `events` = _fetch_events rows (with `valid`). Returns
        {'invalidate': [event ids], 'restore': [correction ids to undo],
         'needs_reach': True when no recorded entry proves `stage`}.

        - an entry above the candidate's stage stops counting;
        - a guessed entry (prep-sheet proxy) gives way to the candidate's answer;
        - the candidate's earlier answer at another stage is superseded;
        - an entry this form invalidated before counts again when the candidate's new
          stage covers it. One invalidated any other way (an admin correction, the
          10-minute rule) stays invalid."""
        rank = STAGE_RANK[stage]
        correctors: Dict[int, List[Dict]] = {}
        for e in events:
            if e["event_type"] == "correction" and e["valid"] and e.get("corrects_event_id"):
                correctors.setdefault(e["corrects_event_id"], []).append(e)
        invalidate: List[int] = []
        restore: List[int] = []
        proven = False
        for e in events:
            status = e.get("to_status")
            if e["event_type"] == "correction" or status not in STAGE_RANK:
                continue
            guess = e.get("source") in INFERRED_EVENT_SOURCES
            if e.get("source") == OUTCOME_SOURCE:
                keep = status == stage
            else:
                keep = not guess and STAGE_RANK[status] <= rank
            valid = e["valid"]
            if valid and not keep:
                invalidate.append(e["id"])
                valid = False
            elif not valid and keep:
                undo = correctors.get(e["id"], [])
                if undo and all(c.get("source") == OUTCOME_SOURCE for c in undo):
                    restore.extend(c["id"] for c in undo)
                    valid = True
            if valid and status == stage and not guess:
                proven = True
        return {"invalidate": invalidate, "restore": restore, "needs_reach": not proven}

    def set_application_outcome(self, app_id: int, stage: str,
                                what_happened: Optional[str] = None,
                                source: str = OUTCOME_SOURCE) -> str:
        """The candidate's answer on a closed card (Rejected / Ghosted / I Withdrew):
        the furthest stage the process reached and, optionally, what happened.
        The candidate can answer or change it any time while the card is closed. Returns
        'saved', 'bad_stage', 'not_found', 'not_closed' or 'failed'.

        The candidate's stage becomes the card's furthest stage, as ledger rows in one
        transaction (nothing is deleted; see _outcome_plan for the rules);
        when no recorded entry proves it, an undated 'stage_reach' adds it,
        shown right before the move that closed the card. An 'outcome' row
        keeps the answer and what happened; the latest one wins."""
        if stage not in OUTCOME_STAGES:
            return "bad_stage"
        text = " ".join(str(what_happened or "").split())[:OUTCOME_NOTE_MAX] or None
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                row = cursor.execute(
                    "SELECT status, job_id FROM applications WHERE id = ?",
                    (int(app_id),)).fetchone()
                if not row:
                    return "not_found"
                status, job_id = row
                if status not in TERMINAL_STATUSES:
                    return "not_closed"
                events = self._fetch_events(app_id)
                by_id = {e["id"]: e for e in events}
                plan = self._outcome_plan(events, stage)
                now = _utc_now()
                for event_id in plan["invalidate"]:
                    self._insert_application_event(
                        cursor, int(app_id), job_id, "correction",
                        from_status=by_id[event_id]["to_status"], effective_at=now,
                        source=source, note=f"furthest stage set to {stage} on the closed card",
                        corrects_event_id=event_id)
                for correction_id in plan["restore"]:
                    self._insert_application_event(
                        cursor, int(app_id), job_id, "correction",
                        from_status=by_id[correction_id]["from_status"], effective_at=now,
                        source=source, note=f"furthest stage set to {stage}: counts again",
                        corrects_event_id=correction_id)
                if plan["needs_reach"]:
                    closing = self._closing_event(events)
                    self._insert_application_event(
                        cursor, int(app_id), job_id, "stage_reach", to_status=stage,
                        effective_at=None, source=source,
                        note=json.dumps({"placed_before": closing["id"] if closing else None}))
                self._insert_application_event(
                    cursor, int(app_id), job_id, "outcome", from_status=status,
                    effective_at=now, source=source,
                    note=json.dumps({"stage": stage, "what_happened": text}))
                return "saved"
        except Exception as e:
            logger.error(f"Error saving the outcome of application {app_id}: {e}")
            return "failed"

    def get_card_outcome(self, app_id: int) -> Dict[str, Any]:
        """One card's history fields plus the candidate's current outcome answer — what
        the Kanban card shows."""
        try:
            events = self._fetch_events(app_id)
            row = self._get_thread_conn().execute(
                "SELECT status FROM applications WHERE id = ?", (int(app_id),)).fetchone()
            reach = self._stage_reach(events)
            reach["outcome"] = self._current_outcome(events, row[0] if row else None)
            return reach
        except Exception as e:
            logger.error(f"Error reading the outcome of application {app_id}: {e}")
            return {"highest_stage": None, "reached_at": None, "reached_inferred": False,
                    "history": [], "outcome": None}

    def get_outcome_summary(self, notes_limit: int = 8) -> Dict[str, Any]:
        """Stats page: how the closed cards (Rejected / Ghosted / I Withdrew,
        current cards only) ended, from the card history and the candidate's outcome answers.

        {'closed': N, 'answered': N with the candidate's answer,
         'withdrew': N the candidate ended (I Withdrew),
         'ended_after': {stage: n} — interview processes (Screening Call or
            later, confirmed) by the furthest stage they reached,
         'to_confirm': n — an interview stage known only from a proxy,
         'notes': the candidate's "what happened" lines, newest first:
            [{'app_id', 'company', 'title', 'status', 'stage', 'what_happened', 'at'}]}"""
        summary: Dict[str, Any] = {"closed": 0, "answered": 0, "withdrew": 0,
                                   "ended_after": {}, "to_confirm": 0, "notes": []}
        try:
            placeholders = ",".join("?" for _ in TERMINAL_STATUSES)
            rows = self._get_thread_conn().execute(f"""
                SELECT a.id, a.status, s.company, s.title
                FROM applications a LEFT JOIN scrapes s ON s.id = a.job_id
                WHERE a.status IN ({placeholders})
            """, TERMINAL_STATUSES).fetchall()
            by_app: Dict[int, List[Dict]] = {}
            for ev in self._fetch_events():
                by_app.setdefault(ev["app_id"], []).append(ev)
            ended: Dict[str, int] = {}
            notes: List[Dict] = []
            for app_id, status, company, title in rows:
                events = by_app.get(app_id, [])
                reach = self._stage_reach(events)
                outcome = self._current_outcome(events, status)
                stage = reach["highest_stage"]
                if stage in REACHED_INTERVIEW_STAGES:
                    if reach["reached_inferred"]:
                        summary["to_confirm"] += 1
                    else:
                        ended[stage] = ended.get(stage, 0) + 1
                if outcome:
                    summary["answered"] += 1
                    if outcome["what_happened"]:
                        notes.append({"app_id": app_id, "company": company, "title": title,
                                      "status": status, "stage": outcome["stage"],
                                      "what_happened": outcome["what_happened"],
                                      "at": outcome["at"]})
            notes.sort(key=lambda n: n["at"] or "", reverse=True)
            summary.update(closed=len(rows), ended_after=ended, notes=notes[:notes_limit],
                           withdrew=sum(1 for r in rows if r[1] == WITHDREW_STATUS))
        except Exception as e:
            logger.error(f"Error summarising closed applications: {e}")
        return summary

    def count_reached_interview(self) -> Dict[str, int]:
        """Applications that ever reached Screening Call or later — including
        cards since rejected, ghosted or deleted — split by what proves it:

        {'confirmed': N, 'unconfirmed': M}
          confirmed   = the card sits at Screening Call or later right now, or
                        a valid, non-inferred ledger event records it getting
                        there (a live move, a backfilled snapshot, a manually
                        recorded entry);
          unconfirmed = known ONLY from an inferred event (a prep-sheet proxy
                        the candidate has not confirmed).

        The current-status rule makes the number right before any backfill
        (cards from before the ledger have no events). It does not apply when
        the card's entry into that stage was corrected (a mis-click invalidated
        by hand while the card still sits there)."""
        try:
            confirmed, inferred, corrected = set(), set(), set()
            for e in self._fetch_events():
                if e["event_type"] == "correction" or e["to_status"] not in REACHED_INTERVIEW_STAGES:
                    continue
                if not e["valid"]:
                    corrected.add((e["app_id"], e["to_status"]))
                elif e.get("source") in INFERRED_EVENT_SOURCES:
                    inferred.add(e["app_id"])
                else:
                    confirmed.add(e["app_id"])
            conn = self._get_thread_conn()
            for app_id, status in conn.execute("SELECT id, status FROM applications"):
                if status in REACHED_INTERVIEW_STAGES and (app_id, status) not in corrected:
                    confirmed.add(app_id)
            return {"confirmed": len(confirmed), "unconfirmed": len(inferred - confirmed)}
        except Exception as e:
            logger.error(f"Error counting interview-stage applications: {e}")
            return {"confirmed": 0, "unconfirmed": 0}

    def correct_application_event(self, app_id: int, event_id: int,
                                  source: str = "admin", note: str = None) -> int:
        """Invalidate one event of an application with a 'correction' event
        (nothing is deleted). Returns the correction's id, or 0 when the event
        does not exist / belongs to another application. Correcting a
        correction undoes it."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                row = cursor.execute(
                    "SELECT app_id, job_id, to_status FROM application_events WHERE id = ?",
                    (int(event_id),)).fetchone()
                if not row or row[0] != int(app_id):
                    return 0
                return self._insert_application_event(
                    cursor, int(app_id), row[1], "correction", from_status=row[2],
                    effective_at=_utc_now(), source=source, note=note,
                    corrects_event_id=int(event_id))
        except Exception as e:
            logger.error(f"Error correcting event {event_id} of application {app_id}: {e}")
            return 0

    # === Sent-CV record: application_artifact_links (Phase 27) ===

    def _insert_artifact_link(self, cursor, app_id: int, job_id: Optional[int],
                              version_id: Optional[int], cv_path: Optional[str],
                              answer: Optional[str], provenance: str) -> int:
        cursor.execute("""
            INSERT INTO application_artifact_links
                (app_id, job_id, version_id, cv_path, answer, provenance, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (app_id, job_id, version_id, cv_path, answer, provenance, _utc_now()))
        return cursor.lastrowid

    def set_sent_document(self, app_id: int, answer: Optional[str], version_id: int = None,
                          provenance: str = "confirmed") -> bool:
        """Record whether the candidate sent the CV we generated. answer: 'as_is' |
        'edited' | 'own' (None only for a non-confirmed link). Append-only —
        the answer can change and the latest row is the answer.

        The link points at the version that was active when the candidate APPLIED (the
        inferred link written at the Applied move) unless version_id is given;
        for a card applied before the ledger existed it falls back to the CV
        on the card now, which is the one the question was asked about."""
        if answer is not None and answer not in SENT_CV_ANSWERS:
            logger.error(f"set_sent_document({app_id}): invalid answer {answer!r}")
            return False
        if answer is None and provenance == "confirmed":
            logger.error(f"set_sent_document({app_id}): a confirmed link needs an answer")
            return False
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                app = cursor.execute(
                    "SELECT job_id FROM applications WHERE id = ?", (int(app_id),)).fetchone()
                if not app:
                    return False
                job_id = app[0]
                cv_path = None
                if version_id is not None:
                    row = cursor.execute(
                        "SELECT cv_path FROM document_versions WHERE id = ?",
                        (int(version_id),)).fetchone()
                    cv_path = row[0] if row else None
                else:
                    prev = cursor.execute("""
                        SELECT version_id, cv_path FROM application_artifact_links
                        WHERE app_id = ? ORDER BY id DESC LIMIT 1
                    """, (int(app_id),)).fetchone()
                    if prev and (prev[0] is not None or prev[1]):
                        version_id, cv_path = prev
                    else:
                        row = cursor.execute(
                            "SELECT cv_path FROM scrapes WHERE id = ?", (job_id,)).fetchone()
                        cv_path = row[0] if row else None
                        version_id = self._latest_document_version_id(cursor, job_id)
                self._insert_artifact_link(cursor, int(app_id), job_id, version_id,
                                           cv_path, answer, provenance)
                return True
        except Exception as e:
            logger.error(f"Error recording sent document for application {app_id}: {e}")
            return False

    def get_sent_document(self, app_id: int) -> Optional[Dict]:
        """The current sent-CV record for an application (latest link), or None:
        {id, app_id, job_id, version_id, cv_path, answer, provenance, recorded_at}.
        answer is None while the link is only inferred."""
        try:
            conn = self._get_thread_conn()
            cols = ("id", "app_id", "job_id", "version_id", "cv_path", "answer",
                    "provenance", "recorded_at")
            row = conn.execute(f"""
                SELECT {', '.join(cols)} FROM application_artifact_links
                WHERE app_id = ? ORDER BY id DESC LIMIT 1
            """, (int(app_id),)).fetchone()
            return dict(zip(cols, row)) if row else None
        except Exception as e:
            logger.error(f"Error fetching sent document for application {app_id}: {e}")
            return None

    # === Saved document versions (Phase 27) ===

    @staticmethod
    def _latest_document_version_id(cursor, job_id: Optional[int]) -> Optional[int]:
        if job_id is None:
            return None
        row = cursor.execute(
            "SELECT MAX(id) FROM document_versions WHERE job_id = ?", (job_id,)).fetchone()
        return row[0] if row else None

    def record_document_version(self, job_id: int, generation_id: str, cv_path: Optional[str],
                                cl_path: Optional[str], meta: dict = None) -> int:
        """Save one generated CV/letter pair as a version row; returns its id
        (0 on failure). Versions are never overwritten — a regeneration adds a
        row. scrapes.cv_path/cl_path (update_job_documents) stay the active
        version the download links serve."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO document_versions
                        (job_id, generation_id, cv_path, cl_path, meta_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (int(job_id), generation_id, cv_path, cl_path,
                      json.dumps(meta or {}, default=str), _utc_now()))
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error recording document version for job {job_id}: {e}")
            return 0

    def get_document_versions(self, job_id: int) -> List[Dict]:
        """Saved versions for a job, newest first: {id, job_id, generation_id,
        cv_path, cl_path, meta (dict), created_at}."""
        try:
            conn = self._get_thread_conn()
            rows = conn.execute("""
                SELECT id, job_id, generation_id, cv_path, cl_path, meta_json, created_at
                FROM document_versions WHERE job_id = ? ORDER BY id DESC
            """, (int(job_id),)).fetchall()
            versions = []
            for vid, jid, gen_id, cv_path, cl_path, meta_json, created_at in rows:
                try:
                    meta = json.loads(meta_json) if meta_json else {}
                except Exception:
                    meta = {}
                versions.append({"id": vid, "job_id": jid, "generation_id": gen_id,
                                 "cv_path": cv_path, "cl_path": cl_path,
                                 "meta": meta, "created_at": created_at})
            return versions
        except Exception as e:
            logger.error(f"Error fetching document versions for job {job_id}: {e}")
            return []

    def update_job_documents(self, job_id: int, cv_path: str = None, cl_path: str = None) -> bool:
        """Point the job's active CV / letter at new files. Sets only the
        values that are not None, so a letter-only update keeps the CV."""
        updates, params = [], []
        if cv_path is not None:
            updates.append("cv_path = ?")
            params.append(cv_path)
        if cl_path is not None:
            updates.append("cl_path = ?")
            params.append(cl_path)
        if not updates:
            return True
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(f"UPDATE scrapes SET {', '.join(updates)} WHERE id = ?",
                               (*params, int(job_id)))
                if cursor.rowcount == 0:
                    logger.warning(f"update_job_documents({job_id}): UPDATE matched 0 rows")
                    return False
                return True
        except Exception as e:
            logger.error(f"Error updating documents for job {job_id}: {e}")
            return False

    def save_generated_documents(self, job_id: int, cv_path: Optional[str], cl_path: Optional[str],
                                 flag: str = None, audit_json: str = None,
                                 domain_used: str = None) -> bool:
        """Make one generation the job's active documents, in ONE UPDATE in one
        transaction (a concurrent generation can never pair its file with this
        one's verdict).

        - cv_path set: cv_path, cl_path, cv_quality_flag, cv_audit_summary,
          cv_generated_at and cv_domain_used all come from this generation; a
          generation that wrote a CV but no letter CLEARS cl_path (the old
          letter belonged to an older CV).
        - cv_path None, cl_path set: only the letter changes (the old CV keeps
          its own verdict).
        - neither: nothing changes (a failed generation never clears a path).
        False when the job does not exist or the write failed."""
        if cv_path is not None:
            sql = ("UPDATE scrapes SET cv_path = ?, cl_path = ?, cv_quality_flag = ?, "
                   "cv_audit_summary = ?, cv_generated_at = ?, cv_domain_used = ? WHERE id = ?")
            params = (cv_path, cl_path, flag, audit_json, datetime.now().isoformat(),
                      domain_used, int(job_id))
        elif cl_path is not None:
            sql, params = "UPDATE scrapes SET cl_path = ? WHERE id = ?", (cl_path, int(job_id))
        else:
            return True
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                if cursor.rowcount == 0:
                    logger.warning(f"save_generated_documents({job_id}): UPDATE matched 0 rows")
                    return False
                return True
        except Exception as e:
            logger.error(f"Error saving generated documents for job {job_id}: {e}")
            return False

    def save_contact(self, rec: dict) -> bool:
        """Upsert a named contact for a job (Spec M2), keyed (job_id, name)."""
        try:
            with self.transaction() as conn:
                conn.cursor().execute("""
                    INSERT INTO contacts
                        (job_id, company, name, title, role_type, email,
                         email_status, source, confidence)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(job_id, name) DO UPDATE SET
                        title=excluded.title, role_type=excluded.role_type,
                        email=COALESCE(excluded.email, email),
                        email_status=excluded.email_status,
                        source=COALESCE(excluded.source, source),
                        confidence=COALESCE(excluded.confidence, confidence)
                """, (rec.get("job_id"), rec.get("company"), rec.get("name"),
                      rec.get("title"), rec.get("role_type"), rec.get("email"),
                      rec.get("email_status"), rec.get("source"),
                      rec.get("confidence")))
                return True
        except Exception as e:
            logger.error(f"Error saving contact {rec.get('name')!r}: {e}")
            return False

    def get_contacts_for_job(self, job_id: int) -> List[Dict]:
        """All stored contacts for a job, verified emails first."""
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute("""
                SELECT * FROM contacts WHERE job_id = ?
                ORDER BY CASE email_status WHEN 'verified' THEN 0
                         WHEN 'pattern_guess' THEN 1 ELSE 2 END, name
            """, (job_id,))]
            conn.row_factory = None
            return rows
        except Exception as e:
            logger.error(f"Error fetching contacts for job {job_id}: {e}")
            return []

    def claim_application_intel(self, app_id: int, stale_seconds: int) -> bool:
        """ATOMICALLY claim intel generation for an application.

        Sets intel_status='pending' ONLY IF no live pending exists (NULL /
        ready / failed, or a pending older than stale_seconds — a daemon killed
        by a service restart). Returns True iff THIS caller won the claim —
        closes the check-then-set race where two rapid Applied transitions
        both spawned a paid generation."""
        try:
            with self.transaction() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE applications
                    SET intel_status = 'pending', intel_generated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                      AND (intel_status IS NULL
                           OR intel_status != 'pending'
                           OR intel_generated_at IS NULL
                           OR (julianday('now') - julianday(intel_generated_at)) * 86400 > ?)
                """, (app_id, stale_seconds))
                return cur.rowcount == 1
        except Exception as e:
            logger.error(f"Error claiming intel for application {app_id}: {e}")
            return False

    def update_application_intel(self, app_id: int, status: str, path: str = None) -> bool:
        """Record outreach-intel generation state for an application.

        status: 'pending' | 'ready' | 'failed'. Path is set when ready.
        intel_generated_at is stamped on EVERY state change — for 'pending' it
        marks when generation started, which the API uses to detect a stale
        pending row (daemon thread killed by a service restart) and allow a
        respawn instead of blocking regeneration forever.
        """
        try:
            with self.transaction() as conn:
                conn.cursor().execute("""
                    UPDATE applications
                    SET intel_status = ?,
                        intel_path = COALESCE(?, intel_path),
                        intel_generated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, path, app_id))
                return True
        except Exception as e:
            logger.error(f"Error updating intel state for application {app_id}: {e}")
            return False

    def claim_application_prep(self, app_id: int, stale_seconds: int) -> bool:
        """ATOMICALLY claim interview-prep generation (mirror of
        claim_application_intel on the prep_* columns). Returns True iff this
        caller won — closes the double-spawn race on rapid stage transitions."""
        try:
            with self.transaction() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE applications
                    SET prep_status = 'pending', prep_generated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                      AND (prep_status IS NULL
                           OR prep_status != 'pending'
                           OR prep_generated_at IS NULL
                           OR (julianday('now') - julianday(prep_generated_at)) * 86400 > ?)
                """, (app_id, stale_seconds))
                return cur.rowcount == 1
        except Exception as e:
            logger.error(f"Error claiming prep for application {app_id}: {e}")
            return False

    def update_application_prep(self, app_id: int, status: str, path: str = None) -> bool:
        """Record interview-prep generation state ('pending'|'ready'|'failed').
        prep_generated_at is re-stamped on every change so a killed-daemon
        pending row can be reclaimed (same stale-recovery contract as intel)."""
        try:
            with self.transaction() as conn:
                conn.cursor().execute("""
                    UPDATE applications
                    SET prep_status = ?,
                        prep_path = COALESCE(?, prep_path),
                        prep_generated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, path, app_id))
                return True
        except Exception as e:
            logger.error(f"Error updating prep state for application {app_id}: {e}")
            return False

    def set_application_prep_stage(self, app_id: int, stage: Optional[str]) -> bool:
        """Remember which interview stage the current prep sheet was built for
        (prep regenerates only when the card reaches a higher one)."""
        try:
            with self.transaction() as conn:
                conn.cursor().execute(
                    "UPDATE applications SET prep_stage = ? WHERE id = ?", (stage, app_id))
                return True
        except Exception as e:
            logger.error(f"Error setting prep stage for application {app_id}: {e}")
            return False

    def restore_application_prep(self, app_id: int, path: str, stage: Optional[str],
                                 generated_at: Optional[str]) -> bool:
        """Put a previous READY prep sheet back after an automatic rebuild
        failed: status, file, the stage it was built for and its original
        date, in one UPDATE. Only a row the failed run left 'failed' is
        touched — a newer claim or sheet is never overwritten. Returns True
        iff the row was restored."""
        try:
            with self.transaction() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE applications
                    SET prep_status = 'ready', prep_path = ?, prep_stage = ?,
                        prep_generated_at = COALESCE(?, prep_generated_at)
                    WHERE id = ? AND prep_status = 'failed'
                """, (path, stage, generated_at, app_id))
                return cur.rowcount == 1
        except Exception as e:
            logger.error(f"Error restoring the previous prep sheet for application {app_id}: {e}")
            return False

    def get_application(self, app_id: int) -> Optional[Dict]:
        """One application row joined with its job details (or None).

        The job description is not a scrapes column — it lives inside the
        raw_data JSON blob, so parse it out (same convention as the doc-
        generation flow)."""
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT a.id AS app_id, a.job_id, a.status, a.notes, a.date_applied,
                       a.intel_status, a.intel_path, a.intel_generated_at,
                       a.prep_status, a.prep_path, a.prep_generated_at, a.prep_stage,
                       s.title, s.company, s.location, s.link, s.raw_data,
                       s.keyword_category, s.cv_path, s.cl_path
                FROM applications a JOIN scrapes s ON a.job_id = s.id
                WHERE a.id = ?
            """, (app_id,)).fetchone()
            conn.row_factory = None
            if not row:
                return None
            d = dict(row)
            raw = d.pop("raw_data", None)
            d["description"] = ""
            d["hiring_contact"] = ""
            if raw:
                try:
                    import json as _json
                    raw_d = _json.loads(raw)
                    d["description"] = str(raw_d.get("description") or "")
                    # Greenhouse hiring-team metadata captured at scrape time
                    # (People Intel M1 source #1 — the posting itself).
                    d["hiring_contact"] = str(raw_d.get("hiring_contact") or "")
                except Exception:
                    pass
            return d
        except Exception as e:
            logger.error(f"Error fetching application {app_id}: {e}")
            return None

    def delete_application(self, app_id: int) -> bool:
        """Delete an application track (does NOT delete the job details from scrapes)."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                # The ledger outlives the card: note the removal, keep the events.
                row = cursor.execute(
                    "SELECT job_id, status FROM applications WHERE id = ?", (app_id,)).fetchone()
                if row:
                    self._insert_application_event(
                        cursor, app_id, row[0], "deleted", from_status=row[1],
                        effective_at=_utc_now(), source="app")
                cursor.execute("DELETE FROM applications WHERE id = ?", (app_id,))
                return cursor.rowcount > 0  # False -> API returns a real 404
        except Exception as e:
            logger.error(f"Error deleting application {app_id}: {e}")
            return False

    def get_job_details(self, job_id: int) -> Optional[Dict]:
        """Fetch full job details by ID."""
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scrapes WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            conn.row_factory = None
            if row:
                data = dict(row)
                if data.get('raw_data'):
                    try:
                        raw = json.loads(data['raw_data'])
                        # jobspy raw_data carries its own "id" (e.g. "li-<digits>")
                        # — letting it clobber the DB integer PK broke
                        # update_job_docs ("UPDATE matched 0 rows": documents were
                        # generated but never linked, 2026-07-17). DB identity
                        # columns always win over the raw blob.
                        for k in ("id", "link", "session_id", "created_at"):
                            raw.pop(k, None)
                        data.update(raw)
                        data["id"] = row["id"]  # belt and braces
                    except Exception:
                        pass
                return data
            return None
        except Exception as e:
            logger.error(f"Error fetching job details {job_id}: {e}")
            return None

    def delete_job(self, job_id: int) -> bool:
        """Delete a job from history (and applications)."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                # Ledger events are kept (title/company were copied in).
                for app_id, status in cursor.execute(
                        "SELECT id, status FROM applications WHERE job_id = ?", (job_id,)).fetchall():
                    self._insert_application_event(
                        cursor, app_id, job_id, "deleted", from_status=status,
                        effective_at=_utc_now(), source="app")
                cursor.execute("DELETE FROM applications WHERE job_id = ?", (job_id,))
                cursor.execute("DELETE FROM scrapes WHERE id = ?", (job_id,))
                return True
        except Exception as e:
            logger.error(f"Error deleting job {job_id}: {e}")
            return False

    def clear_history(self) -> bool:
        """
        Clear untracked history.
        Preserves jobs that are in the applications table.
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM scrapes WHERE id NOT IN (SELECT job_id FROM applications)")
                return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False

    def get_pending_jobs(self) -> List[Dict]:
        """Fetch jobs that have been scraped but not yet analyzed (relevance is NULL)."""
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM scrapes
                WHERE relevance IS NULL OR relevance = ''
                ORDER BY created_at ASC
            """)
            rows = cursor.fetchall()
            conn.row_factory = None
            results = []
            for row in rows:
                data = dict(row)
                if data.get('raw_data'):
                    try:
                        raw = json.loads(data['raw_data'])
                        for k, v in raw.items():
                            if k not in data or data[k] is None:
                                data[k] = v
                    except Exception:
                        pass
                results.append(data)
            return results
        except Exception as e:
            logger.error(f"Error fetching pending jobs: {e}")
            return []

    def get_all_jobs_for_reeval(self) -> List[Dict]:
        """Fetch all scraped jobs (with full data from raw_data) for re-evaluation."""
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM scrapes
                ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()
            conn.row_factory = None
            results = []
            for row in rows:
                data = dict(row)
                if data.get('raw_data'):
                    try:
                        raw = json.loads(data['raw_data'])
                        for k, v in raw.items():
                            if k not in data or data[k] is None:
                                data[k] = v
                    except Exception:
                        pass
                results.append(data)
            return results
        except Exception as e:
            logger.error(f"Error fetching all jobs for re-eval: {e}")
            return []

    # === Chat History (in-app Assistant, Phase 19) ===

    def save_chat_message(self, role: str, content: str, session_id: str = "candidate") -> bool:
        """Persist one chat turn (role: 'user' | 'assistant')."""
        try:
            with self.transaction() as conn:
                conn.execute(
                    "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, role, content),
                )
                return True
        except Exception as e:
            logger.error(f"Error saving chat message: {e}")
            return False

    def get_chat_messages(self, session_id: str = "candidate", limit: int = 40) -> List[Dict]:
        """Most recent chat turns, oldest-first (ready to replay as context)."""
        try:
            conn = self._get_thread_conn()
            rows = conn.execute(
                """SELECT role, content, created_at FROM (
                       SELECT id, role, content, created_at FROM chat_messages
                       WHERE session_id = ? ORDER BY id DESC LIMIT ?
                   ) ORDER BY id ASC""",
                (session_id, limit),
            ).fetchall()
            return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]
        except Exception as e:
            logger.error(f"Error fetching chat messages: {e}")
            return []

    def list_chat_sessions(self) -> List[Dict]:
        """All chat threads: id, first-user-message preview, last activity, turn count.
        Ordered by most recent activity."""
        try:
            conn = self._get_thread_conn()
            rows = conn.execute(
                """SELECT m1.session_id,
                          (SELECT content FROM chat_messages m2
                           WHERE m2.session_id = m1.session_id AND m2.role = 'user'
                           ORDER BY m2.id ASC LIMIT 1) AS preview,
                          MAX(m1.created_at) AS last_at,
                          COUNT(*) AS turns
                   FROM chat_messages m1
                   GROUP BY m1.session_id
                   ORDER BY MAX(m1.id) DESC"""
            ).fetchall()
            return [
                {"session_id": r[0], "preview": (r[1] or "")[:80],
                 "last_at": r[2], "turns": r[3]}
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Error listing chat sessions: {e}")
            return []

    def clear_chat_messages(self, session_id: str = "candidate") -> bool:
        try:
            with self.transaction() as conn:
                conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                return True
        except Exception as e:
            logger.error(f"Error clearing chat messages: {e}")
            return False

    # === Rejection Reason Methods (Phase 15) ===

    def update_rejection_reason(self, job_id: int, reason: str) -> bool:
        """Record why the candidate dismissed a job."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE scrapes SET rejection_reason = ? WHERE id = ?", (reason, job_id))
                return True
        except Exception as e:
            logger.error(f"Error updating rejection reason for job {job_id}: {e}")
            return False

    def dismiss_job(self, job_id: int, reason: str = None,
                    suppress_location: bool = None) -> Optional[str]:
        """Dismiss a job: store the reason, mark it reviewed, and run the
        location feedback loop — when the dismissal is location-driven (explicit
        suppress_location, or a reason that names a location problem) the job's
        location is learned as a no-go. Returns the suppressed location key, or
        None. One path for the HTTP endpoint and the chat tool, so neither
        skips the feedback. add_location_suppression refuses the primary metro / remote."""
        reason = reason or ""
        if reason:
            self.update_rejection_reason(job_id, reason)
        location_driven = bool(suppress_location) or any(
            h in reason.lower() for h in LOCATION_REASON_HINTS)
        suppressed = None
        if location_driven:
            from .config import LOCATION_SUPPRESS_MIN_VOTES
            loc = self.get_job_location(job_id)
            # One dismissal is a vote, not a block (2026-10-04): the
            # location is suppressed only once it has LOCATION_SUPPRESS_MIN_VOTES,
            # or on an explicit suppress_location (a deliberate "block this
            # place", not a passing reason). Both still pass the class guard.
            votes = self.record_location_vote(loc, job_id) if loc else 0
            if loc and (bool(suppress_location) or votes >= LOCATION_SUPPRESS_MIN_VOTES) \
                    and self.add_location_suppression(loc, source_job_id=job_id):
                suppressed = loc.strip().lower()  # matches the stored (normalized) key
        self.mark_reviewed([job_id])
        return suppressed

    def get_rejection_stats(self) -> dict:
        """Count dismissals by reason. Returns {reason: count} for all non-null rejection_reason."""
        try:
            conn = self._get_thread_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT rejection_reason, COUNT(*) as cnt
                FROM scrapes
                WHERE rejection_reason IS NOT NULL
                GROUP BY rejection_reason
                ORDER BY cnt DESC
            """)
            return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error fetching rejection stats: {e}")
            return {}

    # === Feedback loop: location suppression + company scrutiny ===

    def add_location_suppression(self, location: str, source_job_id: int = None) -> bool:
        """Record a location the candidate dismissed as a no-go. Future jobs whose
        location matches are dropped by the pre-filter. Normalized to lowercase.
        Primary-metro / remote locations are refused here so a mis-tagged dismissal
        can never blacklist "Remote" or the candidate's own metro."""
        loc = str(location or "").strip().lower()
        if not loc or loc == "none":
            return False
        # Refuse dangerously short/ambiguous keys ("la", "ca", "us", "nyc").
        # Matching is exact-segment now, but a bare abbreviation is too ambiguous
        # to be a reliable no-go and risks over-broad drops.
        if len(loc) < 4:
            logger.info(f"Refusing to suppress too-short location key: {loc!r}")
            return False
        from .config import location_is_suppressible
        # Only a specific place outside the candidate's area: never the candidate's targets (primary metro /
        # remote), never a vague label ("United States", "Hybrid", "2
        # locations") and never a bare state name — keys that broad
        # would block whole employers.
        if not location_is_suppressible(loc):
            logger.info(f"Refusing to suppress a target or vague location: {loc!r}")
            return False
        try:
            with self.transaction() as conn:
                conn.cursor().execute(
                    "INSERT OR IGNORE INTO location_suppressions (location, source_job_id) VALUES (?, ?)",
                    (loc, source_job_id))
            logger.info(f"Suppressed location (candidate feedback): {loc!r}")
            return True
        except Exception as e:
            logger.error(f"Error suppressing location {location!r}: {e}")
            return False

    def record_location_vote(self, location: str, job_id: int) -> int:
        """Record one location-driven dismissal and return on how many distinct
        DAYS the candidate has dismissed this location (normalized) — a batch dismissal,
        or one job seen on two sites, is still saying it once. 0 on error."""
        loc = str(location or "").strip().lower()
        if not loc:
            return 0
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT OR IGNORE INTO location_votes (location, job_id) VALUES (?, ?)",
                               (loc, int(job_id)))
                return int(cursor.execute(
                    "SELECT COUNT(DISTINCT date(voted_at)) FROM location_votes WHERE location = ?",
                    (loc,)).fetchone()[0])
        except Exception as e:
            logger.error(f"Error recording location vote for {location!r}: {e}")
            return 0

    def get_suppressed_locations(self) -> set:
        """All location strings the candidate has suppressed (lowercased)."""
        try:
            conn = self._get_thread_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT location FROM location_suppressions")
            return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error fetching suppressed locations: {e}")
            return set()

    def remove_location_suppression(self, location: str) -> bool:
        """Un-suppress a location (undo)."""
        try:
            with self.transaction() as conn:
                conn.cursor().execute(
                    "DELETE FROM location_suppressions WHERE location = ?",
                    (str(location or "").strip().lower(),))
            return True
        except Exception as e:
            logger.error(f"Error un-suppressing location {location!r}: {e}")
            return False

    # Only COMPANY-level dismissal reasons feed the company-scrutiny caution.
    # A role dismissed for wrong_location / wrong_industry / social_media /
    # too_junior / already_applied says nothing bad about the COMPANY — counting
    # those would poison the caution against good targets (a large employer
    # flagged purely for posting mislocated roles). Only "this company is a bad
    # fit" and "this is an agency/staffing vendor" are company signals.
    COMPANY_FIT_REASONS = ("bad_company_fit", "agency_vendor")

    def get_followups_due(self, days_list) -> List[Dict]:
        """Applications in 'Applied' whose cadence anchor lands TODAY. Uses the
        SAME weekend rule as the playbook (an anchor on Sat/Sun shifts to the
        next Monday) so the digest and the playbook name the same dates."""
        from datetime import timedelta
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT a.id AS app_id, a.date_applied, a.intel_status,
                       s.title, s.company, s.link
                FROM applications a JOIN scrapes s ON a.job_id = s.id
                WHERE a.status = 'Applied' AND a.date_applied IS NOT NULL
            """).fetchall()
            conn.row_factory = None
            today = datetime.now().date()
            due = []
            for r in rows:
                try:
                    d0 = datetime.strptime(r["date_applied"], "%Y-%m-%d").date()
                except (TypeError, ValueError):
                    continue
                for offset in (days_list or []):
                    d = d0 + timedelta(days=int(offset))
                    if d.weekday() >= 5:  # Sat/Sun → next Monday (mirror build_cadence)
                        d += timedelta(days=7 - d.weekday())
                    if d == today:
                        due.append({**dict(r), "days_since": int(offset)})
                        break
            return due
        except Exception as e:
            logger.error(f"Error fetching follow-ups due: {e}")
            return []

    def get_dismissed_company_counts(self, min_count: int = 1) -> dict:
        """Companies the candidate dismissed for a COMPANY-level reason,
        {company_lower: count}. Feeds the cascade's "scrutinize this company more
        carefully" caution (NOT a suppression). Deliberately ignores role/location
        dismissals so premier targets aren't penalised for mislocated postings."""
        try:
            conn = self._get_thread_conn()
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in self.COMPANY_FIT_REASONS)
            cursor.execute(f"""
                SELECT LOWER(TRIM(company)) AS c, COUNT(*) AS cnt
                FROM scrapes
                WHERE rejection_reason IN ({placeholders})
                  AND company IS NOT NULL AND TRIM(company) != ''
                GROUP BY c
                HAVING cnt >= ?
            """, (*self.COMPANY_FIT_REASONS, min_count))
            return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error fetching dismissed company counts: {e}")
            return {}

    # === Company Intel Methods (Phase 15 — Perplexity cache) ===

    def get_company_intel(self, company_name: str) -> Optional[dict]:
        """Return cached company intel row or None if not found."""
        try:
            conn = self._get_thread_conn()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM company_intel WHERE company_name = ?", (company_name,))
            row = cursor.fetchone()
            conn.row_factory = None
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching company intel for {company_name}: {e}")
            return None

    def save_company_intel(self, data: dict) -> bool:
        """Upsert company intel record. Always updates last_checked to today."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO company_intel (company_name, careers_url, ats_type,
                                              company_profile, last_checked,
                                              last_found_roles, ignored)
                    VALUES (?, ?, ?, ?, date('now'), ?, ?)
                    ON CONFLICT(company_name) DO UPDATE SET
                        careers_url = COALESCE(excluded.careers_url, company_intel.careers_url),
                        ats_type = COALESCE(excluded.ats_type, company_intel.ats_type),
                        company_profile = COALESCE(excluded.company_profile, company_intel.company_profile),
                        last_checked = date('now'),
                        last_found_roles = COALESCE(excluded.last_found_roles, company_intel.last_found_roles),
                        ignored = COALESCE(excluded.ignored, company_intel.ignored)
                """, (
                    data.get("company_name"),
                    data.get("careers_url"),
                    data.get("ats_type"),
                    data.get("company_profile"),
                    data.get("last_found_roles", 0),
                    data.get("ignored", False),
                ))
                return True
        except Exception as e:
            logger.error(f"Error saving company intel for {data.get('company_name')}: {e}")
            return False

    def is_company_ignored(self, company_name: str) -> bool:
        """Return True if company is flagged ignored in company_intel."""
        try:
            conn = self._get_thread_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT ignored FROM company_intel WHERE company_name = ?", (company_name,))
            row = cursor.fetchone()
            return bool(row and row[0])
        except Exception as e:
            logger.error(f"Error checking ignored status for {company_name}: {e}")
            return False

    def set_company_ignored(self, company_name: str, ignored: bool = True) -> bool:
        """Set or clear ignored flag."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO company_intel (company_name, ignored, last_checked)
                    VALUES (?, ?, date('now'))
                    ON CONFLICT(company_name) DO UPDATE SET ignored = ?, last_checked = date('now')
                """, (company_name, ignored, ignored))
                return True
        except Exception as e:
            logger.error(f"Error setting ignored for {company_name}: {e}")
            return False

    # === "Is this posting real?" (Phase 27c) — facts, never a score ===
    # Nothing below reads or writes tier, confidence, match_level, reviewed
    # or ordering. A posting is GONE only when a full-list company job board
    # it was seen on left it out of its two most recent complete, successful
    # fetches; every other case (search sites, failed / partial / empty /
    # shrunken fetches, never seen) is "unknown" — unavailable is not closed.

    def record_board_fetch(self, platform: str, token: str, company: str, *, ok: bool,
                           complete: bool, total_count: int, links: List[Any],
                           run_id: Optional[str] = None) -> Optional[int]:
        """Insert one board_fetches row and upsert the board's sightings in one
        transaction; returns the fetch id (None when ignored or on a DB error).

        Only platforms in POSTING_CHECK_FULL_LIST_PLATFORMS are recorded.
        evidence = ok and complete and total_count >= 1 and links non-empty
        and not a DROP (total_count below POSTING_FETCH_DROP_RATIO x the
        board's previous COMPLETE ok fetch with total_count > 0; a partial
        read's small count is never the baseline — a sudden shrink is
        more likely a broken fetch than a mass closing; the next fetch at the
        new size compares against this one and counts again).
        Also not evidence: CHURN — the board's previous complete ok fetch
        listed at least POSTING_CHURN_MIN_REFS postings and fewer than
        POSTING_CHURN_KEEP_RATIO of them are in this fetch (a board serving
        the same postings under new links, e.g. a new careers domain, would
        otherwise close all of them at once; the next fetch compares against
        this one and counts again).
        Sightings are written whenever ok and links non-empty: presence is
        proven even by a partial read. links: (link, title) pairs or bare
        links; the title is kept as title_key for the re-posted check.

        Catches sqlite3.Error ONLY — main's SIGALRM _ScrapeTimeout raised
        mid-write must propagate (the transaction rolls back)."""
        from .config import (POSTING_CHECK_FULL_LIST_PLATFORMS, POSTING_CHURN_KEEP_RATIO,
                             POSTING_CHURN_MIN_REFS, POSTING_FETCH_DROP_RATIO)
        platform = (platform or "").strip().lower() if isinstance(platform, str) else ""
        token = token.strip() if isinstance(token, str) else ""
        if platform not in POSTING_CHECK_FULL_LIST_PLATFORMS or not token:
            return None
        board_key = f"{platform}:{token}"
        refs: Dict[str, tuple] = {}
        for item in links or ():
            link, title = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item, "")
            ref = posting_ref(link)
            if ref:
                refs[ref] = (link.strip(), title_key(title) or None)
        ok, complete = bool(ok), bool(complete)
        try:
            total = int(total_count or 0)
        except (TypeError, ValueError):
            total = 0
        company = company if isinstance(company, str) else None
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                prev = cursor.execute(
                    "SELECT total_count, id FROM board_fetches "
                    "WHERE board_key = ? AND ok = 1 AND complete = 1 AND total_count > 0 "
                    "ORDER BY id DESC LIMIT 1", (board_key,)).fetchone()
                if not ok:
                    note = "failed"
                elif not complete:
                    note = "partial"
                elif total < 1 or not refs:
                    note = "empty"
                elif prev and total < POSTING_FETCH_DROP_RATIO * prev[0]:
                    note = "drop"
                else:
                    note = None
                if note is None and prev:
                    before = {r[0] for r in cursor.execute(
                        "SELECT ref FROM board_sightings "
                        "WHERE board_key = ? AND last_seen_fetch_id >= ?", (board_key, prev[1]))}
                    if (len(before) >= POSTING_CHURN_MIN_REFS
                            and len(before & set(refs)) < POSTING_CHURN_KEEP_RATIO * len(before)):
                        note = "churn"
                now = _utc_now()
                cursor.execute("""
                    INSERT INTO board_fetches (board_key, platform, token, company, run_id,
                                               fetched_at, ok, complete, total_count,
                                               link_count, evidence, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (board_key, platform, token, company, run_id, now, int(ok),
                      int(complete), total, len(refs), int(note is None), note))
                fetch_id = cursor.lastrowid
                if ok and refs:
                    cursor.executemany("""
                        INSERT INTO board_sightings (board_key, ref, link, first_seen_at,
                                                     last_seen_at, last_seen_fetch_id, title_key)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(board_key, ref) DO UPDATE SET
                            link = excluded.link,
                            last_seen_at = excluded.last_seen_at,
                            last_seen_fetch_id = excluded.last_seen_fetch_id,
                            title_key = COALESCE(excluded.title_key, board_sightings.title_key)
                    """, [(board_key, ref, link, now, now, fetch_id, tk)
                          for ref, (link, tk) in refs.items()])
                return fetch_id
        except sqlite3.Error as e:
            logger.warning(f"Could not record board fetch {board_key}: {e}")
            return None

    @staticmethod
    def _classify_sighting(sighting: Dict[str, Any], latest: Optional[tuple],
                           after: List[tuple], fresh_days: int,
                           gone_after: int, min_span_hours: float = 0) -> Dict[str, Any]:
        """THE CLOSED RULE for one sighting. latest = the board's newest
        evidence fetch (id, fetched_at) or None; after = the board's evidence
        fetches (id, fetched_at) above some floor <= last_seen_fetch_id.
        Gone also needs the missing fetches to span min_span_hours: two
        fetches minutes apart (a manual search right after the pipeline) are
        one look at the board, not two."""
        out = {"status": "unknown", "board_key": sighting["board_key"],
               "checked_at": latest[1] if latest else None, "gone_since": None,
               "first_seen_at": sighting["first_seen_at"],
               "last_seen_at": sighting["last_seen_at"], "relisted_link": None}
        last = sighting["last_seen_fetch_id"]
        if latest is None:
            return {**out, "reason": "no_complete_fetch"}
        if last >= latest[0]:
            checked = _parse_utc(latest[1])
            if checked and datetime.now(timezone.utc) - checked <= timedelta(days=fresh_days):
                return {**out, "status": "listed", "reason": "listed"}
            return {**out, "reason": "stale_check"}
        missed = sorted(f for f in after if f[0] > last)
        if len(missed) >= gone_after:
            first, newest = _parse_utc(missed[0][1]), _parse_utc(missed[-1][1])
            if first and newest and newest - first >= timedelta(hours=min_span_hours):
                return {**out, "status": "gone", "reason": "gone", "gone_since": missed[0][1]}
        return {**out, "reason": "missing_once"}

    @staticmethod
    def _relisted_link(conn, sighting: Dict[str, Any], latest: tuple, ref: str) -> Optional[str]:
        """The link of ANOTHER posting with the same title on the same board in
        its latest evidence fetch (boards often re-post a role under a new id,
        and our title+company dedup keeps the new copy out of the feed) —
        None when there is none."""
        tkey = sighting.get("title_key")
        if not tkey:
            return None
        row = conn.execute(
            "SELECT link FROM board_sightings WHERE board_key = ? AND title_key = ? "
            "AND ref <> ? AND last_seen_fetch_id >= ? ORDER BY last_seen_fetch_id DESC LIMIT 1",
            (sighting["board_key"], tkey, ref, latest[0])).fetchone()
        return row[0] if row else None

    def _listing_statuses(self, conn, links: Dict[int, Any]) -> Dict[int, Dict[str, Any]]:
        """Live listing status for {job_id: link}. Batched: one sighting
        lookup per 500 refs, raw_data read only for rows a board has seen,
        two indexed fetch lookups per involved board."""
        from .config import (POSTING_CHECK_FULL_LIST_PLATFORMS, POSTING_GONE_AFTER_FETCHES,
                             POSTING_GONE_MIN_SPAN_HOURS, POSTING_LISTED_FRESH_DAYS)
        refs = {jid: posting_ref(link) for jid, link in links.items()}
        wanted = sorted({r for r in refs.values() if r})
        by_ref: Dict[str, List[Dict[str, Any]]] = {}
        for i in range(0, len(wanted), 500):
            chunk = wanted[i:i + 500]
            for row in conn.execute(
                    "SELECT board_key, ref, first_seen_at, last_seen_at, last_seen_fetch_id, "
                    f"title_key FROM board_sightings WHERE ref IN ({','.join('?' * len(chunk))})",
                    chunk):
                by_ref.setdefault(row[1], []).append(
                    {"board_key": row[0], "first_seen_at": row[2],
                     "last_seen_at": row[3], "last_seen_fetch_id": row[4], "title_key": row[5]})

        # Where did the row come from? A search site (LinkedIn, Indeed...)
        # stays unknown even when its link points at a company board.
        seen_ids = [jid for jid, r in refs.items() if r in by_ref]
        site: Dict[int, str] = {}
        for i in range(0, len(seen_ids), 500):
            chunk = seen_ids[i:i + 500]
            for jid, raw in conn.execute(
                    f"SELECT id, raw_data FROM scrapes WHERE id IN ({','.join('?' * len(chunk))})",
                    chunk):
                try:
                    site[jid] = str(json.loads(raw or "{}").get("site") or "").strip().lower()
                except Exception:
                    site[jid] = ""
        eligible = {jid for jid in seen_ids if site.get(jid) in POSTING_CHECK_FULL_LIST_PLATFORMS}

        floors: Dict[str, int] = {}
        for jid in eligible:
            for s in by_ref[refs[jid]]:
                k, last = s["board_key"], s["last_seen_fetch_id"]
                floors[k] = min(floors.get(k, last), last)
        latest: Dict[str, Optional[tuple]] = {}
        after: Dict[str, List[tuple]] = {}
        for k, floor in floors.items():
            latest[k] = conn.execute(
                "SELECT id, fetched_at FROM board_fetches WHERE board_key = ? AND evidence = 1 "
                "ORDER BY id DESC LIMIT 1", (k,)).fetchone()
            after[k] = conn.execute(
                "SELECT id, fetched_at FROM board_fetches WHERE board_key = ? AND evidence = 1 "
                "AND id > ?", (k, floor)).fetchall()

        out: Dict[int, Dict[str, Any]] = {}
        for jid, ref in refs.items():
            if ref not in by_ref:
                # No board has listed it since 27c began (or a search-site
                # row) — the caller that knows the site picks not_board.
                out[jid] = _unknown_listing("not_seen")
            elif jid not in eligible:
                out[jid] = _unknown_listing("not_board")
            else:
                results = []
                for s in by_ref[ref]:
                    r = self._classify_sighting(
                        s, latest[s["board_key"]], after[s["board_key"]],
                        POSTING_LISTED_FRESH_DAYS, POSTING_GONE_AFTER_FETCHES,
                        POSTING_GONE_MIN_SPAN_HOURS)
                    if r["reason"] == "gone":
                        # Re-posted under a new link: the role is still on
                        # the board, so it is never hidden.
                        new_link = self._relisted_link(conn, s, latest[s["board_key"]], ref)
                        if new_link:
                            r = {**r, "status": "listed", "reason": "relisted",
                                 "gone_since": None, "relisted_link": new_link}
                    results.append(r)
                out[jid] = min(results, key=lambda r: _LISTING_PRECEDENCE.index(r["reason"]))
        return out

    def get_listing_status(self, job_id: int) -> Dict[str, Any]:
        """Live listing status of one scrapes row (not the denormalized column):
        {"status": "listed"|"gone"|"unknown",
         "reason": "listed"|"relisted"|"gone"|"not_board"|"not_seen"
                   |"no_complete_fetch"|"missing_once"|"stale_check",
         "board_key", "checked_at" (the board's latest evidence fetch),
         "gone_since", "first_seen_at", "last_seen_at", "relisted_link"}.

        THE CLOSED RULE (shared with refresh_listing_status):
          not_board  — the row's raw_data site is not a full-list ATS platform
                       (search sites stay unknown even with a board link)
          not_seen   — no board has listed posting_ref(link) since 27c began
          E = the sighting's board's evidence fetches, newest first
          no_complete_fetch — E is empty
          listed     — last_seen_fetch_id >= E[0].id and E[0] is within
                       POSTING_LISTED_FRESH_DAYS
          stale_check — seen in E[0] but E[0] is older than that
          gone       — the POSTING_GONE_AFTER_FETCHES newest evidence fetches
                       all came after the last sighting, and the missing
                       fetches span at least POSTING_GONE_MIN_SPAN_HOURS;
                       gone_since = the first evidence fetch that missed it
          relisted   — would be gone, but the board's latest evidence fetch
                       lists another posting with the same title (status
                       'listed', relisted_link = that posting's link)
          missing_once — missed by fewer evidence fetches than that, or
                       by fetches too close together
        A later sighting flips gone back to listed. A missing job reads
        unknown/not_seen. Never raises."""
        try:
            conn = self._get_thread_conn()
            row = conn.execute("SELECT link, raw_data FROM scrapes WHERE id = ?",
                               (int(job_id),)).fetchone()
            if not row:
                return _unknown_listing("not_seen")
            try:
                site = str(json.loads(row[1] or "{}").get("site") or "").strip().lower()
            except Exception:
                site = ""
            from .config import POSTING_CHECK_FULL_LIST_PLATFORMS
            if site not in POSTING_CHECK_FULL_LIST_PLATFORMS:
                return _unknown_listing("not_board")
            return self._listing_statuses(conn, {int(job_id): row[0]})[int(job_id)]
        except Exception as e:
            logger.error(f"Error reading listing status for job {job_id}: {e}")
            return _unknown_listing("not_seen")

    def refresh_listing_status(self, job_ids: Optional[List[int]] = None) -> Dict[str, int]:
        """Recompute scrapes.listing_status / listing_checked_at /
        listing_gone_since with the closed rule (see get_listing_status) for
        job_ids, or by default for tiered rows plus every tracked job (a few
        thousand rows: two indexed SELECTs, refs in Python, batched sighting lookups).
        'unknown' stores NULLs. Only rows whose values change are written, in
        one executemany transaction; nothing else on the row is touched.
        Returns {"listed": n, "gone": n, "unknown": n}. Never raises."""
        counts = {"listed": 0, "gone": 0, "unknown": 0}
        try:
            conn = self._get_thread_conn()
            cols = "s.id, s.link, s.listing_status, s.listing_checked_at, s.listing_gone_since"
            rows: List[tuple] = []
            if job_ids is None:
                # Two queries, not one OR: an OR defeats the partial tiered index.
                rows += conn.execute(
                    f"SELECT {cols} FROM scrapes s "
                    "WHERE s.tier IN ('strong', 'worth_a_look')").fetchall()
                rows += conn.execute(
                    f"SELECT {cols} FROM applications a JOIN scrapes s ON s.id = a.job_id"
                ).fetchall()
            else:
                ids = sorted({int(i) for i in job_ids if i is not None})
                for i in range(0, len(ids), 500):
                    chunk = ids[i:i + 500]
                    rows += conn.execute(
                        f"SELECT {cols} FROM scrapes s "
                        f"WHERE s.id IN ({','.join('?' * len(chunk))})", chunk).fetchall()
            current = {r[0]: r for r in rows}
            if not current:
                return counts
            statuses = self._listing_statuses(conn, {jid: r[1] for jid, r in current.items()})
            updates = []
            for jid, r in current.items():
                st = statuses.get(jid) or _unknown_listing("not_seen")
                counts[st["status"]] += 1
                new = ((st["status"], st["checked_at"], st["gone_since"])
                       if st["status"] != "unknown" else (None, None, None))
                if tuple(r[2:5]) != new:
                    updates.append((*new, jid))
            if updates:
                with self.transaction() as wconn:
                    wconn.executemany(
                        "UPDATE scrapes SET listing_status = ?, listing_checked_at = ?, "
                        "listing_gone_since = ? WHERE id = ?", updates)
            if counts["listed"] or counts["gone"]:
                logger.info(f"Listing status refreshed: {counts} ({len(updates)} rows changed)")
            return counts
        except Exception as e:
            logger.error(f"Error refreshing listing status: {e}")
            return counts

    def get_gone_links(self, links: List[str]) -> set:
        """The subset of links whose scrapes row has listing_status 'gone'
        (idx_link lookups in chunks of 500). Never raises."""
        wanted = sorted({l for l in (links or ()) if isinstance(l, str) and l})
        gone: set = set()
        try:
            conn = self._get_thread_conn()
            for i in range(0, len(wanted), 500):
                chunk = wanted[i:i + 500]
                gone.update(r[0] for r in conn.execute(
                    f"SELECT link FROM scrapes WHERE link IN ({','.join('?' * len(chunk))}) "
                    "AND listing_status = 'gone'", chunk))
        except Exception as e:
            logger.error(f"Error reading gone links: {e}")
        return gone

    def get_company_applications(self, company: str,
                                 exclude_job_id: Optional[int] = None) -> List[Dict]:
        """The candidate's earlier applications at the same company (warm_paths.
        normalize_company on both sides; '' never matches) — the candidate's own history,
        not a legitimacy signal. Newest updated_at first:
        [{app_id, job_id, title, company, status, tracked_at (date_applied),
          applied_at (first 'Applied' entry in the card history, else None),
          updated_at, highest_stage, reached_at}]. One join + one ledger read.
        Never raises."""
        from .warm_paths import normalize_company
        target = normalize_company(company) if isinstance(company, str) else ""
        if not target:
            return []
        try:
            conn = self._get_thread_conn()
            rows = conn.execute("""
                SELECT a.id, a.job_id, s.title, s.company, a.status, a.date_applied, a.updated_at
                FROM applications a JOIN scrapes s ON s.id = a.job_id
            """).fetchall()
            apps = [
                {"app_id": r[0], "job_id": r[1], "title": r[2], "company": r[3],
                 "status": r[4], "tracked_at": r[5], "updated_at": r[6]}
                for r in rows
                if (exclude_job_id is None or r[1] != exclude_job_id)
                and normalize_company(r[3] or "") == target
            ]
            if not apps:
                return []
            by_app: Dict[int, List[Dict]] = {}
            for ev in self._fetch_events():
                by_app.setdefault(ev["app_id"], []).append(ev)
            for a in apps:
                reach = self._stage_reach(by_app.get(a["app_id"], []))
                a["applied_at"] = next((h["at"] for h in reach["history"]
                                        if h["status"] == "Applied"), None)
                a["highest_stage"] = reach["highest_stage"]
                a["reached_at"] = reach["reached_at"]
            apps.sort(key=lambda a: str(a.get("updated_at") or ""), reverse=True)
            return apps
        except Exception as e:
            logger.error(f"Error reading applications for company {company}: {e}")
            return []

    def get_posting_quality(self, job_id: int) -> Optional[Dict]:
        """{"job_id", "jd_hash", "model", "checked_at", "result": dict} or None."""
        try:
            row = self._get_thread_conn().execute(
                "SELECT job_id, jd_hash, model, checked_at, result_json FROM posting_quality "
                "WHERE job_id = ?", (int(job_id),)).fetchone()
            if not row:
                return None
            result = json.loads(row[4])
            if not isinstance(result, dict):
                return None
            return {"job_id": row[0], "jd_hash": row[1], "model": row[2],
                    "checked_at": row[3], "result": result}
        except Exception as e:
            logger.error(f"Error reading posting quality for job {job_id}: {e}")
            return None

    def save_posting_quality(self, job_id: int, jd_hash: str, model: str, result: dict) -> bool:
        """Store (replace) the posting quality read with checked_at = now (UTC)."""
        try:
            payload = json.dumps(result, ensure_ascii=False)
            with self.transaction() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO posting_quality "
                    "(job_id, jd_hash, model, checked_at, result_json) VALUES (?, ?, ?, ?, ?)",
                    (int(job_id), jd_hash, model, _utc_now(), payload))
            return True
        except Exception as e:
            logger.error(f"Error saving posting quality for job {job_id}: {e}")
            return False
