"""
FastAPI REST backend for the Next.js frontend (Phase 15).

Wraps existing Python pipeline logic as REST endpoints.
Runs on port 8502; serves the Next.js frontend (port 8503).
"""

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .db_manager import (DBManager, STAGE_RANK, SENT_CV_ANSWERS, OUTCOME_STAGES,
                         OUTCOME_NOTE_MAX)
from .config import (
    KEYWORDS, KEYWORDS_BY_CATEGORY, ALLOWED_LOCATIONS,
    CASCADE_STAGE1_MODEL, CASCADE_STAGE2_MODEL, CASCADE_STAGE3_MODEL,
    DEFAULT_GENERATION_MODEL, AVAILABLE_MODELS, SEARCH_SITES,
    APPLICATION_STANDARD_ANSWERS, AUTO_GENERATE_CV_CL,
    CASCADE_STAGE1_PROMPT, CASCADE_STAGE3_PROMPT,
)
from .candidate_profile import SEARCH
from .gpt_filter import STAGE2_PROMPT_CLAUDE
from .cv_tailor import PROMPT_SUMMARY_TEMPLATE_CLAUDE, PROMPT_EXPERIENCE_TEMPLATE_CLAUDE
from . import posting_check

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Job Agent API", version="1.0.0")
# CORS: ONLY the app's own origins. A wildcard here would let any web page the
# user visits read this API cross-origin from their browser (it serves personal
# data and has cost-incurring endpoints). The Next.js frontend is the only
# legitimate browser caller. When the frontend is opened under another host
# name (a LAN name, a VPN address), list its origins, comma-separated, in the
# CORS_ALLOWED_ORIGINS environment variable; unset, only localhost is allowed.
_DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:8503",
    "http://127.0.0.1:8503",
]
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
] or list(_DEFAULT_ALLOWED_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared DB instance (WAL mode handles concurrent reads)
_db = DBManager()

# Fallback location for company / salary research when a job row or a request
# carries none: the primary metro named in the candidate profile.
_DEFAULT_LOCATION = SEARCH["locations"]["primary_metro_label"]

# Search state (managed by background thread)
_search_state = {
    "active": False,
    "paused": False,
    "progress": 0.0,
    "status_text": "Idle",
    "session_id": None,
    "thread": None,
    "stop_flag": False,
    "session": None,  # the SHARED SessionManager driving the running search
}
# Guards the check-and-claim in /api/search/start so two concurrent requests
# can't both pass the "already active?" check and launch duplicate searches.
_search_lock = threading.Lock()


def _is_nan_or_none(v) -> bool:
    """True for pandas NaN (float), None, or the literal strings 'nan'/'none'."""
    if v is None:
        return True
    if isinstance(v, float):
        import math
        return math.isnan(v)
    if isinstance(v, str) and v.strip().lower() in ("nan", "none", ""):
        return True
    return False


def _merge_without_nan(base: dict, extra: dict) -> dict:
    """Return the entries of extra whose value is not NaN/None (base is not copied in).
    Protects valid top-level DB fields (title, company) from being clobbered by pandas-origin
    NaN values living in raw_data JSON."""
    return {k: v for k, v in extra.items() if not _is_nan_or_none(v)}


# === Pydantic Models ===


class DismissRequest(BaseModel):
    reason: Optional[str] = None
    # When true (or when the reason names a location problem), the candidate's no-go
    # location is learned and future jobs there are dropped by the pre-filter.
    suppress_location: Optional[bool] = None


class ApplicationUpdate(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None


class ManualJobRequest(BaseModel):
    title: str
    company: str
    location: str = ""
    link: str = ""
    description: str = ""   # optional: inbound processes may have no posting


class CompanyIntelRequest(BaseModel):
    # Tolerate null/missing values so Pydantic doesn't 422 the request before the handler
    # can return a helpful 400. A job with company = NULL shouldn't crash the API.
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    location: Optional[str] = None


class SearchStartRequest(BaseModel):
    limit: int = 20
    cascade: bool = True
    resume_path: Optional[str] = None
    base_cv_path: Optional[str] = None
    model_filter: Optional[str] = None
    model_generation: Optional[str] = None
    search_sites: Optional[list[str]] = None
    stage1_model: Optional[str] = None
    stage2_model: Optional[str] = None
    stage3_model: Optional[str] = None
    send_email: bool = False
    target_email: Optional[str] = None
    auto_generate_docs: bool = True


class ChatRequest(BaseModel):
    message: str
    session_id: str = "candidate"


class BatchGenerateRequest(BaseModel):
    job_ids: list[int]


# === Health ===


@app.get("/api/health")
def health():
    try:
        _db._get_thread_conn().execute("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return {"status": "degraded", "db": str(e)}


# === Jobs ===


def _with_ca_highlight(jobs):
    """Annotate each job with ``ca_highlight``: for a multi-location posting, the
    segment inside the candidate's target area that makes it actionable (so the
    candidate sees which of several listed locations is the right one). None
    when not applicable."""
    from .config import extract_actionable_location
    for j in jobs:
        if isinstance(j, dict):
            j["ca_highlight"] = extract_actionable_location(j.get("location") or "")
    return jobs


@app.get("/api/jobs")
def list_jobs(
    session_id: Optional[str] = None,
    match_level: Optional[str] = None,
    reviewed: Optional[bool] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List jobs with optional filters."""
    try:
        conn = _db._get_thread_conn()
        import sqlite3
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        conditions = []
        params = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if match_level:
            conditions.append("match_level = ?")
            params.append(match_level)
        if reviewed is not None:
            conditions.append("reviewed = ?")
            params.append(1 if reviewed else 0)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT id, date, title, company, location, link, created_at,
                   relevance, reasoning, match_level, confidence, session_id,
                   date_posted, keyword_category, reviewed, cv_path, cl_path,
                   cv_quality_flag, cv_audit_summary, cv_domain_used, rejection_reason, tier,
                   comp_text, employment_type, comp_yearly_min, comp_yearly_max, job_level
            FROM scrapes {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        cursor.execute(query, params)
        rows = _with_ca_highlight([dict(r) for r in cursor.fetchall()])
        conn.row_factory = None

        # Total count
        count_query = f"SELECT COUNT(*) FROM scrapes {where}"
        count_params = params[:-2]  # Without limit/offset
        total = conn.execute(count_query, count_params).fetchone()[0]

        return {"jobs": rows, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/latest-session")
def latest_session_jobs(include_irrelevant: bool = False):
    """Get the latest session's results (the 'New Results' view).

    By default returns only relevant (HIGH_MATCH / POTENTIAL_MATCH) jobs.
    Pass include_irrelevant=true to see all jobs from that session.
    """
    session_id = _db.get_latest_session_id()
    if not session_id:
        return {"jobs": [], "total": 0, "session_id": None}

    session_jobs = _db.get_session_results(session_id)

    if not include_irrelevant:
        session_jobs = [
            j for j in session_jobs
            if j.get("relevance") == "relevant"
        ]

    return {
        "jobs": _with_ca_highlight(session_jobs),
        "total": len(session_jobs),
        "session_id": session_id,
    }


@app.get("/api/jobs/unreviewed")
def unreviewed_jobs():
    """Get unreviewed HIGH/POTENTIAL matches.

    Each row carries ``posting_read`` (Phase 27c "Is this posting real?"): the
    deterministic summary of the posting facts, or None. It is display-only —
    the feed's order and membership never depend on it — and a failure there
    never fails the feed. Jobs whose company board no longer lists them are
    already left out by get_unreviewed_jobs."""
    rows = _with_ca_highlight(_db.get_unreviewed_jobs())
    try:
        reads = _feed_posting_reads(rows)
    except Exception as e:
        logger.warning(f"Posting reads for the feed failed: {e}")
        reads = {}
    for r in rows:
        if isinstance(r, dict):
            r["posting_read"] = reads.get(r.get("id"))
    return rows


# ── Feed chips off the request path (Phase 27c) ──────────────────────────────
# feed_reads rebuilds the scanner's company index when it is cold:
# 0.5–2.5 s, against ~40 ms for the feed itself. So the feed only reads a
# cache, and a stale cache (or a row it has not tried yet) is refreshed in a
# background thread. A new row gets its chip on a later load. A failed refresh
# keeps the old chips and is not retried before the TTL, so it cannot loop.
FEED_READS_TTL_SEC = 600
_feed_reads: dict = {"reads": {}, "attempted": set(), "built_at": 0.0, "running": False}
_feed_reads_lock = threading.Lock()
_FEED_READ_FIELDS = ("id", "company", "title", "listing_status", "date_posted")


def _refresh_feed_reads(rows: list) -> None:
    """Thread body: one feed_reads call for the whole feed. Never raises."""
    try:
        reads = posting_check.feed_reads(_db, rows) or {}
    except Exception as e:
        logger.warning(f"Posting reads for the feed failed: {e}")
        reads = None
    with _feed_reads_lock:
        if reads is not None:
            _feed_reads["reads"] = reads
        _feed_reads["attempted"] = {r.get("id") for r in rows}
        _feed_reads["built_at"] = time.time()
        _feed_reads["running"] = False


def _spawn_feed_refresh(rows: list) -> None:
    threading.Thread(target=_refresh_feed_reads, args=(rows,), daemon=True,
                     name="feed-posting-reads").start()


def _feed_posting_reads(rows: list) -> dict:
    """{job_id: posting_read} from the cache, starting a background refresh
    when the cache is older than FEED_READS_TTL_SEC or a row was never tried."""
    ids = {r.get("id") for r in rows if isinstance(r, dict)}
    with _feed_reads_lock:
        stale = (time.time() - _feed_reads["built_at"] > FEED_READS_TTL_SEC
                 or not ids <= _feed_reads["attempted"])
        start = stale and not _feed_reads["running"]
        if start:
            _feed_reads["running"] = True
    if start:
        snapshot = [{k: r.get(k) for k in _FEED_READ_FIELDS} for r in rows if isinstance(r, dict)]
        try:
            _spawn_feed_refresh(snapshot)
        except Exception as e:
            logger.warning(f"Could not start posting reads for the feed: {e}")
            with _feed_reads_lock:
                _feed_reads["running"] = False
    with _feed_reads_lock:
        return dict(_feed_reads["reads"])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int):
    """Get single job detail."""
    try:
        import sqlite3
        conn = _db._get_thread_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scrapes WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        conn.row_factory = None
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        return _with_ca_highlight([dict(row)])[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs")
def create_manual_job(req: ManualJobRequest):
    """Create a manual job entry (User Input).

    Powers the User Input view: saves job, tracks to Kanban,
    marks reviewed, and kicks off CV/CL generation in background.
    """
    job_data = {
        "title": req.title,
        "company": req.company,
        "location": req.location,
        "job_url": req.link or f"manual://{req.company}/{req.title}".replace(" ", "_"),
        "description": req.description,
        "date_posted": datetime.now().strftime("%Y-%m-%d"),
    }
    job_id = _db.save_job_with_relevance(
        job_data,
        relevance="relevant",
        match_level="USER_INPUT",
        confidence=1.0,
        reasoning="Manually submitted",
        tier="strong",
    )
    if not job_id:
        raise HTTPException(status_code=400, detail="Failed to save job (may already exist)")

    # Auto-track to Kanban and mark reviewed (User Input flow)
    _db.track_job(job_id)
    _db.mark_reviewed([job_id])
    _invalidate_stats_cache()

    # Auto-generate CV/CL in background thread
    def _generate():
        try:
            from .docs_builder import generate_package
            from .gpt_filter import load_profile_text

            job = _job_for_generation(job_id)
            if not job:
                logger.error(f"Job {job_id} not found for doc generation")
                return

            profile_text = load_profile_text("resume.pdf")
            # The shared path saves the version, the paths and the QA verdict.
            out = generate_package(job, _db, profile_text, source="user_input")
            logger.info(f"CV/CL {out['status']} for USER_INPUT job {job_id}: {req.title} at {req.company}")
        except Exception as e:
            logger.error(f"Background doc generation failed for USER_INPUT job {job_id}: {e}")

    t = threading.Thread(target=_generate, daemon=True)
    t.start()

    return {"job_id": job_id, "status": "created"}


@app.post("/api/jobs/{job_id}/track")
def track_job(job_id: int):
    """Add job to Kanban tracker and mark as reviewed."""
    success = _db.track_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to track job")
    _db.mark_reviewed([job_id])
    _invalidate_stats_cache()
    spawned = _spawn_hiring_health(job_id)
    return {"status": "tracked", "job_id": job_id, "hiring_health_spawned": spawned}


# ── Hiring health on track (Phase 27c) ───────────────────────────────────────
# Tracking a job researches its company in the background (layoffs, freezes,
# creative-team cuts…) unless that research is still fresh. One run per
# company at a time: two quick Tracks for the same company make one call.
_hiring_health_inflight: set = set()
_hiring_health_lock = threading.Lock()


def _run_hiring_health(company: str, title: Optional[str], key: str) -> None:
    """Thread body: one get_hiring_health call (it has its own hard deadline
    and caches its result). Always releases the in-flight slot."""
    try:
        from .perplexity_utils import get_hiring_health
        get_hiring_health(company, db=_db, job_title=title)
    except Exception as e:
        logger.warning(f"Background hiring health failed for {company}: {e}")
    finally:
        with _hiring_health_lock:
            _hiring_health_inflight.discard(key)


def _spawn_hiring_health(job_id: int) -> bool:
    """Research the tracked job's company in a daemon thread when there is a
    Perplexity key and no fresh cached research. Returns True if spawned.
    Never raises — tracking must succeed whatever happens here."""
    try:
        from .config import USE_PERPLEXITY
        if not USE_PERPLEXITY:
            return False
        job = _db.get_job_details(job_id) or {}
        company = str(job.get("company") or "").strip()
        if _is_nan_or_none(company) or not company or company.lower() == "unknown":
            return False
        title = job.get("title")
        title = None if _is_nan_or_none(title) else (str(title).strip() or None)
        from . import perplexity_utils
        if perplexity_utils.hiring_health_is_fresh(company, _db):
            return False
        key = company.lower()
        with _hiring_health_lock:
            if key in _hiring_health_inflight:
                return False
            _hiring_health_inflight.add(key)
        try:
            threading.Thread(target=_run_hiring_health, args=(company, title, key),
                             daemon=True).start()
        except Exception:
            with _hiring_health_lock:
                _hiring_health_inflight.discard(key)
            raise
        return True
    except Exception as e:
        logger.warning(f"Could not start hiring health for job {job_id}: {e}")
        return False


# ── Posting check (Phase 27c "Is this posting real?") ────────────────────────
# Facts about a posting, never a score: nothing here changes tier, confidence,
# match_level, reviewed or the feed's order.


@app.get("/api/jobs/{job_id}/posting-check")
def get_posting_check(job_id: int):
    """The job's posting facts and their overall read. No network, no LLM."""
    try:
        check = posting_check.build_posting_check(_db, job_id)
    except Exception as e:
        logger.error(f"Posting check failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Posting check failed")
    if check is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return check


@app.post("/api/jobs/{job_id}/posting-check/quality")
def read_posting_quality(job_id: int):
    """The "Read the posting" button: one cheap model read of how specific the posting is
    (tools, team, first-year scope, pay, contradictions, boilerplate), every
    fact quoted from the posting. Cached for 30 days unless the text changes.
    Synchronous. Returns the read's status and the refreshed check."""
    if not _db.get_job_details(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        out = posting_check.run_quality_read(_db, job_id) or {}
    except Exception as e:
        logger.error(f"Posting quality read failed for job {job_id}: {e}")
        out = {"status": "failed"}
    try:
        check = posting_check.build_posting_check(_db, job_id)
    except Exception as e:
        logger.error(f"Posting check failed for job {job_id}: {e}")
        check = None
    return {"status": out.get("status") or "failed", "check": check}


@app.post("/api/jobs/{job_id}/dismiss")
def dismiss_job(job_id: int, req: DismissRequest):
    """Dismiss job + optional rejection reason.

    Feedback loop: if the dismissal is location-driven (explicit
    suppress_location, or a reason that names a location problem), learn that
    location as a no-go so future jobs there are dropped by the pre-filter.
    add_location_suppression refuses to blacklist the primary metro / remote.
    The logic lives in DBManager.dismiss_job so the chat dismiss tool runs the
    very same feedback loop.
    """
    suppressed = _db.dismiss_job(job_id, req.reason, req.suppress_location)
    _invalidate_stats_cache()
    return {"status": "dismissed", "job_id": job_id, "suppressed_location": suppressed}


def _job_for_generation(job_id: int) -> Optional[dict]:
    """The scrapes row merged with its raw_data, ready for generate_package, or
    None when the job does not exist. raw_data values that are NaN/None never
    clobber DB fields, and the DB identity columns always win: jobspy raw_data
    carries its own string "id" ("in-…"), which would cost the shared path its
    job id (no saved version, no paths, no QA verdict)."""
    import sqlite3
    conn = _db._get_thread_conn()
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM scrapes WHERE id = ?", (job_id,)).fetchone()
    finally:
        conn.row_factory = None
    if not row:
        return None
    job = dict(row)
    raw = job.get("raw_data")
    if raw:
        try:
            job.update(_merge_without_nan(job, json.loads(raw)))
        except Exception:
            pass
    for key in ("id", "link", "session_id", "created_at"):
        job[key] = row[key] if key in row.keys() else job.get(key)
    return job


@app.post("/api/jobs/{job_id}/generate-docs")
def generate_docs(job_id: int):
    """Generate a CV + cover letter synchronously (the frontend waits) through
    the shared generation path, which saves the version, the paths and the QA
    verdict."""
    # Get job data first
    try:
        job = _job_for_generation(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        from .docs_builder import generate_package
        from .gpt_filter import load_profile_text
        if not (job.get("title") and job.get("company")):
            raise HTTPException(
                status_code=400,
                detail="Cannot generate CV: job is missing title or company.",
            )
        profile_text = load_profile_text("resume.pdf")
        out = generate_package(job, _db, profile_text, source="api")
        if out["status"] == "in_progress":
            return {"status": "in_progress", "job_id": job_id,
                    "error": "This job's documents are already being generated"}
        if out["cv_path"]:
            logger.info(f"CV/CL generated for job {job_id}: {out['cv_path']}")
            return {"status": "completed", "job_id": job_id, "cv_path": out["cv_path"],
                    "cl_path": out["cl_path"], "cv_quality_flag": out["flag"]}
        if out.get("error"):        # generation raised: same 500 as before the shared path
            raise HTTPException(status_code=500, detail=f"CV generation failed: {out['error']}")
        logger.warning(f"CV generation returned no CV for job {job_id}")
        return {"status": "failed", "job_id": job_id, "error": "Generation returned no files"}
    except HTTPException:
        # Preserve intentional 400s (missing title/company) — don't wrap as 500.
        raise
    except Exception as e:
        logger.error(f"Doc generation failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"CV generation failed: {e}")


@app.get("/api/generating")
def generating_jobs():
    """The jobs whose CV + cover letter are being generated right now (a click,
    a batch, or the background generation a job added by hand starts). The
    Kanban polls this while it is not empty: the card says "Generating…" and
    its View button appears by itself when the documents are saved."""
    from .docs_builder import generating_job_ids
    return {"job_ids": generating_job_ids()}


# Batch CV generation runs in the BACKGROUND. At the xhigh generation pin one
# job — intel, CV, truth judge, hiring-manager grader, cover letter — takes
# roughly 4-6 minutes, so a single synchronous request for several jobs
# outlived every proxy timeout: the UI reported failure ("socket hang up" in
# the frontend proxy log) while the CVs trickled in anyway. POST starts a
# worker and returns at once; the Kanban polls /api/batch-generate/status.
# One batch at a time: a double-click must not start two xhigh generation
# loops in parallel.
_batch_lock = threading.Lock()
_batch_state: dict = {"running": False, "total": 0, "done": 0, "current": None,
                      "results": [], "error": None,
                      "started_at": None, "finished_at": None}


def _batch_snapshot() -> dict:
    with _batch_lock:
        return {**_batch_state, "results": list(_batch_state["results"])}


def _process_batch_job(job_id: int, profile_text: str) -> dict:
    """One job of a batch: Perplexity intel if uncached, then a CV + cover
    letter if the job has none yet. Never raises — errors land in the result."""
    from .docs_builder import generate_package

    result = {"job_id": job_id, "cv": None, "intel": None, "error": None}
    try:
        import sqlite3
        conn = _db._get_thread_conn()
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM scrapes WHERE id = ?", (job_id,)).fetchone()
        conn.row_factory = None
        if not row:
            result["error"] = "Job not found"
            return result

        job = dict(row)
        title = job.get("title", "")
        company = job.get("company", "")
        location = job.get("location", "") or _DEFAULT_LOCATION

        # Enrich with Perplexity if no cached intel
        try:
            from .config import USE_PERPLEXITY
            if USE_PERPLEXITY:
                from .perplexity_utils import get_company_profile, get_hiring_intel, get_salary_intel
                cached = _db.get_company_intel(company)
                if not cached or not cached.get("company_profile"):
                    get_company_profile(company, db=_db)
                    get_hiring_intel(company, title, db=_db)
                    get_salary_intel(title, company, location, db=_db)
                    result["intel"] = "generated"
                else:
                    result["intel"] = "cached"
                # Hiring health has its own 14-day cache, so it is pre-warmed
                # even when the profile is cached. Never fails the batch job.
                try:
                    from .perplexity_utils import get_hiring_health
                    get_hiring_health(company, db=_db, job_title=title)
                except Exception as e:
                    logger.warning(f"Batch hiring health failed for {company}: {e}")
        except Exception as e:
            logger.warning(f"Batch intel failed for {title} @ {company}: {e}")
            result["intel"] = f"error: {e}"

        # Generate CV if not already generated
        if not job.get("cv_path"):
            # raw_data merged back without NaN poisoning, DB identity kept.
            job = _job_for_generation(job_id) or job
            if not (job.get("title") and job.get("company")):
                logger.warning(f"Job {job_id} missing title/company — skipping CV gen")
                result["cv"] = "skipped: missing title or company"
                return result
            # The shared path saves the version, the paths and the QA verdict.
            out = generate_package(job, _db, profile_text, source="batch")
            if out["status"] == "in_progress":
                result["cv"] = "skipped: already generating"
            elif out["cv_path"]:
                result["cv"] = out["cv_path"]
            elif out.get("error"):
                logger.error(f"Batch CV gen failed for job {job_id}: {out['error']}")
                result["cv"] = f"error: {out['error']}"
            else:
                result["cv"] = "failed"
        else:
            result["cv"] = "exists"

    except Exception as e:
        result["error"] = str(e)

    return result


def _run_batch(job_ids: list) -> None:
    from .gpt_filter import load_profile_text
    try:
        profile_text = load_profile_text("resume.pdf")
        for job_id in job_ids:
            with _batch_lock:
                _batch_state["current"] = job_id
            result = _process_batch_job(job_id, profile_text)
            with _batch_lock:
                _batch_state["results"].append(result)
                _batch_state["done"] += 1
            logger.info(f"Batch: job {job_id} — cv={result['cv']}, intel={result['intel']}")
    except Exception as e:   # e.g. the profile PDF is unreadable
        logger.error(f"Batch generation aborted: {e}")
        with _batch_lock:
            _batch_state["error"] = str(e)
    finally:
        with _batch_lock:
            _batch_state.update(running=False, current=None,
                                finished_at=datetime.now().isoformat(timespec="seconds"))


@app.post("/api/batch-generate")
def batch_generate(req: BatchGenerateRequest):
    """Start a background batch: CVs + Perplexity intel for several jobs.
    Returns immediately; poll GET /api/batch-generate/status for progress."""
    job_ids = list(dict.fromkeys(req.job_ids))   # de-dupe, keep order
    if not job_ids:
        raise HTTPException(status_code=400, detail="No jobs selected")
    with _batch_lock:
        if _batch_state["running"]:
            raise HTTPException(status_code=409, detail="A batch generation is already running")
        _batch_state.update(running=True, total=len(job_ids), done=0, current=None,
                            results=[], error=None,
                            started_at=datetime.now().isoformat(timespec="seconds"),
                            finished_at=None)
    threading.Thread(target=_run_batch, args=(job_ids,), daemon=True,
                     name="batch-generate").start()
    return {"status": "started", "total": len(job_ids)}


@app.get("/api/batch-generate/status")
def batch_generate_status():
    """Progress of the current (or last) batch: running, done/total, the job
    being worked on, and per-job results so far."""
    return _batch_snapshot()

@app.get("/api/jobs/{job_id}/cv")
def download_cv(job_id: int):
    """Download generated CV."""
    try:
        import sqlite3
        conn = _db._get_thread_conn()
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT cv_path FROM scrapes WHERE id = ?", (job_id,)).fetchone()
        conn.row_factory = None
        if not row or not row["cv_path"]:
            raise HTTPException(status_code=404, detail="CV not generated yet")
        path = row["cv_path"]
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="CV file not found on disk")
        return FileResponse(path, filename=os.path.basename(path))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}/cl")
def download_cl(job_id: int):
    """Download generated cover letter."""
    try:
        import sqlite3
        conn = _db._get_thread_conn()
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT cl_path FROM scrapes WHERE id = ?", (job_id,)).fetchone()
        conn.row_factory = None
        if not row or not row["cl_path"]:
            raise HTTPException(status_code=404, detail="Cover letter not generated yet")
        path = row["cl_path"]
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Cover letter file not found on disk")
        return FileResponse(path, filename=os.path.basename(path))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Company Intel ===


def _cached_hiring_health(company_name: str, job_title: Optional[str]) -> Optional[str]:
    """Hiring health without a network call: the free scanner signal plus
    whatever public facts are already cached. Never raises."""
    try:
        from .perplexity_utils import get_hiring_health
        return get_hiring_health(company_name, db=_db, job_title=job_title, cached_only=True)
    except Exception as e:
        logger.warning(f"Hiring health read failed for {company_name}: {e}")
        return None


@app.get("/api/company-intel/{company_name}")
def get_company_intel(company_name: str, job_title: Optional[str] = None):
    """Return cached company intel (profile + hiring + salary intel + salary recommendation) if available."""
    try:
        intel = _db.get_company_intel(company_name)
        # Hiring intel is JOB-specific content — read the per-role key
        # (perplexity_utils writes it that way). The legacy company-level key
        # is only safe when no job_title is in play: serving it for a
        # different role put one role's brief on another role's card at the
        # same company.
        import re as _re
        if job_title:
            _safe = _re.sub(r"[^A-Za-z0-9 &/+-]+", "", str(job_title))[:60]
            hiring_key = f"__hiring_intel__{company_name}__{_safe}"[:200]
        else:
            hiring_key = f"__hiring_intel__{company_name}"
        hiring_row = _db.get_company_intel(hiring_key)

        salary_intel = None
        salary_recommendation = None
        if job_title:
            salary_key = f"__salary_intel__{company_name}__{job_title}"
            salary_row = _db.get_company_intel(salary_key)
            salary_intel = salary_row.get("company_profile") if salary_row else None

            salary_rec_key = f"__salary_rec__{company_name}__{job_title}"
            salary_rec_row = _db.get_company_intel(salary_rec_key)
            salary_recommendation = salary_rec_row.get("company_profile") if salary_rec_row else None

        return {
            "company_name": company_name,
            "company_profile": intel.get("company_profile") if intel else None,
            "hiring_intel": hiring_row.get("company_profile") if hiring_row else None,
            "hiring_health": _cached_hiring_health(company_name, job_title),
            "salary_intel": salary_intel,
            "salary_recommendation": salary_recommendation,
            "careers_url": intel.get("careers_url") if intel else None,
            "last_checked": intel.get("last_checked") if intel else None,
        }
    except Exception as e:
        logger.error(f"Error reading company intel for {company_name}: {e}")
        return {
            "company_name": company_name,
            "company_profile": None,
            "hiring_intel": None,
            "hiring_health": None,
            "salary_intel": None,
            "salary_recommendation": None,
            "careers_url": None,
            "last_checked": None,
        }


@app.post("/api/company-intel/enrich")
def enrich_company_intel(req: CompanyIntelRequest):
    """Trigger Perplexity enrichment + Opus salary recommendation for a company."""
    # Validate minimum — handler-level check gives a useful message vs Pydantic's opaque 422.
    company_name = (req.company_name or "").strip()
    if not company_name:
        raise HTTPException(
            status_code=400,
            detail="company_name is required — this job has no company recorded in the database.",
        )
    job_title = (req.job_title or "").strip()

    profile = None
    hiring = None
    health = None
    salary = None
    salary_rec = None
    errors = []

    try:
        from .perplexity_utils import (
            get_company_profile, get_hiring_intel, get_hiring_health, get_salary_intel,
            generate_salary_recommendation,
        )
        from .gpt_filter import load_profile_text
    except ImportError as e:
        logger.error(f"Failed to import enrichment modules: {e}")
        errors.append(f"Import error: {e}")
        # Return what we have from DB cache
        try:
            intel = _db.get_company_intel(company_name)
        except Exception:
            intel = None
        return {
            "company_name": company_name,
            "company_profile": intel.get("company_profile") if intel else None,
            "hiring_intel": None,
            "hiring_health": None,
            "salary_intel": None,
            "salary_recommendation": None,
            "careers_url": intel.get("careers_url") if intel else None,
            "last_checked": intel.get("last_checked") if intel else None,
            "errors": errors,
        }

    # Fetch Perplexity data — each call is independent, catch individually
    try:
        profile = get_company_profile(company_name, db=_db)
    except Exception as e:
        logger.warning(f"Perplexity company profile failed for {company_name}: {e}")
        errors.append(f"Company profile: {e}")

    try:
        hiring = get_hiring_intel(company_name, job_title, db=_db)
    except Exception as e:
        logger.warning(f"Perplexity hiring intel failed for {company_name}: {e}")
        errors.append(f"Hiring intel: {e}")

    # Hiring health: scanner signal + dated, sourced public facts (company name only)
    try:
        health = get_hiring_health(company_name, db=_db, job_title=job_title)
    except Exception as e:
        logger.warning(f"Perplexity hiring health failed for {company_name}: {e}")
        errors.append(f"Hiring health: {e}")

    try:
        salary = get_salary_intel(
            job_title, company_name, req.location or _DEFAULT_LOCATION, db=_db
        )
    except Exception as e:
        logger.warning(f"Perplexity salary intel failed for {company_name}: {e}")
        errors.append(f"Salary intel: {e}")

    # Generate Opus salary recommendation
    try:
        profile_text = load_profile_text("resume.pdf")
        job_dict = {
            "title": job_title,
            "company": company_name,
            "location": req.location or _DEFAULT_LOCATION,
            "description": "",
        }
        # Try to load description from DB
        try:
            import sqlite3
            conn = _db._get_thread_conn()
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT raw_data FROM scrapes WHERE company = ? AND title = ? ORDER BY created_at DESC LIMIT 1",
                (company_name, job_title),
            ).fetchone()
            conn.row_factory = None
            if row and row["raw_data"]:
                raw = json.loads(row["raw_data"])
                job_dict["description"] = raw.get("description", "")
        except Exception:
            pass

        intel_dict = {
            "company_profile": profile,
            "salary_intel": salary,
            "hiring_intel": hiring,
        }
        salary_rec = generate_salary_recommendation(
            job=job_dict,
            company_intel=intel_dict,
            profile_text=profile_text,
            db=_db,
        )
    except Exception as e:
        logger.warning(f"Salary recommendation generation failed: {e}")
        errors.append(f"Salary recommendation: {e}")

    # Re-read from DB to get the latest state (including careers_url, last_checked)
    try:
        intel = _db.get_company_intel(company_name)
        import re as _re
        _safe = _re.sub(r"[^A-Za-z0-9 &/+-]+", "", str(job_title or ""))[:60]
        hiring_key = (f"__hiring_intel__{company_name}__{_safe}"[:200]
                      if job_title else f"__hiring_intel__{company_name}")
        hiring_row = _db.get_company_intel(hiring_key)
        salary_key = f"__salary_intel__{company_name}__{job_title}"
        salary_row = _db.get_company_intel(salary_key)
        salary_rec_key = f"__salary_rec__{company_name}__{job_title}"
        salary_rec_row = _db.get_company_intel(salary_rec_key)
    except Exception as e:
        logger.error(f"Error re-reading company intel from DB: {e}")
        intel = hiring_row = salary_row = salary_rec_row = None

    return {
        "company_name": company_name,
        "company_profile": profile or (intel.get("company_profile") if intel else None),
        "hiring_intel": hiring or (hiring_row.get("company_profile") if hiring_row else None),
        "hiring_health": health or _cached_hiring_health(company_name, job_title),
        "salary_intel": salary or (salary_row.get("company_profile") if salary_row else None),
        "salary_recommendation": salary_rec or (salary_rec_row.get("company_profile") if salary_rec_row else None),
        "careers_url": intel.get("careers_url") if intel else None,
        "last_checked": intel.get("last_checked") if intel else None,
        "errors": errors,
    }


# === Applications (Kanban) ===


@app.get("/api/applications")
def list_applications():
    """Get all Kanban applications with job details."""
    return _db.get_applications()


# Stale-pending window: engine subprocess is capped at OUTREACH_ENGINE_TIMEOUT
# but the post-engine phase (4 Perplexity calls + one Opus synthesis, plus
# rate-limiter waits) is not — 2x engine timeout alone undercut legitimate
# runs. Engine cap + 15 min covers the aux phase generously.
def _intel_stale_seconds() -> int:
    from .config import OUTREACH_ENGINE_TIMEOUT_SEC
    return OUTREACH_ENGINE_TIMEOUT_SEC + 900

# At most 2 concurrent generations: each spawns an engine subprocess + paid API
# calls; 10 rapid Applied transitions must queue, not stampede.
_intel_semaphore = threading.Semaphore(2)

# Statuses where generating (or regenerating) intel makes sense.
_INTEL_ELIGIBLE_STATUSES = {
    "To Apply", "Applied", "Screening Call", "Interview", "Final Interview", "Offer",
}


def _spawn_outreach_intel(app_id: int, force: bool = False) -> bool:
    """Kick off outreach-intel generation in a daemon thread.

    Guards (each closes a reviewed hole):
    - atomic DB claim (claim_application_intel) — two rapid transitions can't
      double-spawn a paid run (TOCTOU fix);
    - already-ready rows are NOT regenerated on a mere status round-trip
      (Applied→Ghosted→Applied); explicit regeneration passes force=True;
    - closed cards (Rejected / Ghosted / I Withdrew) never generate;
    - global semaphore caps concurrent engine subprocesses at 2.
    Returns True if spawned."""
    from .config import OUTREACH_INTEL_ENABLED
    if not OUTREACH_INTEL_ENABLED:
        return False
    row = _db.get_application(app_id)
    if not row:
        return False
    if row.get("status") not in _INTEL_ELIGIBLE_STATUSES:
        return False
    if row.get("intel_status") == "ready" and not force:
        return False  # playbook already exists — regenerate only explicitly
    if not _db.claim_application_intel(app_id, _intel_stale_seconds()):
        return False  # a live pending run owns the claim (double-spawn guard)

    def _run():
        try:
            with _intel_semaphore:
                from .outreach_intel import generate_outreach_intel
                generate_outreach_intel(app_id, _db)
        except Exception as e:
            logger.error(f"Outreach intel generation failed for application {app_id}: {e}")
            _db.update_application_intel(app_id, "failed")

    threading.Thread(target=_run, daemon=True).start()
    return True


# ── Interview prep (spawns on the transition into an interview stage) ─────────
def _prep_stale_seconds() -> int:
    # Company-pulse subprocess + a few Perplexity calls + one Opus synthesis.
    from .config import OUTREACH_ENGINE_TIMEOUT_SEC
    return OUTREACH_ENGINE_TIMEOUT_SEC + 900

_prep_semaphore = threading.Semaphore(2)

# Prep only makes sense once an interview is actually on the table.
_PREP_ELIGIBLE_STATUSES = {
    "Screening Call", "Interview", "Final Interview", "Offer",
}


def _run_interview_prep(app_id: int, stage: Optional[str]) -> None:
    """Generate the sheet for `stage`, then remember the stage it was built
    for. generate_interview_prep takes stage= since Phase 27; an older
    two-argument signature is still called the old way."""
    from .interview_prep import generate_interview_prep
    try:
        result = generate_interview_prep(app_id, _db, stage=stage)
    except TypeError as e:
        # Only the "does not take stage" case falls back — any other TypeError
        # is a real failure and must not trigger a second paid generation.
        if "unexpected keyword argument 'stage'" not in str(e):
            raise
        result = generate_interview_prep(app_id, _db)
    if stage and not (isinstance(result, dict) and result.get("error")):
        _db.set_application_prep_stage(app_id, stage)


def _restore_previous_prep_if_failed(app_id: int, previous: tuple) -> bool:
    """After an AUTOMATIC stage rebuild: if it failed (quality floor or an
    exception) and the card had a ready sheet whose file is still on disk,
    put that sheet back — status, file, its stage and its date — so it stays
    readable on the card (and after a later Rejected). A user-clicked
    "Prep me" is not masked; its failure shows as a failure."""
    prev_status, prev_path, prev_stage, prev_generated_at = previous
    if prev_status != "ready" or not prev_path or not os.path.exists(prev_path):
        return False
    now = _db.get_application(app_id) or {}
    if now.get("prep_status") != "failed":
        return False
    restored = _db.restore_application_prep(app_id, prev_path, prev_stage, prev_generated_at)
    if restored:
        logger.warning(f"Interview prep rebuild failed for application {app_id}; "
                       f"kept the previous sheet (built for {prev_stage or 'an earlier stage'})")
    return restored


def _spawn_interview_prep(app_id: int, force: bool = False) -> bool:
    """Kick off interview-prep generation in a daemon thread. Same guard chain
    as _spawn_outreach_intel (atomic claim, no respawn of ready rows unless
    forced, eligible-status gate, semaphore). Returns True if spawned.

    A ready sheet is rebuilt WITHOUT force only when the card now sits at a
    HIGHER interview stage than the sheet was built for (applications.
    prep_stage) — a round-trip or a move down keeps the existing sheet."""
    from .config import INTERVIEW_PREP_ENABLED
    if not INTERVIEW_PREP_ENABLED:
        return False
    row = _db.get_application(app_id)
    if not row:
        return False
    stage = row.get("status")
    if stage not in _PREP_ELIGIBLE_STATUSES:
        return False
    if row.get("prep_status") == "ready" and not force:
        # A sheet from before prep_stage existed has no recorded stage: treat
        # it as built for the earliest one, so only a move past Screening Call
        # refreshes it.
        built_for = row.get("prep_stage") or "Screening Call"
        if STAGE_RANK.get(stage, 0) <= STAGE_RANK.get(built_for, 0):
            return False
    # A user-clicked "Prep me" (force) is an explicit "run it now" — let it
    # reclaim even a fresh pending row (stale=0). The 23-min stale window only
    # exists to protect a live in-flight daemon from an AUTO respawn; a manual
    # click overrides it (otherwise a killed-daemon pending row is stuck for
    # the full window with no self-heal, since a stage move doesn't re-stamp).
    stale = 0 if force else _prep_stale_seconds()
    # The sheet the candidate can read now, remembered BEFORE the claim flips the row to
    # 'pending': an automatic rebuild that fails must not take it away.
    previous = (row.get("prep_status"), row.get("prep_path"),
                row.get("prep_stage"), row.get("prep_generated_at"))
    if not _db.claim_application_prep(app_id, stale):
        return False

    def _run():
        moved_up = False
        try:
            with _prep_semaphore:
                _run_interview_prep(app_id, stage)
            # The card may have moved to a higher stage while this sheet was
            # being built (that move could not claim the busy row).
            after = _db.get_application(app_id) or {}
            moved_up = (after.get("prep_status") == "ready"
                        and STAGE_RANK.get(after.get("status"), 0) > STAGE_RANK.get(stage, 0))
        except Exception as e:
            logger.error(f"Interview prep generation failed for application {app_id}: {e}")
            _db.update_application_prep(app_id, "failed")
        if not force:
            _restore_previous_prep_if_failed(app_id, previous)
        if moved_up:
            _spawn_interview_prep(app_id)  # builds the sheet for the stage the candidate is at now

    threading.Thread(target=_run, daemon=True).start()
    return True


def on_application_status_change(app_id: int, old_status: Optional[str],
                                 new_status: Optional[str], source: str = "app") -> dict:
    """The ONE place a card move triggers its side effects — called by the
    Kanban PUT and by the chat status tool, after the status (and its ledger
    event) is committed. The chat path used to skip all of this.

    - move to Applied           → outreach playbook
    - move INTO an interview stage (INTERVIEW_PREP_TRIGGER_STAGES, Offer
      included) → interview prep; an existing sheet is rebuilt only for a
      higher stage than it was built for
    - any real move             → stats cache invalidated

    The spawns only claim the row here (a millisecond DB write, so the panel
    shows "pending" straight away); the generation itself runs in a daemon
    thread and the caller returns immediately. Never raises."""
    result = {"intel_spawned": False, "prep_spawned": False}
    if not new_status or new_status == old_status:
        return result
    try:
        if new_status == "Applied":
            result["intel_spawned"] = _spawn_outreach_intel(app_id)
        from .config import INTERVIEW_PREP_TRIGGER_STAGES
        if new_status in INTERVIEW_PREP_TRIGGER_STAGES:
            result["prep_spawned"] = _spawn_interview_prep(app_id)
    except Exception as e:
        logger.error(f"Status side effects failed for application {app_id} "
                     f"({old_status} -> {new_status}, via {source}): {e}")
    _invalidate_stats_cache()
    return result


@app.put("/api/applications/{app_id}")
def update_application(app_id: int, req: ApplicationUpdate):
    """Update application status/notes.

    A real status change is recorded in the outcome ledger (same transaction
    as the update) and then runs on_application_status_change: outreach
    playbook on the move to Applied, interview prep on moves into an
    interview stage."""
    try:
        import sqlite3
        conn = _db._get_thread_conn()
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
        conn.row_factory = None
        if not row:
            raise HTTPException(status_code=404, detail="Application not found")

        status = req.status or row["status"]
        notes = req.notes if req.notes is not None else row["notes"]
        if not _db.update_application(app_id, status, notes, source="app"):
            # Status and ledger event roll back together — never report a
            # move that was not saved.
            raise HTTPException(status_code=500, detail="Could not save the application update")

        effects = on_application_status_change(app_id, row["status"], status, source="app")
        _invalidate_stats_cache()
        return {"status": "updated", "app_id": app_id, **effects}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SentCvRequest(BaseModel):
    # 'as_is' (sent the CV we generated) | 'edited' (sent it with the candidate's edits)
    # | 'own' (rewrote it / sent the candidate's own)
    answer: str


@app.put("/api/applications/{app_id}/sent-cv")
def set_application_sent_cv(app_id: int, req: SentCvRequest):
    """The candidate's one-click answer to "Did you send this CV?" on the Kanban
    card. Stored as a confirmed link to the version that was active when the candidate
    applied; the candidate can change the answer (the latest one wins)."""
    if req.answer not in SENT_CV_ANSWERS:
        raise HTTPException(status_code=400,
                            detail=f"answer must be one of {list(SENT_CV_ANSWERS)}")
    if not _db.get_application(app_id):
        raise HTTPException(status_code=404, detail="Application not found")
    if not _db.set_sent_document(app_id, req.answer, provenance="confirmed"):
        raise HTTPException(status_code=500, detail="Could not save the answer")
    return {"app_id": app_id, "sent_cv": req.answer}


class OutcomeRequest(BaseModel):
    # The furthest stage the process reached (OUTCOME_STAGES) and, optionally,
    # what happened — one line in the candidate's words.
    stage: str
    what_happened: Optional[str] = None


@app.put("/api/applications/{app_id}/outcome")
def set_application_outcome(app_id: int, req: OutcomeRequest):
    """The candidate's answer on a closed card (Rejected / Ghosted / I Withdrew): how
    far the process got and what happened — any time while the card is closed.
    The candidate's stage becomes the card's furthest stage in the outcome
    ledger (nothing is deleted); the latest answer wins. Shown on the card and
    on the Stats page."""
    if req.stage not in OUTCOME_STAGES:
        raise HTTPException(status_code=400,
                            detail=f"stage must be one of {list(OUTCOME_STAGES)}")
    if req.what_happened and len(req.what_happened) > OUTCOME_NOTE_MAX:
        raise HTTPException(status_code=400,
                            detail=f"what_happened is limited to {OUTCOME_NOTE_MAX} characters")
    result = _db.set_application_outcome(app_id, req.stage, req.what_happened)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="Application not found")
    if result == "not_closed":
        raise HTTPException(status_code=409,
                            detail="Only a closed card (Rejected, Ghosted or I Withdrew) "
                                   "has an outcome")
    if result != "saved":
        raise HTTPException(status_code=500, detail="Could not save the outcome")
    _invalidate_stats_cache()
    return {"app_id": app_id, **_db.get_card_outcome(app_id)}


class EventCorrectionRequest(BaseModel):
    note: Optional[str] = None


@app.get("/api/applications/{app_id}/events")
def list_application_events(app_id: int):
    """The outcome ledger for one application (the maintainer's audit view — no UI):
    every event, corrected ones included, plus the derived stage reach. Works
    for a deleted application too: its events are kept."""
    return {"app_id": app_id,
            "events": _db.get_application_events(app_id),
            "reach": _db.get_stage_reach(app_id)}


@app.post("/api/applications/{app_id}/events/{event_id}/correct")
def correct_application_event(app_id: int, event_id: int,
                              req: Optional[EventCorrectionRequest] = None):
    """Invalidate one ledger event (the maintainer only — no UI): e.g. a mis-click the
    10-minute rule did not catch. Nothing is deleted; a 'correction' event is
    appended. Correcting a correction restores the original event."""
    correction_id = _db.correct_application_event(
        app_id, event_id, source="maintainer", note=(req.note if req else None))
    if not correction_id:
        raise HTTPException(status_code=404, detail="Event not found for this application")
    _invalidate_stats_cache()
    return {"app_id": app_id, "corrected_event_id": event_id,
            "correction_event_id": correction_id,
            "reach": _db.get_stage_reach(app_id)}


@app.get("/api/applications/{app_id}/intel")
def get_application_intel(app_id: int):
    """Outreach-intel state + content for an application."""
    row = _db.get_application(app_id)
    if not row:
        raise HTTPException(status_code=404, detail="Application not found")
    content = None
    path = row.get("intel_path")
    if row.get("intel_status") == "ready" and path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed reading intel file {path}: {e}")
    return {
        "app_id": app_id,
        "intel_status": row.get("intel_status"),
        "intel_path": path,
        "intel_generated_at": row.get("intel_generated_at"),
        "content": content,
    }


@app.post("/api/applications/{app_id}/intel")
def regenerate_application_intel(app_id: int):
    """Force (re)generation of outreach intel for an application."""
    row = _db.get_application(app_id)
    if not row:
        raise HTTPException(status_code=404, detail="Application not found")
    spawned = _spawn_outreach_intel(app_id, force=True)
    return {"app_id": app_id, "spawned": spawned,
            "detail": None if spawned else
            "generation disabled, already running, or card is closed"}


@app.get("/api/applications/{app_id}/prep")
def get_application_prep(app_id: int):
    """Interview-prep state + content for an application."""
    row = _db.get_application(app_id)
    if not row:
        raise HTTPException(status_code=404, detail="Application not found")
    content = None
    path = row.get("prep_path")
    if row.get("prep_status") == "ready" and path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed reading prep file {path}: {e}")
    return {
        "app_id": app_id,
        "prep_status": row.get("prep_status"),
        "prep_path": path,
        "prep_generated_at": row.get("prep_generated_at"),
        "content": content,
    }


@app.post("/api/applications/{app_id}/prep")
def regenerate_application_prep(app_id: int):
    """Force (re)generation of interview prep — the Kanban 'Prep me' button.
    Works from any eligible stage (Screening Call and up)."""
    row = _db.get_application(app_id)
    if not row:
        raise HTTPException(status_code=404, detail="Application not found")
    spawned = _spawn_interview_prep(app_id, force=True)
    detail = None
    if not spawned:
        # Distinguish the two refusal reasons so the UI message is truthful.
        eligible = (row.get("status") in _PREP_ELIGIBLE_STATUSES)
        detail = ("a prep run is already in progress — try again in a moment"
                  if eligible else
                  "prep is generated once an interview is booked (Screening Call and up)")
    return {"app_id": app_id, "spawned": spawned, "detail": detail}


# === Warm Paths (Spec M3) ===


def _public_warm_row(m: dict) -> dict:
    """Strip PII not needed by the UI (the CSV's email column) from a match."""
    return {k: v for k, v in m.items() if k != "email"}


@app.get("/api/warm-paths")
def warm_paths_for_company(company: str, limit: int = 50):
    """Warm paths (own-network connections) for a company.

    `matches` = auto-accepted (exact-normalized or anchored-prefix, relevance-
    ranked, capped at `limit` — a large employer can return a very long list);
    `possible` = fuzzy>=90 manual-confirm bucket (never auto-surfaced as a
    match); `total_matches` = uncapped count. Emails from the CSV are never
    exposed here."""
    from .warm_paths import find_warm_paths
    found = find_warm_paths(_db, company)
    return {
        "company": company,
        "matches": [_public_warm_row(m) for m in found["matches"][:max(1, min(limit, 200))]],
        "total_matches": len(found["matches"]),
        "possible": [_public_warm_row(m) for m in found["possible"][:20]],
    }


@app.post("/api/warm-paths/import")
def warm_paths_import():
    """(Re)import every *-connections.csv export from data/linkedin/. Idempotent
    (upsert on profile URL)."""
    from .warm_paths import import_all
    results = import_all(_db)
    if not results:
        raise HTTPException(status_code=404,
                            detail="No *-connections.csv files found in data/linkedin/")
    return {"imports": results}


@app.get("/api/warm-paths/stats")
def warm_paths_stats():
    """Connection counts per source file + total distinct companies."""
    try:
        conn = _db._get_thread_conn()
        by_owner = dict(conn.execute(
            "SELECT owner, COUNT(*) FROM warm_connections GROUP BY owner").fetchall())
        companies = conn.execute(
            "SELECT COUNT(DISTINCT company_norm) FROM warm_connections "
            "WHERE company_norm != ''").fetchone()[0]
        return {"by_owner": by_owner, "distinct_companies": companies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/applications/{app_id}")
def delete_application(app_id: int):
    """Remove an application from the Kanban pipeline."""
    success = _db.delete_application(app_id)
    if not success:
        raise HTTPException(status_code=404, detail="Application not found or could not be deleted")
    return {"status": "deleted", "app_id": app_id}


# === Search Control ===


@app.get("/api/search/status")
def search_status():
    """Get current search status."""
    return {
        "active": _search_state["active"],
        "paused": _search_state["paused"],
        "progress": _search_state["progress"],
        "status_text": _search_state["status_text"],
        "session_id": _search_state["session_id"],
    }


@app.post("/api/search/start")
def search_start(req: SearchStartRequest):
    """Start a search session in background."""
    # Atomically check-and-claim so concurrent starts can't both pass the check.
    # A real SessionManager is created HERE and shared with the pipeline —
    # previously the pipeline made its own private one, so the stop/pause
    # endpoints set flags nobody ever read (complete no-ops).
    from .session import SessionManager
    sm = SessionManager()
    with _search_lock:
        if _search_state["active"]:
            raise HTTPException(status_code=409, detail="Search already in progress")
        _search_state["active"] = True
        _search_state["stop_flag"] = False
        _search_state["paused"] = False
        _search_state["progress"] = 0.0
        _search_state["status_text"] = "Starting search..."
        _search_state["session"] = sm
        _search_state["session_id"] = sm.id

    # Resolve settings: use request overrides, fall back to config defaults
    resume_path = req.resume_path or "resume.pdf"
    base_cv_path = req.base_cv_path or "base_cv.docx"
    model_filter = req.model_filter or CASCADE_STAGE2_MODEL
    model_generation = req.model_generation or DEFAULT_GENERATION_MODEL
    search_sites = req.search_sites or SEARCH_SITES
    send_email = req.send_email
    target_email = req.target_email or None

    # Build cascade config if cascade is enabled. stop_check wires the cascade
    # stages to the shared SessionManager so Stop interrupts AI filtering too.
    cascade_config = None
    if req.cascade:
        cascade_config = {
            "enabled": True,
            "stage1_model": req.stage1_model or CASCADE_STAGE1_MODEL,
            "stage2_model": req.stage2_model or CASCADE_STAGE2_MODEL,
            "stage3_model": req.stage3_model or CASCADE_STAGE3_MODEL,
            "stop_check": sm.check_stop,
        }

    def _run_search():
        try:
            from .main import run_search_session
            # active/stop_flag/progress/status already claimed under _search_lock above.

            def progress_cb(text, pct):
                _search_state["progress"] = pct
                _search_state["status_text"] = text

            run_search_session(
                resume_path=resume_path,
                base_cv_path=base_cv_path,
                model_filter=model_filter,
                model_generation=model_generation,
                limit=req.limit if req.limit > 0 else None,
                site_list=search_sites,
                progress_callback=progress_cb,
                send_email=send_email,
                target_email=target_email,
                cascade_config=cascade_config,
                session=sm,
            )
        except Exception as e:
            logger.error(f"Search session failed: {e}")
            _search_state["status_text"] = f"Error: {e}"
        finally:
            _search_state["active"] = False
            _invalidate_stats_cache()
            _search_state["paused"] = False
            _search_state["progress"] = 1.0
            _search_state["session"] = None

    t = threading.Thread(target=_run_search, daemon=True)
    _search_state["thread"] = t
    t.start()
    return {"status": "started"}


@app.post("/api/search/stop")
def search_stop():
    """Signal the running search to stop (drives the shared SessionManager —
    the old flag was write-only and the button was a complete no-op)."""
    _search_state["stop_flag"] = True
    sm = _search_state.get("session")
    if sm is None:
        return {"status": "no_active_search"}
    sm.request_stop()
    _search_state["status_text"] = "Stopping..."
    return {"status": "stop_requested"}


@app.post("/api/search/pause")
def search_pause():
    """Toggle pause on the running search (drives the shared SessionManager)."""
    sm = _search_state.get("session")
    if sm is None:
        return {"paused": False, "status": "no_active_search"}
    sm.toggle_pause()
    _search_state["paused"] = sm.is_paused
    _search_state["status_text"] = "Paused" if sm.is_paused else "Resuming..."
    return {"paused": sm.is_paused}


@app.get("/api/search/progress")
async def search_progress():
    """SSE stream for real-time pipeline progress."""
    async def event_generator():
        while True:
            status = {
                "active": _search_state["active"],
                "paused": _search_state["paused"],
                "progress": _search_state["progress"],
                "status_text": _search_state["status_text"],
                "session_id": _search_state["session_id"],
            }
            yield f"data: {json.dumps(status)}\n\n"
            if not _search_state["active"] and _search_state["progress"] >= 1.0:
                yield f"data: {json.dumps({**status, 'done': True})}\n\n"
                break
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# === Assistant (chat agent) ===


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream one assistant turn (SSE). Runs the Claude tool-use loop over the
    app's tool layer (chat_tools) with persisted history."""
    from .chat_agent import stream_chat_response

    async def event_generator():
        async for event in stream_chat_response(_db, req.message, session_id=req.session_id):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/chat/sessions")
def chat_sessions():
    """All chat threads (id, preview, last activity, turn count)."""
    return {"sessions": _db.list_chat_sessions()}


@app.get("/api/chat/history")
def chat_history(session_id: str = "candidate", limit: int = Query(default=40, le=200)):
    """Persisted chat turns, oldest-first."""
    return {"messages": _db.get_chat_messages(session_id=session_id, limit=limit)}


@app.delete("/api/chat/history")
def chat_clear(session_id: str = "candidate"):
    """Clear the conversation."""
    ok = _db.clear_chat_messages(session_id=session_id)
    return {"cleared": bool(ok)}


# === Logs ===


@app.get("/api/logs")
def get_logs(lines: int = Query(default=200, le=2000)):
    """Get last N lines from the log file."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
    if not os.path.isdir(log_dir):
        return {"lines": [], "file": None}

    # Find latest log file
    log_files = sorted(
        [f for f in os.listdir(log_dir) if f.endswith(".log")],
        key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
        reverse=True,
    )
    if not log_files:
        return {"lines": [], "file": None}

    log_path = os.path.join(log_dir, log_files[0])
    try:
        with open(log_path, "r") as f:
            all_lines = f.readlines()
        return {"lines": all_lines[-lines:], "file": log_files[0]}
    except Exception as e:
        return {"lines": [f"Error reading log: {e}"], "file": log_files[0]}


# === Stats ===


# Stats cache: the page polls every 60s but the numbers only change after a
# search run or a Kanban/review ACTION — so serve from memory and invalidate
# on those events instead of re-aggregating the whole table per poll.
_stats_cache: dict = {"key": None, "data": None, "ts": 0.0}
_STATS_TTL_SEC = 600  # safety net; event invalidation is the primary mechanism


def _invalidate_stats_cache():
    _stats_cache["key"] = None


@app.get("/api/stats/overview")
def stats_overview(days: int = 30, refresh: bool = False):
    """Dashboard stats for the funnel story: KPIs with week-over-week deltas,
    the 30-day funnel (surfaced → reviewed → applied → in process → offers),
    daily surfaced + applied series, and role-type distribution.

    Cached in memory; invalidated by search-run completion and Kanban/review
    actions (incremental-feeling without incremental bookkeeping)."""
    days = max(7, min(days, 180))
    import time as _time
    if (not refresh and _stats_cache["key"] == days
            and _time.time() - _stats_cache["ts"] < _STATS_TTL_SEC):
        return _stats_cache["data"]
    try:
        conn = _db._get_thread_conn()

        # Daily surfaced (strong/worth split)
        daily_rows = conn.execute(f"""
            SELECT date(created_at) AS day, tier, COUNT(*) AS n
            FROM scrapes
            WHERE tier IN ('strong','worth_a_look')
              AND created_at >= datetime('now', '-{days} day')
            GROUP BY day, tier ORDER BY day
        """).fetchall()
        by_day = {}
        for day, tier, n in daily_rows:
            d = by_day.setdefault(day, {"day": day, "strong": 0, "worth_a_look": 0,
                                        "applied": 0})
            d[tier] = n
        # Daily applied (real application day)
        for day, n in conn.execute(f"""
            SELECT date_applied, COUNT(*) FROM applications
            WHERE status != 'To Apply' AND date_applied IS NOT NULL
              AND date_applied >= date('now', '-{days} day')
            GROUP BY date_applied
        """).fetchall():
            by_day.setdefault(day, {"day": day, "strong": 0, "worth_a_look": 0,
                                    "applied": 0})["applied"] = n

        # FUNNEL — one 30-day cohort, honestly labeled
        surfaced, surfaced_strong, reviewed = conn.execute(f"""
            SELECT COUNT(*),
                   SUM(CASE WHEN tier='strong' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN reviewed=1 THEN 1 ELSE 0 END)
            FROM scrapes
            WHERE tier IN ('strong','worth_a_look')
              AND created_at >= datetime('now', '-{days} day')
        """).fetchone()
        applied = conn.execute(f"""
            SELECT COUNT(*) FROM applications
            WHERE status != 'To Apply' AND date_applied IS NOT NULL
              AND date_applied >= date('now', '-{days} day')
        """).fetchone()[0]
        # Current-state counts (whole pipeline, not window-bound)
        statuses = dict(conn.execute(
            "SELECT status, COUNT(*) FROM applications GROUP BY status").fetchall())
        in_process = sum(statuses.get(s, 0) for s in
                         ("Screening Call", "Interview", "Final Interview"))
        offers = sum(statuses.get(s, 0) for s in ("Offer", "Accepted"))

        # Week-over-week deltas
        def _wow(sql):
            # b must be a valid modifier — "now" as a modifier yields NULL.
            this_w = conn.execute(sql.format(a="-7 day", b="+0 day")).fetchone()[0] or 0
            last_w = conn.execute(sql.format(a="-14 day", b="-7 day")).fetchone()[0] or 0
            return this_w, last_w
        s_this, s_last = _wow("""
            SELECT COUNT(*) FROM scrapes
            WHERE tier IN ('strong','worth_a_look')
              AND created_at >= datetime('now', '{a}')
              AND created_at < datetime('now', '{b}')""")
        st_this, st_last = _wow("""
            SELECT COUNT(*) FROM scrapes
            WHERE tier = 'strong'
              AND created_at >= datetime('now', '{a}')
              AND created_at < datetime('now', '{b}')""")
        a_this, a_last = _wow("""
            SELECT COUNT(*) FROM applications
            WHERE status != 'To Apply' AND date_applied IS NOT NULL
              AND date_applied >= date('now', '{a}')
              AND date_applied < date('now', '{b}')""")

        # Role types (categorized only — 'uncategorized' = ATS finds that skip
        # keyword matching; drowning the chart with it says nothing)
        roletype_rows = conn.execute(f"""
            SELECT keyword_category AS cat, tier, COUNT(*) AS n
            FROM scrapes
            WHERE tier IN ('strong','worth_a_look')
              AND keyword_category IS NOT NULL AND keyword_category != ''
              AND created_at >= datetime('now', '-{days} day')
            GROUP BY cat, tier
        """).fetchall()
        by_cat = {}
        for cat, tier, n in roletype_rows:
            c = by_cat.setdefault(cat, {"category": cat, "strong": 0, "worth_a_look": 0})
            c[tier] = n
        uncategorized = conn.execute(f"""
            SELECT COUNT(*) FROM scrapes
            WHERE tier IN ('strong','worth_a_look')
              AND (keyword_category IS NULL OR keyword_category = '')
              AND created_at >= datetime('now', '-{days} day')
        """).fetchone()[0]

        reached = _db.count_reached_interview()
        data = {
            "days": days,
            "daily": sorted(by_day.values(), key=lambda d: d["day"]),
            "funnel": {
                "surfaced": surfaced or 0,
                "strong": surfaced_strong or 0,
                "reviewed": reviewed or 0,
                "applied": applied or 0,
                "in_process": in_process,
                "offers": offers,
            },
            "week": {
                "surfaced": {"now": s_this, "prev": s_last},
                "strong": {"now": st_this, "prev": st_last},
                "applied": {"now": a_this, "prev": a_last},
            },
            "statuses": statuses,
            # All-time: applications that ever reached Screening Call or
            # later, even if the card is now closed (the
            # current-status counts above lose those). _total = confirmed
            # (sitting there now, or a recorded move); _unconfirmed = known
            # only from a prep-sheet proxy that the candidate has not confirmed yet.
            "reached_interview_total": reached["confirmed"],
            "reached_interview_unconfirmed": reached["unconfirmed"],
            # All-time, current cards: how the closed ones ended
            # (furthest stage of the interview processes + the candidate's "what
            # happened" lines from the outcome form).
            "endings": _db.get_outcome_summary(),
            "role_types": sorted(by_cat.values(),
                                 key=lambda c: -(c["strong"] + c["worth_a_look"])),
            "uncategorized": uncategorized,
        }
        _stats_cache.update(key=days, data=data, ts=_time.time())
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/rejections")
def rejection_stats():
    """Get rejection reason counts."""
    return _db.get_rejection_stats()


@app.get("/api/stats/cascade")
def cascade_stats():
    """Get cascade stats from latest session (placeholder)."""
    # TODO: persist cascade stats to DB for cross-session retrieval
    return {"message": "Cascade stats available during active search via SSE"}


# === Config (read-only) ===


@app.get("/api/config")
def get_config():
    """Return keywords, models, locations, and search config for frontend display."""
    return {
        "keywords": KEYWORDS,
        "keywords_by_category": KEYWORDS_BY_CATEGORY,
        "locations": ALLOWED_LOCATIONS,
        "models": {
            "stage1": CASCADE_STAGE1_MODEL,
            "stage2": CASCADE_STAGE2_MODEL,
            "stage3": CASCADE_STAGE3_MODEL,
        },
        "standard_answers": APPLICATION_STANDARD_ANSWERS,
        # Extended config for Settings panel
        "available_models": AVAILABLE_MODELS,
        "search_sites": SEARCH_SITES,
        "cascade_models": {
            "stage1": CASCADE_STAGE1_MODEL,
            "stage2": CASCADE_STAGE2_MODEL,
            "stage3": CASCADE_STAGE3_MODEL,
        },
        "generation_model": DEFAULT_GENERATION_MODEL,
        "auto_generate_docs": AUTO_GENERATE_CV_CL,
        "default_resume_path": "resume.pdf",
        "default_base_cv_path": "base_cv.docx",
        "prompts": {
            "stage1": CASCADE_STAGE1_PROMPT,
            "stage2_claude": STAGE2_PROMPT_CLAUDE,
            "stage3": CASCADE_STAGE3_PROMPT,
            "cv_summary_claude": PROMPT_SUMMARY_TEMPLATE_CLAUDE,
            "cv_experience_claude": PROMPT_EXPERIENCE_TEMPLATE_CLAUDE,
        },
    }
