"""
Tool layer for the in-app Assistant (chat agent).

Each tool is a thin, SAFE wrapper over existing pipeline code — parameterized
queries only (no raw SQL from the model), file reads restricted to the
generated-docs directories, and JSON-serializable returns. The chat endpoint
(api.py) exposes these to Claude via the Messages API tool-use loop.

Design rule: tools NEVER raise — they return {"error": ...} so the agent can
explain a failure instead of the chat dying.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from .candidate_profile import SEARCH
from .config import METRICS_BANK_TEXT

_DEFAULT_SEARCH_LOCATION = SEARCH["locations"]["primary_metro_label"]

logger = logging.getLogger(__name__)

# Kanban statuses the assistant may set (mirrors the UI columns)
VALID_APP_STATUSES = [
    "To Apply", "Applied", "Screening Call", "Interview",
    "Final Interview", "Offer", "Accepted", "Rejected", "Ghosted", "I Withdrew",
]

_DOC_DIRS = ("cv", "cl")  # read_document is restricted to these trees


def _doc_urls(job_id, cv_path=None, cl_path=None) -> dict:
    """Download links for generated docs (served by /api/jobs/{id}/cv|cl).
    Relative URLs work through the Next.js rewrite, so they work on any host
    name and are clickable directly inside the chat."""
    out = {}
    if cv_path:
        out["cv_url"] = f"/api/jobs/{job_id}/cv"
    if cl_path:
        out["cl_url"] = f"/api/jobs/{job_id}/cl"
    return out


def _row_to_brief(r: dict) -> dict:
    """Compact job representation for list results (keeps tokens down)."""
    return {
        "id": r.get("id"),
        "title": r.get("title"),
        "company": r.get("company"),
        "location": r.get("location"),
        "tier": r.get("tier"),
        "match_level": r.get("match_level"),
        "confidence": r.get("confidence"),
        "date_posted": r.get("date_posted"),
        "reviewed": bool(r.get("reviewed")),
        "link": r.get("link"),
        "has_cv": bool(r.get("cv_path")),
    }


# ── Tool implementations ────────────────────────────────────────────────────

def query_jobs(db, tier: str = None, company: str = None, title_contains: str = None,
               reviewed: bool = None, days_back: int = None, limit: int = 20) -> dict:
    """Search the scrapes table with safe, parameterized filters."""
    try:
        conn = db._get_thread_conn()
        import sqlite3
        conn.row_factory = sqlite3.Row
        conds, params = [], []
        if tier in ("strong", "worth_a_look"):
            conds.append("tier = ?"); params.append(tier)
        elif tier == "reject":
            conds.append("tier IS NULL AND match_level IS NOT NULL")
        if company:
            conds.append("company LIKE ?"); params.append(f"%{company}%")
        if title_contains:
            conds.append("title LIKE ?"); params.append(f"%{title_contains}%")
        if reviewed is not None:
            conds.append("reviewed = ?"); params.append(1 if reviewed else 0)
        if days_back:
            cutoff = (datetime.now() - timedelta(days=int(days_back))).strftime("%Y-%m-%d")
            conds.append("created_at >= ?"); params.append(cutoff)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        limit = max(1, min(int(limit or 20), 50))
        rows = conn.execute(
            f"""SELECT id, title, company, location, link, tier, match_level,
                       confidence, date_posted, reviewed, cv_path
                FROM scrapes {where}
                ORDER BY CASE tier WHEN 'strong' THEN 0 WHEN 'worth_a_look' THEN 1 ELSE 2 END,
                         created_at DESC LIMIT ?""",
            params + [limit],
        ).fetchall()
        conn.row_factory = None
        return {"count": len(rows), "jobs": [_row_to_brief(dict(r)) for r in rows]}
    except Exception as e:
        logger.warning(f"chat_tools.query_jobs failed: {e}")
        return {"error": str(e)}


def get_job(db, job_id: int) -> dict:
    """Full detail for one job, including description + AI reasoning."""
    try:
        data = db.get_job_details(int(job_id))
        if not data:
            return {"error": f"job {job_id} not found"}
        desc = data.get("description")
        if desc is None or isinstance(desc, float):
            desc = ""
        return {
            **_row_to_brief(data),
            "description": str(desc)[:6000],
            "reasoning": data.get("reasoning"),
            "keyword_category": data.get("keyword_category"),
            "cv_path": data.get("cv_path"),
            "cl_path": data.get("cl_path"),
            "cv_quality_flag": data.get("cv_quality_flag"),
            "rejection_reason": data.get("rejection_reason"),
            **_doc_urls(job_id, data.get("cv_path"), data.get("cl_path")),
        }
    except Exception as e:
        return {"error": str(e)}


def check_already_applied(db, company: str, title: str = None) -> dict:
    """Has the candidate already applied/tracked/dismissed anything at this company?

    Checks the applications table (tracked/applied) AND dismissed jobs with a
    rejection_reason — the exact 'did I already deal with this?' question.
    """
    try:
        conn = db._get_thread_conn()
        import sqlite3
        conn.row_factory = sqlite3.Row
        params = [f"%{company}%"]
        title_cond = ""
        if title:
            title_cond = " AND s.title LIKE ?"
            params.append(f"%{title}%")
        apps = conn.execute(
            f"""SELECT a.id AS app_id, a.status, a.date_applied, s.title, s.company, s.id AS job_id
                FROM applications a JOIN scrapes s ON a.job_id = s.id
                WHERE s.company LIKE ?{title_cond}
                ORDER BY a.updated_at DESC LIMIT 20""", params).fetchall()
        dismissed = conn.execute(
            f"""SELECT id AS job_id, title, company, rejection_reason
                FROM scrapes
                WHERE company LIKE ?{title_cond.replace('s.', '')} AND rejection_reason IS NOT NULL
                ORDER BY created_at DESC LIMIT 10""", params).fetchall()
        conn.row_factory = None
        return {
            "applications": [dict(r) for r in apps],
            "dismissed": [dict(r) for r in dismissed],
            "already_engaged": bool(apps) or bool(dismissed),
        }
    except Exception as e:
        return {"error": str(e)}


def get_applications(db, status: str = None) -> dict:
    """Kanban board state, optionally filtered by status column.
    Rows are slimmed to the fields the agent needs — full rows tripled the
    tokens fed back per call."""
    try:
        rows = db.get_applications()
        if status:
            rows = [r for r in rows if r.get("status") == status]
        slim = [{
            "app_id": r.get("app_id"), "job_id": r.get("job_id"),
            "status": r.get("status"), "date_applied": r.get("date_applied"),
            "title": r.get("title"), "company": r.get("company"),
            "match_level": r.get("match_level"),
            "cv_ready": bool(r.get("cv_path")),
            "notes": (str(r.get("notes"))[:200] if r.get("notes") else None),
        } for r in rows[:60]]
        return {"count": len(rows), "applications": slim}
    except Exception as e:
        return {"error": str(e)}


def _service_api(db):
    """The running API module, when this tool call is executing inside the API
    service on the service's own database — else None. Card side effects
    (outreach playbook, interview prep, stats cache) belong to that process;
    a tool pointed at any other database (tests, a one-off script) must never
    start a paid generation, so it simply skips them."""
    import sys
    api = sys.modules.get(f"{__package__}.api")
    if api is not None and getattr(api, "_db", None) is db:
        return api
    return None


def update_application_status(db, app_id: int, status: str, notes: str = None) -> dict:
    """Move a Kanban card / append notes. Status must be a valid column.

    Same path as the Kanban: the move is written to the outcome ledger
    (source 'chat') and then runs api.on_application_status_change — a chat
    move used to skip the outreach playbook and the interview prep."""
    if status and status not in VALID_APP_STATUSES:
        return {"error": f"invalid status {status!r}; valid: {VALID_APP_STATUSES}"}
    try:
        before = db.get_application(int(app_id))
        old_status = before.get("status") if before else None
        ok = db.update_application(int(app_id), status, notes, source="chat")
        out = {"updated": bool(ok), "app_id": int(app_id), "status": status}
        api = _service_api(db)
        if ok and before and api is not None:
            out.update(api.on_application_status_change(
                int(app_id), old_status, status, source="chat"))
        return out
    except Exception as e:
        return {"error": str(e)}


def track_job(db, job_id: int) -> dict:
    """Add a job to the Kanban (To Apply) and mark it reviewed."""
    try:
        ok = db.track_job(int(job_id), source="chat")
        db.mark_reviewed([int(job_id)])
        return {"tracked": bool(ok), "job_id": int(job_id)}
    except Exception as e:
        return {"error": str(e)}


def dismiss_job(db, job_id: int, reason: str = "other") -> dict:
    """Dismiss a job with a reason and mark it reviewed. Runs the same
    location feedback loop as the HTTP dismiss endpoint (a location-driven
    reason teaches the pre-filter that location is a no-go)."""
    try:
        suppressed = db.dismiss_job(int(job_id), reason)
        api = _service_api(db)
        if api is not None:
            api._invalidate_stats_cache()
        return {"dismissed": True, "job_id": int(job_id), "reason": reason,
                "suppressed_location": suppressed}
    except Exception as e:
        return {"error": str(e)}


def generate_cv(db, job_id: int) -> dict:
    """Generate tailored CV + cover letter for a job (slow: ~1-2 min, Opus)."""
    try:
        job = db.get_job_details(int(job_id))
        if not job:
            return {"error": f"job {job_id} not found"}
        if job.get("cv_path") and os.path.exists(str(job.get("cv_path"))):
            return {"status": "already_exists", "cv_path": job["cv_path"], "cl_path": job.get("cl_path"),
                    **_doc_urls(job_id, job.get("cv_path"), job.get("cl_path"))}
        from .docs_builder import generate_package
        from .gpt_filter import load_profile_text
        profile_text = load_profile_text("Profile.pdf")
        # The shared path saves the version, the paths and the QA verdict.
        out = generate_package(job, db, profile_text, source="chat")
        if out["status"] == "in_progress":
            return {"status": "in_progress", "job_id": int(job_id)}
        cv_p, cl_p = out["cv_path"], out["cl_path"]
        return {
            "status": "generated" if cv_p else "failed",
            "cv_path": cv_p, "cl_path": cl_p,
            "quality_flag": out["flag"],
            **({"error": out["error"]} if out.get("error") else {}),
            **_doc_urls(job_id, cv_p, cl_p),
        }
    except Exception as e:
        logger.warning(f"chat_tools.generate_cv failed for {job_id}: {e}")
        return {"error": str(e)}


def read_document(db, path: str) -> dict:
    """Read the text of a generated CV/CL (.docx). Restricted to cv/ and cl/."""
    try:
        norm = os.path.normpath(str(path))
        if os.path.isabs(norm) or norm.startswith(".."):
            return {"error": "only relative paths under cv/ or cl/ are allowed"}
        if norm.split(os.sep)[0] not in _DOC_DIRS:
            return {"error": f"path must be under one of {_DOC_DIRS}"}
        if not os.path.exists(norm):
            return {"error": f"file not found: {norm}"}
        from docx import Document
        doc = Document(norm)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return {"path": norm, "text": text[:12000]}
    except Exception as e:
        return {"error": str(e)}


def get_pipeline_stats(db) -> dict:
    """What's happening: tier counts (today/7d), review backlog, kanban, rejections."""
    try:
        conn = db._get_thread_conn()
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        def counts(since):
            rows = conn.execute(
                """SELECT COALESCE(tier, 'reject'), COUNT(*) FROM scrapes
                   WHERE created_at >= ? AND match_level IS NOT NULL GROUP BY tier""",
                (since,)).fetchall()
            return {k: n for k, n in rows}

        backlog = conn.execute(
            """SELECT COUNT(*) FROM scrapes
               WHERE tier IN ('strong','worth_a_look') AND (reviewed IS NULL OR reviewed = 0)"""
        ).fetchone()[0]
        kanban = dict(conn.execute(
            "SELECT status, COUNT(*) FROM applications GROUP BY status").fetchall())
        stale_applied = conn.execute(
            """SELECT COUNT(*) FROM applications
               WHERE status = 'Applied' AND date_applied <= ?""",
            ((datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"),)).fetchone()[0]
        return {
            "today": counts(today),
            "last_7_days": counts(week_ago),
            "unreviewed_backlog": backlog,
            "kanban": kanban,
            "applied_with_no_followup_14d": stale_applied,
            "rejection_stats": db.get_rejection_stats(),
        }
    except Exception as e:
        return {"error": str(e)}


def get_company_intel(db, company: str) -> dict:
    """Cached Perplexity intel for a company (profile/salary/hiring if cached)."""
    try:
        out = {"company": company}
        base = db.get_company_intel(company)
        if base and base.get("company_profile"):
            out["profile"] = base["company_profile"]
        for key, label in ((f"__salary_intel__{company}", "salary"),
                           (f"__hiring_intel__{company}", "hiring")):
            row = db.get_company_intel(key)
            if row and row.get("company_profile"):
                out[label] = row["company_profile"]
        # salary intel keys are per-title; do a LIKE sweep as fallback
        if "salary" not in out:
            conn = db._get_thread_conn()
            r = conn.execute(
                "SELECT company_profile FROM company_intel WHERE company_name LIKE ? LIMIT 1",
                (f"__salary_intel__{company}%",)).fetchone()
            if r and r[0]:
                out["salary"] = r[0]
        out["cached"] = len(out) > 2
        return out
    except Exception as e:
        return {"error": str(e)}


def list_stale_applications(db, days: int = 14) -> dict:
    """Applications sitting in 'Applied' with no movement for N+ days — the
    follow-up queue. Sorted oldest-first (most overdue on top)."""
    try:
        conn = db._get_thread_conn()
        import sqlite3
        conn.row_factory = sqlite3.Row
        cutoff = (datetime.now() - timedelta(days=int(days or 14))).strftime("%Y-%m-%d")
        rows = conn.execute(
            """SELECT a.id AS app_id, a.status, a.date_applied, a.notes,
                      s.id AS job_id, s.title, s.company, s.link
               FROM applications a JOIN scrapes s ON a.job_id = s.id
               WHERE a.status = 'Applied' AND a.date_applied <= ?
               ORDER BY a.date_applied ASC LIMIT 50""", (cutoff,)).fetchall()
        conn.row_factory = None
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["days_since_applied"] = (datetime.now() - datetime.strptime(d["date_applied"], "%Y-%m-%d")).days
            except Exception:
                d["days_since_applied"] = None
            out.append(d)
        return {"count": len(out), "stale_applications": out,
                "note": f"Applied >= {days} days ago with no status change"}
    except Exception as e:
        return {"error": str(e)}


def _drafting_context(db, title: str, company: str) -> str:
    """Shared candidate + company context for the drafting tools."""
    intel = ""
    try:
        row = db.get_company_intel(company)
        if row and row.get("company_profile"):
            intel = f"\nCompany context: {str(row['company_profile'])[:800]}"
    except Exception:
        pass
    return (
        f"Candidate — verified proof points (the only career facts to use):\n{METRICS_BANK_TEXT}\n"
        f"Target: {title} at {company}.{intel}"
    )


def draft_followup(db, app_id: int) -> dict:
    """Draft a short follow-up for an application in 'Applied' (email + LinkedIn
    variants). Uses the tracked job's details; one premium-model call."""
    try:
        conn = db._get_thread_conn()
        import sqlite3
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT a.id AS app_id, a.date_applied, a.notes, s.title, s.company, s.link
               FROM applications a JOIN scrapes s ON a.job_id = s.id WHERE a.id = ?""",
            (int(app_id),)).fetchone()
        conn.row_factory = None
        if not row:
            return {"error": f"application {app_id} not found"}
        d = dict(row)
        from .gpt_filter import call_llm
        from .config import DEFAULT_GENERATION_MODEL
        prompt = f"""Write TWO short follow-up messages for a job application with no response.

{_drafting_context(db, d['title'], d['company'])}
Applied on: {d.get('date_applied')}
Application notes: {d.get('notes') or '(none)'}

1) EMAIL (subject + <=120 words): polite, confident, restates fit in ONE concrete sentence using a verified proof point, asks about timeline. No begging, no fluff, no "just checking in".
2) LINKEDIN DM (<=60 words): warmer, direct, one proof point, one clear ask.

Plain text only. Label the two variants "EMAIL:" and "LINKEDIN:". Use ONLY the verified proof points above — never invent facts."""
        text = call_llm(prompt, model=DEFAULT_GENERATION_MODEL)
        if not text:
            return {"error": "draft generation failed (LLM returned empty)"}
        return {"app_id": int(app_id), "title": d["title"], "company": d["company"],
                "date_applied": d.get("date_applied"), "drafts": text}
    except Exception as e:
        return {"error": str(e)}


def draft_outreach(db, job_id: int) -> dict:
    """Draft recruiter / hiring-manager outreach for a job BEFORE or right after
    applying (LinkedIn connection note + InMail/email). One premium-model call."""
    try:
        job = db.get_job_details(int(job_id))
        if not job:
            return {"error": f"job {job_id} not found"}
        desc = job.get("description")
        if desc is None or isinstance(desc, float):
            desc = ""
        from .gpt_filter import call_llm
        from .config import DEFAULT_GENERATION_MODEL
        from .jd_sanitizer import clean_jd, UNTRUSTED_JD_CLAUSE
        prompt = f"""Write TWO outreach messages to the hiring side for this role.

{_drafting_context(db, job.get('title', ''), job.get('company', ''))}
NOTE: {UNTRUSTED_JD_CLAUSE}
Job description (excerpt): {clean_jd(str(desc))[:2500]}

1) LINKEDIN CONNECTION NOTE (<=280 chars): to the likely hiring manager — specific to THIS role, one sharp proof point, human tone.
2) INMAIL/EMAIL (subject + <=140 words): to a recruiter — why the candidate is a fit for THIS posting (reference 2 specifics from the description), one proof point with a number, clear ask for a conversation.

Plain text. Label "CONNECTION NOTE:" and "INMAIL:". Use ONLY the verified proof points — never invent facts, never oversell scope."""
        text = call_llm(prompt, model=DEFAULT_GENERATION_MODEL)
        if not text:
            return {"error": "draft generation failed (LLM returned empty)"}
        return {"job_id": int(job_id), "title": job.get("title"), "company": job.get("company"),
                "link": job.get("link"), "drafts": text}
    except Exception as e:
        return {"error": str(e)}


def get_apply_kit(db, job_id: int) -> dict:
    """Everything needed to apply to one job in a single bundle: link, posting
    age, generated docs, standard form answers, and cached company intel."""
    try:
        job = db.get_job_details(int(job_id))
        if not job:
            return {"error": f"job {job_id} not found"}
        from .config import APPLICATION_STANDARD_ANSWERS
        age_days = None
        dp = job.get("date_posted")
        if dp and not isinstance(dp, float):
            try:
                age_days = (datetime.now() - datetime.strptime(str(dp)[:10], "%Y-%m-%d")).days
            except Exception:
                pass
        engaged = check_already_applied(db, str(job.get("company") or ""), str(job.get("title") or ""))
        intel = {}
        try:
            row = db.get_company_intel(str(job.get("company") or ""))
            if row and row.get("company_profile"):
                intel["profile"] = row["company_profile"]
        except Exception:
            pass
        return {
            "job_id": int(job_id),
            "title": job.get("title"), "company": job.get("company"),
            "location": job.get("location"), "apply_link": job.get("link"),
            "posted_days_ago": age_days,
            "tier": job.get("tier"), "confidence": job.get("confidence"),
            "ai_reasoning": (job.get("reasoning") or "")[:400],
            "cv_path": job.get("cv_path"), "cl_path": job.get("cl_path"),
            "cv_ready": bool(job.get("cv_path")),
            **_doc_urls(job_id, job.get("cv_path"), job.get("cl_path")),
            "standard_answers": APPLICATION_STANDARD_ANSWERS,
            "already_engaged": engaged.get("already_engaged"),
            "company_intel": intel,
        }
    except Exception as e:
        return {"error": str(e)}



# ── LIVE SEARCH TOOLS — the agent participates in the search ────────────────
# These hit external sources ON DEMAND (vs the DB-bound tools above). Fresh
# results can optionally be saved into the pipeline (save=True) so they appear
# in the app and flow through review/tracking like any scraped job.


def _brief_live(j: dict) -> dict:
    return {"title": j.get("title"), "company": j.get("company"),
            "location": j.get("location"), "url": j.get("job_url"),
            "posted": j.get("date_posted"), "source": j.get("site"),
            "description_snippet": str(j.get("description") or "")[:300]}


def _maybe_save(db, jobs: list, save: bool) -> int:
    if not save:
        return 0
    saved = 0
    for j in jobs:
        try:
            if db.save_job(j, session_id=None):
                saved += 1
        except Exception:
            pass
    return saved


def live_search_jobs(db, query: str, location: str = _DEFAULT_SEARCH_LOCATION,
                     sites: list = None, results: int = 10,
                     hours_old: int = 168, save: bool = False) -> dict:
    """On-demand board search via jobspy (LinkedIn/Indeed/Google guest
    endpoints — no login). ~10-30s. Free and unlimited."""
    try:
        from jobspy import scrape_jobs
        site_name = sites or ["indeed", "linkedin", "google"]
        df = scrape_jobs(
            site_name=site_name, search_term=query,
            google_search_term=f"{query} jobs in {location}",
            location=location, results_wanted=int(results),
            hours_old=int(hours_old), country_indeed="USA",
        )
        if df is None or df.empty:
            return {"query": query, "found": 0, "jobs": []}
        jobs = []
        for _, r in df.iterrows():
            jobs.append({
                "title": str(r.get("title") or ""), "company": str(r.get("company") or ""),
                "location": str(r.get("location") or ""),
                "job_url": str(r.get("job_url") or ""),
                "description": str(r.get("description") or "")[:6000],
                "date_posted": str(r.get("date_posted") or ""),
                "site": str(r.get("site") or "live_search"),
            })
        saved = _maybe_save(db, jobs, save)
        return {"query": query, "found": len(jobs),
                "saved_to_pipeline": saved,
                "jobs": [_brief_live(j) for j in jobs[:15]]}
    except Exception as e:
        return {"error": f"live search failed: {e}"}


def search_google_jobs_live(db, query: str, location: str = "",
                            remote: bool = False, save: bool = False) -> dict:
    """Google Jobs via SerpAPI — reaches boards we can't scrape (iCIMS,
    SuccessFactors, company career sites). COSTS 1 credit of a 250/month
    budget — use for targeted questions, not broad sweeps."""
    from .scraper_google_jobs import search_google_jobs, _budget_state, _spend, MONTHLY_HARD_STOP
    month, used, idx = _budget_state(db)
    if used >= MONTHLY_HARD_STOP:
        return {"error": f"SerpAPI monthly budget exhausted ({used}/{MONTHLY_HARD_STOP}) — use live_search_jobs instead"}
    jobs = search_google_jobs(query, location=location or SEARCH["locations"]["home"],
                              remote=remote)
    _spend(db, month, 1, idx)
    saved = _maybe_save(db, jobs, save)
    return {"query": query, "found": len(jobs), "budget_used": f"{used + 1}/{MONTHLY_HARD_STOP}",
            "saved_to_pipeline": saved,
            "jobs": [_brief_live(j) for j in jobs[:15]]}


def search_company_board(db, company: str) -> dict:
    """LIVE query of one company's ATS board right now (Greenhouse/Lever/
    Ashby/SmartRecruiters/Recruitee/Workable — the configured companies).
    Fresh from the source, includes roles our last run may have missed."""
    from .config import ATS_COMPANIES, KEYWORDS
    target = (company or "").strip().lower()
    entry = next((e for e in ATS_COMPANIES
                  if e["company"].lower() == target
                  or target in e["company"].lower()), None)
    if not entry:
        near = [e["company"] for e in ATS_COMPANIES if target[:4] in e["company"].lower()][:5]
        return {"error": f"{company!r} is not in the configured ATS boards",
                "did_you_mean": near,
                "hint": "use live_search_jobs or search_google_jobs_live instead"}
    ats, token, name = entry["ats"], entry["token"], entry["company"]
    try:
        if ats == "greenhouse":
            from .ats_scraper import scrape_greenhouse as fn
        elif ats == "lever":
            from .ats_scraper import scrape_lever as fn
        elif ats == "ashby":
            from .scraper_ashby_sr import scrape_ashby as fn
        elif ats == "smartrecruiters":
            from .scraper_ashby_sr import scrape_smartrecruiters as fn
        elif ats == "recruitee":
            from .scraper_rwt import scrape_recruitee as fn
        elif ats == "workable":
            from .scraper_rwt import scrape_workable as fn
        elif ats == "eightfold":
            from .scraper_eightfold_apple import scrape_eightfold as fn
        elif ats == "apple":
            from .scraper_eightfold_apple import scrape_apple as fn
        elif ats == "bamboohr":
            from .scraper_bamboohr_comeet import scrape_bamboohr as fn
        else:
            return {"error": f"{name} uses {ats} — live per-company query not supported "
                             f"for this platform (covered by the scheduled runs)"}
        jobs = fn(token, name, KEYWORDS)
        return {"company": name, "ats": ats, "matching_open_roles": len(jobs),
                "jobs": [_brief_live(j) for j in jobs[:20]]}
    except Exception as e:
        return {"error": f"live board query failed for {name}: {e}"}


def get_warm_paths(db, company: str) -> dict:
    """People from the imported networks at a company (official LinkedIn
    exports). A referral ask beats cold outreach; a contact from someone else's
    imported network → ask that person for an intro."""
    from .warm_paths import find_warm_paths
    found = find_warm_paths(db, company)
    return {
        "company": company,
        "warm_paths": [
            {"name": m["name"], "position": m.get("position"),
             "network": m.get("owner"), "linkedin": m.get("url"),
             "action": ("ask the owner of that network for an intro"
                        if (m.get("owner") or "candidate").lower() != "candidate"
                        else "reach out directly")}
            for m in found["matches"][:8]
        ],
        "total": len(found["matches"]),
        "needs_confirmation": len(found["possible"]),
    }


def find_role_contacts(db, company: str, title: str) -> dict:
    """Hiring-team discovery for a role: likely hiring manager + recruiters
    from PUBLIC sources with citations (never invented), plus verified email
    resolution when available. Cached 14 days per company+team; ~$0.03."""
    from .outreach_intel import discover_contacts
    from .contact_resolution import resolve_contacts_for_job, resolved_contacts_text
    blocks = discover_contacts(company, title, db=db)
    if not blocks:
        return {"error": "contact research unavailable (Perplexity offline?)"}
    resolved = resolve_contacts_for_job(db, 0, company, blocks)
    return {
        "company": company, "role": title,
        "research": [{"finding": b["content"][:800],
                      "sources": [s.get("url") for s in (b.get("search_results") or [])[:4]]}
                     for b in blocks],
        "named_contacts": resolved_contacts_text(resolved) or "(none extracted)",
        "note": "names come from cited public sources only; only VERIFIED emails are sendable",
    }


def interview_prep(db, company: str, title: str, job_id: int = None) -> dict:
    """Build an interview-prep sheet for a role: likely questions, the candidate's strongest
    real STAR stories mapped to them (grounded in the verified transcript, never
    invented), a company angle from cached intel, and smart questions to ask.
    Pass job_id when the role is in the pipeline to use its full description;
    otherwise preps from title + company. One premium-model call (~cents)."""
    try:
        from .interview_prep import build_prep, _gather_company_intel, _transcript_for
        description, location, keyword_category = "", "", ""
        if job_id:
            job = db.get_job_details(int(job_id))
            if job:
                company = company or job.get("company", "")
                title = title or job.get("title", "")
                desc = job.get("description")
                description = "" if (desc is None or isinstance(desc, float)) else str(desc)
                location = job.get("location") or ""
                keyword_category = job.get("keyword_category") or ""
        if not company or not title:
            return {"error": "need at least a company and a role title to prep"}
        company_intel = _gather_company_intel(company, title, db=db)
        transcript = _transcript_for(title, keyword_category)
        sheet = build_prep(company, title, description, location, company_intel, transcript)
        if not sheet or "## " not in sheet:
            return {"error": "prep generation failed (LLM returned nothing usable)"}
        return {"company": company, "role": title, "status": "ready",
                "prep": sheet,
                "note": "stories and numbers are from the candidate's verified transcript only; company facts are cite-or-omit"}
    except Exception as e:
        return {"error": str(e)}


# ── Tool registry (name -> schema + fn), consumed by the chat endpoint ──────

TOOLS: list[dict] = [
    {
        "name": "live_search_jobs",
        "description": "LIVE on-demand job search across LinkedIn/Indeed/Google (guest endpoints, free, ~10-30s). Use when the candidate asks to find fresh roles right now, beyond what's in the database. Optionally save results into the pipeline.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search term, e.g. 'video producer'"},
                "location": {"type": "string", "description": "defaults to the profile's primary metro"},
                "sites": {"type": "array", "items": {"type": "string", "enum": ["indeed", "linkedin", "google", "zip_recruiter", "glassdoor"]}},
                "results": {"type": "integer", "description": "per-site cap, default 10"},
                "hours_old": {"type": "integer", "description": "recency window in hours, default 168 (1 week)"},
                "save": {"type": "boolean", "description": "save findings into the pipeline DB so they appear in the app"},
            },
            "required": ["query"],
        },
        "fn": live_search_jobs,
    },
    {
        "name": "search_google_jobs_live",
        "description": "Google Jobs search via SerpAPI — reaches career sites and ATS platforms we can't scrape (iCIMS, SuccessFactors, Avature, company sites). COSTS 1 credit of a 250/month budget: use for targeted questions ('any <role> roles at <company>?'), never broad sweeps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "location": {"type": "string"},
                "remote": {"type": "boolean", "description": "restrict to remote roles"},
                "save": {"type": "boolean"},
            },
            "required": ["query"],
        },
        "fn": search_google_jobs_live,
    },
    {
        "name": "search_company_board",
        "description": "LIVE query of one company's ATS job board right now (the configured companies on Greenhouse/Lever/Ashby/SmartRecruiters/Recruitee/Workable). Use when the candidate asks 'what's open at <company> today'.",
        "input_schema": {
            "type": "object",
            "properties": {"company": {"type": "string"}},
            "required": ["company"],
        },
        "fn": search_company_board,
    },
    {
        "name": "get_warm_paths",
        "description": "People from the imported LinkedIn networks who work at a company — a referral ask beats cold outreach. Returns names, positions, and whose network they're from (a contact from someone else's network → ask that person for an intro).",
        "input_schema": {
            "type": "object",
            "properties": {"company": {"type": "string"}},
            "required": ["company"],
        },
        "fn": get_warm_paths,
    },
    {
        "name": "find_role_contacts",
        "description": "Research the hiring team for a specific role: likely hiring manager and recruiters from cited public sources (never invented), with verified-email resolution when available. Cached 14 days per company+team.",
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "title": {"type": "string", "description": "the job title, e.g. 'Video Producer'"},
            },
            "required": ["company", "title"],
        },
        "fn": find_role_contacts,
    },
    {
        "name": "query_jobs",
        "description": "Search scraped jobs. Filter by tier ('strong' | 'worth_a_look' | 'reject'), company, title substring, reviewed flag, or recency (days_back). Returns compact job briefs sorted strong-first, newest-first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["strong", "worth_a_look", "reject"]},
                "company": {"type": "string"},
                "title_contains": {"type": "string"},
                "reviewed": {"type": "boolean"},
                "days_back": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
        "fn": query_jobs,
    },
    {
        "name": "get_job",
        "description": "Full detail for one job by id: description, AI reasoning, CV status, links.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
        "fn": get_job,
    },
    {
        "name": "check_already_applied",
        "description": "Check whether the candidate already applied to / tracked / dismissed roles at a company (optionally narrowed by title). Use BEFORE recommending an application.",
        "input_schema": {
            "type": "object",
            "properties": {"company": {"type": "string"}, "title": {"type": "string"}},
            "required": ["company"],
        },
        "fn": check_already_applied,
    },
    {
        "name": "get_applications",
        "description": "Kanban board state — all tracked applications, optionally filtered by status (To Apply, Applied, Screening Call, Interview, Final Interview, Offer, Accepted, Rejected, Ghosted, I Withdrew).",
        "input_schema": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
        },
        "fn": get_applications,
    },
    {
        "name": "update_application_status",
        "description": "Move a Kanban card to a new status and/or append notes. app_id comes from get_applications.",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_id": {"type": "integer"},
                "status": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["app_id", "status"],
        },
        "fn": update_application_status,
    },
    {
        "name": "track_job",
        "description": "Add a job to the Kanban 'To Apply' column and mark it reviewed.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
        "fn": track_job,
    },
    {
        "name": "dismiss_job",
        "description": "Dismiss a job with a reason code (e.g. wrong_location, wrong_industry, already_applied, other) and mark it reviewed.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}, "reason": {"type": "string"}},
            "required": ["job_id"],
        },
        "fn": dismiss_job,
    },
    {
        "name": "generate_cv",
        "description": "Generate the tailored CV + cover letter for a job (SLOW: ~1-2 minutes; uses the premium model). If documents already exist, returns their paths instead of regenerating.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
        "fn": generate_cv,
    },
    {
        "name": "read_document",
        "description": "Read the text of a generated CV or cover letter (.docx). Path must be relative, under cv/ or cl/ (as returned by get_job / generate_cv).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "fn": read_document,
    },
    {
        "name": "get_pipeline_stats",
        "description": "Pipeline overview: tier counts today/this week, unreviewed backlog, Kanban column counts, stale 'Applied' items (14d+ without follow-up), rejection stats.",
        "input_schema": {"type": "object", "properties": {}},
        "fn": get_pipeline_stats,
    },
    {
        "name": "list_stale_applications",
        "description": "The follow-up queue: applications sitting in 'Applied' with no movement for N+ days (default 14), most-overdue first. Use this to drive a follow-up session.",
        "input_schema": {
            "type": "object",
            "properties": {"days": {"type": "integer"}},
        },
        "fn": list_stale_applications,
    },
    {
        "name": "draft_followup",
        "description": "Draft a follow-up for a stale application (EMAIL + LINKEDIN variants, grounded in the candidate's verified metrics). app_id from get_applications / list_stale_applications. One premium-model call (~cents).",
        "input_schema": {
            "type": "object",
            "properties": {"app_id": {"type": "integer"}},
            "required": ["app_id"],
        },
        "fn": draft_followup,
    },
    {
        "name": "draft_outreach",
        "description": "Draft recruiter/hiring-manager outreach for a job (LinkedIn connection note + InMail), referencing the actual job description. job_id from query_jobs. One premium-model call (~cents).",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
        "fn": draft_outreach,
    },
    {
        "name": "get_apply_kit",
        "description": "One-stop apply bundle for a job: apply link, posting age, CV/CL paths + readiness, standard form answers (name, work authorization, etc.), already-applied check, cached company intel. Use when the candidate is about to apply.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
        "fn": get_apply_kit,
    },
    {
        "name": "get_company_intel",
        "description": "Cached company intelligence (Perplexity profile / salary / hiring intel) for a company, if previously fetched.",
        "input_schema": {
            "type": "object",
            "properties": {"company": {"type": "string"}},
            "required": ["company"],
        },
        "fn": get_company_intel,
    },
    {
        "name": "interview_prep",
        "description": "Build an interview-prep sheet for a specific role: likely questions inferred from the posting, the candidate's strongest REAL STAR stories mapped to them (grounded in the candidate's verified transcript — never invented), a company angle from cached intel, and sharp questions for the candidate to ask back. Use when the candidate has a Screening Call/Interview/Final Interview on the Kanban or asks to prep for a company. Pass job_id if the role is in the pipeline (uses the full JD); otherwise company + title is enough.",
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "title": {"type": "string", "description": "the job title, e.g. 'Video Producer'"},
                "job_id": {"type": "integer", "description": "optional — the pipeline job id, to use its full description"},
            },
            "required": ["company", "title"],
        },
        "fn": interview_prep,
    },
]

TOOLS_BY_NAME: dict[str, dict] = {t["name"]: t for t in TOOLS}


def execute_tool(db, name: str, tool_input: dict) -> dict:
    """Dispatch one tool call. Never raises."""
    tool = TOOLS_BY_NAME.get(name)
    if not tool:
        return {"error": f"unknown tool {name!r}"}
    try:
        return tool["fn"](db, **(tool_input or {}))
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        logger.warning(f"chat tool {name} failed: {e}")
        return {"error": str(e)}


def anthropic_tool_specs() -> list[dict]:
    """The tools in Anthropic Messages-API format (name/description/input_schema)."""
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        for t in TOOLS
    ]
