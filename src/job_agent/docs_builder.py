
import os
import re
import json
import shutil
import hashlib
import logging
import threading
from copy import deepcopy
from datetime import date, datetime
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from dotenv import load_dotenv

from .gpt_filter import call_llm
from .cv_tailor import (_strip_markdown, _load_transcript_context, grade_cv_quality,
                        _ATTRIBUTION_LOCK, _ANTI_SLOP_RULES, estimate_docx_pages,
                        check_keyword_coverage, _norm_ws, grade_cover_letter_truth)
from . import cv_facts
from .candidate_profile import letter_template as profile_letter_template
from .company_deriver import display_company
from .jd_sanitizer import sanitize_jd, clean_jd, leaked_trap_tokens, UNTRUSTED_JD_CLAUSE
from .config import (DEFAULT_GENERATION_MODEL, TRUNCATION_PROFILE_SUMMARY,
                     DOMAIN_CV_MAP, DOMAIN_DESCRIPTIONS, KEYWORD_TO_CATEGORY,
                     resolve_cv_domain, detect_model_family, METRICS_BANK_TEXT,
                     MODEL_EFFORT, CV_KEYWORD_COVERAGE_ENABLED, CV_LENGTH_GATE_ENABLED,
                     CV_PAGE_CEILING, CV_MAX_PAGES, CV_HM_GRADER_ENABLED,
                     CV_HM_GRADE_THRESHOLD, CV_CONTENT_MODES, CV_CONTENT_MODE,
                     CV_COHORT_TAG, CV_THIN_POSTING_CHARS, CV_B1_SHADOW_JUDGE,
                     CV_B1_SHADOW_SOURCES, TRUNCATION_JOB_DESC_CV)

# Load environment
load_dotenv()

def _slug(s, max_len=60):
    # Coerce anything (NaN float from pandas, None, int, etc.) to a clean string.
    if s is None:
        s = ""
    else:
        s = str(s)
    if s.lower() in ("nan", "none"):
        s = ""
    s = re.sub(r"[^\w\s\-]+", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:max_len] or "role"

def tailor_text_segment(segment_type: str, original_text: str, job: dict, model: str = DEFAULT_GENERATION_MODEL) -> str:
    """
    Tailor a specific segment (Summary or Bullets) for the target job.
    """
    # The posting is untrusted data (Phase 27): cleaned, and flagged as data.
    desc_text = clean_jd(str(job.get('description') or "")) if job.get('description') is not None else ""
    prompt = f"""
    You are an expert Resume Strategist. Your goal is to rewrite the following {segment_type} section to act as a precision argument for why this candidate fits the Target Job.

    === TARGET JOB ===
    Title: {job.get('title')}
    Company: {job.get('company')}
    NOTE: {UNTRUSTED_JD_CLAUSE}
    Key Requirements (JD): {desc_text}

    === SOURCE MATERIAL ({segment_type.upper()}) ===
    {original_text}

    === REASONING STEPS (Perform these internally) ===
    1. DECONSTRUCT: Identify the top 5 hard skills and problems the Target Job needs solved.
    2. FILTER: Scan the Source Material. strictly isolate *only* the accomplishments that prove the candidate can solve those specific problems.
       - IGNORE generic duties (e.g., "attended meetings") or irrelevant experience not tied to the Target Job.
    3. TRANSLATE: Rewrite the filtered points using the specific terminology and keywords found in the Job Description.

    === OUTPUT RULES ===
    - Output ONLY the rewritten text. No markdown formatting (no bold **, no italic *, no headers #). Plain text only.
    - Use ONLY information explicitly present in the Source Material above. Do NOT invent, fabricate, or assume any skills, achievements, metrics, tools, or qualifications not found in the source. If the source material has no relevant info for a specific requirement, omit it rather than guessing.
    - Ensure the tone is active, results-oriented, and professional.
    """
    result = call_llm(prompt, model=model)
    return _strip_markdown(result) if result else original_text

def _title_domain(title) -> str:
    """The domain whose keyword is the longest match inside the job TITLE, or
    "" when none matches (resolve_cv_domain's title step without its
    producer_core default)."""
    title_lower = str(title or "").lower()
    best, best_len = "", 0
    for kw, cat in KEYWORD_TO_CATEGORY.items():
        if kw in title_lower and len(kw) > best_len:
            best, best_len = cat, len(kw)
    return best


def _resolve_base_cv_path(base_cv_path: str, job: dict, title_first: bool = False) -> tuple[str, str]:
    """Resolve the domain-specific base CV file for a job.

    Returns (resolved_path, domain_key).  Falls back to *base_cv_path*
    (the generic base_cv.docx) if the domain file doesn't exist.

    *title_first* is the B2 routing (Phase 27): the job title decides first,
    then the search-keyword category. B0 keeps the routing below exactly as it
    was (keyword category first), so B0 output does not change.
    """
    title = job.get("title", "")

    # keyword_category is already a domain key (e.g. "youtube_and_creators")
    # matched_keyword is an actual keyword string (e.g. "YouTube Producer")
    kw_category = job.get("keyword_category") or ""
    matched_kw = job.get("matched_keyword") or ""

    if title_first and _title_domain(title):
        domain = _title_domain(title)
    # If we already have a valid domain key, use it directly
    elif kw_category and kw_category in DOMAIN_CV_MAP:
        domain = kw_category
    else:
        domain = resolve_cv_domain(title, matched_kw)

    # Look for domain CV in same directory as base_cv_path
    cv_dir = os.path.dirname(os.path.abspath(base_cv_path))
    domain_filename = DOMAIN_CV_MAP.get(domain, "")
    domain_cv_path = os.path.join(cv_dir, domain_filename) if domain_filename else ""

    if domain_cv_path and os.path.exists(domain_cv_path):
        logging.info(f"Domain CV resolved: {domain} -> {domain_filename}")
        return domain_cv_path, domain

    # Fallback: use the generic base CV
    logging.warning(
        f"Domain CV '{domain_filename}' not found in {cv_dir}, "
        f"falling back to {os.path.basename(base_cv_path)}"
    )
    return base_cv_path, domain


# === Saved versions (Phase 27) ===

def _job_id_of(job: dict):
    """The scrapes row id of *job* as an int, or None."""
    for key in ("id", "job_id"):
        value = (job or {}).get(key)
        if value is None or (isinstance(value, float) and value != value):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _version_dirs(job_id, now: datetime) -> tuple:
    """(cv_dir, cl_dir, generation_id) for ONE generation:
    cv/cv_YYYYMMDD/j<job_id>_<HHMMSS>/ and cl/cl_YYYYMMDD/j<job_id>_<HHMMSS>/.
    Every generation gets its own directory, so a regeneration never overwrites
    an earlier version (the file name inside — what the employer sees — stays
    clean). The CV directory is claimed atomically; a second generation in the
    same second gets a _2, _3 ... suffix. generation_id = YYYYMMDD-HHMMSS-<job_id>."""
    day, hms = now.strftime("%Y%m%d"), now.strftime("%H%M%S")
    jid = str(job_id) if job_id is not None else "0"
    for n in range(1, 1000):
        suffix = f"_{n}" if n > 1 else ""
        leaf = f"j{jid}_{hms}{suffix}"
        cv_dir = os.path.join("cv", f"cv_{day}", leaf)
        cl_dir = os.path.join("cl", f"cl_{day}", leaf)
        try:
            os.makedirs(cv_dir)          # exist_ok=False: the claim
        except FileExistsError:
            continue
        os.makedirs(cl_dir, exist_ok=True)
        return cv_dir, cl_dir, f"{day}-{hms}-{jid}" + (f"-{n}" if n > 1 else "")
    raise RuntimeError(f"no free version directory for job {jid} at {day} {hms}")


# === CV truth gates (Phase 27) ===
# The summary is the only LLM-written text a CV can ship. It ships only when
# every check on the FINAL text passes; otherwise the routed base variant's own
# truth-audited summary ships, verbatim. EXPERIENCE always ships verbatim,
# truth-audited bullets (B0: the routed base's own; B2: selected from the pool).

# Worst first: the overall CV state is the first of these found among the parts.
_STATE_ORDER = ("artifact_mismatch", "input_incomplete", "invalid", "check_unavailable")


def _shipped_text(experience: list, skills_text: str = "") -> str:
    """The verbatim (base / pool) text a CV ships besides its summary: every
    role's title line and bullets, plus SKILLS — where a judge's experience or
    skills quote must be found to count as a dispute about base text.

    The roles are included a second time exactly as the judge reads them
    (cv_tailor._format_experience_for_review joins a role's title and its
    company / date line with " | "). The stored form keeps those two lines
    apart, so a quoted role header was never found here, and a dispute about a
    base header was charged to the summary: a good tailored summary was thrown
    away over one base header line."""
    from .cv_tailor import _format_experience_for_review
    parts = []
    for role in experience or []:
        parts.append(str(role.get("title") or ""))
        parts.append(str(role.get("description") or ""))
    parts.append(str(skills_text or ""))
    parts.append(_format_experience_for_review(experience or []))
    return "\n".join(p for p in parts if p)


def _summary_problems(summary: str, check: dict, corpus: list, sanitized,
                      shipped_text: str = None) -> list:
    """Why a generated summary may NOT ship ([] = it may): the truth check on
    the final text must be complete with no unsupported/composition claim
    that is (or may be) the summary's — including one the judge filed under
    experience/skills whose quote is summary text or is not in *shipped_text*
    (cv_tailor.summary_verdict); every number must trace to METRICS_BANK_TEXT
    + the source bank; no known-stale claim (cv_facts.FORBIDDEN); no posting
    trap token."""
    from .cv_tailor import summary_verdict
    problems = []
    verdict = summary_verdict(check, summary=summary, shipped_text=shipped_text)
    if verdict:
        problems.append(verdict)
    numbers = cv_facts.check_numbers(summary, corpus)
    if numbers:
        problems.append("number(s) not in the candidate's verified sources: " + ", ".join(numbers))
    stale = cv_facts.forbidden_hits(summary)
    if stale:
        problems.append("known-stale claim(s): " + ", ".join(stale))
    leaked = leaked_trap_tokens(summary, sanitized)
    if leaked:
        problems.append("posting trap token(s): " + ", ".join(leaked))
    return problems


def _choose_summary(candidate, base_summary: str, experience: list, judge_args: dict,
                    corpus: list, sanitized) -> dict:
    """The summary gate. Returns {summary, source, reason, state, check,
    corrected, rejected_check}:
      source  'tailored'      the generated summary, every gate clean
              'corrected'     the judge's corrected summary, after a complete
                              and clean RE-check of its own
              'base_fallback' the routed base's summary, verbatim
      state   'ok' | 'check_unavailable' (judge incomplete/errored, or nothing
              to check) | 'invalid' (a gate proved a claim unsupported)
      check   the grade_cv_quality result that decided (None when no judge ran)"""
    out = {"summary": base_summary, "source": "base_fallback", "reason": "",
           "state": "check_unavailable", "check": None, "corrected": None,
           "rejected_check": None}
    if not candidate:
        out["reason"] = "no tailored summary to check"
        return out

    shipped = _shipped_text(experience, judge_args.get("skills_text", ""))
    check = grade_cv_quality({"summary": candidate, "experience": experience}, **judge_args)
    out["check"] = check
    problems = _summary_problems(candidate, check, corpus, sanitized, shipped)
    if not problems:
        out.update(summary=candidate, source="tailored", state="ok")
        return out
    out["state"] = "invalid" if check.get("state") == "ok" else "check_unavailable"
    out["reason"] = "; ".join(problems)

    corrected = check.get("corrected_summary") if check.get("state") == "ok" else None
    corrected = corrected.strip() if isinstance(corrected, str) else ""
    if corrected and _norm_ws(corrected) not in (_norm_ws(candidate), _norm_ws(base_summary)):
        recheck = grade_cv_quality({"summary": corrected, "experience": experience}, **judge_args)
        re_problems = _summary_problems(corrected, recheck, corpus, sanitized, shipped)
        if not re_problems:
            out.update(summary=corrected, source="corrected", state="ok", check=recheck,
                       corrected=corrected, rejected_check=check,
                       reason=f"tailored summary failed ({out['reason']}); the judge's "
                              f"corrected summary passed a clean re-check")
            return out
        out["reason"] += "; corrected summary also failed: " + "; ".join(re_problems)
        out["rejected_check"] = recheck
    return out


def _select_b2(job: dict, base_parsed: dict, base_path: str) -> tuple:
    """B2 selection through cv_pool (lazy import — the flag keeps it off by
    default). Returns (experience | None, info). None = ship the base bullets."""
    try:
        from .cv_pool import select_for_job
        result = select_for_job(job, base_parsed, base_path)
    except Exception as e:
        return None, {"used": False, "fallback_reason": f"selection failed: {e!r}"}
    if not result:
        return None, {"used": False, "fallback_reason": "no B2 selection for this job "
                                                        "(pool missing or stale, old base layout, or an error)"}
    experience = result.get("experience")
    base_exp = base_parsed.get("experience") or []
    if (not isinstance(experience, list) or len(experience) != len(base_exp)
            or any(_norm_ws(a.get("title")) != _norm_ws(b.get("title"))
                   for a, b in zip(experience, base_exp))):
        return None, {"used": False, "fallback_reason": "selection does not match the base layout",
                      "selection": result.get("selection")}
    shipped = [{**b, "description": str(a.get("description") or "")}
               for a, b in zip(experience, base_exp)]
    return shipped, {"used": True, "selection": result.get("selection")}


def _verify_membership(docx_path: str, base_parsed: dict) -> dict:
    try:
        from .cv_pool import verify_membership
        return verify_membership(docx_path, base_parsed)
    except Exception as e:
        return {"ok": False, "violations": [{"kind": "check_failed", "text": f"{type(e).__name__}: {e}"}]}


def _short(items, n: int = 3) -> str:
    return json.dumps((items or [])[:n], default=str)[:400]


def _write_verified_cv(base_path: str, cv_path: str, final: dict, base_data: dict,
                       base_parsed: dict, b2_used: bool, job_id) -> dict:
    """Write *final* into a copy of the base, then prove the SAVED file says
    exactly that (B2: the pool-membership gate first; always: the per-role
    round trip). On a failure the base bullets are written instead (state
    'artifact_mismatch', ERROR log with the CV_ARTIFACT_MISMATCH prefix); if
    even that does not round-trip, the base file itself ships, byte for byte.
    Returns {final, state, notes, b2_used} — final = what the file now says."""
    from .cv_tailor import fill_docx_with_tailored_data, verify_docx_roundtrip
    state, notes = "ok", {}
    fill_docx_with_tailored_data(base_path, cv_path, final)

    if b2_used:
        membership = _verify_membership(cv_path, base_parsed)
        if not membership.get("ok"):
            logging.error(f"CV_ARTIFACT_MISMATCH job {job_id}: B2 membership gate failed — "
                          f"base bullets written instead {_short(membership.get('violations'))}")
            notes["membership"] = (membership.get("violations") or [])[:5]
            state, b2_used = "artifact_mismatch", False
            final = {"summary": final["summary"], "experience": deepcopy(base_data["experience"])}
            fill_docx_with_tailored_data(base_path, cv_path, final)

    roundtrip = verify_docx_roundtrip(cv_path, final, base_docx_path=base_path)
    if not roundtrip.get("ok"):
        mismatches = roundtrip.get("mismatches") or []
        logging.error(f"CV_ARTIFACT_MISMATCH job {job_id}: the saved CV is not what was intended — "
                      f"base bullets written instead {_short(mismatches)}")
        notes["roundtrip"] = mismatches[:5]
        state, b2_used = "artifact_mismatch", False
        keep_summary = not any(m.get("where") in ("summary", "file", "static") for m in mismatches)
        final = {"summary": final["summary"] if keep_summary else base_data["summary"],
                 "experience": deepcopy(base_data["experience"])}
        fill_docx_with_tailored_data(base_path, cv_path, final)
        if not verify_docx_roundtrip(cv_path, final, base_docx_path=base_path).get("ok"):
            logging.error(f"CV_ARTIFACT_MISMATCH job {job_id}: the base-bullet rewrite differs too — "
                          f"shipping the base file verbatim")
            shutil.copyfile(base_path, cv_path)
            final = deepcopy(base_data)
            notes["shipped_base_file"] = True
    return {"final": final, "state": state, "notes": notes, "b2_used": b2_used}


def _drop_lowest_ranked(experience: list) -> bool:
    """B2 length rule (deterministic): remove the LAST bullet — the lowest
    ranked, since B2 puts the proof of the most important requirements first —
    of the role with the most bullets above its floor (ties: the older role).
    False when every role is at its floor."""
    from .cv_tailor import _min_bullets_for_role
    best = None
    for i, role in enumerate(experience):
        lines = [line for line in str(role.get("description") or "").split("\n") if line.strip()]
        surplus = len(lines) - _min_bullets_for_role(i, experience)
        if surplus > 0 and (best is None or surplus >= best[0]):
            best = (surplus, i, lines)
    if best is None:
        return False
    _, i, lines = best
    experience[i] = {**experience[i], "description": "\n".join(lines[:-1])}
    return True


def save_documents_for_job(
    base_cv_path: str,
    profile_text: str,
    job: dict,
    model_generation: str = DEFAULT_GENERATION_MODEL,
    summary_prompt_template: str = None,
    experience_prompt_template: str = None,
    transcript_text: str = "",
    content_mode: str = None,
    shadow_judge: bool = None,
):
    """Generate a CV and a cover letter for *job* into their own version
    directories. Returns (cv_path, cl_path, cv_id, cl_id, cv_quality_result):
    paths only for files that exist, drive ids always None.

    shadow_judge: run the B1 shadow truth judge (None = CV_B1_SHADOW_JUDGE;
    generate_package passes False on the synchronous paths).

    The CV (Phase 27):
    - EXPERIENCE is verbatim, truth-audited text. B0 (default): the routed base
      variant's own bullets. B2 (CV_CONTENT_MODE): the candidate's bullets selected and
      ordered for this job from the pool (cv_pool), title-first routing.
    - The SUMMARY is the only LLM text that can ship, through _choose_summary.
    - The main generation prompt still runs unchanged; its reworded bullets
      never ship — they are kept, with their own truth verdict, as the B1
      shadow (audit["shadow_b1"]).
    - Keyword coverage and the hiring-manager grade are diagnostics on the
      FINAL text; nothing they return changes the file.
    - A posting under CV_THIN_POSTING_CHARS (after untrusted-text stripping)
      ships the general version: base summary + base bullets, and the
      templated cover letter with no model call (audit["cl"] says why).
    - The saved file is verified (round trip; B2: pool membership); a mismatch
      ships the base bullets.

    cv_quality_result: {flag, state, audit, corrected_summary, base_variant,
    content_mode, summary_source, generation_id}. flag stays on the existing
    badges: PASSED only for a complete clean check of exactly what shipped,
    otherwise REVIEW_RECOMMENDED. state: ok / check_unavailable /
    input_incomplete / invalid / artifact_mismatch.
    """
    mode = str(content_mode or CV_CONTENT_MODE or "B0").strip().upper()
    if mode not in CV_CONTENT_MODES:
        mode = "B0"

    # Resolve domain-specific base CV (title-first only in B2)
    resolved_cv_path, domain = _resolve_base_cv_path(base_cv_path, job, title_first=(mode == "B2"))
    # The variant ACTUALLY used: the domain, or the generic file when the
    # domain file is missing.
    base_file = os.path.basename(str(resolved_cv_path))
    base_variant = domain if base_file == DOMAIN_CV_MAP.get(domain) else os.path.splitext(base_file)[0]

    # Setup dirs — one directory per generation, never overwritten
    job_id = _job_id_of(job)
    now = datetime.now()
    cv_dir, cl_dir, generation_id = _version_dirs(job_id, now)

    # File names carry the display name ("Acme", not "Acme (via a staffing
    # board)"); the stored company value is unchanged.
    company_display = display_company(job)
    company_slug = _slug(company_display, max_len=30)
    role_slug = _slug(job.get('title', ''), max_len=40)
    date_str = now.strftime("%Y-%m-%d")
    candidate = os.getenv("CANDIDATE_NAME", "Candidate_Name").replace(" ", "_")
    cv_path = os.path.join(cv_dir, f"{candidate}_Resume_{company_slug}_{role_slug}_{date_str}.docx")
    cl_path = os.path.join(cl_dir, f"{candidate}_Cover_Letter_{company_slug}_{role_slug}_{date_str}.docx")

    # The posting is untrusted data: AI-directed and applicant-instruction
    # sentences are stripped; "thin" is measured on what is left.
    _desc = job.get("description")
    sanitized = sanitize_jd(_desc if isinstance(_desc, str) else "")
    posting = (sanitized.text or "").strip()
    thin = len(posting) < CV_THIN_POSTING_CHARS
    evaluation_date = date.today().isoformat()

    # === CV GENERATION (TEMPLATE EDITING) ===
    # Default keeps the 5-tuple contract intact if generation raises before grading.
    cv_quality_result = {
        "flag": "REVIEW_RECOMMENDED",
        "state": "check_unavailable",
        "audit": {"error": "CV generation did not complete", "generation_id": generation_id,
                  "jd_chars": len(posting), "evaluation_date": evaluation_date},
        "corrected_summary": None,
        "base_variant": base_variant,
        "content_mode": mode,
        "summary_source": "base_fallback",
        "generation_id": generation_id,
    }
    base_data = None
    cv_verified = False
    try:
        logging.info(
            f"Tailoring CV for {job.get('title')} at {job.get('company')} "
            f"using {model_generation} (domain: {domain}, mode: {mode})"
        )

        from .cv_tailor import generate_tailored_cv, parse_docx_to_structure, extract_section_text

        base_parsed = parse_docx_to_structure(resolved_cv_path)
        if not base_parsed.get("summary") and not base_parsed.get("experience"):
            raise RuntimeError(f"base CV {resolved_cv_path} could not be parsed")
        # The base is every fallback's known-good content. One that repeats a
        # retired claim is not, and never ships.
        stale = cv_facts.forbidden_hits("\n".join(
            [base_parsed.get("summary", "")]
            + [str(e.get("description") or "") for e in base_parsed.get("experience", [])]))
        if stale:
            raise RuntimeError(f"base CV {base_file} repeats retired claims ({', '.join(stale)}) — "
                               f"not a known-good fallback, no CV shipped")
        base_data = {"summary": base_parsed.get("summary", ""),
                     "experience": deepcopy(base_parsed.get("experience", []))}
        # Ground the auditor on the WHOLE base CV (summary + every role's
        # bullets), not just the summary — otherwise tools/facts that live
        # only in the base bullets (a named tool, vendor work) get mis-flagged
        # as fabrications.
        base_grounding = "\n".join(
            [base_data["summary"]]
            + [str(role.get("description", "")) for role in base_data["experience"]]
        ).strip()
        skills_text = extract_section_text(resolved_cv_path, "SKILLS")

        # Ground the auditor on the SAME transcript the generator saw (named
        # tools live in its TOOLS INVENTORY), so verified tools are not
        # mis-flagged as fabrications.
        judge_transcript = transcript_text or _load_transcript_context(domain=domain)
        judge_args = {"profile_text": profile_text, "target_job": job,
                      "base_summary": base_grounding, "transcript_context": judge_transcript,
                      "evaluation_date": evaluation_date, "skills_text": skills_text}

        states = {"input": "input_incomplete" if thin else "ok"}
        audit_extra = {"posting": {"chars": len(posting),
                                   "ai_directed_stripped": len(sanitized.ai_directed),
                                   "applicant_instructions": len(sanitized.applicant_instructions)}}

        # 1. Generate (unchanged main prompt). Only the summary can ship; the
        # reworded bullets become the B1 shadow. A thin posting is not tailored.
        tailored_summary, b1_experience, gen_error = None, None, None
        if not thin:
            try:
                tailored = generate_tailored_cv(
                    resolved_cv_path,
                    profile_text,
                    job,
                    model_name=model_generation,
                    summary_prompt_template=summary_prompt_template,
                    experience_prompt_template=experience_prompt_template,
                    domain=domain,
                    transcript_text=transcript_text,
                ) or {}
                tailored_summary = str(tailored.get("summary") or "").strip() or None
                b1_experience = tailored.get("experience")
            except Exception as e:
                gen_error = f"generation failed: {e}"
                logging.error(f"CV generation call failed (base content ships): {e}")

        # 2. What EXPERIENCE ships: the base's own bullets (B0, thin postings,
        # any B2 failure) or the B2 selection.
        shipped_experience = deepcopy(base_data["experience"])
        b2_used = False
        if mode == "B2" and not thin:
            selected, b2_info = _select_b2(job, base_parsed, resolved_cv_path)
            audit_extra["b2"] = b2_info
            if selected is not None:
                shipped_experience, b2_used = selected, True

        # 3. The summary gate, on the FINAL text (summary + shipped bullets + skills).
        candidate_summary = tailored_summary
        if candidate_summary and _norm_ws(candidate_summary) == _norm_ws(base_data["summary"]):
            candidate_summary = None        # the generator echoed the base: nothing tailored
        corpus = cv_facts.verified_corpus()
        chosen = _choose_summary(candidate_summary, base_data["summary"], shipped_experience,
                                 judge_args, corpus, sanitized)
        if thin:
            chosen.update(state="input_incomplete",
                          reason=f"thin posting ({len(posting)} chars after cleaning) — general version")
        elif not candidate_summary:
            chosen["reason"] = gen_error or ("the generator returned the base summary unchanged"
                                             if tailored_summary else "no tailored summary returned")
        if chosen["source"] == "base_fallback" and candidate_summary:
            logging.info(f"CV summary gate: base summary ships ({chosen['reason'][:200]})")

        # 4. Write, then prove the saved file says exactly that.
        final = {"summary": chosen["summary"], "experience": shipped_experience}
        written = _write_verified_cv(resolved_cv_path, cv_path, final, base_data,
                                     base_parsed, b2_used, job_id)
        cv_verified = True
        final, b2_used = written["final"], written["b2_used"]
        states["artifact"] = written["state"]
        if written["notes"]:
            audit_extra["artifact"] = written["notes"]
        if _norm_ws(final["summary"]) != _norm_ws(chosen["summary"]):
            chosen.update(summary=final["summary"], source="base_fallback", corrected=None,
                          reason=(chosen["reason"] + "; " if chosen["reason"] else "")
                                 + "saved-file mismatch")
        states["summary"] = chosen["state"]

        # 5. LENGTH — measured, never LLM-trimmed (B0 keeps the base's length).
        # B2 over the ceiling drops its lowest-ranked selected bullets.
        if CV_LENGTH_GATE_ENABLED:
            try:
                pages = estimate_docx_pages(cv_path)
                dropped = 0
                while b2_used and pages > CV_PAGE_CEILING and dropped < 8:
                    shorter = deepcopy(final["experience"])
                    if not _drop_lowest_ranked(shorter):
                        break
                    dropped += 1
                    cv_verified = False
                    written = _write_verified_cv(resolved_cv_path, cv_path,
                                                 {"summary": final["summary"], "experience": shorter},
                                                 base_data, base_parsed, True, job_id)
                    cv_verified = True
                    final, b2_used = written["final"], written["b2_used"]
                    if written["state"] != "ok":
                        states["artifact"] = written["state"]
                        audit_extra.setdefault("artifact", {}).update(written["notes"])
                    pages = estimate_docx_pages(cv_path)
                audit_extra["page_estimate"] = pages
                if dropped:
                    audit_extra["b2_length_dropped"] = dropped
                if pages > CV_PAGE_CEILING:
                    audit_extra["length_flag"] = "CV_TOO_LONG"
                    logging.warning(f"CV length gate: ~{pages} pages > {CV_MAX_PAGES}")
                else:
                    logging.info(f"CV length gate: ~{pages} pages — fits 2-page rule")
            except Exception as e:
                if not cv_verified:
                    raise       # a half-done rewrite never ships: base fallback below
                logging.warning(f"CV length gate failed (non-fatal): {e}")

        # 6. B1 SHADOW — the generator's reworded bullets, never shipped, kept
        # with their own truth verdict for the offline comparison.
        if b1_experience:
            shadow = {"summary": tailored_summary,
                      "experience": [{"title": r.get("title", ""), "description": r.get("description", "")}
                                     for r in b1_experience if str(r.get("description") or "").strip()]}
            run_shadow = CV_B1_SHADOW_JUDGE if shadow_judge is None else bool(shadow_judge)
            if not run_shadow:
                shadow["judge_skipped"] = True
            if run_shadow:
                try:
                    shadow_check = grade_cv_quality({"summary": tailored_summary or "",
                                                     "experience": b1_experience}, **judge_args)
                    shadow.update(flag=shadow_check.get("flag"), state=shadow_check.get("state"),
                                  claims=(shadow_check.get("audit") or {}).get("claims") or [])
                    if (shadow_check.get("audit") or {}).get("error"):
                        shadow["error"] = shadow_check["audit"]["error"]
                    states["b1_shadow"] = shadow_check.get("state")
                except Exception as e:
                    shadow.update(state="check_unavailable", error=str(e))
            audit_extra["shadow_b1"] = shadow

        # 7. DIAGNOSTICS on the FINAL text (shipped summary + bullets + skills):
        # ATS keyword coverage and the hiring-manager grade. Nothing they return
        # is written into the CV (keyword gaps still feed the letter's
        # never-claim list).
        if not thin and CV_KEYWORD_COVERAGE_ENABLED:
            try:
                coverage = check_keyword_coverage(
                    final, job, profile_text=profile_text, base_summary=base_grounding,
                    transcript_context=judge_transcript, skills_text=skills_text)
                audit_extra["keyword_coverage"] = coverage
                logging.info(f"CV keyword coverage: {coverage.get('summary', '')}")
            except Exception as e:
                logging.warning(f"CV keyword coverage step failed (non-fatal): {e}")
        if not thin and CV_HM_GRADER_ENABLED:
            try:
                from .cv_tailor import grade_cv_hiring_manager
                hm = grade_cv_hiring_manager(final, job, evaluation_date=evaluation_date,
                                             skills_text=skills_text)
                audit_extra["hm_grade"] = hm
                overall = hm.get("overall")
                if isinstance(overall, (int, float)) and overall < CV_HM_GRADE_THRESHOLD:
                    audit_extra["hm_below_threshold"] = True
                logging.info(f"CV hiring-manager grade: {overall} ({hm.get('verdict', '?')})")
            except Exception as e:
                logging.warning(f"CV hiring-manager grade step failed (non-fatal): {e}")

        # 8. Verdict. PASSED only for a complete, clean check of exactly what
        # shipped; everything else is REVIEW_RECOMMENDED (no new badge).
        overall_state = next((s for s in _STATE_ORDER
                              if s in (states.get("input"), states.get("summary"), states.get("artifact"))),
                             "ok")
        check = chosen["check"] or {}
        if (overall_state == "ok" and chosen["source"] in ("tailored", "corrected")
                and check.get("flag") == "PASSED"):
            flag = "PASSED"
        else:
            flag = "REVIEW_RECOMMENDED"

        audit = dict(check.get("audit") or {})
        if not chosen["check"]:
            if thin:
                audit["note"] = chosen["reason"]
            else:
                audit["error"] = chosen["reason"] or "no truth check ran"
        if chosen["source"] != "tailored" and candidate_summary:
            audit["tailored_summary"] = candidate_summary
        if chosen["rejected_check"]:
            rejected = chosen["rejected_check"].get("audit") or {}
            audit["rejected_summary_check"] = {"state": chosen["rejected_check"].get("state"),
                                               "claims": rejected.get("claims"),
                                               "error": rejected.get("error")}
        if chosen["check"]:
            from .cv_tailor import flagged_claims, quote_in
            # Verbatim base/pool text the judge disputes: for the maintainer, never a
            # rewrite. Only a claim whose quote IS in the shipped bullets or
            # skills — a mislabelled summary claim is not a base-CV issue.
            shipped = _shipped_text(final["experience"], skills_text)
            disputed = [c for c in flagged_claims(audit)
                        if c.get("section") in ("experience", "skills")
                        and quote_in(c.get("quote"), shipped)]
            if disputed:
                audit["base_claims_disputed"] = disputed
        audit.update(audit_extra)
        audit.update({
            "state": overall_state,
            "states": states,
            "summary_source": chosen["source"],
            "summary_reason": chosen["reason"],
            "judge_flag": check.get("flag"),
            "content_mode": mode,
            "base_variant": base_variant,
            "evaluation_date": evaluation_date,
            "jd_chars": len(posting),
            "generation_id": generation_id,
        })
        cv_quality_result = {
            "flag": flag,
            "state": overall_state,
            "audit": audit,
            "corrected_summary": chosen["corrected"],
            "base_variant": base_variant,
            "content_mode": mode,
            "summary_source": chosen["source"],
            "generation_id": generation_id,
        }
        logging.info(f"CV quality: {flag} (state {overall_state}, summary {chosen['source']})")
        logging.info(f"Saved Tailored CV: {cv_path}")

    except Exception as e:
        logging.error(f"CV Generation Failed: {e}")
        # Known-good fallback: an unverified file never ships. If the base
        # parsed, the truth-audited base itself ships, byte for byte.
        if not cv_verified:
            try:
                if base_data is not None:
                    shutil.copyfile(resolved_cv_path, cv_path)
                    cv_quality_result["audit"] = {
                        "error": f"CV generation failed ({e}); the base CV shipped verbatim",
                        "state": "check_unavailable", "summary_source": "base_fallback",
                        "content_mode": mode, "base_variant": base_variant,
                        "evaluation_date": evaluation_date, "generation_id": generation_id}
                else:
                    cv_quality_result["audit"]["error"] = f"CV generation failed: {e}"
                    if os.path.exists(cv_path):
                        os.remove(cv_path)
            except Exception as copy_error:
                logging.error(f"CV fallback copy failed: {copy_error}")
                try:
                    if os.path.exists(cv_path):
                        os.remove(cv_path)
                except Exception:
                    pass

    # === COVER LETTER GENERATION (NEW FILE) ===
    # Phase 27: build_cover_letter runs the deterministic checks, the 420-word
    # cap and the letter truth check (templated letter when the check is
    # unavailable); create_formatted_cl always signs. A failed generation
    # writes no letter. The letter's audit rides in the CV audit as "cl".
    # A thin posting gets the templated letter with no model call (thin=).
    cl_audit = {"state": "generation_failed", "template_used": False, "signed": False,
                "company_display": company_display}
    try:
        if not thin:
            logging.info(f"Generating Cover Letter for {job.get('title')} using {model_generation}")
        # Feed the CV keyword auditor's confirmed gaps to the CL as a
        # never-claim list (the CL used to have no gap discipline at all).
        _gaps = ((cv_quality_result.get("audit") or {}).get("keyword_coverage") or {}).get("gaps") or []
        cl_content, cl_audit = build_cover_letter(
            profile_text, job, model=model_generation, domain=domain,
            transcript_text=transcript_text, never_claim=_gaps, base_cv_path=resolved_cv_path,
            company_display=company_display, evaluation_date=evaluation_date, sanitized=sanitized,
            thin=thin,
        )
        if not cl_content:
            raise RuntimeError(cl_audit.get("reason") or "cover letter generation returned no text")
        links = _contact_links(resolved_cv_path)
        cl_audit["links"] = len(links)
        create_formatted_cl(cl_content, job, company_display=company_display, links=links).save(cl_path)
        logging.info(f"Saved CL: {cl_path} ({cl_audit.get('state')}, {cl_audit.get('words')} words"
                     f"{', templated' if cl_audit.get('template_used') else ''})")
    except Exception as e:
        cl_audit = {**cl_audit, "error": str(e)}
        logging.error(f"CL Generation Failed: {e}")
    cl_audit["signed"] = os.path.exists(cl_path)      # every letter code writes is signed
    if isinstance(cv_quality_result.get("audit"), dict):
        cv_quality_result["audit"]["cl"] = cl_audit

    # Only return paths for files that actually exist on disk.
    # If CV or CL generation failed, the intended path was set but the file
    # was never written — returning a phantom path would store a broken
    # reference in the DB and confuse the UI.
    cv_path_final = cv_path if os.path.exists(cv_path) else None
    cl_path_final = cl_path if os.path.exists(cl_path) else None

    # Drive upload RETIRED: the copies were unused. Local files +
    # the app's file serving are the source of truth; nothing is uploaded.
    # The 5-tuple contract is preserved — drive ids are simply always None.
    cv_id, cl_id = None, None

    return cv_path_final, cl_path_final, cv_id, cl_id, cv_quality_result


# === Shared generation path (Phase 27) ===
# Every route that makes documents (API single + batch + manual-add, chat tool,
# pipeline auto-generation) goes through generate_package: one lock per job,
# one saved version per generation, QA persisted every time.

_JOB_LOCKS: dict = {}
_JOB_LOCKS_GUARD = threading.Lock()


def _job_lock(key) -> threading.Lock:
    with _JOB_LOCKS_GUARD:
        return _JOB_LOCKS.setdefault(key, threading.Lock())


def generating_job_ids() -> list:
    """Jobs whose documents this process is generating right now (their lock
    is held), so the board can say "Generating…" instead of looking idle: a
    job added by hand starts its own background generation, and a click on
    Generate meanwhile only answers 'in_progress'."""
    with _JOB_LOCKS_GUARD:
        return sorted(k for k, lock in _JOB_LOCKS.items()
                      if isinstance(k, int) and lock.locked())


def _sha256(path) -> str:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_package(job: dict, db, profile_text: str, *, base_cv_path: str = "base_cv.docx",
                     model: str = None, source: str = "api") -> dict:
    """Generate one CV + cover letter for *job* (the merged job dict, with its
    scrapes 'id') and persist it.

    - A per-job in-process lock: while one generation for a job runs, a second
      caller gets {'status': 'in_progress'} at once (no double generation).
    - save_documents_for_job writes into a fresh version directory. The B1
      shadow judge runs only for the background sources (CV_B1_SHADOW_SOURCES:
      'pipeline', 'batch'); a synchronous click does not wait on it.
    - Persists through db (each call guarded with hasattr so older fakes work):
        record_document_version(job_id, generation_id, cv_path, cl_path, meta)
        save_generated_documents(job_id, cv_path, cl_path, flag, audit_json,
                                 domain_used)
          ONE transaction: the active CV, the active letter and the CV's QA
          verdict always come from this generation. A new CV with no letter
          clears the old letter (it belonged to an older CV); a letter with no
          CV keeps the old CV and its verdict.
        (older DB objects: update_job_documents + update_job_cv_quality)
      A failed generation never clears an existing path, and the quality
      verdict is only replaced together with the CV it describes.

    Returns {'status', 'job_id', 'cv_path', 'cl_path', 'flag', 'state',
    'audit', 'version_id', 'generation_id'} (+ 'error' when generation
    raised). status: 'completed' (both files), 'partial' (one), 'failed'
    (none) or 'in_progress'. Never raises.
    """
    job_id = _job_id_of(job)
    lock_key = job_id if job_id is not None else ("no-id", str(job.get("title")), str(job.get("company")))
    lock = _job_lock(lock_key)
    empty = {"job_id": job_id, "cv_path": None, "cl_path": None, "flag": None, "state": None,
             "audit": None, "version_id": None, "generation_id": None}
    if not lock.acquire(blocking=False):
        logging.info(f"generate_package: job {job_id} is already generating — skipped ({source})")
        return {"status": "in_progress", **empty}
    try:
        model = model or DEFAULT_GENERATION_MODEL
        # A job listed without its posting (Glassdoor, ZipRecruiter, small
        # boards) gets its text looked up first: another saved copy, then the
        # posting page. Without it the documents can only be the general ones.
        try:
            from .posting_text import fill_posting_text
            fill_posting_text(job, db)
        except Exception as e:
            logging.warning(f"generate_package: posting text lookup failed for job {job_id}: {e}")
        try:
            cv_path, cl_path, _cv_id, _cl_id, quality = save_documents_for_job(
                base_cv_path, profile_text, job, model_generation=model,
                shadow_judge=CV_B1_SHADOW_JUDGE and source in CV_B1_SHADOW_SOURCES)
        except Exception as e:
            logging.error(f"generate_package: generation failed for job {job_id} ({source}): {e}")
            return {"status": "failed", **empty, "error": str(e)}
        quality = quality or {}
        audit = quality.get("audit") or {}
        generation_id = quality.get("generation_id") or audit.get("generation_id")
        base_variant = quality.get("base_variant")
        flag = quality.get("flag", "REVIEW_RECOMMENDED")

        meta = {
            "content_mode": quality.get("content_mode"),
            "base_variant": base_variant,
            "cohort_tag": CV_COHORT_TAG,
            "model": model,
            "effort": MODEL_EFFORT.get(model),
            "jd_chars": audit.get("jd_chars"),
            "summary_source": quality.get("summary_source"),
            "state": quality.get("state"),
            "states": audit.get("states"),
            "flag": flag,
            "sha256s": {"cv": _sha256(cv_path), "cl": _sha256(cl_path)},
            "evaluation_date": audit.get("evaluation_date"),
            "source": source,
        }

        version_id = None
        if job_id is not None and db is not None and (cv_path or cl_path):
            if hasattr(db, "record_document_version"):
                try:
                    version_id = db.record_document_version(job_id, generation_id, cv_path, cl_path, meta)
                except Exception as e:
                    logging.error(f"generate_package: version record failed for job {job_id}: {e}")
            audit_json = json.dumps(audit, default=str)
            if hasattr(db, "save_generated_documents"):
                try:
                    if not db.save_generated_documents(job_id, cv_path, cl_path, flag,
                                                       audit_json, base_variant):
                        logging.error(f"generate_package: documents not saved for job {job_id}")
                except Exception as e:
                    logging.error(f"generate_package: documents not saved for job {job_id}: {e}")
            else:
                if hasattr(db, "update_job_documents"):
                    try:
                        db.update_job_documents(job_id, cv_path=cv_path, cl_path=cl_path)
                    except Exception as e:
                        logging.error(f"generate_package: document paths not saved for job {job_id}: {e}")
                if cv_path and hasattr(db, "update_job_cv_quality"):
                    try:
                        db.update_job_cv_quality(job_id, flag, audit_json, base_variant)
                    except Exception as e:
                        logging.error(f"generate_package: CV quality not saved for job {job_id}: {e}")

        status = "completed" if (cv_path and cl_path) else "partial" if (cv_path or cl_path) else "failed"
        logging.info(f"generate_package: job {job_id} {status} ({source}, {generation_id}, {flag})")
        return {"status": status, "job_id": job_id, "cv_path": cv_path, "cl_path": cl_path,
                "flag": flag if cv_path else None, "state": quality.get("state"),
                "audit": audit, "version_id": version_id, "generation_id": generation_id}
    finally:
        lock.release()

# === Cover letters (Phase 27) ===
# The model writes the body; code owns everything an employer could catch us
# on: the salutation, the signature (always "Sincerely," + the candidate's name, exactly
# once), the company name (display_company), the length cap, and the truth of
# every sentence. A sentence ships only if it survives the deterministic
# checks (numbers traced to the candidate sources, no known-stale claim, no posting trap
# token) and a COMPLETE letter truth check. When the check is unavailable, or
# too little survives, a templated letter built from METRICS_BANK facts ships
# instead. A thin posting gets that templated letter directly (no model call).
# A failed generation writes no letter at all (never a placeholder).

CL_TARGET_WORDS = "300-400"   # asked of the model
CL_MAX_BODY_WORDS = 420       # enforced in code (salutation + signature not counted)
CL_MIN_BODY_WORDS = 120       # fewer left after the checks -> templated letter
CL_SALUTATION = "Dear Hiring Manager,"
CL_SIGN_OFF = "Sincerely,"


def _candidate_name() -> str:
    return os.getenv("CANDIDATE_NAME", "Candidate Name")


# --- Sign-off and salutation --------------------------------------------------
# Any sign-off the model wrote is stripped and code always signs (a bare
# "Sincerely," from the model must never ship without a name).
_SIGN_OFF_WORDS = (r"(?:sincerely(?:\s+yours)?|yours\s+(?:sincerely|truly)"
                   r"|(?:best|kind(?:est)?|warm(?:est)?|with\s+best)\s+regards|regards"
                   r"|best\s+wishes|all\s+the\s+best|warmly|respectfully(?:\s+yours)?"
                   r"|best|thank\s+you|many\s+thanks|thanks|cheers)")
_SIGN_OFF_LINE_RE = re.compile(rf"^(?i:{_SIGN_OFF_WORDS})\s*[,.!:;-]*\s*(?P<rest>.*)$")
# A sign-off glued onto the end of the last paragraph: "... Thank you.
# Sincerely, Jane Doe". Only after a sentence end, so "the best" never matches.
_GLUED_SIGN_OFF_RE = re.compile(
    r"(?<=[.!?])\s+(?i:sincerely(?:\s+yours)?|yours\s+(?:sincerely|truly)"
    r"|(?:best|kind(?:est)?|warm(?:est)?)\s+regards|regards|best\s+wishes|respectfully)\s*,?"
    r"(?:\s+(?:\[[^\]\n]{1,40}\]|[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3}))?\s*$")
_SALUTATION_RE = re.compile(
    r"^(?i:dear|hello|hi|greetings|to\s+whom\s+it\s+may\s+concern)\b[^,\n:]{0,60}[,:]\s*")
# The same on a line of its own with no punctuation ("Dear Hiring Manager").
_SALUTATION_LINE_RE = re.compile(r"^(?i:dear|hello|hi|greetings)\b[^,\n:.!?]{0,60}$")
_CONTACT_RE = re.compile(r"@|https?://|linkedin\.com|\+?\d[\d\s().-]{6,}\d", re.I)


def _letters_only(text: str) -> str:
    return re.sub(r"[^a-z]", "", str(text or "").lower())


def _is_name_line(line: str, name: str) -> bool:
    """The candidate's name (full or first), or a placeholder like "[Your Name]"."""
    s = line.strip().strip(",.")
    if re.fullmatch(r"\[[^\]]{1,40}\]", s):
        return True
    key = _letters_only(s)
    parts = str(name or "").split()
    return bool(key) and key in {_letters_only(name), _letters_only(parts[0]) if parts else ""}


def _is_signature_tail(line: str, name: str) -> bool:
    """A line that can follow a sign-off: a name, a placeholder, a contact
    line, or a short name-like line (no sentence punctuation)."""
    s = line.strip()
    if not s or _is_name_line(s, name):
        return True
    words = s.split()
    if len(words) <= 12 and _CONTACT_RE.search(s):
        return True
    # "Jane D." keeps its period; a sentence ("Thank you.") has a lowercase word.
    return (len(words) <= 4 and not re.search(r"[!?:;]$", s)
            and all(w[:1].isupper() for w in words if w[:1].isalpha()))


def _strip_sign_off(text: str, name: str = None) -> str:
    """*text* without any sign-off the model wrote (Sincerely / Best regards /
    Regards / Kind regards / Warm regards ..., alone, with the candidate name on the same
    line or below it, or glued to the last sentence) and without a trailing
    name or "[Your Name]" line. Body sentences are never touched: "Thank you
    for your time." is a sentence, not a sign-off."""
    name = _candidate_name() if name is None else name
    paras = [p.strip() for p in str(text or "").split("\n") if p.strip()]
    while paras:
        cut = None
        for j in range(len(paras) - 1, max(-1, len(paras) - 5), -1):
            m = _SIGN_OFF_LINE_RE.match(paras[j])
            if (m and _is_signature_tail(m.group("rest"), name)
                    and all(_is_signature_tail(p, name) for p in paras[j + 1:])):
                cut = j
                break
        if cut is None:
            break
        paras = paras[:cut]
    while paras and _is_name_line(paras[-1], name):
        paras.pop()
    if paras:
        paras[-1] = _GLUED_SIGN_OFF_RE.sub("", paras[-1]).rstrip()
        if not paras[-1]:
            paras.pop()
    return "\n".join(paras)


def _strip_salutation(paras: list) -> list:
    """Paragraphs without the model's salutation (code writes its own). A
    salutation run into the first paragraph ("Dear X, I am ...") is cut off."""
    if not paras:
        return paras
    if _SALUTATION_LINE_RE.match(paras[0]) and _count_words(paras[0]) <= 8:
        return paras[1:]
    m = _SALUTATION_RE.match(paras[0])
    if not m:
        return paras
    rest = paras[0][m.end():].strip()
    return ([rest] if rest else []) + paras[1:]


# --- Sentences and words ------------------------------------------------------
_ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "st.", "jr.",
                  "sr.", "inc.", "co.", "corp.", "ltd.", "u.s.", "u.k.", "no.", "approx."}


def _split_sentences(paragraph: str) -> list:
    """Sentences of one paragraph. Splits after . ! ? (plus closing quotes or
    brackets) when the next word starts like a sentence; never after a known
    abbreviation or an initial, never inside a dotted name ("Acme.Studio") or "$1.5M"."""
    text = " ".join(str(paragraph or "").split())
    out, start = [], 0
    for m in re.finditer(r"[.!?][\"”’')\]]*\s+", text):
        nxt = text[m.end():m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"“‘'([$"):
            continue
        words = text[start:m.end()].split()
        last = words[-1].lower().rstrip("\"”’')]") if words else ""
        if last in _ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def _count_words(text: str) -> int:
    return len(str(text or "").split())


def _letter_words(paras: list) -> int:
    return sum(_count_words(s) for p in paras for s in p)


def _letter_paragraphs(text: str) -> list:
    """The model's letter as body paragraphs of sentences: sign-off,
    signature and salutation removed."""
    paras = [p.strip() for p in _strip_sign_off(text).split("\n") if p.strip()]
    return [s for s in (_split_sentences(p) for p in _strip_salutation(paras)) if s]


def _render_letter(paras: list) -> str:
    """Salutation + body paragraphs (no signature: create_formatted_cl signs)."""
    return "\n".join([CL_SALUTATION] + [" ".join(p) for p in paras if p])


def _quote(sentence: str) -> str:
    return sentence if len(sentence) <= 160 else sentence[:157] + "..."


# --- Deterministic checks and the length cap ----------------------------------

# A range written out in words: "from A to B percent" restates a source's
# "A–B%". Models often write a source range this way; without this the
# sentence would fail the number check and be dropped.
_WORDED_RANGE_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:%|percent)?\s+to\s+(\d[\d,]*(?:\.\d+)?)(\s*(?:%|percent\b))")


def _untraced_numbers(sentence: str, corpus: list) -> list:
    """cv_facts.check_numbers for a letter sentence. When numbers are missing
    as written, the sentence still passes if reading "A to B percent" as the
    range "A-B%" traces every number (the range itself must be in a source)."""
    missing = cv_facts.check_numbers(sentence, corpus)
    if missing:
        ranged = _WORDED_RANGE_RE.sub(r"\1-\2\3", sentence)
        if ranged != sentence and not cv_facts.check_numbers(ranged, corpus):
            return []
    return missing


# A sentence that claims something about the CANDIDATE (first person, singular
# or the team's "we"). Its numbers must come from the candidate's verified sources; only a
# sentence about the employer may restate a posting / company-context number.
_FIRST_PERSON_RE = re.compile(r"\b(?:I|I['’](?:ve|m|d|ll)|(?i:my|me|mine|myself)|[Ww]e|[Oo]ur|us)\b")


def _drop_unverified_sentences(paras: list, corpus: list, sanitized, context: list = ()) -> list:
    """Drop (in place) every sentence that carries a number found in none of
    *corpus* (the candidate's verified sources; a sentence with no first-person claim may
    also use the employer *context*: posting, company profile), repeats a
    known-stale claim (cv_facts.FORBIDDEN) or leaks a posting trap token.
    Returns what was dropped, for the audit."""
    employer_corpus = list(corpus) + [str(t or "") for t in context]
    dropped = []
    for p in paras:
        for s in list(p):
            numbers = _untraced_numbers(s, corpus if _FIRST_PERSON_RE.search(s) else employer_corpus)
            stale = cv_facts.forbidden_hits(s)
            leaked = leaked_trap_tokens(s, sanitized) if sanitized is not None else []
            if not (numbers or stale or leaked):
                continue
            p.remove(s)
            stage = "numbers" if numbers else "stale" if stale else "trap"
            detail = ", ".join(numbers or stale or leaked)
            dropped.append({"stage": stage, "detail": detail, "quote": _quote(s)})
    paras[:] = [p for p in paras if p]
    return dropped


def _cap_letter(paras: list, cap: int = CL_MAX_BODY_WORDS) -> list:
    """Enforce the word cap (in place) by dropping WHOLE sentences:
      1. the last sentence of the longest middle paragraph (each middle
         paragraph keeps its first sentence while any has more than one);
      2. then whole middle paragraphs, the last one first;
      3. last resort, the opening paragraph's later sentences.
    Never the opening sentence, never the closing paragraph. Returns the
    dropped sentences; the caller checks whether the cap was met."""
    dropped = []
    while _letter_words(paras) > cap:
        middle = [i for i in range(1, len(paras) - 1) if len(paras[i]) > 1]
        if middle:
            i = max(middle, key=lambda k: (_count_words(" ".join(paras[k])), k))
            dropped.append(paras[i].pop())
        elif len(paras) > 2:
            dropped.extend(paras.pop(len(paras) - 2))
        elif len(paras) > 1 and len(paras[0]) > 1:
            dropped.append(paras[0].pop())
        else:
            break
    return [{"stage": "length", "detail": f"over {cap} words", "quote": _quote(s)} for s in dropped]


# --- Never-claim list ---------------------------------------------------------
# The keyword auditor's "gaps" are free text. Only genuine skill/experience
# gaps belong on the letter's never-claim list; the auditor also emits parser
# notes (a remark about what the ATS parser sees, an exact-phrase or synonym
# suggestion) and behavioural traits (soft-skill and culture wording), which
# are not claims a letter could fake.
_NEVER_CLAIM_NOTE_RE = re.compile(
    r"\bparser\b|\bATS\b|stuffable|\bkeywords?\b|exact\s+phrase|\bsynonyms?\b|low\s+risk\s+to\s+add"
    r"|behaviou?ral|soft[-\s](?:skills?|culture)|\bmindset\b|\battitude\b|\bmorale\b"
    r"|psychological\s+safety|inclusive\s+environment|people\s+skills|trust[-\s]building"
    r"|self[-\s](?:directed|starter|motivated)|team\s+player|work\s+ethic|\bpassion(?:ate)?\b"
    r"|\bcuriosity\b|detail[-\s]oriented|\binterpersonal\b", re.I)
# The auditor's own commentary in a trailing parenthetical ("<tool> (not
# confirmed by name in grounding)") goes; the term itself stays.
_NEVER_CLAIM_COMMENT_RE = re.compile(
    r"\s*\((?=[^()]*\b(?:grounding|confirm(?:ed|s)?|supported|verified|the candidate\s+has|adjacent"
    r"|named\s+as\s+such|not\s+enough|scale\s+gap|beyond)\b)[^()]*\)\s*$", re.I)


def _never_claim_terms(terms) -> list:
    """The genuine skill/experience gaps among the keyword auditor's *terms*:
    parser and behavioural notes dropped, auditor commentary trimmed,
    de-duplicated (case-insensitive), order kept."""
    out, seen = [], set()
    for term in terms or []:
        s = " ".join(str(term or "").split())
        if not s or _NEVER_CLAIM_NOTE_RE.search(s):
            continue
        s = _NEVER_CLAIM_COMMENT_RE.sub("", s).strip() or s
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


# --- Grounding helpers ----------------------------------------------------------

def _company_context(job: dict) -> str:
    """The cached Perplexity company profile (populated by AUTO_ENRICH_MATCHES):
    an APPROVED source for the "why this company" sentence. '' when none.
    Looked up by the stored company name (the intel table's key)."""
    try:
        from .db_manager import DBManager
        intel = DBManager().get_company_intel(str(job.get("company") or ""))
        if intel and intel.get("company_profile"):
            return str(intel["company_profile"])[:1500]
    except Exception:
        pass
    return ""


def _letter_transcript(domain: str, transcript_text: str = "") -> str:
    """The transcript block the letter sees (explicit text, or the domain's
    sections plus the Cover Letter Stories)."""
    if transcript_text and transcript_text.strip():
        return transcript_text.strip()
    return _load_transcript_context(domain=domain, for_cover_letter=True)


def _base_grounding(base_cv_path) -> str:
    """The routed base CV as the letter judge reads it: summary, every role
    (title and company line) with its bullets, and SKILLS. '' if unreadable."""
    if not base_cv_path:
        return ""
    try:
        from .cv_tailor import parse_docx_to_structure, extract_section_text, _format_experience_for_review
        parsed = parse_docx_to_structure(base_cv_path)
        parts = [parsed.get("summary", ""), _format_experience_for_review(parsed.get("experience", []))]
        skills = extract_section_text(base_cv_path, "SKILLS")
        if skills:
            parts.append("SKILLS:\n" + skills)
        return "\n".join(p for p in parts if p).strip()
    except Exception as e:
        logging.warning(f"Cover letter: base CV grounding unavailable ({e})")
        return ""


# Links in the base CV's header line (the candidate's LinkedIn and portfolio). An e-mail
# address is never a link.
_LINK_TOKEN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*"
    r"\.[a-z]{2,}(?:/\S*)?$", re.I)


def _link_key(link: str) -> str:
    return re.sub(r"^(?:https?://)?(?:www\.)?", "", link.strip().lower()).rstrip("/")


def _contact_links(base_cv_path) -> list:
    """The candidate's LinkedIn / portfolio links, as the routed base CV's own contact
    line prints them (paragraphs before the first section header), in that
    order; [] when there are none or the file cannot be read. No new setting:
    the truth-audited base is the source."""
    if not base_cv_path:
        return []
    try:
        from .cv_tailor import _section_header
        doc = Document(base_cv_path)
    except Exception:
        return []
    links, keys = [], set()
    for p in doc.paragraphs[:8]:
        if _section_header(p) is not None:
            break
        candidates = re.split(r"[\s|•·,;]+", p.text)
        candidates += [h.address for h in getattr(p, "hyperlinks", []) if getattr(h, "address", "")]
        for token in candidates:
            token = token.strip("()[]<>").rstrip(".")
            if not token or "@" in token or token.lower().startswith("mailto:"):
                continue
            if not _LINK_TOKEN_RE.match(token) or _link_key(token) in keys:
                continue
            keys.add(_link_key(token))
            links.append(token)
    return links[:3]


# --- Templated letter ---------------------------------------------------------
# The known-good fallback. Its sentences are DATA from the candidate profile
# (letter_template: opening, proofs, proofs_by_domain, fallback_proof; see
# candidate_profile.letter_template), never text written into this module:
# a fallback that tells someone else's story, or a number no source
# supports, would reach an employer exactly when the letter check is down.
# Each proof restates one or two lines of the profile's metrics_text (past
# tense, no employer name). templated_letter still re-checks every sentence
# against METRICS_BANK_TEXT and the stale-claim list and drops a failing one,
# so a profile edit that leaves a proof behind costs a sentence, not the truth.
_LETTER_TEMPLATE = profile_letter_template()
# A profile with no letter_template gets this short, number-free letter.
_GENERIC_OPENING = "My experience lines up closely with the work this role describes."
_GENERIC_PROOF = ("In my recent roles, I took projects from plan to delivery and kept the "
                  "people and vendors involved on schedule.")
_TEMPLATE_OPENING = _LETTER_TEMPLATE["opening"] or _GENERIC_OPENING
_TEMPLATE_CLOSE = ("My CV with the full details of this work is attached. "
                   "Thank you for your time and consideration.")
#: {name: sentence}, as the profile writes them.
_TEMPLATE_PROOFS = _LETTER_TEMPLATE["proofs"]
#: {domain: (names...)}: 2-3 lead proofs per routed domain (DOMAIN_CV_MAP keys), strongest first.
_TEMPLATE_PROOFS_BY_DOMAIN = _LETTER_TEMPLATE["proofs_by_domain"]
_TEMPLATE_FALLBACK_PROOF = _LETTER_TEMPLATE["fallback_proof"] or _GENERIC_PROOF
# The routing default (config.resolve_cv_domain): its proofs serve a domain
# the profile does not list.
_TEMPLATE_DEFAULT_DOMAIN = "producer_core"


def _template_problems(sentence: str) -> list:
    """Numbers in *sentence* that METRICS_BANK_TEXT does not carry, plus any
    stale claim. Empty = safe to ship in the fallback letter."""
    return cv_facts.check_numbers(sentence, [METRICS_BANK_TEXT]) + cv_facts.forbidden_hits(sentence)


def _template_opening() -> str:
    problems = _template_problems(_TEMPLATE_OPENING)
    if problems:
        logging.warning(f"Templated letter: profile opening dropped ({', '.join(problems)})")
        return _GENERIC_OPENING
    return _TEMPLATE_OPENING


def _template_proofs(domain: str) -> list:
    """The routed domain's proof sentences that pass the checks; the
    fallback proof (or the generic one) when none does."""
    names = (_TEMPLATE_PROOFS_BY_DOMAIN.get(domain)
             or _TEMPLATE_PROOFS_BY_DOMAIN.get(_TEMPLATE_DEFAULT_DOMAIN)
             or tuple(_TEMPLATE_PROOFS)[:3])
    kept = []
    for name in names:
        sentence = _TEMPLATE_PROOFS.get(name, "")
        problems = _template_problems(sentence)
        if sentence and not problems:
            kept.append(sentence)
        elif sentence:
            logging.warning(f"Templated letter: proof '{name}' dropped ({', '.join(problems)})")
    if kept:
        return kept
    if not _template_problems(_TEMPLATE_FALLBACK_PROOF):
        return [_TEMPLATE_FALLBACK_PROOF]
    return [_GENERIC_PROOF]


def _plain_title(job: dict) -> str:
    title = job.get("title")
    if title is None or isinstance(title, float):
        return ""
    title = " ".join(str(title).split())
    return "" if title.lower() in ("nan", "none") else title


def templated_letter(job: dict, domain: str, company_display: str) -> str:
    """The deterministic letter (salutation + 3 paragraphs, no signature): one
    sentence naming the role and company, the profile's one-line opening, the
    routed domain's 2-3 lead proofs that pass the number and stale-claim
    checks (a number-free proof when none does), and a plain close."""
    title = _plain_title(job)
    role = f"the {title} role" if title else "this role"
    at = f" at {company_display}" if company_display else ""
    return "\n".join([
        CL_SALUTATION,
        f"I am writing to apply for {role}{at}. {_template_opening()}",
        " ".join(_template_proofs(domain)),
        _TEMPLATE_CLOSE,
    ])


# --- Diagnostics (audit only; nothing here changes the letter) -----------------
_STYLE_PATTERNS = (
    ("bridge", re.compile(r"\b(?:that|this)(?:'s|\s+is|\s+was)\s+(?:exactly\s+)?(?:the\s+)?kind\s+of\b", re.I)),
    ("would_love", re.compile(r"\bI(?:\s+would|'d)\s+(?:love|welcome|be\s+glad)\b", re.I)),
    ("not_x_but_y", re.compile(r"\bnot\s+(?:just\s+|only\s+|merely\s+)?[^.,;]{1,40},?\s+but\b", re.I)),
    ("emdash", re.compile("—")),
)


def _style_hits(text: str) -> list:
    return [label for label, rx in _STYLE_PATTERNS if rx.search(text or "")]


# What build_cover_letter enforces on numbers, told to the model up front so it
# does not write sentences the check will delete.
_CL_NUMBERS_RULE = (
    "NUMBERS ARE CHECKED IN CODE: every number about the candidate (a count, team size, percentage, "
    "budget, multiplier) must appear in the CAREER SOURCE BANK or the verified metrics; a "
    "sentence about the candidate with a number found only in the interview transcript or the posting "
    "is deleted. The candidate's years of experience are always the canonical years figure in the "
    "verified metrics, never another number of years.")


def build_cover_letter(
    profile_text: str,
    job: dict,
    model: str = DEFAULT_GENERATION_MODEL,
    domain: str = "producer_core",
    transcript_text: str = "",
    never_claim: list = None,
    base_cv_path: str = None,
    company_display: str = None,
    evaluation_date: str = None,
    sanitized=None,
    thin: bool = False,
) -> tuple:
    """Generate, check and finish one cover letter. Returns (text, audit).

    text: salutation + body paragraphs, WITHOUT a signature (create_formatted_cl
    always signs), or None when generation failed (no letter is written).

    thin: the caller (save_documents_for_job) measured the posting under
    CV_THIN_POSTING_CHARS. There is nothing to tailor a letter to, so the
    templated letter ships at once: no model call, no judge call, state
    input_incomplete with the reason in the audit.

    1. generate_cover_letter_content (display company, sanitized posting).
    2. Strip the model's sign-off / salutation; split into sentences.
    3. Drop sentences with a number found in none of the candidate's VERIFIED sources
       (METRICS_BANK_TEXT, the source bank, the routed base CV — never the
       interview transcript, which is spoken recollection); a
       sentence with no first-person claim may also restate a posting or
       company-context number. Also drop a known-stale claim or a posting
       trap token.
    4. Cap at CL_MAX_BODY_WORDS (_cap_letter).
    5. grade_cover_letter_truth on the FINAL sentences: unsupported and
       composition sentences are dropped.
    6. Check unavailable (including a failed verdict, overall_pass false,
       that flags no sentence), or fewer than CL_MIN_BODY_WORDS left -> the
       templated letter (templated_letter).

    audit (stored as cv_quality_result["audit"]["cl"]): state (ok /
    check_unavailable / invalid / input_incomplete / generation_failed), reason,
    words, model_words, dropped_sentences [{stage, detail, quote}], template_used,
    company_display, signed, posting_chars, never_claim, style_hits, judge."""
    company_display = display_company(job) if company_display is None else company_display
    if sanitized is None:
        desc = job.get("description")
        sanitized = sanitize_jd(desc if isinstance(desc, str) else "")
    jd_text = (sanitized.text or "").strip()[:TRUNCATION_JOB_DESC_CV]
    company_context = _company_context(job)
    audit = {"state": "generation_failed", "reason": "", "words": 0, "model_words": 0,
             "dropped_sentences": [], "template_used": False, "company_display": company_display,
             "signed": False, "posting_chars": len(jd_text),
             "never_claim": _never_claim_terms(never_claim)}

    def ship_template(state: str, reason: str, level: int = logging.WARNING) -> tuple:
        text = templated_letter(job, domain, company_display)
        audit.update(state=state, reason=reason, template_used=True,
                     words=_letter_words(_letter_paragraphs(text)))
        logging.log(level, f"Cover letter: templated letter ships ({state}: {reason})")
        return text, audit

    if thin:
        # A letter written from the title alone is a paid call of over a
        # minute with no posting to work from. This is the intended path for
        # a posting with no text, so it logs at INFO, not as a warning.
        return ship_template("input_incomplete",
                             f"thin posting ({len(jd_text)} chars after cleaning) — "
                             f"standard letter, no model call", level=logging.INFO)

    raw = generate_cover_letter_content(
        profile_text, job, model=model, domain=domain, transcript_text=transcript_text,
        never_claim=never_claim, company_display=company_display, company_context=company_context)
    if not raw or not str(raw).strip():
        audit["reason"] = "cover letter generation returned no text (timeout / refusal / empty)"
        logging.error(f"CL generation failed: {audit['reason']}")
        return None, audit

    paras = _letter_paragraphs(str(raw))
    audit["model_words"] = _letter_words(paras)
    audit["style_hits"] = _style_hits(str(raw))
    dropped = audit["dropped_sentences"]

    transcript_ctx = _letter_transcript(domain, transcript_text)     # the judge's story context
    base_grounding = _base_grounding(base_cv_path)
    corpus = cv_facts.verified_corpus([base_grounding])
    dropped.extend(_drop_unverified_sentences(paras, corpus, sanitized,
                                              context=[jd_text, company_context]))
    dropped.extend(_cap_letter(paras, CL_MAX_BODY_WORDS))

    words = _letter_words(paras)
    if words > CL_MAX_BODY_WORDS:
        return ship_template("invalid", f"{words} words and no droppable sentence left under the "
                                        f"{CL_MAX_BODY_WORDS}-word cap")
    if words < CL_MIN_BODY_WORDS:
        return ship_template("invalid", f"only {words} words left after the deterministic checks "
                                        f"(minimum {CL_MIN_BODY_WORDS})")

    sentences = [s for p in paras for s in p]
    try:
        check = grade_cover_letter_truth(
            sentences, profile_text, job, company=company_display, base_cv=base_grounding,
            transcript_context=transcript_ctx, company_context=company_context,
            evaluation_date=evaluation_date)
    except Exception as e:
        check = {"state": "check_unavailable", "flagged": [], "audit": {"error": f"judge raised: {e}"}}
    audit["judge"] = check.get("audit") or {}
    if check.get("state") != "ok":
        return ship_template("check_unavailable",
                             f"letter truth check unavailable ({audit['judge'].get('error', 'no verdict')})")

    flagged = set(check.get("flagged") or [])
    if audit["judge"].get("overall_pass") is False and not flagged:
        # The judge failed the letter but located nothing to drop: no sentence
        # can be proven clean, so this is no complete check.
        return ship_template("check_unavailable",
                             "letter judge failed the letter (overall_pass false) without flagging a sentence")
    verdicts = check.get("verdicts") or {}
    n = 0
    for p in paras:
        for s in list(p):
            n += 1
            if n in flagged:
                p.remove(s)
                v = verdicts.get(n) or {}
                dropped.append({"stage": "judge", "detail": f"{v.get('verdict', 'flagged')}: "
                                                            f"{str(v.get('reason', ''))[:160]}",
                                "quote": _quote(s)})
    paras[:] = [p for p in paras if p]
    words = _letter_words(paras)
    if words < CL_MIN_BODY_WORDS:
        return ship_template("invalid", f"only {words} words left after the truth check "
                                        f"(minimum {CL_MIN_BODY_WORDS})")
    audit.update(state="ok", words=words,
                 reason=f"{len(flagged)} sentence(s) dropped by the truth check" if flagged else "")
    return _render_letter(paras), audit


def generate_cover_letter_content(
    base_text: str,
    job: dict,
    model: str = DEFAULT_GENERATION_MODEL,
    domain: str = "producer_core",
    transcript_text: str = "",
    never_claim: list = None,
    company_display: str = None,
    company_context: str = None,
) -> str:
    """The model's cover-letter body (plain text), or "" on failure. The
    caller (build_cover_letter) checks it and create_formatted_cl signs it.
    *company_display* defaults to display_company(job); *company_context* to
    the cached company profile."""
    domain_desc = DOMAIN_DESCRIPTIONS.get(domain, "general production")
    family = detect_model_family(model)
    company = display_company(job) if company_display is None else company_display
    if company_context is None:
        company_context = _company_context(job)

    # JD-noun-injection guard for the CL path: the CV keyword
    # auditor identifies posting terms the grounding does NOT support
    # ("missing_gap"), and the CV correctly leaves them out — but the CL had
    # no equivalent discipline and could reframe unrelated work as experience
    # with a posting's own nouns. Gap terms are hard-banned here
    # (Phase 27: genuine skill/experience gaps only, see _never_claim_terms).
    never_claim = _never_claim_terms(never_claim)
    never_claim_block = ""
    if never_claim:
        terms = "; ".join(never_claim)
        never_claim_block = (
            "NEVER-CLAIM LIST: a grounding audit confirmed the candidate has NO supported "
            "experience with these posting requirements — do NOT claim, imply, or reframe "
            f"experience with any of them: {terms}. Acknowledging a gap or bridging via a "
            "genuinely transferable skill is fine; wording that implies direct experience is fraud."
        )

    # Resolve transcript context (auto-load if not provided)
    transcript_ctx = _letter_transcript(domain, transcript_text)
    # Prepend the AUTHORITATIVE source bank: facts come from the candidate's
    # finalized documents; the transcript is kept for stories and voice.
    from .cv_tailor import _load_source_bank
    transcript_ctx = ("=== CAREER SOURCE BANK (AUTHORITATIVE — on any conflict, this wins) ===\n"
                      + _load_source_bank()
                      + "\n\n=== INTERVIEW TRANSCRIPT (stories/voice) ===\n" + transcript_ctx)

    # The posting is untrusted DATA (Phase 27): AI-directed and applicant-
    # instruction sentences stripped (clean_jd), then the full CV window — the
    # old 4,000-character cut hid the requirements of long postings.
    _jd = job.get("description")
    jd_text = clean_jd(_jd if isinstance(_jd, str) else "").strip()[:TRUNCATION_JOB_DESC_CV]
    title = _plain_title(job) or "(untitled role)"

    if family == "claude":
        prompt = f"""Write the body of a cover letter for the role below.

<target_role>
Title: {title}
Company: {company or 'the hiring company'}
</target_role>

<untrusted_data_notice>{UNTRUSTED_JD_CLAUSE}</untrusted_data_notice>
<job_description>
{jd_text or 'No description available — rely on the title and domain focus.'}
</job_description>

<company_context>
{company_context or 'No company context available.'}
</company_context>

<domain_focus>{domain_desc}</domain_focus>

<candidate_background>
{base_text[:TRUNCATION_PROFILE_SUMMARY]}
</candidate_background>

<transcript_context>
{transcript_ctx}
</transcript_context>

<instructions>
- Write ONLY the body. No header (name, date, address), no sign-off, no signature: the header, "Sincerely," and the candidate's name are added by code.
- Start with "Dear Hiring Manager,".
- {CL_TARGET_WORDS} words in 3-4 short paragraphs. Hard limit: anything past {CL_MAX_BODY_WORDS} words is cut.
- The first two sentences say who the candidate is for THIS role and give the candidate's single most relevant outcome for it.
- Identify the posting's top requirement. Open with the story or proof point that demonstrates THAT requirement, chosen from the transcript's "Cover Letter Stories" or the verified metrics. Do not default to a favourite anecdote: a story belongs in this letter only if it proves this posting's top requirement.
- Then address 1-2 further SPECIFIC requirements from the job_description with matching proof points, framed through the domain lens ({domain_desc}). The letter must read as written for THIS posting, not any producer job.
- Use company_context (when available) to ground one "why this company" sentence in something real about them. Never invent facts about the company.
- Register: plain and professional. Direct, specific sentences; no hype and no fluff. Every sentence earns its place.
- Mention a gap only if the posting requires it AND adjacent proof exists in the candidate's sources; otherwise do not mention gaps.
- {_CL_NUMBERS_RULE}
- BANNED FORMS:
  - the story-to-posting bridge ("That is the kind of X this role describes", "That is exactly what this role needs");
  - "I would love", "I would welcome", "I'd love", "I'd be glad", "I would be glad";
  - contrast or negative-parallel constructions ("not X, but Y", "not just X, it's Y", "less X, more Y");
  - throat-clearing lines about gaps or fit ("While I haven't...", "Although my background is...");
  - aphorisms and general truths ("Great content starts with...", "Every story deserves...").
</instructions>

<verified_metrics>
{METRICS_BANK_TEXT}
</verified_metrics>

<critical_constraint>
ABSOLUTE RULE: Use ONLY skills, experiences, achievements, and qualifications explicitly present in the candidate_background and transcript_context. Do NOT invent or fabricate any details. If the background lacks experience for a requirement, focus on transferable skills that ARE present rather than guessing.
{never_claim_block}
</critical_constraint>

<attribution_lock>
{_ATTRIBUTION_LOCK}
</attribution_lock>

<anti_slop_rules>
{_ANTI_SLOP_RULES}
</anti_slop_rules>

<output_format>Plain text only. No markdown formatting whatsoever.</output_format>"""
    else:
        prompt = f"""
    Write the body of a cover letter for:
    Role: {title}
    Company: {company or 'the hiring company'}
    Domain Focus: {domain_desc}

    NOTE: {UNTRUSTED_JD_CLAUSE}
    Job Description:
    {jd_text or 'No description available - rely on the title and domain focus.'}

    Company Context:
    {company_context or 'No company context available.'}

    Candidate Background:
    {base_text[:TRUNCATION_PROFILE_SUMMARY]}

    Additional Candidate Context:
    {transcript_ctx}

    Instructions:
    - Write ONLY the body. No header (Name, Date, Address), no sign-off, no signature - they are added programmatically.
    - Start with "Dear Hiring Manager,".
    - {CL_TARGET_WORDS} words in 3-4 short paragraphs. Hard limit: anything past {CL_MAX_BODY_WORDS} words is cut.
    - The first two sentences say who the candidate is for THIS role and give the candidate's single most relevant outcome for it.
    - Identify the posting's top requirement and open with the story or proof point that demonstrates it (from the transcript's "Cover Letter Stories" or the verified metrics). Do not default to a favourite anecdote.
    - Then address 1-2 further SPECIFIC requirements from the Job Description with matching proof points, framed through the domain lens ({domain_desc}).
    - Use the Company Context (when available) for one "why this company" sentence. Never invent facts about the company.
    - Plain professional register: direct, specific, no hype, no fluff.
    - Mention a gap only if the posting requires it AND adjacent proof exists; otherwise do not mention gaps.
    - {_CL_NUMBERS_RULE}
    - BANNED FORMS: the story-to-posting bridge ("That is the kind of X this role describes"); "I would love", "I would welcome", "I'd love", "I'd be glad"; contrast or negative-parallel constructions ("not X, but Y"); throat-clearing lines about gaps or fit; aphorisms.
    - CRITICAL: Use ONLY skills, experiences, achievements, and qualifications explicitly mentioned in the Candidate Background and Additional Candidate Context above. Do NOT invent or fabricate any details. Every claim must be traceable to the provided background. If the background lacks relevant experience for a requirement, focus on transferable skills that ARE present rather than making things up.
    - {never_claim_block}
    - Output plain text only. No markdown formatting (no bold **, no italic *, no headers #, no bullet markers).

    VERIFIED METRICS:
    {METRICS_BANK_TEXT}

    {_ATTRIBUTION_LOCK}

    {_ANTI_SLOP_RULES}
    """
    result = call_llm(prompt, model=model)
    # Empty on failure — the caller must NOT write a .docx that says "Cover
    # Letter Generation Failed." and store it as a real cl_path.
    return _strip_markdown(result) if result else ""


def _add_hyperlink(paragraph, text: str, size_pt: float = 10, font: str = "Arial") -> None:
    """Append *text* to *paragraph* as a real (clickable) hyperlink, styled like
    the header text around it."""
    url = text if re.match(r"https?://", text, re.I) else "https://" + text
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), font)
    fonts.set(qn("w:hAnsi"), font)
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), str(int(size_pt * 2)))
    rpr.append(fonts)
    rpr.append(size)
    run.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


def create_formatted_cl(body_text: str, job: dict, company_display: str = None,
                        links: list = None) -> Document:
    """
    Creates a Cover Letter with the candidate's header and standard business formatting.

    Phase 27: the company is display_company(job) (unless *company_display* is
    given); *links* (the candidate's LinkedIn / portfolio, from the base CV's contact line)
    go on the contact line as real hyperlinks; any sign-off in *body_text* is
    stripped and the letter is ALWAYS signed "Sincerely," + the candidate's name, once.
    """
    doc = Document()
    company = display_company(job) if company_display is None else company_display
    candidate_name = _candidate_name()

    # --- HEADER ---
    # Name
    name_para = doc.add_paragraph()
    name_run = name_para.add_run(candidate_name)
    name_run.bold = True
    name_run.font.size = Pt(16)
    name_run.font.name = 'Arial'
    name_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Contact Info (+ LinkedIn / portfolio on a second line, as real links)
    contact_para = doc.add_paragraph()
    location = os.getenv("CANDIDATE_LOCATION", "City, State")
    phone = os.getenv("CANDIDATE_PHONE", "555-555-5555")
    email = os.getenv("CANDIDATE_EMAIL", "email@example.com")
    contact_run = contact_para.add_run(f"{location} • {phone} • {email}")
    contact_run.font.size = Pt(10)
    contact_run.font.name = 'Arial'
    for i, link in enumerate(links or []):
        sep = contact_para.add_run(" • " if i else "")
        if not i:
            sep.add_break()
        sep.font.size = Pt(10)
        sep.font.name = 'Arial'
        _add_hyperlink(contact_para, link)
    contact_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph() # Spacer

    # --- DATE ---
    date_para = doc.add_paragraph(datetime.now().strftime("%B %d, %Y"))
    date_para.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_paragraph() # Spacer

    # --- RECIPIENT ---
    doc.add_paragraph("Hiring Manager")
    if company:
        doc.add_paragraph(company)
    # We could add address if we knew it, but blank is safer than wrong

    doc.add_paragraph() # Spacer

    # --- SUBJECT ---
    title = _plain_title(job) or "Application"
    subject_text = f"RE: {title}"
    if company and company.lower() not in title.lower():
        subject_text += f", {company}"
    subject = doc.add_paragraph()
    subject_run = subject.add_run(subject_text)
    subject_run.bold = True

    doc.add_paragraph() # Spacer

    # --- BODY ---
    # One paragraph per line; the salutation is code's own.
    paras = [p.strip() for p in _strip_sign_off(body_text, candidate_name).split('\n') if p.strip()]
    paras = [CL_SALUTATION] + _strip_salutation(paras)
    for para in paras:
        p = doc.add_paragraph(para)
        p.paragraph_format.space_after = Pt(12)

    # --- SIGNATURE --- always, exactly once (the model's sign-off was stripped)
    doc.add_paragraph(CL_SIGN_OFF)
    doc.add_paragraph(candidate_name)

    return doc


def upload_to_google_drive(file_paths):
    """RETIRED — always returns {} (no Drive calls).

    CV/CL Drive copies were unused: the app serves every document and local
    files are the source of truth, so nothing is uploaded. Kept as a stub
    because tests patch this symbol.
    """
    logging.info("CV/CL Drive upload retired — skipping.")
    return {}
