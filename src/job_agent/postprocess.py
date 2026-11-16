
import os
import logging
import re
import smtplib
import time
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
# gspread import removed (Sheets sync retired — see upload_to_google_sheets stub)
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

load_dotenv()

from .utils import get_google_credentials, is_posting_old, normalize_date_posted


def upload_to_google_sheets(df: pd.DataFrame):
    """RETIRED — always returns None.

    Sheets sync was removed: the web app is the interface and the digest
    email already attaches the full CSV. Kept as a stub so legacy call
    sites/tests stay harmless.
    """
    logging.info("Sheets sync retired — skipping.")
    return None


import pytz


def _smtp_send(sender: str, password: str, recipients: list, msg, attempts: int = 3):
    """Send via Gmail SMTP with a connect timeout and retry/backoff.

    Both send paths were one-shot with no timeout — a single transient
    Errno-60 connect timeout could cost the whole digest.
    Raises the last exception if every attempt fails.
    """
    for i in range(attempts):
        try:
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30)
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
            server.quit()
            return
        except Exception as e:
            if i >= attempts - 1:
                raise
            wait = 30 * (i + 1)
            logging.warning(
                f"SMTP send failed (attempt {i + 1}/{attempts}): {e} — retrying in {wait}s")
            time.sleep(wait)


def send_error_alert(subject: str, error_details: str, pipeline_stage: str = "Unknown"):
    """Send an error notification email to the admin.

    Uses the same SMTP config as the regular digest. Fails silently
    (logs a warning) so it never crashes the pipeline.

    Args:
        subject: Short description of the error (e.g. "Cascade filter failed").
        error_details: Full error text / traceback snippet.
        pipeline_stage: Which pipeline phase failed (e.g. "AI Filtering", "CV Generation").
    """
    from .config import EMAIL_SENDER, EMAIL_PASSWORD, ERROR_ALERT_EMAIL

    sender = EMAIL_SENDER
    password = EMAIL_PASSWORD
    recipient = ERROR_ALERT_EMAIL

    if not password or not sender or not recipient:
        logging.warning("Error alert skipped — EMAIL_SENDER/EMAIL_PASSWORD/ERROR_ALERT_EMAIL not configured.")
        return

    try:
        pacific = pytz.timezone("US/Pacific")
        timestamp = datetime.now(pacific).strftime("%Y-%m-%d %H:%M:%S %Z")

        full_subject = f"[AI Job Agent] ERROR: {subject}"
        body = (
            f"Pipeline stage: {pipeline_stage}\n"
            f"Timestamp: {timestamp}\n"
            f"{'=' * 60}\n\n"
            f"{error_details}\n"
        )

        msg = MIMEText(body, "plain")
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = full_subject

        _smtp_send(sender, password, [recipient], msg)
        logging.info(f"Error alert sent to {recipient}: {full_subject}")
    except Exception as e:
        logging.warning(f"Failed to send error alert: {e}")


def _md_to_html(text: str) -> str:
    """Convert basic markdown formatting to inline HTML for email display.

    Handles **bold**, *italic*, and strips stray # headers.
    """
    import re
    if not text:
        return ""
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    # Italic: *text* or _text_ (but not inside HTML tags)
    text = re.sub(r'(?<![<\w])\*(.+?)\*(?![>])', r'<em>\1</em>', text)
    text = re.sub(r'(?<![<\w])_(.+?)_(?![>])', r'<em>\1</em>', text)
    # Headers: strip leading # markers
    text = re.sub(r'^#{1,3}\s+', '', text, flags=re.MULTILINE)
    return text


def _truncate_sentences(text: str, max_sentences: int = 2) -> str:
    """Return the first N sentences of text."""
    import re
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:max_sentences])


def _build_intel_preview(company: str, title: str, db) -> str:
    """Build compact HTML intel preview for a single job row in the email digest.

    Fetches company profile, salary intel, salary recommendation, and hiring intel
    from the company_intel cache. Returns empty string if nothing is cached.
    """
    if db is None:
        return ""

    parts = []

    # 1. Company profile (cached under company name)
    profile_row = db.get_company_intel(company)
    if profile_row and profile_row.get("company_profile"):
        summary = _truncate_sentences(profile_row["company_profile"], 2)
        if summary:
            parts.append(f'<strong>Company:</strong> {_md_to_html(summary)}')

    # 2. Salary intel (cached under __salary_intel__{company}__{title})
    salary_row = db.get_company_intel(f"__salary_intel__{company}__{title}")
    if salary_row and salary_row.get("company_profile"):
        salary_text = _truncate_sentences(salary_row["company_profile"], 1)
        if salary_text:
            parts.append(f'<strong>Salary:</strong> {_md_to_html(salary_text)}')

    # 3. Salary recommendation (cached under __salary_rec__{company}__{title})
    rec_row = db.get_company_intel(f"__salary_rec__{company}__{title}")
    if rec_row and rec_row.get("company_profile"):
        rec_text = _truncate_sentences(rec_row["company_profile"], 1)
        if rec_text:
            parts.append(f'<strong>Strategy:</strong> {_md_to_html(rec_text)}')

    # 4. Hiring intel — per-role key (matches perplexity_utils' writer);
    # legacy company-level fallback for pre-migration cache entries whose
    # role matches is impossible to verify, so only used when per-role misses.
    _safe = re.sub(r"[^A-Za-z0-9 &/+-]+", "", str(title or ""))[:60]
    hiring_row = db.get_company_intel(f"__hiring_intel__{company}__{_safe}"[:200])
    if not (hiring_row and hiring_row.get("company_profile")):
        hiring_row = db.get_company_intel(f"__hiring_intel__{company}")
    if hiring_row and hiring_row.get("company_profile"):
        intel_text = _truncate_sentences(hiring_row["company_profile"], 2)
        if intel_text:
            parts.append(f'<strong>Intel:</strong> {_md_to_html(intel_text)}')

    if not parts:
        return ""

    return (
        '<div style="background:#f8fafc;border-left:3px solid #6366f1;'
        'padding:6px 10px;margin:4px 0 8px 0;font-size:12px;line-height:1.4;color:#555;">'
        + '<br>'.join(parts)
        + '</div>'
    )


def _ascii_html(text: str) -> str:
    """Escape text for the digest HTML and keep it ASCII (non-ASCII becomes
    a numeric entity), so no mail client re-encodes a dash into mojibake."""
    from html import escape
    return escape(str(text), quote=True).encode("ascii", "xmlcharrefreplace").decode("ascii")


def _contract_note_html(etype, comp, company=None) -> str:
    """The in-place contract-pay line for a digest card ('' if not contract).
    The note says only what the row supports (pay basis, agency) — see
    comp_utils.contract_pay_note."""
    from .comp_utils import contract_pay_note
    note = contract_pay_note(etype, comp, company)
    if not note:
        return ""
    return (f'<div style="color:#92400e;font-size:12px;margin-top:3px;line-height:1.4;">'
            f'{_ascii_html(note)}</div>')


# Follow-up steps the digest lists, by cadence position (config
# OUTREACH_FOLLOWUP_DAYS = [2, 7, 14, 21]). The last anchor, day 21, is "close
# the loop — no more messages" in the playbook itself, so it has no line here
# (zip stops at the labels). Each step is listed ONCE: only in the morning
# digest, at most FOLLOWUPS_IN_DIGEST of them, below the Strong matches.
_FOLLOWUP_LABELS = (
    "LinkedIn touch: connect with the hiring manager/recruiter",
    "Email follow-up: one proof point, ask about timeline",
    "Final nudge, then move on",
)
FOLLOWUPS_IN_DIGEST = 5


def _is_morning_digest(now_pt: datetime) -> bool:
    """The morning run's digest (any run before noon Pacific)."""
    return now_pt.hour < 12


def _followups_html(db) -> str:
    """'Follow-ups due today' block for the digest ('' when nothing is due)."""
    from .config import OUTREACH_FOLLOWUP_DAYS
    steps = dict(zip(OUTREACH_FOLLOWUP_DAYS, _FOLLOWUP_LABELS))
    due = [f for f in db.get_followups_due(list(steps)) if f.get("days_since") in steps]
    if not due:
        return ""
    shown = due[:FOLLOWUPS_IN_DIGEST]
    items = []
    for f in shown:
        link = f.get("link")
        posting = (f' &middot; <a href="{_ascii_html(link)}" style="color:#2563eb;font-size:13px;">'
                   f'posting</a>' if link else "")
        items.append(
            f'<div style="padding:6px 0;border-bottom:1px solid #f3f4f6;">'
            f'<b>{_ascii_html(str(f.get("title") or "")[:60])}</b> @ '
            f'{_ascii_html(str(f.get("company") or "")[:40])}'
            f'<br><span style="color:#92400e;font-size:13px;">Day {int(f["days_since"])}: '
            f'{_ascii_html(steps[f["days_since"]])}</span>{posting}</div>')
    more = len(due) - len(shown)
    more_html = (f'<div style="color:#666;font-size:13px;padding-top:6px;">&hellip;and {more} more '
                 f'on the Kanban board.</div>' if more > 0 else "")
    return f"""
    <div style="background:#fffbeb;border-left:4px solid #f59e0b;padding:12px 16px;margin:16px 0;border-radius:4px;">
        <b>&#9200; Follow-ups due today</b> &mdash; open the Kanban card for details (cards applied via the app carry an outreach playbook).
        {"".join(items)}{more_html}
    </div>"""


def _row_link(job) -> str:
    """The row's posting link (job_url, else link), "" when missing/NaN."""
    for key in ("job_url", "link"):
        value = job.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _drop_gone_rows(df: pd.DataFrame, db) -> pd.DataFrame:
    """Postings no longer on their company's job board (checked
    twice — db.get_gone_links) leave the digest. Nothing is deleted; the app
    keeps the row. Any failure keeps the frame as it was."""
    if db is None or df is None or df.empty:
        return df
    try:
        links = [_row_link(r) for _, r in df.iterrows()]
        gone = db.get_gone_links([l for l in links if l]) or set()
        if not gone:
            return df
        keep = [link not in gone for link in links]
        logging.info(f"Digest: leaving out {len(keep) - sum(keep)} posting(s) no longer on the company job board")
        return df[keep].copy()
    except Exception as e:
        logging.warning(f"Digest: gone-posting check skipped: {e}")
        return df


def _drop_stale_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows fresh enough for the email (utils.is_fresh_for_review: Strong up
    to 7 days, Worth a look up to 3 — by posting date, else first seen; a new
    row this run has no created_at yet, so it counts from its posting date)."""
    if df is None or df.empty or "tier" not in df.columns:
        return df
    try:
        from .utils import is_fresh_for_review
        keep = df.apply(lambda r: is_fresh_for_review(
            r.get("tier"), r.get("date_posted"), r.get("created_at")), axis=1)
        return df[keep].copy()
    except Exception as e:
        logging.warning(f"Digest: freshness filter skipped: {e}")
        return df


_GLANCE_READ_STYLE = {
    "looks_active": ("#15803d", "&#10003; "),
    "mixed": ("#b45309", "&#9888; "),
    "check_first": ("#b91c1c", "&#9888; "),
}


def _posting_glance_html(job, db) -> str:
    """One muted "at a glance" line under every card: the
    posting check's read plus its most telling facts — company news, still on
    the company board, re-listing, wide pay, posting red flags, the candidate's earlier
    card at the company, people they know there (posting_check.digest_glance).
    "" when there is nothing to say or on any failure — the digest never breaks
    over it."""
    if db is None:
        return ""
    try:
        link = _row_link(job)
        job_id = db.get_job_id_by_link(link) if link else None
        if not job_id:
            return ""
        from . import posting_check
        glance = posting_check.digest_glance(db, job_id)
    except Exception as e:
        logging.warning(f"Digest: posting check line skipped: {e}")
        return ""
    if not glance:
        return ""
    from html import escape
    parts = []
    style = _GLANCE_READ_STYLE.get(glance.get("read"))
    if style:
        color, mark = style
        text = escape(str(glance.get("label") or ""))
        reasons = [str(r) for r in (glance.get("reasons") or []) if r]
        if glance.get("read") != "looks_active" and reasons:
            text += ": " + escape("; ".join(reasons))
        parts.append(f'<span style="color:{color};font-weight:600;">{mark}{text}</span>')
    for item in glance.get("items") or []:
        text = escape(str(item.get("text") or ""))
        if not text:
            continue
        if item.get("caution"):
            text = f'<span style="color:#b45309;">{text}</span>'
        url = str(item.get("url") or "")
        if url.startswith("http"):
            text += (f' <a href="{escape(url, quote=True)}" '
                     f'style="color:#9ca3af;font-size:11px;">source</a>')
        parts.append(text)
    if not parts:
        return ""
    return ('<div style="color:#6b7280;font-size:12px;margin-top:4px;line-height:1.45;">'
            + " &middot; ".join(parts) + "</div>")


def send_email_with_results(csv_path: str, recipient: str, df: pd.DataFrame, profile_text: str = "", spreadsheet_url: str = "", total_scraped: int = 0, cascade_stats: dict = None, db=None,
                            *, subject: str = None, intro_html: str = "", include_followups: bool = None,
                            recovered_df: pd.DataFrame = None, recovered_note: str = ""):
    """The digest email. The keyword-only options serve a re-sent or combined
    digest (scripts/send_digest_for_session.py): `subject` replaces the
    generated subject, `intro_html` replaces the "search just finished" line,
    `include_followups` forces the follow-ups section on or off (default: the
    morning digest only), `recovered_df` adds a "previously dropped, now
    recovered" section after Worth a look (rows from earlier days,
    re-graded after a rule change; `recovered_note` is its one-line explanation).
    A normal pipeline run passes none of them."""
    logging.info(f"Preparing email digest for {recipient}...")
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")

    if not password:
        logging.warning("No email password found. Skipping email.")
        return

    # Gone postings leave before anything is counted, and so do
    # roles past the freshness limit (Strong 7 days, Worth a look 3).
    df = _drop_stale_rows(_drop_gone_rows(df, db))

    # --- Build match stats (tier-based) ---
    # Count by TIER, not by match_level pairs: the recall pass added
    # WORTH_A_LOOK as a final level — counting only HIGH+POTENTIAL made a
    # worth-a-look-only run send the "Nothing new today" email while the jobs
    # sat invisible in the app.
    if "tier" in df.columns:
        num_high = int((df["tier"] == "strong").sum())
        num_potential = int((df["tier"] == "worth_a_look").sum())
    elif "match_level" in df.columns:
        num_high = int(df["match_level"].isin(["HIGH_MATCH", "USER_INPUT"]).sum())
        num_potential = int(df["match_level"].isin(["POTENTIAL_MATCH", "WORTH_A_LOOK"]).sum())
    else:
        num_high = num_potential = 0
    num_relevant = num_high + num_potential

    # Use cascade_stats for accurate funnel numbers, fall back to df length
    if cascade_stats:
        total_scanned = total_scraped or cascade_stats.get("stage1_total", len(df))
        s1_total = cascade_stats.get("stage1_total", 0)
        s1_filtered = cascade_stats.get("stage1_filtered", 0)
        s2_total = cascade_stats.get("stage2_total", 0)
        s3_total = cascade_stats.get("stage3_total", 0)
        cascade_summary_html = f"""
        <p style="color: #666; font-size: 13px;">
            <b>How I filtered:</b> {total_scanned:,} jobs scraped &rarr;
            {s1_total} passed to AI pre-screen &rarr;
            {s1_filtered} eliminated &rarr;
            {s2_total} graded in detail &rarr;
            <b>{num_relevant} worth your time</b>
        </p>
        """
    elif total_scraped:
        total_scanned = total_scraped
        cascade_summary_html = f"""
        <p style="color: #666; font-size: 13px;">
            <b>How I filtered:</b> {total_scanned:,} jobs scraped &rarr;
            <b>{num_relevant} worth your time</b>
        </p>
        """
    else:
        total_scanned = len(df)
        cascade_summary_html = ""

    # --- Tiered job sections (mirrors the app: Strong / Worth a look) ---
    from .config import location_priority

    def _split_tiers(frame):
        if "tier" in frame.columns:
            return (frame[frame["tier"] == "strong"].copy(),
                    frame[frame["tier"] == "worth_a_look"].copy())
        if "match_level" in frame.columns:
            return (frame[frame["match_level"].isin(["HIGH_MATCH", "USER_INPUT"])].copy(),
                    frame[frame["match_level"].isin(["POTENTIAL_MATCH", "WORTH_A_LOOK"])].copy())
        return frame.iloc[0:0].copy(), frame.iloc[0:0].copy()

    def _sort_for_email(frame):
        """Location priority (config.location_priority), then confidence."""
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["_loc_rank"] = frame.apply(
            lambda r: location_priority(str(r.get("location", "")), str(r.get("title", ""))), axis=1)
        sort_cols = ["_loc_rank"] + (["confidence"] if "confidence" in frame.columns else [])
        ascending = [True] + ([False] if "confidence" in frame.columns else [])
        return frame.sort_values(sort_cols, ascending=ascending).drop(columns=["_loc_rank"])

    def _age_label(job):
        raw = job.get("date_posted", "")
        d = normalize_date_posted(raw) if raw else ""
        if not d:
            return ""
        try:
            days = (datetime.now() - datetime.strptime(d[:10], "%Y-%m-%d")).days
        except Exception:
            return ""
        if days < 0:
            return ""
        if days <= 1:
            # Fresh postings get the strongest nudge: early applications
            # get the most attention, so the badge says to act today.
            txt, bg = ("new today" if days == 0 else "1d") + " · apply today", "#15803d"
        elif days <= 13:
            txt, bg = f"{days}d", "#22c55e"
        elif days <= 29:
            txt, bg = f"{days}d", "#f59e0b"
        else:
            txt, bg = f"{days // 7}w", "#f97316"
        return (f' <span style="background:{bg};color:#fff;padding:1px 7px;'
                f'border-radius:10px;font-size:11px;">{txt}</span>')

    _PILL = {
        "strong": '<span style="background:#22c55e;color:#fff;padding:2px 9px;border-radius:10px;font-size:12px;">Strong</span>',
        "worth": '<span style="background:#8b5cf6;color:#fff;padding:2px 9px;border-radius:10px;font-size:12px;">Worth a look</span>',
    }

    def _clean_reason(job):
        txt = str(job.get("gpt_reasoning") or job.get("reason") or job.get("reasoning") or "")
        txt = txt.split("[Capped-metro on-site")[0].strip()       # policy note is app detail
        if len(txt) > 150:
            # Truncate at a WORD boundary and drop any dangling open-paren
            # fragment — a hard cut at 150 once ended inside a parenthesis on
            # a "($10…" right under the comp line, reading like a broken price.
            cut = txt[:150]
            sp = cut.rfind(" ")
            if sp > 100:
                cut = cut[:sp]
            cut = re.sub(r"\s*\([^)]*$", "", cut).rstrip(" ,;:·-")
            txt = cut + "…"
        return txt

    from .config import extract_actionable_location
    from html import escape as _esc

    def _card(job, tier_key, with_reason):
        """One phone-friendly stacked card (single column, no wide tables)."""
        # Scraped text (title/company/location) is interpolated into HTML —
        # escape it so odd characters (e.g. "English <> German" titles) can't
        # corrupt the email or inject markup.
        link = _esc(str(job.get("job_url") or job.get("link", "#")), quote=True)
        title = _esc(str(job.get("title", "N/A"))[:90])
        company = _esc(str(job.get("company") or "—")[:50])
        location = _esc(str(job.get("location") or "")[:55])
        # For a multi-location post, surface the option that makes it a match
        # (the raw location above is truncated and may hide it).
        _opt = extract_actionable_location(job.get("location") or "")
        loc_badge = (
            f'<div style="color:#0a7d2e;font-size:12px;font-weight:600;margin-top:2px;">'
            f'&#10003; Matching location: {_esc(str(_opt)[:45])}</div>' if _opt else "")
        reason_html = ""
        if with_reason:
            r = _clean_reason(job)
            if r:
                reason_html = f'<div style="color:#555;font-size:13px;margin-top:5px;line-height:1.4;">{r}</div>'
        # Per-job comp + employment type at a glance (2026-07-26). The
        # pipeline's in-memory df carries raw jobspy fields (min_amount/
        # job_type) but not the DB-computed columns — extract inline when
        # missing. Falls back gracefully: no "N/A" noise on cards without data.
        comp = str(job.get("comp_text") or "").strip()
        etype = str(job.get("employment_type") or "").strip()
        if (not comp or comp.lower() == "nan") and (not etype or etype.lower() == "nan"):
            try:
                from .comp_utils import extract_comp
                _c = extract_comp(dict(job))
                comp = _c["comp_text"] or ""
                etype = _c["employment_type"] or ""
            except Exception:
                pass
        comp_bits = []
        if comp and comp.lower() != "nan":
            comp_bits.append(
                f'<span style="color:#0a7d2e;font-weight:600;">&#128176; {_esc(comp)}</span>')
        else:
            # Explicit N/A (2026-07-26): the comp slot always renders on
            # tiered cards — muted grey so the eye can scan past it.
            comp_bits.append(
                '<span style="color:#9ca3af;">&#128176; N/A</span>')
        if etype and etype.lower() != "nan":
            comp_bits.append(
                f'<span style="background:#eef2ff;color:#4338ca;padding:1px 8px;'
                f'border-radius:10px;font-size:12px;">{_esc(etype)}</span>')
        # Employer's internal level, when the posting publishes one (Apple
        # ICT/IC levels, 2026-09-26) — a direct read on seniority.
        level = job.get("job_level")
        if isinstance(level, str) and level.strip() and level.strip().lower() != "nan":
            comp_bits.append(
                f'<span style="background:#f1f5f9;color:#334155;padding:1px 8px;'
                f'border-radius:10px;font-size:12px;">Level: {_esc(level.strip())}</span>')
        comp_html = f'<div style="font-size:13px;margin-top:3px;">{" &nbsp;".join(comp_bits)}</div>'
        # Contract / hourly roles are never penalized, but the card says the
        # pay works differently (2026-10-03).
        comp_html += _contract_note_html(etype, comp, job.get("company"))
        # "Is this posting real?" at a glance: every card gets
        # one muted line with the posting check's read and key facts.
        comp_html += _posting_glance_html(job, db)
        return f"""
        <div style="border:1px solid #e5e7eb;border-radius:10px;padding:11px 13px;margin:8px 0;">
            <div style="line-height:1.5;">
                <a href="{link}" style="font-size:15px;font-weight:600;color:#111;text-decoration:none;">{title}</a>
                {_PILL[tier_key]}{_age_label(job)}
            </div>
            <div style="color:#666;font-size:13px;margin-top:3px;">
                {company}{" &middot; " + location if location else ""}
                &middot; <a href="{link}" style="color:#2563eb;font-size:13px;">Apply &rarr;</a>
            </div>
            {comp_html}
            {loc_badge}
            {reason_html}
        </div>"""

    strong_df, worth_df = _split_tiers(df)
    strong_df = _sort_for_email(strong_df).head(20)
    worth_df = _sort_for_email(worth_df)
    WORTH_IN_EMAIL = 8
    worth_shown = worth_df.head(WORTH_IN_EMAIL)
    worth_more = max(0, num_potential - len(worth_shown))

    strong_html = "".join(_card(j, "strong", with_reason=True) for _, j in strong_df.iterrows()) \
        or '<p style="color:#999;font-size:14px;margin:6px 0;">None this run.</p>'
    worth_html = "".join(_card(j, "worth", with_reason=False) for _, j in worth_shown.iterrows()) \
        or '<p style="color:#999;font-size:14px;margin:6px 0;">None this run.</p>'
    worth_more_html = (
        f'<p style="color:#666;font-size:13px;">&hellip;and <b>{worth_more} more</b> in the app\'s '
        f'Review queue (sorted by location priority).</p>'
        if worth_more else ""
    )

    # --- Previously dropped, now recovered (a combined re-send only) ---
    recovered_html = ""
    if recovered_df is not None and not recovered_df.empty:
        # A job already in today's sections is never shown twice.
        today_links = set()
        for col in ("job_url", "link"):
            if col in df.columns:
                today_links |= {str(x) for x in df[col].dropna()}
        rec_link = recovered_df.get("job_url", recovered_df.get("link"))
        if rec_link is not None and today_links:
            recovered_df = recovered_df[~rec_link.astype(str).isin(today_links)]
        rec_strong, rec_worth = _split_tiers(_drop_stale_rows(_drop_gone_rows(recovered_df, db)))
        rec_cards = ("".join(_card(j, "strong", with_reason=True) for _, j in _sort_for_email(rec_strong).iterrows())
                     + "".join(_card(j, "worth", with_reason=False) for _, j in _sort_for_email(rec_worth).iterrows()))
        if rec_cards:
            note = (f'<p style="color:#666;font-size:13px;margin:2px 0 6px;">{recovered_note}</p>'
                    if recovered_note else "")
            recovered_html = (
                f'<h3 style="font-size:14px;letter-spacing:0.05em;color:#0369a1;margin:22px 0 4px;">'
                f'PREVIOUSLY DROPPED &mdash; NOW RECOVERED ({len(rec_strong)} strong + {len(rec_worth)} worth a look)</h3>'
                f'{note}{rec_cards}')

    # --- Follow-ups due today (outreach cadence: day 2 / 7 / 14 after applying) ---
    # Morning digest only, so each step is shown once; placed BELOW the Strong
    # matches (the day's new roles come first).
    followups_html = ""
    want_followups = (include_followups if include_followups is not None
                      else _is_morning_digest(datetime.now(pytz.timezone("US/Pacific"))))
    if db is not None and want_followups:
        try:
            followups_html = _followups_html(db)
        except Exception as e:
            logging.error(f"Follow-ups-due section failed: {e}")

    # --- Compose email ---
    pacific = pytz.timezone("US/Pacific")
    now_formatted = datetime.now(pacific).strftime("%A, %B %d at %I:%M %p")

    # Time-appropriate greeting, by the profile's first name (plain greeting
    # when the profile has none or cannot be read — the digest never breaks
    # over it).
    try:
        from .candidate_profile import CANDIDATE
        _first = str(CANDIDATE.get("first_name") or "").strip()
    except Exception:
        _first = ""
    _to = f", {_first}" if _first else ""
    hour = datetime.now(pacific).hour
    if 5 <= hour < 12:
        greeting = f"Good morning{_to}!"
        signoff = "Have a great day! I'll search again on the next scheduled run."
    elif 12 <= hour < 17:
        greeting = f"Good afternoon{_to}!"
        signoff = "Have a great afternoon! I'll search again tomorrow."
    elif 17 <= hour < 21:
        greeting = f"Good evening{_to}!"
        signoff = "Have a great evening! I'll search again tomorrow."
    else:
        greeting = f"Hello{_to}!"
        signoff = "Talk soon! I'll search again tomorrow."

    if num_relevant == 0:
        # --- No results email ---
        html_body = f"""
    <html><body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 640px; margin: 0 auto; color: #333; padding: 0 10px; word-break: break-word; overflow-wrap: anywhere;">

    <p>{greeting}</p>

    <p>Your job search just finished. Here's the news as of <b>{now_formatted} PT</b>:</p>

    <div style="background: #fef9c3; border-left: 4px solid #eab308; padding: 12px 16px; margin: 16px 0; border-radius: 4px;">
        <b>Nothing new today</b> &mdash; that's actually okay! It means the filters are working hard so you don't have to review noise.
        {cascade_summary_html}
    </div>
    {followups_html}
    {recovered_html}

    <p style="color: #666; font-size: 14px;">I looked through {total_scanned:,} job postings and none cleared the bar. I'll keep watching &mdash; good roles don't stay open long, and I'll flag them the moment they appear.</p>

    <p style="margin-top: 24px;">{signoff}</p>

    <p style="color: #999; font-size: 12px; margin-top: 32px; border-top: 1px solid #eee; padding-top: 12px;">
        Sent by your personal AI job search assistant.
        <br>Powered by a 3-stage AI cascade.
    </p>

    </body></html>
        """
        subject = subject or f"Your Job Digest \u2014 Nothing new today ({datetime.now(pacific).strftime('%b %d')})"
    else:
        # --- Results email: two sections, exactly like the app ---
        sheets_link = ""
        if spreadsheet_url:
            sheets_link = f'&nbsp;&bull;&nbsp;<a href="{spreadsheet_url}" style="color:#2563eb;">Google Sheets</a>'

        html_body = f"""
    <html><body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 640px; margin: 0 auto; color: #333; padding: 0 10px; word-break: break-word; overflow-wrap: anywhere;">

    <p>{greeting}</p>

    {intro_html or f"<p>Your job search just finished &mdash; <b>{now_formatted} PT</b>.</p>"}

    <div style="background: #f0fdf4; border-left: 4px solid #22c55e; padding: 12px 16px; margin: 16px 0; border-radius: 4px;">
        <b style="font-size: 17px;">{num_high}</b> strong match{"es" if num_high != 1 else ""} &bull;
        <b style="font-size: 17px;">{num_potential}</b> worth a look &bull;
        <span style="color: #666; font-size: 13px;">{total_scanned:,} scanned</span>
    </div>

    <h3 style="font-size:14px;letter-spacing:0.05em;color:#16a34a;margin:18px 0 4px;">STRONG MATCHES &mdash; ACT ON THESE</h3>
    {strong_html}
    {followups_html}

    <h3 style="font-size:14px;letter-spacing:0.05em;color:#8b5cf6;margin:22px 0 4px;">WORTH A LOOK &mdash; OPTIONAL, ADJACENT &amp; STRETCH</h3>
    {worth_html}
    {worth_more_html}
    {recovered_html}

    {cascade_summary_html}

    <p style="color:#666;font-size:13px;margin-top:14px;">Full CSV attached{sheets_link}</p>

    <p style="margin-top: 22px;">{signoff}</p>

    <p style="color: #999; font-size: 12px; margin-top: 30px; border-top: 1px solid #eee; padding-top: 12px;">
        Sent by your personal AI job search assistant.
        <br>Powered by a 3-stage AI cascade (pre-filter &rarr; grading &rarr; senior review).
    </p>

    </body></html>
        """
        subject = subject or (f"Job Digest \u2014 {num_high} strong + {num_potential} worth a look "
                              f"({datetime.now(pacific).strftime('%b %d')})")

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject

    # ASCII HTML: every non-ASCII character (dashes, "≈", accented company
    # names) becomes a numeric entity, so the body travels as plain 7-bit
    # text and no client mis-decodes it.
    msg.attach(MIMEText(html_body.encode("ascii", "xmlcharrefreplace").decode("ascii"), 'html'))

    if csv_path and os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(part)

    try:
        _smtp_send(sender, password, recipient.split(','), msg)
        logging.info(f"Email sent successfully to {recipient}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
