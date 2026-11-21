"""Per-job compensation + employment-type extraction: yearly comp
and W2-vs-contract at a glance in the digest and on cards.

Deterministic and free — no LLM calls:
1. jobspy structured fields first (min_amount / max_amount / interval /
   job_type — present on most rows, but often as the STRING 'nan'
   because pandas NaN survives json.dumps(default=str)).
2. Description-text regex fallback (many postings state comp only in the JD).
3. Employment type from job_type + W2/1099/C2C keyword overrides in the text.

All amounts normalize to YEARLY for the at-a-glance display; the original
unit is kept in the display string when it wasn't yearly.
"""
import re
from typing import Optional

_HOURS_YEAR = 2080
_INTERVAL_FACTOR = {
    "yearly": 1, "hourly": _HOURS_YEAR, "daily": 260, "weekly": 52, "monthly": 12,
}


def _num(v) -> Optional[float]:
    """Parse a jobspy amount that may be float, int, numeric string, or the
    literal string 'nan' (pandas NaN through json.dumps(default=str))."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f <= 0:  # NaN or nonsense
        return None
    return f


def _fmt_k(amount: float) -> str:
    """98000 -> '$98K'; 1200000 -> '$1.2M'."""
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M".replace(".0M", "M")
    if amount >= 1_000:
        return f"${round(amount / 1000):g}K"
    return f"${amount:g}"


def _yearly_display(lo: Optional[float], hi: Optional[float], interval: str) -> Optional[str]:
    factor = _INTERVAL_FACTOR.get(interval)
    if factor is None or (lo is None and hi is None):
        return None
    ylo = lo * factor if lo else None
    yhi = hi * factor if hi else None
    if interval == "yearly":
        core = f"{_fmt_k(ylo)}–{_fmt_k(yhi)}/yr" if ylo and yhi and ylo != yhi else f"{_fmt_k(ylo or yhi)}/yr"
        return core
    # Non-yearly: show original unit + the yearly FORK, not a midpoint avg —
    # "≈$52K/yr" for a $20-$30/hr role read like a lowball while its ceiling
    # was $62K (show the fork, judge the ceiling).
    unit = {"hourly": "/hr", "daily": "/day", "weekly": "/wk", "monthly": "/mo"}[interval]
    if lo and hi and lo != hi:
        orig = f"${lo:g}–${hi:g}{unit}"
        approx = f"{_fmt_k(ylo)}–{_fmt_k(yhi)}"
    else:
        orig = f"${(lo or hi):g}{unit}"
        approx = _fmt_k(ylo or yhi)
    return f"{orig} (≈{approx}/yr)"


# ── Description-text fallback ────────────────────────────────────────────────
# "$120,000 - $150,000", "$120k–$150k", "$45/hr", "$60 - $70 per hour",
# "up to $140,000", "$140,000 per year", "$8,000/month"
_RANGE_RE = re.compile(
    r"\$\s?(\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?)\s*([kK])?"
    r"\s*(?:-|–|—|to)\s*"
    r"\$?\s?(\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?)\s*([kK])?"
    r"\s*(?:(?:per|/|a)\s*(hour|hr|year|yr|annum|month|mo|week|wk|day)s?\b)?", re.I)
_SINGLE_RE = re.compile(
    r"\$\s?(\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?)\s*([kK])?"
    r"\s*(?:(?:per|/|a)\s*(hour|hr|year|yr|annum|month|mo|week|wk|day)s?\b)", re.I)
_UNIT_TO_INTERVAL = {
    "hour": "hourly", "hr": "hourly", "year": "yearly", "yr": "yearly",
    "annum": "yearly", "month": "monthly", "mo": "monthly",
    "week": "weekly", "wk": "weekly", "day": "daily",
}


def _amt(txt: str, k: Optional[str]) -> Optional[float]:
    try:
        v = float(txt.replace(",", ""))
    except ValueError:
        return None
    if k:
        v *= 1000
    return v if v > 0 else None


def _infer_interval(lo: Optional[float], hi: Optional[float], unit: Optional[str]) -> Optional[str]:
    if unit:
        return _UNIT_TO_INTERVAL.get(unit.lower())
    ref = hi or lo
    if ref is None:
        return None
    if ref >= 20_000:
        return "yearly"
    if ref <= 500:
        return "hourly"
    if 2_000 <= ref < 20_000:
        return "monthly"
    return None  # 500-2000: too ambiguous (weekly? daily?) — skip


# "between $40.00 and $60.00/hr" (some employers' phrasing — plain "and" as a range
# separator is too false-positive-prone, so require the "between" anchor)
_BETWEEN_RE = re.compile(
    r"between\s+\$\s?(\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?)\s*([kK])?"
    r"\s+and\s+\$?\s?(\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?)\s*([kK])?"
    r"\s*(?:(?:per|/|a)\s*(hour|hr|year|yr|annum|month|mo|week|wk|day)s?\b)?", re.I)


def _comp_from_text(text: str) -> Optional[tuple[str, int, int]]:
    """Returns (display, yearly_min, yearly_max) or None. The yearly numbers
    matter as much as the display — sorting, the $-filter chips, and the comp
    floor all read comp_yearly_*, so a display-only extraction leaves the row
    invisible to them."""
    for regex in (_BETWEEN_RE, _RANGE_RE):
        m = regex.search(text)
        if m:
            lo, hi = _amt(m.group(1), m.group(2)), _amt(m.group(3), m.group(4))
            interval = _infer_interval(lo, hi, m.group(5))
            if interval and lo and hi and hi >= lo and (hi / max(lo, 1)) < 20:
                f = _INTERVAL_FACTOR[interval]
                return _yearly_display(lo, hi, interval), round(lo * f), round(hi * f)
    m = _SINGLE_RE.search(text)
    if m:
        v = _amt(m.group(1), m.group(2))
        interval = _infer_interval(v, v, m.group(3))
        if interval and v:
            f = _INTERVAL_FACTOR[interval]
            return _yearly_display(v, v, interval), round(v * f), round(v * f)
    return None


# ── Employment type ─────────────────────────────────────────────────────────
_TYPE_LABEL = {
    "fulltime": "Full-time", "parttime": "Part-time",
    "contract": "Contract", "temporary": "Temp", "internship": "Internship",
}


def _employment_type(job_type: Optional[str], text: str) -> Optional[str]:
    parts = []
    jt = str(job_type or "").lower()
    if jt and jt != "nan":
        seen = [lbl for key, lbl in _TYPE_LABEL.items() if key in jt]
        if seen:
            parts.append("/".join(seen[:2]))
    # Keyword overrides from the JD text (jobspy job_type never carries these)
    lowered = text.lower()
    if re.search(r"\bw-?2\b", lowered):
        parts.append("W2")
    if re.search(r"\bc2c\b|corp.to.corp", lowered):
        parts.append("C2C")
    if re.search(r"\b1099\b", lowered):
        parts.append("1099")
    if not parts and re.search(r"\bcontract(or)?\b", lowered):
        parts.append("Contract")
    if not parts:
        return None
    # Dedup while preserving order
    out = []
    for p in parts:
        if p not in out:
            out.append(p)
    return " · ".join(out)


def extract_comp(job_data: dict) -> dict:
    """Returns {comp_text, employment_type, comp_yearly_min, comp_yearly_max,
    comp_source} (values may be None). job_data is the raw scraper dict
    (jobspy row or ATS/board record). comp_source: 'structured' (scraper
    fields) | 'text' (JD regex) | None."""
    desc = str(job_data.get("description") or "")
    lo = _num(job_data.get("min_amount"))
    hi = _num(job_data.get("max_amount"))
    interval = str(job_data.get("interval") or "").lower()
    comp_text = None
    ymin = ymax = None
    source = None

    if (lo or hi) and interval in _INTERVAL_FACTOR:
        comp_text = _yearly_display(lo, hi, interval)
        f = _INTERVAL_FACTOR[interval]
        ymin = round((lo or hi) * f)
        ymax = round((hi or lo) * f)
        source = "structured"
    if not comp_text and desc:
        parsed = _comp_from_text(desc[:6000])
        if parsed:
            comp_text, ymin, ymax = parsed
            source = "text"

    return {
        "comp_text": comp_text,
        "employment_type": _employment_type(job_data.get("job_type"), desc[:6000]),
        "comp_yearly_min": ymin,
        "comp_yearly_max": ymax,
        "comp_source": source,
    }


# ── Contract-pay note ───────────────────────────────────────────────────────
# Agencies and contract roles are NEVER penalized (they are a legitimate route
# into a company) — but the card must say that the pay works differently.
# Shown in place on the digest card and the Review / Jobs
# rows (web/src/app/{review,jobs}/page.tsx mirror is_contract_pay /
# contract_pay_note and STAFFING_AGENCY_RE — keep the three in step).
#
# The note is BUILT FROM WHAT IS KNOWN, never a fixed claim: a fixed
# "paid hourly (W2 through an agency, or 1099)" was wrong on every card
# whose pay was yearly, a day rate, or not stated at all.
#   - head:   "Contract role" (employment type names contract / freelance /
#             1099 / C2C, or the company is a staffing agency), "Temporary
#             role" (only Temp is named), else "Hourly role" (hourly pay and
#             nothing says employee).
#   - basis:  ", paid hourly" only when comp_text has /hr; ", paid a day
#             rate" only when it has /day. Yearly or unknown pay adds nothing.
#   - agency: " (W2 through an agency, or 1099)" only when the employment type
#             names W2 / 1099 / C2C or the company is a staffing agency.
CONTRACT_PAY_TAIL = ("taxes, benefits and paid time off work differently from a salaried "
                     "role; compare take-home, not the annualized figure.")
CONTRACT_PAY_NOTE = f"Contract role — {CONTRACT_PAY_TAIL}"  # the base (nothing else known)
_CONTRACT_TYPE_RE = re.compile(r"\b(?:contract(?:or)?|temp(?:orary)?|freelance|1099|c2c)\b", re.I)
_CONTRACT_WORD_RE = re.compile(r"\b(?:contract(?:or)?|freelance|1099|c2c)\b", re.I)
_EMPLOYEE_TYPE_RE = re.compile(r"\b(?:(?:full|part)-?\s?time|intern(?:ship)?)\b", re.I)
_TAX_FORM_RE = re.compile(r"\b(?:w-?2|1099|c2c)\b", re.I)
# Known staffing agencies / contingent-workforce firms (a hand-kept market
# list), plus the decorations the scrapers add
# ("Agency X (staffing)", "Acme (via StaffCo)"). Conservative: a miss only
# drops the "(W2 through an agency, or 1099)" hint, a false hit would claim it.
STAFFING_AGENCY_RE = re.compile(
    r"\bstaffing\b|\(via\s|\brecruit(?:ing|ment)\b|\bworkforce solutions\b"
    r"|\b(?:aquent|robert half|russell tobin|creative circle|the planet group|workgenius"
    r"|pms for hire|synergis|mindsource|cypress hcm|insight global|inspyr solutions|cella"
    r"|adecco|randstad|kforce|24 seven talent|solomon page|pyramid consulting|intellipro"
    r"|intelliswift|infotree|epitec|atr international|ektello|us tech solutions"
    r"|the judge group|talentburst|syndicatebleu|pride global|horizontal talent"
    r"|hellokindred|harvey nash|experis|dexian|akkodis|yoh|varite|teksystems"
    r"|apex systems|kelly services|magnit|willhire|tcwglobal|the mom project"
    r"|saicon consultants|motion recruitment|mondo|yunojuno|ursus)\b",
    re.I)


def _clean_str(v) -> str:
    s = v.strip() if isinstance(v, str) else ""
    return "" if s.lower() in ("nan", "none") else s


def is_staffing_agency(company) -> bool:
    """True when the company name is a known staffing / contingent-workforce
    agency or carries a scraper's agency decoration ("(staffing)", "(via X)")."""
    return bool(STAFFING_AGENCY_RE.search(_clean_str(company)))


def is_contract_pay(employment_type, comp_text) -> bool:
    """True when the stored employment type says contract (Contract / Temp /
    Freelance / 1099 / C2C), or the pay is hourly and nothing says the role is
    a regular full-time / part-time / intern job (an hourly Apple full-time
    role is an employee role, not a contract)."""
    etype, comp = _clean_str(employment_type), _clean_str(comp_text)
    if _CONTRACT_TYPE_RE.search(etype):
        return True
    return "/hr" in comp and not _EMPLOYEE_TYPE_RE.search(etype)


def contract_pay_note(employment_type, comp_text, company=None) -> Optional[str]:
    """The contract-pay note for a contract / hourly role (None otherwise),
    saying only what the row supports — see the block comment above."""
    if not is_contract_pay(employment_type, comp_text):
        return None
    etype, comp = _clean_str(employment_type), _clean_str(comp_text)
    agency = is_staffing_agency(company)
    if _CONTRACT_WORD_RE.search(etype) or agency:
        head = "Contract role"
    elif _CONTRACT_TYPE_RE.search(etype):
        head = "Temporary role"
    else:
        head = "Hourly role"
    basis = ""
    if "/hr" in comp and head != "Hourly role":
        basis = ", paid hourly"
    elif "/day" in comp:
        basis = ", paid a day rate"
    via = " (W2 through an agency, or 1099)" if (_TAX_FORM_RE.search(etype) or agency) else ""
    return f"{head}{basis}{via} — {CONTRACT_PAY_TAIL}"


# ── LLM fallback: postings the regex can't parse —
# "5000-8000 / month", "$50/hr but 20 hrs/week", prose ranges. STRICTLY
# extraction, never estimation: if the posting states no numbers, N/A stands.
_LLM_COMP_PROMPT = """Extract the stated compensation from this job posting, if any.

<posting_title>{title}</posting_title>
<posting_text>
{description}
</posting_text>

Rules — read carefully:
- Report ONLY compensation figures the posting itself states. If the posting
  contains no pay numbers, respond {{"found": false}}. NEVER estimate market
  rates, NEVER guess from the title or seniority.
- Convert what IS stated to a yearly range in USD:
  * monthly x 12, weekly x 52, daily x 260.
  * hourly: if the posting states hours (e.g. "20 hours/week"), use them
    (rate x hours x 52). Only if no hours are stated assume full-time (x 2080).
  * If the posting states a single number, use it for both min and max.
- "basis" = a terse trace of your arithmetic, e.g. "$6K-8K/mo x12" or
  "$50/hr x 20h/wk". Max 40 chars.
- employment_type: only if the posting states it (W2, 1099, C2C, Contract,
  Full-time, Part-time, Temp) — else null.

Respond with ONLY this JSON:
{{"found": true/false, "yearly_min": <int|null>, "yearly_max": <int|null>, "basis": "<str|null>", "employment_type": "<str|null>"}}"""


def llm_extract_comp(title: str, description: str, model: str = None) -> Optional[dict]:
    """One opus call to extract stated-but-unparseable comp. Returns the same
    shape as extract_comp with comp_source='llm' and an '(est.)'-marked
    display, or None when the posting states nothing / the call fails."""
    if not description or len(description.strip()) < 40:
        return None
    from .gpt_filter import call_llm
    from .config import COMP_EXTRACT_MODEL
    from .jd_sanitizer import clean_jd
    import json as _json

    # Posting text is untrusted data: AI-directed sentences never reach the model.
    prompt = _LLM_COMP_PROMPT.format(
        title=str(title or "")[:120], description=clean_jd(str(description))[:6000])
    try:
        answer = call_llm(prompt, model=model or COMP_EXTRACT_MODEL)
        if not answer:
            return None
        start = answer.find("{")
        if start == -1:
            return None
        parsed = _json.JSONDecoder().raw_decode(answer[start:])[0]
    except Exception:
        return None
    if not parsed.get("found"):
        return None
    ymin, ymax = parsed.get("yearly_min"), parsed.get("yearly_max")
    if not (isinstance(ymin, (int, float)) or isinstance(ymax, (int, float))):
        return None
    ymin = int(ymin or ymax)
    ymax = int(ymax or ymin)
    if ymin <= 0 or ymax < ymin or ymax > 2_000_000:  # sanity rails
        return None
    core = f"{_fmt_k(ymin)}–{_fmt_k(ymax)}/yr" if ymin != ymax else f"{_fmt_k(ymin)}/yr"
    basis = str(parsed.get("basis") or "").strip()[:40]
    comp_text = f"~{core} (est." + (f" {basis})" if basis else ")")
    etype = parsed.get("employment_type")
    return {
        "comp_text": comp_text,
        "employment_type": (str(etype)[:40] if etype else None),
        "comp_yearly_min": ymin,
        "comp_yearly_max": ymax,
        "comp_source": "llm",
    }


# Comp floor: a role whose KNOWN yearly pay ceiling sits below it leaves the
# feed. Applies only when comp IS KNOWN — N/A rows unaffected. Both floors are
# read from the profile (search.comp_floor_yearly /
# search.comp_floor_secondary_metro); these defaults apply when a profile
# leaves a key out.
_DEFAULT_COMP_FLOOR_YEARLY = 50_000
_DEFAULT_COMP_FLOOR_SECONDARY_METRO = 100_000


def _floor_from_profile(key: str, default: int) -> int:
    """A yearly floor from the profile's search block, or the default when the
    key is missing or empty. A value that is not a number fails loudly at
    import rather than silently applying no floor."""
    from .candidate_profile import SEARCH
    value = SEARCH.get(key)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(float(str(value).replace(",", "").replace("_", "")))
    except ValueError:
        raise ValueError(f"search.{key} in the candidate profile must be a yearly "
                         f"amount such as 60000, not {value!r}") from None


COMP_FLOOR_YEARLY = _floor_from_profile("comp_floor_yearly", _DEFAULT_COMP_FLOOR_YEARLY)
# ON-SITE roles in the profile's secondary metro (classify_location class
# 'secondary') must clear a higher floor. Remote postings and multi-location
# postings with a primary-metro option classify as 'remote'/'primary', so they
# keep the base floor.
COMP_FLOOR_SECONDARY_METRO_ONSITE = _floor_from_profile(
    "comp_floor_secondary_metro", _DEFAULT_COMP_FLOOR_SECONDARY_METRO)


def floor_for_location(location) -> int:
    """The comp floor a job must clear to keep its tier, by location class."""
    from .config import classify_location
    if classify_location(str(location or "")) == "secondary":
        return COMP_FLOOR_SECONDARY_METRO_ONSITE
    return COMP_FLOOR_YEARLY


def apply_comp_floor(db) -> int:
    """UNTIER tiered rows whose KNOWN yearly comp CEILING (top of the stated
    fork) sits below their location's floor (COMP_FLOOR_YEARLY; the higher
    COMP_FLOOR_SECONDARY_METRO_ONSITE for secondary-metro on-site roles). Terminal, not a
    one-tier demote: degrading strong -> worth_a_look would leave a role
    paying well under the floor VISIBLE in the digest (below threshold means
    gone). match_level kept, reasoning annotated — recoverable in the DB,
    never surfaced as attractive. If the ceiling clears the floor the role
    stays and the card shows the full fork. N/A comp is untouched. Returns
    number of rows untiered. The tiered set is small, so a Python pass
    (needed for classify_location) costs nothing."""
    import logging
    try:
        conn = db._get_thread_conn()
        rows = conn.execute("""SELECT id, location, comp_yearly_max
            FROM scrapes
            WHERE tier IN ('strong', 'worth_a_look')
              AND comp_yearly_max IS NOT NULL""").fetchall()
        n = 0
        for rid, location, ymax in rows:
            floor = floor_for_location(location)
            if ymax >= floor:
                continue
            why = ("secondary-metro on-site pays" if floor == COMP_FLOOR_SECONDARY_METRO_ONSITE
                   else "posting pays")
            note = f" [Comp floor: {why} under ${floor // 1000}K/yr - removed from feed.]"
            conn.execute("UPDATE scrapes SET tier=NULL, reasoning=COALESCE(reasoning,'') || ? WHERE id=?",
                         (note, rid))
            n += 1
        conn.commit()
        if n:
            logging.info(f"Comp floor: untiered {n} row(s) paying under their location floor")
        return n
    except Exception as e:
        logging.warning(f"Comp floor application failed (non-fatal): {e}")
        return 0


def enrich_tiered_missing_comp(db, cap: int = 20) -> dict:
    """LLM comp fallback for TIERED rows only (the ones the candidate sees) that the
    deterministic pass left empty. Called post-cascade in the daily run and by
    scripts/enrich_comp_llm.py for the backlog. Returns {link: comp_dict} so
    the caller can merge results into an in-memory digest df.

    Cost rail: `cap` bounds calls per invocation.
    """
    import json as _json
    import logging
    updated: dict = {}
    try:
        conn = db._get_thread_conn()
        rows = conn.execute("""
            SELECT id, link, title, raw_data FROM scrapes
            WHERE tier IN ('strong', 'worth_a_look')
              AND (reviewed IS NULL OR reviewed = 0)
              AND (comp_text IS NULL OR comp_text = '')
              AND (comp_source IS NULL OR comp_source != 'llm_none')
              AND raw_data IS NOT NULL
            ORDER BY id DESC LIMIT ?""", (int(cap),)).fetchall()
        for rid, link, title, raw in rows:
            try:
                desc = str(_json.loads(raw).get("description") or "")
            except Exception:
                continue
            comp = llm_extract_comp(title, desc)
            if comp:
                conn.execute("""UPDATE scrapes SET comp_text=?,
                    employment_type=COALESCE(employment_type, ?),
                    comp_yearly_min=?, comp_yearly_max=?, comp_source='llm'
                    WHERE id=?""",
                             (comp["comp_text"], comp["employment_type"],
                              comp["comp_yearly_min"], comp["comp_yearly_max"], rid))
                updated[link] = comp
            else:
                # Mark as tried-and-empty so we never re-pay for this posting
                conn.execute("""UPDATE scrapes SET comp_source='llm_none'
                    WHERE id=? AND comp_text IS NULL""", (rid,))
        conn.commit()
        if rows:
            logging.info(f"LLM comp fallback: {len(updated)}/{len(rows)} postings yielded stated comp")
    except Exception as e:
        logging.warning(f"LLM comp enrichment failed (non-fatal): {e}")
    return updated
