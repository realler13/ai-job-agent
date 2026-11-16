"""Interview Prep — per-role, per-stage preparation sheet.

When the candidate moves a card to an interview stage (Screening Call / Interview /
Final Interview / Offer), this module builds a single markdown sheet for THAT
stage: a 15-minute review, one stage pack (screening walk-through and pay
answer / talk tracks for the CV the candidate sent / final-round angles / offer
questions), likely questions tagged [from posting] or [inferred], the candidate's
strongest STAR stories (from the candidate's own interview-notes transcript —
grounded, never invented), a company angle from cached intel, and an "After the
call" block (thank-you + check-in drafts). It also works with NO posting text
at all (a company that approached the candidate).

Anti-hallucination contract (mirrors the CV + outreach rules):
- STAR stories, numbers, employers, tools: ONLY from the candidate's transcript + verified
  METRICS_BANK. Never invent a project, metric, or scope. Obey the
  canonical-tenure rule (the one years-of-experience figure in the verified
  metrics) and never bind it to a narrow specialty.
- Company facts: only reference what the provided company intel states; label
  anything uncertain rather than asserting it.
- The candidate's own words (config.CANDIDATE_NARRATIVE) are used verbatim when set; when
  unset the sheet stays neutral and never invents a reason for leaving, a pay
  number or people-management scope.
- The CV the candidate sent is read as "what they will ask about" — never as a fact
  source.
- CHECKED IN CODE after generation (the model is not trusted to self-police):
  every number must trace to the candidate's VERIFIED sources — source bank, METRICS_BANK,
  the candidate's narrative words; never the transcript, which is spoken recollection — (or,
  for company facts, the posting / intel) or its sentence is removed; a pay
  number with no pay target of the candidate's on record must come from the posting /
  intel; invented leaving-reasons and management scope are removed; the
  "After the call" drafts fall back to a fixed template unless they pass. If
  a check itself errors, no sheet is shipped.

The generation core (`build_prep`) is shared by the Kanban spawn path
(`generate_interview_prep`) and any other caller, so all produce identical output.
"""

import logging
import os
import re
from datetime import datetime
from typing import Optional

from .config import (
    APPLICATION_STANDARD_ANSWERS,
    CANDIDATE_NARRATIVE,
    INTERVIEW_PREP_DIR,
    METRICS_BANK_TEXT,
    DEFAULT_GENERATION_MODEL,
    TRUNCATION_JOB_DESC_CV,
    TRUNCATION_SOURCE_BANK,
    TRUNCATION_TRANSCRIPT_CONTEXT,
    resolve_cv_domain,
    DOMAIN_CV_MAP,
)
from .jd_sanitizer import UNTRUSTED_JD_CLAUSE, clean_jd
from .outreach_intel import _safe_slug, _trim_sentence, split_sentences

logger = logging.getLogger(__name__)


# Stages a sheet can be built for (the Kanban interview stages). Each gets ONE
# stage pack; an unknown / missing stage falls back to the general Interview pack.
PREP_STAGES = ("Screening Call", "Interview", "Final Interview", "Offer")
DEFAULT_PREP_STAGE = "Interview"

# Every input block has its OWN budget (chars). One shared cut at the end used
# to drop the tail of the transcript silently: the source bank plus the
# transcript can overrun a single shared cut.
_BUDGET_JD = TRUNCATION_JOB_DESC_CV
_BUDGET_COMPANY_INTEL = 8000
_BUDGET_SOURCE_BANK = TRUNCATION_SOURCE_BANK
# The transcript loader budgets its sections to TRUNCATION_TRANSCRIPT_CONTEXT
# but adds a focus line and separators on top — leave headroom for those.
_BUDGET_TRANSCRIPT = TRUNCATION_TRANSCRIPT_CONTEXT + 1000
_BUDGET_SENT_CV = 9000

REVIEW_MAX_WORDS = 150
THANK_YOU_MAX_WORDS = 120
# Quality floor on what the MODEL wrote (a refusal or a truncation is not a
# sheet). Measured before the code adds its own template blocks and footer.
_MIN_SHEET_CHARS = 500

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROMPT_INTERVIEW_PREP = """Today is {today}. You are the candidate's interview coach. Produce a tight, genuinely useful prep sheet for ONE specific interview. Who the candidate is (background, level, years of experience, current situation) comes ONLY from the SOURCE BANK, the VERIFIED METRICS and THE CANDIDATE'S OWN WORDS below; assume nothing else. Use the one canonical years-of-experience figure in the VERIFIED METRICS; never state another and never bind it to a narrow specialty (not "N years of <one specialty>"). Describe every role that has ended in past tense.

GROUND RULES (non-negotiable):
- Every STAR story, number, employer, and tool you cite must come from THE CANDIDATE'S TRANSCRIPT / SOURCE BANK or the VERIFIED METRICS below. On any factual conflict, the CAREER SOURCE BANK section wins (it is the candidate's newest, finalized document). NEVER invent a project, a metric, a company, or a scope. If you don't have a strong real story for a question, say "no direct story — bridge from <closest real experience>" rather than fabricating one.
- Company facts: use ONLY what COMPANY INTEL states. If intel is thin, say "research before the call" — do not assert facts you were not given.
- Anti-slop: plain language, no "passionate"/"synergy"/"I'm excited to", no em-dash chains, one idea per line. This is a working document, not marketing.
- {untrusted_jd_clause}
- THE CANDIDATE'S OWN WORDS: where that block gives the candidate's wording, use it verbatim as the basis of the answer (shorten only; never add a fact to it). Where it says "not provided", follow the instruction next to it exactly — neutral, truthful, forward-looking. NEVER invent a reason for leaving, a pay number, or people-management scope (direct reports, hiring authority, formal line management).
- THE CV THE CANDIDATE SENT is what the interviewer has in hand: use it to predict what they will ask about. It is NOT a fact source — every fact in an answer still comes from the SOURCE BANK or the VERIFIED METRICS. If a CV line is not backed by those, write "verify before the call" next to it instead of a talk track.
- NUMBERS ARE CHECKED IN CODE: a number in the candidate's answers or stories must appear in the SOURCE BANK or VERIFIED METRICS (a transcript figure the bank does not confirm is deleted); a number about the company or the pay range must appear in the JOB POSTING or COMPANY INTEL. A sentence with any other number is deleted from the sheet.

INTERVIEW STAGE: {stage} — {stage_brief}

TARGET ROLE: {job_title} at {company}{location_line}

JOB POSTING:
{jd_text}

COMPANY INTEL (what we know — cite-or-omit; may be sparse):
{company_intel}

THE CANDIDATE'S OWN WORDS (the basis for these three answers):
{narrative}

THE CV THE CANDIDATE SENT FOR THIS ROLE ({sent_cv_label}):
{sent_cv}

THE CANDIDATE'S VERIFIED METRICS (use only these numbers):
{metrics_bank}

THE CANDIDATE'S EXPERIENCE / STAR MATERIAL (real stories, from the candidate's interview transcript):
{transcript_context}

Produce EXACTLY these sections in markdown, with these exact ## and ### headers, in this order. Keep it scannable — the candidate reads this on a phone before a call:

{sections}

Return ONLY the markdown sheet, starting with "## 15-minute review". No preamble."""


_STAGE_BRIEFS = {
    "Screening Call": "a first call with a recruiter or the hiring manager. They check the basics: "
                      "background fit, why the candidate is looking, pay expectations, logistics.",
    "Interview": "a full interview with the hiring manager or the team. Expect depth on the candidate's "
                 "experience, on every claim in the CV that was sent, on working style, and on level fit.",
    "Final Interview": "a final round, often several interviewers in a row. They compare notes: "
                       "consistency across people, judgment, team fit, and a strong close.",
    "Offer": "an offer conversation. The goal is to understand the offer fully and raise open "
             "items as questions before deciding.",
}

_SECTION_REVIEW = """## 15-minute review
AT MOST 150 words in total, as short bullets, in this order: one ANCHOR sentence (who the candidate is for THIS role, built on one verified fact); 3 MESSAGES to land (one line each, each tied to a verified fact); a PAY line (the posted range if the posting states one, otherwise the question "Can you share the band for this role?" — never a floor or a number of the candidate's own unless THE CANDIDATE'S OWN WORDS gives one); a LOGISTICS line (what to confirm — format, who is on the call, timeline, location; as questions when unknown, never guessed); 3 QUESTIONS to ask them."""

_PACK_SCREENING = """## Screening call pack
### Walk-through (60-90 seconds)
The candidate's spoken background walk-through for THIS role type, first person, 130-190 words: where the candidate is now (as the SOURCE BANK / THE CANDIDATE'S OWN WORDS state it, never assumed), the past work most relevant to this role, why this role. Verified facts only, past tense for past roles.
### Why the candidate is looking
The candidate's answer to "why are you looking / why did you leave", built from THE CANDIDATE'S OWN WORDS (why the candidate left). 2-4 sentences, first person.
### Pay
The candidate's answer to "what are your pay expectations", built from THE CANDIDATE'S OWN WORDS (pay). If the posting states a range, quote it exactly as posted. Never volunteer a floor.
### Logistics
What a recruiter usually checks (location and on-site days, start date, contract or full-time, interview process and timeline). Give the candidate's answer only where the candidate's material states it; otherwise write the question the candidate should ask, or "confirm before the call". Never guess."""

_PACK_INTERVIEW = """## Interview pack
### Talk tracks for the CV the candidate sent
One talk track for EACH line under METRIC-BEARING BULLETS: quote the bullet briefly, then 2-3 lines (context, what the candidate did, the result) using only SOURCE BANK / VERIFIED METRICS facts. If the sent CV is unknown, say so in the first line and write tracks for the 4-5 verified metrics most relevant to this role instead.
### Honest bridges for likely doubts
2-4 doubts this interviewer is likely to have (a gap against the posting, a gap in the timeline, a domain switch, level). For each: the doubt, then an honest bridge from the candidate's closest REAL experience. Never hide it, never fake it.
### Why this level
2-3 lines on why THIS role's level is the right level for the candidate now, grounded in what the candidate has actually done. For anything about managing people, follow THE CANDIDATE'S OWN WORDS (people management).
### First 90 days
3 bullets (30 / 60 / 90 days): what the candidate would do, phrased as intentions that depend on what the candidate learns. No promised results, no invented numbers."""

_PACK_FINAL = """## Final interview pack
### Vary the angle across interviewers
For each likely interviewer in a final round (the hiring manager, a peer or cross-functional partner, a senior leader — use a real name only if COMPANY INTEL states it): one line on what that person cares about, and which of the candidate's REAL stories to lead with, so the candidate does not tell every interviewer the same story.
### Closing
The candidate's closing statement (3-4 sentences: what the candidate heard matters most to them, the verified experience that matches it, a plain statement that the candidate wants the role) and 2 closing questions (next steps and timeline; whether anything gives them pause that the candidate can address now)."""

_PACK_OFFER = """## Offer pack
Items to raise, EACH phrased as a question the candidate can ask (never a demand), grouped under short labels: pay and how it is structured (base, bonus, equity — or, for a contract, the hourly rate, W2 or 1099, and what benefits are included); level and title; start date; work location and on-site days; team and who the candidate reports to; the decision deadline and who to send questions to. Use numbers only from the posting or COMPANY INTEL; for the candidate's own target follow THE CANDIDATE'S OWN WORDS (pay). End with the line: "Ask for the offer in writing and for time to review it." """

_SECTION_QUESTIONS = """## Likely questions & how to answer
5-7 likely questions for THIS role at THIS stage (mix behavioral + role-specific). Tag EVERY question: [from posting] when it follows from a line in the posting (quote the few words it comes from), or [inferred] when it comes from the role title, the stage or the company intel. {posting_note}For each: the question with its tag, then a one-line angle on how the candidate should answer, and — where one fits — which of the candidate's REAL stories to use (name it)."""

_SECTION_STORIES = """## The candidate's strongest stories (STAR)
3-4 of the candidate's real stories most relevant to this role, each as tight S/T/A/R bullets with the verified result. Pick the ones that map to this posting. Pull from the candidate's transcript only."""

_SECTION_COMPANY = """## Company angle — "why this company"
3-4 bullets the candidate can weave in, grounded in the company intel. If intel is thin, give the candidate 2-3 specific things to research before the call instead."""

_SECTION_ASK = """## Smart questions to ask them
4-5 sharp questions that show the candidate understands the role and screen for fit (team, success in 6 months, what the work actually is, red flags to probe)."""

_SECTION_WATCHOUTS = """## Watch-outs
2-3 honest notes: any gap this posting exposes and how to bridge it from adjacent real experience (never hide it, never fake it), plus one delivery reminder."""

_SECTION_AFTER = """## After the call
### Thank-you note (send within 24 hours)
A draft of AT MOST 120 words that contains the placeholders [Interviewer name] and [one thing they said] exactly as written (the candidate fills them in). Plain and specific. No new claims and no numbers.
### Check-in (after 5 business days of silence)
A 2-3 sentence draft with the placeholder [Interviewer name]: polite, specific, one clear question about timing. Never a bare "just checking in"."""

_STAGE_PACKS = {
    "Screening Call": _PACK_SCREENING,
    "Interview": _PACK_INTERVIEW,
    "Final Interview": _PACK_FINAL,
    "Offer": _PACK_OFFER,
}

_NO_POSTING_TEXT = ("(no job posting on file — the company approached the candidate, or the posting text is "
                    "unavailable. Do NOT guess requirements as if they were posted: prep from the "
                    "role title, COMPANY INTEL and the candidate's material.)")


def normalize_stage(stage: Optional[str]) -> str:
    """One of PREP_STAGES; anything else → the general Interview pack."""
    wanted = str(stage or "").strip().lower()
    for s in PREP_STAGES:
        if s.lower() == wanted:
            return s
    return DEFAULT_PREP_STAGE


def _sections_for(stage: str, has_posting: bool) -> str:
    """The section spec for one stage: the 15-minute review, ONE stage pack,
    then the shared sections (an Offer sheet skips the question/story drill)."""
    posting_note = "" if has_posting else ("There is NO posting for this role, so every "
                                           "question is [inferred]. ")
    parts = [_SECTION_REVIEW, _STAGE_PACKS[stage]]
    if stage != "Offer":
        parts += [_SECTION_QUESTIONS.format(posting_note=posting_note), _SECTION_STORIES,
                  _SECTION_COMPANY, _SECTION_ASK]
    parts += [_SECTION_WATCHOUTS, _SECTION_AFTER]
    return "\n\n".join(parts)


def _fit(text: str, budget: int, label: str) -> str:
    """Hold ONE input block to its own budget. A block that overruns is cut at
    a line boundary with a visible marker and a warning — never silently, and
    never at the expense of the blocks after it."""
    text = str(text or "")
    if len(text) <= budget:
        return text
    cut = text[:budget]
    if "\n" in cut[budget // 2:]:
        cut = cut[:cut.rfind("\n")]
    logger.warning(f"interview_prep: {label} is {len(text)} chars, over its {budget} budget — "
                   f"cut to {len(cut)}")
    return cut + f"\n[... {label} cut here: over its {budget}-character budget ...]"


def _said(narrative: Optional[dict], key: str) -> Optional[str]:
    value = (narrative or {}).get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _narrative_block(narrative: Optional[dict]) -> str:
    """The candidate's own words for the three answers that must never be made up — or,
    per answer, the neutral rule to follow while the candidate has not given them."""
    why, pay, people = (_said(narrative, k) for k in
                        ("why_left", "comp_target", "people_management"))
    return "\n".join([
        "- Why the candidate left / why the candidate is looking: " + (
            f'CANDIDATE WORDING (use verbatim as the basis): "{why}"' if why else
            "not provided. Write 2-3 neutral, truthful, forward-looking sentences: the candidate "
            "is looking for the next role, then what draws "
            "the candidate to THIS role. Do NOT state or imply any reason the role ended (no layoff, "
            "restructuring, contract, budget or personal reason) — none is on record."),
        "- Pay: " + (
            f'CANDIDATE WORDING (use verbatim as the basis): "{pay}"' if pay else
            "not provided. Never state a number of the candidate's and never volunteer a floor. If the "
            "posting states a pay range, the candidate's answer quotes that range as posted; otherwise the candidate's "
            'answer is the question "Can you share the band for this role?"'),
        "- People management: " + (
            f'CANDIDATE WORDING (use verbatim as the basis): "{people}"' if people else
            "not provided. Describe only what the SOURCE BANK states (teams the candidate built or "
            "coordinated, with their verified sizes). Never claim direct reports, hiring "
            "authority or formal line management."),
    ])


# ── The CV the candidate sent ──────────────────────────────────────────────────────────

_UNKNOWN_CV = {"status": "unknown", "label": "sent version unknown — no CV file on record",
               "text": "", "metric_bullets": []}


def _read_sent_cv(cv_path: Optional[str]) -> Optional[dict]:
    """Text of a generated CV (.docx, python-docx): {"path", "text",
    "metric_bullets"} — the bullets that carry a number are what an interviewer
    will ask about. None when the file is missing or unreadable."""
    if not cv_path:
        return None
    candidates = [cv_path] if os.path.isabs(cv_path) else [
        cv_path, os.path.join(_PROJECT_ROOT, cv_path)]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if not path:
        return None
    try:
        from docx import Document
        paragraphs = [(p.text.strip(), (p.style.name if p.style is not None else "") or "")
                      for p in Document(path).paragraphs]
    except Exception as e:
        logger.warning(f"interview_prep: could not read the sent CV {path}: {e}")
        return None
    lines = [t for t, _ in paragraphs if t]
    if not lines:
        return None
    bullets = [t for t, style in paragraphs
               if t and (style.lower().startswith("list") or t[:1] in "•·▪-–")]
    return {"path": path, "text": "\n".join(lines),
            "metric_bullets": [b for b in bullets if re.search(r"\d", b)]}


def _sent_cv_for(row: dict, db=None) -> dict:
    """What we know about the CV the candidate sent for this application.

    Uses the sent-CV record when the DB has one (db.get_sent_document: answer
    'as_is' | 'edited' | 'own', or an unconfirmed link) and falls back to the
    CV on the card (scrapes.cv_path). Returns {"status": 'confirmed' |
    'edited' | 'unconfirmed' | 'own' | 'unknown', "label", "text",
    "metric_bullets"} — never raises."""
    link = None
    getter = getattr(db, "get_sent_document", None)
    if callable(getter):
        try:
            link = getter(row.get("app_id"))
        except Exception as e:
            logger.warning(f"interview_prep: sent-CV lookup failed: {e}")
    link = link if isinstance(link, dict) else {}
    answer = link.get("answer")
    if answer == "own":
        return {**_UNKNOWN_CV, "status": "own",
                "label": "sent version unknown — the candidate sent a different CV, not the one on file"}
    cv = _read_sent_cv(link.get("cv_path") or row.get("cv_path"))
    if not cv:
        return dict(_UNKNOWN_CV)
    status, label = {
        "as_is": ("confirmed", "the CV the candidate confirmed sending"),
        "edited": ("edited", "the CV on file — the candidate edited it before sending, so the wording "
                             "they saw may differ"),
    }.get(answer, ("unconfirmed", "the CV on file for this card — not confirmed as the one the candidate sent"))
    return {"status": status, "label": label, "text": cv["text"],
            "metric_bullets": cv["metric_bullets"]}


def _sent_cv_block(sent_cv: dict) -> str:
    if not sent_cv.get("text"):
        return ("(Sent version unknown. Say so in the sheet, and build talk tracks from the "
                "VERIFIED METRICS most relevant to this role.)")
    bullets = "\n".join(f"- {b}" for b in sent_cv.get("metric_bullets") or []) or "(none found)"
    return (f"METRIC-BEARING BULLETS (one talk track each):\n{bullets}\n\n"
            f"FULL TEXT:\n{_fit(sent_cv['text'], _BUDGET_SENT_CV, 'sent CV')}")


# ── Deterministic post-checks ────────────────────────────────────────────────

_NUMBER_RE = re.compile(
    r"(?<![A-Za-z\d])"                        # not inside a token like Q1 / W2 / v2
    r"(?:(?:US|CA|C|AU|A)?(?P<cur>[$€£¥])\s?)?"  # a currency sign ("US$" = "$")
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"       # 12,000 / 1200 / 1.5
    r"(?:(?P<mag>k|m|mm|b|bn)\b|\s(?P<word>thousand|million|billion)\b)?", re.IGNORECASE)
_MAGNITUDE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6,
              "b": 1e9, "bn": 1e9, "billion": 1e9}
_CURRENCY_SIGNS = "$€£¥"
_URL_RE = re.compile(r"https?://\S+|\]\([^)]*\)")
_LINE_PREFIX_RE = re.compile(r"^\s*(?:(?:[-*+•]|\d+[.)])\s+|>\s?)*")
_HEADER_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")


def _numbers(text: str) -> set:
    """Every number in a text as a canonical value string, so "2M", "2
    million" and "2,000,000" compare equal ("+" and "%" are ignored). An
    amount with a currency sign keeps it as a marker ("$80K" -> "$80000"):
    a dollar figure must trace to a dollar figure, never to the same number
    in another currency."""
    out = set()
    for m in _NUMBER_RE.finditer(_URL_RE.sub(" ", str(text or ""))):
        value = float(m.group("int").replace(",", "") + (m.group("dec") or ""))
        value *= _MAGNITUDE.get((m.group("mag") or m.group("word") or "").lower(), 1)
        out.add((m.group("cur") or "") + f"{value:.4f}".rstrip("0").rstrip("."))
    return out


def _with_plain(numbers: set) -> set:
    """*numbers* plus each currency amount without its sign: a source "$80K"
    also supports a bare "80K", but a bare source "80K" never supports
    "$80K"."""
    return set(numbers) | {n.lstrip(_CURRENCY_SIGNS) for n in numbers}


# Sections where a number may be a COMPANY fact (posting / intel). Every other
# section — the candidate's stories, walk-through, talk tracks, why-the-candidate-is-looking,
# the after-the-call drafts, and any header we do not recognise — is held to the
# candidate's own sources only.
_COMPANY_FACTS_OK_RE = re.compile(
    r"15-minute|\bpay\b|logistics|bridge|doubt|level|90 days|likely questions|company|"
    r"questions to ask|watch-outs|offer|vary the angle|closing", re.IGNORECASE)
_WHY_LOOKING_RE = re.compile(r"why (?:the candidate|you|i)\b.*(?:looking|left)", re.IGNORECASE)
_HER_OWN_FACTS_RE = re.compile(
    r"walk-through|talk track|stor(?:y|ies)|thank|check-?in|after the call|"
    + _WHY_LOOKING_RE.pattern, re.IGNORECASE)


def _company_facts_ok(section: str, sub: str) -> bool:
    """May a number here come from the posting / intel? The ### sub-header
    decides when it is recognisable; otherwise the ## section does."""
    if sub:
        if _HER_OWN_FACTS_RE.search(sub):
            return False
        if _COMPANY_FACTS_OK_RE.search(sub):
            return True
    return bool(_COMPANY_FACTS_OK_RE.search(section)) and not _HER_OWN_FACTS_RE.search(section)


# Inside the 15-minute review the anchor and the 3 messages are the candidate's claims; only
# the pay line, the logistics line and the questions may carry a posting number.
_REVIEW_RE = re.compile(r"15-minute", re.IGNORECASE)
_REVIEW_COMPANY_LINE_RE = re.compile(
    r"\bpay\b|\bband\b|\brange\b|salary|compensation|\$|logistic|on-?site|hybrid|remote|"
    r"start date|timeline|\?", re.IGNORECASE)


# A reason for leaving that nobody gave us (used only while why_left is unset).
_LEAVING_REASON_RE = re.compile(
    r"\b(?:laid off|lay-?offs?|let go|fired|terminated|restructur\w*|reorgani[sz]\w*|"
    r"downsiz\w*|budget cuts?|cutbacks?|redundan\w*|contract (?:ended|expired|ran out)|"
    r"(?:was|were|wasn't|not) renewed|resigned|quit|burn(?:ed|t)?[- ]?out|"
    r"position was (?:cut|eliminated)|eliminated|company (?:closed|shut)|shut down|"
    r"(?:personal|family|health) reasons?|funding)\b", re.IGNORECASE)
# Formal management scope nobody verified (used only while people_management is unset).
_MGMT_SCOPE_RE = re.compile(
    r"\b(?:direct reports?|line[- ]manag\w*|people manager|performance reviews?|"
    r"hiring authority|hir(?:ed|ing) and fir\w*|headcount)\b", re.IGNORECASE)
# Pay talk (used only while comp_target is unset): a Pay or Offer section, or a
# line about pay. A bare "$" is not enough (a budget in dollars is a story
# fact); "$" with a pay word is ("a floor of $80K", "$80K base").
_PAY_SECTION_RE = re.compile(r"^\W*pay\b|\boffer\b", re.IGNORECASE)
_PAY_WORD_RE = re.compile(
    r"\bpay\b(?!-per|\s+(?:attention|off\b|it\s+forward))|\bsalar(?:y|ies)\b|\bcompensation\b|"
    r"\bwalk[- ]?away\b|/\s?(?:hr|hour|yr|year)\b|\bper\s+(?:hour|annum)\b|"
    r"\bband\s+for\s+(?:this|the)\s+role\b", re.IGNORECASE)
_PAY_WITH_AMOUNT_RE = re.compile(
    r"\b(?:floor|base|target|minimum|ask(?:ing)?|expect\w*|range|band|below|above|comp|rate|"
    r"offer|bonus|equity|hourly|annual(?:ly)?)\b", re.IGNORECASE)


def _is_pay_line(line: str) -> bool:
    return bool(_PAY_WORD_RE.search(line)) or (
        any(sign in line for sign in "$€£¥") and bool(_PAY_WITH_AMOUNT_RE.search(line)))


# "I" only as a capital; "My" / "Me" also at the start of a sentence.
_FIRST_PERSON_RE = re.compile(r"\b(?:I|I've|I'd|I'm|(?i:my|me))\b")
_NEGATION_RE = re.compile(r"\b(?:no|not|never|without|none)\b|n't\b", re.IGNORECASE)
# Other places the candidate's own story is told, where an invented leaving reason can slip
# in ("my role ended in a restructuring"). Checked on sentences about the candidate only:
# a company sentence there may name a reorganisation or funding as a fact.
_HER_STORY_RE = re.compile(r"walk-through|bridge|doubt|15-minute|why this level", re.IGNORECASE)
_ABOUT_HER_RE = re.compile(r"\b(?:I|I've|I'd|I'm|(?i:my|me|the candidate|the candidate's|candidate))\b")

_NEUTRAL_WHY_LOOKING = ("I am looking for my next role. "
                        "This one stood out because of [one specific thing about the role].")


def _line_units(line: str):
    """(prefix, [sentence, ...]) for one sheet line. A table row is one unit."""
    if line.lstrip().startswith("|"):
        return "", [line]
    prefix = _LINE_PREFIX_RE.match(line).group(0)
    return prefix, split_sentences(line[len(prefix):])


def _check_sheet(sheet: str, own_numbers: set, company_numbers: set, own_text: str,
                 narrative: Optional[dict], scaffold_numbers: set = frozenset()) -> tuple:
    """Remove every sentence that fails a deterministic truth check. Returns
    (sheet, removed) where removed = [(section header, reason, sentence)].

    own_numbers / own_text are the candidate's verified sources (source bank,
    METRICS_BANK_TEXT, the candidate's narrative words — never the interview transcript,
    which is spoken recollection); scaffold_numbers are the sheet's
    own header / timing numbers.

    - NUMBERS: each number must be in own_numbers; in sections where company
      facts belong it may also be in company_numbers (in the 15-minute review
      only on its pay / logistics / question lines). A currency amount must
      trace to the same currency ("$2M" is not "2M" in another currency).
    - PAY (comp_target unset): on a pay line, and anywhere in a Pay or Offer
      section, only posting / intel numbers (and scaffold) — no number of
      the candidate's can become a pay floor or target.
    - LEAVING REASON (why_left unset): no stated reason in the
      why-the-candidate-is-looking answer — nor in a sentence about the candidate in
      the walk-through, bridges, review or level answer — unless the candidate's own
      sources contain that wording.
    - MANAGEMENT SCOPE (people_management unset): no claim of direct
      reports / line management / performance reviews / hiring authority in
      a first-person sentence or anywhere in the candidate's own account (stories, talk
      tracks, walk-through, why-looking, after-call drafts); a negation or a
      question is not a claim. The candidate's sources are no exemption here: a
      matching phrase in them does not establish formal authority."""
    why_unset = not _said(narrative, "why_left")
    people_unset = not _said(narrative, "people_management")
    pay_unset = not _said(narrative, "comp_target")
    own_lower = own_text.lower()
    out, removed = [], []
    section = sub = ""
    for line in sheet.split("\n"):
        header = _HEADER_RE.match(line)
        if header:
            if len(header.group(1)) <= 2:
                section, sub = header.group(2), ""
            else:
                sub = header.group(2)
            out.append(line)
            continue
        if not line.strip():
            out.append(line)
            continue
        where = sub or section
        company_ok = _company_facts_ok(section, sub)
        if company_ok and _REVIEW_RE.search(section) and not sub:
            company_ok = bool(_REVIEW_COMPANY_LINE_RE.search(line))
        pay_rule = pay_unset and bool(_PAY_SECTION_RE.search(section)
                                      or _PAY_SECTION_RE.search(sub) or _is_pay_line(line))
        if pay_rule:
            allowed = _with_plain(company_numbers | scaffold_numbers)
        else:
            allowed = _with_plain(own_numbers | company_numbers if company_ok else own_numbers)
        why_section = bool(_WHY_LOOKING_RE.search(where))
        story_section = bool(_HER_STORY_RE.search(where))
        # Sections that are ALL the candidate's own account (stories, talk tracks, walk-
        # through, why-looking, the after-call drafts): a claim there needs no
        # pronoun ("- **A:** Ran headcount planning").
        her_section = bool(_HER_OWN_FACTS_RE.search(where))
        prefix, units = _line_units(line)
        kept = []
        for unit in units:
            reason = None
            untraced = sorted(_numbers(unit) - allowed)
            if untraced and pay_rule:
                reason = (f"pay number(s) not from the posting or intel {', '.join(untraced)} "
                          f"(no pay target of the candidate's on record)")
            elif untraced:
                reason = f"untraced number(s) {', '.join(untraced)}"
            if not reason and why_unset and (
                    why_section or (story_section and _ABOUT_HER_RE.search(unit)
                                    and not _NEGATION_RE.search(unit))):
                m = _LEAVING_REASON_RE.search(unit)
                if m and m.group(0).lower() not in own_lower:
                    reason = f"unverified reason for leaving ({m.group(0)!r})"
            # No exemption from the candidate's sources: a phrase there does not
            # establish formal authority. A question ("Will the role have direct
            # reports?") is not a claim.
            if not reason and people_unset and not _NEGATION_RE.search(unit) \
                    and not unit.rstrip().endswith("?") \
                    and (her_section or _FIRST_PERSON_RE.search(unit)):
                m = _MGMT_SCOPE_RE.search(unit)
                if m:
                    reason = f"unverified management scope ({m.group(0)!r})"
            if reason:
                removed.append((where, reason, unit.strip()))
                logger.warning(f"interview_prep check removed a sentence in {where!r} — "
                               f"{reason}: {unit.strip()[:160]!r}")
            else:
                kept.append(unit)
        if kept:
            out.append(prefix + " ".join(kept))
    return "\n".join(out), removed


def _section_span(sheet: str, title_re: str, level: int = 2):
    """(header_start, body_start, end) of the first section whose header at
    `level` matches title_re; end = the next header of the same or a higher
    level. None when absent."""
    head = re.search(rf"^\s{{0,3}}{'#' * level}\s+[^\n]*(?:{title_re})[^\n]*$", sheet,
                     re.IGNORECASE | re.MULTILINE)
    if not head:
        return None
    nxt = re.search(rf"^\s{{0,3}}#{{1,{level}}}\s", sheet[head.end():], re.MULTILINE)
    end = head.end() + nxt.start() if nxt else len(sheet)
    return head.start(), head.end(), end


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][\w'’+%$/.-]*", re.sub(r"[*_`>#|\[\]]", " ", text)))


# Scaffolding numbers the model may echo from the section instructions.
_SCAFFOLD_ECHOES = "30 / 60 / 90 days; within 24 hours; 5 business days; W2 or 1099"

_PAY_LINE_RE = re.compile(r"\bpay\b|\bband\b|\brange\b|salary|compensation|\$", re.IGNORECASE)


def _drop_last_clause(line: str) -> Optional[str]:
    """The line minus the last clause of its last sentence (deletion only,
    never mid-word, a question stays a question), or None when no clean
    clause cut exists."""
    if line.lstrip().startswith("|"):
        return None
    prefix, units = _line_units(line)
    if not units:
        return None
    last = units[-1].rstrip()
    cut = _trim_sentence(last, len(last) - 1)
    if not cut or _word_count(cut) >= _word_count(last):
        return None
    if last.endswith("?") and cut.endswith("."):
        cut = cut[:-1] + "?"
    return prefix + " ".join(units[:-1] + [cut])


def _cap_review(sheet: str, max_words: int = REVIEW_MAX_WORDS) -> str:
    """Hold the 15-minute review to max_words by deletion only: first drop
    trailing sentences from its longest lines, then the last clause of the
    longest lines (never the pay line). Logged if it still runs over."""
    span = _section_span(sheet, r"15-minute review")
    if not span:
        return sheet
    _, body_start, end = span
    lines = sheet[body_start:end].split("\n")
    while _word_count("\n".join(lines)) > max_words:
        candidates = []
        for i, line in enumerate(lines):
            prefix, units = _line_units(line)
            if len(units) > 1:
                candidates.append((_word_count(line), i, prefix, units))
        if candidates:
            _, i, prefix, units = max(candidates)
            lines[i] = prefix + " ".join(units[:-1])
            continue
        for i in sorted(range(len(lines)), key=lambda k: (-_word_count(lines[k]), k)):
            shorter = None if _PAY_LINE_RE.search(lines[i]) else _drop_last_clause(lines[i])
            if shorter:
                lines[i] = shorter
                break
        else:
            logger.warning(f"interview_prep: 15-minute review is "
                           f"{_word_count(chr(10).join(lines))} words (cap {max_words}) — "
                           f"nothing left to trim cleanly")
            break
    return sheet[:body_start] + "\n".join(lines) + sheet[end:]


def _after_call_template(title: str, company: str) -> str:
    """Known-good "After the call" block: no claims, no numbers, both
    placeholders. Used whenever the model's drafts do not pass the checks."""
    role = f"the {title} role" if title else "the role"
    at = f" at {company}" if company else ""
    name = APPLICATION_STANDARD_ANSWERS.get("first_name") or "[Your name]"
    return (
        "## After the call\n"
        "### Thank-you note (send within 24 hours)\n"
        "Hi [Interviewer name],\n\n"
        f"Thank you for the conversation about {role}{at}. I appreciated hearing about "
        "[one thing they said], and it made the work clearer to me.\n\n"
        "I would be glad to send anything else that helps as you decide on next steps.\n\n"
        f"Best,\n{name}\n\n"
        "### Check-in (after 5 business days of silence)\n"
        "Hi [Interviewer name],\n\n"
        f"I wanted to follow up on {role}. I am still interested and glad to send anything "
        "else you need. Is there an update on timing?\n\n"
        f"Best,\n{name}\n")


def _after_call_ok(block: str) -> Optional[str]:
    """None when the model's "After the call" block is usable, else why not."""
    thanks = _section_span(block, r"thank", level=3)
    checkin = _section_span(block, r"check-?in", level=3)
    if not thanks or not checkin:
        return "thank-you or check-in draft missing"
    note = block[thanks[1]:thanks[2]]
    if not note.strip() or not block[checkin[1]:checkin[2]].strip():
        return "empty draft"
    if _word_count(re.sub(r"\[[^\]]*\]", " x ", note)) > THANK_YOU_MAX_WORDS:
        return f"thank-you note over {THANK_YOU_MAX_WORDS} words"
    placeholders = [p.lower() for p in re.findall(r"\[([^\]]+)\]", note)]
    if not any("name" in p for p in placeholders):
        return "thank-you note has no [Interviewer name] placeholder"
    if not any(w in p for p in placeholders for w in ("said", "thing", "discussed", "topic")):
        return "thank-you note has no [one thing they said] placeholder"
    return None


def _ensure_after_call(sheet: str, title: str, company: str, check_removed: bool) -> str:
    """Keep the model's "After the call" drafts only on a clean pass: present,
    both placeholders, thank-you within the word cap, and untouched by the
    fact check. Anything else → the fixed template (never a patched draft)."""
    span = _section_span(sheet, r"after the call")
    template = _after_call_template(title, company)
    if not span:
        logger.warning("interview_prep: no 'After the call' block — using the template")
        return sheet.rstrip("\n") + "\n\n" + template
    start, _, end = span
    problem = ("the fact check removed a sentence from the drafts" if check_removed
               else _after_call_ok(sheet[start:end]))
    if not problem:
        return sheet
    logger.warning(f"interview_prep: 'After the call' drafts replaced by the template — {problem}")
    tail = sheet[end:].lstrip("\n")
    return sheet[:start] + template + ("\n" + tail if tail else "")


def _ensure_why_looking(sheet: str, narrative: Optional[dict]) -> str:
    """If the checks emptied the why-the-candidate-is-looking answer (why_left unset),
    put the fixed neutral sentence there instead of leaving a hole."""
    if _said(narrative, "why_left"):
        return sheet
    span = _section_span(sheet, _WHY_LOOKING_RE.pattern, level=3)
    if not span or sheet[span[1]:span[2]].strip():
        return sheet
    return (sheet[:span[1]] + "\n" + _NEUTRAL_WHY_LOOKING
            + "\n*(Say this in your own words.)*\n\n" + sheet[span[2]:].lstrip("\n"))


_QUESTION_TAG_RE = re.compile(r"\[(?:from posting|inferred)\]", re.IGNORECASE)


def _fix_question_tags(sheet: str, has_posting: bool) -> str:
    """Every likely question carries a tag. With no posting on file nothing can
    be [from posting]; an untagged question is [inferred] (the weaker claim)."""
    if not has_posting:
        sheet = re.sub(r"\[from (?:the )?posting\]", "[inferred]", sheet, flags=re.IGNORECASE)
    span = _section_span(sheet, r"likely questions")
    if not span:
        return sheet
    _, body_start, end = span
    lines = sheet[body_start:end].split("\n")
    for i, line in enumerate(lines):
        if "?" in line and not _QUESTION_TAG_RE.search(line):
            cut = line.index("?") + 1
            lines[i] = line[:cut] + " [inferred]" + line[cut:]
    return sheet[:body_start] + "\n".join(lines) + sheet[end:]


def _provenance_line(stage: str, today: str, sent_cv: dict) -> str:
    """Closing line written by code: which stage, which day, which CV."""
    status = sent_cv.get("status", "unknown")
    if sent_cv.get("text"):
        cv = f"CV read for the talk tracks: {sent_cv.get('label')}."
    elif status == "own":
        cv = ("Sent CV: unknown — the candidate sent a different version, so talk tracks use the candidate's verified "
              "metrics, not the lines the interviewer has.")
    else:
        cv = ("Sent CV: unknown — no CV file on record, so talk tracks use the candidate's verified "
              "metrics, not the lines the interviewer has.")
    return f"---\n*Prepared {today} for the {stage} stage. {cv}*\n"


def _drop_preamble(sheet: str) -> str:
    """The sheet starts with the 15-minute review: a model preamble or title
    above it is dropped. Only text with no ## section of its own — a section
    the model put before the review is kept, never deleted."""
    span = _section_span(sheet, r"15-minute review")
    if not span or span[0] == 0:
        return sheet
    head = sheet[:span[0]]
    if re.search(r"^\s{0,3}##\s", head, re.MULTILINE):
        return sheet
    if head.strip():
        logger.info(f"interview_prep: dropped a {len(head.strip())}-char preamble above the review")
    return sheet[span[0]:].lstrip()


def _below_floor(sheet: str) -> bool:
    return not sheet or len(sheet) < _MIN_SHEET_CHARS or "## " not in sheet


def build_prep(company: str, title: str, description: str = "", location: str = "",
               company_intel: str = "", transcript_context: str = "",
               model: str = None, stage: str = None, sent_cv: dict = None) -> Optional[str]:
    """Synthesize the prep sheet for ONE stage (one premium LLM call), then run
    the deterministic checks. Returns markdown, or None on failure — including
    a failure of the checks themselves (an unchecked sheet is never returned).

    stage: 'Screening Call' | 'Interview' | 'Final Interview' | 'Offer'
    (anything else → 'Interview'). sent_cv: the dict from _sent_cv_for, or
    None when the sent version is unknown. call_llm is imported lazily so
    tests patch gpt_filter.call_llm."""
    from .gpt_filter import call_llm
    model = model or DEFAULT_GENERATION_MODEL
    stage = normalize_stage(stage)
    sent_cv = sent_cv if isinstance(sent_cv, dict) else dict(_UNKNOWN_CV)
    today = datetime.now().strftime("%Y-%m-%d")
    location_line = f" ({location})" if location else ""
    # Posting text is untrusted DATA: planted "if you are an AI…" sentences are
    # stripped before the model sees it.
    posting = clean_jd(str(description or "")).strip()
    has_posting = bool(posting)
    narrative = CANDIDATE_NARRATIVE
    # Prepend the AUTHORITATIVE source bank to the STAR material — the
    # transcript carries the stories, the bank carries the facts.
    from .cv_tailor import _load_source_bank
    source_bank = _fit(_load_source_bank(), _BUDGET_SOURCE_BANK, "source bank")
    transcript = _fit(transcript_context or "(transcript unavailable)",
                      _BUDGET_TRANSCRIPT, "transcript")
    intel = _fit(company_intel, _BUDGET_COMPANY_INTEL, "company intel")
    star_material = ("=== CAREER SOURCE BANK (AUTHORITATIVE — on any conflict, this wins) ===\n"
                     + source_bank
                     + "\n\n=== INTERVIEW TRANSCRIPT (stories) ===\n"
                     + transcript)
    sections = _sections_for(stage, has_posting)
    prompt = PROMPT_INTERVIEW_PREP.format(
        today=today,
        untrusted_jd_clause=UNTRUSTED_JD_CLAUSE,
        stage=stage, stage_brief=_STAGE_BRIEFS[stage],
        job_title=title or "(role)", company=company or "(company)",
        location_line=location_line,
        jd_text=_fit(posting, _BUDGET_JD, "job posting") if has_posting else _NO_POSTING_TEXT,
        company_intel=(intel or "(no company intel gathered — tell the candidate to research before the call)"),
        narrative=_narrative_block(narrative),
        sent_cv_label=sent_cv.get("label") or _UNKNOWN_CV["label"],
        sent_cv=_sent_cv_block(sent_cv),
        metrics_bank=METRICS_BANK_TEXT,
        transcript_context=star_material,
        sections=sections,
    )
    try:
        text = call_llm(prompt, model=model)
    except Exception as e:
        logger.warning(f"interview_prep synthesis failed for {company}/{title}: {e}")
        return None
    sheet = text.strip() if text else ""
    if _below_floor(sheet):
        return None

    try:
        sheet = _drop_preamble(sheet)
        # The candidate's verified sources only: the source bank, METRICS_BANK_TEXT
        # and the candidate's own narrative words. The transcript carries the
        # stories but NOT the facts: only the source bank and the verified
        # metrics prove a number.
        own_text = "\n".join([source_bank, METRICS_BANK_TEXT]
                             + [v for v in (narrative or {}).values() if isinstance(v, str)])
        # The sheet's own scaffolding (header numbers like "15-minute" / "60-90
        # seconds" / "First 90 days", the after-call timings, today's date) is
        # allowed everywhere; it comes from this module, not from the model.
        # The instructions' word caps ("AT MOST 150 words") are deliberately
        # NOT: "I produced 150 videos" must not pass on the strength of a cap.
        scaffold = _numbers("\n".join(
            [l for l in sections.split("\n") if l.lstrip().startswith("#")]
            + [_STAGE_BRIEFS[stage], today, _SCAFFOLD_ECHOES]))
        own_numbers = _numbers(own_text) | scaffold
        company_numbers = _numbers("\n".join([posting, intel, title or "", company or "",
                                              location or ""]))
        sheet, removed = _check_sheet(sheet, own_numbers, company_numbers, own_text, narrative,
                                      scaffold_numbers=scaffold)
        if _below_floor(sheet):       # the checks left too little to call a sheet
            logger.warning(f"interview_prep: sheet for {company}/{title} fell below the "
                           f"quality floor after the checks — not shipped")
            return None
        after_call_hit = any(re.search(r"after the call|thank|check-?in", where, re.IGNORECASE)
                             for where, _, _ in removed)
        sheet = _ensure_why_looking(sheet, narrative)
        sheet = _ensure_after_call(sheet, title, company, after_call_hit)
        sheet = _fix_question_tags(sheet, has_posting)
        sheet = _cap_review(sheet)
        if removed:
            logger.info(f"interview_prep: {len(removed)} sentence(s) removed by the checks "
                        f"for {company}/{title}")
        return sheet.rstrip("\n") + "\n\n" + _provenance_line(stage, today, sent_cv)
    except Exception as e:
        # Never ship a sheet the checks could not finish.
        logger.error(f"interview_prep checks failed for {company}/{title} — no sheet shipped: {e}")
        return None


def _gather_company_intel(company: str, title: str, db=None) -> str:
    """Assemble the company-intel block from cached Perplexity sources.

    Deliberately FAST — no last30days pulse. Interview prep is time-sensitive
    ("interview tomorrow"), so it must return in ~20-40s, not the ~8 min a pulse
    subprocess takes. The 30-day pulse already lives in the outreach playbook on
    the same card, so no recent-news signal is lost — it's one panel up. Every
    step degrades to empty; the sheet still generates without intel."""
    parts = []
    try:
        from .perplexity_utils import get_company_profile, get_hiring_intel
        prof = get_company_profile(company, db=db)
        if prof:
            parts.append(f"COMPANY PROFILE:\n{prof}")
        hiring = get_hiring_intel(company, title, db=db)
        if hiring:
            parts.append(f"HIRING TEAM / DEPARTMENT:\n{hiring}")
    except Exception as e:
        logger.warning(f"interview_prep company intel failed: {e}")
    return "\n\n".join(parts)


def _transcript_for(title: str, keyword_category: str = "") -> str:
    """Domain-filtered transcript slice, WITH the proudest-project STAR
    stories — for_cover_letter=True is the only way to include that section,
    and those stories are the strongest interview material."""
    from .cv_tailor import _load_transcript_context
    # keyword_category may ALREADY be a domain key (e.g. "youtube_and_creators");
    # only fall through to title/keyword resolution otherwise (the two-path trap).
    if keyword_category and keyword_category in DOMAIN_CV_MAP:
        domain = keyword_category
    else:
        domain = resolve_cv_domain(job_title=title, matched_keyword=keyword_category or None)
    return _load_transcript_context(domain=domain, for_cover_letter=True)


def save_prep(app_id: int, company: str, prep: str, stage: str = None) -> str:
    """Write the prep markdown under INTERVIEW_PREP_DIR, return the path. A
    staged sheet carries the stage in its file name, so the sheet for an
    earlier round is not overwritten by the next one."""
    day_dir = os.path.join(INTERVIEW_PREP_DIR, datetime.now().strftime("%Y%m%d"))
    os.makedirs(day_dir, exist_ok=True)
    suffix = f"_{_safe_slug(stage).lower()}" if stage else ""
    path = os.path.join(day_dir, f"interview_{_safe_slug(company)}_{app_id}{suffix}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(prep)
    return path


def generate_interview_prep(app_id: int, db, stage: str = None) -> dict:
    """Full pipeline for one application: gather sources → synthesize → check
    → save → stamp applications.prep_*. Same shape/contract as
    generate_outreach_intel.

    stage: the interview stage to prep for; defaults to the card's current
    status (anything that is not an interview stage → 'Interview'). Returns
    {"app_id", "path", "company_intel", "stage", "sent_cv"} or {"error"}."""
    row = db.get_application(app_id)
    if not row:
        db.update_application_prep(app_id, "failed")
        return {"error": f"application {app_id} not found"}

    company = row.get("company") or ""
    title = row.get("title") or ""
    stage = normalize_stage(stage or row.get("status"))
    logger.info(f"Interview prep: starting for app {app_id} ({title} @ {company}, {stage})")

    company_intel = _gather_company_intel(company, title, db=db)
    transcript = _transcript_for(title, row.get("keyword_category") or "")
    sent_cv = _sent_cv_for(row, db)

    prep = build_prep(company, title, row.get("description") or "",
                      row.get("location") or "", company_intel, transcript,
                      stage=stage, sent_cv=sent_cv)

    # Quality floor: 'ready' must mean a real sheet, not a refusal/truncation.
    if not prep or len(prep) < 500 or "## " not in prep:
        db.update_application_prep(app_id, "failed")
        return {"error": "interview-prep synthesis failed or below quality floor"}

    path = save_prep(app_id, company, prep, stage=stage)
    db.update_application_prep(app_id, "ready", path)
    logger.info(f"Interview prep: ready for app {app_id} → {path}")
    return {"app_id": app_id, "path": path, "company_intel": bool(company_intel),
            "stage": stage, "sent_cv": sent_cv["status"]}
