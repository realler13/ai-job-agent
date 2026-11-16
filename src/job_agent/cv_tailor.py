"""CV tailoring and review: parse a domain base CV (.docx), tailor its summary
and bullets to one posting, write the result back in place, and run the review
passes (truth judge, length gate, keyword coverage, hiring-manager grader,
cover-letter truth check). Everything about the candidate comes from the
profile's files and verified metrics; nothing personal is written here."""
import json
import os
import re
import time
import logging
from typing import List, Dict, Optional
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from copy import deepcopy
import google.generativeai as genai
from .config import (GEMINI_MAX_RETRIES, GEMINI_INITIAL_BACKOFF,
                     TRUNCATION_JOB_DESC_CV, TRUNCATION_PROFILE_SUMMARY,
                     TRUNCATION_PROFILE_EXPERIENCE, DEFAULT_GENERATION_MODEL,
                     detect_model_family, DOMAIN_DESCRIPTIONS,
                     TRANSCRIPT_CONTEXT_PATH, TRUNCATION_TRANSCRIPT_CONTEXT,
                     CV_QUALITY_MODEL, CV_QUALITY_MAX_RETRIES,
                     METRICS_BANK_TEXT, CV_SINGLE_CALL_GENERATION)
from .gpt_filter import call_llm
from .jd_sanitizer import clean_jd, UNTRUSTED_JD_CLAUSE
from datetime import date

logger = logging.getLogger(__name__)

# The posting is untrusted DATA (Phase 27): every prompt that embeds one runs it
# through clean_jd first and carries this notice right next to it. Braces are
# escaped because the notice is spliced into str.format templates.
_UNTRUSTED_FMT = UNTRUSTED_JD_CLAUSE.replace("{", "{{").replace("}", "}}")
_UNTRUSTED_XML = f"<untrusted_data_notice>{_UNTRUSTED_FMT}</untrusted_data_notice>"
_UNTRUSTED_PLAIN = f"NOTE: {_UNTRUSTED_FMT}"


def _strip_markdown(text: str) -> str:
    """Strip common markdown formatting from LLM output for clean DOCX insertion."""
    if not text:
        return text
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    # Italic: *text* or _text_ (single, not inside words)
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', text)
    # Headers: # Header
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Inline code: `code`
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text

# === Interview Transcript Loader ===
_transcript_cache: Optional[str] = None


# Domain -> the transcript SECTIONs most relevant to that CV variant.
# Section numbers follow the interview-notes template (1 General, 7 Skills &
# Tools, 8 Cover Letter Stories). Sections 1 and 7 are always included;
# section 8 is cover-letter only (for_cover_letter=True).
_TRANSCRIPT_ALWAYS_SECTIONS = [1, 7]
_TRANSCRIPT_COVER_LETTER_SECTIONS = [1, 7, 8]
_TRANSCRIPT_DOMAIN_SECTIONS = {
    "producer_core": [2, 4],
    "youtube_and_creators": [2, 5],
    "podcast_and_audio": [3, 2],
    "live_and_experiential": [4, 5],
    "production_ops_and_management": [4, 2],
    "project_and_program_management": [4, 2],
    "games_production": [2, 5],
    "production_admin": [3, 6],
}


def _split_transcript_sections(text: str) -> dict:
    """Split the structured transcript into {section_number: full_text}."""
    headers = list(re.finditer(r"(?m)^## SECTION (\d+).*$", text))
    sections = {}
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections[int(m.group(1))] = text[m.start():end].strip()
    return sections


def _load_transcript_context(domain: str = "producer_core", for_cover_letter: bool = False) -> str:
    """Load interview-transcript context, selecting sections by CV domain.

    Includes the General/Skills sections plus the sections most relevant to
    *domain*, within TRUNCATION_TRANSCRIPT_CONTEXT. Section 8 (Cover Letter
    Stories) is included ONLY when *for_cover_letter* is True; the CV path
    excludes it. Falls back to head-truncation if the
    file has no section headers.
    """
    global _transcript_cache

    if _transcript_cache is None:
        # Project root: two levels up from src/job_agent/cv_tailor.py
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        transcript_path = os.path.join(project_root, TRANSCRIPT_CONTEXT_PATH)
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                _transcript_cache = f.read()
            logger.info(f"Loaded transcript context from {transcript_path} ({len(_transcript_cache)} chars)")
        except FileNotFoundError:
            logger.info(f"Transcript file not found at {transcript_path}, using fallback")
            return "No additional transcript context provided."
        except Exception as e:
            logger.warning(f"Failed to read transcript context: {e}")
            return "No additional transcript context provided."

    domain_desc = DOMAIN_DESCRIPTIONS.get(domain, "general production")
    sections = _split_transcript_sections(_transcript_cache)

    if not sections:
        # No recognizable section headers — legacy head-truncation
        text = _transcript_cache[:TRUNCATION_TRANSCRIPT_CONTEXT]
        return f"[Focus on {domain_desc} experience in this transcript.]\n{text}"

    always = _TRANSCRIPT_COVER_LETTER_SECTIONS if for_cover_letter else _TRANSCRIPT_ALWAYS_SECTIONS
    # Section 8 (cover-letter stories) must never reach the CV path, not even
    # via the remaining-budget fill below.
    excluded = set() if for_cover_letter else {8}

    order = list(always[:1])                                     # Section 1 first
    order += _TRANSCRIPT_DOMAIN_SECTIONS.get(domain, [2, 4])      # domain sections
    order += always[1:]                                          # Skills (+ CL stories)
    # Fill any remaining budget with the rest, in numeric order
    order += [n for n in sorted(sections) if n not in order and n not in excluded]

    chosen, used = [], 0
    seen = set()
    for n in order:
        if n in seen or n not in sections:
            continue
        seen.add(n)
        block = sections[n]
        if used + len(block) > TRUNCATION_TRANSCRIPT_CONTEXT and chosen:
            continue  # skip sections that don't fit; keep trying smaller ones
        chosen.append(block)
        used += len(block)

    text = "\n\n".join(chosen)
    return f"[Focus on {domain_desc} experience in this transcript.]\n{text}"

# === Career Source Bank loader (2026-07-16) ===
_source_bank_cache: Optional[str] = None


def _load_source_bank() -> str:
    """Load the candidate's canonical career source bank.
    AUTHORITATIVE for facts/numbers/dates/titles — wins over
    the transcript on any conflict. Cached at module level; degrades to an
    explicit marker string if the file is missing (generation still works,
    grounded on metrics bank + base CV)."""
    global _source_bank_cache
    if _source_bank_cache is None:
        from .config import SOURCE_BANK_PATH, TRUNCATION_SOURCE_BANK
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(project_root, SOURCE_BANK_PATH)
        try:
            with open(path, "r", encoding="utf-8") as f:
                _source_bank_cache = f.read()[:TRUNCATION_SOURCE_BANK]
            logger.info(f"Loaded career source bank from {path} ({len(_source_bank_cache)} chars)")
        except FileNotFoundError:
            logger.warning(f"Source bank not found at {path} — run scripts/build_source_bank.py")
            _source_bank_cache = "(source bank unavailable — ground on verified metrics and base CV only)"
        except Exception as e:
            logger.warning(f"Failed to read source bank: {e}")
            _source_bank_cache = "(source bank unavailable — ground on verified metrics and base CV only)"
    return _source_bank_cache


# === Anti-Slop & Attribution Rules (Phase 15) ===

_ANTI_SLOP_RULES = """
MANDATORY ANTI-SLOP REVIEW — apply before finalizing every section:
- One action per bullet. If a bullet has more than one comma-separated accomplishment, split it.
- The breath test: if any sentence in the Professional Summary takes more than one breath to read aloud, shorten it.
- Professional Summary: max 3-4 sentences, each under 20 words, each names at most 2 things.
- No bullet over 2 lines — split it.
- No gerund clauses tacked on: "...delivering X while maintaining Y" → two sentences.
- No preamble clauses: "Leveraging deep expertise in content, led..." → "Led..."

KILL LIST — eliminate every instance before output:
- "demonstrating ability to" / "showcasing expertise in" / "with a track record of"
- "needed to drive" / "spanning" / "leveraging" / "utilizing"
- "cross-functional, data-driven, audience-centric" piles → pick the one relevant word
- "Led X, Y, and Z across [scope]" → three separate bullets
- Compound noun piles like "AI-driven content production opportunity identification"
- Emdashes (—) in any CV or cover letter output — use commas, periods, colons, or parentheses instead.
"""

_ATTRIBUTION_LOCK = """
COMPANY-ATTRIBUTION LOCK — cover letter accuracy rules:
- For every sentence that names a company, verify the metric/achievement cited exists under that company in the candidate's CV (same employer AND date range). Never merge metrics from different employers in one sentence.
- Roles with end date "Present" → current role → present tense allowed.
- Roles with any other end date → past role → must use past tense.
- Never use "in my current role" for a role that has ended.
- Never extend any role's duration beyond what the CV explicitly states.
- Never inflate seniority, scope, or impact beyond what is documented.
- If a detail is ambiguous, omit it — do not assume.
"""


# === PROMPTS ===
# Non-Claude (Gemini/OpenAI/Ollama) variants — plain text with === HEADER === style.
# These include {domain_context} and {transcript_context} placeholders.

PROMPT_SUMMARY_TEMPLATE = """
Role: Executive Resume Strategist
Task: Write a 3-4 sentence Professional Summary for the candidate that targets the Job Description (JD) below.

=== CV DOMAIN CONTEXT ===
{domain_context}

=== TARGET JOB ===
Title: {job_title}
Company: {company}
""" + _UNTRUSTED_PLAIN + """
Key Keywords & Goals: {jd_text}

=== CANDIDATE FULL PROFILE (CONTEXT) ===
{full_profile}

=== CURRENT SUMMARY (DRAFT FROM DOMAIN CV) ===
{original_summary}

=== ADDITIONAL CANDIDATE CONTEXT ===
{transcript_context}

=== INSTRUCTIONS ===
1. The base CV is already domain-optimized. Refine and sharpen it for THIS specific job, do not revert to generic language.
2. The first sentence must state the candidate's core identity as it relates to the target role (e.g., "Documentary Series Producer" not just "Producer").
3. Use specific keywords from the JD verbatim (e.g., if JD says "Cross-functional leadership," use that exact phrase). BUT keyword mirroring applies to how you DESCRIBE the candidate's real work, never to WHAT work you claim: every activity or format named as the candidate's experience must be real work from the sources. Do NOT import the JD's activity nouns (a format named only in the posting) into the candidate's experience claims — if the JD's format is one the candidate hasn't done, name the closest real format the candidate HAS done.
4. Preserve the domain framing from the base CV while weaving in JD-specific terminology.
5. Tone: Senior, authoritative, results-driven. No fluff.

=== VERIFIED METRICS ===
{metrics_bank}

=== CRITICAL CONSTRAINT ===
Use ONLY skills, experiences, achievements, company names, dates, and qualifications that are explicitly present in the Candidate Full Profile, Current Summary, or Additional Candidate Context above. Do NOT invent, fabricate, or assume any information not found in the source materials. Every claim must be traceable to the input data. If the profile lacks relevant experience for a JD requirement, omit it rather than guessing.

{anti_slop_rules}

Output: A single paragraph of plain text. No markdown, no bold (**), no italics (*), no headers (#), no bullet points. Just clean prose.
"""

PROMPT_EXPERIENCE_TEMPLATE = """
Role: Ruthless Headhunter
Task: Rewrite the bullet points for the role of "{role_title}" to prove the candidate is perfect for the Target Job below.

=== CV DOMAIN CONTEXT ===
{domain_context}

=== TARGET JOB ===
Title: {job_title}
""" + _UNTRUSTED_PLAIN + """
Key Requirements: {jd_text}

=== ORIGINAL EXPERIENCE BULLETS (FROM DOMAIN CV) ===
{original_bullets}

=== FULL CANDIDATE CONTEXT (REFERENCE) ===
{full_profile}

=== ADDITIONAL CANDIDATE CONTEXT ===
{transcript_context}

=== EXECUTION RULES (STRICT) ===
1. The base CV bullets are already domain-optimized. Sharpen them for THIS specific JD, do not revert to generic language.
2. Vocabulary Swap: Replace the candidate's jargon with the JD's exact keywords and phrases.
3. The Cut: DELETE any bullet point that is irrelevant to the Target Job. Quality > Quantity.
4. Quantify: Keep numbers (budget size, team size, output) but frame them as evidence of the JD's requirements.
5. Length Constraint: MAX 3-4 bullet points per role. Short, punchy sentences. No multi-sentence bullets.

=== VERIFIED METRICS ===
{metrics_bank}

=== CRITICAL CONSTRAINT ===
Every bullet point must be grounded in facts from the Original Experience Bullets, Full Candidate Context, or Additional Candidate Context above. Do NOT invent accomplishments, metrics, skills, tools, or technologies not present in the source materials. You may reframe and rephrase existing achievements using JD terminology, but never fabricate new ones. Reframing means changing vocabulary, not changing scope or scale — "coordinated vendor deliverables" can become "managed vendor relationships" but cannot become "built and led a vendor management program." Do not inflate team sizes, budget figures, or the candidate's level of ownership beyond what the source materials state. The NOUN naming what the candidate produced is scope, not vocabulary: a deliverable keeps the type the source gives it, and numbers stay with the work they belong to. Keep the deliverable type exactly as the source states it, even when the JD uses a different word.

{anti_slop_rules}

Output: 3-4 high-impact bullet points. Plain text, one per line (start with "- "). No markdown.
"""

# === Claude-Optimized Variants ===
# XML-tagged for optimal Claude structured parsing.
# {domain_context} tells the model which domain variant is being used.
# {transcript_context} carries the interview-notes context.

PROMPT_SUMMARY_TEMPLATE_CLAUDE = """You are an elite executive resume strategist. You will receive a domain-specific base CV that is already optimized for a particular career track. Your job is to further sharpen the Professional Summary for one specific job posting.

<domain_context>
{domain_context}
</domain_context>

<target_job>
<title>{job_title}</title>
<company>{company}</company>
""" + _UNTRUSTED_XML + """
<requirements>{jd_text}</requirements>
</target_job>

<candidate_profile>
{full_profile}
</candidate_profile>

<current_summary_from_domain_cv>
{original_summary}
</current_summary_from_domain_cv>

<transcript_context>
{transcript_context}
</transcript_context>

<instructions>
Write a 3-4 sentence Professional Summary following these rules:

1. The base CV summary is already domain-optimized. Refine it for THIS specific job — do not revert to generic "Producer" language.
2. Open with the candidate's identity framed for this exact role (e.g., "documentary series producer" not "content producer").
3. Mirror the JD's exact terminology — if the JD says "cross-functional program management," use those words. BUT terminology mirroring applies to how you DESCRIBE the candidate's real work, never to WHAT work you claim: every activity/format named as the candidate's experience (e.g., "including X, Y") must be real work from the sources. Do NOT import the JD's activity nouns (a format named only in the posting) into the candidate's experience claims — if the JD's format is one the candidate hasn't done, name the closest real format the candidate HAS done instead.
4. Weave in 2-3 of the candidate's strongest proof points (metrics, scale, outcomes) that directly map to JD requirements.
5. Tone: senior, precise, confident. Every word must earn its place.
</instructions>

<verified_metrics>
{metrics_bank}
</verified_metrics>

<critical_constraint>
ABSOLUTE RULE: Use ONLY facts from the candidate_profile, current_summary_from_domain_cv, and transcript_context. Do NOT invent skills, achievements, metrics, company names, tools, or qualifications. If the candidate lacks experience for a JD requirement, omit it entirely — never guess or fabricate. Every claim must be directly traceable to the provided source materials.
</critical_constraint>

<anti_slop_rules>
{anti_slop_rules}
</anti_slop_rules>

<output_format>
A single paragraph of plain text. No markdown formatting whatsoever — no bold, no italics, no headers, no bullets, no asterisks.
</output_format>"""

PROMPT_EXPERIENCE_TEMPLATE_CLAUDE = """You are a ruthless executive headhunter. You will receive experience bullet points from a domain-specific base CV that is already optimized for a particular career track. Your job is to further sharpen these bullets for one specific job posting.

<domain_context>
{domain_context}
</domain_context>

<target_job>
<title>{job_title}</title>
""" + _UNTRUSTED_XML + """
<requirements>{jd_text}</requirements>
</target_job>

<original_bullets_from_domain_cv>
{original_bullets}
</original_bullets_from_domain_cv>

<candidate_profile>
{full_profile}
</candidate_profile>

<transcript_context>
{transcript_context}
</transcript_context>

<instructions>
Rewrite the bullets for the role "{role_title}" following these rules:

1. These bullets are already domain-optimized. Sharpen them for THIS specific JD — do not flatten back to generic language.
2. Mirror the JD's vocabulary: if the JD says "stakeholder management," use that phrase instead of "team coordination."
3. Cut ruthlessly: delete any bullet that does not directly support THIS job's requirements. Quality over quantity.
4. Preserve hard numbers (budgets, team sizes, output volumes, percentage changes) but reframe them as evidence of JD requirements.
5. Maximum 3-4 bullets per role. Each bullet: one sentence, one clear proof point. No compound bullets.
</instructions>

<verified_metrics>
{metrics_bank}
</verified_metrics>

<critical_constraint>
ABSOLUTE RULE: Every bullet must be grounded in facts from the original_bullets_from_domain_cv, candidate_profile, or transcript_context. Do NOT invent accomplishments, metrics, skills, tools, technologies, or certifications. You may reframe and rephrase existing achievements using JD terminology, but never fabricate new ones. Reframing means changing vocabulary, not changing scope or scale — "coordinated vendor deliverables" can become "managed vendor relationships" but cannot become "built and led a vendor management program." Do not inflate team sizes, budget figures, or the candidate's level of ownership beyond what the source materials state. The NOUN naming what the candidate produced is scope, not vocabulary: a deliverable keeps the type the source gives it, and numbers stay with the work they belong to. Keep the deliverable type exactly as the source states it, even when the JD uses a different word. If the source material has no relevant proof point for a JD requirement, omit it.
</critical_constraint>

<anti_slop_rules>
{anti_slop_rules}
</anti_slop_rules>

<output_format>
3-4 bullet points. Plain text, one per line, each starting with "- ". No markdown formatting.
</output_format>"""

# Consolidated single-call CV prompt: tailors the Professional Summary AND every
# role's bullets in ONE pass, returning structured JSON. Carries the SAME
# guardrails as the per-section prompts (anti-hallucination, reframing-not-
# inflating, anti-slop, verified metrics, domain framing) so quality is held
# while the shared context is sent once instead of once per section.
PROMPT_CV_CONSOLIDATED_CLAUDE = """You are an elite executive resume strategist and ruthless headhunter. Tailor a domain-specific base CV (already optimized for a career track) to ONE specific job posting, in a single pass: the Professional Summary AND the bullets for EVERY listed role.

<domain_context>
{domain_context}
</domain_context>

<target_job>
<title>{job_title}</title>
<company>{company}</company>
""" + _UNTRUSTED_XML + """
<requirements>{jd_text}</requirements>
</target_job>

<candidate_profile>
{full_profile}
</candidate_profile>

<career_source_bank>
AUTHORITATIVE fact source (the candidate's career source bank). On ANY conflict with candidate_profile or transcript_context, THIS BLOCK WINS — it is newer and verified by the candidate.
{source_bank}
</career_source_bank>

<transcript_context>
{transcript_context}
</transcript_context>

<current_summary_from_domain_cv>
{original_summary}
</current_summary_from_domain_cv>

<roles_to_rewrite>
{roles_block}
</roles_to_rewrite>

<verified_metrics>
{metrics_bank}
</verified_metrics>

<critical_constraint>
ABSOLUTE RULE: Use ONLY facts from career_source_bank (highest precedence), candidate_profile, current_summary_from_domain_cv, the role originals in roles_to_rewrite, and transcript_context. Do NOT invent skills, achievements, metrics, company names, tools, or qualifications. Reframing means changing vocabulary, NOT scope or scale — "coordinated vendor deliverables" can become "managed vendor relationships" but never "built and led a vendor management program." Do not inflate team sizes, budgets, or ownership beyond what the source states. The NOUN naming what the candidate produced is scope, not vocabulary: a deliverable keeps the type the source gives it, and numbers stay with the work they belong to. Keep the deliverable type exactly as the source states it, even when the JD uses a different word. If the source has no relevant proof point for a JD requirement, omit it.
</critical_constraint>

<anti_slop_rules>
{anti_slop_rules}
</anti_slop_rules>

<summary_guardrails>
- VOICE: a résumé summary, not a cover letter. State capability and proof. NO first-person motivational or aspirational lines — never "I am excited", "I love", "I have always wanted", "my dream is", "passionate about", or any sentence about what the candidate wants/feels. Those belong in the cover letter, never here.
- TITLE/LEVEL: keep the candidate's identity at the level the sources document. Do NOT label the candidate a "manager", "director", "head of", "leader", or "people leader" in the summary unless the career_source_bank or the base CV gives the candidate that title — even if the target job carries it. Mirroring the JD's vocabulary does NOT extend to inflating the candidate's title or level.
- YEARS: state the candidate's experience exactly as the canonical years figure in verified_metrics. Never compute or state a different number.
- EMPLOYMENT STATUS: follow verified_metrics. A role with an end date there is a PAST role: past tense only; never "currently", never "Present".
</summary_guardrails>

<instructions>
1) SUMMARY — write a 3-4 sentence Professional Summary refined for THIS job, obeying summary_guardrails: keep the domain framing (do not revert to generic "Producer" language), open with the candidate's real-level identity framed for this role, mirror the JD's terminology for the WORK (not the title/level), and weave in 2-3 of the strongest proof points that map to the JD. Plain text, no markdown.
2) ROLES — for EACH role in roles_to_rewrite, rewrite its bullets sharpened for THIS JD: mirror the JD's vocabulary, cut any bullet that doesn't support this job, preserve hard numbers reframed as JD evidence, max 3-4 bullets per role, one sentence each, no compound bullets. Each bullet starts with "- ". Plain text, no markdown.
3) MIX FROM THE SOURCE BANK: when the JD demands experience that the role's current bullets don't show but the career_source_bank DOES contain for that role's era (check the bank's attributions), you may REPLACE a weaker bullet with a bullet built from those bank lines — real work, the candidate's wording professionalized, JD vocabulary applied. This maximizes relevant experience per JD without inventing anything. Never import a bank line into a role/era it isn't attributed to.
4) Keep each role's "index" exactly as given so the bullets map back to the right role.
</instructions>

<output_format>
Return ONLY a single JSON object — no prose, no code fences, no markdown:
{{"summary": "<one paragraph>", "roles": [{{"index": <int>, "bullets": "- ...\\n- ..."}}]}}
Include exactly one entry per role in roles_to_rewrite, each with its original index.
</output_format>"""

# === PARSER ===

# The section headers a base CV may use. A paragraph is a section header only
# when its WHOLE stripped text is one of these AND it is not a list/bullet
# paragraph. The earlier substring test ("SKILLS" in text) let any bullet or
# company line that merely contains one of these words end the Experience
# section mid-CV.
_SECTION_HEADERS = {
    "PROFESSIONAL SUMMARY": "SUMMARY",
    "SUMMARY": "SUMMARY",
    "PROFESSIONAL EXPERIENCE": "EXPERIENCE",
    "EXPERIENCE": "EXPERIENCE",
    # Every other section is static template content — never parsed as a
    # work role, never rewritten by tailoring.
    "VOLUNTEER EXPERIENCE": "OTHER",
    "SKILLS": "OTHER",
    "EDUCATION": "OTHER",
}
_BULLET_CHARS = ('•', '-', '*', '⁃')


def _is_list_paragraph(p) -> bool:
    """True for a list/bullet paragraph: a 'List…' style, paragraph-level
    numbering (w:numPr), or a literal leading bullet character."""
    try:
        if p.style.name.startswith('List'):
            return True
    except Exception:
        pass
    pPr = p._element.pPr
    if pPr is not None and pPr.find(qn('w:numPr')) is not None:
        return True
    return p.text.strip().startswith(_BULLET_CHARS)


def _section_header(p) -> Optional[str]:
    """SUMMARY / EXPERIENCE / OTHER when *p* is a real section header, else None."""
    if _is_list_paragraph(p):
        return None
    return _SECTION_HEADERS.get(" ".join(p.text.split()).upper())


def _segment_cv(doc) -> Dict:
    """Walk the body once and return the paragraphs the parser reads and the
    writer rewrites:

        {"summary": [p, ...], "entries": [{"header": p, "bullets": [p, ...]}, ...]}

    ONE segmentation shared by parse_docx_to_structure, the writer and the
    round-trip check, so they can never disagree about which bullets belong to
    which role. Convention (unchanged): inside EXPERIENCE every non-bullet line
    opens a new entry, so each real role yields TWO entries — an empty title
    placeholder ("Producer") and the company/location/date line that carries
    the bullets."""
    seg = {"summary": [], "entries": []}
    section = None
    entry = None
    for p in doc.paragraphs:
        if not p.text.strip():
            continue
        header = _section_header(p)
        if header is not None:
            section = header
            entry = None
            continue
        if section == "SUMMARY":
            seg["summary"].append(p)
        elif section == "EXPERIENCE":
            if _is_list_paragraph(p):
                # A bullet before any role header has no owner — skipped.
                if entry is not None:
                    entry["bullets"].append(p)
            else:
                entry = {"header": p, "bullets": []}
                seg["entries"].append(entry)
    return seg


def parse_docx_to_structure(file_path: str) -> Dict:
    """
    Parses the base CV DOCX to extract Summary and Experience sections.
    Returns:
        {
            "summary": "...",
            "experience": [
                {"title": "...", "description": "...", "company": ""},
                ...
            ]
        }
    Each real role is TWO entries (see _segment_cv): the title line with an
    empty description, then the company/date line whose description holds the
    newline-joined bullets.
    """
    if not os.path.exists(file_path):
        logger.error(f"Base CV not found at: {file_path}")
        return {"summary": "", "experience": []}

    try:
        doc = Document(file_path)
    except Exception as e:
        logger.error(f"Failed to load DOCX: {e}")
        return {"summary": "", "experience": []}

    seg = _segment_cv(doc)
    return {
        "summary": "\n".join(p.text.strip() for p in seg["summary"]),
        "experience": [
            {
                "title": entry["header"].text.strip(),
                "company": "",
                "description": "\n".join(b.text.strip() for b in entry["bullets"]),
            }
            for entry in seg["entries"]
        ],
    }


def get_summary_prompt(model: str = None) -> str:
    """Return the best summary prompt template for the given model."""
    if model and detect_model_family(model) == "claude":
        return PROMPT_SUMMARY_TEMPLATE_CLAUDE
    return PROMPT_SUMMARY_TEMPLATE

def get_experience_prompt(model: str = None) -> str:
    """Return the best experience prompt template for the given model."""
    if model and detect_model_family(model) == "claude":
        return PROMPT_EXPERIENCE_TEMPLATE_CLAUDE
    return PROMPT_EXPERIENCE_TEMPLATE

def _build_domain_context(domain: str) -> str:
    """Build human-readable domain context string for prompt injection."""
    desc = DOMAIN_DESCRIPTIONS.get(domain, "general production")
    return (
        f"This CV uses the '{domain}' variant, optimized for {desc}. "
        f"The base CV already emphasizes {desc} experience and terminology. "
        f"Preserve this domain framing while tailoring for the specific job."
    )

def _generate_cv_consolidated(
    candidate_profile: Dict,
    target_job: Dict,
    profile_text: str,
    domain_context: str,
    transcript_context: str,
    model_name: str,
) -> Optional[Dict]:
    """Tailor the whole CV (summary + all role bullets) in ONE LLM call.

    Returns a complete ``tailored_data`` dict, or None if the call fails or the
    response is malformed/incomplete — in which case the caller falls back to
    the per-section path so quality is never silently degraded.
    """
    experience = candidate_profile.get("experience", [])
    # Only roles with real bullets are rewritten; index = position in the
    # experience list, so results map back to the exact role dict.
    sendable = [(i, r) for i, r in enumerate(experience) if str(r.get("description", "")).strip()]
    if not sendable:
        return None

    roles_block = ""
    for i, role in sendable:
        roles_block += (
            f'<role index="{i}" title="{role.get("title", "")}">\n'
            f'{role.get("description", "")}\n</role>\n'
        )

    prompt = PROMPT_CV_CONSOLIDATED_CLAUDE.format(
        domain_context=domain_context,
        job_title=target_job.get("title", ""),
        company=target_job.get("company", ""),
        jd_text=target_job.get("description", "")[:TRUNCATION_JOB_DESC_CV],
        full_profile=profile_text[:TRUNCATION_PROFILE_EXPERIENCE],
        source_bank=_load_source_bank(),
        transcript_context=transcript_context,
        original_summary=candidate_profile.get("summary", ""),
        roles_block=roles_block,
        metrics_bank=METRICS_BANK_TEXT,
        anti_slop_rules=_ANTI_SLOP_RULES,
    )

    answer = call_llm(prompt, model=model_name)
    if not answer:
        return None

    # Parse JSON via raw_decode from the first "{" — tolerates code fences and
    # trailing prose (same robust pattern used at cascade Stage 3).
    start = answer.find("{")
    if start == -1:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(answer[start:])
    except Exception:
        return None

    summary = data.get("summary")
    roles = data.get("roles")
    if not isinstance(summary, str) or not summary.strip() or not isinstance(roles, list):
        return None

    bullets_by_index: Dict[int, str] = {}
    for r in roles:
        if isinstance(r, dict) and isinstance(r.get("bullets"), str) and "index" in r:
            try:
                bullets_by_index[int(r["index"])] = r["bullets"]
            except (TypeError, ValueError):
                continue
    # Require every sent role to come back — otherwise fall back for full quality.
    if any(i not in bullets_by_index for i, _ in sendable):
        return None

    tailored = {"summary": _strip_markdown(summary), "experience": []}
    for i, role in enumerate(experience):
        new_role = role.copy()
        if i in bullets_by_index:
            new_role["description"] = _strip_markdown(bullets_by_index[i])
        tailored["experience"].append(new_role)
    return tailored


def generate_tailored_cv(
    base_cv_path: str,
    profile_text: str,
    target_job: Dict,
    model_name=DEFAULT_GENERATION_MODEL,
    summary_prompt_template: str = None,
    experience_prompt_template: str = None,
    domain: str = "producer_core",
    transcript_text: str = "",
) -> Dict:
    """
    Generates a tailored CV by parsing a domain-specific DOCX, using LLMs
    to rewrite sections, and returning the structure.

    *domain*: key from DOMAIN_CV_MAP (e.g. "games_production").
    *transcript_text*: optional deep-dive interview transcript for richer context.
    """
    # Coerce description once for BOTH paths: job_details merges raw_data JSON
    # where description is frequently None (LinkedIn fetch failures) or float
    # NaN (pandas NaN round-tripped through json) — slicing those crashes.
    _desc = target_job.get("description")
    if _desc is None or isinstance(_desc, float):
        _desc = ""
    # The posting is untrusted data: AI-directed and applicant-instruction
    # sentences are stripped before it reaches either generation path.
    target_job = {**target_job, "description": clean_jd(str(_desc))}

    # Auto-select prompts for model family if not explicitly provided
    if summary_prompt_template is None:
        summary_prompt_template = get_summary_prompt(model_name)
    if experience_prompt_template is None:
        experience_prompt_template = get_experience_prompt(model_name)

    # Build domain and transcript context strings
    domain_context = _build_domain_context(domain)
    if transcript_text and transcript_text.strip():
        transcript_context = transcript_text.strip()
    else:
        transcript_context = _load_transcript_context(domain=domain)

    # 1. Parse Candidate Data
    logger.info(f"Parsing base CV: {base_cv_path} (domain: {domain})")
    candidate_profile = parse_docx_to_structure(base_cv_path)

    if not candidate_profile['summary'] and not candidate_profile['experience']:
        logger.warning("Parser found no data! Tailoring will be empty.")

    tailored_data = {
        "summary": candidate_profile['summary'],
        "experience": []
    }

    # Fast path: tailor the whole CV in ONE structured call (sends the shared
    # context once instead of ~6x). Falls through to the per-section path below
    # if it fails or returns an incomplete/malformed response.
    if CV_SINGLE_CALL_GENERATION:
        try:
            consolidated = _generate_cv_consolidated(
                candidate_profile, target_job, profile_text,
                domain_context, transcript_context, model_name,
            )
        except Exception as e:
            # The promised automatic fallback must hold for EXCEPTIONS too,
            # not just None returns — never let the fast path kill a CV.
            logger.warning(f"Consolidated CV generation raised {e!r} — falling back")
            consolidated = None
        if consolidated is not None:
            logger.info(f"CV generated via single consolidated call (domain: {domain})")
            return consolidated
        logger.warning("Consolidated CV generation failed/incomplete — falling back to per-section calls")

    # 2. Rewrite Summary
    logger.info(f"Generating Tailored Summary using {model_name} (domain: {domain})...")

    # The per-section templates have no dedicated source-bank block (they are
    # caller-overridable), so inject the AUTHORITATIVE bank through the
    # transcript channel every template already carries. Bank first — it wins.
    grounded_ctx = ("=== CAREER SOURCE BANK (AUTHORITATIVE — on any conflict, this wins) ===\n"
                    + _load_source_bank()
                    + "\n\n=== INTERVIEW TRANSCRIPT (stories/voice) ===\n"
                    + transcript_context)

    summary_prompt = summary_prompt_template.format(
        job_title=target_job.get('title', ''),
        company=target_job.get('company', ''),
        jd_text=target_job.get('description', '')[:TRUNCATION_JOB_DESC_CV],
        full_profile=profile_text[:TRUNCATION_PROFILE_SUMMARY],
        original_summary=candidate_profile['summary'],
        domain_context=domain_context,
        transcript_context=grounded_ctx,
        metrics_bank=METRICS_BANK_TEXT,
        anti_slop_rules=_ANTI_SLOP_RULES,
    )
    new_summary = call_llm(summary_prompt, model=model_name)
    if new_summary:
        tailored_data['summary'] = _strip_markdown(new_summary)

    # 3. Rewrite Experience (Loop)
    logger.info(f"Tailoring {len(candidate_profile['experience'])} experience roles...")
    for role in candidate_profile['experience']:
        original_bullets = role['description']
        if not original_bullets.strip():
            tailored_data['experience'].append(role)
            continue

        role_prompt = experience_prompt_template.format(
            job_title=target_job.get('title', ''),
            jd_text=target_job.get('description', '')[:TRUNCATION_JOB_DESC_CV],
            role_title=role.get('title', ''),
            original_bullets=original_bullets,
            full_profile=profile_text[:TRUNCATION_PROFILE_EXPERIENCE],
            domain_context=domain_context,
            transcript_context=grounded_ctx,
            metrics_bank=METRICS_BANK_TEXT,
            anti_slop_rules=_ANTI_SLOP_RULES,
        )

        new_bullets = call_llm(role_prompt, model=model_name)

        new_role = role.copy()
        if new_bullets:
             new_role['description'] = _strip_markdown(new_bullets)

        tailored_data['experience'].append(new_role)

    return tailored_data

# === WRITER ===

def _norm_ws(text) -> str:
    """Collapse whitespace runs — the comparison form for header lines and bullets."""
    return " ".join(str(text or "").split())


def _bullet_lines(description) -> List[str]:
    """A role's newline-joined description as the clean bullet texts that go
    into the file (leading '-' / '•' markers stripped, blank lines dropped)."""
    cleaned = (line.strip().lstrip('•-*⁃ ').strip() for line in str(description or "").split('\n'))
    return [line for line in cleaned if line]


def _delete_paragraph(paragraph) -> None:
    p = paragraph._element
    p.getparent().remove(p)
    paragraph._p = paragraph._element = None


def _set_paragraph_text(paragraph, text: str) -> None:
    """Replace a paragraph's text in place. Its paragraph properties (style,
    numPr, ind, spacing) are not touched and the first run keeps its character
    formatting; every other child is dropped."""
    keep = paragraph.runs[0]._r if paragraph.runs else None
    for child in list(paragraph._p):
        if child is not keep and child.tag != qn('w:pPr'):
            paragraph._p.remove(child)
    if keep is None:
        paragraph.add_run(text)
    else:
        paragraph.runs[0].text = text


def _clone_paragraph_after(template, anchor):
    """Insert a copy of *template* (same pPr: style, numbering, indent, spacing)
    right after *anchor* and return it."""
    new_p = deepcopy(template._p)
    for attr in list(new_p.attrib):
        # w14:paraId / w14:textId identify ONE paragraph — never duplicate them.
        if attr.endswith('}paraId') or attr.endswith('}textId'):
            del new_p.attrib[attr]
    anchor._p.addnext(new_p)
    return Paragraph(new_p, template._parent)


def _write_lines_in_place(paras: list, lines: List[str]) -> None:
    """Make an existing block of paragraphs (one role's bullets, or the summary
    body) carry exactly *lines*, REUSING the block's own paragraphs so every
    line keeps the base formatting — nothing is restyled and no pPr child is
    appended or duplicated.

    - A paragraph whose text is already right is left untouched (B0: writing a
      base's own bullets back changes nothing).
    - Fewer lines than paragraphs: the surplus is removed from just before the
      last paragraph. More lines: copies of the last-but-one paragraph are
      inserted there. First and last paragraph stay first and last — in the
      legacy base they carry the block's leading/trailing spacing.
    """
    n, m = len(paras), len(lines)
    if not n or not m:
        return
    if m < n:
        for p in paras[m - 1:n - 1]:
            _delete_paragraph(p)
        paras = paras[:m - 1] + [paras[-1]]
    elif m > n:
        template = paras[-2] if n >= 2 else paras[-1]
        added, anchor = [], template
        for _ in range(m - n):
            anchor = _clone_paragraph_after(template, anchor)
            added.append(anchor)
        paras = (paras[:-1] + added + [paras[-1]]) if n >= 2 else (paras + added)
    for p, line in zip(paras, lines):
        current = p.text.strip()
        if current == line or current.lstrip('•-*⁃ ').strip() == line:
            continue
        _set_paragraph_text(p, line)


def fill_docx_with_tailored_data(base_docx_path: str, output_path: str, tailored_data: Dict):
    """
    Overwrites the base DOCX with the tailored data: the summary body and each
    role's bullets. Everything else (section headers, title and company lines,
    SKILLS, EDUCATION and any other section) stays exactly as in the base.

    Entries map by POSITION onto the base's own segmentation (_segment_cv) and
    are written only when the header line matches too. The base content stays
    in place (known-good fallback, with a warning) when:
      - the base entry has no bullets — an empty title placeholder is frozen
        and never receives bullets;
      - the tailored entry's title is not the base header line at that position;
      - the tailored entry, or the summary, is empty.
    Run verify_docx_roundtrip() on the saved file to learn whether everything
    intended actually landed.
    """
    try:
        doc = Document(base_docx_path)
    except Exception as e:
        logger.error(f"Failed to load base DOCX for writing: {e}")
        return

    seg = _segment_cv(doc)

    # --- Summary ---
    summary_lines = [line.strip() for line in str(tailored_data.get('summary') or '').split('\n')
                     if line.strip()]
    if summary_lines and seg["summary"]:
        _write_lines_in_place(seg["summary"], summary_lines)
    elif summary_lines:
        logger.warning("CV writer: base has no summary paragraph — summary not written")

    # --- Experience ---
    # (The summary branch must stop at the EXPERIENCE header; an earlier
    # version did not.)
    roles = tailored_data.get('experience') or []
    entries = seg["entries"]
    if len(roles) != len(entries):
        logger.warning(f"CV writer: {len(roles)} tailored entries vs {len(entries)} in the base — "
                       f"writing only where position and header line agree")
    for idx, (entry, role) in enumerate(zip(entries, roles)):
        lines = _bullet_lines(role.get('description'))
        header = _norm_ws(entry["header"].text)
        if not entry["bullets"]:
            if lines:
                logger.warning(f"CV writer: entry {idx} ({header!r}) is an empty title placeholder — "
                               f"{len(lines)} bullet(s) not written")
            continue
        if _norm_ws(role.get('title')) != header:
            logger.warning(f"CV writer: entry {idx} title does not match the base header line "
                           f"({header!r}) — base bullets kept")
            continue
        if not lines:
            logger.warning(f"CV writer: entry {idx} ({header!r}) has no tailored bullets — base bullets kept")
            continue
        _write_lines_in_place(entry["bullets"], lines)

    doc.save(output_path)


def verify_docx_roundtrip(docx_path: str, tailored_data: Dict, base_docx_path: str = None) -> Dict:
    """Re-open the SAVED .docx, parse it, and compare it with what was intended:
    the summary and, per role, the list of bullets (whitespace-normalised) under
    that role's own header line.

    Returns {"ok": bool, "mismatches": [{"where": ..., "reason": ..., ...}]}.
    With *base_docx_path*, also checks that every paragraph outside the summary
    body and the role bullets (headers, title/company lines, SKILLS,
    EDUCATION and any other section) is unchanged from the base, in order.

    Never raises. A file that cannot be read is a mismatch — an unchecked
    document is never reported ok. On a mismatch the caller ships the base
    bullets/summary instead of the unverified file."""
    try:
        if not os.path.exists(docx_path):
            return {"ok": False, "mismatches": [{"where": "file", "reason": "saved file not found"}]}
        doc = Document(docx_path)
        seg = _segment_cv(doc)
        mismatches = []

        want_summary = _norm_ws(tailored_data.get("summary"))
        got_summary = _norm_ws(" ".join(p.text for p in seg["summary"]))
        if want_summary != got_summary:
            mismatches.append({"where": "summary", "reason": "summary text differs",
                               "expected": want_summary, "actual": got_summary})

        roles = tailored_data.get("experience") or []
        entries = seg["entries"]
        if len(roles) != len(entries):
            mismatches.append({"where": "experience", "reason": "role count differs",
                               "expected": len(roles), "actual": len(entries)})
        for idx, (role, entry) in enumerate(zip(roles, entries)):
            header = _norm_ws(entry["header"].text)
            if _norm_ws(role.get("title")) != header:
                mismatches.append({"where": "experience", "index": idx, "title": header,
                                   "reason": "header line differs",
                                   "expected": _norm_ws(role.get("title")), "actual": header})
            want = [_norm_ws(b) for b in _bullet_lines(role.get("description"))]
            got = [_norm_ws(b) for p in entry["bullets"] for b in _bullet_lines(p.text)]
            if want != got:
                mismatches.append({"where": "experience", "index": idx, "title": header,
                                   "reason": "bullets differ", "expected": want, "actual": got})

        if base_docx_path:
            if _static_paragraph_texts(doc, seg) != _static_paragraph_texts(Document(base_docx_path)):
                mismatches.append({"where": "static",
                                   "reason": "paragraphs outside the summary/bullets differ from the base"})

        return {"ok": not mismatches, "mismatches": mismatches}
    except Exception as e:
        logger.warning(f"verify_docx_roundtrip failed for {docx_path}: {e}")
        return {"ok": False, "mismatches": [{"where": "file", "reason": f"could not verify: {e}"}]}


def _static_paragraph_texts(doc, seg: Dict = None) -> List[str]:
    """Texts of every paragraph the writer must never touch: all of the body
    except the summary paragraphs and the roles' bullets."""
    seg = seg or _segment_cv(doc)
    dynamic = {p._p for p in seg["summary"]} | {b._p for e in seg["entries"] for b in e["bullets"]}
    return [p.text for p in doc.paragraphs if p._p not in dynamic]


# === CV Quality Judge ===

_CV_QUALITY_PROMPT_CLAUDE = """You are a CV truth auditor. You will receive a CV that is about to be sent to an employer and ALL the verified sources about the candidate. Your job is to check every factual claim in the CV against those sources, then audit its quality.

<evaluation_date>
Today is {evaluation_date}. Any date on or before today is in the PAST. A role that ended on or before today is a past role: past tense is correct for it, and its end date is never "a future date".
</evaluation_date>

<source_precedence>
When sources disagree, the higher one wins:
1. The CAREER SOURCE BANK (first part of transcript_context, marked AUTHORITATIVE): the candidate's verified career facts.
2. verified_metrics.
3. base_cv: the truth-audited base CV this CV was built from (summary, every role's bullets, skills).
4. source_profile: the candidate's resume or profile export. On its own it is NOT sufficient support for any number, team size, budget or scale.
The interview transcript supports stories, tools and process detail, never a number that a higher source contradicts.
</source_precedence>

<source_profile>
{profile_text}
</source_profile>

<base_cv>
{base_summary}
</base_cv>

<transcript_context>
{transcript_context}
</transcript_context>

<verified_metrics>
These numbers and named facts are PRE-VERIFIED and approved. Any of them appearing in the CV, attached to the right employer, is CORRECT, never a hallucination:
{metrics_bank}
</verified_metrics>

<target_job>
<title>{job_title}</title>
<company>{company}</company>
""" + _UNTRUSTED_XML + """
<requirements>{jd_text}</requirements>
</target_job>

<generated_cv>
<summary>{cv_summary}</summary>
<experience>
{cv_experience}
</experience>
<skills>{cv_skills}</skills>
</generated_cv>

<audit_dimensions>
1. HALLUCINATION: Does the CV contain claims that appear in NONE of the sources, or that a higher source contradicts: invented company names, numbers, tools, degrees, certifications or achievements? A fact is supported if a source states it (respecting source_precedence). Rephrasing, summarizing or reframing a supported fact is NOT hallucination. ALSO flag these composition frauds even when each ingredient fact is real: (a) TENURE BINDING: the canonical years figure in verified_metrics attached to a specialty narrower than the one verified_metrics states it for (the same figure attached to a narrower specialty is a lie built from true parts), or any years figure other than the canonical one; (b) JD-NOUN INJECTION: an activity format claimed as the candidate's experience that appears only in the job description and in none of the sources (e.g., a format named only in the posting); (c) TENSE FRAUD: a role that has an end date in verified_metrics or the sources described in present tense or as "current"; (d) METRIC REASSIGNMENT: a number or achievement moved to a different employer, project or era than the source attaches it to.
2. ENRICHMENT: Does the CV add meaningful value over the source profile, or is it a near-copy?
3. KEYWORD_ALIGNMENT: Does the CV use terminology from the target job description?
4. FORMAT: Is the output clean plain text without markdown artifacts (**, ##, `, etc.)?
5. LENGTH: Is the summary 3-4 sentences? Are experience bullets 3-4 per role?
</audit_dimensions>

<claims>
List EVERY factual claim in the summary (each number, employer, title, tool, scope, outcome, tenure and tense statement), plus every claim in experience or skills that is unsupported or a composition fraud. One entry per claim:
- "section": "summary" | "experience" | "skills"
- "role_index": for experience, the number in [brackets] before the role; otherwise null
- "quote": the exact words from the CV
- "verdict": "supported" (a source states it, per source_precedence) | "unsupported" (in no source, or contradicted) | "composition" (each part is real, the combination is false: tenure binding, JD-noun injection, tense fraud, metric reassignment)
- "reason": one short sentence naming the supporting source, or why it fails
</claims>

<output_format>
Return valid JSON only. Booleans are JSON true/false, never strings:
{{
  "overall_pass": true/false,
  "hallucination": {{"found": true/false, "details": "specific fabricated claim or 'none'"}},
  "claims": [{{"section": "summary", "role_index": null, "quote": "...", "verdict": "supported", "reason": "..."}}],
  "enrichment": {{"adequate": true/false, "details": "..."}},
  "keyword_alignment": {{"adequate": true/false, "details": "..."}},
  "format": {{"clean": true/false, "details": "..."}},
  "length": {{"appropriate": true/false, "details": "..."}},
  "corrected_summary": null or "the summary with every unsupported or composition claim removed or fixed, using ONLY supported facts"
}}

hallucination.found is true exactly when at least one claim is "unsupported" or "composition". Set overall_pass to true if the CV is usable, even if minor improvements are possible; false only for clear fabrications or serious quality issues. A well-tailored CV that rephrases source material should PASS.
</output_format>"""

_CV_QUALITY_PROMPT_DEFAULT = """You are a CV truth auditor. Check every factual claim in the CV below against ALL the verified sources, then audit its quality.

=== EVALUATION DATE ===
Today is {evaluation_date}. Any date on or before today is in the PAST: a role that ended on or before today is a past role, and its end date is never "a future date".

=== SOURCE PRECEDENCE (higher wins on any conflict) ===
1. Career source bank (first part of the transcript block, marked AUTHORITATIVE). 2. Verified metrics. 3. Base CV (truth-audited). 4. Source profile: the candidate's resume or profile export, NOT sufficient on its own for any number, team size, budget or scale. The interview transcript supports stories, tools and process detail only.

=== SOURCE PROFILE (resume or profile export) ===
{profile_text}

=== BASE CV (truth-audited) ===
{base_summary}

=== SOURCE BANK + INTERVIEW TRANSCRIPT ===
{transcript_context}

=== VERIFIED METRICS (pre-approved; correct when attached to the right employer) ===
{metrics_bank}

=== TARGET JOB ===
Title: {job_title}
Company: {company}
""" + _UNTRUSTED_PLAIN + """
Requirements: {jd_text}

=== GENERATED CV ===
Summary: {cv_summary}
Experience:
{cv_experience}
Skills: {cv_skills}

=== AUDIT DIMENSIONS ===
1. HALLUCINATION: claims in NONE of the sources, or contradicted by a higher source. Rephrasing/reframing a supported fact is NOT hallucination. ALSO flag composition frauds built from true parts: (a) the canonical years figure in verified_metrics bound to a specialty narrower than the one verified_metrics states it for, or any other years figure; (b) an activity format claimed as experience that appears only in the JD; (c) a role that has an end date in verified_metrics or the sources described in present tense ("currently"); (d) a number or achievement moved to another employer, project or era.
2. ENRICHMENT: Does the CV add value over source, or near-copy?
3. KEYWORD_ALIGNMENT: Does the CV use JD terminology?
4. FORMAT: Clean plain text, no markdown artifacts?
5. LENGTH: Summary 3-4 sentences? Experience 3-4 bullets per role?

=== CLAIMS ===
List EVERY factual claim in the summary, plus every unsupported or composition claim in experience or skills: {{"section": "summary"|"experience"|"skills", "role_index": <the [n] before the role, or null>, "quote": "<exact CV words>", "verdict": "supported"|"unsupported"|"composition", "reason": "<one sentence>"}}.

=== OUTPUT ===
Return valid JSON only (booleans as JSON true/false, never strings). hallucination.found is true exactly when a claim is "unsupported" or "composition". Set overall_pass to true if the CV is usable; false only for clear fabrications or serious issues.
{{"overall_pass": true/false, "hallucination": {{"found": true/false, "details": "specific fabrication or none"}}, "claims": [...], "enrichment": {{"adequate": true/false, "details": "..."}}, "keyword_alignment": {{"adequate": true/false, "details": "..."}}, "format": {{"clean": true/false, "details": "..."}}, "length": {{"appropriate": true/false, "details": "..."}}, "corrected_summary": null or "the summary with every unsupported claim removed or fixed, using only supported facts"}}"""


# === CV Length Gate (deterministic page estimate — no PDF rendering) ===
# The .docx is the deliverable; we never render it. This reproduces the page
# height a Word/LibreOffice renderer would produce, from the template geometry.
# Calibrated against rendered two-page CVs in the base template.
_EMU_PER_PT = 12700.0


def _paragraph_font_pt(paragraph, default_pt: float = 11.0) -> float:
    """Resolve a paragraph's effective font size in points: an explicit run
    size wins (section headers carry 12pt runs), then the style size, else the
    docDefault 11pt. Mirrors how the renderer resolves inherited sizing."""
    try:
        for run in paragraph.runs:
            if run.font is not None and run.font.size is not None:
                return float(run.font.size.pt)
        style = paragraph.style
        if style is not None and style.font is not None and style.font.size is not None:
            return float(style.font.size.pt)
    except Exception:
        pass
    return default_pt


def estimate_docx_pages(docx_path: str) -> float:
    """Estimate the rendered page count of a .docx as a FRACTIONAL number
    (e.g. 1.6 = fills 1.6 of the 2-page box), WITHOUT rendering.

    Height model (calibrated on rendered CVs; see config CV length-gate block):
      text box  = page - margins  (US Letter 0.5in → 540x720pt)
      per para  chars_per_line = floor(text_w / (0.48 * fontsize))
                wrapped_lines  = ceil(len(text)/chars_per_line), min 1
                height += wrapped_lines * fontsize * 1.15 + 10pt (docDefault after)
      pages     = total_height / text_box_height

    Fail-open: any error returns 1.5 (a safe "fits 2 pages" value) so the gate
    never blocks document delivery on an estimation bug.
    """
    import math
    from .config import (_CV_LINE_MULT, _CV_SPACE_AFTER_PT, _CV_CHAR_WIDTH_FACTOR)
    try:
        doc = Document(docx_path)
        sec = doc.sections[0]
        text_w_pt = (sec.page_width - sec.left_margin - sec.right_margin) / _EMU_PER_PT
        text_h_pt = (sec.page_height - sec.top_margin - sec.bottom_margin) / _EMU_PER_PT
        if text_w_pt <= 0 or text_h_pt <= 0:
            text_w_pt, text_h_pt = 540.0, 720.0
        paras = doc.paragraphs
        total_h = 0.0
        for i, p in enumerate(paras):
            text = p.text or ""
            fs = _paragraph_font_pt(p)
            chars_per_line = max(1, int(text_w_pt / (_CV_CHAR_WIDTH_FACTOR * fs)))
            wrapped = max(1, math.ceil(len(text) / chars_per_line)) if text else 1
            total_h += wrapped * fs * _CV_LINE_MULT
            # docDefault 10pt space-after — BUT the List Bullet style carries
            # w:contextualSpacing, which the renderer honours by suppressing the
            # gap between two consecutive same-style bullets. Counting it on
            # every bullet over-read by ~0.3 pages and false-tripped the gate.
            # Suppress it
            # between adjacent same-style paragraphs when that style opts in.
            nxt = paras[i + 1] if i + 1 < len(paras) else None
            if not (nxt is not None and _same_style_contextual(p, nxt)):
                total_h += _CV_SPACE_AFTER_PT
        return round(total_h / text_h_pt, 3)
    except Exception as e:
        logger.warning(f"page estimate failed for {docx_path}: {e}")
        return 1.5


def _same_style_contextual(p, nxt) -> bool:
    """True if p and nxt share a style whose definition sets w:contextualSpacing
    — the renderer collapses the inter-paragraph gap in that case (e.g. the CV's
    List Bullet runs)."""
    from docx.oxml.ns import qn
    try:
        sp, sn = p.style, nxt.style
        if sp is None or sn is None or sp.name != sn.name:
            return False
        el = getattr(sp, "element", None)
        pPr = el.find(qn("w:pPr")) if el is not None else None
        return pPr is not None and pPr.find(qn("w:contextualSpacing")) is not None
    except Exception:
        return False


def _tailored_data_to_json(tailored_data: Dict) -> str:
    """Compact JSON of the editable content (summary + per-role bullets) for a
    revise/trim LLM call. Bullets are split from the newline-joined description."""
    roles = []
    for role in tailored_data.get("experience", []):
        bullets = [b for b in str(role.get("description", "")).split("\n") if b.strip()]
        roles.append({"title": role.get("title", ""), "company": role.get("company", ""),
                      "bullets": bullets})
    return json.dumps({"summary": tailored_data.get("summary", ""), "experience": roles},
                      ensure_ascii=False, indent=2)


def _description_bullets(role: Dict) -> List[str]:
    """A role's bullets as stored: the non-blank lines of its description."""
    return [b.strip() for b in str((role or {}).get("description") or "").split("\n") if b.strip()]


def _min_bullets_for_role(index: int, experience: Optional[list] = None) -> int:
    """Floor the revise/trim step must respect: >=3 bullets on the top (most
    recent) role, >=2 on every other, so a too-aggressive trim can't gut a role.

    The top role is the first entry that actually HAS bullets — pass the
    experience list. The parser puts an empty title placeholder before each
    real entry, so index 0 is never the role the floor is meant to protect
    (without the list the old "index 0" rule applies)."""
    top = 0
    for i, role in enumerate(experience or []):
        if _description_bullets(role):
            top = i
            break
    return 3 if index == top else 2


def _apply_revised_json(tailored_data: Dict, revised: Dict) -> Dict:
    """Fold a revise/trim call's {summary, experience:[{title,bullets}]} back
    into the canonical tailored_data shape (description = newline-joined bullets).

    A revised entry is accepted for an original entry only when the mapping is
    unambiguous — never by title alone, because two roles can share the
    same title:
      1. same position AND same header line (the answer kept the shape it was
         sent), or
      2. the answer was reordered or shortened, and the header line identifies
         exactly one original and one revised entry (a company line on both
         sides settles a repeated title).
    Anything else — a repeated title that cannot be told apart, a role the LLM
    dropped — keeps its ORIGINAL bullets: a guess would put one employer's
    bullets under another's heading (a fabricated CV, and a violation of the
    attribution lock).

    Entries without bullets are the parser's empty title placeholders. They are
    FROZEN: a revision never puts bullets under them.

    A trim below the per-role bullet floor also keeps the original bullets."""
    out = dict(tailored_data)
    if isinstance(revised.get("summary"), str) and revised["summary"].strip():
        out["summary"] = revised["summary"].strip()

    def _norm(s):
        return " ".join(str(s or "").lower().split())

    originals = tailored_data.get("experience") or []
    rev = [rr for rr in (revised.get("experience") or []) if isinstance(rr, dict)]

    def _fits(rr, role):
        # Same header line; the company may differ only by being absent on one side.
        if _norm(rr.get("title")) != _norm(role.get("title")):
            return False
        rc, oc = _norm(rr.get("company")), _norm(role.get("company"))
        return not (rc and oc and rc != oc)

    def _pick(i, role):
        if len(rev) == len(originals) and _fits(rev[i], role):
            return rev[i]
        cands = [rr for rr in rev if _fits(rr, role)]
        twins = [o for o in originals if _norm(o.get("title")) == _norm(role.get("title"))]
        if len(twins) > 1:
            # Repeated title: only an explicit company line on both sides settles it.
            company = _norm(role.get("company"))
            cands = [rr for rr in cands if company and _norm(rr.get("company")) == company]
            twins = [o for o in twins if _norm(o.get("company")) == company]
        return cands[0] if len(cands) == 1 and len(twins) == 1 else None

    picks = [_pick(i, role) for i, role in enumerate(originals)]
    # One revised entry answers for ONE original; claimed twice = ambiguous.
    claimed = [id(rr) for rr in picks if rr is not None]

    new_exp = []
    for i, role in enumerate(originals):
        role = dict(role)
        orig_n = len(_description_bullets(role))
        rr = picks[i] if picks[i] is not None and claimed.count(id(picks[i])) == 1 else None
        if rr is not None:
            raw = rr.get("bullets") or []
            if isinstance(raw, str):
                raw = raw.split("\n")
            bullets = [str(b).strip() for b in raw if str(b).strip()] if isinstance(raw, list) else []
            # Reject an over-aggressive TRIM: keep the original when the revision
            # REDUCES the role below the floor. (A revision that keeps the count
            # — e.g. keyword weaving — or a role that was already short is fine;
            # the floor only guards against gutting, not against short roles.)
            floor = _min_bullets_for_role(i, originals)
            if orig_n == 0:
                # Empty title placeholder — frozen. Bullets here would land above
                # the company line, attached to no employer.
                if bullets:
                    logger.info(f"revise: entry {i} ({role.get('title')!r}) is an empty title "
                                f"placeholder — ignoring {len(bullets)} revised bullet(s)")
            elif bullets and not (len(bullets) < orig_n and len(bullets) < floor):
                role["description"] = "\n".join(bullets)
            elif bullets:
                logger.info(f"revise: role {i} ({role.get('title')!r}) trimmed below floor "
                            f"({len(bullets)}<{floor}, was {orig_n}) — keeping original bullets")
        # no unambiguous revised entry → keep the original bullets untouched
        new_exp.append(role)
    out["experience"] = new_exp
    return out


def _parse_llm_json_object(answer: str) -> Optional[Dict]:
    """Extract the first complete JSON object from an LLM answer (respects
    string escaping — unlike a greedy regex). Returns None on failure."""
    if not answer:
        return None
    start = answer.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(answer[start:])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


_PROMPT_TRIM_CV = """You are trimming a tailored CV so it fits EXACTLY 2 pages (US Letter). It currently overflows to ~{pages} pages of content and must lose roughly {overflow_lines} lines of bullet text.

Cut using RELEVANCE-WEIGHTED trimming, never mechanically by section age. Score each bullet by:
  (a) relevance to THIS job posting's requirements and keywords,
  (b) uniqueness — is the same point already made in another bullet?,
  (c) whether it carries a number/result (metric-bearing bullets are worth more).
Cut the LOWEST-scoring bullets first, from any role. Rules:
- DELETION ONLY: remove whole bullets. Never reword, shorten, merge, reorder or add a bullet. Every bullet you keep must be copied character-for-character — the result is checked in code, and a single changed character voids the whole trim.
- Keep at least 3 bullets on the most recent (top) role and at least 2 on every other role.
- Prefer keeping bullets that carry a verified metric (a number from verified_metrics).
- Do not change the summary, any title, or the order of the roles. Entries whose bullets list is empty are heading lines: return them as they are.

TARGET ROLE: {job_title} at {company}
JOB POSTING (for relevance scoring):
{jd_text}

CURRENT CV CONTENT (JSON):
{cv_json}

Return ONLY a JSON object with the SAME shape ({{"summary": "...", "experience": [{{"title": "...", "bullets": ["...", ...]}}, ...]}}), with the lowest-value bullets removed so the CV fits 2 pages. No prose, no markdown fences."""


def _trim_violations(before: Dict, after: Dict) -> List[str]:
    """Deletion-only check for a trim. An empty list means *after* differs from
    *before* only by whole bullets removed: same summary, same roles in the same
    order with the same header fields, and each role's surviving bullets are
    byte-identical to that role's own input bullets, in their original order."""
    problems = []
    if after.get("summary") != before.get("summary"):
        problems.append("summary changed")
    b_roles, a_roles = before.get("experience") or [], after.get("experience") or []
    if len(b_roles) != len(a_roles):
        return problems + [f"role count changed ({len(b_roles)} -> {len(a_roles)})"]
    for i, (b, a) in enumerate(zip(b_roles, a_roles)):
        if ({k: v for k, v in b.items() if k != "description"}
                != {k: v for k, v in a.items() if k != "description"}):
            problems.append(f"role {i}: header changed")
        remaining = iter(_description_bullets(b))
        if not all(any(kept == src for src in remaining) for kept in _description_bullets(a)):
            problems.append(f"role {i}: a bullet was rewritten, moved or added")
    return problems


def trim_cv_to_length(tailored_data: Dict, target_job: Dict, current_pages: float,
                      model: str = None) -> Optional[Dict]:
    """One relevance-weighted trim pass — DELETION-ONLY. The model only chooses
    which bullets to drop; the answer is validated in code (_trim_violations)
    and the survivors are re-taken from the input, so nothing reworded can ship.

    Returns the trimmed tailored_data (same shape), or None — the caller keeps
    the untrimmed version — when the call fails, the answer rewrites, moves or
    adds anything, or nothing was removed."""
    from .config import DEFAULT_GENERATION_MODEL, TRUNCATION_JOB_DESC_CV, CV_MAX_PAGES
    model = model or DEFAULT_GENERATION_MODEL
    # ~ lines to shed: (fraction over 2 pages) * (~42 lines per page)
    overflow_lines = max(2, int((current_pages - CV_MAX_PAGES) * 42) + 1)
    prompt = _PROMPT_TRIM_CV.format(
        pages=current_pages, overflow_lines=overflow_lines,
        job_title=target_job.get("title", ""), company=target_job.get("company", ""),
        jd_text=_clean_posting(target_job)[:TRUNCATION_JOB_DESC_CV],
        cv_json=_tailored_data_to_json(tailored_data),
    )
    try:
        answer = call_llm(prompt, model=model)
        revised = _parse_llm_json_object(answer)
        if not revised:
            logger.warning("trim_cv_to_length: could not parse revised JSON")
            return None
        # A trim never touches the summary — only bullets can go.
        revised.pop("summary", None)
        trimmed = _apply_revised_json(tailored_data, revised)
        problems = _trim_violations(tailored_data, trimmed)
        if problems:
            logger.warning("trim_cv_to_length: rejected, not deletion-only — "
                           + "; ".join(problems[:3]))
            return None
        removed = 0
        for before, after in zip(tailored_data.get("experience") or [], trimmed["experience"]):
            # Survivors are the INPUT's own lines, not the model's retyping.
            source = [line for line in str(before.get("description") or "").split("\n") if line.strip()]
            kept, lines = _description_bullets(after), iter(source)
            if len(kept) < len(source):
                after["description"] = "\n".join(
                    next(line for line in lines if line.strip() == k) for k in kept)
                removed += len(source) - len(kept)
            elif "description" in before:
                after["description"] = before["description"]
        if not removed:
            logger.info("trim_cv_to_length: nothing was removed")
            return None
        return trimmed
    except Exception as e:
        logger.warning(f"trim_cv_to_length failed: {e}")
        return None


# === ATS Keyword Coverage (grounded; never stuffs) ===
_PROMPT_KEYWORD_COVERAGE = """You are an ATS keyword-coverage auditor. An ATS ranks CVs by how well they literally match the posting's required and preferred terms. Your job: score coverage HONESTLY and never invent.

STEP 1 — Extract the concrete required and preferred skills/keywords from the JOB POSTING (tools, methods, competencies, named platforms, domain terms). Ignore fluff ("team player", "fast-paced"). Cap at ~18 keywords, required before preferred.

STEP 2 — For each keyword, classify against the CV TEXT, using the CANDIDATE GROUNDING to decide whether the candidate genuinely has it:
  - "covered"          — the term (or a trivial inflection) already appears in the CV.
  - "synonym_only"     — the concept is in the CV under a different word. If the posting's exact term is truthfully applicable per the grounding, prefer switching to the posting's term.
  - "missing_have_it"  — NOT in the CV, but the CANDIDATE GROUNDING shows the candidate genuinely has this skill/experience. It SHOULD be added to a bullet.
  - "missing_gap"      — NOT in the CV and NOT supported by the grounding. A genuine gap. Leave it out — NEVER stuff it.

HONESTY RULE (non-negotiable): a keyword goes to "missing_have_it" ONLY if the grounding really supports it. If in doubt, it is a "missing_gap". Do not fabricate.

CANDIDATE GROUNDING (authoritative — source bank + verified metrics + base CV + transcript + profile; on a conflict the source bank wins):
--- career source bank (the candidate's verified career facts) ---
{source_bank}
--- verified metrics ---
{metrics_bank}
--- base CV (full) ---
{base_summary}
--- transcript ---
{transcript_context}
--- profile (resume or profile export) ---
{profile_text}

TARGET ROLE: {job_title} at {company}
""" + _UNTRUSTED_PLAIN + """
JOB POSTING:
{jd_text}

CV TEXT (what a parser sees):
SUMMARY: {cv_summary}
EXPERIENCE:
{cv_experience}
SKILLS: {cv_skills}

Return ONLY a JSON object:
{{"coverage": [{{"keyword": "...", "priority": "required"|"preferred", "status": "covered"|"synonym_only"|"missing_have_it"|"missing_gap", "note": "where it appears, or why absent"}}, ...],
  "add_keywords": ["<the missing_have_it terms, exact posting wording>"],
  "gaps": ["<the missing_gap terms>"],
  "summary": "<one line: N/M required covered, K have-it to add, G genuine gaps>"}}
No prose, no markdown fences."""


def check_keyword_coverage(tailored_data: Dict, target_job: Dict, profile_text: str = "",
                           base_summary: str = "", transcript_context: str = "",
                           model: str = None, skills_text: str = "") -> Dict:
    """Score the posting's keywords against the CV. Returns
    {coverage:[...], add_keywords:[...], gaps:[...], summary:str}.
    Grounded on the SAME source bank the generator reads (the candidate's
    verified career facts), so a skill the candidate really has is not classed
    as a gap; the CV text includes the SKILLS section. Since Phase 27 this is a diagnostic on the
    final CV text — nothing it returns is woven into a shipped CV.
    Fail-open: on any error returns an empty coverage with no add/gaps so the
    pipeline proceeds (the CV is still delivered)."""
    from .config import CV_KEYWORD_MODEL, TRUNCATION_JOB_DESC_CV, TRUNCATION_PROFILE_SUMMARY, TRUNCATION_TRANSCRIPT_CONTEXT
    model = model or CV_KEYWORD_MODEL
    cv_summary = tailored_data.get("summary", "")
    cv_experience = _format_experience_for_review(tailored_data.get("experience", []))
    prompt = _PROMPT_KEYWORD_COVERAGE.format(
        source_bank=_load_source_bank(),
        profile_text=(profile_text or "(not provided)")[:TRUNCATION_PROFILE_SUMMARY],
        base_summary=(base_summary or "(not provided)"),
        transcript_context=(transcript_context or "(not provided)")[:TRUNCATION_TRANSCRIPT_CONTEXT],
        metrics_bank=METRICS_BANK_TEXT,
        job_title=target_job.get("title", ""), company=target_job.get("company", ""),
        jd_text=_clean_posting(target_job)[:TRUNCATION_JOB_DESC_CV],
        cv_summary=cv_summary, cv_experience=cv_experience,
        cv_skills=skills_text or "(not provided)",
    )
    empty = {"coverage": [], "add_keywords": [], "gaps": [], "summary": ""}
    try:
        answer = call_llm(prompt, model=model, temperature=0.0)
        parsed = _parse_llm_json_object(answer)
        if not parsed:
            return {**empty, "summary": "coverage check unavailable"}
        parsed.setdefault("coverage", [])
        parsed.setdefault("add_keywords", [])
        parsed.setdefault("gaps", [])
        parsed.setdefault("summary", "")
        # Defensive: only ever act on keywords the model marked have-it.
        have_it = {c.get("keyword") for c in parsed["coverage"]
                   if c.get("status") == "missing_have_it"}
        parsed["add_keywords"] = [k for k in parsed["add_keywords"] if k in have_it] or list(have_it)
        return parsed
    except Exception as e:
        logger.warning(f"check_keyword_coverage failed: {e}")
        return {**empty, "summary": f"coverage check error: {e}"}


_PROMPT_INCORPORATE_KEYWORDS = """The tailored CV below is missing some posting keywords that the candidate GENUINELY has (verified against the candidate's grounding). Weave EACH listed keyword into the single most relevant EXISTING experience bullet — as a natural edit to real work, never a bolted-on claim.

STRICT RULES:
- Only surface skills the candidate truly has (they were pre-verified as "have it"). Do NOT invent tools, employers, metrics, or scope.
- Edit existing bullets in place; do NOT add new bullets or lengthen the CV materially (it must still fit 2 pages). Prefer swapping a vaguer word for the posting's exact term.
- Keep every verified metric intact. No em-dashes. One idea per bullet.
- If a keyword genuinely cannot be placed truthfully in any bullet, leave it out (do not force it).

KEYWORDS TO INCORPORATE (all pre-verified as genuinely the candidate's): {keywords}

TARGET ROLE: {job_title} at {company}

CURRENT CV CONTENT (JSON):
{cv_json}

Return ONLY the revised JSON object, SAME shape ({{"summary": "...", "experience": [{{"title": "...", "bullets": ["...", ...]}}, ...]}}). No prose, no fences."""


def incorporate_keywords(tailored_data: Dict, keywords: list, target_job: Dict,
                         model: str = None) -> Optional[Dict]:
    """Surgically weave grounded, have-it keywords into existing bullets. One
    LLM call. Returns revised tailored_data (same shape) or None on failure."""
    from .config import DEFAULT_GENERATION_MODEL
    if not keywords:
        return None
    model = model or DEFAULT_GENERATION_MODEL
    prompt = _PROMPT_INCORPORATE_KEYWORDS.format(
        keywords=", ".join(str(k) for k in keywords[:12]),
        job_title=target_job.get("title", ""), company=target_job.get("company", ""),
        cv_json=_tailored_data_to_json(tailored_data),
    )
    try:
        answer = call_llm(prompt, model=model)
        revised = _parse_llm_json_object(answer)
        if not revised:
            return None
        return _apply_revised_json(tailored_data, revised)
    except Exception as e:
        logger.warning(f"incorporate_keywords failed: {e}")
        return None


# === Hiring-Manager Grader (2026-07-16) — competitiveness, not truth ===
# The truth/fabrication judge is grade_cv_quality. THIS judge answers the other
# question: "would a real hiring manager for THIS role shortlist THIS CV?"

_PROMPT_HM_GRADER = """Act as a senior hiring manager with over 20 years of experience hiring for roles like "{job_title}" in this industry. You have firsthand expertise in this position and a deep understanding of what it takes to succeed in it. You are reviewing ONE candidate CV against ONE job posting.

Today is {evaluation_date}. Any date on or before today is in the PAST: a role that ended on or before today is a past role, never "a future date" and never a red flag for that reason.

STEP 1 — IDEAL CANDIDATE PROFILE. From the job posting alone, define what an exceptional hire looks like: the key qualifications, technical and soft skills, the shape of relevant experience and project work, domain expertise, and what would differentiate a good candidate from a perfect one. Note the red flags you would screen out.

STEP 2 — ASSESS THIS CV against that profile, the way you actually read CVs: 30 seconds first pass, then a close read. Score each dimension 0-100:
- relevance: does the candidate's experience map to what this role actually does day to day?
- evidence: are claims backed by concrete outcomes and numbers (not adjectives)?
- ats_alignment: would this CV surface for the posting's keywords, phrased the way the ATS will match them?
- differentiation: what makes the candidate memorable vs the other 200 applicants — is it visible in the first half page?
- clarity: structure, scannability, one-idea-per-bullet, no filler.

STEP 3 — VERDICT AND FIXES. Overall 0-100, verdict, and the 3-6 highest-leverage CONCRETE fixes (quote the line to change and give the replacement direction).

HARD CONSTRAINT on fixes: this candidate's facts are FIXED — the fix set may reframe, reorder, retitle bullets, change emphasis, surface a buried fact from the verified metrics below, or cut weak material, but must NEVER suggest adding experience, numbers, tools, or scope the candidate does not have. A genuine gap is handled by angle and adjacent evidence, never by invention.

<verified_metrics>
{metrics_bank}
</verified_metrics>

<career_source_bank>
The candidate's verified career facts. A skill or experience stated here is REAL: never list it as a red flag or a gap. It may be surfaced by a fix only under the employer and era it belongs to.
{source_bank}
</career_source_bank>

<job_posting>
Title: {job_title} at {company}
""" + _UNTRUSTED_XML + """
{jd_text}
</job_posting>

<cv>
SUMMARY: {cv_summary}
EXPERIENCE:
{cv_experience}
SKILLS: {cv_skills}
</cv>

Return ONLY a JSON object:
{{"ideal_candidate": "<3-4 line snapshot of the perfect hire for this posting>",
  "scores": {{"relevance": 0, "evidence": 0, "ats_alignment": 0, "differentiation": 0, "clarity": 0}},
  "score_notes": {{"relevance": "...", "evidence": "...", "ats_alignment": "...", "differentiation": "...", "clarity": "..."}},
  "red_flags": ["..."],
  "top_fixes": ["<concrete fix 1>", "..."],
  "overall": 0,
  "verdict": "interview" | "maybe" | "pass"}}
No prose, no fences."""


HM_GRADER_PROMPT_VERSION = "hm-v2-grounded-dated"


def grade_cv_hiring_manager(tailored_data: Dict, target_job: Dict, model: str = None,
                            evaluation_date: str = None, skills_text: str = "") -> Dict:
    """Competitiveness grade from a hiring-manager persona. Returns the parsed
    verdict dict, or {"overall": None, "error": ...} on failure (fail-open —
    informational layer, never blocks delivery).

    Since Phase 27: told today's date (without it a judge can read a recent
    end date as a future one), grounded on the candidate's source bank so real
    skills stop showing up as red flags, and shown the SKILLS section. The result carries
    evaluation_date and prompt_version."""
    from .config import CV_HM_GRADER_MODEL, TRUNCATION_JOB_DESC_CV
    model = model or CV_HM_GRADER_MODEL
    evaluation_date = evaluation_date or date.today().isoformat()
    # Omit the parenthetical when duration is empty — the generation JSON has
    # no duration key, so "Title (): ..." phantom artifacts polluted BOTH
    # grader passes on 2026-07-18 (clarity dinged, fix lists filled with
    # unfixable items, revisions stalled).
    cv_experience = _format_experience_for_review(tailored_data.get("experience", []))
    prompt = _PROMPT_HM_GRADER.format(
        job_title=target_job.get("title", ""), company=target_job.get("company", ""),
        jd_text=_clean_posting(target_job)[:TRUNCATION_JOB_DESC_CV],
        metrics_bank=METRICS_BANK_TEXT,
        source_bank=_load_source_bank(),
        evaluation_date=evaluation_date,
        cv_summary=tailored_data.get("summary", ""),
        cv_experience=cv_experience,
        cv_skills=skills_text or "(not provided)",
    )
    meta = {"evaluation_date": evaluation_date, "prompt_version": HM_GRADER_PROMPT_VERSION}
    try:
        answer = call_llm(prompt, model=model, temperature=0.0)
        parsed = _parse_llm_json_object(answer)
        if not parsed or "overall" not in parsed:
            return {"overall": None, "error": "grader returned no usable JSON", **meta}
        return {**parsed, **meta}
    except Exception as e:
        logger.warning(f"hiring-manager grader failed: {e}")
        return {"overall": None, "error": str(e), **meta}


_PROMPT_HM_REVISE = """A senior hiring manager reviewed this tailored CV for "{job_title}" at {company} and returned concrete fixes. Apply them.

STRICT RULES:
- Apply ONLY the fixes listed. Reframe, reorder, sharpen, cut — but NEVER add experience, numbers, tools, or scope that are not already in the CV or the verified metrics below. A fix that would require inventing something is skipped.
- Keep every verified number intact and attributed to the right employer.
- Anti-slop: one idea per bullet, no em-dashes, plain confident language.
- Keep the same roles; 3-4 bullets per role; the CV must still fit 2 pages.

<verified_metrics>
{metrics_bank}
</verified_metrics>

HIRING-MANAGER FIXES TO APPLY:
{fixes}

CURRENT CV (JSON):
{cv_json}

Return ONLY the revised JSON object, SAME shape ({{"summary": "...", "experience": [{{"title": "...", "bullets": ["...", ...]}}, ...]}}). No prose, no fences."""


def revise_cv_with_hm_feedback(tailored_data: Dict, target_job: Dict, fixes: list,
                               model: str = None) -> Optional[Dict]:
    """One revision pass applying the hiring-manager grader's fixes. Returns the
    revised tailored_data (title-matched fold-back, floors enforced) or None."""
    from .config import DEFAULT_GENERATION_MODEL
    if not fixes:
        return None
    model = model or DEFAULT_GENERATION_MODEL
    prompt = _PROMPT_HM_REVISE.format(
        job_title=target_job.get("title", ""), company=target_job.get("company", ""),
        metrics_bank=METRICS_BANK_TEXT,
        fixes="\n".join(f"- {f}" for f in fixes[:6]),
        cv_json=_tailored_data_to_json(tailored_data),
    )
    try:
        answer = call_llm(prompt, model=model)
        revised = _parse_llm_json_object(answer)
        if not revised:
            return None
        return _apply_revised_json(tailored_data, revised)
    except Exception as e:
        logger.warning(f"HM revision pass failed: {e}")
        return None


# === Shared review helpers (truth judge, keyword auditor, HM grader) ===

def _clean_posting(target_job: Dict) -> str:
    """The posting as the review prompts may embed it: coerced to text (None /
    NaN become empty) and run through clean_jd (untrusted data)."""
    desc = (target_job or {}).get("description")
    if desc is None or isinstance(desc, float):
        desc = ""
    return clean_jd(str(desc))


def _format_experience_for_review(experience: list) -> str:
    """Experience as a reviewer reads it: each role's title placeholder joined
    to its company/date line, bullets underneath, and the role's index in
    [brackets] so a per-claim verdict can point at it (role_index). An entry
    without bullets is a title placeholder: it prefixes the next role."""
    lines, pending = [], []
    for i, role in enumerate(experience or []):
        title = " ".join(str(role.get("title") or "").split())
        if role.get("duration"):
            title = f"{title} ({role.get('duration')})"
        bullets = _description_bullets(role)
        if not bullets:
            if title:
                pending.append(title)
            continue
        header = " | ".join(pending + [title]) if title else " | ".join(pending)
        pending = []
        lines.append(f"[{i}] {header}")
        lines.extend(f"- {b.lstrip('•-*⁃ ').strip()}" for b in bullets)
    lines.extend(pending)
    return "\n".join(lines)


def extract_section_text(docx_path: str, header: str = "SKILLS") -> str:
    """The text of one static section of a CV file (default SKILLS): every
    non-empty paragraph after the *header* line up to the next real section
    header. "" when the file or the section is missing. The truth judge, the
    keyword auditor and the HM grader read Skills through this."""
    try:
        doc = Document(docx_path)
    except Exception as e:
        logger.warning(f"extract_section_text: cannot open {docx_path}: {e}")
        return ""
    want = " ".join(header.split()).upper()
    lines, inside = [], False
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        if _section_header(p) is not None:
            inside = " ".join(text.split()).upper() == want
            continue
        if inside:
            lines.append(text)
    return "\n".join(lines)


# Internal truth-check states (Phase 27). They map onto the EXISTING flags —
# no new badge:
#   ok                 complete, correctly typed verdict on the checked text
#   check_unavailable  empty / unparseable / incomplete / mistyped answer or an
#                      exception -> REVIEW_RECOMMENDED, never PASSED
#   input_incomplete   posting missing or too thin -> the general version ships
#   invalid            a claim was proven unsupported (judge, number gate,
#                      forbidden phrase, leaked trap token) -> base fallback
#   artifact_mismatch  the saved file is not what was intended -> base bullets
TRUTH_STATES = ("ok", "check_unavailable", "input_incomplete", "invalid", "artifact_mismatch")
CLAIM_SECTIONS = ("summary", "experience", "skills")
CLAIM_VERDICTS = ("supported", "unsupported", "composition")
TRUTH_JUDGE_PROMPT_VERSION = "truth-v2-claims-dated"


def _extract_json_object(answer: str, required_key: str = None) -> Optional[Dict]:
    """First JSON object in an LLM answer that decodes completely (and carries
    *required_key*, when given). Walks every "{" with raw_decode, so prose or
    a quoted CV line before the object — and braces inside its strings — do
    not break it (the old greedy regex did)."""
    if not answer:
        return None
    decoder = json.JSONDecoder()
    first = None
    for m in re.finditer(r"\{", answer):
        try:
            obj, _ = decoder.raw_decode(answer[m.start():])
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if required_key is None or required_key in obj:
            return obj
        if first is None:
            first = obj
    return first if required_key is None else None


def _is_bool(value) -> bool:
    return type(value) is bool


def _validate_truth_audit(audit, cv_has_summary: bool) -> Optional[str]:
    """None when *audit* is a complete, correctly typed truth verdict, else the
    reason it is not. Strict on purpose: the string "false" is a type error,
    {} is incomplete, and a CV with a summary must come back with at least one
    summary claim — an unchecked summary is never a pass."""
    if not isinstance(audit, dict) or not audit:
        return "verdict is not a JSON object"
    if not _is_bool(audit.get("overall_pass")):
        return "overall_pass missing or not a boolean"
    hallucination = audit.get("hallucination")
    if not isinstance(hallucination, dict) or not _is_bool(hallucination.get("found")):
        return "hallucination.found missing or not a boolean"
    corrected = audit.get("corrected_summary")
    if corrected is not None and not isinstance(corrected, str):
        return "corrected_summary is neither null nor a string"
    claims = audit.get("claims")
    if not isinstance(claims, list):
        return "claims missing or not a list"
    for n, claim in enumerate(claims):
        if not isinstance(claim, dict):
            return f"claim {n} is not an object"
        if claim.get("section") not in CLAIM_SECTIONS:
            return f"claim {n}: section must be one of {CLAIM_SECTIONS}"
        role_index = claim.get("role_index")
        if role_index is not None and (type(role_index) is not int):
            return f"claim {n}: role_index must be an integer or null"
        if not isinstance(claim.get("quote"), str) or not claim["quote"].strip():
            return f"claim {n}: quote missing"
        if claim.get("verdict") not in CLAIM_VERDICTS:
            return f"claim {n}: verdict must be one of {CLAIM_VERDICTS}"
        if not isinstance(claim.get("reason"), str):
            return f"claim {n}: reason missing"
    if cv_has_summary and not any(c["section"] == "summary" for c in claims):
        return "no summary claim was checked"
    return None


def flagged_claims(audit: Dict, section: str = None) -> List[Dict]:
    """The unsupported / composition claims of a complete verdict (optionally
    only those in *section*)."""
    return [c for c in (audit or {}).get("claims") or []
            if isinstance(c, dict) and c.get("verdict") in ("unsupported", "composition")
            and (section is None or c.get("section") == section)]


_QUOTE_CHARS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                              " ": " "})


def _quote_key(text) -> str:
    """Comparison form for locating a judge's quote in CV text: lower case,
    curly quotes and dashes unified, bullet markers and whitespace collapsed."""
    s = str(text or "").translate(_QUOTE_CHARS).lower()
    s = re.sub(r"(?m)^\s*[-•*⁃▪]\s+", " ", s)
    return " ".join(s.split())


def quote_in(quote, text) -> bool:
    """Is the judge's *quote* text that *text* actually contains? Surrounding
    quote marks / punctuation are ignored and an elided quote ("Produced ...
    episodes") must match fragment by fragment. An empty quote is never found."""
    q = _quote_key(quote).strip(" .,;:'\"")
    hay = _quote_key(text)
    fragments = [f.strip(" .,;:'\"") for f in re.split(r"\.\.\.|…", q)]
    fragments = [f for f in fragments if f]
    return bool(fragments) and all(f in hay for f in fragments)


def summary_verdict(check: Dict, summary: str = None, shipped_text: str = None) -> Optional[str]:
    """Why the summary in a grade_cv_quality result may NOT ship, or None when
    it may: the check is complete (state ok), the summary was checked, and no
    summary claim is unsupported or a composition.

    A flagged claim counts against the summary when the judge filed it under
    "summary"; when its quote is text of *summary*, whatever section the judge
    named; or — given *shipped_text* (the shipped bullets + SKILLS, verbatim
    base / pool text) — when it is an experience/skills claim whose quote is
    not in that text, since then it is the summary's or cannot be located.
    A fabrication the judge reports without pinning it to a claim, and a
    failed verdict (overall_pass false) with nothing flagged, count against
    the summary too."""
    if not isinstance(check, dict) or check.get("state") != "ok":
        return f"truth check unavailable ({((check or {}).get('audit') or {}).get('error', 'no verdict')})"
    audit = check.get("audit") or {}
    flagged = flagged_claims(audit)
    bad = []
    for c in flagged:
        quote = c.get("quote")
        if c.get("section") == "summary":
            bad.append(str(quote)[:80])
        elif summary is not None and quote_in(quote, summary):
            bad.append(f"{str(quote)[:80]} (filed under {c.get('section')}, but it is summary text)")
        elif shipped_text is not None and not quote_in(quote, shipped_text):
            bad.append(f"{str(quote)[:80]} (filed under {c.get('section')}, not in the shipped "
                       f"bullets or skills)")
    if bad:
        return "summary claim unsupported: " + "; ".join(bad[:3])
    if (audit.get("hallucination") or {}).get("found") and not flagged:
        return "judge reported a fabrication without locating it"
    if audit.get("overall_pass") is False and not flagged:
        return "judge failed the CV (overall_pass false) without flagging a claim"
    return None


def grade_cv_quality(
    tailored_data: Dict,
    profile_text: str,
    target_job: Dict,
    model: str = CV_QUALITY_MODEL,
    base_summary: str = "",
    transcript_context: str = "",
    evaluation_date: str = None,
    skills_text: str = "",
) -> Dict:
    """Truth-audit a CV against the candidate's verified sources.

    The auditor is grounded on the SAME sources the generator is allowed to use
    — the career source bank (authoritative), METRICS_BANK_TEXT, the full base
    CV (*base_summary*: summary + every role's bullets) and the transcript —
    with an explicit precedence (source bank > metrics > base CV > the older
    profile export, which alone never supports a number), today's date
    (*evaluation_date*, default today) and the SKILLS text.

    A verdict counts only when it is complete and correctly typed (see
    _validate_truth_audit), including a per-claim list:
      claims: [{section: summary|experience|skills, role_index: int|null,
                quote, verdict: supported|unsupported|composition, reason}]
    An empty answer, no JSON, a parse error, {}, a missing field, a wrong type
    (the string "false") or any exception returns flag REVIEW_RECOMMENDED with
    state "check_unavailable" and audit["error"] set — NEVER "PASSED".

    Returns {"flag", "state", "audit", "corrected_summary"}:
      flag  PASSED / REVIEW_RECOMMENDED / HALLUCINATION (judge level; the
            package flag is decided by docs_builder)
      state "ok" or "check_unavailable"
      audit the parsed verdict plus evaluation_date and prompt_version
      corrected_summary  the judge's fixed summary (complete verdicts only)
    """
    evaluation_date = evaluation_date or date.today().isoformat()
    cv_summary = str(tailored_data.get("summary") or "")
    cv_experience = _format_experience_for_review(tailored_data.get("experience", []))

    family = detect_model_family(model)
    prompt_template = _CV_QUALITY_PROMPT_CLAUDE if family == "claude" else _CV_QUALITY_PROMPT_DEFAULT

    # Ground the judge on the AUTHORITATIVE source bank too (same channel trick
    # as generation) — a fact from the candidate's source bank must never be
    # flagged as fabrication just because older sources omit it.
    judge_ctx = ("=== CAREER SOURCE BANK (AUTHORITATIVE — on any conflict, this wins) ===\n"
                 + _load_source_bank()
                 + "\n\n=== INTERVIEW TRANSCRIPT ===\n"
                 + (transcript_context or "(not provided)"))

    prompt = prompt_template.format(
        profile_text=(profile_text or "")[:TRUNCATION_PROFILE_SUMMARY],
        base_summary=base_summary or "(not provided)",
        transcript_context=judge_ctx[:TRUNCATION_TRANSCRIPT_CONTEXT + 30000],
        metrics_bank=METRICS_BANK_TEXT,
        job_title=target_job.get("title", ""),
        company=target_job.get("company", ""),
        jd_text=_clean_posting(target_job)[:TRUNCATION_JOB_DESC_CV],
        cv_summary=cv_summary,
        cv_experience=cv_experience,
        cv_skills=skills_text or "(not provided)",
        evaluation_date=evaluation_date,
    )
    meta = {"evaluation_date": evaluation_date, "prompt_version": TRUTH_JUDGE_PROMPT_VERSION}

    def unavailable(error: str, raw: str = None) -> Dict:
        logger.warning(f"CV truth check unavailable: {error}")
        audit = {"error": error, **meta}
        if raw:
            audit["raw"] = raw[:500]
        return {"flag": "REVIEW_RECOMMENDED", "state": "check_unavailable",
                "audit": audit, "corrected_summary": None}

    try:
        answer = call_llm(prompt, model=model, temperature=0.0)
    except Exception as e:
        logger.error(f"CV quality judge failed: {e}")
        return unavailable(f"judge call failed: {e}")
    if not answer or not str(answer).strip():
        return unavailable("empty response")

    audit = _extract_json_object(str(answer), required_key="overall_pass")
    if audit is None:
        # Distinguish "no JSON at all" from "JSON without the verdict fields".
        any_obj = _extract_json_object(str(answer))
        return unavailable("no JSON verdict in response" if any_obj is None
                           else "verdict incomplete: overall_pass missing", str(answer))
    problem = _validate_truth_audit(audit, cv_has_summary=bool(cv_summary.strip()))
    if problem:
        return unavailable(f"verdict incomplete or mistyped: {problem}", str(answer))

    overall_pass = audit["overall_pass"]
    # Any unsupported/composition claim IS a finding, even if the judge forgot
    # to set hallucination.found.
    found = audit["hallucination"]["found"] or bool(flagged_claims(audit))
    # Respect overall_pass: HALLUCINATION only when the auditor judged the CV
    # NOT usable AND flagged a fabrication; a usable CV with a finding, or a
    # failed verdict that flags nothing, is REVIEW_RECOMMENDED — never PASSED.
    # (Nothing regenerates on either since Phase 27 — the summary gate in
    # docs_builder decides what ships.)
    if found and not overall_pass:
        flag = "HALLUCINATION"
    elif found or not overall_pass:
        flag = "REVIEW_RECOMMENDED"
    else:
        flag = "PASSED"
    corrected = audit.get("corrected_summary")
    corrected = corrected.strip() if isinstance(corrected, str) and corrected.strip() else None
    return {"flag": flag, "state": "ok", "audit": {**audit, **meta},
            "corrected_summary": corrected}


# === Cover-letter truth check (Phase 27) ===
# The FINAL letter text is checked sentence by sentence against the same
# sources as the CV judge, with the same complete/typed contract: an answer
# that skips a sentence, mistypes a field or does not parse is
# "check_unavailable", never a pass. docs_builder drops every unsupported or
# composition sentence, and ships the templated letter when the check is
# unavailable. One prompt for every model family (plain XML-tagged text).
LETTER_VERDICTS = ("supported", "unsupported", "composition", "no_claim")
LETTER_FLAGGED_VERDICTS = ("unsupported", "composition")
LETTER_JUDGE_PROMPT_VERSION = "letter-truth-v1"

_LETTER_TRUTH_PROMPT = """You are a cover-letter truth auditor. The cover letter below is about to be sent to an employer. Check EVERY numbered sentence: claims about the candidate against ALL the verified sources about the candidate, claims about the employer or the role against the job posting and company_context.

<evaluation_date>
Today is {evaluation_date}. Any date on or before today is in the PAST. A role that ended on or before today is a past role: past tense is correct for it, and its end date is never "a future date".
</evaluation_date>

<source_precedence>
When sources disagree, the higher one wins:
1. The CAREER SOURCE BANK (first part of transcript_context, marked AUTHORITATIVE): the candidate's verified career facts.
2. verified_metrics.
3. base_cv: the candidate's truth-audited base CV (summary, every role's bullets, skills).
4. source_profile: the candidate's resume or profile export. On its own it is NOT sufficient support for any number, team size, budget or scale.
The interview transcript supports stories, tools and process detail, never a number that a higher source contradicts.
</source_precedence>

<source_profile>
{profile_text}
</source_profile>

<base_cv>
{base_cv}
</base_cv>

<transcript_context>
{transcript_context}
</transcript_context>

<verified_metrics>
These numbers and named facts are PRE-VERIFIED. Any of them attached to the right employer is CORRECT:
{metrics_bank}
</verified_metrics>

<target_job>
<title>{job_title}</title>
<company>{company}</company>
""" + _UNTRUSTED_XML + """
<requirements>{jd_text}</requirements>
<company_context>{company_context}</company_context>
</target_job>

<letter>
{letter}
</letter>

<verdicts>
- "supported": every factual claim in the sentence is stated by a source (respecting source_precedence). Rephrasing or summarizing a supported fact is fine. A claim about the employer or the role is supported only by the job posting or company_context.
- "no_claim": the sentence makes no factual claim (a greeting, a statement of interest, a thank-you).
- "unsupported": a claim appears in no source, or a higher source contradicts it: an invented number, employer, title, tool, project, outcome, or fact about the company.
- "composition": each ingredient is real but the combination is false:
  (a) TENURE BINDING: the canonical years figure in verified_metrics attached to a specialty narrower than the one verified_metrics states it for, or any tenure figure other than that canonical figure;
  (b) JD-NOUN INJECTION: an activity, format, tool or domain claimed as the candidate's experience that appears only in the job posting;
  (c) TENSE FRAUD: a role that has an end date in verified_metrics or the sources described in the present tense or as current;
  (d) METRIC REASSIGNMENT: a number or achievement moved to a different employer, project or era than the source attaches it to;
  (e) SCOPE INFLATION: work the candidate supervised, coordinated or contributed to presented as personally executed or solely owned, or a team, budget or scale larger than the source states;
  (f) AGENCY AS EMPLOYER: a staffing agency, contingent-workforce marketplace or job board described as the employer, as the company whose team or mission the candidate would join, or as the place the candidate's past work happened, when the posting shows the work is for a client company.
</verdicts>

<output_format>
Return valid JSON only. Booleans are JSON true/false, never strings. Exactly one entry for EVERY numbered sentence, using its number as "id":
{{
  "overall_pass": true/false,
  "sentences": [{{"id": 1, "verdict": "supported", "reason": "one short sentence naming the supporting source, or why it fails"}}]
}}
"verdict" is one of "supported", "unsupported", "composition", "no_claim". overall_pass is true exactly when no sentence is "unsupported" or "composition".
</output_format>"""


def _validate_letter_audit(audit, n_sentences: int) -> Optional[str]:
    """None when *audit* is a complete, correctly typed letter verdict, else the
    reason it is not. Complete = a verdict for EVERY sentence id 1..n (an
    unchecked sentence is never a pass); a repeated id is allowed (the worst
    verdict counts), an id outside 1..n is a type error."""
    if not isinstance(audit, dict) or not audit:
        return "verdict is not a JSON object"
    if not _is_bool(audit.get("overall_pass")):
        return "overall_pass missing or not a boolean"
    entries = audit.get("sentences")
    if not isinstance(entries, list):
        return "sentences missing or not a list"
    seen = set()
    for k, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return f"entry {k} is not an object"
        sid = entry.get("id")
        if type(sid) is not int or not 1 <= sid <= n_sentences:
            return f"entry {k}: id must be an integer from 1 to {n_sentences}"
        if entry.get("verdict") not in LETTER_VERDICTS:
            return f"entry {k}: verdict must be one of {LETTER_VERDICTS}"
        if not isinstance(entry.get("reason"), str):
            return f"entry {k}: reason missing"
        seen.add(sid)
    missing = [i for i in range(1, n_sentences + 1) if i not in seen]
    if missing:
        return f"no verdict for sentence(s) {missing[:10]}"
    return None


def grade_cover_letter_truth(
    sentences: List[str],
    profile_text: str,
    target_job: Dict,
    company: str = "",
    base_cv: str = "",
    transcript_context: str = "",
    company_context: str = "",
    evaluation_date: str = None,
    model: str = CV_QUALITY_MODEL,
) -> Dict:
    """Truth-check the FINAL sentences of a cover letter (no salutation, no
    signature) against the candidate's verified sources — the same grounding and
    precedence as grade_cv_quality (source bank > METRICS_BANK_TEXT > base CV
    *base_cv* > profile export; transcript for stories), plus the posting and
    *company_context* for claims about the employer. *company* is the display
    name the letter uses.

    Returns {"state", "verdicts", "flagged", "audit"}:
      state     "ok" (complete, typed verdict) or "check_unavailable" (empty
                answer, no JSON, a missing sentence, a wrong type, an exception)
      verdicts  {sentence_id: {"verdict", "reason"}} (1-based; worst verdict
                when an id repeats) — {} when unavailable
      flagged   sorted ids judged unsupported / composition
      audit     {prompt_version, evaluation_date, overall_pass | error}
    """
    evaluation_date = evaluation_date or date.today().isoformat()
    meta = {"prompt_version": LETTER_JUDGE_PROMPT_VERSION, "evaluation_date": evaluation_date}

    def unavailable(error: str, raw: str = None) -> Dict:
        logger.warning(f"Cover-letter truth check unavailable: {error}")
        audit = {"error": error, **meta}
        if raw:
            audit["raw"] = raw[:500]
        return {"state": "check_unavailable", "verdicts": {}, "flagged": [], "audit": audit}

    sentences = [str(s or "").strip() for s in (sentences or [])]
    if not sentences or not all(sentences):
        return unavailable("no letter sentences to check")

    judge_ctx = ("=== CAREER SOURCE BANK (AUTHORITATIVE — on any conflict, this wins) ===\n"
                 + _load_source_bank()
                 + "\n\n=== INTERVIEW TRANSCRIPT ===\n"
                 + (transcript_context or "(not provided)"))
    prompt = _LETTER_TRUTH_PROMPT.format(
        evaluation_date=evaluation_date,
        profile_text=(profile_text or "")[:TRUNCATION_PROFILE_SUMMARY],
        base_cv=base_cv or "(not provided)",
        transcript_context=judge_ctx[:TRUNCATION_TRANSCRIPT_CONTEXT + 30000],
        metrics_bank=METRICS_BANK_TEXT,
        job_title=(target_job or {}).get("title", ""),
        company=company or (target_job or {}).get("company", ""),
        jd_text=_clean_posting(target_job)[:TRUNCATION_JOB_DESC_CV] or "(no posting text)",
        company_context=company_context or "(none)",
        letter="\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, 1)),
    )

    try:
        answer = call_llm(prompt, model=model, temperature=0.0)
    except Exception as e:
        logger.error(f"Cover-letter truth judge failed: {e}")
        return unavailable(f"judge call failed: {e}")
    if not answer or not str(answer).strip():
        return unavailable("empty response")
    audit = _extract_json_object(str(answer), required_key="sentences")
    if audit is None:
        return unavailable("no JSON verdict in response", str(answer))
    problem = _validate_letter_audit(audit, len(sentences))
    if problem:
        return unavailable(f"verdict incomplete or mistyped: {problem}", str(answer))

    rank = {"no_claim": 0, "supported": 0, "composition": 1, "unsupported": 2}
    verdicts: Dict[int, Dict] = {}
    for entry in audit["sentences"]:
        sid = entry["id"]
        if sid not in verdicts or rank[entry["verdict"]] > rank[verdicts[sid]["verdict"]]:
            verdicts[sid] = {"verdict": entry["verdict"], "reason": entry["reason"]}
    flagged = sorted(i for i, v in verdicts.items() if v["verdict"] in LETTER_FLAGGED_VERDICTS)
    return {"state": "ok", "verdicts": verdicts, "flagged": flagged,
            "audit": {"overall_pass": audit["overall_pass"], **meta}}
