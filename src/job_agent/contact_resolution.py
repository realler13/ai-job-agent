"""Contact resolution (Spec M2) — turn a named contact into a VERIFIED email.

Waterfall (Hunter.io — free tier includes the full finder+verifier API, credits
charged only on success):
    Hunter Email Finder (full_name + company)
      → inline verification.status
        → email_status: verified | pattern_guess | unknown

GROUND RULES (spec §3, enforced here, not just in prompts):
- Only `verified` may ever be presented as a sendable address. `pattern_guess`
  (finder hit without a 'valid' verification) is stored for reference but the
  playbook labels it DO-NOT-EMAIL — bounces from the candidate's real address damage
  deliverability for everything after.
- Licensed data only: Hunter sources emails from public web pages and returns
  the source URLs per result (stored alongside).
- Hard per-run cap (a runaway loop must not eat the month's credit budget in
  an afternoon).

Degrades silently when HUNTER_API_KEY is absent — the playbook simply keeps
its LinkedIn-first channel ladder.
"""

import json
import logging
import os
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

HUNTER_API_KEY = os.getenv("HUNTER_API_KEY", "")
HUNTER_TIMEOUT = 20
MAX_LOOKUPS_PER_RUN = 2   # named contacts per intel run worth a credit


def hunter_enabled() -> bool:
    return bool(HUNTER_API_KEY)


def resolve_email(full_name: str, company: str, domain: str = None) -> Optional[dict]:
    """One Hunter Email Finder call. Returns
    {"email","email_status","score","position","sources"} or None.

    Prefer the DOMAIN param when known: live-tested, a generic scraped company
    name (a common word) resolves nothing while the company's own domain
    returns a verified address immediately. Charged only on success.

    email_status: 'verified' only when Hunter's inline verification says
    'valid'; a found-but-unverified address is 'pattern_guess'; no result →
    None (caller records nothing)."""
    if not hunter_enabled() or not full_name or not (company or domain):
        return None
    params = {"full_name": full_name, "api_key": HUNTER_API_KEY}
    if domain:
        params["domain"] = domain
    else:
        params["company"] = company
    try:
        resp = requests.get(
            "https://api.hunter.io/v2/email-finder",
            params=params,
            timeout=HUNTER_TIMEOUT,
        )
        if resp.status_code == 429:
            logger.warning("Hunter rate/credit limit hit — skipping resolution")
            return None
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or {}
        email = data.get("email")
        if not email:
            return None
        verification = (data.get("verification") or {}).get("status")
        status = "verified" if verification == "valid" else "pattern_guess"
        sources = [s.get("uri") for s in (data.get("sources") or [])[:5]
                   if isinstance(s, dict) and s.get("uri")]
        return {
            "email": email,
            "email_status": status,
            "score": data.get("score"),
            "position": data.get("position") or "",
            "sources": sources,
        }
    except Exception as e:
        logger.warning(f"Hunter email-finder failed for {full_name!r}@{company!r}: {e}")
        return None


_EXTRACT_PROMPT = """From the hiring-team research below, list the INDIVIDUAL PEOPLE employed at {company} (any obvious variant counts: "{company}.com", "{company} Inc", the company's careers/leadership page). Full name + job title. Output STRICT JSON only — an array of at most 3 objects:
[{{"name": "Full Name", "title": "Their title", "role_type": "recruiter|hiring_manager|peer"}}]
INCLUDE anyone the text identifies as a {company} employee, executive, or team member — even if their exact team isn't confirmed (e.g. a VP People listed on the careers page IS a valid recruiter-side contact).
EXCLUDE people who work at OTHER companies mentioned in passing (platform executives, competitors, journalists), partial names ("Jane D."), and former employees.
Empty array [] only if the text names nobody at {company}.

RESEARCH:
{research}"""


def extract_named_contacts(contact_blocks: list, company: str = "") -> list:
    """Cheap LLM pass pulling {name,title,role_type} for people the discovery
    research actually named AT the target company (a role whose title names a
    platform, e.g. a "Shopify Developer" at an agency, must not extract
    executives of the platform itself)."""
    if not contact_blocks:
        return []
    research = "\n\n".join(str(c.get("content", ""))[:2000] for c in contact_blocks)
    if not research.strip():
        return []
    try:
        from .gpt_filter import call_llm
        from .config import CASCADE_STAGE2_MODEL
        text = call_llm(_EXTRACT_PROMPT.format(research=research[:6000],
                                               company=company or "the hiring company"),
                        model=CASCADE_STAGE2_MODEL)
        if not text:
            return []
        start = text.find("[")
        if start == -1:
            return []
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        out = []
        for p in parsed if isinstance(parsed, list) else []:
            name = str(p.get("name", "")).strip()
            # full name = at least two words, letters only-ish
            if len(name.split()) >= 2 and re.match(r"^[\w .'’-]+$", name):
                out.append({"name": name,
                            "title": str(p.get("title", "")).strip()[:120],
                            "role_type": str(p.get("role_type", "peer")).strip()})
        return out[:3]
    except Exception as e:
        logger.warning(f"Contact extraction failed: {e}")
        return []


# ATS/job-board hosts that must never be mistaken for the company's own domain.
_NON_COMPANY_HOSTS = re.compile(
    r"greenhouse|lever\.co|ashbyhq|myworkdayjobs|smartrecruiters|bamboohr|"
    r"workable|recruitee|teamtailor|linkedin|indeed|ziprecruiter|glassdoor")


def company_domain(db, company: str) -> Optional[str]:
    """Best-effort company domain from the cached careers-page lookup (free —
    resolve_careers_page has a 7-day Perplexity cache). Live-tested: Hunter's
    finder needs a domain for generically-named companies (a common word)."""
    try:
        from urllib.parse import urlparse
        from .perplexity_utils import resolve_careers_page
        info = resolve_careers_page(company, db=db) or {}
        url = info.get("careers_url") or ""
        host = (urlparse(url).netloc or "").lower()
        if not host or _NON_COMPANY_HOSTS.search(host):
            return None
        # careers.foo.com / jobs.foo.com / www.foo.com → foo.com
        parts = host.split(".")
        if len(parts) > 2 and parts[0] in ("www", "careers", "jobs", "boards", "apply"):
            parts = parts[1:]
        return ".".join(parts)
    except Exception:
        return None


def resolve_contacts_for_job(db, job_id: int, company: str, contact_blocks: list) -> list:
    """Extract named people from discovery research and resolve emails for up
    to MAX_LOOKUPS_PER_RUN of them. Persists every attempt to the contacts
    table; returns the stored rows (with email only when found)."""
    people = extract_named_contacts(contact_blocks, company)
    if not people:
        return []
    domain = company_domain(db, company) if hunter_enabled() else None
    resolved = []
    lookups = 0
    for p in people:
        rec = {"job_id": job_id, "company": company, **p,
               "email": None, "email_status": "unknown", "source": None,
               "confidence": None}
        if hunter_enabled() and lookups < MAX_LOOKUPS_PER_RUN:
            lookups += 1
            hit = resolve_email(p["name"], company, domain=domain)
            if hit:
                rec.update({
                    "email": hit["email"],
                    "email_status": hit["email_status"],
                    "source": "; ".join(hit.get("sources") or [])[:500] or "hunter",
                    "confidence": hit.get("score"),
                })
        try:
            db.save_contact(rec)
        except Exception as e:
            logger.warning(f"Failed saving contact {p['name']!r}: {e}")
        resolved.append(rec)
    return resolved


def resolved_contacts_text(resolved: list) -> str:
    """Render resolved contacts for the playbook prompt. Only 'verified'
    addresses are presented as sendable; pattern guesses are explicitly
    marked DO-NOT-EMAIL (deliverability protection)."""
    if not resolved:
        return ""
    lines = []
    for r in resolved:
        base = f"- {r['name']} ({r.get('title') or r.get('role_type')})"
        if r.get("email") and r.get("email_status") == "verified":
            lines.append(f"{base} — VERIFIED email: {r['email']} "
                         f"(source: {r.get('source') or 'hunter'})")
        elif r.get("email"):
            lines.append(f"{base} — address found but UNVERIFIED "
                         f"(DO-NOT-EMAIL; use LinkedIn instead)")
        else:
            lines.append(f"{base} — no email found; use LinkedIn")
    return "\n".join(lines)
