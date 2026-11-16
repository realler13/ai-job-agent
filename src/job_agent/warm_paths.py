"""Warm Paths (Spec M3, Phase P1) — LinkedIn-CSV referral layer.

Ingests the OFFICIAL LinkedIn connections self-export (the only LinkedIn data
this project ever touches — no automation, no scraping, per project ground
rules) and surfaces warm paths: people in the imported networks (the candidate's
own export, plus any other owner's export placed next to it) who work at a
company with an active pipeline job.

Parser handles the real-world export quirks: a "Notes:" preamble before the
header, UTF-8 BOM, quoted commas inside company names (real csv module, never
split(',')). Import is idempotent — re-import upserts keyed on profile URL.

Matching: normalize company names on both sides (trailing qualifiers such as
"(via X)" / "(Contractor)" / "| tagline" are stripped) → exact match →
anchored prefix → fuzzy (rapidfuzz token_set_ratio). Exact and anchored-prefix
matches surface as warm paths; fuzzy pairs land in a separate "possible"
bucket for manual confirmation — never silently auto-matched.

The normalized connection names are stored at import time (company_norm), so
after changing normalize_company re-run the import (idempotent):
POST /api/warm-paths/import, or import_all(db).
"""

import csv
import io
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

LINKEDIN_DIR = "data/linkedin"
# Calibration note: fuzzy token_set_ratio must NEVER auto-accept — its worst
# false positives score exactly 100 (subset trap: "Acme Fantasy" → 100 vs
# "Acme"). Auto-accept = exact-normalized + anchored-prefix subsidiary rule;
# fuzzy >=90 feeds the manual-confirm bucket only; sub-90 is discarded.
FUZZY_QUEUE_THRESHOLD = 90
PREFIX_MIN_ANCHOR = 4      # anchored-prefix rule needs >=4-char target
# Universe-side junk: job boards anonymize employers with generic strings that
# must never drive prefix/fuzzy matching.
_TARGET_STOPLIST = {
    "confidential", "team", "independent", "media", "studio", "agency",
    "jobs",   # "Jobs via Dice" → "jobs" once the " via X" tail is stripped
}
# CSV-side junk: free-text "company" values that aren't companies.
_CSV_JUNK_MARKERS = ("available on", "confidential")

_CONNECTIONS_HEADER = ["First Name", "Last Name", "URL", "Email Address",
                       "Company", "Position", "Connected On"]

# Corporate suffixes stripped during normalization.
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(?:inc|incorporated|llc|ltd|gmbh|corp|corporation|co|company|plc|sa|"
    r"ag|srl|bv|limited|holdings?|group|technologies|labs)\b\.?", re.IGNORECASE)

# Trailing qualifiers that are NOT part of the employer's name. Job boards add
# trailing qualifiers ("Acme (via StaffingCo)", "Acme (Contractor)"), and so do
# LinkedIn profiles ("Acme (formerly OldName)", "Acme (acme.com)", "Acme | The
# Parent Group"). Left in, the decorated string never equals the plain one, and
# every connection at that employer stays hidden.
# Deliberately a CLOSED list — a generic "drop any parenthetical" rule would
# turn "Acme (Acme Surgical)" into a bare "acme" anchor for the
# prefix rule.
_PAREN_QUALIFIER_RE = re.compile(
    r"\s*\(\s*(?:"
    r"via\b[^()]*"                                            # (via X)       
    r"|contract(?:ors?)?|contingent|staffing(?:\s+agency)?"   # (Contractor) / (staffing)
    r"|(?:formerly|former|fka|f/k/a|previously|ex\.?)\s[^()]*"  # (formerly X)
    r"|(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}/?"  # (domain.tld)
    r")\s*\)\s*$", re.IGNORECASE)
# Bare " via X" tail ("Acme via StaffingCo"). Case-SENSITIVE on purpose: the
# qualifier is always lowercase, while a capitalized "Via" is part of a real
# name ("Acme Via Center").
_VIA_TAIL_RE = re.compile(r"\s+via\s+\S.*$")
_TAGLINE_RE = re.compile(r"\s*\|.*$", re.DOTALL)            # "Name | tagline"


def _strip_qualifiers(name: str) -> str:
    """Drop trailing non-name qualifiers, repeatedly ("Acme via StaffingCo
    (formerly OldName)" → "Acme"). Never strips down to nothing: a
    string that is ONLY a qualifier is returned unchanged."""
    s = str(name or "").strip()
    for _ in range(4):  # qualifiers nest at most 2-3 deep
        prev = s
        for rx in (_TAGLINE_RE, _PAREN_QUALIFIER_RE, _VIA_TAIL_RE):
            cut = rx.sub("", s).strip()
            if cut:
                s = cut
        if s == prev:
            break
    return s


def normalize_company(name: str) -> str:
    """Canonicalize a company name for matching: strip trailing qualifiers
    ("(via X)", " via X", "(Contractor)", "(staffing)", "(formerly X)",
    " | tagline", "(domain.tld)"), lowercase, strip legal suffixes, domain
    endings, punctuation, collapse whitespace. Applied to BOTH sides (job
    company and connection company)."""
    s = _strip_qualifiers(name).lower().strip()
    if not s:
        return ""
    s = re.sub(r"\.(?:com|io|ai|tv|co|net|org)\b", "", s)   # acme.com → acme
    s = _COMPANY_SUFFIX_RE.sub("", s)
    # Keep unicode letters so non-ASCII company names still match.
    s = re.sub(r"[^\w&+ ]+", " ", s, flags=re.UNICODE).replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_connections_csv(path: str) -> list:
    """Parse a LinkedIn Connections.csv export into row dicts.

    Skips the "Notes:" preamble (newer exports prepend ~3 lines before the
    real header), tolerates a UTF-8 BOM, and uses the csv module so quoted
    commas in company names survive. Rows without a URL or a company are
    kept only if they have a name (they can never match, but are recorded)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:20]):
        # Detect the real header row rather than assuming a fixed preamble size.
        if line.startswith("First Name,") and "URL" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No LinkedIn connections header found in {path}")
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    rows = []
    for r in reader:
        name = f"{(r.get('First Name') or '').strip()} {(r.get('Last Name') or '').strip()}".strip()
        if not name:
            continue
        rows.append({
            "name": name,
            "url": (r.get("URL") or "").strip(),
            "email": (r.get("Email Address") or "").strip(),  # ~always blank; bonus only
            "company": (r.get("Company") or "").strip(),
            "position": (r.get("Position") or "").strip(),
            "connected_on": (r.get("Connected On") or "").strip(),
        })
    return rows


def import_connections(db, path: str, owner: str) -> dict:
    """Idempotent import of one export file (upsert keyed on profile URL —
    falls back to owner+name for the rare URL-less row)."""
    rows = parse_connections_csv(path)
    inserted = updated = 0
    with db.transaction() as conn:
        cur = conn.cursor()
        for r in rows:
            key_url = r["url"] or f"no-url:{owner}:{r['name']}"
            existing = cur.execute(
                "SELECT id FROM warm_connections WHERE url = ? AND owner = ?",
                (key_url, owner)).fetchone()
            norm = normalize_company(r["company"])
            if existing:
                cur.execute("""
                    UPDATE warm_connections
                    SET name=?, email=?, company=?, company_norm=?, position=?,
                        connected_on=?, imported_at=CURRENT_TIMESTAMP
                    WHERE id=?
                """, (r["name"], r["email"], r["company"], norm, r["position"],
                      r["connected_on"], existing[0]))
                updated += 1
            else:
                cur.execute("""
                    INSERT INTO warm_connections
                        (owner, name, url, email, company, company_norm,
                         position, connected_on)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (owner, r["name"], key_url, r["email"], r["company"], norm,
                      r["position"], r["connected_on"]))
                inserted += 1
    logger.info(f"Warm connections import ({owner}): {inserted} new, {updated} updated")
    return {"owner": owner, "parsed": len(rows), "inserted": inserted, "updated": updated}


def import_all(db, directory: str = LINKEDIN_DIR) -> list:
    """Import every {owner}-connections.csv found in the data directory."""
    results = []
    if not os.path.isdir(directory):
        return results
    for fname in sorted(os.listdir(directory)):
        m = re.match(r"([a-z]+)-connections\.csv$", fname, re.IGNORECASE)
        if m:
            results.append(import_connections(
                db, os.path.join(directory, fname), owner=m.group(1).lower()))
    return results


def _csv_company_is_junk(company_norm: str, company_raw: str) -> bool:
    """Free-text 'company' values that aren't companies (book listings, etc)."""
    if len(company_norm.split()) > 6:
        return True
    raw = str(company_raw or "").lower()
    return any(m in raw for m in _CSV_JUNK_MARKERS)


def find_warm_paths(db, company: str) -> dict:
    """Warm paths for one company.

    AUTO-ACCEPT (`matches`): exact normalized match and the
    anchored-prefix subsidiary rule — the job's normalized company is the
    leading-token prefix of the connection's ("Acme" ⊂ "Acme Europe", anchor
    >=4 chars, stoplisted generics excluded).
    MANUAL-CONFIRM (`possible`): fuzzy token_set_ratio >= 90 pairs that failed
    auto-accept — never silently surfaced (the 100-score subset trap lives
    here). Sub-90 is discarded."""
    target = normalize_company(company)
    if not target:
        return {"matches": [], "possible": []}
    try:
        conn = db._get_thread_conn()
        import sqlite3
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT owner, name, url, email, company, company_norm, position, connected_on "
            "FROM warm_connections WHERE company_norm != ''")]
        conn.row_factory = None
    except Exception as e:
        logger.error(f"Warm-path lookup failed for {company!r}: {e}")
        return {"matches": [], "possible": []}

    matches, possible = [], []
    matched_urls = set()
    prefix_ok = (len(target) >= PREFIX_MIN_ANCHOR
                 and target not in _TARGET_STOPLIST)
    for r in rows:
        if _csv_company_is_junk(r["company_norm"], r["company"]):
            continue
        if r["company_norm"] == target:
            matches.append({**r, "matched_via": "exact", "score": 100})
            matched_urls.add(r["url"])
        elif prefix_ok and r["company_norm"].startswith(target + " "):
            # subsidiary/division form: "acme europe", "acme global tech"
            matches.append({**r, "matched_via": "prefix", "score": 95})
            matched_urls.add(r["url"])
    if rows and prefix_ok:
        from rapidfuzz import fuzz
        for r in rows:
            if r["url"] in matched_urls or _csv_company_is_junk(r["company_norm"], r["company"]):
                continue
            score = fuzz.token_set_ratio(r["company_norm"], target)
            if score >= FUZZY_QUEUE_THRESHOLD:
                possible.append({**r, "matched_via": "needs_confirm", "score": int(score)})
    # The candidate's own connections first (they can be messaged directly),
    # then relevance of the connection's position to the candidate's field;
    # position relevance beats match score.
    # Without the owner key the candidate's own connections can fall outside
    # every capped list.
    matches.sort(key=lambda m: ((m.get("owner") or "").lower() != "candidate",
                                -_position_relevance(m.get("position")), -m["score"]))
    possible.sort(key=lambda m: -m["score"])
    return {"matches": matches, "possible": possible}


_RELEVANT_POSITION_TERMS = (
    ("producer", 5), ("video", 4), ("content", 4), ("creative", 4),
    ("production", 4), ("editor", 3), ("marketing", 3), ("brand", 3),
    ("events", 3), ("studio", 3), ("media", 2), ("communications", 2),
    ("design", 1), ("recruit", 5), ("talent", 4),
)


def _position_relevance(position: str) -> int:
    """Heuristic: how useful is this connection for a content-producer job hunt?
    Recruiters and producer/creative roles rank highest."""
    p = str(position or "").lower()
    return max((w for t, w in _RELEVANT_POSITION_TERMS if t in p), default=0)


def warm_paths_summary(db, company: str) -> str:
    """One-line summary for prompts/badges: '2 warm paths — <Owner> knows Jane
    Doe (Sr. Producer); <Owner> knows Bob Roe (Editor)', where <Owner> is the
    owner key of the export the connection came from. Empty string if none."""
    found = find_warm_paths(db, company)
    if not found["matches"]:
        return ""
    parts = []
    for m in found["matches"][:4]:
        who = (m.get("owner") or "").capitalize() or "Network"
        pos = f" ({m['position']})" if m.get("position") else ""
        parts.append(f"{who} knows {m['name']}{pos}")
    n = len(found["matches"])
    return f"{n} warm path{'s' if n != 1 else ''} — " + "; ".join(parts)
