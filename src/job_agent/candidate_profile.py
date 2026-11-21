"""Candidate profile — everything about *who is job hunting* lives here.

The rest of this codebase is deliberately impersonal. Scrapers, the 3-stage
cascade, the comp floor, CV generation and the web UI know nothing about any
particular person: they read this profile.

That split is the whole point. To run the agent for yourself you write one
YAML file; you never edit prompts, filters or Python.

    JOB_AGENT_PROFILE=profiles/my_profile.yaml python -m job_agent.main ...

Resolution order for the profile path:
    1. ``$JOB_AGENT_PROFILE``
    2. ``profiles/profile.yaml``     (your real one — gitignored)
    3. ``profiles/example.yaml``     (the shipped fictional example)

Every consumer goes through :data:`PROFILE`, so a missing key fails loudly at
import time rather than silently producing a generic CV for nobody.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEARCH_PATH = (
    os.getenv("JOB_AGENT_PROFILE"),
    _REPO_ROOT / "profiles" / "profile.yaml",
    _REPO_ROOT / "profiles" / "example.yaml",
)


def _resolve_profile_path() -> Path:
    for candidate in _SEARCH_PATH:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No candidate profile found. Copy profiles/example.yaml to "
        "profiles/profile.yaml and edit it, or set $JOB_AGENT_PROFILE."
    )


class Profile:
    """Dot/bracket access over the profile YAML with loud failures.

    ``profile["search"]["locations"]["home"]`` raises a KeyError naming the
    missing key and the file it came from — much easier to debug than a
    ``None`` that only shows up as a strange prompt six calls later.
    """

    def __init__(self, data: dict, source: Path):
        self._data = data
        self.source = source

    def __getitem__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError:
            raise KeyError(
                f"'{key}' missing from candidate profile {self.source}. "
                f"Compare against profiles/example.yaml."
            ) from None
        return Profile(value, self.source) if isinstance(value, dict) else value

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        return Profile(value, self.source) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def raw(self) -> dict:
        """The underlying plain dict (for json.dumps, DataFrame building, …)."""
        return self._data

    def __repr__(self) -> str:
        return f"Profile({sorted(self._data)!r} from {self.source.name})"


def load_profile(path: str | Path | None = None) -> Profile:
    """Load a profile YAML. Exposed for tests, which use fixtures."""
    resolved = Path(path) if path else _resolve_profile_path()
    with open(resolved, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Candidate profile {resolved} must be a YAML mapping.")
    return Profile(data, resolved)


PROFILE = load_profile()

# --- Convenience accessors used across the codebase -------------------------

CANDIDATE = PROFILE["candidate"]
SEARCH = PROFILE["search"]
RULES = PROFILE["rules"]

#: Pronouns feed the prompts, so a profile reads naturally in the third person
#: instead of the prompts hardcoding one person's gender.
PRONOUNS = CANDIDATE.get("pronouns") or {
    "subject": "they", "object": "them", "possessive": "their",
}


def pronoun(kind: str = "subject") -> str:
    return (PRONOUNS.get(kind) if hasattr(PRONOUNS, "get") else None) or "they"


def fill(template: str) -> str:
    """Substitute ``{{profile.*}}`` markers in a prompt template.

    Double braces on purpose: the prompts also carry single-brace
    ``{job_title}`` placeholders filled per-call by ``str.format``, and those
    must survive this pass untouched.
    """
    replacements = {
        "{{profile.summary}}": RULES["candidate_summary"].strip(),
        "{{profile.target_roles}}": RULES["target_roles"].strip(),
        "{{profile.reject_rules}}": RULES["reject_rules"].strip(),
        "{{profile.seniority_rule}}": RULES["seniority_rule"].strip(),
        "{{profile.craft_rule}}": RULES.get("craft_rule", "").strip(),
        "{{profile.industry_rule}}": RULES.get("industry_rule", "").strip(),
        "{{profile.name}}": CANDIDATE["name"],
        "{{profile.headline}}": CANDIDATE["headline"],
        "{{profile.home_location}}": SEARCH["locations"]["home"],
        "{{profile.pronoun_subject}}": pronoun("subject"),
        "{{profile.pronoun_object}}": pronoun("object"),
        "{{profile.pronoun_possessive}}": pronoun("possessive"),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def metrics_text() -> str:
    """The verified-metrics block injected into every generation prompt.

    Generators may use ONLY these numbers — it is the anti-hallucination
    contract. Prose form is what the model reads best, so a profile can supply
    either a ready-made ``metrics_text`` block or a list of structured
    ``metrics`` entries that get rendered here.
    """
    if "metrics_text" in PROFILE:
        return PROFILE["metrics_text"].strip()
    lines = ["VERIFIED METRICS (use only these numbers — never invent others):"]
    for entry in PROFILE.get("metrics") or []:
        value = entry.get("value", "")
        context = entry.get("context", "")
        employer = entry.get("employer", "")
        period = entry.get("period", "")
        where = " · ".join(p for p in (employer, period) if p)
        lines.append(f"- {value} — {context}" + (f" ({where})" if where else ""))
    return "\n".join(lines)


def apple_target_band(profile: Profile | None = None) -> tuple[int, int] | None:
    """(low, high) Apple IC level numbers the candidate targets, or None.

    Read from ``search.apple_target_band: {low: 3, high: 4}``. Apple publishes
    the level on its own job pages, so this is the one seniority rule the
    cascade applies in code (cascade_filter.apple_level_verdict). There is no
    default band: a profile that leaves the key out gets None, and the Apple
    level rule is off. A malformed band fails loudly, because a wrong band
    silently rejects or caps every Apple posting.
    """
    search = (profile or PROFILE).get("search") or {}
    band = search.get("apple_target_band")
    if band is None:
        return None
    source = (profile or PROFILE).source
    try:
        low, high = int(band["low"]), int(band["high"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"search.apple_target_band in {source} must be a mapping with whole-number "
            f"'low' and 'high' level numbers, e.g. {{low: 3, high: 4}}."
        ) from None
    if not 1 <= low <= high <= 9:
        raise ValueError(
            f"search.apple_target_band in {source} must satisfy 1 <= low <= high <= 9 "
            f"(got low={low}, high={high})."
        )
    return low, high


def letter_template(profile: Profile | None = None) -> dict:
    """The profile's ``letter_template`` section, checked and as plain values.

    The templated cover letter (docs_builder.templated_letter) is the fallback
    that ships when the letter check is unavailable, so its sentences must be
    the candidate's own verified facts, never text written for someone else:

        letter_template:
          opening: one sentence on what the candidate does (no numbers needed)
          proofs: {name: one past-tense sentence restating metrics_text}
          proofs_by_domain: {domain: [2-3 proof names, strongest first]}
          fallback_proof: optional number-free sentence

    Returns ``{"opening": str, "proofs": {name: str}, "proofs_by_domain":
    {domain: (names...)}, "fallback_proof": str}``; every part is empty when
    the profile has no section (docs_builder then writes a short generic
    letter). A domain list naming a proof that does not exist fails loudly.
    """
    section = (profile or PROFILE).get("letter_template") or {}
    source = (profile or PROFILE).source

    def text(value) -> str:
        return " ".join(str(value).split()) if value is not None else ""

    proofs = {str(name): text(sentence)
              for name, sentence in (section.get("proofs") or {}).items() if text(sentence)}
    by_domain = {}
    for domain, names in (section.get("proofs_by_domain") or {}).items():
        names = (names,) if isinstance(names, str) else tuple(str(n) for n in (names or ()))
        unknown = [n for n in names if n not in proofs]
        if unknown:
            raise ValueError(
                f"letter_template.proofs_by_domain.{domain} in {source} names "
                f"{', '.join(unknown)}, which letter_template.proofs does not define."
            )
        by_domain[str(domain)] = names
    return {
        "opening": text(section.get("opening")),
        "proofs": proofs,
        "proofs_by_domain": by_domain,
        "fallback_proof": text(section.get("fallback_proof")),
    }


def builtin_sites(profile: Profile | None = None) -> tuple[str, ...]:
    """The Built In regional sites to scrape, as base URLs, or ().

    Read from ``search.builtin_sites: [https://www.builtinseattle.com, ...]``.
    Built In runs one site per metro, so the list is the candidate's market,
    not the tool's: there is no default, and a profile without the key turns
    the Built In scraper off. Each entry must be an https base URL (no path,
    query, fragment or port) on a Built In host (``builtin<metro>.com``,
    ``www.`` optional); anything else fails loudly, because a wrong address
    would quietly scrape nothing every run. Entries come back as
    ``https://<host>`` (lower case, no trailing slash), each once, in order.
    """
    import re
    from urllib.parse import urlsplit

    search = (profile or PROFILE).get("search") or {}
    sites = search.get("builtin_sites")
    if sites is None:
        return ()
    source = (profile or PROFILE).source
    example = "e.g. [https://www.builtinseattle.com]"
    if isinstance(sites, str) or not isinstance(sites, (list, tuple)):
        raise ValueError(
            f"search.builtin_sites in {source} must be a list of Built In base URLs, {example}."
        )
    cleaned: list[str] = []
    for site in sites:
        url = str(site).strip().rstrip("/") if isinstance(site, str) else ""
        parts = urlsplit(url) if url else None
        host = (parts.hostname or "") if parts else ""
        label = host[4:] if host.startswith("www.") else host
        if (
            not parts
            or parts.scheme != "https"
            or not re.fullmatch(r"builtin[a-z0-9-]*\.[a-z]{2,}", label)
            or parts.port is not None
            or parts.path
            or parts.query
            or parts.fragment
            or parts.username
            or parts.password
        ):
            raise ValueError(
                f"search.builtin_sites in {source} has {site!r}: each entry must be an "
                f"https base URL on a Built In host (builtin*), {example}."
            )
        base = f"https://{host}"
        if base not in cleaned:
            cleaned.append(base)
    return tuple(cleaned)
