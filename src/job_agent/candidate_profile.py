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
