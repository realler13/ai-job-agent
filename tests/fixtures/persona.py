"""Constants for the suite's FICTIONAL person, read from the fixture files so a
test never retypes them (and never drifts from them).

    from tests.fixtures.persona import YEARS, NAME, EMPLOYERS
    summary = f"Video producer with {YEARS} of experience ..."

Values come from tests/fixtures/test_profile.yaml (the profile tests/conftest.py
pins) and tests/fixtures/fictional_career.md (the base CV fixture).
"""

import re
from pathlib import Path

import yaml

FIXTURES = Path(__file__).resolve().parent
PROFILE_PATH = FIXTURES / "test_profile.yaml"
CAREER_PATH = FIXTURES / "fictional_career.md"

_PROFILE = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
_CANDIDATE = _PROFILE["candidate"]
_LOCATIONS = _PROFILE["search"]["locations"]

NAME = _CANDIDATE["name"]                      # "Marisol Vantongeren"
FIRST_NAME = _CANDIDATE["first_name"]
LAST_NAME = _CANDIDATE["last_name"]
EMAIL = _CANDIDATE["email"]
YEARS = _CANDIDATE["years_experience"]         # canonical figure, e.g. "11+ years"
YEARS_NUMBER = int(re.match(r"\d+", YEARS).group(0))

HOME = _LOCATIONS["home"]                      # "Seattle, WA"
PRIMARY_METRO_LABEL = _LOCATIONS["primary_metro_label"]
PRIMARY_METRO_TERMS = tuple(_LOCATIONS["primary_metro_terms"])
SECONDARY_METRO_LABEL = _LOCATIONS["secondary_metro_label"]
SECONDARY_METRO_TERMS = tuple(_LOCATIONS["secondary_metro_terms"])

COMP_FLOOR_YEARLY = _PROFILE["search"]["comp_floor_yearly"]
COMP_FLOOR_SECONDARY_METRO = _PROFILE["search"]["comp_floor_secondary_metro"]

#: (low, high) Apple IC level numbers, from search.apple_target_band (3, 4).
#: The tool has no default band: a raw_profile() variant without the key
#: turns the Apple level rule off.
APPLE_TARGET_BAND = (_PROFILE["search"]["apple_target_band"]["low"],
                     _PROFILE["search"]["apple_target_band"]["high"])

#: Built In base URLs, from search.builtin_sites ("https://www.builtinseattle.com").
#: A raw_profile() variant without the key turns the Built In scraper off.
BUILTIN_SITES = tuple(_PROFILE["search"]["builtin_sites"])

METRICS_TEXT = _PROFILE["metrics_text"]        # the only numbers a generator may use
#: The fallback cover letter's sentences: {opening, proofs, proofs_by_domain, fallback_proof}.
LETTER_TEMPLATE = _PROFILE["letter_template"]


def raw_profile() -> dict:
    """A fresh copy of the whole profile YAML, for a test that writes a
    variant of it (a section removed, a value changed) to a temp file."""
    return yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))


def _employers() -> tuple:
    """Company names from the career's EXPERIENCE lines, newest first."""
    names, in_experience = [], False
    for line in CAREER_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_experience = line[3:].strip() == "EXPERIENCE"
        elif in_experience and " | " in line and not line.startswith(("#", "-")):
            names.append(line.split(" | ")[0].strip())
    return tuple(names)


EMPLOYERS = _employers()
