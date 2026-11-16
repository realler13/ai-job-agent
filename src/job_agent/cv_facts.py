"""Deterministic fact checks for text that reaches an employer (zero LLM).

Two checks, both cheap enough to run on every generated summary or letter
sentence:

- check_numbers(text, corpus_texts): every numeric claim in *text* must appear
  in at least one verified source text. "2M" = "2 million" = "2,000,000",
  "2,300" is one number, "50-70" is one range, spelled-out numbers count
  ("thirty", "twenty-five", "two hundred", "a dozen"; the vague "hundreds" /
  "dozens" must be in a source as words), and bare years (1990-2035) are
  exempt. "+" is ignored (20+ and 20 are the same number); a percentage only
  matches a percentage, a multiplier ("5x") only a multiplier, and an amount
  with a currency sign ("$2M") only the same currency (a bare "2M" may
  restate "$2M").
- forbidden_hits(text): phrases from FORBIDDEN, the known-stale claims: a
  years-of-experience figure other than the profile's canonical one, an ended
  role written as a current one, and any claim a deployment lists because it
  was retired from the base CVs. A hit means the text repeats a claim that
  has already been corrected.

Neither check proves a sentence true: they catch invented numbers and known
regressions. The semantic truth judge (cv_tailor.grade_cv_quality) does the
rest. The verified corpus is METRICS_BANK_TEXT + the career source bank, never
the interview transcript (spoken notes can carry figures the bank supersedes).
"""
import os
import re
from functools import lru_cache
from typing import Iterable, List, NamedTuple, Optional, Tuple

# ── Numbers ──────────────────────────────────────────────────────────────────
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_WORD_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
              "seventy": 70, "eighty": 80, "ninety": 90}
_WORD_UNITS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "mm": 1e6,
           "b": 1e9, "bn": 1e9, "billion": 1e9, "hundred": 1e2, "dozen": 12}
_YEAR_MIN, _YEAR_MAX = 1990, 2035
# "$", "€", "£", "¥", optionally with a country prefix ("US$", "C$"):
# normalised to the sign, which is what has to match.
_CURRENCY = r"(?:US|USD\s?|CA|C|AU|A|NZ)?[$€£¥]"

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
# A scale letter sits right on the number ("2M", "250K"); a scale word may
# follow a space ("2 million", "2 hundred"). The letter must end the word:
# "4Kids" is not 4K.
_SCALE = (r"(?:\s?(?i:thousand|million|billion|hundred|dozen)\b"
          r"|(?:[KkMmBb]|bn|BN|MM)(?![A-Za-z]))")


def _one(cur: str, num: str, scale: str) -> str:
    return rf"(?P<{cur}>{_CURRENCY}\s?)?(?P<{num}>{_NUM})(?P<{scale}>{_SCALE})?"


_TOKEN = re.compile(
    # Not glued to a word or another number: "Q4", "MP4", "H.264".
    r"(?<![\w$€£¥.,])"
    + _one("cur", "a", "sa")
    # A dash range ("50-70", "2,500-3,000", "$200K-$1.2M"). A dash followed by
    # a word ("6-videos-weekly", "4-camera") is not a range.
    + r"(?:\s?[-–—]\s?" + _one("cur2", "b", "sb") + r")?"
    # A multiplier ("5x", "3X", "2-3x", "12 ×") is its own kind of claim.
    + r"(?P<mult>\s?[xX×])?"
    + r"(?P<plus>\s?\+)?"
    + r"(?P<pct>\s?(?:%|percent\b|per\s?cent\b))?"
    # Glued to letters after the number = part of a word ("1st", "3D", "60fps").
    + r"(?![A-Za-z0-9])")


def _alt(words) -> str:
    return "|".join(sorted(words, key=len, reverse=True))


# Spelled-out numbers: "seven", "thirty", "twenty-five", "two hundred",
# "fifty thousand", "a dozen", "a million". The plural magnitudes and a scale
# word with no number ("hundreds", "dozens of", "several thousand") are vague
# claims of their own: they match only the same word in a source. A scale word
# right after digits ("2 hundred", "2 million") belongs to _TOKEN.
_SCALE_WORDS = ("hundred", "dozen", "thousand", "million", "billion")
_WORD_TOKEN = re.compile(
    r"\b(?:"
    r"(?P<vague>(?:hundred|dozen|thousand|million|billion)s)"
    r"|(?:(?:(?P<a>an?)\s+(?P<ascale>" + _alt(_SCALE_WORDS) + r"))"
    r"|(?P<tens>" + _alt(_WORD_TENS) + r")(?:[\s-](?P<unit>" + _alt(_WORD_UNITS) + r"))?"
    r"|(?P<small>" + _alt(_WORD_NUMBERS) + r"))"
    r"(?:\s+(?P<wscale>" + _alt(_SCALE_WORDS) + r"))?"
    r"|(?<!\d)(?<!\d\s)(?P<bare>" + _alt(_SCALE_WORDS) + r")"
    r")\b", re.I)


class NumberToken(NamedTuple):
    text: str                       # as written in the text
    norm: Optional[Tuple]           # comparison form, None = exempt (a bare year)
    cur: str = ""                   # currency sign ("$", "€", ...), "" = none


def _value(num: str, scale: Optional[str]) -> float:
    value = float(num.replace(",", ""))
    if scale:
        value *= _SCALES.get(scale.strip().lower(), 1.0)
    return round(value, 6)


def _is_bare_year(num: str, scale, cur, plus, pct) -> bool:
    if scale or cur or plus or pct or "," in num or "." in num:
        return False
    return _YEAR_MIN <= int(num) <= _YEAR_MAX


def _currency(*signs) -> str:
    """The currency sign of a token ("US$ " -> "$"), "" when none."""
    for sign in signs:
        if sign:
            return sign.strip()[-1]
    return ""


def _word_value(m) -> Tuple:
    """Comparison form of one _WORD_TOKEN match: ('val', v, v), or
    ('vague', word) for "hundreds" / "several thousand"."""
    vague = m.group("vague") or m.group("bare")
    if vague:
        return ("vague", vague.lower().rstrip("s"))
    if m.group("a"):
        value = _SCALES[m.group("ascale").lower()]
    elif m.group("tens"):
        value = _WORD_TENS[m.group("tens").lower()] + (
            _WORD_UNITS.index(m.group("unit").lower()) + 1 if m.group("unit") else 0)
    else:
        value = _WORD_NUMBERS[m.group("small").lower()]
    if m.group("wscale"):
        value *= _SCALES[m.group("wscale").lower()]
    value = round(float(value), 6)
    return ("val", value, value)


def numeric_tokens(text: str) -> List[NumberToken]:
    """Every numeric claim in *text*, in order. norm = ('pct'|'mult'|'val', lo,
    hi), or ('vague', word) for "hundreds" / "dozens"; norm None marks an
    exempt bare year (or a year range like 2015-2018). cur = the currency sign
    the amount carries ("" = none)."""
    text = str(text or "")
    tokens: List[NumberToken] = []
    for m in _TOKEN.finditer(text):
        a, b = m.group("a"), m.group("b")
        kind = "pct" if m.group("pct") else "mult" if m.group("mult") else "val"
        plus = m.group("plus")
        raw = m.group(0).strip()
        cur = _currency(m.group("cur"), m.group("cur2"))
        if b is None:
            if kind == "val" and _is_bare_year(a, m.group("sa"), m.group("cur"), plus, m.group("pct")):
                tokens.append(NumberToken(raw, None))
                continue
            v = _value(a, m.group("sa"))
            tokens.append(NumberToken(raw, (kind, v, v), cur))
            continue
        # Year ranges ("2015-2018", "2015-18") are dates, not claims.
        if (kind == "val" and _is_bare_year(a, None, m.group("cur"), None, None)
                and not m.group("sa") and not m.group("sb") and not m.group("cur2")
                and (_is_bare_year(b, None, None, None, None)
                     or (len(b) == 2 and b.isdigit()))):
            tokens.append(NumberToken(raw, None))
            continue
        # "$200K-$1.2M": a scale written only on the second number applies to
        # both ("50-70K" = 50K-70K) unless the first carries its own.
        scale_a = m.group("sa") or m.group("sb")
        lo, hi = _value(a, scale_a), _value(b, m.group("sb"))
        tokens.append(NumberToken(raw, (kind, lo, hi), cur))
    for m in _WORD_TOKEN.finditer(text):
        tokens.append(NumberToken(m.group(0), _word_value(m)))
    return tokens


@lru_cache(maxsize=16)
def _corpus_forms(corpus: Tuple[str, ...]) -> frozenset:
    """Comparison forms in the corpus: every norm, plus norm + (currency,) for
    an amount that carries a currency sign."""
    forms = set()
    for text in corpus:
        for token in numeric_tokens(text):
            if token.norm is not None:
                forms.add(token.norm)
                if token.cur:
                    forms.add(token.norm + (token.cur,))
    return frozenset(forms)


def check_numbers(text: str, corpus_texts: Iterable[str]) -> List[str]:
    """Numeric tokens of *text* that appear in none of *corpus_texts* after
    normalisation (as written in *text*, de-duplicated, in order).

    An empty list means every number in *text* is traceable to the corpus. A
    range must appear as the same range; a percentage must match a percentage,
    a multiplier a multiplier; an amount with a currency sign must match an
    amount with the SAME sign ("$2M" is not "2M EUR").
    """
    forms = _corpus_forms(tuple(str(t or "") for t in corpus_texts))
    unsupported: List[str] = []
    for token in numeric_tokens(text):
        if token.norm is None:
            continue
        if (token.norm + (token.cur,) if token.cur else token.norm) in forms:
            continue
        if token.text not in unsupported:
            unsupported.append(token.text)
    return unsupported


# ── Known-stale claims ───────────────────────────────────────────────────────
class ForbiddenClaim(NamedTuple):
    label: str          # what forbidden_hits returns
    pattern: "re.Pattern"
    correction: str     # what the sources say instead


def _f(label: str, pattern: str, correction: str) -> ForbiddenClaim:
    return ForbiddenClaim(label, re.compile(pattern, re.I), correction)


def _canonical_years() -> Tuple[Optional[int], bool]:
    """The profile's canonical years-of-experience figure, and whether it is
    written with a "+": candidate.years_experience "8+ years" -> (8, True).
    (None, False) when the profile states no figure."""
    from .candidate_profile import CANDIDATE
    m = re.search(r"(\d{1,2})\s*(\+)?", str(CANDIDATE.get("years_experience") or ""))
    return (int(m.group(1)), bool(m.group(2))) if m else (None, False)


def _years_word(n: Optional[int]) -> Optional[str]:
    """Regex for the spelled-out form of *n*: "eight", or "twenty" + space or
    hyphen + "five"."""
    if n is None:
        return None
    for word, value in _WORD_NUMBERS.items():
        if value == n:
            return word
    for word, value in _WORD_TENS.items():
        if value == n - n % 10:
            return word if n % 10 == 0 else word + r"[\s-]" + _WORD_UNITS[n % 10 - 1]
    return None


# The canonical figure comes from the profile, never from this file.
_CANON_YEARS, _CANON_PLUS = _canonical_years()
_CANON_WORD = _years_word(_CANON_YEARS)

# A years-of-experience figure other than the canonical one (and other than 1:
# "a one-year contract" is a duration, not a tenure claim).
# Digits, "+" and spelled forms ("7+ years of", "fifteen years in",
# "twenty-five years producing", "a 5-year producing career", "with 12 years").
# A canonical figure of 20 or more is exempt in its spelled form as well: the
# exact word only ("twenty" does not excuse "twenty-five", nor the reverse).
_CANON_TENS_SKIP = _CANON_UNIT_SKIP = ""
if _CANON_YEARS is not None and _CANON_YEARS >= 20 and _CANON_WORD:
    if _CANON_YEARS % 10:
        _CANON_TENS_SKIP = r"(?!" + _CANON_WORD + r"\b)"
        # ... and its unit word is not read on its own ("five" in "twenty-five").
        _CANON_UNIT_SKIP = r"(?<!" + _CANON_WORD.split("[")[0] + r"[\s-])"
    else:
        _CANON_TENS_SKIP = r"(?!" + _CANON_WORD + r"\b(?![\s-](?:" + _alt(_WORD_UNITS) + r")\b))"
_TENURE_NUM = (
    r"(?:(?!(?:" + str(_CANON_YEARS if _CANON_YEARS is not None else 1) + r"|1)(?![.\d])\b)\d{1,2}(?:\.\d+)?"
    r"|" + _CANON_TENS_SKIP + r"(?:" + _alt(_WORD_TENS) + r")(?:[\s-](?:" + _alt(_WORD_UNITS) + r"))?"
    r"|" + _CANON_UNIT_SKIP + r"(?:"
    + _alt([w for w, v in _WORD_NUMBERS.items() if v not in (1, _CANON_YEARS)]) + r"))")
_TENURE_UNIT = r"\s*\+?\s*(?:-\s*)?(?:years?|yrs?|decades?)"
_TENURE_RE = (
    r"\b" + _TENURE_NUM + _TENURE_UNIT + r"['’]?\s+(?:of|in|producing|experience|expertise|"
    r"working|leading|managing|running|creating|making|directing|developing|building|as|across)\b"
    r"|\bwith\s+(?:over\s+|nearly\s+|almost\s+|about\s+|around\s+|more\s+than\s+)?"
    + _TENURE_NUM + _TENURE_UNIT + r"\b")


def _tenure_claims() -> Tuple[ForbiddenClaim, ...]:
    """The years-of-experience rules, built from the profile's canonical
    figure. No figure in the profile = no rule to state."""
    if _CANON_YEARS is None:
        return ()
    correction = "experience is the canonical years figure in verified_metrics"
    claims = []
    if _CANON_PLUS:
        # The profile writes the figure with a "+": the bare number understates it.
        bare = "|".join(filter(None, (str(_CANON_YEARS), _CANON_WORD)))
        # Not the tail of another number ("2.8 years", "1,8 years").
        claims.append(_f("canonical years without the +",
                         r"(?<![\d.,])\b(?:" + bare + r")(?!\s*\+)\s*(?:-\s*)?years?\b", correction))
    claims.append(_f("non-canonical tenure", _TENURE_RE,
                     correction + " (never state any other years figure)"))
    return tuple(claims)


# Claims a generated text must never repeat. Built in: the years rules above
# and an ended role written as a current one (a generic default; drop that
# entry for a candidate whose most recent role is current). A deployment adds
# one _f(label, pattern, correction) per claim retired from its base CVs, for
# instance:
#     _f("old team size", r"\bteam\s+of\s+30\b",
#        "verified_metrics: the current team size"),
# Check every pattern against the base CVs, METRICS_BANK_TEXT and the source
# bank's facts (tests/test_cv_truth_gates.py): a phrase the verified sources
# themselves use can never be "forbidden".
FORBIDDEN: Tuple[ForbiddenClaim, ...] = (
    *_tenure_claims(),
    _f("an ended role written as a current role",
       r"\b20\d\d\s*[-–—]\s*(?:present|current|now)\b"
       r"|\b(?:currently|presently)\s+(?:at|with|working|employed|serving|producing|leading|"
       r"managing|running|a|an|the)\b",
       "the most recent role has an end date in verified_metrics (a past role)"),
)


def forbidden_hits(text: str) -> List[str]:
    """Labels of the FORBIDDEN (known-stale) claims that *text* repeats."""
    text = str(text or "")
    return [claim.label for claim in FORBIDDEN if claim.pattern.search(text)]


# ── Verified corpus ──────────────────────────────────────────────────────────
def _read_source_bank() -> str:
    """The FULL career source bank (the prompt loader truncates it)."""
    from .config import SOURCE_BANK_PATH
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        with open(os.path.join(root, SOURCE_BANK_PATH), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def verified_corpus(extra: Iterable[str] = ()) -> List[str]:
    """The texts a number in a generated document must trace to:
    METRICS_BANK_TEXT and the career source bank (never the interview
    transcript), plus any *extra* verified text (e.g. the routed base CV)."""
    from .config import METRICS_BANK_TEXT
    return [METRICS_BANK_TEXT, _read_source_bank(), *[str(t or "") for t in extra]]
