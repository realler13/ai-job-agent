"""
3-Stage Cascade AI Filtering Pipeline.

Stage 1 — Pre-filter (Cheap API)
    Model: claude-sonnet-5  |  Binary YES/NO  |  2.5K char descriptions

Stage 2 — Full Grading (Cheap API)
    Model: claude-sonnet-5  |  Structured recruiter grading  |  Batched  |  XML-tagged prompt

Stage 3 — Judge (Premium API, Borderline Only)
    Model: claude-fable-5-2  |  Tier-sort POTENTIAL_MATCH (recall mode)  |  One-at-a-time

Output tiers (the `tier` column on the returned DataFrame):
    strong        → HIGH_MATCH / USER_INPUT (top of the feed)
    worth_a_look  → POTENTIAL_MATCH / WORTH_A_LOOK (adjacent/stretch, lower tier)
    None          → genuine reject (stored, but not surfaced)
"""

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from typing import Callable, Optional

import pandas as pd

from .config import (
    CASCADE_STAGE1_FALLBACKS,
    CASCADE_STAGE1_MODEL,
    CASCADE_STAGE1_PROMPT,
    CASCADE_STAGE2_BATCH_SIZE,
    CASCADE_STAGE2_FALLBACKS,
    CASCADE_STAGE2_MODEL,
    CASCADE_STAGE3_FALLBACKS,
    CASCADE_STAGE3_MODEL,
    CASCADE_STAGE3_PROMPT,
    CASCADE_STAGE3_PROMPT_CLAUDE,
    DOMAIN_DESCRIPTIONS,
    TRUNCATION_JOB_DESC_FILTER,
    TRUNCATION_JOB_DESC_STAGE1,
    USE_PERPLEXITY,
    detect_model_family,
    get_stage3_prompt,
    location_priority,
    resolve_cv_domain,
)
from .gpt_filter import DEFAULT_PROMPT_TEMPLATE, STAGE2_PROMPT_CLAUDE, call_llm
from .jd_sanitizer import SanitizedJD, sanitize_jd

logger = logging.getLogger(__name__)

# ── Tiering ──────────────────────────────────────────────────────────────
# A new orthogonal axis on top of match_level. match_level stays granular
# (HIGH/POTENTIAL/WORTH_A_LOOK/SOCIAL_TRAP/…); `tier` is the single thing the
# web feed groups on. "strong" = top of the feed, "worth_a_look" = adjacent /
# stretch shown under its own label, None = genuine reject (stored, not shown).
STRONG_MATCH_LEVELS = {"HIGH_MATCH", "USER_INPUT"}
WORTH_A_LOOK_MATCH_LEVELS = {"POTENTIAL_MATCH", "WORTH_A_LOOK"}


def assign_tier(match_level) -> Optional[str]:
    """Map a final match_level to its user-facing tier (or None for reject)."""
    if match_level in STRONG_MATCH_LEVELS:
        return "strong"
    if match_level in WORTH_A_LOOK_MATCH_LEVELS:
        return "worth_a_look"
    return None


# Stage 2 verdict parsing: EARLIEST token in the line wins. The model outputs
# "Job X: VERDICT - reasoning"; substring-anywhere matching let reasoning text
# ("...this is not a high match...") flip an IRRELEVANT into HIGH_MATCH.
_STAGE2_VERDICT_RE = re.compile(
    r"HIGH[ _]MATCH|POTENTIAL[ _]MATCH|WORTH[ _]A[ _]LOOK|SOCIAL[ _]TRAP|JUNIOR\s*[/_ ]\s*WEAK|IRRELEVANT"
)
_STAGE2_VERDICT_MAP = {
    "HIGH MATCH": ("HIGH_MATCH", 0.9, "NONE"),
    "POTENTIAL MATCH": ("POTENTIAL_MATCH", 0.6, "NONE"),
    "WORTH A LOOK": ("WORTH_A_LOOK", 0.45, "NONE"),
    "SOCIAL TRAP": ("SOCIAL_TRAP", 0.8, "SOCIAL_TRAP"),
    "JUNIOR WEAK": ("JUNIOR_WEAK", 0.8, "JUNIOR_WEAK"),
    "IRRELEVANT": ("IRRELEVANT", 0.7, "NONE"),
}


# A line that LEADS with "REJECT" is a reject verdict even though REJECT is
# not one of the Stage-2 tokens (the Stage-3 vocabulary and the BELOW-LEVEL
# WALL's "→ REJECT (JUNIOR/WEAK)" wording leak into Stage-2 answers). Without
# this, "Job 1: REJECT - org-seniority Director" parsed as no-verdict and sat
# in STAGE2_PENDING forever, and "REJECT - coordinator; not even worth a look"
# parsed as WORTH_A_LOOK via the earliest-token rule.
# A qualifier right after the leading REJECT names the reject kind.
_STAGE2_LEADING_REJECT_RE = re.compile(
    r"^\W*REJECT(?:ED)?\b(?:[\s:\-\u2013\u2014(\[]*(?P<kind>JUNIOR\s*[/_ ]\s*WEAK|SOCIAL[ _]TRAP)\b)?"
)


def _parse_stage2_verdict(content: str) -> Optional[dict]:
    """Extract the leading Stage 2 verdict from a graded line, or None."""
    upper = content.upper()
    lead = _STAGE2_LEADING_REJECT_RE.match(upper)
    if lead:
        kind = re.sub(r"[\s/_]+", " ", lead.group("kind") or "IRRELEVANT").strip()
        ml, conf, rule = _STAGE2_VERDICT_MAP[kind]
        return {"match_level": ml, "confidence": conf, "rejection_rule": rule}
    m = _STAGE2_VERDICT_RE.search(upper)
    if not m:
        return None
    token = re.sub(r"[\s/_]+", " ", m.group(0)).strip()
    ml, conf, rule = _STAGE2_VERDICT_MAP[token]
    return {"match_level": ml, "confidence": conf, "rejection_rule": rule}


# A Stage-2 job the model never gave a verdict TOKEN for was not judged — it
# stays STAGE2_PENDING (relevance NULL), never a permanent IRRELEVANT. One
# in-run re-grade pass is tried (run_stage2 → _regrade_unparsed); after that
# NOTHING re-grades it automatically: scrape dedup skips the stored link and
# main.process_pending_db_jobs (the manual re-grade path) has no scheduled
# caller. A pending row stays visible as ungraded rather than wrongly rejected.
_STAGE2_NO_VERDICT = "Stage 2: no verdict token in response (retryable)"
_STAGE2_NO_LINE = "Stage 2: no verdict line in response (retryable)"
_STAGE2_JOB_LINE_RE = re.compile(r"Job\s+(\d+)", re.IGNORECASE)


def _stage2_pending(reason: str) -> dict:
    return {"match_level": "STAGE2_PENDING", "confidence": 0.0,
            "reason": reason, "rejection_rule": "NONE"}


def _parse_stage2_answer(answer: str, n_jobs: int) -> list[dict]:
    """One result per job from a Stage-2 batch answer ("Job N: VERDICT - why").

    Default is the retryable STAGE2_PENDING sentinel, NOT a reject: a job whose
    verdict line is missing was never judged — burying it as IRRELEVANT made it
    permanently unrecoverable (dedup + get_pending_jobs both skip graded rows).
    A "Job N:" line without a verdict token stays pending too; when that header
    stands alone ("**Job 3:**" then the verdict on the next line), the next
    non-header line is read as its content."""
    results = [_stage2_pending(_STAGE2_NO_LINE) for _ in range(n_jobs)]
    open_idx = None  # a bare "Job N:" header waiting for its content line
    for line in (answer or "").splitlines():
        stripped = line.strip(" \t*_#>-")
        if not stripped:
            continue
        m = _STAGE2_JOB_LINE_RE.search(line.split(":", 1)[0]) if ":" in line else None
        if m and "job" in line.lower():
            idx = int(m.group(1)) - 1
            content = line.split(":", 1)[1].strip(" \t*_")
            open_idx = None
            if not 0 <= idx < n_jobs:
                continue
            if not content:
                open_idx = idx
                continue
        elif open_idx is not None:
            idx, content, open_idx = open_idx, stripped, None
        else:
            continue
        verdict = _parse_stage2_verdict(content)
        if verdict:
            verdict["reason"] = content
            results[idx] = verdict
        elif results[idx]["match_level"] == "STAGE2_PENDING":
            # Judged line but no verdict token: not a verdict — keep it
            # retryable (the old code stored it as a permanent IRRELEVANT).
            results[idx] = _stage2_pending(f"{_STAGE2_NO_VERDICT}: {content[:200]}")
    return results


def _stage3_priority(value) -> int:
    """apply_priority as an int 1-10. The judge sometimes returns "8", "8/10",
    null or 7.5; the old arithmetic on a raw string raised, and the except-path
    silently turned a HIGH verdict into a 'Stage 3 error' POTENTIAL row."""
    if isinstance(value, bool):
        return 5
    if isinstance(value, (int, float)):
        n = value
    else:
        m = re.search(r"\d+(?:\.\d+)?", str(value or ""))
        if not m:
            return 5
        n = float(m.group(0))
    if n != n:  # NaN
        return 5
    return int(min(10, max(1, round(n))))


# ── Posting text the judges read ─────────────────────────────────────────
# Posting text is DATA. Sentences aimed at AI tools ("if you are an AI ...")
# or at applicants ("mention the word X in your application") are stripped
# in code before Stage 1/2/3 see the description (jd_sanitizer); the prompts
# themselves are unchanged. Cached: each description is read by up to 3 stages.
@lru_cache(maxsize=4096)
def _sanitized(text: str) -> SanitizedJD:
    return sanitize_jd(text)


def _raw_description(row) -> str:
    d = row.get("description", "")
    if d is None or (isinstance(d, float) and d != d):
        return ""
    return str(d)


def _model_description(row) -> str:
    """The description a judge may read (sanitized, untruncated)."""
    return _sanitized(_raw_description(row)).text


# ── Apple internal levels (added 2026-10-03) ─────────────────────────────
# Apple publishes the level on the detail page (scrapes.job_level: "Producer
# IC4", "Human Interface Producer ICT4–ICT5", "Content Production MGR2"), so
# a posting can be compared with a target band without reading the title.
# The band below is a built-in default for a mid-level individual contributor
# (the level of the shipped example profile); it is not read from the profile.
# Only Apple-format codes count — LinkedIn's "Mid-Senior level" lives in the
# same column and is ignored.
_APPLE_LEVEL_RE = re.compile(r"\b(ICT|IC|MGR)\s?(\d)\b")
APPLE_TARGET_BAND = "IC2–IC3"


def apple_level_verdict(job_level) -> Optional[tuple[str, str]]:
    """(action, note) for an Apple-format level, or None when there is none.

    action: 'reject_manager' | 'reject_below' | 'reject_above' | 'cap' |
    'in_band'. Rules, on the low and high ends of the published range: the
    MGR (people-manager) ladder → reject; high end <= 1 → reject (below the candidate's
    level); low end >= 5 → reject (above the candidate's level); low end == 4 → cap at
    worth_a_look; a level or range that includes 2 or 3 → in band."""
    if not isinstance(job_level, str):
        return None
    codes = _APPLE_LEVEL_RE.findall(job_level)
    if not codes:
        return None
    nums = [int(n) for _, n in codes]
    lo_i, hi_i = nums.index(min(nums)), nums.index(max(nums))
    lo, hi = nums[lo_i], nums[hi_i]
    low_code = "".join(codes[lo_i])
    label = low_code if lo == hi else f"{low_code}–{''.join(codes[hi_i])}"
    if any(ladder == "MGR" for ladder, _ in codes):
        return "reject_manager", (f"[Apple level {label} — people-manager ladder, outside the candidate's "
                                  f"target band {APPLE_TARGET_BAND}]")
    if hi <= 1:
        return "reject_below", f"[Apple level {label} — below the candidate's target band {APPLE_TARGET_BAND}]"
    if lo >= 5:
        return "reject_above", f"[Apple level {label} — above the candidate's target band {APPLE_TARGET_BAND}]"
    if lo == 4:
        return "cap", f"[Apple level {low_code} — above the candidate's target band {APPLE_TARGET_BAND}]"
    return "in_band", f"[Apple level {label}: in the candidate's target band]"


def _append_note(df: pd.DataFrame, mask, note: str, sep: str = " ") -> None:
    """Append a policy note to the reasoning of the masked rows (in place).
    Both columns: main.py stores gpt_reasoning (falling back to reason)."""
    for col in ("reason", "gpt_reasoning"):
        if col in df.columns:
            df.loc[mask, col] = df.loc[mask, col].apply(
                lambda s: f"{s}{sep}{note}" if isinstance(s, str) and s.strip() else note)


def _apply_apple_level_rule(result: pd.DataFrame) -> None:
    """The Apple level rule, deterministic, where the tier is stamped (in
    place). A reject uses the standard reject representation (match_level,
    tier None, rejection_rule, gpt_relevance) so feeds and stats stay
    coherent; it applies to tiered rows and to rows Stage 2 left pending
    (the level alone decides those). Rows a judge already rejected keep
    their own reason."""
    if "job_level" not in result.columns:
        return
    for idx, level in result["job_level"].items():
        verdict = apple_level_verdict(level)
        if verdict is None:
            continue
        action, note = verdict
        tiered = not pd.isna(result.at[idx, "tier"])
        if action.startswith("reject"):
            if not tiered and result.at[idx, "match_level"] != "STAGE2_PENDING":
                continue
            result.at[idx, "match_level"] = "JUNIOR_WEAK" if action == "reject_below" else "IRRELEVANT"
            result.at[idx, "tier"] = None
            result.at[idx, "confidence"] = 0.9
            result.loc[idx, "rejection_rule"] = "APPLE_LEVEL"
            if "gpt_relevance" in result.columns:
                result.at[idx, "gpt_relevance"] = "irrelevant"
        elif not tiered:
            continue
        elif action == "cap":
            result.at[idx, "tier"] = "worth_a_look"
        _append_note(result, [idx], note)


# ── Apple contingent marketplace ─────────────────────────────────────────
# WillHire / "(Contractor)" Apple listings come from a contingent-workforce
# marketplace, a channel that works mostly through referrals. Capped at
# worth_a_look with a one-line explanation. Staffing agencies and contract
# roles in general are NEVER capped: they are ordinary hiring channels.
CONTINGENT_MARKETPLACE_NOTE = ("[Contingent marketplace listing — this channel usually "
                               "works through a referral]")


def is_contingent_marketplace(row) -> bool:
    def _s(key):
        v = row.get(key)
        return v if isinstance(v, str) else ""
    link = f"{_s('job_url')} {_s('link')}".lower()
    company = _s("company").lower()
    return "willhire.co" in link or "(via willhire)" in company or "(contractor)" in company


def _apply_contingent_cap(result: pd.DataFrame) -> None:
    """Cap Apple contingent-marketplace listings at worth_a_look and say why
    (in place). Agencies and contract roles in general are never capped."""
    if "tier" not in result.columns:
        return
    mask = result["tier"].notna() & result.apply(is_contingent_marketplace, axis=1)
    if not mask.any():
        return
    result.loc[mask & (result["tier"] == "strong"), "tier"] = "worth_a_look"
    _append_note(result, mask, CONTINGENT_MARKETPLACE_NOTE)
    logger.info(f"Contingent marketplace: {int(mask.sum())} listing(s) capped at worth_a_look")


def _append_applicant_notes(result: pd.DataFrame) -> None:
    """Tiered jobs whose posting asks applicants to do something specific
    ("mention the word X in your application") get ONE line appended to their
    reasoning, so the candidate can act on it personally — the instruction
    itself was kept out of every model's input (in place)."""
    if "tier" not in result.columns or "description" not in result.columns:
        return
    for idx in result.index[result["tier"].notna()]:
        note = _sanitized(_raw_description(result.loc[idx])).applicant_note
        if note:
            _append_note(result, [idx], note, sep="\n")


# Approximate cost per 1M tokens (input/output)
# Used for rough billing estimates, not exact billing.
MODEL_PRICING = {
    # Anthropic
    "claude-opus-4-8":             {"input": 5.00, "output": 25.00},
    "claude-opus-4-7":             {"input": 5.00, "output": 25.00},
    "claude-opus-4-6":             {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6":           {"input": 3.00, "output": 15.00},
    # Official pricing per platform.claude.com models overview (fetched
    # 2026-07-25): opus-5 $5/$25 (opus-4-8 parity), fable-5 $10/$50 (2x! our
    # earlier 5/25 guess undercounted generation cost), sonnet-5 $3/$15
    # standard (intro $2/$10 through Aug 31, 2026 — we book the standard rate).
    "claude-opus-5-5":             {"input": 4.00, "output": 20.00},   # models overview, fetched 2026-09-26
    "claude-opus-5":               {"input": 5.00, "output": 25.00},   # legacy since Opus 5.5
    "claude-sonnet-5":             {"input": 2.00, "output": 10.00},    # $2/$10 per models overview 2026-09-01 (intro rate became standard)
    "claude-fable-5":              {"input": 10.00, "output": 50.00},
    "claude-fable-5-1":            {"input": 10.00, "output": 50.00},   # models overview, fetched 2026-09-01
    "claude-fable-5-2":            {"input": 8.00, "output": 40.00},    # Anthropic's 5.2 migration notes, 2026-11-16
    "claude-haiku-4-5-20251001":   {"input": 1.00, "output": 5.00},    # $1/$5 per models overview 2026-09-01
    # Gemini (free tier exists, but paid rates for reference)
    "gemini-2.5-flash":            {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite":       {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro":              {"input": 1.25, "output": 10.00},
    "gemini-3-pro-preview":        {"input": 1.25, "output": 10.00},
    "gemini-3-flash-preview":      {"input": 0.15, "output": 0.60},
    # OpenAI
    "gpt-5.2":                     {"input": 2.00, "output": 8.00},
    "gpt-5.2-pro":                 {"input": 10.00, "output": 30.00},
    "gpt-5":                       {"input": 10.00, "output": 30.00},
    "gpt-5-mini":                  {"input": 1.50, "output": 6.00},
    "gpt-4.1":                     {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini":                {"input": 0.40, "output": 1.60},
    "gpt-4o":                      {"input": 2.50, "output": 10.00},
    # Ollama (local — free)
}

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)

def _cost_for_model(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a model call. Returns 0.0 for local/unknown models."""
    if model.startswith("ollama/"):
        return 0.0
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    return (input_tokens / 1_000_000 * pricing["input"]) + (output_tokens / 1_000_000 * pricing["output"])


# Rate limits per model family (input tokens per minute).
# Conservative at ~80% of actual limit to avoid hitting the wall.
_MODEL_TPM_LIMITS = {
    "claude": 40_000,   # Actual: 50K input TPM
    "gemini": 1_000_000,  # Generous; gemini_utils has its own RateLimiter
    "gpt": 500_000,     # OpenAI is generally generous
}


class TokenBucketRateLimiter:
    """Simple token-bucket rate limiter for API calls.

    Tracks estimated input tokens consumed per minute and sleeps
    when the bucket is empty, preventing 429 errors.
    """

    def __init__(self, tokens_per_minute: int):
        self.tpm = tokens_per_minute
        self._tokens_used: list[tuple[float, int]] = []  # (timestamp, token_count)
        # Stage 1 calls from 4 worker threads share one limiter — guard state.
        self._state_lock = threading.Lock()

    def _prune_old(self):
        """Remove entries older than 60 seconds."""
        cutoff = time.time() - 60.0
        with self._state_lock:
            self._tokens_used = [(t, n) for t, n in self._tokens_used if t > cutoff]

    def tokens_remaining(self) -> int:
        self._prune_old()
        with self._state_lock:
            used = sum(n for _, n in self._tokens_used)
        return max(0, self.tpm - used)

    def wait_if_needed(self, estimated_tokens: int, progress_fn=None):
        """Block until there's room for estimated_tokens in the bucket."""
        if estimated_tokens >= self.tpm:
            # A single request bigger than the bucket would spin forever —
            # clamp to "wait for a full bucket" and let the API's own 429
            # handling take it from there.
            logger.warning(
                f"Rate limiter: single-request estimate {estimated_tokens} >= bucket "
                f"{self.tpm} — clamping (consider smaller batches/truncation)"
            )
            estimated_tokens = self.tpm
        while True:
            self._prune_old()
            remaining = self.tokens_remaining()
            if estimated_tokens <= remaining:
                break
            # Wait until the oldest entry expires
            if self._tokens_used:
                oldest_ts = self._tokens_used[0][0]
                wait = max(0.5, (oldest_ts + 60.0) - time.time() + 0.5)
            else:
                wait = 1.0
            wait = min(wait, 30.0)  # Cap at 30s
            if progress_fn:
                progress_fn(f"Rate limit: waiting {wait:.0f}s...")
            time.sleep(wait)

    def record(self, tokens: int):
        """Record that tokens were consumed."""
        with self._state_lock:
            self._tokens_used.append((time.time(), tokens))


def _get_rate_limiter(model: str) -> Optional[TokenBucketRateLimiter]:
    """Create a rate limiter appropriate for the model, or None for local models."""
    lower = model.lower()
    if lower.startswith("ollama/"):
        return None
    for prefix, tpm in _MODEL_TPM_LIMITS.items():
        if lower.startswith(prefix):
            return TokenBucketRateLimiter(tpm)
    return TokenBucketRateLimiter(100_000)  # Conservative default


class CascadeStats:
    """Track token usage and timing per stage."""

    def __init__(self):
        self.stage1_total = 0
        self.stage1_passed = 0
        self.stage1_filtered = 0
        self.stage2_total = 0
        self.stage2_high = 0
        self.stage2_potential = 0
        self.stage2_rejected = 0
        self.stage3_total = 0
        self.stage3_upgraded = 0
        self.stage3_confirmed = 0
        self.stage3_downgraded = 0
        self.start_time = None
        self.end_time = None
        # Token/cost tracking
        self.stage1_input_tokens = 0
        self.stage1_output_tokens = 0
        self.stage2_input_tokens = 0
        self.stage2_output_tokens = 0
        self.stage3_input_tokens = 0
        self.stage3_output_tokens = 0
        self.stage1_cost = 0.0
        self.stage2_cost = 0.0
        self.stage3_cost = 0.0

    def summary(self) -> dict:
        elapsed = 0
        if self.start_time and self.end_time:
            elapsed = (self.end_time - self.start_time).total_seconds()
        return {
            "stage1": {
                "total": self.stage1_total,
                "passed": self.stage1_passed,
                "filtered": self.stage1_filtered,
            },
            "stage2": {
                "total": self.stage2_total,
                "high_match": self.stage2_high,
                "potential_match": self.stage2_potential,
                "rejected": self.stage2_rejected,
            },
            "stage3": {
                "total": self.stage3_total,
                "upgraded": self.stage3_upgraded,
                "confirmed": self.stage3_confirmed,
                "downgraded": self.stage3_downgraded,
            },
            "elapsed_seconds": round(elapsed, 1),
        }


class CascadeFilter:
    """
    3-Stage Cascade AI Filtering Pipeline.

    Usage:
        cf = CascadeFilter(profile_text)
        result_df = cf.filter_jobs(df)
    """

    def __init__(
        self,
        profile_text: str,
        stage1_model: str = CASCADE_STAGE1_MODEL,
        stage2_model: str = CASCADE_STAGE2_MODEL,
        stage3_model: str = CASCADE_STAGE3_MODEL,
        stage1_prompt: Optional[str] = None,
        stage2_prompt: Optional[str] = None,
        stage3_prompt: Optional[str] = None,
        stage2_batch_size: int = CASCADE_STAGE2_BATCH_SIZE,
        stage1_fallbacks: Optional[list[str]] = None,
        stage2_fallbacks: Optional[list[str]] = None,
        stage3_fallbacks: Optional[list[str]] = None,
        progress_callback: Optional[Callable] = None,
        stop_check: Optional[Callable] = None,
        db=None,
    ):
        self.profile_text = profile_text
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.stage3_model = stage3_model
        # Auto-select prompts optimized for the model family (Claude uses XML tags)
        if stage1_prompt:
            self.stage1_prompt = stage1_prompt
        elif detect_model_family(stage1_model) == "claude":
            from .config import CASCADE_STAGE1_PROMPT_CLAUDE
            self.stage1_prompt = CASCADE_STAGE1_PROMPT_CLAUDE
        else:
            self.stage1_prompt = CASCADE_STAGE1_PROMPT
        if stage2_prompt:
            self.stage2_prompt = stage2_prompt
        elif detect_model_family(stage2_model) == "claude":
            self.stage2_prompt = STAGE2_PROMPT_CLAUDE
        else:
            self.stage2_prompt = DEFAULT_PROMPT_TEMPLATE
        self.stage3_prompt = stage3_prompt or get_stage3_prompt(stage3_model)
        self.stage2_batch_size = stage2_batch_size
        # Stage-level fallbacks (tried after primary model + its FALLBACK_CHAIN all fail)
        self.stage1_fallbacks = stage1_fallbacks if stage1_fallbacks is not None else list(CASCADE_STAGE1_FALLBACKS)
        self.stage2_fallbacks = stage2_fallbacks if stage2_fallbacks is not None else list(CASCADE_STAGE2_FALLBACKS)
        self.stage3_fallbacks = stage3_fallbacks if stage3_fallbacks is not None else list(CASCADE_STAGE3_FALLBACKS)
        # If the local Ollama server isn't answering, drop ollama/* fallbacks
        # up front: each dead fallback otherwise costs a full request timeout
        # PER Stage-1 failure (one outage burned 25 minutes this way).
        # One 3s probe replaces hundreds of doomed 60s waits.
        if any(m.startswith("ollama/") for m in self.stage1_fallbacks):
            from .gpt_filter import is_ollama_responsive
            if not is_ollama_responsive():
                logging.warning(
                    "Ollama server not responsive — removing ollama/* Stage-1 fallbacks for this run"
                )
                self.stage1_fallbacks = [m for m in self.stage1_fallbacks if not m.startswith("ollama/")]
        # LangFuse observability — a no-op unless keys are configured; every
        # tracer method is fail-safe (tracing must never break the pipeline).
        from .observability import CascadeTracer
        self._tracer = CascadeTracer()
        self.progress_callback = progress_callback
        self.stop_check = stop_check
        self.stats = CascadeStats()
        self._stopped = False
        self._db = db
        self._dismissed_companies = None  # lazy cache for company-scrutiny caution
        # Rate limiters: ONE bucket per SERVER-SIDE limit. Anthropic enforces
        # TPM PER MODEL (haiku / sonnet-5 / opus each draw from their own
        # bucket), so claude models are keyed by model name — a single shared
        # "claude" bucket made Stage-1 haiku traffic throttle Stage-3 opus for
        # nothing. Other providers keep a family bucket. Stages sharing a model
        # share its bucket (separate per-stage buckets would double-spend one
        # server limit; Stage 1's 4 parallel Claude calls once had none).
        # Actual usage is recorded in _call_llm_with_fallback against the
        # model that SERVED the call (a fallback charge lands on its own bucket).
        self._provider_limiters: dict = {}
        self._stage1_limiter = self._limiter_for(self.stage1_model)
        self._stage2_limiter = self._limiter_for(self.stage2_model)
        self._stage3_limiter = self._limiter_for(self.stage3_model)

    def _progress(self, text: str, pct: float):
        logger.info(text)
        if self.progress_callback:
            self.progress_callback(text, pct)

    def _should_stop(self) -> bool:
        if self._stopped:
            return True
        if self.stop_check and self.stop_check():
            self._stopped = True
            logger.info("Cascade filter: stop requested by user")
            self._progress("Stopping...", 0.99)
            return True
        return False

    def _limiter_for(self, model: str) -> Optional[TokenBucketRateLimiter]:
        """Bucket for the server-side limit this model draws from.

        Claude models are keyed per model (Anthropic TPM limits are per model);
        other providers share one bucket per family. Returns None for local
        (ollama) models.
        """
        fam = detect_model_family(model)
        key = model if fam == "claude" else fam
        if key not in self._provider_limiters:
            self._provider_limiters[key] = _get_rate_limiter(model)
        return self._provider_limiters[key]

    def _call_llm_with_fallback(
        self, prompt: str, primary_model: str, fallbacks: list[str], temperature: float = 0.0,
        usage_out: dict = None,
    ) -> tuple[str, str]:
        """Try primary model, then fallback models in order.

        Returns (answer_text, model_actually_used).
        Raises RuntimeError if ALL models fail.

        Records limiter usage against the model that ACTUALLY served the call —
        the call sites only pre-wait on the primary model's bucket, so a
        fallback's spend must land on the fallback's own bucket here.
        """
        est = _estimate_tokens(prompt) + 200
        # Try primary model (call_llm handles its own internal FALLBACK_CHAIN for Gemini)
        try:
            answer = call_llm(prompt, model=primary_model, temperature=temperature,
                              usage_out=usage_out)
            if answer:
                lim = self._limiter_for(primary_model)
                if lim:
                    lim.record(est)
                return answer, primary_model
        except Exception as e:
            logger.warning(f"Primary model {primary_model} failed: {e}")

        # Try stage-level fallbacks (skip if same as primary)
        for fb_model in fallbacks:
            if fb_model == primary_model:
                continue
            try:
                logger.info(f"Trying stage fallback model: {fb_model}")
                lim = self._limiter_for(fb_model)
                if lim:
                    # Nobody pre-waited on the fallback's bucket — do it here.
                    lim.wait_if_needed(est)
                answer = call_llm(prompt, model=fb_model, temperature=temperature,
                                  usage_out=usage_out)
                if answer:
                    logger.info(f"Fallback {fb_model} succeeded")
                    if lim:
                        lim.record(est)
                    return answer, fb_model
            except Exception as e:
                logger.warning(f"Fallback model {fb_model} failed: {e}")

        raise RuntimeError(f"All models failed for stage: {primary_model} + {fallbacks}")

    # ── Stage 1: Plausibility Pre-filter ─────────────────────────────────

    def _stage1_single(self, row: dict) -> bool:
        """Classify a single job as YES (pass) or NO (filter out)."""
        title = row.get("title", "")
        location = row.get("location", "")
        desc = _model_description(row)[:TRUNCATION_JOB_DESC_STAGE1]

        prompt = self.stage1_prompt.format(
            title=title,
            location=location,
            description_snippet=desc,
        )

        try:
            if self._stage1_limiter:
                est = _estimate_tokens(prompt) + 10
                # progress_fn makes throttle waits VISIBLE — Stage 1 spent
                # 6m12s (72% of its wall clock) in silent 42-48s TPM stalls on
                # 2026-07-18; the limiter was working correctly but unlogged.
                self._stage1_limiter.wait_if_needed(
                    est, progress_fn=lambda msg: logger.info(f"Stage 1 {msg}"))
                # record happens in _call_llm_with_fallback, on the ACTUAL model
            usage: dict = {}
            answer, _model_used = self._call_llm_with_fallback(
                prompt, self.stage1_model, self.stage1_fallbacks, temperature=0.0,
                usage_out=usage,
            )
            # Track tokens
            # Real usage when the provider reports it (thinking tokens count!)
            self.stats.stage1_input_tokens += usage.get("input_tokens") or _estimate_tokens(prompt)
            self.stats.stage1_output_tokens += usage.get("output_tokens") or _estimate_tokens(answer)
            # Ollama qwen3 may include <think>...</think> tags before the answer
            # Strip thinking tags if present
            clean = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            # First explicit verdict token wins. startswith("YES") silently
            # dropped any answer with a preamble word — a thinking model that
            # frames its reply ("Yes." / "Verdict: NO") must not lose the job.
            m = re.search(r"\b(YES|NO)\b", clean.upper())
            passed = m.group(1) == "YES" if m else True
            if not m:
                logger.warning(
                    f"Stage 1 unparseable verdict for '{title}': {clean[:80]!r} — failing open")
            if not usage:
                usage = {"input_tokens": _estimate_tokens(prompt),
                         "output_tokens": _estimate_tokens(answer)}
            self._tracer.log_stage1(
                row, prompt=prompt, completion=answer, model=_model_used,
                usage=usage,
                cost=_cost_for_model(_model_used, usage["input_tokens"], usage["output_tokens"]),
                passed=passed,
            )
            return passed
        except Exception as e:
            logger.warning(f"Stage 1 all models failed for '{title}': {e}")
            # On error, pass through to Stage 2 (don't lose jobs)
            return True

    def run_stage1(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run Stage 1 pre-filter on all jobs.

        Returns (passed_df, filtered_df).
        """
        self.stats.stage1_total = len(df)
        self._progress(f"Stage 1: Pre-filtering {len(df)} jobs with {self.stage1_model}...", 0.80)

        passed_indices = []
        filtered_indices = []

        # Run in parallel with ThreadPoolExecutor (the calls are I/O-bound)
        records = df.to_dict("records")
        idx_list = list(df.index)

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_idx = {
                executor.submit(self._stage1_single, rec): idx
                for rec, idx in zip(records, idx_list)
            }
            for i, future in enumerate(as_completed(future_to_idx)):
                if self._should_stop():
                    # Cancel pending futures and pass remaining through
                    for f in future_to_idx:
                        f.cancel()
                    remaining = set(idx_list) - set(passed_indices) - set(filtered_indices)
                    passed_indices.extend(remaining)
                    break

                idx = future_to_idx[future]
                try:
                    passed = future.result()
                    if passed:
                        passed_indices.append(idx)
                    else:
                        filtered_indices.append(idx)
                except Exception:
                    passed_indices.append(idx)  # On error, pass through

                if (i + 1) % 5 == 0 or i == len(records) - 1:
                    self._progress(
                        f"Stage 1: {i + 1}/{len(records)} processed",
                        0.80 + ((i + 1) / len(records)) * 0.03,
                    )

        passed_df = df.loc[passed_indices].copy() if passed_indices else pd.DataFrame()
        filtered_df = df.loc[filtered_indices].copy() if filtered_indices else pd.DataFrame()

        self.stats.stage1_passed = len(passed_df)
        self.stats.stage1_filtered = len(filtered_df)

        # Mark filtered jobs
        if not filtered_df.empty:
            filtered_df["match_level"] = "FILTERED_OUT"
            filtered_df["confidence"] = 0.0
            filtered_df["reason"] = "Stage 1: Pre-filter rejected (not relevant creative/media role)"
            filtered_df["rejection_rule"] = "STAGE1_PREFILTER"
            filtered_df["gpt_relevance"] = "irrelevant"
            filtered_df["gpt_reasoning"] = filtered_df["reason"]
            filtered_df["cascade_stage"] = 1

        logger.info(
            f"Stage 1 complete: {len(passed_df)} passed, {len(filtered_df)} filtered out "
            f"({self.stats.stage1_filtered}/{self.stats.stage1_total} = "
            f"{self.stats.stage1_filtered / max(self.stats.stage1_total, 1) * 100:.0f}% eliminated)"
        )
        return passed_df, filtered_df

    # ── Stage 2: Full Grading (Batched) ──────────────────────────────────

    def _resolve_domain_for_job(self, job: dict) -> tuple[str, str]:
        """Resolve domain key and description for a job."""
        kw_category = job.get("keyword_category") or ""
        matched_kw = job.get("matched_keyword") or ""
        title = job.get("title", "")
        domain = resolve_cv_domain(title, matched_kw) if not kw_category else kw_category
        domain_desc = DOMAIN_DESCRIPTIONS.get(domain, "general production")
        return domain, domain_desc

    def _stage2_batch(self, batch_df: pd.DataFrame) -> list[dict]:
        """Process a single batch through Stage 2."""
        jobs_block = ""
        for idx, job in enumerate(batch_df.to_dict("records"), 1):
            desc = _model_description(job)[:TRUNCATION_JOB_DESC_FILTER]
            _domain, domain_desc = self._resolve_domain_for_job(job)
            jobs_block += f"""
Job {idx}:
Title: {job.get('title', '')}
Company: {job.get('company', '')}
Location: {job.get('location', '')}
Date: {job.get('date_posted', 'Unknown')}
Domain Match: {domain_desc}
Description: {desc}
---
"""

        prompt = self.stage2_prompt.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
        prompt = prompt.replace("{profile_text}", self.profile_text)
        prompt = prompt.replace("{jobs_block}", jobs_block)

        usage: dict = {}
        answer, _model_used = self._call_llm_with_fallback(
            prompt, self.stage2_model, self.stage2_fallbacks, temperature=0.0,
            usage_out=usage,
        )
        self.stats.stage2_input_tokens += usage.get("input_tokens") or _estimate_tokens(prompt)
        self.stats.stage2_output_tokens += usage.get("output_tokens") or _estimate_tokens(answer)

        results = _parse_stage2_answer(answer, len(batch_df))

        if not usage:
            usage = {"input_tokens": _estimate_tokens(prompt),
                     "output_tokens": _estimate_tokens(answer)}
        self._tracer.log_stage2_batch(
            batch_df.to_dict("records"), prompt=prompt, completion=answer,
            model=_model_used, usage=usage,
            cost=_cost_for_model(_model_used, usage["input_tokens"], usage["output_tokens"]),
            results=results,
        )
        return results

    def _regrade_unparsed(self, df: pd.DataFrame, all_results: list) -> None:
        """One in-run re-grade for jobs the model skipped or answered without a
        verdict token (in place). Anything still unparsed stays STAGE2_PENDING
        (relevance NULL) — it is NOT retried by a later run (dedup skips the
        stored link; process_pending_db_jobs is manual only). Batch errors and
        user stops are not re-graded here (a failing API would just fail
        again)."""
        retry = [i for i, r in enumerate(all_results[:len(df)])
                 if r["match_level"] == "STAGE2_PENDING"
                 and str(r.get("reason", "")).startswith((_STAGE2_NO_LINE, _STAGE2_NO_VERDICT))]
        if not retry or self._should_stop():
            return
        logger.info(f"Stage 2: re-grading {len(retry)} job(s) with no parseable verdict")
        for start in range(0, len(retry), self.stage2_batch_size):
            if self._should_stop():
                return
            chunk = retry[start:start + self.stage2_batch_size]
            if self._stage2_limiter:
                self._stage2_limiter.wait_if_needed(
                    _estimate_tokens(self.profile_text) + len(chunk) * (TRUNCATION_JOB_DESC_FILTER // 4) + 200)
            try:
                regraded = self._stage2_batch(df.iloc[chunk])
            except Exception as e:
                logger.warning(f"Stage 2 re-grade failed (rows stay pending): {e}")
                continue
            for i, r in zip(chunk, regraded):
                all_results[i] = r

    def _enrich_thin_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rescue thin-description jobs before Stage 2 grading (Perplexity, optional).

        A job whose scraped description is too short gets graded IRRELEVANT
        ("INSUFFICIENT DATA") at Stage 2 through no fault of its own. When
        Perplexity is enabled, fetch a concise role summary and append it so
        Stage 2 can judge the role on its merits. Bounded + cached for cost;
        a complete no-op when USE_PERPLEXITY is False.
        """
        if df.empty or not (USE_PERPLEXITY and self._db is not None):
            return df
        if "description" not in df.columns:
            return df
        from .config import THIN_DESCRIPTION_THRESHOLD, PERPLEXITY_JD_RESCUE_MAX
        from .perplexity_utils import enrich_job_description

        df = df.copy()
        desc_len = df["description"].astype(str).str.len()
        thin_mask = desc_len < THIN_DESCRIPTION_THRESHOLD
        # Empty descriptions first (browser-scraped list views ship ''), then
        # merely-short ones, within the per-run budget.
        thin_idx = list(desc_len[thin_mask].sort_values().index)[:PERPLEXITY_JD_RESCUE_MAX]
        if not thin_idx:
            return df

        self._progress(f"Enriching {len(thin_idx)} thin-description jobs (Perplexity)...", 0.815)
        enriched = 0
        for idx in thin_idx:
            if self._should_stop():
                break
            row = df.loc[idx]
            try:
                summary = enrich_job_description(
                    str(row.get("title", "")), str(row.get("company", "")),
                    str(row.get("location", "")), self._db,
                )
            except Exception as e:
                logger.warning(f"Thin-desc enrichment failed for '{row.get('title')}': {e}")
                summary = None
            if summary:
                base = str(row.get("description", "")).strip()
                df.at[idx, "description"] = f"{base}\n\n[Role context (web search): {summary}]".strip()
                enriched += 1
        if enriched:
            logger.info(f"Thin-description rescue: enriched {enriched}/{len(thin_idx)} jobs before Stage 2")
        return df

    def run_stage2(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run Stage 2 full grading on passed jobs.

        Returns (high_match_df, rejected_df, potential_match_df).
        """
        if df.empty:
            empty = pd.DataFrame()
            return empty, empty, empty

        self.stats.stage2_total = len(df)
        self._progress(f"Stage 2: Grading {len(df)} jobs with {self.stage2_model}...", 0.83)

        all_results = []
        batch_size = self.stage2_batch_size
        total_batches = (len(df) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(df), batch_size):
            if self._should_stop():
                # Fill remaining with the retryable STAGE2_PENDING sentinel —
                # these jobs were never judged; marking them IRRELEVANT buried
                # them permanently (Stage 1/3 fail open; Stage 2 must too).
                remaining = len(df) - len(all_results)
                all_results.extend([{
                    "match_level": "STAGE2_PENDING",
                    "confidence": 0.0,
                    "reason": "Stopped by user before Stage 2 grading (retryable)",
                    "rejection_rule": "NONE",
                }] * remaining)
                break

            batch_num = batch_idx // batch_size + 1
            batch = df.iloc[batch_idx : batch_idx + batch_size]

            self._progress(
                f"Stage 2: Batch {batch_num}/{total_batches}",
                0.83 + (batch_num / total_batches) * 0.07,
            )

            # Rate limit: estimate tokens and wait if needed
            if self._stage2_limiter:
                est_tokens = _estimate_tokens(self.profile_text) + len(batch) * (TRUNCATION_JOB_DESC_FILTER // 4) + 200
                self._stage2_limiter.wait_if_needed(
                    est_tokens,
                    progress_fn=lambda msg: self._progress(msg, 0.83 + (batch_num / total_batches) * 0.07),
                )

            try:
                batch_results = self._stage2_batch(batch)
                all_results.extend(batch_results)
                # limiter record happens in _call_llm_with_fallback (actual model)
            except Exception as e:
                logger.error(f"Stage 2 batch {batch_num} failed: {e}")
                all_results.extend([{
                    "match_level": "STAGE2_PENDING",
                    "confidence": 0.0,
                    "reason": f"Stage 2 error (retryable): {e}",
                    "rejection_rule": "NONE",
                }] * len(batch))

        self._regrade_unparsed(df, all_results)

        # Apply results. STAGE2_PENDING maps to gpt_relevance=None so the DB
        # row keeps relevance NULL (ungraded, not rejected) — only the manual
        # main.process_pending_db_jobs (get_pending_jobs) re-grades it.
        df = df.copy()
        df["match_level"] = [r["match_level"] for r in all_results]
        df["confidence"] = [r["confidence"] for r in all_results]
        df["reason"] = [r["reason"] for r in all_results]
        df["rejection_rule"] = [r["rejection_rule"] for r in all_results]
        df["gpt_relevance"] = df["match_level"].apply(
            lambda ml: None if ml == "STAGE2_PENDING"
            else ("relevant" if ml in ("HIGH_MATCH", "POTENTIAL_MATCH", "WORTH_A_LOOK") else "irrelevant")
        )
        df["gpt_reasoning"] = df["reason"]
        df["cascade_stage"] = 2

        # Split into categories. Only POTENTIAL_MATCH goes to the premium Stage-3
        # judge. WORTH_A_LOOK keeps its Stage-2 verdict and lands directly in
        # the worth_a_look tier — premium-judging the deliberately-loose tier cost
        # ~$0.3-0.6/run for verdicts that almost never changed ("confirmed").
        # (Note: WORTH_A_LOOK rows therefore travel in rejected_df below — the
        # name is historical; they keep gpt_relevance='relevant' + their tier.)
        high_mask = df["match_level"] == "HIGH_MATCH"
        potential_mask = df["match_level"] == "POTENTIAL_MATCH"
        rejected_mask = ~high_mask & ~potential_mask

        high_df = df[high_mask].copy()
        potential_df = df[potential_mask].copy()
        rejected_df = df[rejected_mask].copy()

        self.stats.stage2_high = len(high_df)
        self.stats.stage2_potential = len(potential_df)
        self.stats.stage2_rejected = len(rejected_df)

        logger.info(
            f"Stage 2 complete: {len(high_df)} HIGH, {len(potential_df)} POTENTIAL, "
            f"{len(rejected_df)} rejected"
        )
        return high_df, rejected_df, potential_df

    # ── Stage 3: Judge (One at a time) ───────────────────────────────────

    def _company_caution(self, company: str) -> str:
        """Feedback loop: if the candidate has repeatedly dismissed roles from this
        company, tell the judge to scrutinize — a second look, NOT a ban.
        Requires 2+ prior dismissals so a one-off (e.g. 'already applied')
        doesn't taint the company."""
        if not company or self._db is None:
            return ""
        if self._dismissed_companies is None:
            try:
                self._dismissed_companies = self._db.get_dismissed_company_counts(min_count=2)
            except Exception:
                self._dismissed_companies = {}
        n = self._dismissed_companies.get(str(company).strip().lower(), 0)
        if n >= 2:
            return (f"CAUTION — the candidate has previously dismissed {n} role(s) from this "
                    f"company. Scrutinize fit carefully; do NOT upgrade on brand alone. "
                    f"If it resembles what the candidate already rejected, prefer WORTH_A_LOOK or REJECT.")
        return ""

    def _stage3_single(self, row: dict) -> dict:
        """Judge a single POTENTIAL_MATCH job."""
        desc = _model_description(row)[:TRUNCATION_JOB_DESC_FILTER]
        _domain, domain_desc = self._resolve_domain_for_job(row)

        # Optional Perplexity enrichment for Stage 3
        perplexity_context = ""
        if USE_PERPLEXITY and self._db:
            try:
                from .perplexity_utils import enrich_borderline_job, get_company_profile
                company = row.get("company", "")
                company_profile = get_company_profile(company, self._db)
                job_enrichment = enrich_borderline_job(row.get("title", ""), company, self._db)
                parts = []
                if company_profile:
                    parts.append(f"Company context: {company_profile}")
                if job_enrichment:
                    parts.append(f"Role context: {job_enrichment}")
                perplexity_context = "\n".join(parts)
            except Exception as e:
                logger.warning(f"Perplexity enrichment failed for Stage 3: {e}")

        # Feedback loop: prepend a scrutiny caution for repeatedly-dismissed
        # companies (piggybacks the perplexity_context field — no new prompt arg).
        caution = self._company_caution(row.get("company", ""))
        if caution:
            perplexity_context = (caution + "\n" + perplexity_context).strip()

        prompt = self.stage3_prompt.format(
            profile_text=self.profile_text,
            title=row.get("title", ""),
            company=row.get("company", ""),
            location=row.get("location", ""),
            description=desc,
            stage2_reason=row.get("reason", "No assessment available"),
            domain_hint=f"Domain match: {domain_desc}",
            perplexity_context=perplexity_context,
        )

        try:
            usage: dict = {}
            answer, _model_used = self._call_llm_with_fallback(
                prompt, self.stage3_model, self.stage3_fallbacks, temperature=0.0,
                usage_out=usage,
            )
            self.stats.stage3_input_tokens += usage.get("input_tokens") or _estimate_tokens(prompt)
            self.stats.stage3_output_tokens += usage.get("output_tokens") or _estimate_tokens(answer)

            # Parse JSON from response. raw_decode from the first "{" handles
            # braces inside string values (e.g. reason: "fit for {your team}")
            # and trailing prose, which the old r"\{[^{}]*\}" regex mis-extracted.
            json_start = answer.find("{")
            data = None
            if json_start != -1:
                data, _ = json.JSONDecoder().raw_decode(answer[json_start:])
            if isinstance(data, dict):
                # Validated, not trusted: a null final_level or an
                # apply_priority of "8" / "8/10" / null used to raise below
                # and turn a HIGH verdict into a 'Stage 3 error' row.
                final_level = str(data.get("final_level") or "CONFIRMED_POTENTIAL").upper()
                reason = data.get("reason") or "No reason given"
                priority = _stage3_priority(data.get("apply_priority", 5))

                # Recall-mode tier sort. Stage 3 keeps the job VISIBLE unless it
                # is a genuinely different job (REJECT) — a downgrade lands in the
                # lower "worth_a_look" tier, never in oblivion. Order matters:
                # check REJECT before WORTH so an explicit reject isn't caught by
                # a stray "worth" in the prose.
                if "HIGH" in final_level:
                    result = {
                        "match_level": "HIGH_MATCH",
                        "confidence": min(0.95, 0.7 + priority * 0.025),
                        "reason": f"Stage 3 upgraded: {reason}",
                        "rejection_rule": "NONE",
                        "apply_priority": priority,
                    }
                elif "REJECT" in final_level:
                    result = {
                        "match_level": "IRRELEVANT",
                        "confidence": 0.8,
                        "reason": f"Stage 3 rejected (different job): {reason}",
                        "rejection_rule": "STAGE3_REJECT",
                        "apply_priority": priority,
                    }
                elif "WORTH" in final_level or "DOWNGRAD" in final_level:
                    result = {
                        "match_level": "WORTH_A_LOOK",
                        "confidence": min(0.55, 0.35 + priority * 0.02),
                        "reason": f"Stage 3 worth-a-look: {reason}",
                        "rejection_rule": "NONE",
                        "apply_priority": priority,
                    }
                else:
                    # CONFIRMED_POTENTIAL
                    result = {
                        "match_level": "POTENTIAL_MATCH",
                        "confidence": min(0.85, 0.5 + priority * 0.035),
                        "reason": f"Stage 3 confirmed: {reason}",
                        "rejection_rule": "NONE",
                        "apply_priority": priority,
                    }
            else:
                logger.warning(f"Stage 3: No JSON found in response for '{row.get('title')}'")
                result = {
                    "match_level": "POTENTIAL_MATCH",
                    "confidence": 0.5,
                    "reason": f"Stage 3 parse error; original: {row.get('reason', '')}",
                    "rejection_rule": "NONE",
                    "apply_priority": 5,
                }
            if not usage:
                usage = {"input_tokens": _estimate_tokens(prompt),
                         "output_tokens": _estimate_tokens(answer)}
            self._tracer.log_stage3(
                row, prompt=prompt, completion=answer, model=_model_used,
                usage=usage,
                cost=_cost_for_model(_model_used, usage["input_tokens"], usage["output_tokens"]),
                result=result,
            )
            return result
        except Exception as e:
            logger.warning(f"Stage 3 failed for '{row.get('title')}': {e}")
            return {
                "match_level": "POTENTIAL_MATCH",
                "confidence": 0.5,
                "reason": f"Stage 3 error: {e}; original: {row.get('reason', '')}",
                "rejection_rule": "NONE",
                "apply_priority": 5,
            }

    def run_stage3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run Stage 3 judge on POTENTIAL_MATCH jobs (one at a time).

        Returns DataFrame with updated match decisions.
        """
        if df.empty:
            return df

        self.stats.stage3_total = len(df)
        self._progress(f"Stage 3: Judging {len(df)} borderline jobs with {self.stage3_model}...", 0.90)

        records = df.to_dict("records")
        results = []

        for i, rec in enumerate(records):
            if self._should_stop():
                # Keep remaining as POTENTIAL_MATCH (don't lose data)
                for remaining_rec in records[i:]:
                    results.append({
                        "match_level": "POTENTIAL_MATCH",
                        "confidence": 0.5,
                        "reason": f"Stopped by user; original: {remaining_rec.get('reason', '')}",
                        "rejection_rule": "NONE",
                        "apply_priority": 5,
                    })
                    self.stats.stage3_confirmed += 1
                break

            self._progress(
                f"Stage 3: {i + 1}/{len(records)} - {rec.get('title', '')}",
                0.90 + ((i + 1) / len(records)) * 0.08,
            )

            # Rate limit for Stage 3
            if self._stage3_limiter:
                est_tokens = _estimate_tokens(self.profile_text) + (TRUNCATION_JOB_DESC_FILTER // 4) + 200
                self._stage3_limiter.wait_if_needed(
                    est_tokens,
                    progress_fn=lambda msg: self._progress(msg, 0.90 + ((i + 1) / len(records)) * 0.08),
                )

            result = self._stage3_single(rec)

            # limiter record happens in _call_llm_with_fallback (actual model)
            results.append(result)

            if result["match_level"] == "HIGH_MATCH":
                self.stats.stage3_upgraded += 1
            elif result["match_level"] == "IRRELEVANT":
                self.stats.stage3_downgraded += 1
            else:
                self.stats.stage3_confirmed += 1

        df = df.copy()
        df["match_level"] = [r["match_level"] for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        df["reason"] = [r["reason"] for r in results]
        df["rejection_rule"] = [r["rejection_rule"] for r in results]
        df["gpt_relevance"] = df["match_level"].apply(
            lambda ml: "relevant" if ml in ("HIGH_MATCH", "POTENTIAL_MATCH", "WORTH_A_LOOK") else "irrelevant"
        )
        df["gpt_reasoning"] = df["reason"]
        df["cascade_stage"] = 3

        logger.info(
            f"Stage 3 complete: {self.stats.stage3_upgraded} upgraded, "
            f"{self.stats.stage3_confirmed} confirmed, {self.stats.stage3_downgraded} downgraded"
        )
        return df

    # ── Full Pipeline ────────────────────────────────────────────────────

    def _finalize(self, parts: list) -> pd.DataFrame:
        """Concatenate result partitions and stamp the ``tier`` column.

        Single choke point so every return path from ``filter_jobs`` (including
        the early stop-paths) gets a consistent tier assignment.
        """
        parts = [p for p in parts if p is not None and not p.empty]
        if not parts:
            return pd.DataFrame()
        result = pd.concat(parts, ignore_index=True)
        if "match_level" in result.columns:
            result["tier"] = result["match_level"].apply(assign_tier)
            # LOCATION POLICY: an on-site role in the capped metro
            # (location_priority == 3) is capped at worth_a_look no matter how
            # strong the match. Still fully visible; just not in the top tier.
            # The note text below is matched literally by postprocess.py.
            if "location" in result.columns:
                capped_mask = result.apply(
                    lambda r: location_priority(r.get("location"), r.get("title")) == 3, axis=1
                ) & (result["tier"] == "strong")
                if capped_mask.any():
                    result.loc[capped_mask, "tier"] = "worth_a_look"
                    _append_note(result, capped_mask,
                                 "[Capped-metro on-site — tiered to worth-a-look per location priority]")
                    logger.info(f"Location policy: {int(capped_mask.sum())} capped-metro on-site strong match(es) tiered down to worth_a_look")
            # SENIORITY POLICY: fully judge-arbitrated — no deterministic cap
            # here. The Stage-2/Stage-3 prompts REJECT confirmed people-managers
            # and cap ambiguous leadership titles at worth_a_look by reading the
            # JD (see config.py "Seniority Policy"). A title-only regex backstop
            # lived here through 2026-06-18 but kept springing brittle leaks, so
            # it was removed in favour of the agentic test.
            # APPLE LEVEL POLICY (added 2026-10-03) — the one deterministic
            # seniority rule, because it reads the employer's OWN published
            # level, not the title: a level inside APPLE_TARGET_BAND stays in
            # band; one level above is capped at worth_a_look; anything lower,
            # two or more levels above, or on the MGR ladder is rejected.
            _apply_apple_level_rule(result)
            # Apple contingent marketplace (WillHire / "(Contractor)"): capped.
            _apply_contingent_cap(result)
            # Applicant instructions found in the posting: one line for the candidate.
            _append_applicant_notes(result)
        return result

    def filter_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full 3-stage cascade pipeline.

        Returns a single DataFrame with all jobs annotated with:
        - match_level, confidence, reason, rejection_rule (standard columns)
        - gpt_relevance, gpt_reasoning (legacy compat)
        - cascade_stage (which stage produced the final decision)
        """
        result = None
        try:
            result = self._filter_jobs_inner(df)
            return result
        finally:
            # Short-lived process: flush + shutdown LangFuse explicitly or
            # buffered events are lost. finish() is fail-safe (no-op/swallow).
            self._tracer.finish(result)

    def _filter_jobs_inner(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        self.stats.start_time = datetime.now()

        # Pre-stage: Dedup by normalized title+company within the batch.
        # Same job from different sources (LinkedIn vs Greenhouse) has different
        # URLs but identical title+company — no need to run AI on duplicates.
        df = df.copy()
        _t = df["title"].astype(str).str.strip().str.lower()
        _c = df["company"].astype(str).str.strip().str.lower()
        if "job_url" in df.columns:
            _l = df["job_url"].astype(str)
        elif "link" in df.columns:
            _l = df["link"].astype(str)
        else:
            _l = pd.Series("", index=df.index)
        # Empty-company rows (RSS feeds) dedup by LINK — 'title|' collapsed
        # every distinct same-title job into one.
        df["_dedup_key"] = (_t + "|" + _c).where(_c != "", "link::" + _l)
        dup_mask = df.duplicated(subset=["_dedup_key"], keep="first")
        dedup_count = dup_mask.sum()
        if dedup_count > 0:
            logger.info(f"Cascade dedup: removing {dedup_count} cross-source duplicates (same title+company)")
            dup_df = df[dup_mask].copy()
            dup_df["match_level"] = "DUPLICATE"
            dup_df["confidence"] = 0.0
            dup_df["reason"] = "Cross-source duplicate (same title+company)"
            dup_df["rejection_rule"] = "DUPLICATE"
            dup_df["gpt_relevance"] = "irrelevant"
            dup_df["gpt_reasoning"] = dup_df["reason"]
            dup_df["cascade_stage"] = 0
            dup_df = dup_df.drop(columns=["_dedup_key"])
            df = df[~dup_mask].copy()
        else:
            dup_df = pd.DataFrame()
        df = df.drop(columns=["_dedup_key"])

        # Stage 1: Pre-filter
        passed_df, filtered_df = self.run_stage1(df)

        if passed_df.empty or self._should_stop():
            self.stats.end_time = datetime.now()
            self._log_summary()
            if self._stopped and not passed_df.empty:
                # Return what we have so far — passed jobs get a default annotation
                passed_df = passed_df.copy()
                passed_df["match_level"] = "POTENTIAL_MATCH"
                passed_df["confidence"] = 0.5
                passed_df["reason"] = "Stopped by user before grading"
                passed_df["rejection_rule"] = "NONE"
                passed_df["gpt_relevance"] = "relevant"
                passed_df["gpt_reasoning"] = passed_df["reason"]
                passed_df["cascade_stage"] = 1
                return self._finalize([dup_df, filtered_df, passed_df])
            return self._finalize([dup_df, filtered_df])

        # Thin-description rescue (Stage 1 -> 2): fill in role context for jobs
        # whose scraped description is too thin for Stage 2 to judge fairly.
        passed_df = self._enrich_thin_descriptions(passed_df)

        # Stage 2: Full grading
        high_df, rejected_df, potential_df = self.run_stage2(passed_df)

        if self._should_stop():
            # Return everything processed so far
            self.stats.end_time = datetime.now()
            self._log_summary()
            return self._finalize([dup_df, filtered_df, rejected_df, high_df, potential_df])

        # Stage 3: Judge borderline — CAPPED. Only the top slice gets the
        # premium judge (its only power is upgrading to strong); the overflow
        # keeps its Stage-2 POTENTIAL verdict and still surfaces in the
        # worth_a_look tier. Uncapped, one surge run made 69 premium calls.
        from .config import CASCADE_STAGE3_MAX_JOBS
        overflow_df = pd.DataFrame()
        if not potential_df.empty:
            skip_mask = pd.Series(False, index=potential_df.index)
            if "location" in potential_df.columns:
                skip_mask |= potential_df.apply(
                    lambda r: location_priority(r.get("location"), r.get("title")) == 3, axis=1
                )
            # NOTE: leadership/business titles are NO LONGER skipped here — they
            # now go through the premium judge so it can REJECT confirmed people-
            # managers (vs. cap ambiguous leadership at worth_a_look). Only the
            # capped-metro on-site location cap still short-circuits the premium judge.
            if skip_mask.any():
                logger.info(
                    f"Stage 3: skipping {int(skip_mask.sum())} jobs "
                    f"(capped-metro on-site — cannot reach strong by location policy)"
                )
                overflow_df = potential_df[skip_mask].copy()
                potential_df = potential_df[~skip_mask].copy()
        if len(potential_df) > CASCADE_STAGE3_MAX_JOBS:
            if "confidence" in potential_df.columns:
                potential_df = potential_df.sort_values("confidence", ascending=False)
            overflow_df = pd.concat(
                [overflow_df, potential_df.iloc[CASCADE_STAGE3_MAX_JOBS:]], ignore_index=False
            )
            potential_df = potential_df.iloc[:CASCADE_STAGE3_MAX_JOBS].copy()
            logger.info(
                f"Stage 3 cap: judging top {len(potential_df)}, "
                f"{len(overflow_df)} keep their Stage-2 verdict (still surfaced)"
            )
        judged_df = self.run_stage3(potential_df)

        # Combine all results (including pre-stage duplicates) + stamp tier
        result = self._finalize([dup_df, filtered_df, rejected_df, high_df, judged_df, overflow_df])

        self.stats.end_time = datetime.now()
        self._progress("Cascade complete!", 0.98)
        self._log_summary()

        return result

    def get_stats_for_ui(self) -> dict:
        """Return stats dict for display in the UI."""
        s = self.stats
        s1_cost = _cost_for_model(self.stage1_model, s.stage1_input_tokens, s.stage1_output_tokens)
        s2_cost = _cost_for_model(self.stage2_model, s.stage2_input_tokens, s.stage2_output_tokens)
        s3_cost = _cost_for_model(self.stage3_model, s.stage3_input_tokens, s.stage3_output_tokens)
        elapsed = 0
        if s.start_time and s.end_time:
            elapsed = (s.end_time - s.start_time).total_seconds()
        return {
            "stage1_model": self.stage1_model,
            "stage1_total": s.stage1_total,
            "stage1_passed": s.stage1_passed,
            "stage1_filtered": s.stage1_filtered,
            "stage1_tokens": s.stage1_input_tokens + s.stage1_output_tokens,
            "stage1_cost": s1_cost,
            "stage2_model": self.stage2_model,
            "stage2_total": s.stage2_total,
            "stage2_high": s.stage2_high,
            "stage2_potential": s.stage2_potential,
            "stage2_rejected": s.stage2_rejected,
            "stage2_tokens": s.stage2_input_tokens + s.stage2_output_tokens,
            "stage2_cost": s2_cost,
            "stage3_model": self.stage3_model,
            "stage3_total": s.stage3_total,
            "stage3_upgraded": s.stage3_upgraded,
            "stage3_confirmed": s.stage3_confirmed,
            "stage3_downgraded": s.stage3_downgraded,
            "stage3_tokens": s.stage3_input_tokens + s.stage3_output_tokens,
            "stage3_cost": s3_cost,
            "total_cost": s1_cost + s2_cost + s3_cost,
            "elapsed_seconds": round(elapsed, 1),
        }

    def _log_summary(self):
        s = self.stats.summary()

        # Compute costs
        self.stats.stage1_cost = _cost_for_model(
            self.stage1_model, self.stats.stage1_input_tokens, self.stats.stage1_output_tokens
        )
        self.stats.stage2_cost = _cost_for_model(
            self.stage2_model, self.stats.stage2_input_tokens, self.stats.stage2_output_tokens
        )
        self.stats.stage3_cost = _cost_for_model(
            self.stage3_model, self.stats.stage3_input_tokens, self.stats.stage3_output_tokens
        )
        total_cost = self.stats.stage1_cost + self.stats.stage2_cost + self.stats.stage3_cost

        s1_label = "FREE (local)" if self.stage1_model.startswith("ollama/") else f"${self.stats.stage1_cost:.4f}"
        s2_label = f"${self.stats.stage2_cost:.4f}"
        s3_label = f"${self.stats.stage3_cost:.4f}"

        logger.info(
            f"=== CASCADE SUMMARY ({s['elapsed_seconds']}s) ===\n"
            f"  Stage 1 [{self.stage1_model}]: {s['stage1']['total']} total → "
            f"{s['stage1']['passed']} passed, {s['stage1']['filtered']} filtered | "
            f"~{self.stats.stage1_input_tokens + self.stats.stage1_output_tokens} tokens | Cost: {s1_label}\n"
            f"  Stage 2 [{self.stage2_model}]: {s['stage2']['total']} total → "
            f"{s['stage2']['high_match']} HIGH, {s['stage2']['potential_match']} POTENTIAL, "
            f"{s['stage2']['rejected']} rejected | "
            f"~{self.stats.stage2_input_tokens + self.stats.stage2_output_tokens} tokens | Cost: {s2_label}\n"
            f"  Stage 3 [{self.stage3_model}]: {s['stage3']['total']} total → "
            f"{s['stage3']['upgraded']} upgraded, {s['stage3']['confirmed']} confirmed, "
            f"{s['stage3']['downgraded']} downgraded | "
            f"~{self.stats.stage3_input_tokens + self.stats.stage3_output_tokens} tokens | Cost: {s3_label}\n"
            f"  TOTAL FILTERING COST: ${total_cost:.4f}"
        )
