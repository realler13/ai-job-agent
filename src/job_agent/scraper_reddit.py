"""
Reddit scraper using PRAW (Python Reddit API Wrapper).

Covers producer/video/podcast job threads across active subreddits. Uses
application-only (read-only) OAuth — no user login needed.

Setup: create a "script" app at https://www.reddit.com/prefs/apps, then add
to .env:
    REDDIT_CLIENT_ID=...
    REDDIT_CLIENT_SECRET=...
    REDDIT_USER_AGENT="AIJobAgent/1.0 (by /u/{username})"

If the env vars aren't set, scrape_reddit() returns [] at once (INFO log) —
the scheduled run won't fail. Once credentials are added, Reddit jumps in
automatically.

Filtering strategy (cut freelance/junk, keep FT producer roles):
  1. Flair / title must be [HIRING] (not [FOR HIRE])
  2. Reject freelance signals in title: gig, $/edit, per-video
  3. Filter by keyword match on title (same as other scrapers)
  4. Scope to posts from the last 7 days
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone

from .ats_scraper import _matches_keywords

logger = logging.getLogger(__name__)

# Subreddits where full-time producer / video / podcast roles appear.
REDDIT_JOB_SUBREDDITS = [
    "forhire",              # primary job marketplace, strict [HIRING] enforcement
    "editors",              # pro editors; occasional full-time postings
    "podcasting",           # podcast community; occasional FT
    "VideoProduction",      # commercial/broadcast producers
    "entertainmentcareers", # entertainment industry jobs
    "hiring",               # general cross-industry
]

MAX_POSTS_PER_SUB = 50        # how many recent posts to scan per subreddit
MAX_AGE_DAYS = 7              # only consider posts from the last week

# Title tokens that mean "I am hiring" (accept)
_HIRING_TAGS = ("[hiring]", "[hiring!]", "[ft]", "[full-time]", "[fulltime]")
# Title tokens that mean "I am for hire" — those are candidates, not jobs (reject)
_FORHIRE_TAGS = ("[for hire]", "[forhire]", "[for-hire]")
# Freelance/gig signals that usually mean this isn't a FT role
_FREELANCE_SIGNALS = re.compile(
    r"\b(?:gig|one[- ]off|per[- ]video|per[- ]edit|freelance only|project[- ]based)\b"
    # Rate patterns — no leading \b because "$" is a non-word character
    r"|\$\d+\s*/?\s*(?:hr|hour|video|edit)\b",
    re.I,
)


def _reddit_client():
    """Lazy-construct a PRAW read-only client, or None if env vars missing."""
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT")
    if not (cid or csec or ua):
        # Not set up is a normal state. INFO, not WARNING: a warning on every
        # run for a source nobody enabled is noise that hides the warnings
        # that matter.
        logger.info("Reddit: no credentials configured — skipping.")
        return None
    if not (cid and csec and ua):
        # Half-configured is a real mistake worth a warning.
        logger.warning(
            "Reddit: credentials incomplete "
            "(need REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT). Skipping."
        )
        return None
    try:
        import praw
        return praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)
    except ImportError:
        logger.warning("Reddit: praw not installed. Skipping.")
        return None
    except Exception as e:
        logger.warning(f"Reddit: client init failed: {e}")
        return None


def _title_is_hiring_post(title: str) -> bool:
    """Accept only titles that look like a hiring post (not for-hire offers)."""
    t = title.lower()
    if any(tag in t for tag in _FORHIRE_TAGS):
        return False
    # Either an explicit [HIRING] tag OR a title like "Hiring: ..."
    if any(tag in t for tag in _HIRING_TAGS):
        return True
    if re.search(r"\b(hiring|seeking|looking\s+for|wanted)\b", t):
        return True
    return False


def _strip_hiring_tags(title: str) -> str:
    """Remove [HIRING] / [FT] / [REMOTE] style prefixes so the cleaned title
    goes into the cascade looking like a regular job title."""
    return re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", title).strip()


def _freelance_signal(text: str) -> bool:
    """True if text looks like a freelance/gig ad rather than full-time hiring."""
    return bool(_FREELANCE_SIGNALS.search(text or ""))


def _post_age_days(created_utc: float) -> float:
    now = datetime.now(timezone.utc).timestamp()
    return (now - created_utc) / 86400.0


def scrape_reddit(keywords: list[str], max_age_days: int = MAX_AGE_DAYS) -> list[dict]:
    """Scan the configured subreddits for recent [HIRING] posts that match
    the keyword set. Returns standardized job dicts.

    Silent no-op when Reddit credentials aren't configured — so the pipeline
    keeps running on partial coverage.
    """
    reddit = _reddit_client()
    if reddit is None:
        return []

    results: list[dict] = []

    for sub_name in REDDIT_JOB_SUBREDDITS:
        try:
            sub = reddit.subreddit(sub_name)
            posts = list(sub.new(limit=MAX_POSTS_PER_SUB))
        except Exception as e:
            logger.warning(f"Reddit /r/{sub_name} fetch error: {e}")
            continue

        for post in posts:
            try:
                title = post.title or ""
                if not _title_is_hiring_post(title):
                    continue

                if _post_age_days(post.created_utc) > max_age_days:
                    continue

                cleaned_title = _strip_hiring_tags(title)
                if not _matches_keywords(cleaned_title, keywords):
                    # Also try matching against the body preview
                    body = (post.selftext or "")[:500]
                    if not _matches_keywords(body, keywords):
                        continue

                # Skip obvious freelance posts
                body_full = post.selftext or ""
                if _freelance_signal(title) or _freelance_signal(body_full[:1000]):
                    continue

                date_posted = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime("%Y-%m-%d")

                results.append({
                    "title": cleaned_title,
                    "company": f"r/{sub_name}",  # poster is usually individual
                    "location": "",  # not reliably structured in Reddit text
                    "job_url": f"https://reddit.com{post.permalink}",
                    "description": body_full[:3000],
                    "date_posted": date_posted,
                    "site": "reddit",
                    "department": sub_name,
                })
            except Exception as e:
                logger.debug(f"Reddit post parse error in /r/{sub_name}: {e}")
                continue

        time.sleep(0.3)  # polite between subreddits

    logger.info(f"Reddit: {len(results)} matching hiring posts across {len(REDDIT_JOB_SUBREDDITS)} subreddits")
    return results
