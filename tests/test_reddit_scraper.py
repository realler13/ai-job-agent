"""Tests for scraper_reddit — PRAW-based subreddit scanner.

PRAW is mocked throughout so tests don't hit Reddit's API.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from job_agent.scraper_reddit import (
    scrape_reddit,
    _title_is_hiring_post,
    _strip_hiring_tags,
    _freelance_signal,
    _reddit_client,
)


# --- _title_is_hiring_post -------------------------------------------------


class TestTitleIsHiring:
    def test_explicit_hiring_tag(self):
        assert _title_is_hiring_post("[HIRING] Senior Video Producer")
        assert _title_is_hiring_post("[hiring] podcast producer")
        assert _title_is_hiring_post("[FT] Production coordinator")

    def test_for_hire_rejected(self):
        assert not _title_is_hiring_post("[FOR HIRE] Video editor, 7 years exp")
        assert not _title_is_hiring_post("[forhire] producer available")

    def test_informal_hiring_phrase(self):
        assert _title_is_hiring_post("Hiring a producer for a new podcast")
        assert _title_is_hiring_post("Looking for a creative producer")
        assert _title_is_hiring_post("Seeking video editor")

    def test_unrelated_post(self):
        assert not _title_is_hiring_post("What microphone do you use?")
        assert not _title_is_hiring_post("My first documentary release")


# --- _strip_hiring_tags ----------------------------------------------------


class TestStripHiringTags:
    def test_strips_single_tag(self):
        assert _strip_hiring_tags("[HIRING] Video Producer") == "Video Producer"

    def test_strips_multiple_tags(self):
        assert _strip_hiring_tags("[HIRING][Remote] Producer") == "Producer"

    def test_no_tags_passthrough(self):
        assert _strip_hiring_tags("Regular title") == "Regular title"


# --- _freelance_signal -----------------------------------------------------


class TestFreelanceSignal:
    def test_detects_gig(self):
        assert _freelance_signal("looking for a gig")
        assert _freelance_signal("one-off project for a producer")

    def test_detects_per_video_rate(self):
        assert _freelance_signal("$50/video for editing work")
        assert _freelance_signal("$30 per edit, short turnaround")

    def test_ignores_full_time(self):
        assert not _freelance_signal("Full-time salaried producer role, benefits included")


# --- _reddit_client --------------------------------------------------------


class TestRedditClient:
    def test_missing_creds_returns_none(self, monkeypatch):
        # Wipe all three env vars
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
        assert _reddit_client() is None

    def test_partial_creds_returns_none(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "xxx")
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
        assert _reddit_client() is None


# --- scrape_reddit (end-to-end with mocked PRAW) --------------------------


def _make_fake_submission(title: str, selftext: str = "", age_hours: float = 1.0):
    s = MagicMock()
    s.title = title
    s.selftext = selftext
    s.created_utc = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).timestamp()
    s.permalink = f"/r/test/comments/abc/{title[:20].replace(' ', '_')}/"
    s.score = 5
    return s


class TestScrapeReddit:
    def test_returns_empty_without_creds(self, monkeypatch):
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        assert scrape_reddit(keywords=["producer"]) == []

    @patch("job_agent.scraper_reddit.time.sleep", lambda _: None)
    @patch("job_agent.scraper_reddit._reddit_client")
    def test_filters_hiring_posts(self, mock_client_factory):
        fake_reddit = MagicMock()
        sub = MagicMock()
        # Three posts: 1 matching hiring post, 1 for-hire (rejected), 1 unrelated
        sub.new.return_value = iter([
            _make_fake_submission("[HIRING] Senior Video Producer - Remote, FT"),
            _make_fake_submission("[FOR HIRE] Editor with 7 yrs exp"),
            _make_fake_submission("What camera do you use?"),
        ])
        fake_reddit.subreddit.return_value = sub
        mock_client_factory.return_value = fake_reddit

        jobs = scrape_reddit(keywords=["producer", "video producer"])

        # Only the [HIRING] post should come through, once per subreddit (7 subs by default)
        # With all 7 subreddits returning the same 3 posts, we get 7 matching jobs
        assert all(j["site"] == "reddit" for j in jobs)
        assert all("Senior Video Producer" in j["title"] for j in jobs)
        # Title should have [HIRING] stripped
        assert all(not j["title"].startswith("[") for j in jobs)

    @patch("job_agent.scraper_reddit.time.sleep", lambda _: None)
    @patch("job_agent.scraper_reddit._reddit_client")
    def test_rejects_stale_posts(self, mock_client_factory):
        fake_reddit = MagicMock()
        sub = MagicMock()
        # Post is 30 days old — should be skipped
        sub.new.return_value = iter([
            _make_fake_submission("[HIRING] Producer wanted", age_hours=24 * 30),
        ])
        fake_reddit.subreddit.return_value = sub
        mock_client_factory.return_value = fake_reddit

        jobs = scrape_reddit(keywords=["producer"], max_age_days=7)
        assert jobs == []

    @patch("job_agent.scraper_reddit.time.sleep", lambda _: None)
    @patch("job_agent.scraper_reddit._reddit_client")
    def test_rejects_freelance_signals(self, mock_client_factory):
        fake_reddit = MagicMock()
        sub = MagicMock()
        sub.new.return_value = iter([
            _make_fake_submission("[HIRING] Video producer $50/video one-off",
                                   selftext="looking for freelance only"),
        ])
        fake_reddit.subreddit.return_value = sub
        mock_client_factory.return_value = fake_reddit

        jobs = scrape_reddit(keywords=["producer"])
        assert jobs == []

    @patch("job_agent.scraper_reddit.time.sleep", lambda _: None)
    @patch("job_agent.scraper_reddit._reddit_client")
    def test_subreddit_fetch_error_skipped(self, mock_client_factory):
        fake_reddit = MagicMock()
        sub = MagicMock()
        sub.new.side_effect = Exception("Reddit is grumpy")
        fake_reddit.subreddit.return_value = sub
        mock_client_factory.return_value = fake_reddit

        # Should not raise — logs warning and returns []
        jobs = scrape_reddit(keywords=["producer"])
        assert jobs == []
