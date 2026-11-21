"""Tests for scraper_remote_boards — WeWorkRemotely RSS + Remote OK JSON API."""
from unittest.mock import patch, MagicMock

import pytest

from job_agent.scraper_remote_boards import (
    scrape_weworkremotely_rss,
    scrape_remoteok_api,
    scrape_remote_boards,
)


# --- WeWorkRemotely fixtures ------------------------------------------------

WWR_RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Remote Jobs</title>
    <item>
      <title>Acme Studios: Senior Video Producer</title>
      <link>https://weworkremotely.com/remote-jobs/acme-video-producer</link>
      <description>Lead video production for a remote-first studio.</description>
      <pubDate>Mon, 20 Apr 2026 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>WidgetCo: Backend Engineer</title>
      <link>https://weworkremotely.com/remote-jobs/widget-backend</link>
      <description>Python + Postgres. No producer work.</description>
      <pubDate>Mon, 20 Apr 2026 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>MediaHouse: Podcast Producer</title>
      <link>https://weworkremotely.com/remote-jobs/podcast-producer</link>
      <description>Produce daily podcasts.</description>
      <pubDate>Mon, 20 Apr 2026 10:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""


class TestWeWorkRemotely:
    def test_filters_by_keyword(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.text = WWR_RSS_SAMPLE
        mock_resp.raise_for_status = MagicMock()

        mock_get = MagicMock(return_value=mock_resp)
        monkeypatch.setattr(
            "job_agent.scraper_remote_boards._session.get", mock_get
        )
        # Disable politeness sleep to keep tests fast
        monkeypatch.setattr("job_agent.scraper_remote_boards.time.sleep", lambda _: None)

        jobs = scrape_weworkremotely_rss(keywords=["producer", "podcast"])

        # 2 feeds x 2 matching items = but seen_links dedups within the run
        # Each of the 3 feeds returns same 3 items; 2 match; dedupe to 2.
        titles = {j["title"] for j in jobs}
        assert "Senior Video Producer" in titles
        assert "Podcast Producer" in titles
        assert "Backend Engineer" not in titles

    def test_splits_company_from_title(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.text = WWR_RSS_SAMPLE
        mock_resp.raise_for_status = MagicMock()

        monkeypatch.setattr(
            "job_agent.scraper_remote_boards._session.get",
            MagicMock(return_value=mock_resp),
        )
        monkeypatch.setattr("job_agent.scraper_remote_boards.time.sleep", lambda _: None)

        jobs = scrape_weworkremotely_rss(keywords=["producer"])
        by_title = {j["title"]: j for j in jobs}

        assert by_title["Senior Video Producer"]["company"] == "Acme Studios"
        assert by_title["Podcast Producer"]["company"] == "MediaHouse"

    def test_network_error_returns_empty(self, monkeypatch):
        def fail(*_a, **_k):
            raise Exception("network down")

        monkeypatch.setattr(
            "job_agent.scraper_remote_boards._session.get", fail
        )
        monkeypatch.setattr("job_agent.scraper_remote_boards.time.sleep", lambda _: None)

        jobs = scrape_weworkremotely_rss(keywords=["producer"])
        assert jobs == []

    def test_dedupes_across_feeds(self, monkeypatch):
        # Same item returned by all 3 feeds — should appear once
        mock_resp = MagicMock()
        mock_resp.text = WWR_RSS_SAMPLE
        mock_resp.raise_for_status = MagicMock()

        monkeypatch.setattr(
            "job_agent.scraper_remote_boards._session.get",
            MagicMock(return_value=mock_resp),
        )
        monkeypatch.setattr("job_agent.scraper_remote_boards.time.sleep", lambda _: None)

        jobs = scrape_weworkremotely_rss(keywords=["producer"])
        links = [j["job_url"] for j in jobs]
        assert len(links) == len(set(links))  # no duplicates


# --- Remote OK fixtures -----------------------------------------------------

REMOTEOK_API_SAMPLE = [
    # First element is the usual legal notice — no "position" field
    {"legal": "This API is free to use", "terms": "..."},
    {
        "id": "001",
        "position": "Senior Video Producer",
        "company": "StreamCo",
        "location": "Remote",
        "description": "<p>Lead video production.</p>",
        "tags": ["video", "producer", "remote"],
        "date": "2026-04-20T10:00:00+00:00",
        "url": "https://remoteok.com/remote-jobs/001",
    },
    {
        "id": "002",
        "position": "Backend Engineer",
        "company": "WidgetCo",
        "location": "",
        "description": "<p>Python + Postgres</p>",
        "tags": ["python", "backend"],
        "date": "2026-04-20T11:00:00+00:00",
        "url": "https://remoteok.com/remote-jobs/002",
    },
    {
        "id": "003",
        "position": "Growth Marketer",
        "company": "MediaHouse",
        "location": "Remote",
        "description": "<p>Drive growth.</p>",
        "tags": ["marketing", "producer", "growth"],  # note: 'producer' in tags
        "date": "2026-04-20T12:00:00Z",  # Z suffix variant
        "url": "https://remoteok.com/remote-jobs/003",
        "apply_url": "https://mediahouse.com/apply",
    },
]


class TestRemoteOK:
    def test_filters_by_position_title(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.json = MagicMock(return_value=REMOTEOK_API_SAMPLE)
        mock_resp.raise_for_status = MagicMock()

        monkeypatch.setattr(
            "job_agent.scraper_remote_boards._session.get",
            MagicMock(return_value=mock_resp),
        )

        jobs = scrape_remoteok_api(keywords=["producer"])
        titles = {j["title"] for j in jobs}

        # Direct title match
        assert "Senior Video Producer" in titles
        # Tag fallback match (position is "Growth Marketer" but tags include 'producer')
        assert "Growth Marketer" in titles
        # No match
        assert "Backend Engineer" not in titles

    def test_handles_iso_z_suffix(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.json = MagicMock(return_value=REMOTEOK_API_SAMPLE)
        mock_resp.raise_for_status = MagicMock()

        monkeypatch.setattr(
            "job_agent.scraper_remote_boards._session.get",
            MagicMock(return_value=mock_resp),
        )

        jobs = scrape_remoteok_api(keywords=["producer"])
        by_title = {j["title"]: j for j in jobs}
        assert by_title["Growth Marketer"]["date_posted"] == "2026-04-20"
        assert by_title["Senior Video Producer"]["date_posted"] == "2026-04-20"

    def test_defaults_missing_location_to_remote(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.json = MagicMock(return_value=REMOTEOK_API_SAMPLE)
        mock_resp.raise_for_status = MagicMock()

        monkeypatch.setattr(
            "job_agent.scraper_remote_boards._session.get",
            MagicMock(return_value=mock_resp),
        )

        jobs = scrape_remoteok_api(keywords=["producer"])
        by_title = {j["title"]: j for j in jobs}
        # Both have "Remote" or fallback
        assert by_title["Senior Video Producer"]["location"] == "Remote"
        assert by_title["Growth Marketer"]["location"] == "Remote"

    def test_ignores_non_dict_entries(self, monkeypatch):
        # The legal notice (first element) should not blow up
        mock_resp = MagicMock()
        mock_resp.json = MagicMock(
            return_value=[{"legal": "x"}, "notastring", None,
                          {"position": "Producer", "company": "X", "url": "http://x"}]
        )
        mock_resp.raise_for_status = MagicMock()

        monkeypatch.setattr(
            "job_agent.scraper_remote_boards._session.get",
            MagicMock(return_value=mock_resp),
        )

        jobs = scrape_remoteok_api(keywords=["producer"])
        assert len(jobs) == 1
        assert jobs[0]["title"] == "Producer"

    def test_network_error_returns_empty(self, monkeypatch):
        def fail(*_a, **_k):
            raise Exception("api down")

        monkeypatch.setattr(
            "job_agent.scraper_remote_boards._session.get", fail
        )

        jobs = scrape_remoteok_api(keywords=["producer"])
        assert jobs == []


# --- Orchestrator -----------------------------------------------------------


class TestOrchestrator:
    def test_empty_when_both_sources_empty(self, monkeypatch):
        monkeypatch.setattr(
            "job_agent.scraper_remote_boards.scrape_weworkremotely_rss",
            lambda *_a, **_k: [],
        )
        monkeypatch.setattr(
            "job_agent.scraper_remote_boards.scrape_remoteok_api",
            lambda *_a, **_k: [],
        )

        df = scrape_remote_boards(keywords=["producer"])
        assert df.empty

    def test_merges_both_sources(self, monkeypatch):
        monkeypatch.setattr(
            "job_agent.scraper_remote_boards.scrape_weworkremotely_rss",
            lambda *_a, **_k: [{"title": "A", "company": "X", "site": "weworkremotely",
                                 "job_url": "http://a", "description": "", "location": "",
                                 "date_posted": "", "department": ""}],
        )
        monkeypatch.setattr(
            "job_agent.scraper_remote_boards.scrape_remoteok_api",
            lambda *_a, **_k: [{"title": "B", "company": "Y", "site": "remoteok",
                                 "job_url": "http://b", "description": "", "location": "",
                                 "date_posted": "", "department": ""}],
        )

        df = scrape_remote_boards(keywords=["producer"])
        assert len(df) == 2
        assert set(df["site"]) == {"weworkremotely", "remoteok"}
