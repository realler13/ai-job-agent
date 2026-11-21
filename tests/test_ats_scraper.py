"""Tests for the ATS (Greenhouse/Lever) scraper module."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from job_agent.ats_scraper import (
    scrape_greenhouse,
    scrape_lever,
    scrape_ats_boards,
    _strip_html,
    _matches_keywords,
    _extract_brand_from_metadata,
)


# === Fixtures ===

GREENHOUSE_RESPONSE = {
    "jobs": [
        {
            "title": "Senior Content Producer",
            "location": {"name": "San Francisco, CA"},
            "absolute_url": "https://boards.greenhouse.io/airbnb/jobs/123",
            "content": "<p>We are looking for a <strong>Content Producer</strong>.</p>",
            "updated_at": "2025-01-15T10:00:00Z",
            "departments": [{"name": "Marketing"}],
        },
        {
            "title": "Software Engineer",
            "location": {"name": "New York, NY"},
            "absolute_url": "https://boards.greenhouse.io/airbnb/jobs/456",
            "content": "<p>Build distributed systems.</p>",
            "updated_at": "2025-01-14T10:00:00Z",
            "departments": [{"name": "Engineering"}],
        },
        {
            "title": "Video Project Manager",
            "location": {"name": "Remote"},
            "absolute_url": "https://boards.greenhouse.io/airbnb/jobs/789",
            "content": "Manage video production workflows.",
            "updated_at": "2025-01-13T10:00:00Z",
            "departments": [],
        },
    ]
}

LEVER_RESPONSE = [
    {
        "text": "Creative Producer",
        "categories": {"location": "Los Angeles, CA", "department": "Content"},
        "hostedUrl": "https://jobs.lever.co/company/abc-123",
        "descriptionPlain": "Lead creative projects across multiple channels.",
        "createdAt": 1705000000000,
    },
    {
        "text": "Backend Developer",
        "categories": {"location": "Remote", "department": "Engineering"},
        "hostedUrl": "https://jobs.lever.co/company/def-456",
        "descriptionPlain": "Build APIs and services.",
        "createdAt": 1704900000000,
    },
]

KEYWORDS = ["Content Producer", "Video Project Manager", "Creative Producer"]


# === Tests ===


class TestMatchesKeywords:
    def test_exact_match(self):
        assert _matches_keywords("Content Producer", ["Content Producer"])

    def test_case_insensitive(self):
        assert _matches_keywords("CONTENT PRODUCER", ["content producer"])

    def test_partial_phrase_in_longer_title(self):
        assert _matches_keywords("Senior Content Producer - West Coast", ["Content Producer"])

    def test_no_match(self):
        assert not _matches_keywords("Software Engineer", ["Content Producer"])

    def test_empty_keywords(self):
        assert not _matches_keywords("Any Title", [])


class TestStripHtml:
    def test_strips_tags(self):
        assert _strip_html("<p>Hello <strong>world</strong></p>") == "Hello world"

    def test_empty_input(self):
        assert _strip_html("") == ""
        assert _strip_html(None) == ""

    def test_plain_text_passthrough(self):
        assert _strip_html("No HTML here") == "No HTML here"


class TestScrapeGreenhouse:
    @patch("job_agent.ats_scraper.requests.get")
    def test_parses_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = GREENHOUSE_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_greenhouse("airbnb", "Airbnb", KEYWORDS)

        assert len(results) == 2  # "Software Engineer" filtered out
        assert results[0]["title"] == "Senior Content Producer"
        assert results[0]["company"] == "Airbnb"
        assert results[0]["location"] == "San Francisco, CA"
        assert results[0]["site"] == "greenhouse"
        assert results[0]["department"] == "Marketing"
        assert "Content Producer" in results[0]["description"]
        assert "<p>" not in results[0]["description"]  # HTML stripped

        assert results[1]["title"] == "Video Project Manager"
        assert results[1]["department"] == ""  # No departments

    @patch("job_agent.ats_scraper.requests.get")
    def test_handles_api_error(self, mock_get):
        mock_get.side_effect = Exception("Connection timeout")
        results = scrape_greenhouse("badtoken", "Bad", KEYWORDS)
        assert results == []

    @patch("job_agent.ats_scraper.requests.get")
    def test_handles_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_resp

        results = scrape_greenhouse("nonexistent", "Ghost", KEYWORDS)
        assert results == []

    @patch("job_agent.ats_scraper.requests.get")
    def test_brand_metadata_overrides_company(self, mock_get):
        """For multi-brand boards (PMC hosts THR/Deadline/Variety/etc. under the
        'pmc' Greenhouse token), the per-posting Brand Name should win over the
        configured company name."""
        pmc_response = {
            "jobs": [
                {
                    "title": "Senior Content Producer",
                    "location": {"name": "Los Angeles, CA"},
                    "absolute_url": "https://boards.greenhouse.io/pmc/jobs/1",
                    "content": "Lead video production for THR.",
                    "updated_at": "2026-04-20T10:00:00Z",
                    "departments": [{"name": "Editorial"}],
                    "metadata": [
                        {"id": 1, "name": "Brand Name", "type": "short_text",
                         "value": "The Hollywood Reporter"},
                        {"id": 2, "name": "Team Department", "type": "short_text",
                         "value": "Video"},
                    ],
                },
                {
                    "title": "Video Project Manager",
                    "location": {"name": "New York, NY"},
                    "absolute_url": "https://boards.greenhouse.io/pmc/jobs/2",
                    "content": "Coordinate editorial video.",
                    "updated_at": "2026-04-20T11:00:00Z",
                    "departments": [],
                    "metadata": [
                        {"id": 1, "name": "Brand Name", "type": "short_text",
                         "value": "Deadline"},
                    ],
                },
                {
                    "title": "Creative Producer",
                    "location": {"name": "Remote"},
                    "absolute_url": "https://boards.greenhouse.io/pmc/jobs/3",
                    "content": "Run Variety video.",
                    "updated_at": "2026-04-20T12:00:00Z",
                    "departments": [],
                    # No metadata at all — fallback to configured company name
                },
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = pmc_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_greenhouse("pmc", "Penske Media", KEYWORDS)

        assert len(results) == 3
        assert results[0]["company"] == "The Hollywood Reporter"
        assert results[1]["company"] == "Deadline"
        # No brand metadata → falls back to configured company name
        assert results[2]["company"] == "Penske Media"


class TestExtractBrandFromMetadata:
    def test_returns_brand_name(self):
        job = {"metadata": [
            {"id": 1, "name": "Brand Name", "value": "Variety"},
        ]}
        assert _extract_brand_from_metadata(job) == "Variety"

    def test_case_insensitive_name_match(self):
        job = {"metadata": [{"id": 1, "name": "brand name", "value": "Billboard"}]}
        assert _extract_brand_from_metadata(job) == "Billboard"

    def test_alternate_brand_key(self):
        # Accept "Brand" in addition to "Brand Name"
        job = {"metadata": [{"id": 1, "name": "Brand", "value": "Rolling Stone"}]}
        assert _extract_brand_from_metadata(job) == "Rolling Stone"

    def test_no_metadata(self):
        assert _extract_brand_from_metadata({}) == ""

    def test_metadata_wrong_shape(self):
        assert _extract_brand_from_metadata({"metadata": "not-a-list"}) == ""

    def test_metadata_no_brand_field(self):
        job = {"metadata": [
            {"id": 1, "name": "Team Department", "value": "Video"},
        ]}
        assert _extract_brand_from_metadata(job) == ""

    def test_empty_brand_value(self):
        job = {"metadata": [{"id": 1, "name": "Brand Name", "value": ""}]}
        assert _extract_brand_from_metadata(job) == ""

    def test_non_string_value(self):
        job = {"metadata": [{"id": 1, "name": "Brand Name", "value": 42}]}
        assert _extract_brand_from_metadata(job) == ""

    def test_strips_whitespace(self):
        job = {"metadata": [{"id": 1, "name": "Brand Name", "value": "  Deadline  "}]}
        assert _extract_brand_from_metadata(job) == "Deadline"


class TestScrapeLever:
    @patch("job_agent.ats_scraper.requests.get")
    def test_parses_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = LEVER_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_lever("company", "TestCo", KEYWORDS)

        assert len(results) == 1  # "Backend Developer" filtered out
        assert results[0]["title"] == "Creative Producer"
        assert results[0]["company"] == "TestCo"
        assert results[0]["location"] == "Los Angeles, CA"
        assert results[0]["site"] == "lever"
        assert results[0]["department"] == "Content"
        assert "creative projects" in results[0]["description"].lower()

    @patch("job_agent.ats_scraper.requests.get")
    def test_handles_api_error(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        results = scrape_lever("badtoken", "Bad", KEYWORDS)
        assert results == []

    @patch("job_agent.ats_scraper.requests.get")
    def test_handles_unexpected_format(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "not found"}  # dict instead of list
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_lever("badtoken", "Bad", KEYWORDS)
        assert results == []


class TestScrapeAtsBoards:
    @patch("job_agent.ats_scraper.scrape_lever")
    @patch("job_agent.ats_scraper.scrape_greenhouse")
    def test_combines_results(self, mock_gh, mock_lv):
        mock_gh.return_value = [
            {"title": "Producer", "company": "GHCo", "site": "greenhouse",
             "location": "SF", "job_url": "http://gh/1", "description": "",
             "date_posted": "", "department": ""},
        ]
        mock_lv.return_value = [
            {"title": "Producer", "company": "LVCo", "site": "lever",
             "location": "LA", "job_url": "http://lv/1", "description": "",
             "date_posted": "", "department": ""},
        ]

        companies = [
            {"company": "GHCo", "ats": "greenhouse", "token": "ghco"},
            {"company": "LVCo", "ats": "lever", "token": "lvco"},
        ]

        df = scrape_ats_boards(companies, ["Producer"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(df["site"]) == {"greenhouse", "lever"}

    @patch("job_agent.ats_scraper.scrape_greenhouse")
    def test_empty_results(self, mock_gh):
        mock_gh.return_value = []
        companies = [{"company": "Empty", "ats": "greenhouse", "token": "empty"}]

        df = scrape_ats_boards(companies, ["Producer"])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("job_agent.ats_scraper.scrape_greenhouse")
    def test_progress_callback(self, mock_gh):
        mock_gh.return_value = []
        callback = MagicMock()

        companies = [{"company": "Co", "ats": "greenhouse", "token": "co"}]
        scrape_ats_boards(companies, ["Producer"], progress_callback=callback)

        assert callback.call_count >= 2  # At least per-company + final

    def test_unknown_ats_type(self):
        companies = [{"company": "X", "ats": "myunknownats", "token": "x"}]
        df = scrape_ats_boards(companies, ["Producer"])
        assert df.empty


class TestKeywordFiltering:
    @patch("job_agent.ats_scraper.requests.get")
    def test_filters_by_keyword(self, mock_get):
        """Verify only jobs matching keywords are returned."""
        response = {
            "jobs": [
                {"title": "Content Producer", "location": {"name": ""}, "absolute_url": "", "content": "", "updated_at": "", "departments": []},
                {"title": "Data Scientist", "location": {"name": ""}, "absolute_url": "", "content": "", "updated_at": "", "departments": []},
                {"title": "Video Project Manager", "location": {"name": ""}, "absolute_url": "", "content": "", "updated_at": "", "departments": []},
                {"title": "Accountant", "location": {"name": ""}, "absolute_url": "", "content": "", "updated_at": "", "departments": []},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_greenhouse("test", "Test", ["Content Producer", "Video Project Manager"])
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert titles == {"Content Producer", "Video Project Manager"}
