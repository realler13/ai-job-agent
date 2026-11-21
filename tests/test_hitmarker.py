"""Tests for scraper_hitmarker — sitemap + JSON-LD based scraper."""
import json
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from job_agent.scraper_hitmarker import (
    scrape_hitmarker,
    _extract_job_posting_ld,
    _parse_jobposting,
    _fetch_job_sitemap_urls,
)


SITEMAP_INDEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://hitmarker.net/sitemap-jobs.xml/p1</loc></sitemap>
  <sitemap><loc>https://hitmarker.net/sitemap-jobs.xml/p2</loc></sitemap>
  <sitemap><loc>https://hitmarker.net/sitemap-companies.xml/p1</loc></sitemap>
</sitemapindex>
"""

SITEMAP_JOBS_P1_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://hitmarker.net/jobs/driftwood-arcade-senior-events-producer-1000101</loc></url>
  <url><loc>https://hitmarker.net/jobs/ubisoft-backend-engineer-1631067</loc></url>
  <url><loc>https://hitmarker.net/jobs/lanternhook-games-content-producer-1000102</loc></url>
</urlset>
"""

SITEMAP_JOBS_P2_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://hitmarker.net/jobs/activision-video-producer-events-1693737</loc></url>
</urlset>
"""


JOBPOSTING_LD = {
    "@context": "https://schema.org/",
    "@type": "JobPosting",
    "title": "Senior Events Producer",
    "description": "<p>Lead live event production for Driftwood Arcade. Travel required.</p>",
    "datePosted": "2026-04-18",
    "employmentType": "FULL_TIME",
    "hiringOrganization": {
        "@type": "Organization",
        "name": "Driftwood Arcade",
    },
    "jobLocation": {
        "@type": "Place",
        "address": {
            "@type": "PostalAddress",
            "addressLocality": "Austin",
            "addressRegion": "TX",
            "addressCountry": "US",
        },
    },
}


def _make_detail_html(ld_payload: dict) -> str:
    """Wrap a JSON-LD dict in a minimal page, alongside a noise JSON-LD block."""
    noise = '{"@type": "Product", "name": "Hitmarker"}'
    return f"""
<!doctype html><html><body>
<script type="application/ld+json">{noise}</script>
<script type="application/ld+json">{json.dumps(ld_payload)}</script>
</body></html>
"""


# --- _extract_job_posting_ld ------------------------------------------------


class TestExtractJobPostingLd:
    def test_picks_jobposting_from_multiple_blocks(self):
        soup = BeautifulSoup(_make_detail_html(JOBPOSTING_LD), "html.parser")
        result = _extract_job_posting_ld(soup)
        assert result is not None
        assert result["@type"] == "JobPosting"
        assert result["title"] == "Senior Events Producer"

    def test_no_jobposting_returns_none(self):
        html = '<html><body><script type="application/ld+json">{"@type":"Organization"}</script></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        assert _extract_job_posting_ld(soup) is None

    def test_handles_list_form(self):
        # JSON-LD can be a list of objects
        html = f'<html><body><script type="application/ld+json">{json.dumps([JOBPOSTING_LD])}</script></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        result = _extract_job_posting_ld(soup)
        assert result is not None
        assert result["title"] == "Senior Events Producer"

    def test_handles_graph_form(self):
        # @graph form is also common
        payload = {"@context": "https://schema.org/", "@graph": [JOBPOSTING_LD]}
        html = f'<html><body><script type="application/ld+json">{json.dumps(payload)}</script></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        result = _extract_job_posting_ld(soup)
        assert result is not None
        assert result["title"] == "Senior Events Producer"

    def test_bad_json_skipped(self):
        html = '<html><body><script type="application/ld+json">{not valid json</script></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        assert _extract_job_posting_ld(soup) is None


# --- _parse_jobposting ------------------------------------------------------


class TestParseJobPosting:
    def test_full_extraction(self):
        job = _parse_jobposting(JOBPOSTING_LD, "https://hitmarker.net/jobs/driftwood-arcade-senior-events-producer-1000101")
        assert job["title"] == "Senior Events Producer"
        assert job["company"] == "Driftwood Arcade"
        assert job["location"] == "Austin, TX"
        assert job["site"] == "hitmarker"
        assert job["department"] == "FULL_TIME"
        assert "Lead live event production" in job["description"]
        assert "<p>" not in job["description"]  # HTML stripped
        assert job["job_url"].endswith("1000101")

    def test_telecommute_becomes_remote(self):
        data = dict(JOBPOSTING_LD)
        data["jobLocationType"] = "TELECOMMUTE"
        data.pop("jobLocation", None)
        job = _parse_jobposting(data, "https://x")
        assert job["location"] == "Remote"

    def test_hiring_organization_as_string(self):
        data = dict(JOBPOSTING_LD)
        data["hiringOrganization"] = "Indie Studio"
        job = _parse_jobposting(data, "https://x")
        assert job["company"] == "Indie Studio"

    def test_missing_fields_tolerated(self):
        # Minimum viable JobPosting
        data = {"@type": "JobPosting", "title": "Producer"}
        job = _parse_jobposting(data, "https://x")
        assert job["title"] == "Producer"
        assert job["company"] == ""
        assert job["location"] == ""


# --- _fetch_job_sitemap_urls ------------------------------------------------


class TestFetchSitemapUrls:
    @patch("job_agent.scraper_hitmarker._session")
    def test_walks_index_and_collects_urls(self, mock_session):
        # Set up 3 sequential responses: index, p1, p2
        resp_index = MagicMock(); resp_index.text = SITEMAP_INDEX_XML; resp_index.raise_for_status = MagicMock()
        resp_p1 = MagicMock(); resp_p1.text = SITEMAP_JOBS_P1_XML; resp_p1.raise_for_status = MagicMock()
        resp_p2 = MagicMock(); resp_p2.text = SITEMAP_JOBS_P2_XML; resp_p2.raise_for_status = MagicMock()
        mock_session.get.side_effect = [resp_index, resp_p1, resp_p2]

        urls = _fetch_job_sitemap_urls()

        # 3 in p1 + 1 in p2 = 4 total. Companies sitemap should NOT be fetched.
        assert len(urls) == 4
        assert "https://hitmarker.net/jobs/driftwood-arcade-senior-events-producer-1000101" in urls
        # Only /jobs/ URLs included
        assert all("/jobs/" in u for u in urls)

    @patch("job_agent.scraper_hitmarker._session")
    def test_index_fetch_failure_returns_empty(self, mock_session):
        mock_session.get.side_effect = Exception("network down")
        assert _fetch_job_sitemap_urls() == []


# --- scrape_hitmarker (end-to-end with mocked HTTP) ------------------------


class TestScrapeHitmarker:
    @patch("job_agent.scraper_hitmarker.time.sleep", lambda _: None)
    @patch("job_agent.scraper_hitmarker._session")
    @patch("job_agent.scraper_hitmarker._fetch_job_sitemap_urls")
    def test_filters_slugs_and_fetches_details(self, mock_urls, mock_session):
        mock_urls.return_value = [
            "https://hitmarker.net/jobs/driftwood-arcade-senior-events-producer-1000101",
            "https://hitmarker.net/jobs/ubisoft-backend-engineer-1631067",
            "https://hitmarker.net/jobs/lanternhook-games-content-producer-1000102",
        ]

        # Every detail fetch returns the same JobPosting (test doesn't care about variety)
        detail_resp = MagicMock()
        detail_resp.text = _make_detail_html(JOBPOSTING_LD)
        detail_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = detail_resp

        jobs = scrape_hitmarker(keywords=["producer", "content producer"])

        # Only producer-slug URLs should be fetched — backend-engineer filtered out
        # by slug match, not by title.
        assert mock_session.get.call_count == 2
        assert len(jobs) == 2
        assert all(j["site"] == "hitmarker" for j in jobs)

    @patch("job_agent.scraper_hitmarker.time.sleep", lambda _: None)
    @patch("job_agent.scraper_hitmarker._session")
    @patch("job_agent.scraper_hitmarker._fetch_job_sitemap_urls")
    def test_respects_max_jobs_cap(self, mock_urls, mock_session):
        # 10 matching URLs but cap is 3
        mock_urls.return_value = [
            f"https://hitmarker.net/jobs/company-{i}-producer-{1000000+i}" for i in range(10)
        ]
        detail_resp = MagicMock()
        detail_resp.text = _make_detail_html(JOBPOSTING_LD)
        detail_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = detail_resp

        jobs = scrape_hitmarker(keywords=["producer"], max_jobs=3)
        assert len(jobs) == 3
        assert mock_session.get.call_count == 3

    @patch("job_agent.scraper_hitmarker._fetch_job_sitemap_urls")
    def test_empty_sitemap_returns_empty(self, mock_urls):
        mock_urls.return_value = []
        jobs = scrape_hitmarker(keywords=["producer"])
        assert jobs == []

    @patch("job_agent.scraper_hitmarker.time.sleep", lambda _: None)
    @patch("job_agent.scraper_hitmarker._session")
    @patch("job_agent.scraper_hitmarker._fetch_job_sitemap_urls")
    def test_detail_fetch_error_skips_gracefully(self, mock_urls, mock_session):
        mock_urls.return_value = [
            "https://hitmarker.net/jobs/co-producer-1111111",
            "https://hitmarker.net/jobs/co-producer-2222222",
        ]
        good = MagicMock(); good.text = _make_detail_html(JOBPOSTING_LD); good.raise_for_status = MagicMock()
        mock_session.get.side_effect = [Exception("500"), good]

        jobs = scrape_hitmarker(keywords=["producer"])
        assert len(jobs) == 1
