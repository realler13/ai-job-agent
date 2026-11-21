"""Tests for scraper_productionhub — StealthyFetcher-based HTML scraper."""
from unittest.mock import patch, MagicMock

import pytest
from bs4 import BeautifulSoup

from job_agent.scraper_productionhub import (
    scrape_productionhub,
    _parse_job_card,
    _extract_footer_field,
    _parse_date,
)


# --- Fixtures ---------------------------------------------------------------

# Real card structure captured from a live probe of productionhub.com/jobs
SAMPLE_CARD_HTML = """
<div class="media">
<a class="d-none d-md-flex mr-3" href="/jobs/details/64432"><img alt="logo"/></a>
<div class="media-body">
  <div class="media-header">
    <h5 class="media-title">
      <a href="/job/64432/video-producer-editing-focused">Video Producer (Editing-Focused)</a>
    </h5>
    <span class="media-meta">
      <a href="/jobs?locationID=3200">
        <span>
          <span itemprop="addressLocality">Alexandria</span>,
          <span itemprop="addressRegion">Virginia</span>
        </span>
      </a>
    </span>
  </div>
  <div class="media-desc">
    <a href="/job/64432/video-producer-editing-focused">
      Rescue Media is a full-service production shop in Alexandria.
    </a>
  </div>
  <footer>
    <span><span class="glyphicons glyphicons-th-list"></span>
      <a href="/jobs/type/commercial">Commercial Video</a></span>
    <span><span class="glyphicons glyphicons-time"></span>
      <a href="/jobs?typeID=1">Full Time</a></span>
    <span><span class="glyphicons glyphicons-briefcase"></span> Rescue Media</span>
    <span><span class="glyphicons glyphicons-calendar"></span> 4/19/2026</span>
  </footer>
</div>
</div>
"""

FULL_PAGE_HTML = f"""
<html><body>
  <div class="container">{SAMPLE_CARD_HTML}</div>
  <div class="container">
    <div class="media">
      <div class="media-body">
        <div class="media-header">
          <h5 class="media-title">
            <a href="/job/64401/software-engineer">Software Engineer</a>
          </h5>
          <span class="media-meta">
            <span itemprop="addressLocality">New York</span>,
            <span itemprop="addressRegion">NY</span>
          </span>
        </div>
        <div class="media-desc"><a href="/job/64401/software-engineer">Build APIs.</a></div>
        <footer>
          <span><span class="glyphicons glyphicons-briefcase"></span> WidgetCo</span>
          <span><span class="glyphicons glyphicons-calendar"></span> 4/18/2026</span>
        </footer>
      </div>
    </div>
  </div>
</body></html>
"""


# --- _parse_date -----------------------------------------------------------


class TestParseDate:
    def test_slash_format(self):
        assert _parse_date("4/19/2026") == "2026-04-19"

    def test_iso_passthrough(self):
        assert _parse_date("2026-04-19") == "2026-04-19"

    def test_dash_format(self):
        assert _parse_date("4-19-2026") == "2026-04-19"

    def test_empty(self):
        assert _parse_date("") == ""
        assert _parse_date("   ") == ""

    def test_unparseable(self):
        assert _parse_date("sometime in April") == ""


# --- _extract_footer_field -------------------------------------------------


class TestExtractFooterField:
    def test_finds_briefcase(self):
        soup = BeautifulSoup(SAMPLE_CARD_HTML, "html.parser")
        footer = soup.find("footer")
        assert _extract_footer_field(footer, "glyphicons-briefcase") == "Rescue Media"

    def test_finds_calendar(self):
        soup = BeautifulSoup(SAMPLE_CARD_HTML, "html.parser")
        footer = soup.find("footer")
        assert _extract_footer_field(footer, "glyphicons-calendar") == "4/19/2026"

    def test_icon_not_present_returns_empty(self):
        soup = BeautifulSoup(SAMPLE_CARD_HTML, "html.parser")
        footer = soup.find("footer")
        assert _extract_footer_field(footer, "glyphicons-nonexistent") == ""

    def test_none_footer(self):
        assert _extract_footer_field(None, "glyphicons-briefcase") == ""


# --- _parse_job_card -------------------------------------------------------


class TestParseJobCard:
    def test_extracts_all_fields(self):
        soup = BeautifulSoup(SAMPLE_CARD_HTML, "html.parser")
        card = soup.select_one("div.media")
        job = _parse_job_card(card)

        assert job["title"] == "Video Producer (Editing-Focused)"
        assert job["company"] == "Rescue Media"
        assert job["location"] == "Alexandria, Virginia"
        assert job["job_url"] == "https://www.productionhub.com/job/64432/video-producer-editing-focused"
        assert job["date_posted"] == "2026-04-19"
        assert "Rescue Media" in job["description"]
        assert job["site"] == "productionhub"

    def test_card_without_title_returns_none(self):
        soup = BeautifulSoup('<div class="media"><p>no title link</p></div>', "html.parser")
        card = soup.select_one("div.media")
        assert _parse_job_card(card) is None

    def test_relative_url_gets_prefixed(self):
        soup = BeautifulSoup(SAMPLE_CARD_HTML, "html.parser")
        card = soup.select_one("div.media")
        job = _parse_job_card(card)
        assert job["job_url"].startswith("https://www.productionhub.com/")


# --- scrape_productionhub (full flow) --------------------------------------


class TestScrapeProductionhub:
    @patch("job_agent.scraper_productionhub._fetch_stealth")
    def test_filters_by_keyword(self, mock_fetch):
        mock_fetch.return_value = BeautifulSoup(FULL_PAGE_HTML, "html.parser")

        jobs = scrape_productionhub(["producer", "video producer"], max_pages=1)

        assert len(jobs) == 1
        assert jobs[0]["title"] == "Video Producer (Editing-Focused)"
        # "Software Engineer" filtered out
        assert all(j["title"] != "Software Engineer" for j in jobs)

    @patch("job_agent.scraper_productionhub._fetch_stealth")
    def test_stops_when_fetch_returns_none(self, mock_fetch):
        mock_fetch.return_value = None
        jobs = scrape_productionhub(["producer"], max_pages=3)
        assert jobs == []
        # Should only try page 1, then bail
        assert mock_fetch.call_count == 1

    @patch("job_agent.scraper_productionhub._fetch_stealth")
    def test_stops_when_page_empty(self, mock_fetch):
        empty_soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        mock_fetch.return_value = empty_soup
        jobs = scrape_productionhub(["producer"], max_pages=3)
        assert jobs == []

    @patch("job_agent.scraper_productionhub._fetch_stealth")
    def test_dedupes_across_pages(self, mock_fetch):
        # Both pages return the same job — scraper should stop after page 2
        # because it saw zero NEW jobs
        page_soup = BeautifulSoup(FULL_PAGE_HTML, "html.parser")
        mock_fetch.return_value = page_soup

        jobs = scrape_productionhub(["producer"], max_pages=3)

        assert len(jobs) == 1  # Only the one unique producer job
        # Confirmed early-break on dupe-only page
        assert mock_fetch.call_count == 2
