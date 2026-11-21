"""Tests for industry HTML scrapers and RSS feed scrapers."""

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

KEYWORDS = ["Content Producer", "Video Producer", "Podcast Producer", "Game Producer"]


# ============================================================
# RSS Parsing Helpers
# ============================================================

class TestParseRssItems:
    SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Test Feed</title>
        <item>
          <title>Content Producer - Netflix</title>
          <link>https://example.com/job/123</link>
          <description>Lead content production for original series.</description>
          <pubDate>Mon, 15 Jan 2026 10:30:00 +0000</pubDate>
          <category>Los Angeles</category>
          <category>CA</category>
          <category>Full Time</category>
        </item>
        <item>
          <title>Software Engineer</title>
          <link>https://example.com/job/456</link>
          <description>Build backend systems.</description>
          <pubDate>Tue, 16 Jan 2026 08:00:00 +0000</pubDate>
        </item>
      </channel>
    </rss>"""

    def test_parses_items(self):
        from job_agent.scraper_rss import _parse_rss_items

        items = _parse_rss_items(self.SAMPLE_RSS)
        assert len(items) == 2
        assert items[0]["title"] == "Content Producer - Netflix"
        assert items[0]["link"] == "https://example.com/job/123"
        assert "content production" in items[0]["description"].lower()
        assert items[0]["pub_date"] == "Mon, 15 Jan 2026 10:30:00 +0000"
        assert "Los Angeles" in items[0]["categories"]
        assert "CA" in items[0]["categories"]
        assert "Full Time" in items[0]["categories"]

    def test_handles_invalid_xml(self):
        from job_agent.scraper_rss import _parse_rss_items

        items = _parse_rss_items("<not valid xml!!!")
        assert items == []

    def test_handles_empty_channel(self):
        from job_agent.scraper_rss import _parse_rss_items

        xml = """<?xml version="1.0"?><rss><channel></channel></rss>"""
        items = _parse_rss_items(xml)
        assert items == []


class TestParseRssDate:
    def test_rfc822_format(self):
        from job_agent.scraper_rss import _parse_rss_date

        result = _parse_rss_date("Mon, 15 Jan 2026 10:30:00 +0000")
        assert result == "2026-01-15"

    def test_iso_format(self):
        from job_agent.scraper_rss import _parse_rss_date

        result = _parse_rss_date("2026-01-15T10:30:00+0000")
        assert result == "2026-01-15"

    def test_already_formatted(self):
        from job_agent.scraper_rss import _parse_rss_date

        result = _parse_rss_date("2026-01-15")
        assert result == "2026-01-15"

    def test_empty_string(self):
        from job_agent.scraper_rss import _parse_rss_date

        result = _parse_rss_date("")
        assert result == ""

    def test_no_timezone(self):
        from job_agent.scraper_rss import _parse_rss_date

        result = _parse_rss_date("Mon, 15 Jan 2026 10:30:00")
        assert result == "2026-01-15"


# ============================================================
# EntertainmentCareers RSS
# ============================================================

class TestScrapeEntertainmentCareersRss:
    SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>EntertainmentCareers.Net - Music</title>
        <item>
          <title>Content Producer</title>
          <link>https://www.entertainmentcareers.net/job/12345</link>
          <description>Produce content for streaming platform.</description>
          <pubDate>Mon, 15 Jan 2026 10:30:00 +0000</pubDate>
          <category>Los Angeles</category>
          <category>CA</category>
          <category>Full Time</category>
        </item>
        <item>
          <title>Podcast Producer</title>
          <link>https://www.entertainmentcareers.net/job/12346</link>
          <description>Manage podcast production pipeline.</description>
          <pubDate>Tue, 16 Jan 2026 08:00:00 +0000</pubDate>
          <category>New York</category>
          <category>NY</category>
        </item>
        <item>
          <title>Accountant</title>
          <link>https://www.entertainmentcareers.net/job/12347</link>
          <description>Handle financials.</description>
          <pubDate>Wed, 17 Jan 2026 09:00:00 +0000</pubDate>
        </item>
      </channel>
    </rss>"""

    @patch("job_agent.scraper_rss._session")
    def test_parses_and_filters(self, mock_session):
        from job_agent.scraper_rss import scrape_entertainmentcareers_rss

        mock_resp = MagicMock()
        mock_resp.text = self.SAMPLE_RSS
        mock_resp.raise_for_status.return_value = None
        mock_session.get.return_value = mock_resp

        results = scrape_entertainmentcareers_rss(KEYWORDS, feed_ids=["ecnjcat119"])

        # Should match Content Producer and Podcast Producer, not Accountant
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert "Content Producer" in titles
        assert "Podcast Producer" in titles

        # Check fields
        cp = next(r for r in results if r["title"] == "Content Producer")
        assert cp["site"] == "entertainmentcareers"
        assert cp["job_url"] == "https://www.entertainmentcareers.net/job/12345"
        assert cp["date_posted"] == "2026-01-15"
        assert "Los Angeles" in cp["location"]

    @patch("job_agent.scraper_rss._session")
    def test_deduplicates_across_feeds(self, mock_session):
        from job_agent.scraper_rss import scrape_entertainmentcareers_rss

        mock_resp = MagicMock()
        mock_resp.text = self.SAMPLE_RSS
        mock_resp.raise_for_status.return_value = None
        mock_session.get.return_value = mock_resp

        # Same RSS returned for 2 feeds — should dedup
        results = scrape_entertainmentcareers_rss(KEYWORDS, feed_ids=["ecnjcat119", "ecnjcat168"])
        assert len(results) == 2  # Not 4

    @patch("job_agent.scraper_rss._session")
    def test_handles_http_error(self, mock_session):
        from job_agent.scraper_rss import scrape_entertainmentcareers_rss

        mock_session.get.side_effect = Exception("Connection timeout")

        results = scrape_entertainmentcareers_rss(KEYWORDS, feed_ids=["ecnjcat119"])
        assert results == []

    @patch("job_agent.scraper_rss._session")
    def test_filters_job_type_from_location(self, mock_session):
        from job_agent.scraper_rss import scrape_entertainmentcareers_rss

        mock_resp = MagicMock()
        mock_resp.text = self.SAMPLE_RSS
        mock_resp.raise_for_status.return_value = None
        mock_session.get.return_value = mock_resp

        results = scrape_entertainmentcareers_rss(KEYWORDS, feed_ids=["ecnjcat119"])
        cp = next(r for r in results if r["title"] == "Content Producer")
        # "Full Time" should not be in location
        assert "Full Time" not in cp["location"]


# ============================================================
# Games-Career RSS
# ============================================================

class TestScrapeGamesCareerRss:
    SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>games-career.com</title>
        <item>
          <title>Game Producer (m/f/d)</title>
          <link>https://www.games-career.com/job/99999</link>
          <description>Manage game production pipeline.</description>
          <pubDate>Mon, 15 Jan 2026 12:00:00 +0000</pubDate>
        </item>
        <item>
          <title>3D Artist</title>
          <link>https://www.games-career.com/job/99998</link>
          <description>Create 3D assets.</description>
          <pubDate>Tue, 16 Jan 2026 12:00:00 +0000</pubDate>
        </item>
      </channel>
    </rss>"""

    @patch("job_agent.scraper_rss._session")
    def test_parses_and_filters(self, mock_session):
        from job_agent.scraper_rss import scrape_gamescareer_rss

        mock_resp = MagicMock()
        mock_resp.text = self.SAMPLE_RSS
        mock_resp.raise_for_status.return_value = None
        mock_session.get.return_value = mock_resp

        results = scrape_gamescareer_rss(KEYWORDS)

        assert len(results) == 1
        assert results[0]["title"] == "Game Producer (m/f/d)"
        assert results[0]["site"] == "gamescareer"
        assert results[0]["date_posted"] == "2026-01-15"

    @patch("job_agent.scraper_rss._session")
    def test_handles_empty_feed(self, mock_session):
        from job_agent.scraper_rss import scrape_gamescareer_rss

        empty_rss = """<?xml version="1.0"?><rss><channel></channel></rss>"""
        mock_resp = MagicMock()
        mock_resp.text = empty_rss
        mock_resp.raise_for_status.return_value = None
        mock_session.get.return_value = mock_resp

        results = scrape_gamescareer_rss(KEYWORDS)
        assert results == []

    @patch("job_agent.scraper_rss._session")
    def test_handles_http_error(self, mock_session):
        from job_agent.scraper_rss import scrape_gamescareer_rss

        mock_session.get.side_effect = Exception("503 Service Unavailable")
        results = scrape_gamescareer_rss(KEYWORDS)
        assert results == []


# ============================================================
# RSS Orchestrator
# ============================================================

class TestScrapeRssFeeds:
    @patch("job_agent.scraper_rss.scrape_aquent_feed", return_value=[])
    @patch("job_agent.scraper_rss.scrape_gamescareer_rss")
    @patch("job_agent.scraper_rss.scrape_entertainmentcareers_rss")
    def test_combines_results(self, mock_ec, mock_gc, mock_aq):
        from job_agent.scraper_rss import scrape_rss_feeds

        mock_ec.return_value = [
            {"title": "Content Producer", "company": "", "location": "LA", "job_url": "https://ec.net/1",
             "description": "", "date_posted": "", "site": "entertainmentcareers", "department": ""},
        ]
        mock_gc.return_value = [
            {"title": "Game Producer", "company": "", "location": "", "job_url": "https://gc.com/1",
             "description": "", "date_posted": "", "site": "gamescareer", "department": ""},
        ]

        df = scrape_rss_feeds(KEYWORDS)
        assert len(df) == 2
        assert set(df["site"]) == {"entertainmentcareers", "gamescareer"}

    @patch("job_agent.scraper_rss.scrape_aquent_feed", return_value=[])
    @patch("job_agent.scraper_rss.scrape_gamescareer_rss")
    @patch("job_agent.scraper_rss.scrape_entertainmentcareers_rss")
    def test_returns_empty_df_on_no_results(self, mock_ec, mock_gc, mock_aq):
        from job_agent.scraper_rss import scrape_rss_feeds

        mock_ec.return_value = []
        mock_gc.return_value = []

        df = scrape_rss_feeds(KEYWORDS)
        assert df.empty

    @patch("job_agent.scraper_rss.scrape_gamescareer_rss")
    @patch("job_agent.scraper_rss.scrape_entertainmentcareers_rss")
    def test_calls_progress_callback(self, mock_ec, mock_gc):
        from job_agent.scraper_rss import scrape_rss_feeds

        mock_ec.return_value = []
        mock_gc.return_value = []
        cb = MagicMock()

        scrape_rss_feeds(KEYWORDS, progress_callback=cb)
        assert cb.call_count >= 2


# ============================================================
# ShowbizJobs HTML
# ============================================================

class TestScrapeShowbizjobs:
    SAMPLE_HTML = """<html><body>
    <table id="postings">
      <tr>
        <td>
          <h5><a href="/jobs/content-producer-123">Content Producer</a></h5>
          <h6><a href="/company/acme">Acme Studios</a></h6>
        </td>
        <td>Los Angeles, CA</td>
      </tr>
      <tr>
        <td>
          <h5><a href="/jobs/accountant-456">Staff Accountant</a></h5>
          <h6><a href="/company/bigco">BigCo</a></h6>
        </td>
        <td>New York, NY</td>
      </tr>
    </table>
    </body></html>"""

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_parses_and_filters(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_showbizjobs
        from bs4 import BeautifulSoup

        mock_fetch.return_value = BeautifulSoup(self.SAMPLE_HTML, "html.parser")

        results = scrape_showbizjobs(KEYWORDS, categories=["production-crew"], max_pages=1)

        assert len(results) == 1
        assert results[0]["title"] == "Content Producer"
        assert results[0]["company"] == "Acme Studios"
        assert results[0]["site"] == "showbizjobs"
        assert "showbizjobs.com" in results[0]["job_url"]

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_handles_empty_table(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_showbizjobs
        from bs4 import BeautifulSoup

        html = "<html><body><table id='postings'></table></body></html>"
        mock_fetch.return_value = BeautifulSoup(html, "html.parser")

        results = scrape_showbizjobs(KEYWORDS, categories=["production-crew"], max_pages=1)
        assert results == []

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_handles_fetch_failure(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_showbizjobs

        mock_fetch.return_value = None

        results = scrape_showbizjobs(KEYWORDS, categories=["production-crew"], max_pages=1)
        assert results == []

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_deduplicates_across_pages(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_showbizjobs
        from bs4 import BeautifulSoup

        # Return same HTML for both pages
        mock_fetch.return_value = BeautifulSoup(self.SAMPLE_HTML, "html.parser")

        results = scrape_showbizjobs(KEYWORDS, categories=["production-crew"], max_pages=2)
        # Should only have 1 result despite 2 pages with same content
        assert len(results) == 1


# ============================================================
# Podnews HTML
# ============================================================

class TestScrapePodnews:
    SAMPLE_HTML = """<html><body>
    <div class="artblock">
      <a href="/job/podcast-producer-at-spotify">Podcast Producer</a>
      <span>Spotify</span>
    </div>
    <div class="artblock">
      <a href="/job/marketing-manager">Marketing Manager</a>
      <span>Apple</span>
    </div>
    </body></html>"""

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_parses_and_filters(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_podnews
        from bs4 import BeautifulSoup

        mock_fetch.return_value = BeautifulSoup(self.SAMPLE_HTML, "html.parser")

        results = scrape_podnews(KEYWORDS)

        assert len(results) == 1
        assert results[0]["title"] == "Podcast Producer"
        assert results[0]["site"] == "podnews"
        assert "podnews.net" in results[0]["job_url"]

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_handles_fetch_failure(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_podnews

        mock_fetch.return_value = None
        results = scrape_podnews(KEYWORDS)
        assert results == []


# ============================================================
# MBW HTML
# ============================================================

class TestScrapeMbw:
    SAMPLE_HTML = """<html><body>
    <ul>
      <li class="job_listing">
        <a href="https://musicbusinessworldwide.com/job/content-producer/"><strong>Content Producer</strong></a>
        <a href="#">Universal Music</a>
        <a href="#">London</a>
      </li>
      <li class="job_listing">
        <a href="https://musicbusinessworldwide.com/job/data-analyst/"><strong>Data Analyst</strong></a>
        <a href="#">Spotify</a>
        <a href="#">Stockholm</a>
      </li>
    </ul>
    </body></html>"""

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_parses_and_filters(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_mbw
        from bs4 import BeautifulSoup

        mock_fetch.return_value = BeautifulSoup(self.SAMPLE_HTML, "html.parser")

        results = scrape_mbw(KEYWORDS, max_pages=1)

        assert len(results) == 1
        assert results[0]["title"] == "Content Producer"
        assert results[0]["company"] == "Universal Music"
        assert results[0]["location"] == "London"
        assert results[0]["site"] == "mbw"

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_handles_fetch_failure(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_mbw

        mock_fetch.return_value = None
        results = scrape_mbw(KEYWORDS, max_pages=1)
        assert results == []


# ============================================================
# DoorsOpen HTML
# ============================================================

class TestScrapeDoorsopen:
    SAMPLE_HTML = """<html><body>
    <article class="listing-item">
      <h4 class="listing-item__title"><a href="/jobs/video-producer-hulu">Video Producer</a></h4>
      <ul>
        <li class="listing-item__info--item">Hulu</li>
        <li class="listing-item__info--item">Santa Monica, CA</li>
      </ul>
      <span class="listing-item__date">January 15, 2026</span>
    </article>
    <article class="listing-item">
      <h4 class="listing-item__title"><a href="/jobs/data-engineer">Data Engineer</a></h4>
      <ul>
        <li class="listing-item__info--item">Netflix</li>
        <li class="listing-item__info--item">Remote</li>
      </ul>
    </article>
    </body></html>"""

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_parses_and_filters(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_doorsopen
        from bs4 import BeautifulSoup

        mock_fetch.return_value = BeautifulSoup(self.SAMPLE_HTML, "html.parser")

        results = scrape_doorsopen(KEYWORDS, max_pages=1)

        assert len(results) == 1
        assert results[0]["title"] == "Video Producer"
        assert results[0]["company"] == "Hulu"
        assert results[0]["location"] == "Santa Monica, CA"
        assert results[0]["date_posted"] == "January 15, 2026"
        assert results[0]["site"] == "doorsopen"
        assert "doorsopen.co" in results[0]["job_url"]

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_handles_fetch_failure(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_doorsopen

        mock_fetch.return_value = None
        results = scrape_doorsopen(KEYWORDS, max_pages=1)
        assert results == []


# ============================================================
# Digilogue HTML
# ============================================================

class TestScrapeDigilogue:
    SAMPLE_HTML = """<html><body>
    <div class="w-dyn-item">
      <h3>Content Producer</h3>
      <a href="https://apply.example.com/content-producer">Apply</a>
      <div>Acme Corp</div>
      <div>San Francisco, CA</div>
    </div>
    <div class="w-dyn-item">
      <h3>Backend Developer</h3>
      <a href="https://apply.example.com/backend-dev">Apply</a>
      <div>TechCo</div>
      <div>Remote</div>
    </div>
    </body></html>"""

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_parses_and_filters(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_digilogue
        from bs4 import BeautifulSoup

        mock_fetch.return_value = BeautifulSoup(self.SAMPLE_HTML, "html.parser")

        results = scrape_digilogue(KEYWORDS)

        assert len(results) == 1
        assert results[0]["title"] == "Content Producer"
        assert results[0]["site"] == "digilogue"

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_handles_fetch_failure(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_digilogue

        mock_fetch.return_value = None
        results = scrape_digilogue(KEYWORDS)
        assert results == []


# ============================================================
# RemoteGameJobs HTML
# ============================================================

class TestScrapeRemoteGameJobs:
    SAMPLE_HTML = """<html><body>
    <div class="job-box">
      <h3>Game Producer</h3>
      <span class="company">Riot Games</span>
      <a href="/jobs/game-producer-riot">View Job</a>
    </div>
    <div class="job-box">
      <h3>UI Designer</h3>
      <span class="company">Epic Games</span>
      <a href="/jobs/ui-designer-epic">View Job</a>
    </div>
    </body></html>"""

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_parses_and_filters(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_remotegamejobs
        from bs4 import BeautifulSoup

        mock_fetch.return_value = BeautifulSoup(self.SAMPLE_HTML, "html.parser")

        results = scrape_remotegamejobs(KEYWORDS)

        assert len(results) == 1
        assert results[0]["title"] == "Game Producer"
        assert results[0]["company"] == "Riot Games"
        assert results[0]["location"] == "Remote"
        assert results[0]["site"] == "remotegamejobs"
        assert "remotegamejobs.com" in results[0]["job_url"]

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_handles_fetch_failure(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_remotegamejobs

        mock_fetch.return_value = None
        results = scrape_remotegamejobs(KEYWORDS)
        assert results == []


# ============================================================
# WillHire (Apple Contingent Workforce)
# ============================================================

class TestWillhireDates:
    """WillHire's relative 'Posted:' values become ISO dates at scrape time
    (stored raw they never parsed: no age pill, no stale gate)."""

    @pytest.mark.parametrize("raw,expected", [
        ("Today", "2026-10-04"),
        ("Yesterday", "2026-10-03"),
        ("An hour ago", "2026-10-04"),
        ("3 hours ago", "2026-10-03"),     # 01:30 minus 3h is yesterday
        ("45 minutes ago", "2026-10-04"),
        ("2 days ago", "2026-10-02"),
        ("a week ago", "2026-09-27"),
        ("Jan 22, 2026", "2026-01-22"),
        ("2026-09-30", "2026-09-30"),
        ("", ""),
        ("Posted recently", "Posted recently"),  # unknown text is kept, not guessed
    ])
    def test_parse_posted_date(self, raw, expected):
        from datetime import datetime
        from job_agent.scraper_industry_html import parse_posted_date
        assert parse_posted_date(raw, now=datetime(2026, 10, 4, 1, 30)) == expected


class TestScrapeWillhire:
    SAMPLE_HTML = """<html><body>
    <div class="col-md-9">
      <div class="kt-portlet p-4 job-card job-card-layout">
        <div class="d-flex flex-column job-information">
          <a href="/jobs/900001-content-producer-onsite" class="mb-3 job-title">
            Content Producer (Onsite)
          </a>
          <div class="d-flex flex-row align-items-center">
            <span class="info-text">Austin, Texas (On-Site)</span>
          </div>
          <div class="d-flex flex-row align-items-center">
            <span class="info-text">Posted: 3 hours ago</span>
          </div>
        </div>
      </div>
      <div class="kt-portlet p-4 job-card job-card-layout">
        <div class="d-flex flex-column job-information">
          <a href="/jobs/900002-software-engineer-remote" class="mb-3 job-title">
            Software Engineer (Remote)
          </a>
          <div class="d-flex flex-row align-items-center">
            <span class="info-text">Remote</span>
          </div>
          <div class="d-flex flex-row align-items-center">
            <span class="info-text">Posted: 1 day ago</span>
          </div>
        </div>
      </div>
      <div class="kt-portlet p-4 job-card job-card-layout">
        <div class="d-flex flex-column job-information">
          <a href="/jobs/900003-video-producer-hybrid" class="mb-3 job-title">
            Video Producer (Hybrid)
          </a>
          <div class="d-flex flex-row align-items-center">
            <span class="info-text">Los Angeles, California (Hybrid)</span>
          </div>
          <div class="d-flex flex-row align-items-center">
            <span class="info-text">Posted: 5 days ago</span>
          </div>
        </div>
      </div>
    </div>
    </body></html>"""

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_parses_and_filters(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_willhire
        from bs4 import BeautifulSoup

        mock_fetch.return_value = BeautifulSoup(self.SAMPLE_HTML, "html.parser")

        results = scrape_willhire(KEYWORDS)

        # Should match Content Producer and Video Producer, not Software Engineer
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert "Content Producer (Onsite)" in titles
        assert "Video Producer (Hybrid)" in titles

        # Check fields
        cp = next(r for r in results if "Content" in r["title"])
        assert cp["company"] == "Apple (via WillHire)"
        assert cp["location"] == "Austin, Texas (On-Site)"
        # Relative WillHire dates are converted to ISO at scrape time
        # ("3 hours ago" is today, or yesterday shortly after midnight).
        from datetime import datetime, timedelta
        today = datetime.now().date()
        assert cp["date_posted"] in {today.isoformat(), (today - timedelta(days=1)).isoformat()}
        vp = next(r for r in results if "Video" in r["title"])
        assert vp["date_posted"] == (today - timedelta(days=5)).isoformat()
        assert cp["site"] == "willhire"
        assert "willhire.co/jobs/900001" in cp["job_url"]

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_handles_fetch_failure(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_willhire

        mock_fetch.return_value = None
        results = scrape_willhire(KEYWORDS)
        assert results == []

    @patch("job_agent.scraper_industry_html._fetch_page")
    def test_deduplicates_by_url(self, mock_fetch):
        from job_agent.scraper_industry_html import scrape_willhire
        from bs4 import BeautifulSoup

        # Two pages return the same job
        mock_fetch.return_value = BeautifulSoup(self.SAMPLE_HTML, "html.parser")

        results = scrape_willhire(KEYWORDS, max_pages=2)
        urls = [r["job_url"] for r in results]
        assert len(urls) == len(set(urls)), "Should not have duplicate URLs"


# ============================================================
# HTML Orchestrator
# ============================================================

class TestScrapeIndustryHtmlBoards:
    @patch("job_agent.scraper_industry_html.scrape_builtin")
    @patch("job_agent.scraper_industry_html.scrape_hitmarker")
    @patch("job_agent.scraper_industry_html.scrape_productionhub")
    @patch("job_agent.scraper_industry_html.scrape_apmg")
    @patch("job_agent.scraper_industry_html.scrape_willhire")
    @patch("job_agent.scraper_industry_html.scrape_remotegamejobs")
    @patch("job_agent.scraper_industry_html.scrape_digilogue")
    @patch("job_agent.scraper_industry_html.scrape_doorsopen")
    @patch("job_agent.scraper_industry_html.scrape_mbw")
    @patch("job_agent.scraper_industry_html.scrape_podnews")
    @patch("job_agent.scraper_industry_html.scrape_showbizjobs")
    def test_combines_all_scrapers(self, mock_sbj, mock_pod, mock_mbw, mock_doors, mock_digi, mock_rgj, mock_wh, mock_apmg, mock_phub, mock_hit, mock_bi):
        from job_agent.scraper_industry_html import scrape_industry_html_boards

        mock_sbj.return_value = [{"title": "A", "company": "", "location": "", "job_url": "u1",
                                   "description": "", "date_posted": "", "site": "showbizjobs", "department": ""}]
        mock_pod.return_value = [{"title": "B", "company": "", "location": "", "job_url": "u2",
                                   "description": "", "date_posted": "", "site": "podnews", "department": ""}]
        mock_mbw.return_value = []
        mock_doors.return_value = []
        mock_digi.return_value = []
        mock_rgj.return_value = [{"title": "C", "company": "", "location": "", "job_url": "u3",
                                   "description": "", "date_posted": "", "site": "remotegamejobs", "department": ""}]
        mock_wh.return_value = [{"title": "D", "company": "Apple (via WillHire)", "location": "", "job_url": "u4",
                                  "description": "", "date_posted": "", "site": "willhire", "department": ""}]
        mock_apmg.return_value = [{"title": "E", "company": "American Public Media Group", "location": "St Paul, MN",
                                    "job_url": "u5", "description": "", "date_posted": "", "site": "apmg", "department": ""}]
        mock_phub.return_value = [{"title": "F", "company": "Rescue Media", "location": "Alexandria, VA",
                                    "job_url": "u6", "description": "", "date_posted": "", "site": "productionhub", "department": ""}]
        mock_hit.return_value = [{"title": "G", "company": "Riot Games", "location": "Los Angeles, CA",
                                   "job_url": "u7", "description": "", "date_posted": "", "site": "hitmarker", "department": "FULL_TIME"}]
        mock_bi.return_value = [{"title": "H", "company": "Netflix", "location": "Los Angeles",
                                  "job_url": "u8", "description": "", "date_posted": "", "site": "builtin", "department": ""}]

        df = scrape_industry_html_boards(KEYWORDS)

        assert len(df) == 8
        assert set(df["site"]) == {"showbizjobs", "podnews", "remotegamejobs", "willhire", "apmg", "productionhub", "hitmarker", "builtin"}

        # Verify all scrapers were called
        mock_sbj.assert_called_once_with(KEYWORDS)
        mock_pod.assert_called_once_with(KEYWORDS)
        mock_mbw.assert_called_once_with(KEYWORDS)
        mock_doors.assert_called_once_with(KEYWORDS)
        mock_digi.assert_called_once_with(KEYWORDS)
        mock_rgj.assert_called_once_with(KEYWORDS)
        mock_wh.assert_called_once_with(KEYWORDS)
        mock_apmg.assert_called_once_with(KEYWORDS)
        mock_phub.assert_called_once_with(KEYWORDS)
        mock_hit.assert_called_once_with(KEYWORDS)
        mock_bi.assert_called_once_with(KEYWORDS)

    @patch("job_agent.scraper_industry_html.scrape_builtin")
    @patch("job_agent.scraper_industry_html.scrape_hitmarker")
    @patch("job_agent.scraper_industry_html.scrape_productionhub")
    @patch("job_agent.scraper_industry_html.scrape_apmg")
    @patch("job_agent.scraper_industry_html.scrape_willhire")
    @patch("job_agent.scraper_industry_html.scrape_remotegamejobs")
    @patch("job_agent.scraper_industry_html.scrape_digilogue")
    @patch("job_agent.scraper_industry_html.scrape_doorsopen")
    @patch("job_agent.scraper_industry_html.scrape_mbw")
    @patch("job_agent.scraper_industry_html.scrape_podnews")
    @patch("job_agent.scraper_industry_html.scrape_showbizjobs")
    def test_returns_empty_df_on_no_results(self, mock_sbj, mock_pod, mock_mbw, mock_doors, mock_digi, mock_rgj, mock_wh, mock_apmg, mock_phub, mock_hit, mock_bi):
        from job_agent.scraper_industry_html import scrape_industry_html_boards

        mock_sbj.return_value = []
        mock_pod.return_value = []
        mock_mbw.return_value = []
        mock_doors.return_value = []
        mock_digi.return_value = []
        mock_rgj.return_value = []
        mock_wh.return_value = []
        mock_apmg.return_value = []
        mock_phub.return_value = []
        mock_hit.return_value = []
        mock_bi.return_value = []

        df = scrape_industry_html_boards(KEYWORDS)
        assert df.empty

    @patch("job_agent.scraper_industry_html.scrape_builtin")
    @patch("job_agent.scraper_industry_html.scrape_hitmarker")
    @patch("job_agent.scraper_industry_html.scrape_productionhub")
    @patch("job_agent.scraper_industry_html.scrape_apmg")
    @patch("job_agent.scraper_industry_html.scrape_willhire")
    @patch("job_agent.scraper_industry_html.scrape_remotegamejobs")
    @patch("job_agent.scraper_industry_html.scrape_digilogue")
    @patch("job_agent.scraper_industry_html.scrape_doorsopen")
    @patch("job_agent.scraper_industry_html.scrape_mbw")
    @patch("job_agent.scraper_industry_html.scrape_podnews")
    @patch("job_agent.scraper_industry_html.scrape_showbizjobs")
    def test_handles_scraper_exception(self, mock_sbj, mock_pod, mock_mbw, mock_doors, mock_digi, mock_rgj, mock_wh, mock_apmg, mock_phub, mock_hit, mock_bi):
        from job_agent.scraper_industry_html import scrape_industry_html_boards

        mock_sbj.side_effect = Exception("ShowbizJobs is down")
        mock_pod.return_value = [{"title": "Podcast Producer", "company": "X", "location": "", "job_url": "u1",
                                   "description": "", "date_posted": "", "site": "podnews", "department": ""}]
        mock_mbw.return_value = []
        mock_doors.return_value = []
        mock_digi.return_value = []
        mock_rgj.return_value = []
        mock_wh.return_value = []
        mock_apmg.return_value = []
        mock_phub.return_value = []
        mock_hit.return_value = []
        mock_bi.return_value = []

        df = scrape_industry_html_boards(KEYWORDS)
        # Should still return results from podnews despite showbizjobs failure
        assert len(df) == 1
        assert df.iloc[0]["site"] == "podnews"

    @patch("job_agent.scraper_industry_html.scrape_builtin")
    @patch("job_agent.scraper_industry_html.scrape_hitmarker")
    @patch("job_agent.scraper_industry_html.scrape_productionhub")
    @patch("job_agent.scraper_industry_html.scrape_apmg")
    @patch("job_agent.scraper_industry_html.scrape_willhire")
    @patch("job_agent.scraper_industry_html.scrape_remotegamejobs")
    @patch("job_agent.scraper_industry_html.scrape_digilogue")
    @patch("job_agent.scraper_industry_html.scrape_doorsopen")
    @patch("job_agent.scraper_industry_html.scrape_mbw")
    @patch("job_agent.scraper_industry_html.scrape_podnews")
    @patch("job_agent.scraper_industry_html.scrape_showbizjobs")
    def test_calls_progress_callback(self, mock_sbj, mock_pod, mock_mbw, mock_doors, mock_digi, mock_rgj, mock_wh, mock_apmg, mock_phub, mock_hit, mock_bi):
        from job_agent.scraper_industry_html import scrape_industry_html_boards

        for m in (mock_sbj, mock_pod, mock_mbw, mock_doors, mock_digi, mock_rgj, mock_wh, mock_apmg, mock_phub, mock_hit, mock_bi):
            m.return_value = []

        cb = MagicMock()
        scrape_industry_html_boards(KEYWORDS, progress_callback=cb)

        # Should call progress for each scraper + final summary
        assert cb.call_count >= 11


# ============================================================
# Config Integration
# ============================================================

class TestConfig:
    def test_industry_boards_in_search_sites(self):
        from job_agent.config import SEARCH_SITES
        assert "industry_boards" in SEARCH_SITES

    def test_showbizjobs_categories_defined(self):
        from job_agent.config import SHOWBIZJOBS_CATEGORIES
        assert len(SHOWBIZJOBS_CATEGORIES) >= 5
        assert "production-crew" in SHOWBIZJOBS_CATEGORIES
        assert "podcast" in SHOWBIZJOBS_CATEGORIES

    def test_entertainmentcareers_feeds_defined(self):
        from job_agent.config import ENTERTAINMENTCAREERS_FEEDS
        assert len(ENTERTAINMENTCAREERS_FEEDS) >= 8
        # Feeds outside the configured title scope (e.g. ecnjcat166) are not fetched.
        assert "ecnjcat166" not in ENTERTAINMENTCAREERS_FEEDS
        assert "ecnjcat119" in ENTERTAINMENTCAREERS_FEEDS
        assert "ecnjcat168" in ENTERTAINMENTCAREERS_FEEDS


class TestResolveKeywordCategoryLongestMatch:
    def test_longest_match_wins(self):
        """'Senior Podcast Producer' must resolve via 'Podcast Producer'
        (podcast_and_audio), not bare 'Producer' (producer_core)."""
        from job_agent.ats_scraper import _resolve_keyword_category
        from job_agent.config import KEYWORDS
        assert _resolve_keyword_category("Senior Podcast Producer", KEYWORDS) == "podcast_and_audio"
        assert _resolve_keyword_category("YouTube Producer", KEYWORDS) == "youtube_and_creators"
        assert _resolve_keyword_category("Senior Producer", KEYWORDS) == "producer_core"
        assert _resolve_keyword_category("Software Engineer", KEYWORDS) == ""


class TestAquentFeed:
    """Aquent creative-staffing XML feed (first-party, no auth)."""

    FEED = '''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item><job_id>1</job_id><title><![CDATA[Video Producer [211111]]]></title>
<location><city><![CDATA[Seattle]]></city><state>WA</state><country>US</country></location>
<company>Aquent</company><placement_type>Temporary</placement_type><remotetype>On site</remotetype>
<description><![CDATA[<p>Produce <strong>video</strong> content.</p>]]></description><salary>$50/hr</salary>
<segment>Video</segment><pubDate>Thu, 02 Jul 2026 01:00:00 GMT</pubDate>
<url>https://aquent.com/find-work/211111</url></item>
<item><job_id>2</job_id><title><![CDATA[Content Producer [222222]]]></title>
<location><city><![CDATA[New York]]></city><state>NY</state><country>US</country></location>
<company>Aquent</company><placement_type>Contract</placement_type><remotetype>Fully remote</remotetype>
<description><![CDATA[Remote content work.]]></description><salary></salary>
<segment>Content</segment><pubDate>Thu, 02 Jul 2026 01:00:00 GMT</pubDate>
<url>https://aquent.com/find-work/222222</url></item>
<item><job_id>3</job_id><title><![CDATA[Creative Producer [333333]]]></title>
<location><city><![CDATA[Boston]]></city><state>MA</state><country>US</country></location>
<company>Aquent</company><placement_type>Contract</placement_type><remotetype>Hybrid remote</remotetype>
<description><![CDATA[Hybrid role.]]></description><salary></salary>
<segment>Creative</segment><pubDate>Thu, 02 Jul 2026 01:00:00 GMT</pubDate>
<url>https://aquent.com/find-work/333333</url></item>
<item><job_id>4</job_id><title><![CDATA[Video Producer [444444]]]></title>
<location><city><![CDATA[Sydney]]></city><state>NSW</state><country>AU</country></location>
<company>Aquent</company><placement_type>Temporary</placement_type><remotetype>Fully remote</remotetype>
<description><![CDATA[AU role.]]></description><salary></salary>
<segment>Video</segment><pubDate>Thu, 02 Jul 2026 01:00:00 GMT</pubDate>
<url>https://aquent.com/find-work/444444</url></item>
<item><job_id>5</job_id><title><![CDATA[Video Producer [555555]]]></title>
<location><city><![CDATA[San Diego]]></city><state>CA</state><country>US</country></location>
<company>Aquent</company><placement_type>Temporary</placement_type><remotetype>On site</remotetype>
<description><![CDATA[On-site in another state.]]></description><salary></salary>
<segment>Video</segment><pubDate>Thu, 02 Jul 2026 01:00:00 GMT</pubDate>
<url>https://aquent.com/find-work/555555</url></item>
<item><job_id>6</job_id><title><![CDATA[Content Producer [666666]]]></title>
<location><city><![CDATA[Spokane]]></city><state>Washington</state><country>US</country></location>
<company>Aquent</company><placement_type>Contract</placement_type><remotetype>On site</remotetype>
<description><![CDATA[Home state, written in full.]]></description><salary></salary>
<segment>Content</segment><pubDate>Thu, 02 Jul 2026 01:00:00 GMT</pubDate>
<url>https://aquent.com/find-work/666666</url></item>
</channel></rss>'''

    def _scrape(self, monkeypatch):
        from job_agent import scraper_rss as sr
        class R:
            status_code = 200
            text = self.FEED
            def raise_for_status(self): pass
        monkeypatch.setattr(sr.requests, "get", lambda *a, **k: R())
        return sr.scrape_aquent_feed(["Video Producer", "Content Producer", "Creative Producer"])

    def test_home_region_and_fully_remote_only(self, monkeypatch):
        jobs = self._scrape(monkeypatch)
        urls = {j["job_url"] for j in jobs}
        assert "https://aquent.com/find-work/211111" in urls   # on-site in the profile's state (WA)
        assert "https://aquent.com/find-work/666666" in urls   # same, state written in full
        assert "https://aquent.com/find-work/222222" in urls   # US fully remote
        assert "https://aquent.com/find-work/333333" not in urls  # hybrid outside the state → out
        assert "https://aquent.com/find-work/444444" not in urls  # AU → out
        assert "https://aquent.com/find-work/555555" not in urls  # on-site California → out

    def test_onsite_home_state_item_keeps_its_city(self, monkeypatch):
        jobs = self._scrape(monkeypatch)
        sea = next(j for j in jobs if "211111" in j["job_url"])
        assert sea["location"] == "Seattle, WA"

    def test_title_id_suffix_stripped(self, monkeypatch):
        jobs = self._scrape(monkeypatch)
        assert all("[" not in j["title"] for j in jobs)

    def test_fully_remote_reads_remote_for_location_gate(self, monkeypatch):
        jobs = self._scrape(monkeypatch)
        ny = next(j for j in jobs if "222222" in j["job_url"])
        assert ny["location"] == "Remote"           # gate-compatible
        assert "Team base: New York" in ny["description"]  # city kept as context

    def test_html_stripped_and_salary_placement_noted(self, monkeypatch):
        jobs = self._scrape(monkeypatch)
        sea = next(j for j in jobs if "211111" in j["job_url"])
        assert "<" not in sea["description"]
        assert "Temporary via Aquent" in sea["description"]
        assert "$50/hr" in sea["description"]
