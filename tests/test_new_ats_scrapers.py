"""Tests for new ATS scrapers: Ashby, SmartRecruiters, Workday, Recruitee, Workable, Teamtailor, and Autodiscovery."""

import json
from unittest.mock import MagicMock, patch

import pytest

KEYWORDS = ["Content Producer", "Video Producer", "Creative Producer"]


# ============================================================
# Ashby
# ============================================================

class TestScrapeAshby:
    ASHBY_RESPONSE = {
        "jobs": [
            {
                "title": "Content Producer",
                "location": "San Francisco, CA",
                "isRemote": False,
                "isListed": True,
                "department": "Marketing",
                "team": "Content",
                "descriptionPlain": "Lead content production across channels.",
                "descriptionHtml": "<p>Lead content production.</p>",
                "publishedAt": "2026-01-15T10:00:00Z",
                "jobUrl": "https://jobs.ashbyhq.com/openai/abc-123",
            },
            {
                "title": "Software Engineer",
                "location": "Remote",
                "isRemote": True,
                "isListed": True,
                "department": "Engineering",
                "team": "Backend",
                "descriptionPlain": "Build systems.",
                "publishedAt": "2026-01-14",
                "jobUrl": "https://jobs.ashbyhq.com/openai/def-456",
            },
            {
                "title": "Video Producer",
                "location": "",
                "isRemote": True,
                "isListed": False,  # unlisted — should be skipped
                "department": "Creative",
                "team": "",
                "descriptionPlain": "Video work.",
                "publishedAt": "2026-01-13",
                "jobUrl": "https://jobs.ashbyhq.com/openai/ghi-789",
            },
        ]
    }

    @patch("job_agent.scraper_ashby_sr.requests.get")
    def test_parses_response(self, mock_get):
        from job_agent.scraper_ashby_sr import scrape_ashby

        mock_resp = MagicMock()
        mock_resp.json.return_value = self.ASHBY_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_ashby("openai", "OpenAI", KEYWORDS)

        # Only "Content Producer" should match (Software Engineer no match, Video Producer unlisted)
        assert len(results) == 1
        assert results[0]["title"] == "Content Producer"
        assert results[0]["company"] == "OpenAI"
        assert results[0]["site"] == "ashby"
        assert results[0]["department"] == "Marketing — Content"
        assert "content production" in results[0]["description"].lower()

    @patch("job_agent.scraper_ashby_sr.requests.get")
    def test_handles_api_error(self, mock_get):
        from job_agent.scraper_ashby_sr import scrape_ashby

        mock_get.side_effect = Exception("Timeout")
        results = scrape_ashby("bad", "Bad", KEYWORDS)
        assert results == []

    @patch("job_agent.scraper_ashby_sr.requests.get")
    def test_remote_appended_to_location(self, mock_get):
        from job_agent.scraper_ashby_sr import scrape_ashby

        response = {
            "jobs": [{
                "title": "Content Producer",
                "location": "New York",
                "isRemote": True,
                "isListed": True,
                "department": "",
                "team": "",
                "descriptionPlain": "Remote-friendly role.",
                "publishedAt": "",
                "jobUrl": "https://jobs.ashbyhq.com/test/123",
            }]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_ashby("test", "Test", KEYWORDS)
        assert len(results) == 1
        assert "Remote" in results[0]["location"]
        assert "New York" in results[0]["location"]


# ============================================================
# SmartRecruiters
# ============================================================

class TestScrapeSmartRecruiters:
    SR_RESPONSE = {
        "content": [
            {
                "name": "Creative Producer",
                "id": "post-123",
                "location": {"city": "San Francisco", "region": "CA", "country": "US", "remote": False},
                "department": {"label": "Marketing"},
                "releasedDate": "2026-01-15T00:00:00Z",
                "ref": "https://api.smartrecruiters.com/...",
            },
            {
                "name": "Data Analyst",
                "id": "post-456",
                "location": {"city": "NYC", "region": "NY", "country": "US", "remote": False},
                "department": {"label": "Analytics"},
                "releasedDate": "2026-01-14",
                "ref": "https://api.smartrecruiters.com/...",
            },
        ],
        "totalFound": 2,
    }

    @patch("job_agent.scraper_ashby_sr._sr_fetch_posting_description")
    @patch("job_agent.scraper_ashby_sr.requests.get")
    def test_parses_response(self, mock_get, mock_desc):
        from job_agent.scraper_ashby_sr import scrape_smartrecruiters

        mock_resp = MagicMock()
        mock_resp.json.return_value = self.SR_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        mock_desc.return_value = "Full description text."

        results = scrape_smartrecruiters("Visa", "Visa", KEYWORDS)

        assert len(results) == 1
        assert results[0]["title"] == "Creative Producer"
        assert results[0]["company"] == "Visa"
        assert results[0]["site"] == "smartrecruiters"
        assert "San Francisco" in results[0]["location"]
        assert results[0]["description"] == "Full description text."

    @patch("job_agent.scraper_ashby_sr.requests.get")
    def test_handles_api_error(self, mock_get):
        from job_agent.scraper_ashby_sr import scrape_smartrecruiters

        mock_get.side_effect = Exception("Connection refused")
        results = scrape_smartrecruiters("bad", "Bad", KEYWORDS)
        assert results == []


# ============================================================
# Workday
# ============================================================

class TestScrapeWorkday:
    @patch("job_agent.scraper_workday.requests.Session")
    def test_parses_response(self, mock_session_cls):
        from job_agent.scraper_workday import scrape_workday

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        # Mock POST for job listings
        list_resp = MagicMock()
        list_resp.json.return_value = {
            "jobPostings": [
                {
                    "title": "Content Producer",
                    "externalPath": "/job/content-producer/123",
                    "locationsText": "Burbank, CA",
                    "postedOn": "Posted 2 Days Ago",
                },
                {
                    "title": "Accountant",
                    "externalPath": "/job/accountant/456",
                    "locationsText": "New York",
                    "postedOn": "Posted Today",
                },
            ],
            "total": 2,
        }
        list_resp.raise_for_status.return_value = None
        mock_session.post.return_value = list_resp

        # Mock GET for job detail
        detail_resp = MagicMock()
        detail_resp.json.return_value = {
            "jobPostingInfo": {
                "jobDescription": "<p>Manage content production.</p>",
                "startDate": "2026-01-15",
            },
            "hiringOrganization": {"name": "Disney Studios"},
        }
        detail_resp.raise_for_status.return_value = None
        mock_session.get.return_value = detail_resp

        results = scrape_workday(
            "https://disney.wd5.myworkdayjobs.com/disneycareer",
            "Disney",
            KEYWORDS,
            fetch_details=True,
        )

        assert len(results) == 1
        assert results[0]["title"] == "Content Producer"
        assert results[0]["company"] == "Disney"
        assert results[0]["site"] == "workday"
        assert "Burbank" in results[0]["location"]
        assert "content production" in results[0]["description"].lower()

    def test_parse_base_url(self):
        from job_agent.scraper_workday import _parse_base_url

        host, slug, board = _parse_base_url(
            "https://disney.wd5.myworkdayjobs.com/disneycareer"
        )
        assert host == "https://disney.wd5.myworkdayjobs.com"
        assert slug == "disney"
        assert board == "disneycareer"

    def test_parse_base_url_with_locale(self):
        from job_agent.scraper_workday import _parse_base_url

        host, slug, board = _parse_base_url(
            "https://warnerbros.wd5.myworkdayjobs.com/en-US/global"
        )
        assert slug == "warnerbros"
        assert board == "global"

    def test_parse_base_url_invalid(self):
        from job_agent.scraper_workday import _parse_base_url

        with pytest.raises(ValueError):
            _parse_base_url("not-a-url")

    def test_parse_workday_relative_date(self):
        from job_agent.scraper_workday import _parse_workday_relative_date
        from datetime import datetime, timedelta

        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        assert _parse_workday_relative_date("Posted Today") == today
        assert _parse_workday_relative_date("Posted Yesterday") == yesterday
        assert _parse_workday_relative_date("") == ""

        result = _parse_workday_relative_date("Posted 5 Days Ago")
        expected = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        assert result == expected

    @patch("job_agent.scraper_workday.requests.Session")
    def test_handles_api_error(self, mock_session_cls):
        from job_agent.scraper_workday import scrape_workday

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        import requests as req
        mock_session.post.side_effect = req.exceptions.RequestException("Connection refused")

        results = scrape_workday(
            "https://disney.wd5.myworkdayjobs.com/disneycareer",
            "Disney",
            KEYWORDS,
        )
        assert results == []

    @patch("job_agent.scraper_workday.requests.Session")
    def test_deduplicates_across_keywords(self, mock_session_cls):
        from job_agent.scraper_workday import scrape_workday

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        # Same job appears for different keyword searches
        list_resp = MagicMock()
        list_resp.json.return_value = {
            "jobPostings": [{
                "title": "Content Producer",
                "externalPath": "/job/content-producer/123",
                "locationsText": "Remote",
                "postedOn": "Posted Today",
            }],
            "total": 1,
        }
        list_resp.raise_for_status.return_value = None
        mock_session.post.return_value = list_resp

        detail_resp = MagicMock()
        detail_resp.json.return_value = {
            "jobPostingInfo": {"jobDescription": "Desc"},
            "hiringOrganization": {},
        }
        detail_resp.raise_for_status.return_value = None
        mock_session.get.return_value = detail_resp

        results = scrape_workday(
            "https://disney.wd5.myworkdayjobs.com/disneycareer",
            "Disney",
            ["Content Producer", "Producer"],  # two keywords, same job returned
        )

        # Should only appear once despite matching both keywords
        assert len(results) == 1


# ============================================================
# Recruitee
# ============================================================

class TestScrapeRecruitee:
    RECRUITEE_RESPONSE = {
        "offers": [
            {
                "title": "Video Producer",
                "location": "Berlin, Germany",
                "department": "Content",
                "description": "<p>Create video content.</p>",
                "created_at": "2026-01-15T10:00:00Z",
                "careers_url": "https://fly-io.recruitee.com/o/video-producer",
                "slug": "video-producer",
                "status": "published",
            },
            {
                "title": "Software Engineer",
                "location": "Remote",
                "department": "Engineering",
                "description": "<p>Build infra.</p>",
                "created_at": "2026-01-14",
                "careers_url": "https://fly-io.recruitee.com/o/swe",
                "slug": "swe",
                "status": "published",
            },
            {
                "title": "Creative Producer",
                "location": "SF",
                "department": "Marketing",
                "description": "Closed role.",
                "created_at": "2025-06-01",
                "careers_url": "",
                "slug": "creative-producer",
                "status": "closed",
            },
        ]
    }

    @patch("job_agent.scraper_rwt.requests.get")
    def test_parses_response(self, mock_get):
        from job_agent.scraper_rwt import scrape_recruitee

        mock_resp = MagicMock()
        mock_resp.json.return_value = self.RECRUITEE_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_recruitee("fly-io", "Fly.io", KEYWORDS)

        # Only "Video Producer" matches (SWE no match, Creative Producer is closed)
        assert len(results) == 1
        assert results[0]["title"] == "Video Producer"
        assert results[0]["company"] == "Fly.io"
        assert results[0]["site"] == "recruitee"
        assert results[0]["department"] == "Content"
        assert "video content" in results[0]["description"].lower()

    @patch("job_agent.scraper_rwt.requests.get")
    def test_handles_api_error(self, mock_get):
        from job_agent.scraper_rwt import scrape_recruitee

        mock_get.side_effect = Exception("Timeout")
        results = scrape_recruitee("bad", "Bad", KEYWORDS)
        assert results == []

    @patch("job_agent.scraper_rwt.requests.get")
    def test_fallback_url_from_slug(self, mock_get):
        from job_agent.scraper_rwt import scrape_recruitee

        response = {
            "offers": [{
                "title": "Content Producer",
                "location": "Remote",
                "department": "",
                "description": "Great role.",
                "created_at": "2026-01-15",
                "careers_url": "",  # empty — should construct from slug
                "slug": "content-prod",
                "status": "published",
            }]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_recruitee("myco", "MyCo", KEYWORDS)
        assert len(results) == 1
        assert "myco.recruitee.com/o/content-prod" in results[0]["job_url"]


# ============================================================
# Workable
# ============================================================

class TestScrapeWorkable:
    WORKABLE_RESPONSE = {
        "results": [
            {
                "title": "Content Producer",
                "shortcode": "ABC123",
                "department": "Marketing",
                "shortDescription": "Manage content pipeline.",
                "location": {"city": "San Francisco", "region": "CA", "country": "US", "remote": True},
                "published": "2026-01-15T10:00:00Z",
                "url": "https://apply.workable.com/traveloka/j/ABC123/",
            },
            {
                "title": "Data Scientist",
                "shortcode": "DEF456",
                "department": "Analytics",
                "shortDescription": "Analyze data.",
                "location": {"city": "NYC", "region": "NY", "country": "US", "remote": False},
                "published": "2026-01-14",
                "url": "",
            },
        ],
        "nextToken": None,
    }

    @patch("job_agent.scraper_rwt.requests.post")
    def test_parses_response(self, mock_post):
        from job_agent.scraper_rwt import scrape_workable

        mock_resp = MagicMock()
        mock_resp.json.return_value = self.WORKABLE_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        results = scrape_workable("traveloka", "Traveloka", KEYWORDS)

        assert len(results) == 1
        assert results[0]["title"] == "Content Producer"
        assert results[0]["company"] == "Traveloka"
        assert results[0]["site"] == "workable"
        assert "San Francisco" in results[0]["location"]
        assert "Remote" in results[0]["location"]
        assert "content pipeline" in results[0]["description"].lower()

    @patch("job_agent.scraper_rwt.requests.post")
    def test_handles_api_error(self, mock_post):
        from job_agent.scraper_rwt import scrape_workable

        mock_post.side_effect = Exception("Timeout")
        results = scrape_workable("bad", "Bad", KEYWORDS)
        assert results == []


# ============================================================
# Teamtailor
# ============================================================

class TestScrapeTeamtailor:
    TEAMTAILOR_HTML_WITH_JSONLD = """
    <html><head>
    <script type="application/ld+json">
    [
        {"@type": "JobPosting", "title": "Video Producer",
         "jobLocation": {"address": {"addressLocality": "Stockholm", "addressRegion": ""}},
         "jobLocationType": "TELECOMMUTE",
         "url": "https://northvolt.teamtailor.com/jobs/123",
         "description": "Create video content for brand.",
         "datePosted": "2026-01-15"},
        {"@type": "JobPosting", "title": "Software Engineer",
         "jobLocation": {"address": {"addressLocality": "Berlin"}},
         "url": "https://northvolt.teamtailor.com/jobs/456",
         "description": "Build systems.",
         "datePosted": "2026-01-14"}
    ]
    </script>
    </head><body></body></html>
    """

    @patch("job_agent.scraper_rwt.requests.get")
    def test_parses_jsonld(self, mock_get):
        from job_agent.scraper_rwt import scrape_teamtailor

        mock_resp = MagicMock()
        mock_resp.text = self.TEAMTAILOR_HTML_WITH_JSONLD
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_teamtailor("northvolt", "Northvolt", KEYWORDS)

        assert len(results) == 1
        assert results[0]["title"] == "Video Producer"
        assert results[0]["company"] == "Northvolt"
        assert results[0]["site"] == "teamtailor"
        assert "Stockholm" in results[0]["location"]
        assert "Remote" in results[0]["location"]

    @patch("job_agent.scraper_rwt.requests.get")
    def test_handles_fetch_error(self, mock_get):
        from job_agent.scraper_rwt import scrape_teamtailor

        mock_get.side_effect = Exception("Timeout")
        results = scrape_teamtailor("bad", "Bad", KEYWORDS)
        assert results == []


# ============================================================
# ATS Autodiscovery
# ============================================================

class TestAtsAutodiscovery:
    def test_detect_greenhouse(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        result = detect_ats_from_url("https://boards.greenhouse.io/airbnb")
        assert result == ("greenhouse", "airbnb")

    def test_detect_greenhouse_api(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        result = detect_ats_from_url("https://boards-api.greenhouse.io/v1/boards/stripe/jobs")
        assert result == ("greenhouse", "stripe")

    def test_detect_lever(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        result = detect_ats_from_url("https://jobs.lever.co/spotify")
        assert result == ("lever", "spotify")

    def test_detect_ashby(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        result = detect_ats_from_url("https://jobs.ashbyhq.com/ramp")
        assert result == ("ashby", "ramp")

    def test_detect_smartrecruiters(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        result = detect_ats_from_url("https://jobs.smartrecruiters.com/Visa")
        assert result == ("smartrecruiters", "Visa")

    def test_detect_workable(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        result = detect_ats_from_url("https://apply.workable.com/traveloka/")
        assert result == ("workable", "traveloka")

    def test_detect_recruitee(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        result = detect_ats_from_url("https://fly-io.recruitee.com/")
        assert result == ("recruitee", "fly-io")

    def test_detect_teamtailor(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        result = detect_ats_from_url("https://northvolt.teamtailor.com/jobs")
        assert result == ("teamtailor", "northvolt")

    def test_detect_workday(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        result = detect_ats_from_url("https://disney.wd5.myworkdayjobs.com/disneycareer")
        assert result is not None
        assert result[0] == "workday"
        assert "disney" in result[1]

    def test_detect_unknown(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        assert detect_ats_from_url("https://example.com/careers") is None
        assert detect_ats_from_url("https://www.google.com") is None

    def test_detect_invalid_url(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        assert detect_ats_from_url("not-a-url") is None

    def test_detect_bare_host_no_token(self):
        from job_agent.ats_autodiscovery import detect_ats_from_url

        assert detect_ats_from_url("https://boards.greenhouse.io/") is None
        assert detect_ats_from_url("https://boards.greenhouse.io") is None

    @patch("job_agent.ats_autodiscovery.requests.get")
    def test_discover_from_redirect(self, mock_get):
        from job_agent.ats_autodiscovery import discover_ats_from_careers_page

        mock_resp = MagicMock()
        mock_resp.url = "https://boards.greenhouse.io/airbnb"  # redirected to GH
        mock_resp.text = "<html></html>"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = discover_ats_from_careers_page("https://airbnb.com/careers")
        assert result == ("greenhouse", "airbnb")

    @patch("job_agent.ats_autodiscovery.requests.get")
    def test_discover_from_embedded_link(self, mock_get):
        from job_agent.ats_autodiscovery import discover_ats_from_careers_page

        mock_resp = MagicMock()
        mock_resp.url = "https://company.com/careers"  # not an ATS URL itself
        mock_resp.text = '''
        <html><body>
            <a href="https://jobs.lever.co/mycompany">See all jobs</a>
        </body></html>
        '''
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = discover_ats_from_careers_page("https://company.com/careers")
        assert result == ("lever", "mycompany")

    def test_resolve_company_ats_already_specified(self):
        from job_agent.ats_autodiscovery import resolve_company_ats

        entry = {"company": "Airbnb", "ats": "greenhouse", "token": "airbnb"}
        result = resolve_company_ats(entry, cache={})
        assert result["ats"] == "greenhouse"
        assert result["token"] == "airbnb"

    def test_resolve_company_ats_no_careers_url(self):
        from job_agent.ats_autodiscovery import resolve_company_ats

        entry = {"company": "Unknown"}
        result = resolve_company_ats(entry, cache={})
        assert "ats" not in result or not result.get("ats")


# ============================================================
# Dispatcher integration — all ATS types routed correctly
# ============================================================

class TestDispatcherIntegration:
    @patch("job_agent.ats_scraper.scrape_greenhouse")
    def test_dispatches_greenhouse(self, mock_gh):
        from job_agent.ats_scraper import scrape_ats_boards

        mock_gh.return_value = [{"title": "P", "company": "A", "site": "greenhouse",
                                  "location": "", "job_url": "", "description": "",
                                  "date_posted": "", "department": ""}]
        companies = [{"company": "A", "ats": "greenhouse", "token": "a"}]
        df = scrape_ats_boards(companies, ["P"])
        assert len(df) == 1
        mock_gh.assert_called_once()

    @patch("job_agent.scraper_workday.scrape_workday")
    def test_dispatches_workday(self, mock_wd):
        from job_agent.ats_scraper import scrape_ats_boards

        mock_wd.return_value = [{"title": "P", "company": "D", "site": "workday",
                                  "location": "", "job_url": "", "description": "",
                                  "date_posted": "", "department": ""}]
        companies = [{"company": "D", "ats": "workday", "token": "https://d.wd5.myworkdayjobs.com/x"}]
        df = scrape_ats_boards(companies, ["P"])
        assert len(df) == 1
        mock_wd.assert_called_once()

    @patch("job_agent.scraper_rwt.scrape_recruitee")
    def test_dispatches_recruitee(self, mock_rec):
        from job_agent.ats_scraper import scrape_ats_boards

        mock_rec.return_value = []
        companies = [{"company": "R", "ats": "recruitee", "token": "r"}]
        df = scrape_ats_boards(companies, ["P"])
        assert df.empty
        mock_rec.assert_called_once()

    @patch("job_agent.scraper_rwt.scrape_workable")
    def test_dispatches_workable(self, mock_wa):
        from job_agent.ats_scraper import scrape_ats_boards

        mock_wa.return_value = []
        companies = [{"company": "W", "ats": "workable", "token": "w"}]
        df = scrape_ats_boards(companies, ["P"])
        assert df.empty
        mock_wa.assert_called_once()

    @patch("job_agent.scraper_rwt.scrape_teamtailor")
    def test_dispatches_teamtailor(self, mock_tt):
        from job_agent.ats_scraper import scrape_ats_boards

        mock_tt.return_value = []
        companies = [{"company": "T", "ats": "teamtailor", "token": "t"}]
        df = scrape_ats_boards(companies, ["P"])
        assert df.empty
        mock_tt.assert_called_once()


# ============================================================
# Eightfold.ai (Netflix)
# ============================================================

class TestScrapeEightfold:
    EIGHTFOLD_RESPONSE = {
        "count": 3,
        "positions": [
            {
                "id": "abc123",
                "name": "Content Producer",
                "location": "Los Angeles, CA",
                "department": "Marketing",
                "t_create": "2026-01-10T00:00:00Z",
                "t_update": "2026-01-15T00:00:00Z",
            },
            {
                "id": "def456",
                "name": "Software Engineer",
                "location": "San Jose, CA",
                "department": "Engineering",
                "t_create": "2026-01-12T00:00:00Z",
                "t_update": "2026-01-14T00:00:00Z",
            },
            {
                "id": "ghi789",
                "name": "Video Producer",
                "location": "New York, NY",
                "department": "Content",
                "t_create": "2026-01-11T00:00:00Z",
                "t_update": "2026-01-13T00:00:00Z",
            },
        ],
    }

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_parse_and_filter(self, mock_get):
        from job_agent.scraper_eightfold_apple import scrape_eightfold

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self.EIGHTFOLD_RESPONSE
        mock_get.return_value = mock_resp

        jobs = scrape_eightfold("netflix.com", "Netflix", KEYWORDS)
        assert len(jobs) == 2  # Content Producer + Video Producer
        assert jobs[0]["title"] == "Content Producer"
        assert jobs[0]["company"] == "Netflix"
        assert "abc123" in jobs[0]["link"]
        assert jobs[1]["title"] == "Video Producer"

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_fetch_failure(self, mock_get):
        from job_agent.scraper_eightfold_apple import scrape_eightfold

        mock_get.side_effect = Exception("Connection refused")
        jobs = scrape_eightfold("netflix.com", "Netflix", KEYWORDS)
        assert jobs == []

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_pagination(self, mock_get):
        from job_agent.scraper_eightfold_apple import scrape_eightfold

        page1 = {"count": 150, "positions": [
            {"id": f"p{i}", "name": "Content Producer", "location": "LA", "department": ""}
            for i in range(100)
        ]}
        page2 = {"count": 150, "positions": [
            {"id": f"q{i}", "name": "Content Producer", "location": "LA", "department": ""}
            for i in range(50)
        ]}

        mock_resp1 = MagicMock()
        mock_resp1.json.return_value = page1
        mock_resp2 = MagicMock()
        mock_resp2.json.return_value = page2

        mock_get.side_effect = [mock_resp1, mock_resp2]

        jobs = scrape_eightfold("netflix.com", "Netflix", KEYWORDS)
        assert len(jobs) == 150
        assert mock_get.call_count == 2


# ============================================================
# Apple Careers (SSR hydration)
# ============================================================

class TestScrapeApple:
    HYDRATION_DATA = {
        "loaderData": {
            "root": {"appType": "external"},
            "search": {
                "searchResults": [
                    {
                        "id": 111,
                        "positionId": "200601234",
                        "postingTitle": "Creative Producer",
                        "jobSummary": "Lead creative production.",
                        "postingDate": "Jan 10, 2026",
                        "postDateInGMT": "2026-01-10T00:00:00.000Z",
                        "locations": [
                            {"city": "Austin", "stateProvince": "Texas", "countryName": "United States"},
                        ],
                    },
                    {
                        "id": 222,
                        "positionId": "200605678",
                        "postingTitle": "Data Scientist",
                        "jobSummary": "Build ML models.",
                        "postingDate": "Jan 12, 2026",
                        "postDateInGMT": "2026-01-12T00:00:00.000Z",
                        "locations": [
                            {"city": "Seattle", "stateProvince": "Washington", "countryName": "United States"},
                        ],
                    },
                ],
            },
        },
        "actionData": None,
        "errors": None,
    }

    def _make_html(self, data=None):
        if data is None:
            data = self.HYDRATION_DATA
        escaped = json.dumps(json.dumps(data))  # double-encode for JS
        return f'<html><script>window.__staticRouterHydrationData = JSON.parse({escaped});</script></html>'

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_parse_and_filter(self, mock_get):
        from job_agent.scraper_eightfold_apple import scrape_apple

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = self._make_html()
        mock_get.return_value = mock_resp

        jobs = scrape_apple("en-us", "Apple", KEYWORDS)
        assert len(jobs) == 1  # Only Creative Producer
        assert jobs[0]["title"] == "Creative Producer"
        assert jobs[0]["company"] == "Apple"
        assert "200601234" in jobs[0]["link"]
        assert jobs[0]["location"] == "Austin, Texas, United States"

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_fetch_failure(self, mock_get):
        from job_agent.scraper_eightfold_apple import scrape_apple

        mock_get.side_effect = Exception("Connection refused")
        jobs = scrape_apple("en-us", "Apple", KEYWORDS)
        assert jobs == []

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_empty_hydration(self, mock_get):
        from job_agent.scraper_eightfold_apple import scrape_apple

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>No jobs</body></html>"
        mock_get.return_value = mock_resp

        jobs = scrape_apple("en-us", "Apple", KEYWORDS)
        assert jobs == []

    def test_extract_hydration_helper(self):
        from job_agent.scraper_eightfold_apple import _extract_apple_hydration

        html = self._make_html()
        results = _extract_apple_hydration(html)
        assert len(results) == 2
        assert results[0]["postingTitle"] == "Creative Producer"


# ============================================================
# Dispatcher: eightfold + apple
# ============================================================

class TestDispatcherNewTypes:
    @patch("job_agent.scraper_eightfold_apple.scrape_eightfold")
    def test_eightfold_dispatch(self, mock_ef):
        from job_agent.ats_scraper import scrape_ats_boards

        mock_ef.return_value = [{"title": "Producer", "company": "Netflix"}]
        companies = [{"company": "Netflix", "ats": "eightfold", "token": "netflix.com"}]
        df = scrape_ats_boards(companies, ["Producer"])
        assert len(df) == 1
        mock_ef.assert_called_once_with("netflix.com", "Netflix", ["Producer"])

    @patch("job_agent.scraper_eightfold_apple.scrape_apple")
    def test_apple_dispatch(self, mock_ap):
        from job_agent.ats_scraper import scrape_ats_boards

        mock_ap.return_value = [{"title": "Producer", "company": "Apple"}]
        companies = [{"company": "Apple", "ats": "apple", "token": "en-us"}]
        df = scrape_ats_boards(companies, ["Producer"])
        assert len(df) == 1
        mock_ap.assert_called_once_with("en-us", "Apple", ["Producer"])
