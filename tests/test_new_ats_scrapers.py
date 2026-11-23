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

    @staticmethod
    def _one_page_session(mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        list_resp = MagicMock()
        list_resp.json.return_value = {
            "jobPostings": [{
                "title": "Content Producer",
                "externalPath": "/job/Austin/Content-Producer_R-1",
                "locationsText": "Austin, TX",
                "postedOn": "Posted Today",
            }],
            "total": 1,
        }
        list_resp.raise_for_status.return_value = None
        mock_session.post.return_value = list_resp
        return mock_session

    @patch("job_agent.scraper_workday.requests.Session")
    def test_sends_named_user_agent(self, mock_session_cls):
        """Workday's Cloudflare answers 429 to the python-requests default UA
        (on every board), so the session must name itself."""
        from job_agent.scraper_workday import scrape_workday

        mock_session = self._one_page_session(mock_session_cls)
        scrape_workday("https://disney.wd5.myworkdayjobs.com/disneycareer",
                       "Disney", KEYWORDS, fetch_details=False)

        headers = mock_session.headers.update.call_args[0][0]
        assert headers["User-Agent"]
        assert "python-requests" not in headers["User-Agent"]

    @patch("job_agent.scraper_workday.requests.Session")
    def test_token_query_becomes_applied_facets(self, mock_session_cls):
        """A shared board (Omnicom's) is filtered to one agency by the facet in
        the token's query string; the job link must not carry that query."""
        from job_agent.scraper_workday import scrape_workday

        mock_session = self._one_page_session(mock_session_cls)
        results = scrape_workday(
            "https://interpublic.wd5.myworkdayjobs.com/OMC?Agency_Facet=abc123",
            "GSD&M", KEYWORDS, fetch_details=False,
        )

        url = mock_session.post.call_args[0][0]
        payload = mock_session.post.call_args[1]["json"]
        assert url == "https://interpublic.wd5.myworkdayjobs.com/wday/cxs/interpublic/OMC/jobs"
        assert payload["appliedFacets"] == {"Agency_Facet": ["abc123"]}
        assert results[0]["job_url"] == (
            "https://interpublic.wd5.myworkdayjobs.com/OMC/job/Austin/Content-Producer_R-1"
        )

    @patch("job_agent.scraper_workday.requests.Session")
    def test_plain_token_sends_no_facets(self, mock_session_cls):
        from job_agent.scraper_workday import scrape_workday

        mock_session = self._one_page_session(mock_session_cls)
        scrape_workday("https://condenast.wd115.myworkdayjobs.com/CondeCareers",
                       "Condé Nast", KEYWORDS, fetch_details=False)

        assert "appliedFacets" not in mock_session.post.call_args[1]["json"]


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

    @pytest.fixture(autouse=True)
    def _no_sleep(self):
        # The scraper pauses between requests and backs off after a timeout.
        with patch("job_agent.scraper_eightfold_apple.time.sleep") as sleep:
            self.sleep = sleep
            yield

    def _make_html(self, data=None):
        if data is None:
            data = self.HYDRATION_DATA
        escaped = json.dumps(json.dumps(data))  # double-encode for JS
        return f'<html><script>window.__staticRouterHydrationData = JSON.parse({escaped});</script></html>'

    @staticmethod
    def _days_ago(days=0):
        from datetime import datetime, timedelta, timezone
        return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def _search_html(self, results):
        return self._make_html({"loaderData": {"search": {"searchResults": results}}})

    @staticmethod
    def _resp(text="", payload=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = text
        resp.json.return_value = payload if payload is not None else {}
        resp.raise_for_status.return_value = None
        return resp

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

    def test_hydration_keeps_non_ascii_titles(self):
        """unicode_escape read the UTF-8 bytes as Latin-1 and stored an en dash
        in a title as mojibake ('Video Producer â...')."""
        from job_agent.scraper_eightfold_apple import _extract_apple_hydration

        title = "Video Producer – Example Team for Music & Podcasts"
        payload = json.dumps({"loaderData": {"search": {"searchResults": [{"postingTitle": title}]}}},
                             ensure_ascii=False)
        escaped = json.dumps(payload, ensure_ascii=False)[1:-1]
        html = f'<script>window.__staticRouterHydrationData = JSON.parse("{escaped}");</script>'
        assert _extract_apple_hydration(html)[0]["postingTitle"] == title

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_one_row_per_posting_with_its_locations(self, mock_get):
        """Apple lists a posting once per location (id '<positionId>-<loc>');
        it is one job, so one row, every city, one detail read."""
        from job_agent.scraper_eightfold_apple import scrape_apple

        fresh = self._days_ago(1)
        rows = [
            {"id": "200600009-0101", "positionId": "200600009", "postingTitle": "Creative Producer",
             "jobSummary": "s", "postDateInGMT": fresh,
             "locations": [{"city": "", "name": "Austin", "countryName": "United States of America"}]},
            {"id": "200600009-0102", "positionId": "200600009", "postingTitle": "Creative Producer",
             "jobSummary": "s", "postDateInGMT": fresh,
             "locations": [{"city": "", "name": "Seattle", "countryName": "United States of America"}]},
        ]
        search_html = self._search_html(rows)

        def get(url, params=None, **kw):
            if "/api/v1/jobDetails/" in url:
                return self._resp(payload={"res": {"postingTitle": "Creative Producer",
                                                   "highJobTitle": "Producer IC3"}})
            return self._resp(search_html if params["page"] == 1 else "<html></html>")
        mock_get.side_effect = get

        jobs = scrape_apple("en-us", "Apple", KEYWORDS)

        assert len(jobs) == 1
        assert jobs[0]["location"] == ("Austin, United States of America; "
                                       "Seattle, United States of America")
        assert jobs[0]["link"] == "https://jobs.apple.com/en-us/details/200600009"
        assert jobs[0]["job_level"] == "Producer IC3"
        detail_calls = [c for c in mock_get.call_args_list if "/api/v1/jobDetails/" in c.args[0]]
        assert len(detail_calls) == 1
        search_call = next(c for c in mock_get.call_args_list if "/search" in c.args[0])
        assert search_call.kwargs["params"]["location"] == "united-states-USA"

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_timeout_is_retried_after_a_wait(self, mock_get):
        import requests as _requests
        from job_agent.scraper_eightfold_apple import scrape_apple

        search_html = self._search_html([{
            "id": "1-1", "positionId": "1", "postingTitle": "Creative Producer",
            "postDateInGMT": self._days_ago(), "locations": []}])
        answers = iter([_requests.ReadTimeout("read timed out"), self._resp(search_html)])

        def get(url, params=None, **kw):
            if "/search" in url and params["search"] == "producer" and params["page"] == 1:
                answer = next(answers)
                if isinstance(answer, Exception):
                    raise answer
                return answer
            if "/api/v1/jobDetails/" in url:
                return self._resp(payload={"res": {"postingTitle": "Creative Producer"}})
            return self._resp("<html></html>")
        mock_get.side_effect = get

        jobs = scrape_apple("en-us", "Apple", KEYWORDS)

        assert [j["title"] for j in jobs] == ["Creative Producer"]
        assert 5 in [c.args[0] for c in self.sleep.call_args_list]   # the backoff wait
        assert mock_get.call_args.kwargs["timeout"][1] > 15          # read timeout raised

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_stops_when_apple_does_not_answer(self, mock_get):
        """When every seed's page 1 times out: two requests that fail after
        all their retries end the Apple scrape for the run."""
        import requests as _requests
        from job_agent.scraper_eightfold_apple import scrape_apple

        mock_get.side_effect = _requests.ReadTimeout("read timed out")
        jobs = scrape_apple("en-us", "Apple", KEYWORDS)

        assert jobs == []
        assert mock_get.call_count == 2 * 3   # 2 requests x (1 try + 2 retries)

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_a_404_does_not_stop_the_scrape(self, mock_get):
        import requests as _requests
        from job_agent.scraper_eightfold_apple import scrape_apple

        def get(url, params=None, **kw):
            if "/search" in url:
                if params["search"] in ("producer", "production"):
                    resp = self._resp()
                    resp.raise_for_status.side_effect = _requests.HTTPError(
                        response=MagicMock(status_code=404))
                    return resp
                if params["search"] == "content" and params["page"] == 1:
                    return self._resp(self._search_html([{
                        "id": "9-1", "positionId": "9", "postingTitle": "Video Producer",
                        "postDateInGMT": self._days_ago(), "locations": []}]))
            if "/api/v1/jobDetails/" in url:
                return self._resp(payload={"res": {"postingTitle": "Video Producer"}})
            return self._resp("<html></html>")
        mock_get.side_effect = get

        jobs = scrape_apple("en-us", "Apple", KEYWORDS)
        assert [j["title"] for j in jobs] == ["Video Producer"]

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_stale_postings_end_paging_and_skip_the_detail_read(self, mock_get):
        """The search is newest first, and the pre-filter drops anything older
        than STALE_POSTING_DAYS whatever its level, so neither later pages nor
        that posting's job page are worth a request."""
        from job_agent.scraper_eightfold_apple import scrape_apple

        old = self._search_html([{
            "id": "5-1", "positionId": "5", "postingTitle": "Creative Producer",
            "jobSummary": "s", "postDateInGMT": self._days_ago(60), "locations": []}])
        mock_get.side_effect = lambda url, params=None, **kw: self._resp(old)

        jobs = scrape_apple("en-us", "Apple", KEYWORDS)

        assert len(jobs) == 1 and jobs[0]["job_level"] == ""   # still returned, as before
        urls = [c.args[0] for c in mock_get.call_args_list]
        assert not [u for u in urls if "jobDetails" in u or "/details/" in u]
        pages = [c.kwargs["params"]["page"] for c in mock_get.call_args_list]
        assert pages == [1, 1, 1, 1]          # one page per seed

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_a_page_of_postings_seen_under_another_seed_does_not_end_paging(self, mock_get):
        """The seeds overlap ('producer' / 'production'): a page holding only
        postings already seen must not end that seed's paging."""
        from job_agent.scraper_eightfold_apple import scrape_apple

        def row(pid, title):
            return {"id": f"{pid}-1", "positionId": pid, "postingTitle": title,
                    "postDateInGMT": self._days_ago(), "locations": []}

        pages = {("producer", 1): [row("1", "Creative Producer")],
                 ("production", 1): [row("1", "Creative Producer")],
                 ("production", 2): [row("2", "Video Producer")]}

        def get(url, params=None, **kw):
            if "/api/v1/jobDetails/" in url:
                return self._resp(payload={"res": {"postingTitle": "x"}})
            found = pages.get((params["search"], params["page"]))
            return self._resp(self._search_html(found) if found else "<html></html>")
        mock_get.side_effect = get

        jobs = scrape_apple("en-us", "Apple", KEYWORDS)
        assert sorted(j["title"] for j in jobs) == ["Creative Producer", "Video Producer"]

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_time_budget_ends_the_scrape(self, mock_get):
        from job_agent import scraper_eightfold_apple as ap

        clock = iter(range(0, 10_000, 200))   # each look at the clock: +200s
        mock_get.side_effect = lambda url, params=None, **kw: self._resp(self._search_html([{
            "id": f"{params['search']}-{params['page']}",
            "positionId": f"{params['search']}{params['page']}",
            "postingTitle": "Creative Producer", "postDateInGMT": self._days_ago(), "locations": []}]))
        with patch.object(ap.time, "monotonic", lambda: next(clock)):
            ap.scrape_apple("en-us", "Apple", KEYWORDS)
        assert mock_get.call_count < 4

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_search_leaves_half_the_budget_for_the_job_reads(self, mock_get):
        """Apple slow but answering: each search page takes 40s. Paging used
        to spend the whole 300s, and every row came back without level and
        pay. It now stops past 150s, and the job reads still happen."""
        from job_agent import scraper_eightfold_apple as ap

        now = [0.0]

        def get(url, params=None, **kw):
            if "/api/v1/jobDetails/" in url:
                now[0] += 3
                return self._resp(payload={"res": {"postingTitle": "x",
                                                   "highJobTitle": "Producer IC3"}})
            now[0] += 40
            pid = f"{params['search']}{params['page']}"
            return self._resp(self._search_html([{
                "id": f"{pid}-1", "positionId": pid, "postingTitle": "Creative Producer",
                "postDateInGMT": self._days_ago(), "locations": []}]))
        mock_get.side_effect = get

        with patch.object(ap.time, "monotonic", lambda: now[0]):
            jobs = ap.scrape_apple("en-us", "Apple", KEYWORDS)

        search_calls = [c for c in mock_get.call_args_list if "/search" in c.args[0]]
        assert len(search_calls) == 4          # pages at 0, 40, 80, 120s; none after 150s
        assert len(jobs) == 4
        assert all(j["job_level"] == "Producer IC3" for j in jobs)

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_job_reads_go_newest_first_before_the_cap(self, mock_get):
        from job_agent import scraper_eightfold_apple as ap

        rows = [{"id": f"{pid}-1", "positionId": pid, "postingTitle": "Creative Producer",
                 "postDateInGMT": self._days_ago(days), "locations": []}
                for pid, days in (("old", 5), ("newest", 0), ("middle", 2))]
        search_html = self._search_html(rows)

        def get(url, params=None, **kw):
            if "/api/v1/jobDetails/" in url:
                return self._resp(payload={"res": {"postingTitle": "x",
                                                   "highJobTitle": "Producer IC3"}})
            return self._resp(search_html if params["page"] == 1 else "<html></html>")
        mock_get.side_effect = get

        with patch.object(ap, "_APPLE_MAX_DETAILS", 2):
            jobs = ap.scrape_apple("en-us", "Apple", KEYWORDS)

        read = [c.args[0].rsplit("/", 1)[1] for c in mock_get.call_args_list
                if "/api/v1/jobDetails/" in c.args[0]]
        assert read == ["newest", "middle"]
        levels = {j["link"].rsplit("/", 1)[1]: j["job_level"] for j in jobs}
        assert levels == {"old": "", "newest": "Producer IC3", "middle": "Producer IC3"}

    def test_hydration_with_a_quote_then_paren_in_a_posting(self):
        """'(.+?)")' stopped at the first '")' inside a posting text and the
        page parsed to 0 rows, silently."""
        from job_agent.scraper_eightfold_apple import _extract_apple_hydration

        data = json.loads(json.dumps(self.HYDRATION_DATA))
        summary = 'Produce shows for the "Studio") and more (see "Notes")'
        data["loaderData"]["search"]["searchResults"][0]["jobSummary"] = summary
        html = self._make_html(data) + '<script>JSON.parse("{}")</script>'
        results = _extract_apple_hydration(html)
        assert len(results) == 2
        assert results[0]["jobSummary"] == summary
        assert results[1]["postingTitle"] == "Data Scientist"

    def test_hydration_with_odd_entries_keeps_the_good_ones(self):
        from job_agent import scraper_eightfold_apple as ap

        html = self._search_html(["oops", None, {
            "id": "7-1", "positionId": "7", "postingTitle": "Creative Producer",
            "postDateInGMT": self._days_ago(60), "locations": ["Austin", None]}])
        with patch.object(ap.requests, "get", return_value=self._resp(html)):
            jobs = ap.scrape_apple("en-us", "Apple", KEYWORDS)
        assert [j["title"] for j in jobs] == ["Creative Producer"]


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
