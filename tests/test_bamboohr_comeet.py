"""Tests for BambooHR and Comeet scrapers."""

from unittest.mock import MagicMock, patch

import pytest

KEYWORDS = ["Content Producer", "Video Producer", "Creative Producer"]


# ============================================================
# BambooHR
# ============================================================

class TestScrapeBambooHR:
    BAMBOOHR_RESPONSE = {
        "meta": {"totalCount": 4},
        "result": [
            {
                "id": "101",
                "jobOpeningName": "Content Producer",
                "departmentId": "1000",
                "departmentLabel": "Marketing",
                "employmentStatusLabel": "Full-Time",
                "location": {"city": "Los Angeles", "state": "California"},
                "isRemote": None,
                "locationType": "0",
            },
            {
                "id": "102",
                "jobOpeningName": "Software Engineer",
                "departmentId": "2000",
                "departmentLabel": "Engineering",
                "employmentStatusLabel": "Full-Time",
                "location": {"city": "San Francisco", "state": "California"},
                "isRemote": None,
                "locationType": "0",
            },
            {
                "id": "103",
                "jobOpeningName": "Video Producer",
                "departmentId": "1000",
                "departmentLabel": "Marketing",
                "employmentStatusLabel": "Full-Time",
                "location": {"city": "", "state": ""},
                "isRemote": True,
                "locationType": "1",
            },
            {
                "id": "104",
                "jobOpeningName": "HR Coordinator",
                "departmentId": "3000",
                "departmentLabel": "Admin",
                "employmentStatusLabel": "Full-Time",
                "location": {"city": "Austin", "state": "Texas"},
                "isRemote": None,
                "locationType": "0",
            },
        ],
    }

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_parses_response(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_bamboohr

        mock_resp = MagicMock()
        mock_resp.json.return_value = self.BAMBOOHR_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_bamboohr("testcompany", "TestCo", KEYWORDS)

        assert len(results) == 2
        assert results[0]["title"] == "Content Producer"
        assert results[0]["company"] == "TestCo"
        assert results[0]["site"] == "bamboohr"
        assert results[0]["location"] == "Los Angeles, California"
        assert results[0]["department"] == "Marketing"
        assert results[0]["job_url"] == "https://testcompany.bamboohr.com/careers/101"

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_remote_location(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_bamboohr

        mock_resp = MagicMock()
        mock_resp.json.return_value = self.BAMBOOHR_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_bamboohr("testcompany", "TestCo", KEYWORDS)

        video_producer = [r for r in results if r["title"] == "Video Producer"][0]
        assert video_producer["location"] == "Remote"

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_handles_api_error(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_bamboohr

        mock_get.side_effect = Exception("Connection refused")
        results = scrape_bamboohr("bad", "Bad", KEYWORDS)
        assert results == []

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_handles_empty_result(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_bamboohr

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"meta": {"totalCount": 0}, "result": []}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_bamboohr("empty", "Empty", KEYWORDS)
        assert results == []

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_handles_unexpected_format(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_bamboohr

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"meta": {}, "result": "not a list"}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_bamboohr("weird", "Weird", KEYWORDS)
        assert results == []

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_handles_null_location(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_bamboohr

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "meta": {"totalCount": 1},
            "result": [
                {
                    "id": "200",
                    "jobOpeningName": "Creative Producer",
                    "departmentLabel": "Content",
                    "location": None,
                    "isRemote": None,
                },
            ],
        }
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_bamboohr("nullloc", "NullLoc", KEYWORDS)
        assert len(results) == 1
        assert results[0]["location"] == ""

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_correct_api_url(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_bamboohr

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"meta": {"totalCount": 0}, "result": []}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        scrape_bamboohr("mycompany", "MyCo", KEYWORDS)

        mock_get.assert_called_once_with(
            "https://mycompany.bamboohr.com/careers/list",
            headers={"Accept": "application/json"},
            timeout=15,
        )


# ============================================================
# Comeet
# ============================================================

class TestScrapeComeet:
    COMEET_RESPONSE = [
        {
            "uid": "A1.001",
            "name": "Content Producer",
            "department": "Marketing",
            "employment_type": "Full-time",
            "experience_level": "Intermediate",
            "workplace_type": "Remote",
            "url_comeet_hosted_page": "https://www.comeet.com/jobs/testco/99.001/content-producer/A1.001",
            "url_active_page": "https://careers.testco.com/content-producer",
            "time_updated": "2026-01-20T14:30:00Z",
            "company_name": "TestCo",
            "is_internal": False,
            "location": {
                "name": "New York, NY",
                "country": "US",
                "city": "New York",
                "state": "NY",
                "is_remote": True,
            },
            "details": [
                {"name": "Description", "value": "<p>Lead content production across channels.</p>", "order": 0},
                {"name": "Requirements", "value": "<ul><li>5+ years experience</li></ul>", "order": 1},
            ],
        },
        {
            "uid": "A1.002",
            "name": "Software Engineer",
            "department": "Engineering",
            "employment_type": "Full-time",
            "workplace_type": "On-site",
            "url_comeet_hosted_page": "https://www.comeet.com/jobs/testco/99.001/swe/A1.002",
            "url_active_page": "",
            "time_updated": "2026-01-19T10:00:00Z",
            "company_name": "TestCo",
            "is_internal": False,
            "location": {
                "name": "Tel Aviv",
                "country": "IL",
                "city": "Tel Aviv",
                "state": "",
                "is_remote": False,
            },
            "details": [],
        },
        {
            "uid": "A1.003",
            "name": "Video Producer",
            "department": "Creative",
            "employment_type": "Full-time",
            "workplace_type": "Hybrid",
            "url_comeet_hosted_page": "https://www.comeet.com/jobs/testco/99.001/video-producer/A1.003",
            "url_active_page": "",
            "time_updated": "2026-01-18T08:00:00Z",
            "company_name": "TestCo",
            "is_internal": True,  # Internal — should be skipped
            "location": {
                "name": "London",
                "country": "GB",
                "city": "London",
                "state": "",
                "is_remote": False,
            },
            "details": [
                {"name": "Description", "value": "<p>Video production role.</p>", "order": 0},
            ],
        },
        {
            "uid": "A1.004",
            "name": "Creative Producer",
            "department": "Content",
            "employment_type": "Full-time",
            "workplace_type": "On-site",
            "url_comeet_hosted_page": "https://www.comeet.com/jobs/testco/99.001/creative-producer/A1.004",
            "url_active_page": "https://careers.testco.com/creative-producer",
            "time_updated": "2026-01-17T12:00:00Z",
            "company_name": "TestCo",
            "is_internal": False,
            "location": {
                "name": "Austin, TX",
                "country": "US",
                "city": "Austin",
                "state": "TX",
                "is_remote": False,
            },
            "details": [
                {"name": "Description", "value": "<p>Produce creative campaigns.</p>", "order": 0},
            ],
        },
    ]

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_parses_response(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_comeet

        mock_resp = MagicMock()
        mock_resp.json.return_value = self.COMEET_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_comeet("99.001:TESTTOKEN123", "TestCo", KEYWORDS)

        # Content Producer + Creative Producer (SWE no match, Video Producer is internal)
        assert len(results) == 2
        assert results[0]["title"] == "Content Producer"
        assert results[0]["company"] == "TestCo"
        assert results[0]["site"] == "comeet"
        assert results[0]["department"] == "Marketing"
        # Should prefer url_active_page
        assert results[0]["job_url"] == "https://careers.testco.com/content-producer"

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_location_parsing(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_comeet

        mock_resp = MagicMock()
        mock_resp.json.return_value = self.COMEET_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_comeet("99.001:TESTTOKEN123", "TestCo", KEYWORDS)

        # Content Producer: US location with remote
        assert "New York" in results[0]["location"]
        assert "Remote" in results[0]["location"]

        # Creative Producer: US location, no remote
        assert results[1]["location"] == "Austin, TX"

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_description_extraction(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_comeet

        mock_resp = MagicMock()
        mock_resp.json.return_value = self.COMEET_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_comeet("99.001:TESTTOKEN123", "TestCo", KEYWORDS)

        # Content Producer has two detail sections
        assert "content production" in results[0]["description"].lower()
        assert "5+ years" in results[0]["description"]

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_skips_internal_postings(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_comeet

        mock_resp = MagicMock()
        mock_resp.json.return_value = self.COMEET_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_comeet("99.001:TESTTOKEN123", "TestCo", KEYWORDS)

        titles = [r["title"] for r in results]
        assert "Video Producer" not in titles

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_handles_api_error(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_comeet

        mock_get.side_effect = Exception("Timeout")
        results = scrape_comeet("99.001:BADTOKEN", "Bad", KEYWORDS)
        assert results == []

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_handles_invalid_token_format(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_comeet

        results = scrape_comeet("no_colon_token", "Bad", KEYWORDS)
        assert results == []
        mock_get.assert_not_called()

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_handles_unexpected_format(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_comeet

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "invalid"}  # Not a list
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_comeet("99.001:TOKEN", "TestCo", KEYWORDS)
        assert results == []

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_fallback_url(self, mock_get):
        """When url_active_page is empty, fall back to url_comeet_hosted_page."""
        from job_agent.scraper_bamboohr_comeet import scrape_comeet

        mock_resp = MagicMock()
        # Single position with empty url_active_page
        mock_resp.json.return_value = [
            {
                "uid": "B1.001",
                "name": "Content Producer",
                "department": "Content",
                "url_comeet_hosted_page": "https://www.comeet.com/jobs/co/1/cp/B1.001",
                "url_active_page": "",
                "time_updated": "2026-01-15T10:00:00Z",
                "is_internal": False,
                "location": {"name": "Remote", "city": "", "state": "", "country": "", "is_remote": True},
                "details": [],
            }
        ]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        results = scrape_comeet("1.001:TOKEN", "Co", KEYWORDS)
        assert len(results) == 1
        assert results[0]["job_url"] == "https://www.comeet.com/jobs/co/1/cp/B1.001"

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_correct_api_url(self, mock_get):
        from job_agent.scraper_bamboohr_comeet import scrape_comeet

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        scrape_comeet("76.008:MYTOKEN", "TestCo", KEYWORDS)

        mock_get.assert_called_once_with(
            "https://www.comeet.co/careers-api/2.0/company/76.008/positions",
            params={"token": "MYTOKEN", "details": "true"},
            timeout=60,
        )


# ============================================================
# Dispatcher integration
# ============================================================

class TestDispatcherIntegration:
    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_bamboohr_dispatched(self, mock_get):
        from job_agent.ats_scraper import scrape_ats_boards

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "meta": {"totalCount": 1},
            "result": [
                {
                    "id": "1",
                    "jobOpeningName": "Content Producer",
                    "departmentLabel": "Marketing",
                    "location": {"city": "LA", "state": "CA"},
                    "isRemote": None,
                },
            ],
        }
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        companies = [{"company": "TestCo", "ats": "bamboohr", "token": "testco"}]
        df = scrape_ats_boards(companies, KEYWORDS)

        assert len(df) == 1
        assert df.iloc[0]["site"] == "bamboohr"

    @patch("job_agent.scraper_bamboohr_comeet.requests.get")
    def test_comeet_dispatched(self, mock_get):
        from job_agent.ats_scraper import scrape_ats_boards

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "uid": "A1.001",
                "name": "Video Producer",
                "department": "Content",
                "url_comeet_hosted_page": "https://example.com/job",
                "url_active_page": "",
                "time_updated": "2026-01-20T14:30:00Z",
                "is_internal": False,
                "location": {"name": "Remote", "city": "", "state": "", "country": "", "is_remote": True},
                "details": [],
            },
        ]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        companies = [{"company": "TestCo", "ats": "comeet", "token": "99.001:TOKEN"}]
        df = scrape_ats_boards(companies, KEYWORDS)

        assert len(df) == 1
        assert df.iloc[0]["site"] == "comeet"
