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
             "location": "Seattle, WA", "job_url": "http://lv/1", "description": "",
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


class TestPerBoardGuard:
    @patch("job_agent.ats_scraper.scrape_lever")
    @patch("job_agent.ats_scraper.scrape_greenhouse")
    def test_one_failing_board_does_not_stop_later_boards(self, mock_gh, mock_lv):
        mock_gh.side_effect = TypeError("expected string or bytes-like object, got 'NoneType'")
        mock_lv.return_value = [
            {"title": "Producer", "company": "LVCo", "site": "lever",
             "location": "Seattle, WA", "job_url": "http://lv/1", "description": "",
             "date_posted": "", "department": ""},
        ]
        companies = [
            {"company": "GHCo", "ats": "greenhouse", "token": "ghco"},
            {"company": "LVCo", "ats": "lever", "token": "lvco"},
        ]
        df = scrape_ats_boards(companies, ["Producer"])
        assert list(df["company"]) == ["LVCo"]

    @patch("job_agent.ats_scraper.scrape_lever")
    @patch("job_agent.ats_scraper.scrape_greenhouse")
    def test_the_phase_timeout_still_ends_the_scan(self, mock_gh, mock_lv):
        from job_agent import main
        mock_gh.side_effect = main._ScrapeTimeout("ATS phase timed out after 900s")
        companies = [
            {"company": "GHCo", "ats": "greenhouse", "token": "ghco"},
            {"company": "LVCo", "ats": "lever", "token": "lvco"},
        ]
        with pytest.raises(main._ScrapeTimeout):
            scrape_ats_boards(companies, ["Producer"])
        mock_lv.assert_not_called()


class TestDeadline:
    """A deadline stops the scan between boards and keeps what it found; a
    SIGALRM at the budget would throw it all away."""

    @staticmethod
    def _row(company):
        return [{"title": "Producer", "company": company, "site": "greenhouse",
                 "location": "Seattle, WA", "job_url": f"http://gh/{company}", "description": "",
                 "date_posted": "", "department": ""}]

    def _scan(self, deadline, board_seconds, caplog):
        clock = {"t": 1000.0}
        done = []

        def board(token, company, keywords):
            clock["t"] += board_seconds
            done.append(company)
            return self._row(company)

        companies = [{"company": f"Co{i}", "ats": "greenhouse", "token": f"co{i}"} for i in range(5)]
        with patch("job_agent.ats_scraper.scrape_greenhouse", side_effect=board), \
                patch("job_agent.ats_scraper.time.monotonic", side_effect=lambda: clock["t"]), \
                caplog.at_level("WARNING", logger="job_agent.ats_scraper"):
            df = scrape_ats_boards(companies, ["Producer"],
                                   deadline=None if deadline is None else 1000.0 + deadline)
        return df, done

    def test_stops_starting_boards_when_one_workday_board_no_longer_fits(self, caplog):
        from job_agent.scraper_workday import _BOARD_TIME_BUDGET
        # 100 s a board, 400 s to the deadline: boards start at 0, 100, 200 s
        # (400, 300, 200 s left); at 300 s only 100 s are left, under 150 + 20.
        assert _BOARD_TIME_BUDGET + 20 == 170
        df, done = self._scan(400, 100, caplog)
        assert done == ["Co0", "Co1", "Co2"]
        assert list(df["company"]) == ["Co0", "Co1", "Co2"]  # kept, not discarded
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1 and "2 of 5 boards skipped" in warnings[0].getMessage()

    def test_a_deadline_already_too_close_scans_nothing(self, caplog):
        df, done = self._scan(100, 10, caplog)
        assert done == [] and df.empty
        assert "5 of 5 boards skipped" in caplog.text

    def test_without_a_deadline_every_board_runs(self, caplog):
        df, done = self._scan(None, 10_000, caplog)
        assert len(done) == 5
        assert "skipped" not in caplog.text


class TestAtsPhaseSplit:
    """main runs non-Workday and Workday boards as two timed calls, so a
    Workday overrun keeps the other boards' results."""

    def test_each_call_gets_a_deadline_30s_inside_its_budget(self, monkeypatch):
        from job_agent import config, main

        companies = [
            {"company": "GHCo", "ats": "greenhouse", "token": "ghco"},
            {"company": "WDCo", "ats": "workday", "token": "https://x.wd5.myworkdayjobs.com/X"},
        ]
        monkeypatch.setattr(config, "ATS_COMPANIES", companies)
        monkeypatch.setattr(main.time, "monotonic", lambda: 5000.0)
        calls = []

        def fake_timeout(func, args=(), kwargs=None, timeout_sec=120, label="", **kw):
            calls.append((timeout_sec, kwargs["deadline"]))
            return pd.DataFrame()

        monkeypatch.setattr(main, "_run_with_timeout", fake_timeout)
        monkeypatch.setattr(main, "load_profile_text", lambda p: "profile")
        monkeypatch.setattr(main, "load_proxies", lambda: None)
        monkeypatch.setattr(main, "REMOTE_SECOND_PASS", False)
        monkeypatch.setattr(main, "SECONDARY_SECOND_PASS", False)
        monkeypatch.setattr(main, "_parallel_jobspy_pass",
                            lambda *a, **kw: pytest.fail("no jobspy pass in this test"))

        class _Stop(Exception):
            pass

        def stop(*a, **kw):
            raise _Stop

        # The phase is what's under test; stop the session right after it.
        monkeypatch.setattr(main, "_refresh_listing_status", stop)
        with pytest.raises(_Stop):
            main.run_search_session("r.pdf", "cv.docx", "m", "g", site_list=["ats_boards"],
                                    send_email=False, session=main.SessionManager(), db=MagicMock())
        assert calls == [(900, 5000.0 + 900 - 30), (1500, 5000.0 + 1500 - 30)]

    def test_workday_timeout_keeps_the_other_boards_results(self, monkeypatch):
        from job_agent import config, main

        class _Saved(Exception):
            pass

        companies = [
            {"company": "GHCo", "ats": "greenhouse", "token": "ghco"},
            {"company": "WDCo", "ats": "workday", "token": "https://x.wd5.myworkdayjobs.com/X"},
            {"company": "LVCo", "ats": "lever", "token": "lvco"},
        ]
        monkeypatch.setattr(config, "ATS_COMPANIES", companies)
        calls = []

        def fake_timeout(func, args=(), kwargs=None, timeout_sec=120, label="", **kw):
            calls.append((label, [c["company"] for c in args[0]], timeout_sec, kwargs))
            if label.endswith("(Workday boards)"):
                raise main._ScrapeTimeout(f"{label} timed out after {timeout_sec}s")
            return pd.DataFrame([{"title": "Producer", "company": "GHCo", "site": "greenhouse",
                                  "location": "Seattle, WA", "job_url": "http://gh/1", "description": "",
                                  "date_posted": "", "department": ""}])

        class DB:
            saved = []

            def save_job(self, rec, session_id=None):
                self.saved.append(rec)
                raise _Saved  # stop the session here; the phase is what's under test

        monkeypatch.setattr(main, "_run_with_timeout", fake_timeout)
        monkeypatch.setattr(main, "load_profile_text", lambda p: "profile")
        monkeypatch.setattr(main, "load_proxies", lambda: None)
        monkeypatch.setattr(main, "REMOTE_SECOND_PASS", False)
        monkeypatch.setattr(main, "SECONDARY_SECOND_PASS", False)
        monkeypatch.setattr(main, "_parallel_jobspy_pass",
                            lambda *a, **kw: pytest.fail("no jobspy pass in this test"))
        db = DB()
        session = main.SessionManager()
        with pytest.raises(_Saved):
            main.run_search_session("r.pdf", "cv.docx", "m", "g", site_list=["ats_boards"],
                                    send_email=False, session=session, db=db)

        assert [c[1] for c in calls] == [["GHCo", "LVCo"], ["WDCo"]]
        assert sum(c[2] for c in calls) == 2400
        assert all({k: v for k, v in c[3].items() if k != "deadline"} == {"db": db, "run_id": session.id}
                   for c in calls)
        assert db.saved[0]["job_url"] == "http://gh/1"
