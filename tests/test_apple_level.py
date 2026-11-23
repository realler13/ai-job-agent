"""Apple's internal job level (2026-09-26).

Apple's job data publishes lowJobTitle / highJobTitle ("Machine Learning ICT4"
/ "Machine Learning ICT5"; either may be missing) — read from the jobDetails
JSON, with the detail page's SSR hydration as the fallback.
The level is a direct read on seniority, so it is stored per job, shown in the
digest and the UI, and put on the description's first line for the cascade.
"""
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from job_agent import scraper_eightfold_apple as ap


def _detail_html(**levels):
    """A detail page as Apple serves it: hydration JSON inside JSON.parse("..."),
    so every quote in the payload arrives backslash-escaped."""
    payload = json.dumps({"loaderData": {"jobDetails": {"jobsData": {
        "postingTitle": "Producer", **levels, "teamNames": ["Marketing"]}}}})
    escaped = json.dumps(payload)[1:-1]
    return f'<html><script>window.__staticRouterHydrationData = JSON.parse("{escaped}");</script></html>'


class TestExtract:
    def test_range(self):
        html = _detail_html(highJobTitle="Machine Learning ICT5", lowJobTitle="Machine Learning ICT4")
        assert ap.extract_apple_levels(html) == ("Machine Learning ICT4", "Machine Learning ICT5")

    def test_high_only(self):
        """A posting that publishes highJobTitle only."""
        html = _detail_html(highJobTitle="Engineering Project/Program ICT3")
        assert ap.extract_apple_levels(html) == ("", "Engineering Project/Program ICT3")

    def test_absent(self):
        assert ap.extract_apple_levels(_detail_html()) == ("", "")
        assert ap.extract_apple_levels("") == ("", "")


class TestFormat:
    @pytest.mark.parametrize("low,high,expected", [
        ("Machine Learning ICT4", "Machine Learning ICT5", "Machine Learning ICT4–ICT5"),
        ("Marketing Program Mgmt IC4", "Marketing Program Mgmt IC5", "Marketing Program Mgmt IC4–IC5"),
        ("", "Engineering Project/Program ICT3", "Engineering Project/Program ICT3"),
        ("Data Scientist ICT2", "", "Data Scientist ICT2"),
        ("Producer IC3", "Producer IC3", "Producer IC3"),
        ("Producer IC3", "Program Manager IC4", "Producer IC3 – Program Manager IC4"),
        ("", "", ""),
    ])
    def test_display(self, low, high, expected):
        assert ap.format_apple_level(low, high) == expected


def _fresh_date():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _resp(text="", payload=None):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = text
    resp.json.return_value = payload if payload is not None else {}
    return resp


# The JSON the job page is built from (jobs.apple.com/api/v1/jobDetails/<id>),
# in the real answer's shape, with made-up values.
_DETAIL_JSON = {"res": {
    "postingTitle": "Video Producer, Example Team",
    "jobSummary": "We are seeking an enthusiastic Video Producer.",
    "highJobTitle": "Video Producer ICT3",
    "selectedLocation": {"id": "postLocation-BBB"},
    "postingFooters": [
        {"postLocationId": "postLocation-AAA", "localizations": {"en_US": [
            {"displayOrder": 1, "name": "Pay & Benefits",
             "content": "The base pay range for this role is between $120,000 and $180,000.<br><br>Benefits."},
            {"displayOrder": 2, "name": "EEO Statement", "content": "<p>Apple is an equal opportunity employer.</p>"},
        ]}},
        {"postLocationId": "postLocation-BBB", "localizations": {"en_US": [
            {"displayOrder": 1, "name": "Pay & Benefits",
             "content": "The base pay range for this role is between $110,000 and $170,000.<br><br>Benefits."},
        ]}},
    ],
}}


class TestScrape:
    @pytest.fixture(autouse=True)
    def _no_sleep(self):
        with patch("job_agent.scraper_eightfold_apple.time.sleep"):
            yield

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_level_reaches_the_job_dict_and_the_description(self, mock_get):
        search = {"loaderData": {"search": {"searchResults": [{
            "id": "1", "positionId": "200600001", "postingTitle": "Video Producer",
            "jobSummary": "Produce video.", "locations": [{"city": "Austin"}],
            "postDateInGMT": _fresh_date()}]}}}
        search_html = ('<html><script>window.__staticRouterHydrationData = JSON.parse("'
                       + json.dumps(json.dumps(search))[1:-1] + '");</script></html>')
        detail = {"res": {"postingTitle": "Video Producer", "highJobTitle": "Producer ICT3"}}

        def get(url, **kw):
            if "/api/v1/jobDetails/200600001" in url:
                return _resp(payload=detail)
            return _resp(search_html if kw.get("params", {}).get("page") == 1 else "<html></html>")
        mock_get.side_effect = get

        jobs = ap.scrape_apple("en-us", "Apple", ["producer"])
        assert len(jobs) == 1
        job = jobs[0]
        assert job["job_level"] == "Producer ICT3"
        assert job["description"].startswith("Apple job level: Producer ICT3 (Apple IC ladder")
        assert "Produce video." in job["description"]

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_json_gives_level_and_the_pay_of_the_page_s_location(self, mock_get):
        mock_get.return_value = _resp(payload=_DETAIL_JSON)
        fields = ap._fetch_apple_detail_fields("200600002")
        assert fields["level"] == "Video Producer ICT3"
        # The page opens on its selected location (the second footer), so that pay.
        assert fields["pay"] == ("Pay & Benefits: The base pay range for this role is "
                                 "between $110,000 and $170,000. Benefits.")
        assert mock_get.call_count == 1

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_json_range(self, mock_get):
        mock_get.return_value = _resp(payload={"res": {
            "postingTitle": "Managing Producer", "lowJobTitle": "Creative Production Mgmt MGR2",
            "highJobTitle": "Creative Production Mgmt MGR3"}})
        assert ap._fetch_apple_detail_fields("200600003")["level"] == "Creative Production Mgmt MGR2–MGR3"

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_page_is_read_when_the_json_is_gone(self, mock_get):
        """If Apple retires the JSON, the level still comes from the job page."""
        import requests as _requests

        def get(url, **kw):
            if "/api/v1/jobDetails/" in url:
                resp = _resp()
                resp.raise_for_status.side_effect = _requests.HTTPError(response=MagicMock(status_code=404))
                return resp
            return _resp(_detail_html(highJobTitle="Producer ICT3"))
        mock_get.side_effect = get

        assert ap._fetch_apple_detail_fields("200600001")["level"] == "Producer ICT3"
        assert mock_get.call_args.args[0] == "https://jobs.apple.com/en-us/details/200600001"

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_page_is_not_read_after_a_timeout(self, mock_get):
        """A slow Apple gets no second, heavier request for the same posting."""
        import requests as _requests

        mock_get.side_effect = _requests.ReadTimeout("read timed out")
        assert ap._fetch_apple_detail_fields("200600001") == {}
        assert all("/api/v1/jobDetails/" in c.args[0] for c in mock_get.call_args_list)

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_detail_fetch_failure_leaves_level_empty(self, mock_get):
        mock_get.side_effect = Exception("boom")
        assert ap._fetch_apple_detail_fields("1") == {}

    # Nested values of a type the parser does not expect. Each one raised
    # AttributeError out of scrape_apple and ended the whole ATS phase. The
    # posting loses only its pay: the level is worked out first.
    @pytest.mark.parametrize("odd", [
        {"postingFooters": [None]},
        {"postingFooters": "Pay & Benefits"},
        {"postingFooters": [{"postLocationId": "x", "localizations": [{"content": "$1"}]}]},
        {"postingFooters": [{"postLocationId": "x", "localizations": {"en_US": [None]}}]},
        {"selectedLocation": "postLocation-BBB", "postingFooters": [{"postLocationId": "x"}]},
    ])
    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_odd_nested_pay_data_keeps_the_level_and_does_not_raise(self, mock_get, odd):
        mock_get.return_value = _resp(payload={"res": {
            "postingTitle": "Producer", "highJobTitle": "Producer IC3", **odd}})
        assert ap._fetch_apple_detail_fields("200600001") == {"pay": "", "level": "Producer IC3"}
        # The JSON answered, so no second, heavier request for the page.
        assert mock_get.call_count == 1

    @pytest.mark.parametrize("low,high", [(3, ["Producer IC3"]), ({"t": "x"}, None)])
    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_level_titles_that_are_not_text_are_ignored(self, mock_get, low, high):
        mock_get.return_value = _resp(payload={"res": {
            "postingTitle": "Producer", "lowJobTitle": low, "highJobTitle": high}})
        assert ap._fetch_apple_detail_fields("200600001")["level"] == ""

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_odd_job_data_does_not_end_the_scrape(self, mock_get):
        search = {"loaderData": {"search": {"searchResults": [{
            "id": "1", "positionId": "200600001", "postingTitle": "Video Producer",
            "jobSummary": "Produce video.", "locations": [{"city": "Austin"}],
            "postDateInGMT": _fresh_date()}]}}}
        search_html = ('<html><script>window.__staticRouterHydrationData = JSON.parse("'
                       + json.dumps(json.dumps(search))[1:-1] + '");</script></html>')

        def get(url, **kw):
            if "/api/v1/jobDetails/" in url:
                return _resp(payload={"res": {"postingTitle": "Video Producer",
                                              "postingFooters": [None]}})
            return _resp(search_html if kw.get("params", {}).get("page") == 1 else "<html></html>")
        mock_get.side_effect = get

        jobs = ap.scrape_apple("en-us", "Apple", ["producer"])
        assert [(j["title"], j["job_level"]) for j in jobs] == [("Video Producer", "")]


class _ScrapeTimeout(Exception):
    """Stands in for main._ScrapeTimeout (matched by name, as in ats_scraper)."""


class TestPhaseTimeout:
    """main.py's SIGALRM phase budget raises _ScrapeTimeout wherever the
    scraper happens to be; it must end the ATS phase, not be swallowed."""

    @pytest.fixture(autouse=True)
    def _no_sleep(self):
        with patch("job_agent.scraper_eightfold_apple.time.sleep"):
            yield

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_raised_during_a_request(self, mock_get):
        mock_get.side_effect = _ScrapeTimeout("ATS timed out")
        with pytest.raises(_ScrapeTimeout):
            ap.scrape_apple("en-us", "Apple", ["producer"])
        assert mock_get.call_count == 1

    @patch("job_agent.scraper_eightfold_apple._apple_pay_from_json")
    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_raised_while_reading_the_pay(self, mock_get, mock_pay):
        mock_get.return_value = _resp(payload={"res": {
            "postingTitle": "Producer", "highJobTitle": "Producer IC3"}})
        mock_pay.side_effect = _ScrapeTimeout("ATS timed out")
        with pytest.raises(_ScrapeTimeout):
            ap._fetch_apple_detail_fields("200600001")

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_raised_while_reading_the_job_data(self, mock_get):
        resp = _resp()
        resp.json.side_effect = _ScrapeTimeout("ATS timed out")
        mock_get.return_value = resp
        with pytest.raises(_ScrapeTimeout):
            ap._fetch_apple_detail_fields("200600001")


class TestStorage:
    def test_saved_and_returned(self, tmp_path):
        from job_agent.db_manager import DBManager
        db = DBManager(db_path=str(tmp_path / "lvl.db"))
        base = {"title": "Producer", "company": "Apple", "location": "Austin, TX", "description": "d"}
        a = db.save_job({**base, "job_url": "https://jobs.apple.com/1", "job_level": "Producer ICT3"})
        b = db.save_job({**base, "title": "Editor", "job_url": "https://x.example/2", "job_level": float("nan")})
        conn = db._get_thread_conn()
        levels = dict(conn.execute("SELECT id, job_level FROM scrapes").fetchall())
        assert levels[a] == "Producer ICT3"
        assert levels[b] is None          # DataFrame NaN for non-Apple rows is not stored as "nan"


class TestDigest:
    def _send(self, monkeypatch, level):
        from job_agent.postprocess import send_email_with_results
        captured = {}

        class FakeSMTP:
            def login(self, *a): pass
            def sendmail(self, sender, recips, msg_str): captured["msg"] = msg_str
            def quit(self): pass

        monkeypatch.setenv("EMAIL_SENDER", "sender@example.invalid")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret")
        monkeypatch.setattr("job_agent.postprocess.smtplib.SMTP_SSL", lambda *a, **k: FakeSMTP())
        df = pd.DataFrame([{
            "title": "Video Producer", "company": "Apple", "location": "Austin, TX",
            "job_url": "https://jobs.apple.com/1", "confidence": 0.9, "match_level": "HIGH_MATCH",
            "tier": "strong", "gpt_relevance": "relevant", "gpt_reasoning": "fit", "job_level": level,
        }])
        send_email_with_results(None, "recipient@example.invalid", df, db=None)
        return captured["msg"]

    def test_level_shown_on_the_card(self, monkeypatch):
        assert "Level: Machine Learning ICT3" in self._send(monkeypatch, "Machine Learning ICT3")

    def test_no_chip_without_a_level(self, monkeypatch):
        assert "Level:" not in self._send(monkeypatch, float("nan"))


class TestTargetBand:
    """The band comes from the profile (search.apple_target_band)."""

    def test_band_is_read_from_the_pinned_profile(self):
        from job_agent import cascade_filter as cf
        from tests.fixtures.persona import APPLE_TARGET_BAND
        low, high = APPLE_TARGET_BAND
        assert (cf.APPLE_TARGET_LOW, cf.APPLE_TARGET_HIGH) == (low, high)
        assert cf.APPLE_TARGET_BAND == f"IC{low}\u2013IC{high}"

    def _profile(self, tmp_path, band):
        from job_agent.candidate_profile import load_profile
        from tests.fixtures.persona import raw_profile
        data = raw_profile()
        if band is None:
            del data["search"]["apple_target_band"]
        else:
            data["search"]["apple_target_band"] = band
        path = tmp_path / "p.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return load_profile(path)

    def test_missing_band_means_no_band(self, tmp_path):
        """There is no default band: a profile without the key gets None."""
        from job_agent import candidate_profile
        assert candidate_profile.apple_target_band(self._profile(tmp_path, None)) is None
        assert not hasattr(candidate_profile, "DEFAULT_APPLE_TARGET_BAND")

    def test_profile_without_a_band_turns_the_rule_off(self, tmp_path, monkeypatch):
        """No band in the profile: no level gets a verdict (the same result as
        a posting with no level), and the judges' line carries the level and
        the ladder key with no band clause."""
        from job_agent import cascade_filter as cf
        constants = cf._band_constants(self._profile(tmp_path, None))
        assert constants == (None, None, None)
        for name, value in zip(("APPLE_TARGET_LOW", "APPLE_TARGET_HIGH", "APPLE_TARGET_BAND"), constants):
            monkeypatch.setattr(cf, name, value)
        for level in ("Producer IC1", "Producer IC2", "Producer ICT3", "Producer IC4", "Producer IC5",
                      "Producer IC7", "Producer IC4\u2013IC5", "Content Production MGR3"):
            assert cf.apple_level_verdict(level) is None, level
        line = ap.apple_level_line("Producer IC5")
        assert line == ("Apple job level: Producer IC5 (Apple IC ladder: 2 entry, 3 mid, 4 senior, "
                        "5 staff, 6 principal; MGR is the people-manager ladder)")
        assert "band" not in line

    def test_profile_band_wins(self, tmp_path):
        from job_agent import cascade_filter as cf
        from job_agent.candidate_profile import apple_target_band
        profile = self._profile(tmp_path, {"low": 4, "high": 5})
        assert apple_target_band(profile) == (4, 5)
        assert cf._band_constants(profile) == (4, 5, "IC4\u2013IC5")

    @pytest.mark.parametrize("band", [{"low": 3}, {"low": "three", "high": 4}, "IC3-IC4",
                                      {"low": 4, "high": 3}, {"low": 0, "high": 2}])
    def test_malformed_band_fails_loudly(self, tmp_path, band):
        from job_agent import cascade_filter as cf
        from job_agent.candidate_profile import apple_target_band
        profile = self._profile(tmp_path, band)
        with pytest.raises(ValueError, match="apple_target_band"):
            apple_target_band(profile)
        with pytest.raises(ValueError, match="apple_target_band"):
            cf._band_constants(profile)

    @pytest.mark.parametrize("level,action", [
        ("Producer IC3", "reject_below"),
        ("Producer IC3\u2013IC4", "in_band"),
        ("Producer IC4", "in_band"),
        ("Producer ICT5", "in_band"),
        ("Producer IC6", "cap"),
        ("Producer IC7", "reject_above"),
        ("Producer MGR4", "reject_manager"),
    ])
    def test_another_band_moves_every_edge(self, monkeypatch, level, action):
        """A senior band (IC4-IC5): the same rules, one level up."""
        from job_agent import cascade_filter as cf
        monkeypatch.setattr(cf, "APPLE_TARGET_LOW", 4)
        monkeypatch.setattr(cf, "APPLE_TARGET_HIGH", 5)
        got, note = cf.apple_level_verdict(level)
        assert got == action
        if action != "in_band":
            assert note.endswith("target band IC4\u2013IC5]")

    def test_level_line_follows_another_band(self, tmp_path, monkeypatch):
        """The judges' line moves with the band; a band that starts at IC1
        has no level below it."""
        from job_agent import cascade_filter as cf
        constants = cf._band_constants(self._profile(tmp_path, {"low": 1, "high": 1}))
        assert constants == (1, 1, "IC1")
        for name, value in zip(("APPLE_TARGET_LOW", "APPLE_TARGET_HIGH", "APPLE_TARGET_BAND"), constants):
            monkeypatch.setattr(cf, name, value)
        assert ap.apple_level_line("Producer IC2") == (
            "Apple job level: Producer IC2 (Apple IC ladder: 2 entry, 3 mid, 4 senior, 5 staff, "
            "6 principal; the candidate's target band is IC1 / ICT1 \u2014 IC2 is above it, "
            "the MGR people-manager ladder are outside it)")

    def test_single_level_band(self, monkeypatch):
        from job_agent import cascade_filter as cf
        monkeypatch.setattr(cf, "APPLE_TARGET_LOW", 3)
        monkeypatch.setattr(cf, "APPLE_TARGET_HIGH", 3)
        assert cf.apple_level_verdict("Producer IC3")[0] == "in_band"
        assert cf.apple_level_verdict("Producer IC4") == (
            "cap", "[Apple level IC4 \u2014 above the candidate's target band IC3]")
        assert cf.apple_level_verdict("Producer IC2")[0] == "reject_below"
