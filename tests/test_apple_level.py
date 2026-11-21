"""Apple's internal job level (2026-09-26).

Apple detail pages publish lowJobTitle / highJobTitle in their SSR hydration
("Machine Learning ICT4" / "Machine Learning ICT5"; either may be missing).
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


class TestScrape:
    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_level_reaches_the_job_dict_and_the_description(self, mock_get):
        search = {"loaderData": {"search": {"searchResults": [{
            "id": "1", "positionId": "200600001", "postingTitle": "Video Producer",
            "jobSummary": "Produce video.", "locations": [{"city": "Austin"}],
            "postDateInGMT": "2026-09-20"}]}}}
        search_html = ('<html><script>window.__staticRouterHydrationData = JSON.parse("'
                       + json.dumps(json.dumps(search))[1:-1] + '");</script></html>')
        detail_html = _detail_html(highJobTitle="Producer ICT3")

        def get(url, **kw):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.text = detail_html if "/details/" in url else (
                search_html if kw.get("params", {}).get("page") == 1 else "<html></html>")
            return resp
        mock_get.side_effect = get

        with patch("time.sleep"):
            jobs = ap.scrape_apple("en-us", "Apple", ["producer"])
        assert len(jobs) == 1
        job = jobs[0]
        assert job["job_level"] == "Producer ICT3"
        assert job["description"].startswith("Apple job level: Producer ICT3 (Apple IC ladder")
        assert "Produce video." in job["description"]

    @patch("job_agent.scraper_eightfold_apple.requests.get")
    def test_detail_fetch_failure_leaves_level_empty(self, mock_get):
        mock_get.side_effect = Exception("boom")
        assert ap._fetch_apple_detail_fields("https://jobs.apple.com/en-us/details/1") == {}


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
