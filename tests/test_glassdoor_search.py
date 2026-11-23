"""Tests for the Glassdoor crawl4ai search scraper and posting-page enrichment."""
import asyncio
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

import job_agent.glassdoor_crawler as gc
from tests.fixtures.persona import HOME


def _fake_run(returns=None, raises=None):
    """Build a fake asyncio.run that closes the passed coroutine (no warning)."""
    def run(coro, *a, **k):
        if hasattr(coro, "close"):
            coro.close()
        if raises:
            raise raises
        return returns
    return run


def test_returns_none_when_crawl4ai_unavailable(monkeypatch):
    monkeypatch.setattr(gc, "CRAWL4AI_AVAILABLE", False)
    assert gc.scrape_glassdoor("content producer") is None


def test_wraps_jobs_in_dataframe(monkeypatch):
    monkeypatch.setattr(gc, "CRAWL4AI_AVAILABLE", True)
    jobs = [
        {"site": "glassdoor", "title": "Multimedia Producer", "company": "Boston University",
         "location": "Boston, MA", "job_url": "https://www.glassdoor.com/job-listing/x",
         "description": "", "date_posted": None},
    ]
    # _scrape_glassdoor_search is async; patch asyncio.run to return the fixture.
    monkeypatch.setattr(gc.asyncio, "run", _fake_run(returns=jobs))
    df = gc.scrape_glassdoor("content producer", HOME)
    assert df is not None
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Multimedia Producer"
    assert df.iloc[0]["site"] == "glassdoor"


def test_empty_results_returns_empty_df(monkeypatch):
    # Rendered fine with 0 cards = legitimate empty -> empty DataFrame
    # (resets the circuit breaker), NOT None.
    monkeypatch.setattr(gc, "CRAWL4AI_AVAILABLE", True)
    monkeypatch.setattr(gc.asyncio, "run", _fake_run(returns=[]))
    df = gc.scrape_glassdoor("content producer")
    assert df is not None and df.empty


def test_crawl_exception_returns_none(monkeypatch):
    monkeypatch.setattr(gc, "CRAWL4AI_AVAILABLE", True)

    monkeypatch.setattr(gc.asyncio, "run", _fake_run(raises=RuntimeError("browser error")))
    assert gc.scrape_glassdoor("content producer") is None


# --- enrich_jobs_with_crawl4ai: posting-page circuit breaker ----------------

BLOCKED = "Blocked by anti-bot protection: Cloudflare JS challenge"


def _result(ok_text=None, error=""):
    if ok_text:
        return SimpleNamespace(success=True, markdown=SimpleNamespace(raw_markdown=ok_text), error_message="")
    return SimpleNamespace(success=False, markdown=None, error_message=error)


def _fake_crawler(outcomes, calls):
    """AsyncWebCrawler stand-in: arun returns outcomes[url] (default: blocked)."""
    class FakeCrawler:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def arun(self, url, config=None):
            calls.append(url)
            return outcomes.get(url, _result(error=BLOCKED))
    return FakeCrawler


def _jobs(n):
    return [{"title": f"Producer {i}", "job_url": f"https://www.glassdoor.com/job-listing/p{i}",
             "description": ""} for i in range(n)]


@pytest.fixture
def breaker(monkeypatch):
    """Closed breaker, crawl4ai present, no real browser."""
    monkeypatch.setattr(gc, "CRAWL4AI_AVAILABLE", True)
    monkeypatch.setattr(gc, "_breaker_open_until", 0.0)
    monkeypatch.setattr(gc, "BrowserConfig", lambda **k: None, raising=False)
    monkeypatch.setattr(gc, "CrawlerRunConfig", lambda **k: None, raising=False)
    monkeypatch.setattr(gc, "CacheMode", SimpleNamespace(BYPASS="bypass"), raising=False)
    calls = []

    def install(outcomes=None):
        monkeypatch.setattr(gc, "AsyncWebCrawler", _fake_crawler(outcomes or {}, calls), raising=False)
        return calls
    return install


def test_all_blocked_stops_after_three_with_one_warning(breaker, caplog):
    calls = breaker()
    jobs = _jobs(20)
    with caplog.at_level(logging.WARNING, logger=gc.logger.name):
        out = asyncio.run(gc.enrich_jobs_with_crawl4ai(jobs))
    assert len(calls) == gc.BLOCKED_PAGES_TO_TRIP == 3
    # Rows come back unchanged in number and shape: list data only.
    assert out is jobs and len(out) == 20
    assert all(j["description"] == "" and "description_source" not in j for j in out)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and "in a row blocked" in warnings[0]


def test_open_breaker_skips_later_batches(breaker):
    calls = breaker()
    asyncio.run(gc.enrich_jobs_with_crawl4ai(_jobs(20)))
    assert len(calls) == 3
    out = asyncio.run(gc.enrich_jobs_with_crawl4ai(_jobs(20)))
    assert len(calls) == 3  # no page tried for the second keyword's batch
    assert len(out) == 20


def test_breaker_closes_after_hold(breaker, monkeypatch):
    calls = breaker()
    now = [1000.0]
    monkeypatch.setattr(gc.time, "monotonic", lambda: now[0])
    asyncio.run(gc.enrich_jobs_with_crawl4ai(_jobs(5)))
    assert len(calls) == 3
    now[0] += gc.BREAKER_HOLD_SECONDS + 1  # next run, or a search from the app later
    asyncio.run(gc.enrich_jobs_with_crawl4ai(_jobs(5)))
    assert len(calls) == 6


def test_success_resets_the_count_and_fills_text(breaker):
    jobs = _jobs(6)
    calls = breaker({jobs[2]["job_url"]: _result(ok_text="Full posting text"),
                     jobs[5]["job_url"]: _result(ok_text="Another posting")})
    asyncio.run(gc.enrich_jobs_with_crawl4ai(jobs))
    # blocked, blocked, ok, blocked, blocked, ok: never three in a row.
    assert len(calls) == 6
    assert jobs[2]["description"] == "Full posting text"
    assert jobs[2]["description_source"] == "crawl4ai_enriched"
    assert jobs[5]["description"] == "Another posting"
    assert gc._breaker_open_until == 0.0


def test_other_failures_do_not_trip(breaker):
    jobs = _jobs(5)
    calls = breaker({j["job_url"]: _result(error="net::ERR_CONNECTION_RESET") for j in jobs})
    asyncio.run(gc.enrich_jobs_with_crawl4ai(jobs))
    assert len(calls) == 5
    assert gc._breaker_open_until == 0.0


# --- search URL: Glassdoor needs its city id --------------------------------
# locKeyword alone gives the nationwide results ("... jobs in United States");
# only Glassdoor's own location id (locT/locId) narrows a search. The id comes
# from the profile (search.glassdoor_city_id); the fixture profile has none,
# so these tests set a made-up one.

FAKE_CITY = ("C", 9999999)   # a placeholder, not a real Glassdoor id


@pytest.fixture
def home_city_id(monkeypatch):
    """GLASSDOOR_LOCATION_IDS as built from a profile that names a city id."""
    monkeypatch.setattr(gc, "GLASSDOOR_LOCATION_IDS", {HOME.strip().lower(): FAKE_CITY})


def test_home_search_uses_the_city_id(home_city_id):
    url = gc._search_url("Video Producer", HOME)
    assert url.endswith("=Video+Producer&locT=C&locId=9999999")
    assert "locKeyword" not in url


def test_city_lookup_ignores_case_and_spaces(home_city_id):
    assert gc._search_url("Video Producer", f"  {HOME.upper()} ").endswith("&locT=C&locId=9999999")


def test_unknown_city_keeps_keyword_search_and_warns(home_city_id, caplog):
    with caplog.at_level(logging.WARNING, logger=gc.logger.name):
        url = gc._search_url("Video Producer", "Boise, ID")
    assert url.endswith("&locKeyword=Boise%2C+ID")
    assert "no location id" in caplog.text


def test_search_renders_the_city_id_url(home_city_id, monkeypatch):
    # The browser is asked for the city-id URL (not the nationwide one).
    seen = []

    class _Crawler:
        def __init__(self, config=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def arun(self, url, config=None):
            seen.append(url)
            card = ('<li data-test="jobListing"><a data-test="job-title" href="/job-listing/a">Video Producer</a>'
                    '<div data-test="emp-location">Bellevue, WA</div></li>')
            return SimpleNamespace(html="<html><ul>" + card + "</ul>" + "<div>x</div>" * 12000 + "</html>")

    monkeypatch.setattr(gc, "AsyncWebCrawler", _Crawler, raising=False)
    monkeypatch.setattr(gc, "BrowserConfig", lambda **k: None, raising=False)
    monkeypatch.setattr(gc, "CrawlerRunConfig", lambda **k: None, raising=False)
    monkeypatch.setattr(gc, "CacheMode", SimpleNamespace(BYPASS=None), raising=False)

    jobs = asyncio.run(gc._scrape_glassdoor_search("Video Producer", HOME, 20))
    assert seen == [gc._search_url("Video Producer", HOME)]
    assert [j["location"] for j in jobs] == ["Bellevue, WA"]


def test_no_location_is_the_nationwide_search_without_a_warning(caplog):
    with caplog.at_level("WARNING"):
        for loc in (None, "", "  "):
            assert gc._search_url("Video Producer", loc) == \
                "https://www.glassdoor.com/Job/jobs.htm?sc.keyword=Video+Producer"
    assert not caplog.records


# --- which lanes a keyword is searched in -----------------------------------

def test_without_a_city_id_the_search_is_nationwide_only(caplog):
    # The fixture profile has no search.glassdoor_city_id.
    assert gc.GLASSDOOR_LOCATION_IDS == {}
    with caplog.at_level(logging.WARNING, logger=gc.logger.name):
        assert gc.search_locations() == [None]
    assert not caplog.records


def test_with_a_city_id_the_home_city_then_nationwide(home_city_id):
    assert gc.search_locations() == [HOME, None]


def test_scrape_glassdoor_defaults_to_nationwide(monkeypatch):
    monkeypatch.setattr(gc, "CRAWL4AI_AVAILABLE", True)
    seen = []

    async def fake_search(keyword, location, results_wanted):
        seen.append(location)
        return []

    monkeypatch.setattr(gc, "_scrape_glassdoor_search", fake_search)
    gc.scrape_glassdoor("video producer")
    assert seen == [None]


def test_pipeline_searches_each_keyword_in_every_lane():
    """main.py runs each Glassdoor keyword in the lanes search_locations()
    gives: the home city when the profile has its id, then nationwide."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "job_agent" / "main.py").read_text()
    assert '_gd_searches = [(kw, loc) for kw in BROWSER_SCRAPE_KEYWORDS' in src
    assert 'for loc in _gd_locations()]' in src
    assert 'gdf = _scrape_gd(kw, gd_loc, results_wanted=results_wanted_per_site)' in src
    ast.parse(src)


# --- the profile key ----------------------------------------------------------

def _profile_with(tmp_path, value):
    import yaml
    from job_agent.candidate_profile import load_profile
    from tests.fixtures.persona import raw_profile

    data = raw_profile()
    if value is None:
        data["search"].pop("glassdoor_city_id", None)
    else:
        data["search"]["glassdoor_city_id"] = value
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return load_profile(path)


def test_profile_city_id_is_read(tmp_path):
    from job_agent.candidate_profile import glassdoor_city_id
    assert glassdoor_city_id(_profile_with(tmp_path, {"type": "c", "id": 9999999})) == ("C", 9999999)
    assert glassdoor_city_id(_profile_with(tmp_path, {"id": "9999999"})) == ("C", 9999999)


def test_profile_without_the_key_has_no_city_id(tmp_path):
    from job_agent.candidate_profile import glassdoor_city_id
    assert glassdoor_city_id(_profile_with(tmp_path, None)) is None


@pytest.mark.parametrize("bad", [
    9999999, "C9999999", {"type": "C"}, {"id": "abc"}, {"id": 0}, {"id": -5},
    {"id": 1.5}, {"id": True}, {"type": "city", "id": 1}, {"type": "", "id": 1},
])
def test_malformed_city_id_fails_loudly(tmp_path, bad):
    from job_agent.candidate_profile import glassdoor_city_id
    with pytest.raises(ValueError, match="search.glassdoor_city_id"):
        glassdoor_city_id(_profile_with(tmp_path, bad))
