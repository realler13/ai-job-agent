"""Tests for the Glassdoor crawl4ai search scraper (parsing + wrapper)."""
import pandas as pd

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
