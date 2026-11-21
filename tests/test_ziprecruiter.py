"""Tests for the ZipRecruiter scraper (Scrapling solve_cloudflare + parsing)."""
import pandas as pd

import job_agent.ziprecruiter_scraper as zr
from tests.fixtures.persona import HOME


# Minimal fixture mirroring the live DOM: article[id^='job-card'] with an h2 title
# and data-testid company/location anchors. The first listing is doubled (same id)
# to exercise dedup.
FIXTURE_HTML = f"""
<html><body>
  <article id="job-card-AAA111">
    <h2>Senior Video Producer</h2>
    <a data-testid="job-card-company" href="/co/Acme">Acme Studios</a>
    <a data-testid="job-card-location" href="/x">{HOME}</a>
  </article>
  <article id="job-card-AAA111">
    <h2>Senior Video Producer</h2>
    <a data-testid="job-card-company">Acme Studios</a>
    <a data-testid="job-card-location">{HOME}</a>
  </article>
  <article id="job-card-BBB222">
    <h2>Podcast Producer</h2>
    <a data-testid="job-card-company">MediaCo</a>
    <a data-testid="job-card-location">Remote</a>
  </article>
</body></html>
"""


class _FakePage:
    def __init__(self, body):
        self.body = body
        self.status = 200


def test_parses_and_dedupes(monkeypatch):
    monkeypatch.setattr(
        zr.StealthyFetcher, "fetch",
        staticmethod(lambda *a, **k: _FakePage(FIXTURE_HTML)),
    )
    df = zr.scrape_ziprecruiter("video producer", HOME, results_wanted=20)
    assert df is not None
    # 3 article cards, but the doubled AAA111 dedups -> 2 unique
    assert len(df) == 2
    titles = set(df["title"])
    assert titles == {"Senior Video Producer", "Podcast Producer"}
    row = df[df["title"] == "Senior Video Producer"].iloc[0]
    assert row["company"] == "Acme Studios"
    assert row["location"] == HOME
    assert row["job_url"] == "https://www.ziprecruiter.com/jobs//j?lvk=AAA111"
    assert row["site"] == "zip_recruiter"


def test_respects_results_wanted(monkeypatch):
    monkeypatch.setattr(
        zr.StealthyFetcher, "fetch",
        staticmethod(lambda *a, **k: _FakePage(FIXTURE_HTML)),
    )
    df = zr.scrape_ziprecruiter("producer", results_wanted=1)
    assert len(df) == 1


def test_blocked_page_returns_none(monkeypatch):
    # Block markers / tiny shell -> None (counts toward the circuit breaker).
    monkeypatch.setattr(
        zr.StealthyFetcher, "fetch",
        staticmethod(lambda *a, **k: _FakePage("<html><body>Access denied</body></html>")),
    )
    assert zr.scrape_ziprecruiter("producer") is None


def test_rendered_empty_page_returns_empty_df(monkeypatch):
    # A real-size rendered page with zero cards = legitimate empty result ->
    # empty DataFrame (resets the breaker), NOT None.
    big_empty = "<html><body>" + "<div>filler</div>" * 4000 + "</body></html>"
    monkeypatch.setattr(
        zr.StealthyFetcher, "fetch",
        staticmethod(lambda *a, **k: _FakePage(big_empty)),
    )
    df = zr.scrape_ziprecruiter("producer")
    assert df is not None and df.empty


def test_fetch_exception_returns_none(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("browser crashed")
    monkeypatch.setattr(zr.StealthyFetcher, "fetch", staticmethod(boom))
    assert zr.scrape_ziprecruiter("producer") is None
