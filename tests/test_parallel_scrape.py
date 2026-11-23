"""Tests for _parallel_jobspy_pass (2026-07-18: per-site worker threads
replace the sequential site-alternating jobspy loops — 66% of run wall-clock).

All hermetic: scrape_jobs is patched at the main module (from jobspy import
scrape_jobs creates a main-local reference), SITE_MIN_GAP_SEC is zeroed so
tests don't sleep.
"""
import threading
import time
import logging

import pandas as pd
import pytest

import job_agent.main as main_mod
from job_agent.main import _parallel_jobspy_pass, _ScrapeTimeout

logger = logging.getLogger("test-parallel")


class _FakeSession:
    id = "test-session"

    def __init__(self, stop_after=None):
        self._stop_after = stop_after
        self.checks = 0

    def check_stop(self):
        self.checks += 1
        return self._stop_after is not None and self.checks > self._stop_after

    def wait_if_paused(self):
        pass


class _FakeDB:
    def __init__(self):
        self.saved = []
        self._lock = threading.Lock()

    def save_job(self, rec, session_id=None, keyword_category=None):
        with self._lock:
            self.saved.append((rec.get("title"), session_id, keyword_category))


@pytest.fixture(autouse=True)
def _zero_gaps(monkeypatch):
    monkeypatch.setattr(main_mod, "SITE_MIN_GAP_SEC", {})


def _df(title):
    return pd.DataFrame([{"title": title, "company": "Co", "job_url": f"http://x/{title}"}])


def _run(sites, keywords, scrape_fn, monkeypatch, *, limit=None, session=None):
    monkeypatch.setattr(main_mod, "scrape_jobs", scrape_fn)
    session = session or _FakeSession()
    db = _FakeDB()
    all_jobs: list = []
    _parallel_jobspy_pass(
        sites, keywords,
        lambda site, keyword: dict(site_name=[site], search_term=keyword),
        session=session, control_flags=None,
        progress_callback=lambda *a, **k: None,
        db=db, all_jobs=all_jobs, limit=limit,
        label="test", base_pct=0.1, logger=logger,
    )
    return all_jobs, db


def test_all_site_keyword_pairs_scraped(monkeypatch):
    calls = []
    lock = threading.Lock()

    def fake_scrape(site_name=None, search_term=None, **kw):
        with lock:
            calls.append((site_name[0], search_term))
        return _df(f"{site_name[0]}-{search_term}")

    all_jobs, db = _run(["linkedin", "indeed"], ["kw1", "kw2", "kw3"], fake_scrape, monkeypatch)
    assert len(calls) == 6                      # 2 sites x 3 keywords
    assert len(all_jobs) == 6                   # every row collected
    assert len(db.saved) == 6                   # every row persisted
    # each site walked its keyword list IN ORDER (sequential per site)
    for site in ("linkedin", "indeed"):
        site_kws = [kw for s, kw in calls if s == site]
        assert site_kws == ["kw1", "kw2", "kw3"]


def test_sites_actually_run_concurrently(monkeypatch):
    """Two sites x 4 keywords at 0.15s per call: sequential would be >=1.2s,
    parallel should be close to 0.6s. Generous ceiling avoids flakiness."""
    def slow_scrape(site_name=None, search_term=None, **kw):
        time.sleep(0.15)
        return _df(f"{site_name[0]}-{search_term}")

    t0 = time.monotonic()
    all_jobs, _ = _run(["linkedin", "indeed"], ["a", "b", "c", "d"], slow_scrape, monkeypatch)
    elapsed = time.monotonic() - t0
    assert len(all_jobs) == 8
    assert elapsed < 1.1, f"not parallel: {elapsed:.2f}s for 8x0.15s calls"


def test_one_site_erroring_does_not_kill_the_other(monkeypatch):
    def fake_scrape(site_name=None, search_term=None, **kw):
        if site_name[0] == "linkedin":
            raise RuntimeError("boom")
        return _df(f"indeed-{search_term}")

    all_jobs, _ = _run(["linkedin", "indeed"], ["kw1", "kw2"], fake_scrape, monkeypatch)
    titles = sorted(j["title"] for j in all_jobs)
    assert titles == ["indeed-kw1", "indeed-kw2"]


def test_timeout_is_abandoned_not_fatal(monkeypatch):
    """A _ScrapeTimeout from the timeout wrapper skips the keyword and moves on."""
    def fake_run_with_timeout(func, args=(), kwargs=None, timeout_sec=120, label=""):
        if "kw2" in label:
            raise _ScrapeTimeout(f"{label} timed out")
        return func(**(kwargs or {}))

    monkeypatch.setattr(main_mod, "_run_with_timeout", fake_run_with_timeout)

    def fake_scrape(site_name=None, search_term=None, **kw):
        return _df(f"{site_name[0]}-{search_term}")

    all_jobs, _ = _run(["linkedin"], ["kw1", "kw2", "kw3"], fake_scrape, monkeypatch)
    titles = sorted(j["title"] for j in all_jobs)
    assert titles == ["linkedin-kw1", "linkedin-kw3"]


def test_limit_stops_workers_early(monkeypatch):
    calls = []
    lock = threading.Lock()

    def fake_scrape(site_name=None, search_term=None, **kw):
        with lock:
            calls.append((site_name[0], search_term))
        return _df(f"{site_name[0]}-{search_term}")

    all_jobs, _ = _run(["linkedin", "indeed"], [f"kw{i}" for i in range(50)],
                       fake_scrape, monkeypatch, limit=4)
    # workers stop soon after the limit trips — nowhere near the 100-call grid
    assert len(all_jobs) >= 4
    assert len(calls) < 20


def test_stop_flag_honored(monkeypatch):
    def fake_scrape(site_name=None, search_term=None, **kw):
        return _df(f"{site_name[0]}-{search_term}")

    session = _FakeSession(stop_after=2)  # stops after 2 check_stop() calls
    all_jobs, _ = _run(["linkedin", "indeed"], [f"kw{i}" for i in range(50)],
                       fake_scrape, monkeypatch, session=session)
    assert len(all_jobs) < 100  # far from the full grid


def test_daemon_thread_timeout_does_not_block(monkeypatch):
    """_run_in_thread_with_timeout abandons a hung call via a daemon thread."""
    from job_agent.main import _run_in_thread_with_timeout

    def hang_forever():
        time.sleep(60)

    t0 = time.monotonic()
    with pytest.raises(_ScrapeTimeout):
        _run_in_thread_with_timeout(hang_forever, (), {}, timeout_sec=0.3, label="hang")
    assert time.monotonic() - t0 < 2
    # the abandoned thread must be a daemon (never blocks interpreter exit)
    hung = [t for t in threading.enumerate() if t.name.startswith("scrape-hang")]
    assert all(t.daemon for t in hung)


def test_tripped_breaker_skips_only_its_own_site(monkeypatch):
    """breakers= is per site: an open LinkedIn breaker (tripped earlier in the
    run) sends only its one probe request at the start of the pass (still
    bad here, so it stays open), and Indeed's worker is untouched."""
    from job_agent.main import _SiteBreaker

    calls = []
    lock = threading.Lock()

    def fake_scrape(site_name=None, search_term=None, **kw):
        with lock:
            calls.append((site_name[0], search_term))
        if site_name[0] == "linkedin":
            return pd.DataFrame()
        return _df(f"{site_name[0]}-{search_term}")

    breaker = _SiteBreaker("linkedin", 1)
    breaker.record(False)                       # already open
    monkeypatch.setattr(main_mod, "scrape_jobs", fake_scrape)
    all_jobs: list = []
    _parallel_jobspy_pass(
        ["linkedin", "indeed"], ["kw1", "kw2"],
        lambda site, keyword: dict(site_name=[site], search_term=keyword),
        session=_FakeSession(), control_flags=None,
        progress_callback=lambda *a, **k: None,
        db=_FakeDB(), all_jobs=all_jobs, limit=None,
        label="remote", base_pct=0.1, logger=logger,
        breakers={"linkedin": breaker},
    )
    assert sorted(calls) == [("indeed", "kw1"), ("indeed", "kw2"), ("linkedin", "kw1")]
    assert breaker.skipped == {"remote": 1}
    assert breaker.tripped and breaker.probes == 1
