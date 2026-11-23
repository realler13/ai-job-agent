"""LinkedIn circuit breaker: after SITE_BREAKER_TRIP["linkedin"] searches in a
row that time out or come back empty, the run's LinkedIn searches (primary,
remote and secondary passes) are skipped, except one probe at the start of
each later pass and after every 10 skips in a pass; a good probe closes the
breaker. One summary line per run; Indeed and the other phases
carry on.

Hermetic: scrape_jobs and _run_with_timeout are patched at the main module,
pacing gaps are zeroed unless a test sets them, no network, no LLM.
"""
import logging
import threading
import time

import pandas as pd
import pytest

import job_agent.main as main_mod
from job_agent.config import SITE_BREAKER_TRIP
from job_agent.main import _SiteBreaker, _ScrapeTimeout, _parallel_jobspy_pass

logger = logging.getLogger("test-linkedin-breaker")

BREAKER_MSG = "circuit breaker"


class _FakeSession:
    id = "test-session"

    def check_stop(self):
        return False

    def wait_if_paused(self):
        pass


class _FakeDB:
    def __init__(self):
        self.saved = []
        self._lock = threading.Lock()

    def save_job(self, rec, session_id=None, keyword_category=None):
        with self._lock:
            self.saved.append(rec.get("title"))


@pytest.fixture(autouse=True)
def _zero_gaps(monkeypatch):
    monkeypatch.setattr(main_mod, "SITE_MIN_GAP_SEC", {})


def _df(title):
    return pd.DataFrame([{"title": title, "company": "Co", "job_url": f"http://x/{title}"}])


class _Recorder:
    """Fake scrape_jobs: per-site answers, records every call in order."""

    def __init__(self, linkedin_answer, indeed_answer=None):
        self.calls = []
        self._lock = threading.Lock()
        self.linkedin_answer = linkedin_answer
        self.indeed_answer = indeed_answer or (lambda kw: _df(f"indeed-{kw}"))

    def __call__(self, site_name=None, search_term=None, **kw):
        site = site_name[0]
        with self._lock:
            self.calls.append((site, search_term))
        if site == "linkedin":
            return self.linkedin_answer(search_term)
        return self.indeed_answer(search_term)

    def keywords(self, site):
        return [kw for s, kw in self.calls if s == site]


def _pass(sites, keywords, recorder, monkeypatch, breakers, label="primary"):
    monkeypatch.setattr(main_mod, "scrape_jobs", recorder)
    all_jobs: list = []
    _parallel_jobspy_pass(
        sites, keywords,
        lambda site, keyword: dict(site_name=[site], search_term=keyword),
        session=_FakeSession(), control_flags=None,
        progress_callback=lambda *a, **k: None,
        db=_FakeDB(), all_jobs=all_jobs, limit=None,
        label=label, base_pct=0.1, logger=logger, breakers=breakers,
    )
    return all_jobs


# ── The breaker itself ───────────────────────────────────────────────────────

def test_trips_after_n_bad_in_a_row():
    b = _SiteBreaker("linkedin", 3)
    assert b.record(False) is None
    assert b.record(False) is None
    assert b.record(False, "primary/'kw3'") == "tripped"   # the tripping call says so
    assert b.tripped and b.tripped_at == "primary/'kw3'"
    assert b.record(False) is None                    # trips once, not again
    assert b.trips == 1


def test_good_search_resets_the_count():
    b = _SiteBreaker("linkedin", 3)
    for ok in (False, False, True, False, False, True, False, False):
        b.record(ok)
    assert not b.tripped


def test_only_a_probe_can_close_it():
    """A good result that was not a probe (none was let through) changes nothing."""
    b = _SiteBreaker("linkedin", 2)
    b.allow("primary"); b.record(False); b.allow("primary"); b.record(False)
    assert b.record(True) is None
    assert b.tripped
    assert b.allow("primary") is False


def test_allow_counts_skips_per_pass_and_probes_each_later_pass():
    b = _SiteBreaker("linkedin", 1)
    assert b.allow("primary") is True                      # nothing skipped before a trip
    assert b.record(False) == "tripped"
    assert b.allow("primary") is False
    assert b.allow("primary") is False
    assert b.allow("remote") is True                  # probe at the start of a later pass
    assert b.record(False) is None                    # bad probe: stays open
    assert b.allow("remote") is False
    assert b.allow("secondary") is True                      # probe again
    b.record(False)
    for _ in range(3):
        assert b.allow("secondary") is False
    assert b.skipped == {"primary": 2, "remote": 1, "secondary": 3}
    assert b.probes == 2 and b.tripped


def test_probe_after_every_10_skips_within_a_pass():
    b = _SiteBreaker("linkedin", 1)
    b.allow("primary"); b.record(False)
    pattern = []
    for _ in range(25):
        ok = b.allow("primary")
        pattern.append(ok)
        if ok:
            b.record(False)
    # 10 skips, probe, 10 skips, probe, 3 skips
    assert [i for i, ok in enumerate(pattern) if ok] == [10, 21]
    assert b.skipped == {"primary": 23} and b.probes == 2


def test_good_probe_closes_and_resets_the_count():
    b = _SiteBreaker("linkedin", 2)
    b.allow("primary"); b.record(False); b.allow("primary"); b.record(False)
    for _ in range(4):
        b.allow("primary")                                 # 4 skips
    assert b.allow("remote") is True
    assert b.record(True) == "recovered"
    assert not b.tripped and b.recoveries == 1 and b.recovered_after == 4
    assert b.allow("remote") is True
    assert b.record(False) is None                    # count starts again from 0
    assert not b.tripped
    assert b.record(False) == "tripped"               # and it can trip again
    assert b.trips == 2


def test_bad_probe_keeps_it_open():
    b = _SiteBreaker("linkedin", 1)
    b.allow("primary"); b.record(False)
    assert b.allow("remote") is True
    assert b.record(False) is None
    assert b.tripped and b.recoveries == 0


def test_report_is_one_warning_with_the_counts_and_reason(caplog):
    b = _SiteBreaker("linkedin", 2)
    b.allow("primary"); b.record(False); b.allow("primary"); b.record(False, "primary/'Producer'")
    for _ in range(5):
        b.allow("primary")
    b.allow("remote"); b.record(False)                # bad probe
    b.allow("remote"); b.allow("remote")
    with caplog.at_level(logging.INFO, logger="test-linkedin-breaker"):
        b.report(logger)
        b.report(logger)                               # second call is silent
    assert len(caplog.records) == 1
    rec = caplog.records[0]
    assert rec.levelno == logging.WARNING
    msg = rec.getMessage()
    assert "linkedin circuit breaker: tripped 1 time(s)" in msg
    assert "7 linkedin searches skipped (primary 5, remote 2)" in msg
    assert "1 probe search(es), 0 recovery(ies), still open at the end" in msg
    assert "timed out or returned nothing" in msg
    assert "primary/'Producer'" in msg


def test_report_counts_trips_and_recoveries(caplog):
    b = _SiteBreaker("linkedin", 1)
    b.allow("primary"); b.record(False, "primary/'a'")
    b.allow("primary")                                      # 1 skip
    b.allow("remote"); b.record(True)                  # recovered
    b.allow("remote"); b.record(False, "remote/'b'")   # trips again
    b.allow("secondary"); b.record(True)                      # recovered again
    with caplog.at_level(logging.INFO, logger="test-linkedin-breaker"):
        b.report(logger)
    msg = caplog.records[0].getMessage()
    assert "tripped 2 time(s)" in msg and "(first at primary/'a')" in msg
    assert "1 linkedin searches skipped (primary 1)" in msg
    assert "2 probe search(es), 2 recovery(ies), closed at the end" in msg


def test_report_is_one_info_line_when_not_tripped(caplog):
    b = _SiteBreaker("linkedin", 2)
    b.record(False)
    with caplog.at_level(logging.INFO, logger="test-linkedin-breaker"):
        b.report(logger)
        b.report(logger)
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, "linkedin circuit breaker: not tripped this run")]


def test_thread_safe_trips_exactly_once():
    """Many workers recording bad searches at once: exactly one record() call
    reports the trip, a bad probe never closes it, and every allow() is
    either a run search, a probe or a counted skip."""
    b = _SiteBreaker("linkedin", 50)
    events = []
    start = threading.Barrier(8)

    def worker():
        start.wait()
        for _ in range(200):
            ev = b.record(False)
            if ev:
                events.append(ev)
            b.allow("primary")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert events == ["tripped"]
    assert b.tripped and b.trips == 1 and b.recoveries == 0
    # 1600 allow() calls: at most ~50 before the trip, then skips with a probe
    # let through after every 10 (or held while one is in flight)
    assert b.skipped["primary"] + b.probes <= 1600
    assert b.skipped["primary"] >= 1000


def test_fresh_breakers_per_run():
    a, b = main_mod._site_breakers(), main_mod._site_breakers()
    assert set(a) == set(SITE_BREAKER_TRIP)
    assert a["linkedin"] is not b["linkedin"]
    assert not a["linkedin"].tripped


def test_trip_point_leaves_room_for_a_slow_working_day():
    """Calibration guard. On a slow day LinkedIn still works between bursts of
    timeouts, and the trip point must sit above such bursts so a working day
    never trips it. Replay real run logs before lowering it."""
    assert SITE_BREAKER_TRIP["linkedin"] > 10


# ── Inside a jobspy pass ─────────────────────────────────────────────────────

def test_dead_linkedin_stops_after_trip_and_indeed_runs_everything(monkeypatch):
    kws = [f"kw{i}" for i in range(10)]
    rec = _Recorder(linkedin_answer=lambda kw: pd.DataFrame())
    breakers = {"linkedin": _SiteBreaker("linkedin", 3)}
    all_jobs = _pass(["linkedin", "indeed"], kws, rec, monkeypatch, breakers)
    assert rec.keywords("linkedin") == ["kw0", "kw1", "kw2"]
    assert rec.keywords("indeed") == kws
    assert len(all_jobs) == 10
    assert breakers["linkedin"].skipped == {"primary": 7}


def test_timeouts_count_as_bad(monkeypatch):
    def fake_timeout(func, args=(), kwargs=None, timeout_sec=120, label=""):
        if "/linkedin/" in label:
            raise _ScrapeTimeout(f"{label} timed out")
        return func(**(kwargs or {}))

    monkeypatch.setattr(main_mod, "_run_with_timeout", fake_timeout)
    rec = _Recorder(linkedin_answer=lambda kw: _df(kw))
    breakers = {"linkedin": _SiteBreaker("linkedin", 2)}
    _pass(["linkedin"], [f"kw{i}" for i in range(6)], rec, monkeypatch, breakers)
    assert breakers["linkedin"].tripped
    assert breakers["linkedin"].skipped == {"primary": 4}


def test_errors_count_as_bad(monkeypatch):
    def boom(kw):
        raise RuntimeError("429")

    rec = _Recorder(linkedin_answer=boom)
    breakers = {"linkedin": _SiteBreaker("linkedin", 2)}
    _pass(["linkedin"], ["a", "b", "c"], rec, monkeypatch, breakers)
    assert rec.keywords("linkedin") == ["a", "b"]


def test_scattered_bad_searches_never_trip(monkeypatch):
    """A slow-but-working day: bad searches between good ones run in full."""
    pattern = {"kw1", "kw2", "kw4", "kw5", "kw7", "kw8"}       # pairs, never 3 in a row
    kws = [f"kw{i}" for i in range(10)]
    rec = _Recorder(linkedin_answer=lambda kw: pd.DataFrame() if kw in pattern else _df(kw))
    breakers = {"linkedin": _SiteBreaker("linkedin", 3)}
    _pass(["linkedin"], kws, rec, monkeypatch, breakers)
    assert rec.keywords("linkedin") == kws
    assert not breakers["linkedin"].tripped


def test_skipped_searches_do_not_wait_for_pacing(monkeypatch):
    """A skip must cost nothing: no pacing sleep per skipped keyword (10
    skips: the next keyword would be a probe, a real paced search)."""
    monkeypatch.setattr(main_mod, "SITE_MIN_GAP_SEC", {"linkedin": 5})
    rec = _Recorder(linkedin_answer=lambda kw: pd.DataFrame())
    breakers = {"linkedin": _SiteBreaker("linkedin", 1)}
    t0 = time.monotonic()
    _pass(["linkedin"], [f"kw{i}" for i in range(11)], rec, monkeypatch, breakers)
    assert time.monotonic() - t0 < 3
    assert breakers["linkedin"].skipped == {"primary": 10}


def test_dead_site_gets_one_probe_per_later_pass(monkeypatch):
    rec = _Recorder(linkedin_answer=lambda kw: pd.DataFrame())
    breakers = {"linkedin": _SiteBreaker("linkedin", 2)}
    _pass(["linkedin", "indeed"], ["a", "b", "c"], rec, monkeypatch, breakers, label="primary")
    _pass(["linkedin", "indeed"], ["r1", "r2"], rec, monkeypatch, breakers, label="remote")
    _pass(["linkedin", "indeed"], ["l1", "l2", "l3"], rec, monkeypatch, breakers, label="secondary")
    assert rec.keywords("linkedin") == ["a", "b", "r1", "l1"]
    assert breakers["linkedin"].skipped == {"primary": 1, "remote": 1, "secondary": 2}
    assert breakers["linkedin"].probes == 2 and breakers["linkedin"].tripped
    assert rec.keywords("indeed") == ["a", "b", "c", "r1", "r2", "l1", "l2", "l3"]


def test_dead_site_probes_every_10_skips_in_a_pass(monkeypatch):
    rec = _Recorder(linkedin_answer=lambda kw: pd.DataFrame())
    breakers = {"linkedin": _SiteBreaker("linkedin", 2)}
    kws = [f"kw{i}" for i in range(25)]
    _pass(["linkedin"], kws, rec, monkeypatch, breakers)
    assert rec.keywords("linkedin") == ["kw0", "kw1", "kw12", "kw23"]
    assert breakers["linkedin"].skipped == {"primary": 21}


def test_site_that_comes_back_is_picked_up_by_a_probe(monkeypatch, caplog):
    """A long bad streak, then LinkedIn works again."""
    kws = [f"kw{i}" for i in range(30)]
    rec = _Recorder(linkedin_answer=lambda kw: pd.DataFrame() if int(kw[2:]) < 8 else _df(kw))
    breakers = {"linkedin": _SiteBreaker("linkedin", 2)}
    with caplog.at_level(logging.INFO, logger="test-linkedin-breaker"):
        all_jobs = _pass(["linkedin"], kws, rec, monkeypatch, breakers)
    # kw0, kw1 trip it; kw2..kw11 skipped; kw12 probes, works, and the rest run
    assert rec.keywords("linkedin") == ["kw0", "kw1"] + kws[12:]
    assert len(all_jobs) == 18
    b = breakers["linkedin"]
    assert not b.tripped and b.trips == 1 and b.recoveries == 1
    assert b.skipped == {"primary": 10}
    recovered = [r for r in caplog.records if "recovered" in r.getMessage()]
    assert [(r.levelno, r.getMessage()) for r in recovered] == [
        (logging.INFO, "linkedin recovered after 10 skipped searches")]


def test_streak_carries_across_passes(monkeypatch):
    """The count is per run, not per pass: 2 bad at the end of the primary pass + 1 bad at
    the start of the remote pass trips a breaker set to 3."""
    rec = _Recorder(linkedin_answer=lambda kw: _df(kw) if kw == "good" else pd.DataFrame())
    breakers = {"linkedin": _SiteBreaker("linkedin", 3)}
    _pass(["linkedin"], ["good", "b1", "b2"], rec, monkeypatch, breakers, label="primary")
    assert not breakers["linkedin"].tripped
    _pass(["linkedin"], ["b3", "r2", "r3"], rec, monkeypatch, breakers, label="remote")
    assert breakers["linkedin"].tripped
    assert breakers["linkedin"].skipped == {"remote": 2}


# ── The whole jobspy phase through run_search_session ───────────────────────

class _Done(Exception):
    pass


def _session(monkeypatch, rec, primary_kws, logger_name, caplog):
    monkeypatch.setattr(main_mod, "scrape_jobs", rec)
    monkeypatch.setattr(main_mod, "load_profile_text", lambda p: "profile")
    monkeypatch.setattr(main_mod, "load_proxies", lambda: None)
    monkeypatch.setattr(main_mod, "SITE_BREAKER_TRIP", {"linkedin": 3})
    monkeypatch.setattr(main_mod, "REMOTE_SECOND_PASS", True)
    monkeypatch.setattr(main_mod, "SECONDARY_SECOND_PASS", True)
    monkeypatch.setattr(main_mod, "REMOTE_PASS_SITES", ["linkedin", "indeed"])
    monkeypatch.setattr(main_mod, "REMOTE_PASS_KEYWORDS", ["r1", "r2"])
    monkeypatch.setattr(main_mod, "SECONDARY_PASS_KEYWORDS_EXTRA", ["l1"])

    def stop_here(*a, **k):
        raise _Done
    monkeypatch.setattr(main_mod, "dedup_scraped_jobs", stop_here)
    with caplog.at_level(logging.INFO, logger=logger_name), pytest.raises(_Done):
        main_mod.run_search_session(
            "r.pdf", "cv.docx", "m", "g",
            site_list=["linkedin", "indeed"], keywords_list=primary_kws,
            send_email=False, session=main_mod.SessionManager(), db=_FakeDB(),
            logger=logging.getLogger(logger_name),
        )
    return [r for r in caplog.records if BREAKER_MSG in r.getMessage()]


def test_run_search_session_recovers_when_linkedin_comes_back(monkeypatch, caplog):
    """Dead in the primary pass, working again by the remote pass: the remote
    probe closes the breaker and the remote and secondary passes run in full."""
    primary_kws = [f"kw{i}" for i in range(8)]
    rec = _Recorder(linkedin_answer=lambda kw: pd.DataFrame() if kw.startswith("kw") else _df(kw))
    lines = _session(monkeypatch, rec, primary_kws, "t-run3", caplog)
    assert rec.keywords("linkedin") == ["kw0", "kw1", "kw2", "r1", "r2", "r1", "r2", "l1"]
    assert [r.getMessage() for r in caplog.records
            if "recovered after" in r.getMessage()] == ["linkedin recovered after 5 skipped searches"]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    msg = lines[0].getMessage()
    assert "tripped 1 time(s)" in msg
    assert "5 linkedin searches skipped (primary 5)" in msg
    assert "1 probe search(es), 1 recovery(ies), closed at the end" in msg


def test_run_search_session_trips_once_and_logs_one_warning(monkeypatch, caplog):
    rec = _Recorder(linkedin_answer=lambda kw: pd.DataFrame())
    monkeypatch.setattr(main_mod, "scrape_jobs", rec)
    monkeypatch.setattr(main_mod, "load_profile_text", lambda p: "profile")
    monkeypatch.setattr(main_mod, "load_proxies", lambda: None)
    monkeypatch.setattr(main_mod, "SITE_BREAKER_TRIP", {"linkedin": 3})
    monkeypatch.setattr(main_mod, "REMOTE_SECOND_PASS", True)
    monkeypatch.setattr(main_mod, "SECONDARY_SECOND_PASS", True)
    monkeypatch.setattr(main_mod, "REMOTE_PASS_SITES", ["linkedin", "indeed"])
    monkeypatch.setattr(main_mod, "REMOTE_PASS_KEYWORDS", ["r1", "r2"])
    monkeypatch.setattr(main_mod, "SECONDARY_PASS_KEYWORDS_EXTRA", ["l1"])

    # Stop right after scraping: everything downstream (dedup, the AI cascade)
    # is out of scope and must not run in a test.
    def stop_here(*a, **k):
        raise _Done
    monkeypatch.setattr(main_mod, "dedup_scraped_jobs", stop_here)

    primary_kws = [f"kw{i}" for i in range(8)]
    with caplog.at_level(logging.INFO, logger="t-run"), pytest.raises(_Done):
        main_mod.run_search_session(
            "r.pdf", "cv.docx", "m", "g",
            site_list=["linkedin", "indeed"], keywords_list=primary_kws,
            send_email=False, session=main_mod.SessionManager(), db=_FakeDB(),
            logger=logging.getLogger("t-run"),
        )

    # 3 to trip, then one probe at the start of the remote and the secondary pass
    assert rec.keywords("linkedin") == ["kw0", "kw1", "kw2", "r1", "r1"]
    # Indeed ran every keyword of every pass
    assert rec.keywords("indeed") == primary_kws + ["r1", "r2"] + ["r1", "r2", "l1"]
    lines = [r for r in caplog.records if BREAKER_MSG in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    msg = lines[0].getMessage()
    # 5 primary + 1 remote + 2 secondary
    assert "8 linkedin searches skipped (primary 5, remote 1, secondary 2)" in msg
    assert "2 probe search(es), 0 recovery(ies), still open at the end" in msg


def test_run_search_session_normal_day_skips_nothing(monkeypatch, caplog):
    rec = _Recorder(linkedin_answer=lambda kw: _df(f"li-{kw}"))
    monkeypatch.setattr(main_mod, "scrape_jobs", rec)
    monkeypatch.setattr(main_mod, "load_profile_text", lambda p: "profile")
    monkeypatch.setattr(main_mod, "load_proxies", lambda: None)
    monkeypatch.setattr(main_mod, "REMOTE_SECOND_PASS", False)
    monkeypatch.setattr(main_mod, "SECONDARY_SECOND_PASS", False)

    def stop_here(*a, **k):
        raise _Done
    monkeypatch.setattr(main_mod, "dedup_scraped_jobs", stop_here)

    kws = [f"kw{i}" for i in range(6)]
    with caplog.at_level(logging.INFO, logger="t-run2"), pytest.raises(_Done):
        main_mod.run_search_session(
            "r.pdf", "cv.docx", "m", "g",
            site_list=["linkedin", "indeed"], keywords_list=kws,
            send_email=False, session=main_mod.SessionManager(), db=_FakeDB(),
            logger=logging.getLogger("t-run2"),
        )
    assert rec.keywords("linkedin") == kws
    lines = [r for r in caplog.records if BREAKER_MSG in r.getMessage()]
    assert [(r.levelno, r.getMessage()) for r in lines] == [
        (logging.INFO, "linkedin circuit breaker: not tripped this run")]
