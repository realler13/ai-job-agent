"""Phase 27c "Is this posting real?" — pipeline stream.

Board-fetch reports from the full-list ATS scrapers (every listed link BEFORE
the keyword filter + an honest "complete" flag), recording through
scrape_ats_boards, the proof that recording never changes which jobs are
scraped / deduped / filtered, the Strong-job steps (auto hiring health,
posting quality reads), the hiring-health query's sixth topic, the scanner
stats split, and the digest (gone rows out, one caution line on Strong cards).
Hermetic: HTTP, Perplexity, the LLM and SMTP are fakes.
"""
import email
import re
import sys
from email.header import decode_header, make_header
import types
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
import requests

from job_agent import ats_scraper, main, perplexity_utils, postprocess
from job_agent.ats_scraper import _take_board_report, scrape_ats_boards
from job_agent.scraper_ashby_sr import scrape_ashby, scrape_smartrecruiters
from job_agent.scraper_bamboohr_comeet import scrape_bamboohr, scrape_comeet
from job_agent.scraper_eightfold_apple import scrape_apple, scrape_eightfold
from job_agent.scraper_rwt import scrape_recruitee, scrape_teamtailor, scrape_workable
from job_agent.scraper_workday import scrape_workday
from tests.fixtures.persona import FIRST_NAME, HOME, LAST_NAME

KW = ["Video Producer"]


# ── HTTP fakes ───────────────────────────────────────────────────────────────
class FakeResp:
    def __init__(self, payload=None, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


def _route(monkeypatch, handler, method="get"):
    """Send every requests.<method> call (module-level and Session) to handler."""
    calls = []

    def fake(*args, **kwargs):
        url = args[-1] if args and isinstance(args[-1], str) else kwargs.get("url", "")
        calls.append((url, kwargs))
        return handler(url, kwargs)

    monkeypatch.setattr(requests, method, fake)
    return calls


def _no_network(monkeypatch):
    def boom(*a, **kw):
        raise requests.exceptions.ConnectionError("no network in tests")
    for m in ("get", "post"):
        monkeypatch.setattr(requests, m, boom)
    monkeypatch.setattr(requests.Session, "get", lambda self, *a, **kw: boom())
    monkeypatch.setattr(requests.Session, "post", lambda self, *a, **kw: boom())


def _links(rep):
    """The links of a board report (reports carry (link, title) pairs)."""
    return [link for link, _title in rep["links"]]


@pytest.fixture(autouse=True)
def _clean_slot():
    ats_scraper._clear_board_report()
    yield
    ats_scraper._clear_board_report()


# ── per-scraper reports ──────────────────────────────────────────────────────
def _gh_jobs():
    return [
        {"title": "Senior Video Producer", "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1"},
        {"title": "Software Engineer", "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/2"},
        {"title": "Accountant", "absolute_url": "https://acme.com/careers?gh_jid=3"},
    ]


def test_greenhouse_reports_every_link_before_the_keyword_filter(monkeypatch):
    _route(monkeypatch, lambda url, kw: FakeResp({"jobs": _gh_jobs(), "meta": {"total": 3}}))
    jobs = ats_scraper.scrape_greenhouse("acme", "Acme", KW)
    assert [j["title"] for j in jobs] == ["Senior Video Producer"]
    rep = _take_board_report()
    assert rep["platform"] == "greenhouse" and rep["token"] == "acme"
    assert rep["ok"] and rep["complete"] and rep["total_count"] == 3
    assert rep["links"] == [(j["absolute_url"], j["title"]) for j in _gh_jobs()]
    # the matched job's job_url is built with the same expression
    assert jobs[0]["job_url"] in _links(rep)


def test_greenhouse_meta_mismatch_is_partial_and_error_is_failed(monkeypatch):
    _route(monkeypatch, lambda url, kw: FakeResp({"jobs": _gh_jobs(), "meta": {"total": 40}}))
    ats_scraper.scrape_greenhouse("acme", "Acme", KW)
    rep = _take_board_report()
    assert rep["ok"] and not rep["complete"]

    _route(monkeypatch, lambda url, kw: FakeResp(status=500))
    assert ats_scraper.scrape_greenhouse("acme", "Acme", KW) == []
    rep = _take_board_report()
    assert not rep["ok"] and not rep["complete"] and rep["links"] == []


def test_lever_reports_full_list(monkeypatch):
    data = [{"text": "Video Producer", "hostedUrl": "https://jobs.lever.co/x/a"},
            {"text": "Barista", "hostedUrl": "https://jobs.lever.co/x/b"}]
    _route(monkeypatch, lambda url, kw: FakeResp(data))
    assert len(ats_scraper.scrape_lever("x", "X", KW)) == 1
    rep = _take_board_report()
    assert rep["complete"] and rep["links"] == [("https://jobs.lever.co/x/a", "Video Producer"),
                                                 ("https://jobs.lever.co/x/b", "Barista")]

    _route(monkeypatch, lambda url, kw: FakeResp({"error": "nope"}))
    ats_scraper.scrape_lever("x", "X", KW)
    assert not _take_board_report()["ok"]


def test_ashby_excludes_unlisted_postings(monkeypatch):
    data = {"jobs": [
        {"title": "Video Producer", "jobUrl": "https://jobs.ashbyhq.com/r/1", "isListed": True},
        {"title": "Video Producer II", "jobUrl": "https://jobs.ashbyhq.com/r/2", "isListed": False},
        {"title": "Engineer", "jobUrl": "https://jobs.ashbyhq.com/r/3"},
    ]}
    _route(monkeypatch, lambda url, kw: FakeResp(data))
    jobs = scrape_ashby("r", "R", KW)
    assert [j["job_url"] for j in jobs] == ["https://jobs.ashbyhq.com/r/1"]
    rep = _take_board_report()
    assert rep["complete"] and rep["total_count"] == 2
    assert _links(rep) == ["https://jobs.ashbyhq.com/r/1", "https://jobs.ashbyhq.com/r/3"]


def _sr_page(ids, total):
    return {"totalFound": total, "content": [
        {"id": i, "name": "Video Producer" if i == "p1" else f"Role {i}", "location": {}} for i in ids]}


def test_smartrecruiters_complete_only_when_all_pages_read(monkeypatch):
    monkeypatch.setattr("job_agent.scraper_ashby_sr._sr_fetch_posting_description",
                        lambda slug, pid: "")
    pages = {0: _sr_page(["p1", "p2"], 3), 2: _sr_page(["p3"], 3)}
    _route(monkeypatch, lambda url, kw: FakeResp(pages[kw["params"]["offset"]]))
    jobs = scrape_smartrecruiters("Acme", "Acme", KW)
    assert len(jobs) == 1
    rep = _take_board_report()
    assert rep["ok"] and rep["complete"] and rep["total_count"] == 3
    assert _links(rep) == [f"https://jobs.smartrecruiters.com/Acme/p{i}" for i in (1, 2, 3)]
    assert jobs[0]["job_url"] == _links(rep)[0]


def test_smartrecruiters_error_mid_pagination_is_partial(monkeypatch):
    monkeypatch.setattr("job_agent.scraper_ashby_sr._sr_fetch_posting_description",
                        lambda slug, pid: "")

    def handler(url, kw):
        if kw["params"]["offset"] == 0:
            return FakeResp(_sr_page(["p1", "p2"], 3))
        return FakeResp(status=503)
    _route(monkeypatch, handler)
    scrape_smartrecruiters("Acme", "Acme", KW)
    rep = _take_board_report()
    assert rep["ok"] and not rep["complete"] and len(rep["links"]) == 2


def test_smartrecruiters_dead_slug_reads_as_empty_board(monkeypatch):
    _route(monkeypatch, lambda url, kw: FakeResp({"totalFound": 0, "content": []}))
    assert scrape_smartrecruiters("Dead", "Dead", KW) == []
    rep = _take_board_report()
    assert rep["total_count"] == 0 and rep["links"] == []   # 'empty' -> never evidence

    _route(monkeypatch, lambda url, kw: FakeResp(status=404))
    scrape_smartrecruiters("Dead", "Dead", KW)
    assert not _take_board_report()["ok"]


def test_recruitee_excludes_closed_offers(monkeypatch):
    data = {"offers": [
        {"title": "Video Producer", "careers_url": "https://r.recruitee.com/o/a", "status": "published"},
        {"title": "Video Producer", "slug": "b", "status": "closed"},
        {"title": "Designer", "slug": "c", "status": "published"},
    ]}
    _route(monkeypatch, lambda url, kw: FakeResp(data))
    assert len(scrape_recruitee("r", "R", KW)) == 1
    rep = _take_board_report()
    assert rep["complete"] and _links(rep) == ["https://r.recruitee.com/o/a", "https://r.recruitee.com/o/c"]


def test_workable_complete_only_when_pages_run_out(monkeypatch):
    def page(n, more):
        out = {"results": [{"title": f"Video Producer {n}", "shortcode": f"S{n}"}]}
        if more:
            out["nextToken"] = f"t{n}"
        return out

    _route(monkeypatch, lambda url, kw: FakeResp(page(1 if not kw["json"].get("token") else 2,
                                                      more=not kw["json"].get("token"))),
           method="post")
    jobs = scrape_workable("w", "W", KW)
    rep = _take_board_report()
    assert rep["complete"] and _links(rep) == ["https://apply.workable.com/w/j/S1/",
                                                "https://apply.workable.com/w/j/S2/"]
    assert {j["job_url"] for j in jobs} == set(_links(rep))

    # Never-ending nextToken -> stops at the max_pages cap -> not complete.
    _route(monkeypatch, lambda url, kw: FakeResp(page(1, more=True)), method="post")
    scrape_workable("w", "W", KW)
    rep = _take_board_report()
    assert rep["ok"] and not rep["complete"]

    # Error on page 2 -> partial.
    def err(url, kw):
        return FakeResp(page(1, True)) if not kw["json"].get("token") else FakeResp(status=500)
    _route(monkeypatch, err, method="post")
    scrape_workable("w", "W", KW)
    rep = _take_board_report()
    assert rep["ok"] and not rep["complete"]


def test_eightfold_complete_and_capped(monkeypatch):
    def handler(url, kw):
        start = kw["params"]["start"]
        if start == 0:
            return FakeResp({"count": 2, "positions": [{"id": 11, "name": "Video Producer"},
                                                       {"id": 12, "name": "Engineer"}]})
        return FakeResp({"count": 2, "positions": []})
    _route(monkeypatch, handler)
    assert len(scrape_eightfold("netflix.com", "Netflix", KW)) == 1
    rep = _take_board_report()
    assert rep["complete"] and _links(rep) == ["https://explore.jobs.netflix.net/careers/job/11",
                                                "https://explore.jobs.netflix.net/careers/job/12"]

    # count larger than the max_pages cap -> not complete
    _route(monkeypatch, lambda url, kw: FakeResp({"count": 50000, "positions": [
        {"id": kw["params"]["start"] + i + 1, "name": "Engineer"} for i in range(100)]}))
    scrape_eightfold("netflix.com", "Netflix", KW)
    rep = _take_board_report()
    assert rep["ok"] and not rep["complete"] and rep["total_count"] == 8000


def _netflix_api(board, page_cap=10, pages=None):
    """The live Eightfold API: at most `page_cap` positions per page whatever
    `num` asks for; `pages` = explicit {start: positions} overrides."""
    def handler(url, kw):
        start = kw["params"]["start"]
        if pages is not None and start in pages:
            return FakeResp({"count": len(board), "positions": pages[start]})
        page = board[start:start + min(page_cap, kw["params"]["num"])]
        return FakeResp({"count": len(board), "positions": page})
    return handler


def test_eightfold_reads_every_page_when_the_api_caps_pages_at_10(monkeypatch):
    """Regression (2026-10-04 live probe): num=100 returns 10 positions, and
    stepping the offset by 100 read 50 of 471 Netflix jobs."""
    board = [{"id": 1000 + i, "name": "Video Producer" if i in (3, 17, 24) else "Engineer"}
             for i in range(25)]
    calls = _route(monkeypatch, _netflix_api(board))
    jobs = scrape_eightfold("netflix.com", "Netflix", KW)
    assert sorted(j["job_url"].rsplit("/", 1)[1] for j in jobs) == ["1003", "1017", "1024"]
    rep = _take_board_report()
    assert rep["ok"] and rep["complete"] and rep["total_count"] == 25
    assert len(set(_links(rep))) == 25
    assert [kw["params"]["start"] for _, kw in calls] == [0, 10, 20]


def test_eightfold_shifted_pages_are_not_complete(monkeypatch):
    """A page shift serves one job twice and skips another: 25 positions read
    but 24 distinct ids -> not complete (the skipped job must not read as gone)."""
    board = [{"id": 1000 + i, "name": "Engineer"} for i in range(25)]
    # page 10 repeats job 9; job 19 is never served
    _route(monkeypatch, _netflix_api(board, pages={10: board[9:19], 20: board[20:25]}))
    scrape_eightfold("netflix.com", "Netflix", KW)
    rep = _take_board_report()
    assert rep["ok"] and not rep["complete"] and rep["total_count"] == 24


def test_bamboohr_and_comeet(monkeypatch):
    _route(monkeypatch, lambda url, kw: FakeResp({"meta": {"totalCount": 2}, "result": [
        {"id": 7, "jobOpeningName": "Video Producer"}, {"id": 8, "jobOpeningName": "Nurse"}]}))
    assert len(scrape_bamboohr("acme", "Acme", KW)) == 1
    rep = _take_board_report()
    assert rep["complete"] and _links(rep) == ["https://acme.bamboohr.com/careers/7",
                                                "https://acme.bamboohr.com/careers/8"]
    _route(monkeypatch, lambda url, kw: FakeResp({"meta": {"totalCount": 9}, "result": [
        {"id": 7, "jobOpeningName": "Video Producer"}]}))
    scrape_bamboohr("acme", "Acme", KW)
    assert not _take_board_report()["complete"]

    _route(monkeypatch, lambda url, kw: FakeResp([
        {"name": "Video Producer", "url_active_page": "https://c.co/1"},
        {"name": "Video Producer", "url_active_page": "https://c.co/2", "is_internal": True},
        {"name": "Chef", "url_comeet_hosted_page": "https://c.co/3"},
    ]))
    assert len(scrape_comeet("1.1:tok", "C", KW)) == 1
    rep = _take_board_report()
    assert rep["complete"] and rep["links"] == [("https://c.co/1", "Video Producer"),
                                                 ("https://c.co/3", "Chef")]


def test_keyword_search_platforms_never_report(monkeypatch):
    """Workday / Apple are seed keyword searches, Teamtailor a paginated HTML
    page: absence there proves nothing, so they never report."""
    _no_network(monkeypatch)
    scrape_workday("acme|wd1|External", "Acme", KW)
    assert _take_board_report() is None
    scrape_apple("en-us", "Apple", KW)
    assert _take_board_report() is None
    scrape_teamtailor("t", "T", KW)
    assert _take_board_report() is None


# ── scrape_ats_boards recording ──────────────────────────────────────────────
class RecordingDB:
    def __init__(self, raise_exc=None):
        self.calls = []
        self.raise_exc = raise_exc

    def record_board_fetch(self, platform, token, company, **kw):
        if self.raise_exc:
            raise self.raise_exc
        self.calls.append((platform, token, company, kw))
        return len(self.calls)


def _fake_http(monkeypatch):
    def get(url, kw):
        if "greenhouse" in url:
            return FakeResp({"jobs": _gh_jobs()})
        if "lever" in url:
            return FakeResp([{"text": "Senior Video Producer", "hostedUrl": "https://jobs.lever.co/l/1",
                              "categories": {"location": "Bellevue, WA"}},
                             {"text": "Chef", "hostedUrl": "https://jobs.lever.co/l/2"}])
        raise requests.exceptions.ConnectionError("unexpected")
    _route(monkeypatch, get)


COMPANIES = [
    {"company": "Acme", "ats": "greenhouse", "token": "acme"},
    {"company": "Lev", "ats": "lever", "token": "lev"},
    {"company": "Day", "ats": "workday", "token": "x|y|z"},
]


def test_scrape_ats_boards_records_each_full_list_board_with_run_id(monkeypatch):
    _fake_http(monkeypatch)
    monkeypatch.setattr("job_agent.scraper_workday.scrape_workday", lambda t, c, k: [])
    db = RecordingDB()
    scrape_ats_boards(COMPANIES, KW, db=db, run_id="run-1")
    assert [(p, t, c) for p, t, c, _ in db.calls] == [("greenhouse", "acme", "Acme"), ("lever", "lev", "Lev")]
    kw = db.calls[0][3]
    assert kw["run_id"] == "run-1" and kw["ok"] and kw["complete"] and kw["total_count"] == 3
    assert len(kw["links"]) == 3


def test_a_mocked_scraper_without_a_report_records_nothing(monkeypatch):
    monkeypatch.setattr(ats_scraper, "scrape_greenhouse",
                        lambda t, c, k: [{"title": "Video Producer", "job_url": "u"}])
    # a stale report left by an earlier board must not leak onto this one
    ats_scraper.report_board_fetch("lever", "old", ok=True, complete=True, total_count=1, links=["x"])
    db = RecordingDB()
    df = scrape_ats_boards([{"company": "Acme", "ats": "greenhouse", "token": "acme"}], KW, db=db)
    assert db.calls == [] and len(df) == 1


def test_record_failure_never_stops_the_scan_but_the_phase_timeout_does(monkeypatch):
    _fake_http(monkeypatch)
    monkeypatch.setattr("job_agent.scraper_workday.scrape_workday", lambda t, c, k: [])
    df = scrape_ats_boards(COMPANIES, KW, db=RecordingDB(raise_exc=RuntimeError("db gone")))
    assert len(df) == 2

    with pytest.raises(main._ScrapeTimeout):
        scrape_ats_boards(COMPANIES, KW, db=RecordingDB(raise_exc=main._ScrapeTimeout("ATS phase")))


def test_recording_changes_nothing_scraped_deduped_or_filtered(monkeypatch, tmp_db):
    """Same boards, same HTTP: the job list, the cross-source dedup and the
    pre-filter come out identical with and without recording (real DB)."""
    from job_agent.utils import dedup_scraped_jobs, process_and_filter_jobs
    fresh = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
    gh = _gh_jobs() + [
        {"title": "Video Producer", "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/4",
         "location": {"name": "Seattle, WA"}, "first_published": fresh},
        {"title": "Senior Video Producer", "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/5",
         "location": {"name": "London, UK"}, "first_published": fresh},
    ]
    gh[0]["location"] = {"name": "Bellevue, WA"}
    gh[0]["first_published"] = fresh

    def get(url, kw):
        if "greenhouse" in url:
            return FakeResp({"jobs": gh})
        return FakeResp([{"text": "Senior Video Producer", "hostedUrl": "https://jobs.lever.co/l/1",
                          "categories": {"location": "Bellevue, WA"}, "createdAt": fresh}])
    _route(monkeypatch, get)
    companies = COMPANIES[:2]
    existing_links = {"https://job-boards.greenhouse.io/acme/jobs/4"}

    def run(db):
        df = scrape_ats_boards(companies, KW, db=db, run_id="r")
        records = df.to_dict("records")
        deduped = dedup_scraped_jobs(records, set(existing_links), {})
        filtered = process_and_filter_jobs(pd.DataFrame(deduped))
        return df, deduped, filtered

    df0, dd0, f0 = run(None)
    df1, dd1, f1 = run(tmp_db)
    pd.testing.assert_frame_equal(df0, df1)
    assert dd0 == dd1
    pd.testing.assert_frame_equal(f0.reset_index(drop=True), f1.reset_index(drop=True))
    assert len(dd0) < len(df0)             # the known link was deduped either way
    assert "https://job-boards.greenhouse.io/acme/jobs/4" not in {j["job_url"] for j in dd0}
    # and the recording really happened
    conn = tmp_db._get_thread_conn()
    assert conn.execute("SELECT COUNT(*) FROM board_fetches").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM board_sightings").fetchone()[0] == 6


# ── main wiring ──────────────────────────────────────────────────────────────
class WiringDB:
    def __init__(self):
        self.refreshed = 0

    def _get_thread_conn(self):
        raise RuntimeError("no baseline in this test")

    def refresh_listing_status(self, job_ids=None):
        self.refreshed += 1
        return {"listed": 0, "gone": 0, "unknown": 0}


def test_run_search_session_passes_db_and_run_id_and_refreshes(monkeypatch):
    seen = {}

    def fake_timeout(func, args=(), kwargs=None, **kw):
        seen["func"], seen["kwargs"] = func, kwargs
        return pd.DataFrame()

    monkeypatch.setattr(main, "_run_with_timeout", fake_timeout)
    monkeypatch.setattr(main, "load_profile_text", lambda p: "profile")
    monkeypatch.setattr(main, "load_proxies", lambda: None)
    # the remote / secondary-metro jobspy passes run whatever site_list says — keep them off
    monkeypatch.setattr(main, "REMOTE_SECOND_PASS", False)
    monkeypatch.setattr(main, "SECONDARY_SECOND_PASS", False)
    monkeypatch.setattr(main, "_parallel_jobspy_pass",
                        lambda *a, **kw: pytest.fail("no jobspy pass in this test"))
    db = WiringDB()
    session = main.SessionManager()
    out = main.run_search_session("r.pdf", "cv.docx", "m", "g", site_list=["ats_boards"],
                                  send_email=False, session=session, db=db)
    assert out == []
    assert seen["func"] is scrape_ats_boards
    # The ATS calls also carry a deadline (their own budget, minus 30 s).
    assert seen["kwargs"]["db"] is db and seen["kwargs"]["run_id"] == session.id
    assert set(seen["kwargs"]) == {"db", "run_id", "deadline"}
    assert db.refreshed == 1


def test_refresh_failure_is_non_fatal(caplog):
    class Bad:
        def refresh_listing_status(self):
            raise RuntimeError("locked")
    main._refresh_listing_status(Bad(), main.logging.getLogger("t"), "test")   # no raise


# ── Strong-job steps ─────────────────────────────────────────────────────────
class HealthDB:
    def __init__(self, ignored=()):
        self.ignored = {c.lower() for c in ignored}

    def is_company_ignored(self, company):
        return company.lower() in self.ignored


def _strong(companies):
    return pd.DataFrame([
        {"company": c, "title": f"Producer {i}", "tier": "strong", "match_level": "HIGH_MATCH",
         "confidence": 0.9 - i * 0.01, "job_url": f"https://ex.com/{i}"}
        for i, c in enumerate(companies)])


@pytest.fixture
def health_env(monkeypatch):
    calls = []
    fresh = set()
    monkeypatch.setattr("job_agent.config.AUTO_HIRING_HEALTH", True)
    monkeypatch.setattr("job_agent.config.USE_PERPLEXITY", True)
    monkeypatch.setattr("job_agent.config.AUTO_HIRING_HEALTH_MAX_COMPANIES", 20)
    monkeypatch.setattr("job_agent.config.AUTO_HIRING_HEALTH_BUDGET_SEC", 480)
    monkeypatch.setattr(perplexity_utils, "hiring_health_is_fresh", lambda c, db: c in fresh)

    def fake_health(company, db=None, job_title=None, cached_only=False):
        calls.append((company, job_title))
        return "facts"
    monkeypatch.setattr(perplexity_utils, "get_hiring_health", fake_health)
    return calls, fresh


def test_auto_hiring_health_one_call_per_company_skips_fresh_and_ignored(health_env):
    calls, fresh = health_env
    fresh.add("Fresh Co")
    df = _strong(["Harbor Lights Media", "harbor lights media ", "Fresh Co", "Ignored Co", "Unknown", "",
                  "Kestrel Bay Studios"])
    before = df.copy()
    out = main._auto_hiring_health(df, HealthDB(ignored=["Ignored Co"]), main.logging.getLogger("t"))
    assert calls == [("Harbor Lights Media", None), ("Kestrel Bay Studios", None)]   # company only, no job title
    assert out["checked"] == 2 and out["cached"] == 1 and out["skipped"] == 3
    pd.testing.assert_frame_equal(df, before)                # tier/confidence/order untouched


def test_auto_hiring_health_highest_confidence_first_and_capped(health_env, monkeypatch):
    calls, _ = health_env
    monkeypatch.setattr("job_agent.config.AUTO_HIRING_HEALTH_MAX_COMPANIES", 2)
    df = _strong(["Low", "Mid", "High"])
    df["confidence"] = [0.5, 0.7, 0.95]
    out = main._auto_hiring_health(df, HealthDB(), main.logging.getLogger("t"))
    assert [c for c, _ in calls] == ["High", "Mid"]
    assert out["checked"] == 2 and out["skipped"] == 1


def test_auto_hiring_health_stops_starting_calls_after_the_budget(health_env, monkeypatch):
    calls, _ = health_env
    clock = iter([0.0, 10.0, 500.0, 600.0, 700.0])
    monkeypatch.setattr(main, "_clock", lambda: next(clock))
    out = main._auto_hiring_health(_strong(["A", "B", "C", "D"]), HealthDB(), main.logging.getLogger("t"))
    assert [c for c, _ in calls] == ["A"]
    assert out["checked"] == 1 and out["skipped"] == 3


def test_auto_hiring_health_switches(health_env, monkeypatch):
    calls, _ = health_env
    monkeypatch.setattr("job_agent.config.AUTO_HIRING_HEALTH", False)
    main._auto_hiring_health(_strong(["A"]), HealthDB(), main.logging.getLogger("t"))
    monkeypatch.setattr("job_agent.config.AUTO_HIRING_HEALTH", True)
    monkeypatch.setattr("job_agent.config.USE_PERPLEXITY", False)
    main._auto_hiring_health(_strong(["A"]), HealthDB(), main.logging.getLogger("t"))
    assert calls == []


def test_auto_hiring_health_survives_a_failing_company(health_env, monkeypatch):
    calls, _ = health_env

    def flaky(company, db=None, **kw):
        calls.append((company, None))
        if company == "A":
            raise RuntimeError("perplexity down")
        return "ok"
    monkeypatch.setattr(perplexity_utils, "get_hiring_health", flaky)
    out = main._auto_hiring_health(_strong(["A", "B"]), HealthDB(), main.logging.getLogger("t"))
    assert [c for c, _ in calls] == ["A", "B"] and out["failed"] == 1 and out["checked"] == 1


def test_auto_hiring_health_honors_stop(health_env, monkeypatch):
    """Regression: Stop pressed during the Strong-job step kept researching
    every remaining company (up to 20 Perplexity calls, ~9 min)."""
    calls, _ = health_env
    flags = {"stop": False}

    def stop_after_first(company, db=None, **kw):
        calls.append((company, None))
        flags["stop"] = True
        return "facts"
    monkeypatch.setattr(perplexity_utils, "get_hiring_health", stop_after_first)
    out = main._auto_hiring_health(_strong(["A", "B", "C"]), HealthDB(), main.logging.getLogger("t"),
                                   should_stop=lambda: flags["stop"])
    assert [c for c, _ in calls] == ["A"]
    assert out["checked"] == 1 and out["skipped"] == 2


def test_process_found_jobs_skips_strong_steps_when_the_ui_stop_was_pressed(monkeypatch):
    """Regression: the UI search passes no control_flags — its Stop only sets
    session.check_stop(), which the Phase 27c gate never read."""
    graded = pd.DataFrame([
        {"title": "Video Producer", "company": "A", "location": "Bellevue, WA", "job_url": "https://ex.com/a",
         "match_level": "HIGH_MATCH", "confidence": 0.8, "tier": "strong", "gpt_relevance": True}])
    monkeypatch.setattr(main, "load_profile_text", lambda p: "profile")
    monkeypatch.setattr("job_agent.utils.process_and_filter_jobs", lambda df, **kw: df)
    monkeypatch.setattr("job_agent.config.AUTO_ENRICH_MATCHES", False)
    monkeypatch.setattr(main, "AUTO_GENERATE_CV_CL", False)
    from job_agent.session import SessionManager
    sm = SessionManager()

    def grade_then_stop(df, *a, **kw):
        sm.request_stop()                      # Stop pressed while the cascade ran
        return graded.copy()
    monkeypatch.setattr(main, "batch_job_relevance_check", grade_then_stop)
    steps = []
    monkeypatch.setattr(main, "_auto_hiring_health", lambda *a, **kw: steps.append("health"))
    monkeypatch.setattr(main, "_auto_posting_quality", lambda *a, **kw: steps.append("quality"))

    class DB:
        def get_suppressed_locations(self):
            return []

        def save_job_with_relevance(self, job, **kw):
            return 1

        def refresh_listing_status(self, job_ids=None):
            return {}

    main.process_found_jobs(graded.to_dict("records"), "r.pdf", "cv.docx", "m", "g",
                            db=DB(), session=sm, send_email=False)
    assert steps == []


def test_auto_posting_quality_passes_strong_ids_with_caps(monkeypatch):
    got = {}
    fake_pc = types.SimpleNamespace(
        run_quality_reads=lambda db, ids, *, cap, budget_sec, should_stop=None: got.update(
            ids=ids, cap=cap, budget=budget_sec)
        or {"fresh": len(ids), "cached": 0, "skipped": 0, "failed": 0})
    monkeypatch.setitem(sys.modules, "job_agent.posting_check", fake_pc)
    import job_agent
    monkeypatch.setattr(job_agent, "posting_check", fake_pc, raising=False)

    class DB:
        def get_job_id_by_link(self, link):
            return {"https://ex.com/0": 10, "https://ex.com/1": 11, "https://ex.com/l": 12}.get(link)

    df = _strong(["A", "B", "C"])
    df.loc[2, "job_url"] = float("nan")
    df["link"] = [None, None, "https://ex.com/l"]
    out = main._auto_posting_quality(df, DB(), main.logging.getLogger("t"))
    from job_agent.config import POSTING_QUALITY_BUDGET_SEC, POSTING_QUALITY_MAX_PER_RUN
    assert got == {"ids": [10, 11, 12], "cap": POSTING_QUALITY_MAX_PER_RUN,
                   "budget": POSTING_QUALITY_BUDGET_SEC}
    assert out["fresh"] == 3


# ── perplexity_utils ─────────────────────────────────────────────────────────
def test_hiring_health_query_has_the_creative_cuts_topic_and_nothing_about_her():
    q = perplexity_utils._hiring_health_query("Acme")
    assert "on six topics" in q
    assert ("6. Layoffs or restructuring in content, video, creative, production, marketing "
            "or events teams — date, scale, teams affected.") in q
    assert "- **Creative team cuts:**" in q
    low = q.lower()
    home_city = HOME.split(",")[0].strip().lower()
    for private in (FIRST_NAME.lower(), LAST_NAME.lower(), "candidate", "resume", home_city,
                    "profile", "her", "she"):
        assert not re.search(rf"\b{re.escape(private)}\b", low), private


def test_creative_cuts_no_signal_label_is_a_topic_label():
    assert perplexity_utils._is_topic_label("- **Creative team cuts:** ")
    assert perplexity_utils._is_topic_label("- **Layoffs or restructuring in creative and video teams:** ")
    assert perplexity_utils._is_topic_label("- **Events team cuts:** ")


class StatsDB:
    db_path = "stats-test"

    def __init__(self, rows):
        import sqlite3
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE scrapes (id INTEGER PRIMARY KEY, title TEXT, company TEXT, "
                          "link TEXT, location TEXT, created_at TEXT)")
        self.conn.executemany("INSERT INTO scrapes (title, company, link, location, created_at) "
                              "VALUES (?,?,?,?,?)", rows)

    def _get_thread_conn(self):
        return self.conn


def test_scanner_stats_feed_the_unchanged_signal(monkeypatch):
    monkeypatch.setattr(perplexity_utils, "_scanner_index", {"db_path": None, "built_at": 0.0, "ids": {}})
    d = lambda n: (date.today() - timedelta(days=n)).isoformat() + " 10:00:00"
    db = StatsDB([
        ("Video Producer", "Acme", "https://boards.greenhouse.io/acme/1", "Bellevue", d(40)),
        ("Video Producer", "Acme", "https://boards.greenhouse.io/acme/2", "Bellevue", d(20)),
        ("Video Producer", "Acme", "https://www.linkedin.com/jobs/3", "Bellevue", d(19)),
        ("Editor", "Acme", "https://boards.greenhouse.io/acme/4", "Bellevue", d(5)),
    ])
    stats = perplexity_utils.get_scanner_stats("Acme", db, "video producer")
    assert stats["listings"] == 4 and stats["distinct_titles"] == 2 and stats["window_days"] == 90
    assert stats["title"] == {"listings": 3, "boards": 2, "first_seen": d(40)[:10],
                              "latest_seen": d(19)[:10], "relisted": 1}
    assert perplexity_utils.get_scanner_stats("Acme", db)["title"] is None
    assert perplexity_utils.get_scanner_stats("Acme", db, "Chef")["title"]["listings"] == 0
    assert perplexity_utils.get_scanner_stats("Nobody", db) is None
    sig = perplexity_utils.get_scanner_signal("Acme", db, "Video Producer")
    assert "re-listed 1 time (a new listing on the same job board" in sig
    assert "3 listings on 2 job boards" in sig


def test_hiring_health_is_fresh():
    class DB:
        def __init__(self, row):
            self.row = row
            self.keys = []

        def get_company_intel(self, key):
            self.keys.append(key)
            return self.row

    today = date.today()
    db = DB({"company_profile": "- facts", "last_checked": today.isoformat()})
    assert perplexity_utils.hiring_health_is_fresh(" Acme ", db)
    assert db.keys == ["__hiring_health__Acme"]
    old = (today - timedelta(days=perplexity_utils.HIRING_HEALTH_TTL_DAYS)).isoformat()
    assert not perplexity_utils.hiring_health_is_fresh("Acme", DB({"company_profile": "x", "last_checked": old}))
    assert not perplexity_utils.hiring_health_is_fresh("Acme", DB({"company_profile": None,
                                                                   "last_checked": today.isoformat()}))
    assert not perplexity_utils.hiring_health_is_fresh("Acme", DB(None))
    assert not perplexity_utils.hiring_health_is_fresh("", db)


# ── digest ───────────────────────────────────────────────────────────────────
class DigestDB:
    def __init__(self, gone=(), glances=None):
        self.gone = set(gone)
        self.ids = {"https://ex.com/1": 1, "https://ex.com/2": 2, "https://ex.com/3": 3}
        self.glances = glances or {}

    def get_gone_links(self, links):
        return {l for l in links if l in self.gone}

    def get_job_id_by_link(self, link):
        return self.ids.get(link)

    def get_followups_due(self, days):
        return []

    def get_company_intel(self, key):
        return None


def _digest_jobs():
    return pd.DataFrame([
        {"title": "Senior Video Producer", "company": "Acme", "location": "Bellevue, WA",
         "job_url": "https://ex.com/1", "tier": "strong", "match_level": "HIGH_MATCH", "confidence": 0.9,
         "gpt_reasoning": "fit", "comp_text": "$95K–$130K/yr", "employment_type": "Full-time"},
        {"title": "Studio Producer", "company": "Gone Co", "location": "Redmond, WA",
         "job_url": "https://ex.com/2", "tier": "strong", "match_level": "HIGH_MATCH", "confidence": 0.85,
         "gpt_reasoning": "fit", "comp_text": None, "employment_type": None},
        {"title": "Content Producer", "company": "Media Co", "location": "Remote",
         "job_url": "https://ex.com/3", "tier": "worth_a_look", "match_level": "WORTH_A_LOOK",
         "confidence": 0.5, "gpt_reasoning": "Adjacent", "comp_text": None, "employment_type": None},
    ])


def _send(monkeypatch, df, db=None):
    captured = {}

    class FakeSMTP:
        def login(self, *a):
            pass

        def sendmail(self, sender, recips, msg_str):
            captured["msg"] = msg_str
            captured["subject"] = str(make_header(decode_header(email.message_from_string(msg_str)["Subject"])))

        def quit(self):
            pass

    monkeypatch.setenv("EMAIL_SENDER", "digest@test.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "secret")
    monkeypatch.setattr("job_agent.postprocess.smtplib.SMTP_SSL", lambda *a, **kw: FakeSMTP())
    monkeypatch.setattr(postprocess, "_is_morning_digest", lambda now: False)
    postprocess.send_email_with_results(None, "candidate@test.com", df, db=db)
    msg = email.message_from_string(captured["msg"])
    part = next(p for p in msg.walk() if p.get_content_type() == "text/html")
    return part.get_payload(decode=True).decode("ascii"), captured["subject"]


@pytest.fixture
def fake_glance(monkeypatch):
    calls = []

    def digest_glance(db, job_id):
        calls.append(job_id)
        return db.glances.get(job_id)
    fake_pc = types.SimpleNamespace(digest_glance=digest_glance)
    monkeypatch.setitem(sys.modules, "job_agent.posting_check", fake_pc)
    import job_agent
    monkeypatch.setattr(job_agent, "posting_check", fake_pc, raising=False)
    return calls


def test_gone_rows_leave_the_digest_and_its_counts(monkeypatch, fake_glance):
    html, subject = _send(monkeypatch, _digest_jobs(), db=DigestDB(gone={"https://ex.com/2"}))
    assert "Studio Producer" not in html and "Gone Co" not in html
    assert "Senior Video Producer" in html
    assert "1 strong + 1 worth a look" in subject


def test_glance_line_on_every_card_escaped_with_cautions_and_sources(monkeypatch, fake_glance):
    """The posting check shows at a glance on every digest card, Strong and
    Worth-a-look alike."""
    db = DigestDB(glances={
        1: {"read": "check_first", "label": "Check before investing time",
            "reasons": ["re-listed 3 times", "layoffs reported 2026-06-12"],
            "items": [{"text": "Layoffs reported 2026-06-12 & <more>", "caution": True,
                       "url": "https://news.example/a?b=1&c=2"},
                      {"text": "People you know here: 3 contacts", "caution": False, "url": None}]},
        3: {"read": "looks_active", "label": "Looks active", "reasons": ["still on the board"],
            "items": [{"text": "On Media Co's job board", "caution": False, "url": None}]},
    })
    html, _ = _send(monkeypatch, _digest_jobs(), db=db)
    assert ("&#9888; Check before investing time: re-listed 3 times; layoffs reported 2026-06-12"
            in html)
    assert '<span style="color:#b45309;">Layoffs reported 2026-06-12 &amp; &lt;more&gt;</span>' in html
    assert 'href="https://news.example/a?b=1&amp;c=2"' in html
    assert "People you know here: 3 contacts" in html
    # Looks active: the label only, never its reasons; and the worth-a-look card has its line too
    assert "&#10003; Looks active</span> &middot; On Media Co&#x27;s job board" in html
    assert "still on the board" not in html
    assert fake_glance == [1, 2, 3]


def test_no_line_when_there_is_nothing_to_say(monkeypatch, fake_glance):
    db = DigestDB(glances={1: {"read": "not_enough_info", "label": "Not enough to go on",
                               "reasons": [], "items": []}})
    html, _ = _send(monkeypatch, _digest_jobs(), db=db)
    assert "Not enough to go on" not in html
    assert "line-height:1.45;" not in html


def test_digest_survives_a_failing_posting_check(monkeypatch, fake_glance):
    class Broken(DigestDB):
        def get_gone_links(self, links):
            raise RuntimeError("db locked")

        def get_job_id_by_link(self, link):
            raise RuntimeError("db locked")
    html, subject = _send(monkeypatch, _digest_jobs(), db=Broken())
    assert "Studio Producer" in html and "2 strong + 1 worth a look" in subject


def test_no_db_digest_is_unchanged(monkeypatch, fake_glance):
    html_none, subj_none = _send(monkeypatch, _digest_jobs(), db=None)
    html_quiet, subj_quiet = _send(monkeypatch, _digest_jobs(), db=DigestDB())
    assert subj_none == subj_quiet
    assert html_none == html_quiet
    assert fake_glance == [1, 2, 3]                  # every card is asked


def test_process_found_jobs_refreshes_and_runs_strong_steps_without_touching_the_verdicts(
        monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                       # the CSV export lands here
    graded = pd.DataFrame([
        {"title": "Video Producer", "company": "A", "location": "Bellevue, WA", "job_url": "https://ex.com/a",
         "match_level": "HIGH_MATCH", "confidence": 0.8, "tier": "strong", "gpt_relevance": True},
        {"title": "Content Producer", "company": "B", "location": "Remote", "job_url": "https://ex.com/b",
         "match_level": "WORTH_A_LOOK", "confidence": 0.6, "tier": "worth_a_look", "gpt_relevance": True},
        {"title": "Video Editor", "company": "C", "location": "Bellevue, WA", "job_url": "https://ex.com/c",
         "match_level": "HIGH_MATCH", "confidence": 0.9, "tier": "strong", "gpt_relevance": True},
    ])
    monkeypatch.setattr(main, "load_profile_text", lambda p: "profile")
    monkeypatch.setattr("job_agent.utils.process_and_filter_jobs", lambda df, **kw: df)
    monkeypatch.setattr(main, "batch_job_relevance_check", lambda df, *a, **kw: graded.copy())
    monkeypatch.setattr(main, "AUTO_GENERATE_CV_CL", False)
    monkeypatch.setattr("job_agent.config.AUTO_ENRICH_MATCHES", False)
    steps = {}
    monkeypatch.setattr(main, "_auto_hiring_health",
                        lambda df, db, lg, **kw: steps.setdefault("health", df[["company", "tier"]].copy()))
    monkeypatch.setattr(main, "_auto_posting_quality",
                        lambda df, db, lg, **kw: (_ for _ in ()).throw(RuntimeError("quality boom")))

    class DB:
        refreshed = 0
        saved = []

        def get_suppressed_locations(self):
            return []

        def save_job_with_relevance(self, job, **kw):
            self.saved.append((job["job_url"], kw["tier"], kw["confidence"], kw["match_level"]))
            return 1

        def save_job(self, job, **kw):
            return 1

        def refresh_listing_status(self, job_ids=None):
            DB.refreshed += 1
            return {}

    db = DB()
    out = main.process_found_jobs(graded.to_dict("records"), "r.pdf", "cv.docx", "m", "g",
                                  db=db, send_email=False)
    assert DB.refreshed == 1
    assert list(steps["health"]["company"]) == ["C", "A"]          # strong only
    assert set(steps["health"]["tier"]) == {"strong"}
    # the failing quality step did not stop the run, and verdicts/ordering are intact
    assert [(r["job_url"], r["tier"], r["confidence"], r["match_level"]) for r in out] == [
        ("https://ex.com/c", "strong", 0.9, "HIGH_MATCH"),
        ("https://ex.com/a", "strong", 0.8, "HIGH_MATCH"),
        ("https://ex.com/b", "worth_a_look", 0.6, "WORTH_A_LOOK"),
    ]
    assert sorted(db.saved) == sorted(
        (r["job_url"], r["tier"], r["confidence"], r["match_level"]) for r in graded.to_dict("records"))


def test_end_to_end_two_complete_fetches_without_the_posting_make_it_gone(monkeypatch, tmp_db):
    """Real DB: scrape -> save -> the board drops the posting -> one fetch is
    not enough -> a failed fetch is no evidence -> the second complete fetch
    makes it gone, and the digest leaves it out."""
    board = {"jobs": _gh_jobs(), "fail": False}
    # runs 8 hours apart: "gone" needs the misses to span POSTING_GONE_MIN_SPAN_HOURS
    clock = {"t": datetime.now(timezone.utc) - timedelta(days=2)}
    monkeypatch.setattr("job_agent.db_manager._utc_now",
                        lambda: clock["t"].strftime("%Y-%m-%d %H:%M:%S"))

    def scrape(run_id):
        clock["t"] += timedelta(hours=8)
        return scrape_ats_boards(companies, KW, db=tmp_db, run_id=run_id)

    def get(url, kw):
        if board["fail"]:
            return FakeResp(status=502)
        return FakeResp({"jobs": board["jobs"]})
    _route(monkeypatch, get)
    companies = [{"company": "Acme", "ats": "greenhouse", "token": "acme"}]
    df = scrape("r1")
    job = df.to_dict("records")[0]
    job_id = tmp_db.save_job_with_relevance(job, relevance=True, reasoning="fit", match_level="HIGH_MATCH",
                                            confidence=0.9, session_id="r1", tier="strong")
    assert tmp_db.get_listing_status(job_id)["status"] == "listed"

    board["jobs"] = _gh_jobs()[1:]                       # the matched posting leaves the board
    scrape("r2")
    assert tmp_db.get_listing_status(job_id)["reason"] == "missing_once"
    board["fail"] = True
    scrape("r3")
    assert tmp_db.get_listing_status(job_id)["reason"] == "missing_once"
    board["fail"] = False
    scrape("r4")
    assert tmp_db.get_listing_status(job_id)["status"] == "gone"
    tmp_db.refresh_listing_status()
    assert tmp_db.get_gone_links([job["job_url"]]) == {job["job_url"]}
    row = tmp_db._get_thread_conn().execute(
        "SELECT tier, confidence, match_level FROM scrapes WHERE id = ?", (job_id,)).fetchone()
    assert tuple(row) == ("strong", 0.9, "HIGH_MATCH")
