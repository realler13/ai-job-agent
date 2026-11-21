"""Phase 27c "Is this posting real?" — the API surface.

posting_check (the logic stream) is replaced by a fake on the api module, so
these tests pin only what api.py does: endpoint shapes, the feed annotation
that can never fail the feed, and hiring health spawned on Track (once per
company, never when fresh or without a key). No network, no model calls.
"""
import threading
from datetime import date
from types import SimpleNamespace

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from job_agent.db_manager import DBManager
    from job_agent import api
    db = DBManager(db_path=str(tmp_path / "posting_check_api.db"))
    monkeypatch.setattr(api, "_db", db)
    with api._hiring_health_lock:
        api._hiring_health_inflight.clear()
    _reset_feed_cache(api)
    # Feed chips are refreshed in a background thread in production; run the
    # refresh inline here so a request sees its own result. The async path has
    # its own tests below.
    monkeypatch.setattr(api, "_spawn_feed_refresh", api._refresh_feed_reads)
    return TestClient(api.app), db, api


def _reset_feed_cache(api):
    with api._feed_reads_lock:
        api._feed_reads.update(reads={}, attempted=set(), built_at=0.0, running=False)


def _fake_check(job_id, read="mixed"):
    return {
        "job_id": job_id, "company": "Harbor Lights Media", "read": read, "label": "Mixed signals",
        "reasons": ["posted 19 days ago"],
        "facts": [{"key": "posted", "text": "Posted 19 days ago", "source": "posting_text",
                   "url": None, "quotes": [], "caution": False}],
        "quality": None, "quality_available": False,
    }


def _install_fake(monkeypatch, api, **overrides):
    calls = {"build": [], "quality": [], "feed": []}

    def build_posting_check(db, job_id):
        calls["build"].append(job_id)
        return _fake_check(job_id)

    def run_quality_read(db, job_id):
        calls["quality"].append(job_id)
        return {"status": "fresh", "quality": {"tools": []}}

    def feed_reads(db, jobs):
        calls["feed"].append([j["id"] for j in jobs])
        return {j["id"]: {"read": "check_first", "label": "Check before investing time",
                          "reasons": ["re-listed 3 times"]} for j in jobs}

    fake = SimpleNamespace(build_posting_check=build_posting_check,
                           run_quality_read=run_quality_read, feed_reads=feed_reads)
    for k, v in overrides.items():
        setattr(fake, k, v)
    monkeypatch.setattr(api, "posting_check", fake)
    return calls


def _save(db, n=1, company="Harbor Lights Media", tier="strong", confidence=0.9):
    return db.save_job_with_relevance(
        {"title": f"Video Producer {n}", "company": company, "location": "Bellevue, WA",
         "job_url": f"https://boards.greenhouse.io/harborlights/jobs/{1000 + n}",
         "description": "Make videos.", "date_posted": date.today().isoformat()},
        relevance="relevant", match_level="HIGH_MATCH", confidence=confidence,
        reasoning="fits", tier=tier,
    )


def _row(db, job_id):
    conn = db._get_thread_conn()
    return conn.execute(
        "SELECT tier, confidence, match_level, COALESCE(reviewed, 0) FROM scrapes WHERE id = ?",
        (job_id,)).fetchone()


# ── GET /api/jobs/{id}/posting-check ─────────────────────────────────────────
def test_get_returns_the_check(client, monkeypatch):
    c, db, api = client
    calls = _install_fake(monkeypatch, api)
    jid = _save(db)
    r = c.get(f"/api/jobs/{jid}/posting-check")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == jid and body["read"] == "mixed"
    assert body["facts"][0]["source"] == "posting_text"
    assert body["quality_available"] is False
    assert calls["build"] == [jid] and calls["quality"] == []   # GET never reads the posting


def test_get_404_when_the_job_is_unknown(client, monkeypatch):
    c, db, api = client
    _install_fake(monkeypatch, api, build_posting_check=lambda db, job_id: None)
    assert c.get("/api/jobs/999999/posting-check").status_code == 404


def test_get_500_when_the_check_raises(client, monkeypatch):
    c, db, api = client

    def boom(db, job_id):
        raise RuntimeError("bad row")
    _install_fake(monkeypatch, api, build_posting_check=boom)
    jid = _save(db)
    assert c.get(f"/api/jobs/{jid}/posting-check").status_code == 500


# ── POST /api/jobs/{id}/posting-check/quality ────────────────────────────────
@pytest.mark.parametrize("status", ["fresh", "cached", "no_text", "failed"])
def test_quality_passes_the_status_through_with_the_refreshed_check(client, monkeypatch, status):
    c, db, api = client
    calls = _install_fake(monkeypatch, api,
                          run_quality_read=lambda db, job_id: {"status": status, "quality": None})
    jid = _save(db)
    r = c.post(f"/api/jobs/{jid}/posting-check/quality")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == status
    assert body["check"]["job_id"] == jid
    assert calls["build"] == [jid]


def test_quality_404_for_an_unknown_job_and_no_read(client, monkeypatch):
    c, db, api = client
    calls = _install_fake(monkeypatch, api)
    assert c.post("/api/jobs/999999/posting-check/quality").status_code == 404
    assert calls["quality"] == []


def test_quality_read_that_raises_reports_failed(client, monkeypatch):
    c, db, api = client

    def boom(db, job_id):
        raise RuntimeError("model down")
    _install_fake(monkeypatch, api, run_quality_read=boom)
    jid = _save(db)
    r = c.post(f"/api/jobs/{jid}/posting-check/quality")
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    assert r.json()["check"]["job_id"] == jid


# ── GET /api/jobs/unreviewed ─────────────────────────────────────────────────
def test_feed_rows_carry_posting_read(client, monkeypatch):
    c, db, api = client
    calls = _install_fake(monkeypatch, api)
    a, b = _save(db, 1), _save(db, 2, tier="worth_a_look")
    rows = c.get("/api/jobs/unreviewed").json()
    assert {r["id"] for r in rows} == {a, b}
    for r in rows:
        assert r["posting_read"]["read"] == "check_first"
        assert r["posting_read"]["reasons"] == ["re-listed 3 times"]
    assert len(calls["feed"]) == 1          # one batched call for the whole feed
    assert sorted(calls["feed"][0]) == sorted([a, b])


def test_feed_survives_a_raising_feed_reads(client, monkeypatch):
    c, db, api = client

    def boom(db, jobs):
        raise RuntimeError("scanner index broke")
    _install_fake(monkeypatch, api, feed_reads=boom)
    a = _save(db)
    r = c.get("/api/jobs/unreviewed")
    assert r.status_code == 200
    rows = r.json()
    assert [x["id"] for x in rows] == [a]
    assert rows[0]["posting_read"] is None


def test_feed_rows_without_a_read_get_null(client, monkeypatch):
    c, db, api = client
    a, b = _save(db, 1), _save(db, 2)
    _install_fake(monkeypatch, api, feed_reads=lambda db, jobs: {
        a: {"read": "mixed", "label": "Mixed signals", "reasons": ["posted 20 days ago"]}})
    rows = {x["id"]: x for x in c.get("/api/jobs/unreviewed").json()}
    assert rows[a]["posting_read"]["read"] == "mixed"
    assert rows[b]["posting_read"] is None


def test_feed_order_and_job_fields_do_not_depend_on_the_read(client, monkeypatch):
    """Facts, never a score: the read must not move, hide or regrade a row."""
    c, db, api = client
    ids = [_save(db, 1, confidence=0.95), _save(db, 2, confidence=0.7),
           _save(db, 3, tier="worth_a_look", confidence=0.6)]
    before = {i: _row(db, i) for i in ids}
    _install_fake(monkeypatch, api, feed_reads=lambda db, jobs: {})
    plain = [x["id"] for x in c.get("/api/jobs/unreviewed").json()]
    _install_fake(monkeypatch, api, feed_reads=lambda db, jobs: {
        ids[0]: {"read": "check_first", "label": "Check before investing time",
                 "reasons": ["no longer on the company's job board"]}})
    flagged = c.get("/api/jobs/unreviewed").json()
    assert [x["id"] for x in flagged] == plain
    for x in flagged:
        assert x["tier"] == before[x["id"]][0]
        assert x["confidence"] == before[x["id"]][1]
    c.get(f"/api/jobs/{ids[0]}/posting-check")
    c.post(f"/api/jobs/{ids[0]}/posting-check/quality")
    assert {i: _row(db, i) for i in ids} == before


def test_feed_leaves_out_gone_jobs(client, monkeypatch):
    c, db, api = client
    _install_fake(monkeypatch, api, feed_reads=lambda db, jobs: {})
    keep, gone = _save(db, 1), _save(db, 2)
    conn = db._get_thread_conn()
    conn.execute("UPDATE scrapes SET listing_status = 'gone', listing_gone_since = "
                 "'2026-10-02 12:00:00' WHERE id = ?", (gone,))
    conn.commit()
    assert [x["id"] for x in c.get("/api/jobs/unreviewed").json()] == [keep]


def test_feed_never_waits_for_the_posting_reads(client, monkeypatch):
    """The cold scanner index costs seconds; the feed must answer from the
    cache at once and pick the reads up on a later load."""
    c, db, api = client
    monkeypatch.setattr(api, "_spawn_feed_refresh", _thread_spawn(api))   # the real, threaded path
    release = threading.Event()
    calls = []

    def slow_feed_reads(db, jobs):
        calls.append([j["id"] for j in jobs])
        release.wait(5)
        return {j["id"]: {"read": "mixed", "label": "Mixed signals", "reasons": ["x"]} for j in jobs}
    _install_fake(monkeypatch, api, feed_reads=slow_feed_reads)
    a = _save(db)
    rows = c.get("/api/jobs/unreviewed").json()          # returns while the read is blocked
    assert rows[0]["posting_read"] is None
    release.set()
    for t in threading.enumerate():
        if t.name == "feed-posting-reads":
            t.join(5)
    rows = c.get("/api/jobs/unreviewed").json()
    assert rows[0]["posting_read"]["read"] == "mixed"
    assert len(calls) == 1                                # the second load used the cache


def _thread_spawn(api):
    def spawn(rows):
        threading.Thread(target=api._refresh_feed_reads, args=(rows,), daemon=True,
                         name="feed-posting-reads").start()
    return spawn


def test_feed_reads_are_cached_and_refreshed_for_new_rows(client, monkeypatch):
    c, db, api = client
    calls = _install_fake(monkeypatch, api)
    a = _save(db, 1)
    c.get("/api/jobs/unreviewed")
    c.get("/api/jobs/unreviewed")
    assert len(calls["feed"]) == 1                        # inside the TTL: cache only
    b = _save(db, 2)
    rows = {x["id"]: x for x in c.get("/api/jobs/unreviewed").json()}
    assert len(calls["feed"]) == 2                        # a row never tried: refresh
    assert rows[b]["posting_read"]["read"] == "check_first"
    with api._feed_reads_lock:
        api._feed_reads["built_at"] -= api.FEED_READS_TTL_SEC + 1
    c.get("/api/jobs/unreviewed")
    assert len(calls["feed"]) == 3                        # past the TTL: refresh


def test_a_failing_feed_read_is_not_retried_on_every_load(client, monkeypatch):
    c, db, api = client
    n = []

    def boom(db, jobs):
        n.append(1)
        raise RuntimeError("scanner index broke")
    _install_fake(monkeypatch, api, feed_reads=boom)
    _save(db)
    for _ in range(3):
        assert c.get("/api/jobs/unreviewed").status_code == 200
    assert len(n) == 1


# ── POST /api/jobs/{id}/track → hiring health in the background ──────────────
@pytest.fixture
def spawn_env(client, monkeypatch):
    """USE_PERPLEXITY on, nothing cached, and a thread target that records the
    call and HOLDS the in-flight slot (like a still-running research call)."""
    c, db, api = client
    from job_agent import config, perplexity_utils
    monkeypatch.setattr(config, "USE_PERPLEXITY", True)
    fresh_calls = []

    def not_fresh(company, db_):
        fresh_calls.append(company)
        return False
    monkeypatch.setattr(perplexity_utils, "hiring_health_is_fresh", not_fresh, raising=False)
    started = []
    done = threading.Event()

    def fake_run(company, title, key):
        started.append((company, title, key))
        done.set()
    monkeypatch.setattr(api, "_run_hiring_health", fake_run)
    return SimpleNamespace(c=c, db=db, api=api, started=started, done=done,
                           fresh_calls=fresh_calls, perplexity_utils=perplexity_utils,
                           config=config)


def test_track_spawns_hiring_health_once_per_company(spawn_env):
    e = spawn_env
    a = _save(e.db, 1, company="Harbor Lights Media")
    b = _save(e.db, 2, company="  harbor lights media ")
    r = e.c.post(f"/api/jobs/{a}/track")
    assert r.status_code == 200
    assert r.json()["status"] == "tracked"
    assert r.json()["hiring_health_spawned"] is True
    assert e.done.wait(5)
    assert e.started == [("Harbor Lights Media", "Video Producer 1", "harbor lights media")]
    # The first run still holds the slot → the same company does not run twice.
    r2 = e.c.post(f"/api/jobs/{b}/track")
    assert r2.status_code == 200 and r2.json()["hiring_health_spawned"] is False
    assert len(e.started) == 1


def test_track_spawns_again_after_the_slot_is_released(spawn_env):
    e = spawn_env
    a, b = _save(e.db, 1, company="Harbor Lights Media"), _save(e.db, 2, company="Harbor Lights Media")
    assert e.c.post(f"/api/jobs/{a}/track").json()["hiring_health_spawned"] is True
    with e.api._hiring_health_lock:
        e.api._hiring_health_inflight.clear()
    assert e.c.post(f"/api/jobs/{b}/track").json()["hiring_health_spawned"] is True


def test_track_does_not_spawn_when_research_is_fresh(spawn_env, monkeypatch):
    e = spawn_env
    monkeypatch.setattr(e.perplexity_utils, "hiring_health_is_fresh",
                        lambda company, db: True, raising=False)
    a = _save(e.db, 1)
    r = e.c.post(f"/api/jobs/{a}/track")
    assert r.status_code == 200 and r.json()["hiring_health_spawned"] is False
    assert e.started == []


def test_track_does_not_spawn_without_a_perplexity_key(spawn_env, monkeypatch):
    e = spawn_env
    monkeypatch.setattr(e.config, "USE_PERPLEXITY", False)
    a = _save(e.db, 1)
    r = e.c.post(f"/api/jobs/{a}/track")
    assert r.status_code == 200 and r.json()["hiring_health_spawned"] is False
    assert e.started == [] and e.fresh_calls == []


def test_track_skips_unknown_company(spawn_env):
    e = spawn_env
    a = _save(e.db, 1, company="Unknown")
    assert e.c.post(f"/api/jobs/{a}/track").json()["hiring_health_spawned"] is False
    assert e.started == []


def test_track_succeeds_when_the_spawn_check_raises(spawn_env, monkeypatch):
    e = spawn_env

    def boom(company, db):
        raise RuntimeError("cache read failed")
    monkeypatch.setattr(e.perplexity_utils, "hiring_health_is_fresh", boom, raising=False)
    a = _save(e.db, 1)
    r = e.c.post(f"/api/jobs/{a}/track")
    assert r.status_code == 200
    assert r.json() == {"status": "tracked", "job_id": a, "hiring_health_spawned": False}
    assert a in [x["job_id"] for x in e.db.get_applications()]


def test_run_hiring_health_calls_research_and_releases_the_slot(client, monkeypatch):
    c, db, api = client
    from job_agent import perplexity_utils
    seen = []

    def fake_health(company, db=None, job_title=None, **kw):
        seen.append((company, job_title, db))
        raise RuntimeError("deadline")   # a failure must still release the slot
    monkeypatch.setattr(perplexity_utils, "get_hiring_health", fake_health)
    with api._hiring_health_lock:
        api._hiring_health_inflight.add("harbor lights media")
    api._run_hiring_health("Harbor Lights Media", "Video Producer", "harbor lights media")
    assert seen == [("Harbor Lights Media", "Video Producer", db)]
    assert "harbor lights media" not in api._hiring_health_inflight
