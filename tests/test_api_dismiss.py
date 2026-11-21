"""Dismiss endpoint: location-driven dismissals close the feedback loop."""
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from job_agent.db_manager import DBManager
    from job_agent import api
    db = DBManager(db_path=str(tmp_path / "api_test.db"))
    monkeypatch.setattr(api, "_db", db)
    return TestClient(api.app), db


def _save(db, location):
    return db.save_job({
        "title": "Producer", "company": "X", "location": location,
        "job_url": f"https://ex/{location}", "description": "d",
    })


def _save_n(db, location, n):
    return db.save_job({
        "title": f"Producer {n}", "company": "X", "location": location,
        "job_url": f"https://ex/{location}/{n}", "description": "d",
    })


def _vote_yesterday(db, location, job_id):
    conn = db._get_thread_conn()
    conn.execute("INSERT INTO location_votes (location, job_id, voted_at) "
                 "VALUES (?, ?, datetime('now', '-1 day'))", (location.lower(), job_id))
    conn.commit()


def test_location_reason_suppresses_on_a_second_day(client):
    """Saying it once about a location must not block it: a second vote on
    another day does."""
    c, db = client
    first, second = _save_n(db, "Austin, TX", 1), _save_n(db, "Austin, TX", 2)
    r = c.post(f"/api/jobs/{first}/dismiss", json={"reason": "wrong location, not remote"})
    assert r.status_code == 200
    assert r.json()["suppressed_location"] is None
    assert db.get_suppressed_locations() == set()
    _vote_yesterday(db, "Austin, TX", 77)                 # the user said it on another day too
    r = c.post(f"/api/jobs/{second}/dismiss", json={"reason": "wrong location"})
    assert r.json()["suppressed_location"] == "austin, tx"
    assert "austin, tx" in db.get_suppressed_locations()


def test_a_batch_on_one_day_is_saying_it_once(client):
    c, db = client
    for n in range(3):
        c.post(f"/api/jobs/{_save_n(db, 'Austin, TX', n)}/dismiss", json={"reason": "wrong location"})
    assert db.get_suppressed_locations() == set()


def test_same_job_twice_is_one_vote(client):
    c, db = client
    jid = _save(db, "Austin, TX")
    for _ in range(2):
        c.post(f"/api/jobs/{jid}/dismiss", json={"reason": "wrong location"})
    assert db.get_suppressed_locations() == set()


def _never_suppressed(client, location):
    c, db = client
    _vote_yesterday(db, location, 77)
    for n in range(3):
        jid = _save_n(db, location, n)
        r = c.post(f"/api/jobs/{jid}/dismiss", json={"reason": "wrong location"})
        assert r.json()["suppressed_location"] is None
    r = c.post(f"/api/jobs/{_save_n(db, location, 9)}/dismiss",
               json={"reason": "location", "suppress_location": True})
    assert r.json()["suppressed_location"] is None
    assert db.get_suppressed_locations() == set()


@pytest.mark.parametrize("home_state", ["Washington", "WA, US"])
def test_bare_home_state_label_is_never_suppressed(client, home_state):
    """A bare label for the profile's own state (search.locations state_code /
    state_name: wa / washington in the test profile) names no place: blocking
    it would drop every statewide posting, however often the user dismisses it."""
    _never_suppressed(client, home_state)


@pytest.mark.parametrize("vague", ["United States", "United States of America", "Hybrid", "2 locations",
                                   "Bellevue, WA, US", "Redmond, WA", "Kirkland, WA"])
def test_vague_locations_are_never_suppressed(client, vague):
    """A bare country label, 'Hybrid', or a primary-metro city is never
    blocked, however often the user dismisses it."""
    _never_suppressed(client, vague)


def test_explicit_flag_suppresses(client):
    c, db = client
    jid = _save(db, "Denver, CO")
    r = c.post(f"/api/jobs/{jid}/dismiss", json={"reason": "nope", "suppress_location": True})
    assert r.json()["suppressed_location"] == "denver, co"


def test_non_location_reason_does_not_suppress(client):
    c, db = client
    jid = _save(db, "Austin, TX")
    r = c.post(f"/api/jobs/{jid}/dismiss", json={"reason": "already applied"})
    assert r.json()["suppressed_location"] is None
    assert db.get_suppressed_locations() == set()


def test_never_suppresses_remote(client):
    c, db = client
    jid = _save(db, "Remote")
    r = c.post(f"/api/jobs/{jid}/dismiss", json={"reason": "location", "suppress_location": True})
    assert r.json()["suppressed_location"] is None  # refused by add_location_suppression
