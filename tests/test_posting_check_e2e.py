"""Phase 27c end to end: the "closed" rule across a fake sequence of pipeline
runs on a real temporary DB, through every surface that reads it.

Runs go through the real scrape_ats_boards + the real Greenhouse scraper
(HTTP faked), the real DBManager, posting_check and the digest filter.

  run 1  complete fetch, posting listed         -> listed
  run 2  complete fetch without it              -> missing_once (unknown)
  run 3  failed fetch (HTTP 502)                -> still missing_once
  run 4  partial fetch without it (meta short)  -> still missing_once
  run 5  complete fetch without it              -> gone
  run 6  the posting is back                    -> listed again

A LinkedIn row whose link points at the very same board posting stays
"unknown" forever: search results are never evidence, whatever the board does.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import requests

from job_agent import ats_scraper, posting_check, postprocess
from job_agent.ats_scraper import scrape_ats_boards

KW = ["Video Producer"]
BOARD = [{"company": "Acme", "ats": "greenhouse", "token": "acme"}]
MATCHED_URL = "https://job-boards.greenhouse.io/acme/jobs/1"
OTHERS = [
    {"title": "Software Engineer", "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/2"},
    {"title": "Accountant", "absolute_url": "https://acme.com/careers?gh_jid=3"},
    {"title": "Designer", "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/4"},
]


class _Resp:
    def __init__(self, payload=None, status=200):
        self._payload, self.status_code, self.text = payload, status, ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


_CLOCK = {"t": None}


@pytest.fixture(autouse=True)
def _run_clock(monkeypatch):
    """Pipeline runs are 8 hours apart (_run moves the clock): "gone" needs
    the missing fetches to span POSTING_GONE_MIN_SPAN_HOURS."""
    _CLOCK["t"] = datetime.now(timezone.utc) - timedelta(days=3)
    monkeypatch.setattr("job_agent.db_manager._utc_now",
                        lambda: _CLOCK["t"].strftime("%Y-%m-%d %H:%M:%S"))


@pytest.fixture(autouse=True)
def _clean_slot():
    ats_scraper._clear_board_report()
    yield
    ats_scraper._clear_board_report()


@pytest.fixture
def board(monkeypatch):
    """A fake Greenhouse board. mode: 'ok' | 'fail' | 'partial'."""
    fresh = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00Z")
    state = {"listed": True, "mode": "ok"}

    def jobs():
        out = [dict(j) for j in OTHERS]
        if state["listed"]:
            out.insert(0, {"title": "Senior Video Producer", "absolute_url": MATCHED_URL,
                           "location": {"name": "Seattle, WA"}, "first_published": fresh,
                           "content": "Produce video. " * 40})
        return out

    def fake_get(*args, **kwargs):
        if state["mode"] == "fail":
            return _Resp(status=502)
        listed = jobs()
        if state["mode"] == "partial":
            # The API says the board has more postings than it returned.
            return _Resp({"jobs": listed, "meta": {"total": len(listed) + 5}})
        return _Resp({"jobs": listed, "meta": {"total": len(listed)}})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests.Session, "get", lambda self, *a, **kw: fake_get(*a, **kw))
    return state


def _run(db, run_id):
    _CLOCK["t"] += timedelta(hours=8)
    df = scrape_ats_boards(BOARD, KW, db=db, run_id=run_id)
    db.refresh_listing_status()         # what run_search_session does after the ATS phase
    return df


def _row(db, job_id):
    return tuple(db._get_thread_conn().execute(
        "SELECT tier, confidence, match_level, reviewed, listing_status FROM scrapes WHERE id = ?",
        (job_id,)).fetchone())


def _feed_ids(db):
    return {r["id"] for r in db.get_unreviewed_jobs()}


def test_closed_rule_across_a_sequence_of_pipeline_runs(board, tmp_db):
    # ── run 1: the posting is scraped, graded Strong and tracked by nobody yet
    df = _run(tmp_db, "r1")
    job = df.to_dict("records")[0]
    assert job["job_url"] == MATCHED_URL
    job_id = tmp_db.save_job_with_relevance(
        job, relevance=True, reasoning="fit", match_level="HIGH_MATCH",
        confidence=0.9, session_id="r1", tier="strong")
    # A second Strong job from the same board that she tracks (cards are never hidden).
    tracked = dict(job, title="Video Producer", job_url=MATCHED_URL.replace("/1", "/4"),
                   link=MATCHED_URL.replace("/1", "/4"))
    tracked_id = tmp_db.save_job_with_relevance(
        tracked, relevance=True, reasoning="fit", match_level="HIGH_MATCH",
        confidence=0.8, session_id="r1", tier="strong")
    assert tmp_db.track_job(tracked_id)
    tmp_db.refresh_listing_status()
    assert tmp_db.get_listing_status(job_id)["status"] == "listed"
    assert _row(tmp_db, job_id) == ("strong", 0.9, "HIGH_MATCH", 0, "listed")
    assert job_id in _feed_ids(tmp_db)

    # ── run 2: the board stops listing it — one miss is not enough
    board["listed"] = False
    _run(tmp_db, "r2")
    st = tmp_db.get_listing_status(job_id)
    assert (st["status"], st["reason"]) == ("unknown", "missing_once")
    assert job_id in _feed_ids(tmp_db)

    # ── run 3: the board fetch fails — no evidence either way
    board["mode"] = "fail"
    _run(tmp_db, "r3")
    assert tmp_db.get_listing_status(job_id)["reason"] == "missing_once"

    # ── run 4: a partial read without it — still no evidence of absence
    board["mode"] = "partial"
    _run(tmp_db, "r4")
    assert tmp_db.get_listing_status(job_id)["reason"] == "missing_once"
    assert job_id in _feed_ids(tmp_db)

    # ── run 5: the second COMPLETE fetch without it -> gone
    board["mode"] = "ok"
    _run(tmp_db, "r5")
    st = tmp_db.get_listing_status(job_id)
    assert st["status"] == "gone" and st["gone_since"]
    conn = tmp_db._get_thread_conn()
    r2_fetched = conn.execute(
        "SELECT fetched_at FROM board_fetches WHERE run_id = 'r2'").fetchone()[0]
    assert st["gone_since"] == r2_fetched          # since the FIRST complete fetch that missed it
    assert [r[0] for r in conn.execute(
        "SELECT note FROM board_fetches ORDER BY id")] == [None, None, "failed", "partial", None]

    # Surfaces: hidden from Review and the digest, nothing changed on the row.
    assert _row(tmp_db, job_id) == ("strong", 0.9, "HIGH_MATCH", 0, "gone")
    assert job_id not in _feed_ids(tmp_db)
    digest_df = pd.DataFrame([{"title": job["title"], "company": "Acme", "job_url": MATCHED_URL,
                               "tier": "strong"}])
    assert postprocess._drop_gone_rows(digest_df, tmp_db).empty
    check = posting_check.build_posting_check(tmp_db, job_id)
    listing = next(f for f in check["facts"] if f["key"] == "listing")
    assert listing["text"].startswith("No longer on Acme's job board since") and listing["caution"]
    assert listing["source"] == "company_board"
    assert check["read"] == posting_check.READ_CHECK_FIRST
    assert posting_check.caution_line(tmp_db, job_id)

    # ── the tracked card on the same board: it was still listed, so it stays
    # on the board with no gone line; then it too leaves the board twice.
    apps = {a["job_id"]: a for a in tmp_db.get_applications()}
    assert apps[tracked_id]["listing_status"] == "listed"
    OTHERS[2:3], removed = [], OTHERS[2:3]
    try:
        _run(tmp_db, "r6")
        _run(tmp_db, "r7")
    finally:
        OTHERS[2:3] = removed
    apps = {a["job_id"]: a for a in tmp_db.get_applications()}
    assert apps[tracked_id]["listing_status"] == "gone"       # the card stays, with the fact
    assert apps[tracked_id]["listing_gone_since"]

    # ── run 8: the posting comes back -> listed again, back in the feed
    board["listed"] = True
    _run(tmp_db, "r8")
    assert tmp_db.get_listing_status(job_id)["status"] == "listed"
    assert _row(tmp_db, job_id) == ("strong", 0.9, "HIGH_MATCH", 0, "listed")
    assert job_id in _feed_ids(tmp_db)


def test_a_search_board_job_stays_unknown_forever(board, tmp_db):
    """A LinkedIn row whose link is the very posting a company board lists:
    the board dropping it, many complete fetches later, still proves nothing."""
    linkedin = {"title": "Senior Video Producer", "company": "Acme", "location": "Seattle, WA",
                "job_url": "https://acme.com/careers?gh_jid=3", "link": "https://acme.com/careers?gh_jid=3",
                "site": "linkedin", "description": "Produce video. " * 40,
                "date_posted": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")}
    job_id = tmp_db.save_job_with_relevance(
        linkedin, relevance=True, reasoning="fit", match_level="HIGH_MATCH",
        confidence=0.85, session_id="r1", tier="strong")
    _run(tmp_db, "r1")                                       # board lists gh_jid=3 ...
    accountant = OTHERS.pop(1)                               # ... then drops it for good
    try:
        for i in range(2, 8):
            _run(tmp_db, f"r{i}")
            st = tmp_db.get_listing_status(job_id)
            assert (st["status"], st["reason"]) == ("unknown", "not_board"), i
    finally:
        OTHERS.insert(1, accountant)
    assert _row(tmp_db, job_id) == ("strong", 0.85, "HIGH_MATCH", 0, None)
    assert job_id in _feed_ids(tmp_db)
    assert tmp_db.get_gone_links([linkedin["job_url"]]) == set()
    check = posting_check.build_posting_check(tmp_db, job_id)
    listing = next(f for f in check["facts"] if f["key"] == "listing")
    assert listing["text"].startswith("Came from LinkedIn") and not listing["caution"]
    assert check["read"] != posting_check.READ_CHECK_FIRST or "no longer" not in " ".join(check["reasons"])


def test_a_role_reposted_under_a_new_id_stays_in_the_feed(board, tmp_db):
    """Regression (review): boards re-post a role under a new id; title+company
    dedup keeps the new copy out of the cascade, so the old row is her only
    copy. It must stay visible, pointing at the new link — not go 'gone'."""
    df = _run(tmp_db, "r1")
    job = df.to_dict("records")[0]
    job_id = tmp_db.save_job_with_relevance(
        job, relevance=True, reasoning="fit", match_level="HIGH_MATCH",
        confidence=0.9, session_id="r1", tier="strong")
    board["listed"] = False
    reposted = {"title": "Senior Video Producer", "absolute_url": MATCHED_URL.replace("/1", "/77")}
    OTHERS.append(reposted)
    try:
        _run(tmp_db, "r2")
        _run(tmp_db, "r3")
    finally:
        OTHERS.remove(reposted)
    st = tmp_db.get_listing_status(job_id)
    assert (st["status"], st["reason"]) == ("listed", "relisted")
    assert job_id in _feed_ids(tmp_db)
    assert _row(tmp_db, job_id) == ("strong", 0.9, "HIGH_MATCH", 0, "listed")
    listing = next(f for f in posting_check.build_posting_check(tmp_db, job_id)["facts"]
                   if f["key"] == "listing")
    assert listing["text"].startswith("Still on Acme's job board under a new link")
    assert listing["url"] == reposted["absolute_url"] and not listing["caution"]
