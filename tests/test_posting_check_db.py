"""Phase 27c "Is this posting real?" — the DB layer.

Board fetches + sightings, THE CLOSED RULE (listed / gone / unknown), the
denormalized listing_status column, the Review-feed exclusion, earlier
applications at the same company and the posting-quality cache. Facts only:
none of this may change tier, confidence, match_level or reviewed.
"""
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from job_agent.db_manager import DBManager, posting_ref


GH = "greenhouse"
BOARD = "greenhouse:harborlightsmedia"


def _job(link, site=GH, company="Harbor Lights Media", title="Video Producer", **extra):
    return {"job_url": link, "site": site, "company": company, "title": title,
            "location": "Bellevue, WA", **extra}


def _save(db, link, site=GH, tier="strong", company="Harbor Lights Media", title="Video Producer"):
    return db.save_job_with_relevance(
        _job(link, site=site, company=company, title=title), relevance="relevant",
        reasoning="fits", match_level="HIGH_MATCH", confidence=0.9, tier=tier)


_CLOCK = {"t": None}
_RUN_STEP = timedelta(hours=7)   # any step >= POSTING_GONE_MIN_SPAN_HOURS


@pytest.fixture(autouse=True)
def _board_clock(monkeypatch):
    """Board fetches are one pipeline run apart (_fetch moves the clock by
    _RUN_STEP): "gone" needs the missing fetches to span
    POSTING_GONE_MIN_SPAN_HOURS. Starts 2 days ago so a fetch is recent."""
    _CLOCK["t"] = datetime.now(timezone.utc) - timedelta(days=2)
    monkeypatch.setattr("job_agent.db_manager._utc_now",
                        lambda: _CLOCK["t"].strftime("%Y-%m-%d %H:%M:%S"))


def _fetch(db, links, *, ok=True, complete=True, total=None, platform=GH, token="harborlightsmedia"):
    _CLOCK["t"] += _RUN_STEP
    return db.record_board_fetch(platform, token, "Harbor Lights Media", ok=ok, complete=complete,
                                 total_count=len(links) if total is None else total,
                                 links=links, run_id="run-1")


def _fetch_row(db, fetch_id):
    conn = db._get_thread_conn()
    conn.row_factory = sqlite3.Row
    row = dict(conn.execute("SELECT * FROM board_fetches WHERE id = ?", (fetch_id,)).fetchone())
    conn.row_factory = None
    return row


def _age_fetches(db, days):
    """Push every recorded fetch `days` into the past."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    with db.transaction() as conn:
        conn.execute("UPDATE board_fetches SET fetched_at = ?", (ts,))


def _scores(db, job_id):
    return db._get_thread_conn().execute(
        "SELECT tier, confidence, match_level, reviewed FROM scrapes WHERE id = ?",
        (job_id,)).fetchone()


# ── posting_ref ──────────────────────────────────────────────────────────────

class TestPostingRef:
    def test_greenhouse_forms_share_one_identity(self):
        forms = [
            "https://boards.greenhouse.io/harborlightsmedia/jobs/4012345?gh_jid=4012345",
            "https://job-boards.greenhouse.io/harborlightsmedia/jobs/4012345",
            "https://boards.greenhouse.io/harborlightsmedia/jobs/4012345",
            "https://careers.harborlightsmedia.com/jobs/4012345?gh_jid=4012345",
            "http://www.harborlightsmedia.com/careers/open-roles?gh_jid=4012345&utm=x",
        ]
        assert {posting_ref(f) for f in forms} == {"gh:4012345"}

    def test_plain_links_ignore_scheme_www_trailing_slash_query_fragment(self):
        a = posting_ref("https://jobs.lever.co/tidepool/5a1f09c2-3d4e?lever-source=x")
        b = posting_ref("http://www.jobs.lever.co/tidepool/5a1f09c2-3d4e/#apply")
        c = posting_ref("HTTPS://Jobs.Lever.co/tidepool/5a1f09c2-3d4e/")
        assert a == b == c == "jobs.lever.co/tidepool/5a1f09c2-3d4e"

    def test_non_greenhouse_jobs_path_is_not_a_gh_id(self):
        assert posting_ref("https://careers.copperkettle.example/jobs/6100042") == \
            "careers.copperkettle.example/jobs/6100042"

    @pytest.mark.parametrize("bad", ["", "   ", None, float("nan"), "nan", "not a url", 12])
    def test_empty_or_unparseable(self, bad):
        assert posting_ref(bad) == ""


# ── record_board_fetch: evidence flags ───────────────────────────────────────

class TestRecordBoardFetch:
    LINKS = [f"https://job-boards.greenhouse.io/harborlightsmedia/jobs/{i}" for i in range(1, 11)]

    def test_complete_fetch_is_evidence_and_records_sightings(self, tmp_db):
        fid = _fetch(tmp_db, self.LINKS)
        row = _fetch_row(tmp_db, fid)
        assert (row["evidence"], row["note"], row["board_key"], row["run_id"]) == \
            (1, None, BOARD, "run-1")
        assert row["link_count"] == 10
        n = tmp_db._get_thread_conn().execute(
            "SELECT COUNT(*) FROM board_sightings WHERE board_key = ?", (BOARD,)).fetchone()[0]
        assert n == 10

    def test_failed_partial_empty_are_no_evidence(self, tmp_db):
        failed = _fetch_row(tmp_db, _fetch(tmp_db, [], ok=False, total=0))
        partial = _fetch_row(tmp_db, _fetch(tmp_db, self.LINKS[:3], complete=False, total=10))
        empty = _fetch_row(tmp_db, _fetch(tmp_db, [], total=0))
        assert [(r["evidence"], r["note"]) for r in (failed, partial, empty)] == \
            [(0, "failed"), (0, "partial"), (0, "empty")]

    def test_partial_fetch_still_proves_presence(self, tmp_db):
        fid = _fetch(tmp_db, self.LINKS[:3], complete=False, total=10)
        last = tmp_db._get_thread_conn().execute(
            "SELECT DISTINCT last_seen_fetch_id FROM board_sightings").fetchall()
        assert last == [(fid,)]

    def test_drop_then_recover(self, tmp_db):
        _fetch(tmp_db, self.LINKS)                                    # 10 jobs
        drop = _fetch_row(tmp_db, _fetch(tmp_db, self.LINKS[:4]))     # 4 < 0.5 * 10
        again = _fetch_row(tmp_db, _fetch(tmp_db, self.LINKS[:4]))    # 4 vs 4 — real now
        back = _fetch_row(tmp_db, _fetch(tmp_db, self.LINKS))         # recovered
        assert (drop["evidence"], drop["note"]) == (0, "drop")
        assert (again["evidence"], again["note"]) == (1, None)
        assert (back["evidence"], back["note"]) == (1, None)

    def test_ignored_platforms_record_nothing(self, tmp_db):
        assert _fetch(tmp_db, self.LINKS, platform="workday") is None
        assert _fetch(tmp_db, self.LINKS, platform="linkedin") is None
        assert _fetch(tmp_db, self.LINKS, token="") is None
        assert tmp_db._get_thread_conn().execute(
            "SELECT COUNT(*) FROM board_fetches").fetchone()[0] == 0

    def test_first_seen_kept_last_seen_moves(self, tmp_db):
        f1 = _fetch(tmp_db, self.LINKS[:1])
        with tmp_db.transaction() as conn:
            conn.execute("UPDATE board_sightings SET first_seen_at = '2026-01-01 00:00:00'")
        f2 = _fetch(tmp_db, self.LINKS[:1])
        row = tmp_db._get_thread_conn().execute(
            "SELECT first_seen_at, last_seen_fetch_id FROM board_sightings").fetchone()
        assert row == ("2026-01-01 00:00:00", f2) and f2 > f1

    def test_timeout_inside_upsert_propagates_and_rolls_back(self, tmp_db):
        class _ScrapeTimeoutLike(Exception):
            pass

        real = tmp_db._get_thread_conn()

        class _Cursor:
            def __init__(self, cur):
                self._cur = cur

            def executemany(self, *a, **kw):
                raise _ScrapeTimeoutLike("alarm")

            def __getattr__(self, name):
                return getattr(self._cur, name)

        class _Conn:
            def cursor(self):
                return _Cursor(real.cursor())

            def __getattr__(self, name):
                return getattr(real, name)

        with patch.object(tmp_db, "_get_thread_conn", return_value=_Conn()):
            with pytest.raises(_ScrapeTimeoutLike):
                _fetch(tmp_db, self.LINKS)
        assert real.execute("SELECT COUNT(*) FROM board_fetches").fetchone()[0] == 0

    def test_db_error_is_swallowed(self, tmp_db):
        with tmp_db.transaction() as conn:
            conn.execute("DROP TABLE board_sightings")
        assert _fetch(tmp_db, self.LINKS) is None


# ── THE CLOSED RULE ──────────────────────────────────────────────────────────

class TestListingStatus:
    LINK = "https://boards.greenhouse.io/harborlightsmedia/jobs/111?gh_jid=111"
    OTHER = ["https://job-boards.greenhouse.io/harborlightsmedia/jobs/222",
             "https://job-boards.greenhouse.io/harborlightsmedia/jobs/333"]
    # How the board itself links the posting (a different host than the row).
    BOARD_FORM = "https://job-boards.greenhouse.io/harborlightsmedia/jobs/111"

    def test_listed_missing_once_gone_and_back(self, tmp_db):
        jid = _save(tmp_db, self.LINK)
        assert tmp_db.get_listing_status(jid)["reason"] == "not_seen"

        f1 = _fetch(tmp_db, [self.BOARD_FORM] + self.OTHER)
        st = tmp_db.get_listing_status(jid)
        assert (st["status"], st["reason"], st["board_key"]) == ("listed", "listed", BOARD)
        assert st["checked_at"] == _fetch_row(tmp_db, f1)["fetched_at"]

        f2 = _fetch(tmp_db, self.OTHER)
        st = tmp_db.get_listing_status(jid)
        assert (st["status"], st["reason"]) == ("unknown", "missing_once")

        _fetch(tmp_db, self.OTHER)
        st = tmp_db.get_listing_status(jid)
        assert (st["status"], st["reason"]) == ("gone", "gone")
        assert st["gone_since"] == _fetch_row(tmp_db, f2)["fetched_at"]

        _fetch(tmp_db, [self.BOARD_FORM] + self.OTHER)      # re-listed
        assert tmp_db.get_listing_status(jid)["status"] == "listed"

    def test_failed_partial_empty_fetches_between_do_not_count(self, tmp_db):
        jid = _save(tmp_db, self.LINK)
        _fetch(tmp_db, [self.BOARD_FORM] + self.OTHER)
        _fetch(tmp_db, self.OTHER)                                   # missing once
        _fetch(tmp_db, [], ok=False, total=0)                        # failed
        _fetch(tmp_db, self.OTHER[:1], complete=False, total=3)      # partial
        _fetch(tmp_db, [], total=0)                                  # empty
        assert tmp_db.get_listing_status(jid)["reason"] == "missing_once"

    def test_drop_fetch_does_not_count(self, tmp_db):
        jid = _save(tmp_db, self.LINK)
        many = [f"https://job-boards.greenhouse.io/harborlightsmedia/jobs/{i}" for i in range(500, 520)]
        _fetch(tmp_db, [self.BOARD_FORM] + many)
        _fetch(tmp_db, many[:2])      # drop
        _fetch(tmp_db, many[:2])      # first evidence miss at the new size
        assert tmp_db.get_listing_status(jid)["reason"] == "missing_once"

    def test_no_complete_fetch(self, tmp_db):
        jid = _save(tmp_db, self.LINK)
        _fetch(tmp_db, [self.BOARD_FORM], complete=False, total=5)
        st = tmp_db.get_listing_status(jid)
        assert (st["status"], st["reason"]) == ("unknown", "no_complete_fetch")
        assert st["first_seen_at"] and st["last_seen_at"]

    def test_stale_check(self, tmp_db):
        jid = _save(tmp_db, self.LINK)
        _fetch(tmp_db, [self.BOARD_FORM] + self.OTHER)
        _age_fetches(tmp_db, 30)
        st = tmp_db.get_listing_status(jid)
        assert (st["status"], st["reason"]) == ("unknown", "stale_check")

    def test_linkedin_row_with_board_link_is_not_board(self, tmp_db):
        jid = _save(tmp_db, self.LINK, site="linkedin")
        _fetch(tmp_db, self.OTHER)
        _fetch(tmp_db, [self.BOARD_FORM] + self.OTHER)
        _fetch(tmp_db, self.OTHER)
        _fetch(tmp_db, self.OTHER)
        st = tmp_db.get_listing_status(jid)
        assert (st["status"], st["reason"]) == ("unknown", "not_board")
        tmp_db.refresh_listing_status()
        assert tmp_db.get_job_details(jid)["listing_status"] is None

    def test_missing_job_and_errors_read_unknown(self, tmp_db):
        assert tmp_db.get_listing_status(99999)["status"] == "unknown"
        with patch.object(tmp_db, "_listing_statuses", side_effect=RuntimeError("x")):
            jid = _save(tmp_db, self.LINK)
            assert tmp_db.get_listing_status(jid)["status"] == "unknown"


class TestFalseClosureGuards:
    """Review fixes: a role re-posted under a new id, a board that changes its
    link format, and two fetches minutes apart must never read as gone."""
    OLD = "https://job-boards.greenhouse.io/harborlightsmedia/jobs/111"
    NEW = "https://job-boards.greenhouse.io/harborlightsmedia/jobs/999"
    OTHER = [("https://job-boards.greenhouse.io/harborlightsmedia/jobs/222", "Editor"),
             ("https://job-boards.greenhouse.io/harborlightsmedia/jobs/333", "Designer")]

    def test_role_reposted_under_a_new_id_stays_visible_with_the_new_link(self, tmp_db):
        """Regression: a role re-posted under a new id a few days later is kept
        out of the cascade by title+company dedup, so the old row went 'gone'
        and the open role vanished."""
        jid = _save(tmp_db, self.OLD, title="Producer/Editor, Social Media")
        _fetch(tmp_db, [(self.OLD, "Producer/Editor, Social Media")] + self.OTHER)
        _fetch(tmp_db, [(self.NEW, "Producer / Editor - Social Media")] + self.OTHER)
        _fetch(tmp_db, [(self.NEW, "Producer / Editor - Social Media")] + self.OTHER)
        st = tmp_db.get_listing_status(jid)
        assert (st["status"], st["reason"], st["relisted_link"]) == ("listed", "relisted", self.NEW)
        assert st["gone_since"] is None
        tmp_db.refresh_listing_status()
        assert jid in {r["id"] for r in tmp_db.get_unreviewed_jobs()}
        assert tmp_db.get_gone_links([self.OLD]) == set()

    def test_a_different_title_still_goes(self, tmp_db):
        jid = _save(tmp_db, self.OLD, title="Producer/Editor, Social Media")
        _fetch(tmp_db, [(self.OLD, "Producer/Editor, Social Media")] + self.OTHER)
        _fetch(tmp_db, [(self.NEW, "Senior Producer")] + self.OTHER)
        _fetch(tmp_db, [(self.NEW, "Senior Producer")] + self.OTHER)
        assert tmp_db.get_listing_status(jid)["status"] == "gone"

    def test_board_changing_its_link_format_closes_nothing(self, tmp_db):
        """Regression: Recruitee careers_url follows a custom domain; the
        same 12 postings under new links kept the board's count, so both
        fetches were evidence and every posting went gone together."""
        old = [(f"https://framestore.recruitee.com/o/slug-{i}", f"Role {i}") for i in range(12)]
        new = [(f"https://careers.framestore.com/o/slug-{i}", f"Role {i}") for i in range(12)]
        jid = tmp_db.save_job_with_relevance(
            {"job_url": old[3][0], "site": "recruitee", "company": "Framestore", "title": "Role 3",
             "location": "London"}, relevance="relevant", reasoning="fits",
            match_level="HIGH_MATCH", confidence=0.9, tier="strong")
        kw = dict(platform="recruitee", token="framestore")
        _fetch(tmp_db, old, **kw)
        f2 = _fetch(tmp_db, new, **kw)
        f3 = _fetch(tmp_db, new, **kw)
        assert (_fetch_row(tmp_db, f2)["evidence"], _fetch_row(tmp_db, f2)["note"]) == (0, "churn")
        assert _fetch_row(tmp_db, f3)["evidence"] == 1      # the new format is the baseline now
        assert tmp_db.get_listing_status(jid)["reason"] == "missing_once"
        _fetch(tmp_db, new, **kw)
        st = tmp_db.get_listing_status(jid)
        assert st["status"] == "listed" and st["relisted_link"] == new[3][0]
        tmp_db.refresh_listing_status()
        assert jid in {r["id"] for r in tmp_db.get_unreviewed_jobs()}

    def test_churn_alone_holds_back_one_fetch(self, tmp_db):
        """Without titles the churn guard still makes the first new-format
        fetch no evidence: one run later than the title check, never at once."""
        old = [f"https://framestore.recruitee.com/o/slug-{i}" for i in range(12)]
        new = [f"https://careers.framestore.com/o/slug-{i}" for i in range(12)]
        jid = tmp_db.save_job_with_relevance(
            {"job_url": old[3], "site": "recruitee", "company": "Framestore", "title": "Role 3",
             "location": "London"}, relevance="relevant", reasoning="fits",
            match_level="HIGH_MATCH", confidence=0.9, tier="strong")
        kw = dict(platform="recruitee", token="framestore")
        _fetch(tmp_db, old, **kw)
        _fetch(tmp_db, new, **kw)
        _fetch(tmp_db, new, **kw)
        assert tmp_db.get_listing_status(jid)["reason"] == "missing_once"

    def test_small_or_healthy_boards_are_not_churn(self, tmp_db):
        links = [f"https://job-boards.greenhouse.io/harborlightsmedia/jobs/{i}" for i in range(1, 11)]
        _fetch(tmp_db, links)
        f = _fetch(tmp_db, links[:6] + [f"https://job-boards.greenhouse.io/harborlightsmedia/jobs/{i}"
                                         for i in range(50, 54)])
        assert _fetch_row(tmp_db, f)["note"] is None          # 6 of 10 kept
        _fetch(tmp_db, links[:3], token="tiny")
        f = _fetch(tmp_db, [f"https://job-boards.greenhouse.io/tiny/jobs/{i}" for i in (7, 8, 9)],
                   token="tiny")
        assert _fetch_row(tmp_db, f)["note"] is None          # under POSTING_CHURN_MIN_REFS

    def test_two_fetches_minutes_apart_are_one_look(self, tmp_db):
        """Regression: a UI search right after the pipeline fetched every
        board again 60 s later; the two misses closed the posting."""
        jid = _save(tmp_db, self.OLD)
        _fetch(tmp_db, [self.OLD] + [l for l, _ in self.OTHER])
        f2 = _fetch(tmp_db, [l for l, _ in self.OTHER])
        _CLOCK["t"] -= _RUN_STEP - timedelta(seconds=60)
        _fetch(tmp_db, [l for l, _ in self.OTHER])
        assert tmp_db.get_listing_status(jid)["reason"] == "missing_once"
        _fetch(tmp_db, [l for l, _ in self.OTHER])                 # the next run
        st = tmp_db.get_listing_status(jid)
        assert st["status"] == "gone" and st["gone_since"] == _fetch_row(tmp_db, f2)["fetched_at"]


# ── refresh_listing_status, feed, board, digest helper ───────────────────────

class TestRefreshAndSurfaces:
    LINK = "https://job-boards.greenhouse.io/harborlightsmedia/jobs/111"
    KEEP = "https://job-boards.greenhouse.io/harborlightsmedia/jobs/222"

    def _gone_setup(self, db):
        gone_id = _save(db, self.LINK)
        keep_id = _save(db, self.KEEP)
        _fetch(db, [self.LINK, self.KEEP])
        _fetch(db, [self.KEEP])
        _fetch(db, [self.KEEP])
        return gone_id, keep_id

    def test_refresh_writes_columns_and_counts(self, tmp_db):
        gone_id, keep_id = self._gone_setup(tmp_db)
        li_id = _save(tmp_db, "https://www.linkedin.com/jobs/view/1", site="linkedin")
        counts = tmp_db.refresh_listing_status()
        assert counts == {"listed": 1, "gone": 1, "unknown": 1}
        gone = tmp_db.get_job_details(gone_id)
        keep = tmp_db.get_job_details(keep_id)
        assert (gone["listing_status"], keep["listing_status"]) == ("gone", "listed")
        assert gone["listing_gone_since"] and keep["listing_checked_at"]
        assert tmp_db.get_job_details(li_id)["listing_status"] is None

    def test_refresh_never_touches_scores(self, tmp_db):
        gone_id, keep_id = self._gone_setup(tmp_db)
        before = {j: _scores(tmp_db, j) for j in (gone_id, keep_id)}
        tmp_db.refresh_listing_status()
        tmp_db.refresh_listing_status([gone_id, keep_id])
        assert {j: _scores(tmp_db, j) for j in (gone_id, keep_id)} == before

    def test_refresh_reappearance_clears_gone(self, tmp_db):
        gone_id, _ = self._gone_setup(tmp_db)
        tmp_db.refresh_listing_status()
        _fetch(tmp_db, [self.LINK, self.KEEP])
        tmp_db.refresh_listing_status([gone_id])
        row = tmp_db.get_job_details(gone_id)
        assert (row["listing_status"], row["listing_gone_since"]) == ("listed", None)

    def test_refresh_covers_tracked_untiered_jobs(self, tmp_db):
        gone_id, _ = self._gone_setup(tmp_db)
        with tmp_db.transaction() as conn:
            conn.execute("UPDATE scrapes SET tier = NULL WHERE id = ?", (gone_id,))
        tmp_db.track_job(gone_id)
        tmp_db.refresh_listing_status()
        assert tmp_db.get_job_details(gone_id)["listing_status"] == "gone"

    def test_refresh_never_raises(self, tmp_db):
        with patch.object(tmp_db, "_listing_statuses", side_effect=RuntimeError("boom")):
            _save(tmp_db, self.LINK)
            assert tmp_db.refresh_listing_status() == {"listed": 0, "gone": 0, "unknown": 0}

    def test_feed_hides_gone_but_board_keeps_card_with_fact(self, tmp_db):
        gone_id, keep_id = self._gone_setup(tmp_db)
        tmp_db.track_job(gone_id)
        tmp_db.refresh_listing_status()

        feed = tmp_db.get_unreviewed_jobs()
        assert [j["id"] for j in feed] == [keep_id]
        assert feed[0]["listing_status"] == "listed"
        assert "listing_checked_at" in feed[0] and "listing_gone_since" in feed[0]

        apps = tmp_db.get_applications()
        assert [a["job_id"] for a in apps] == [gone_id]
        assert apps[0]["listing_status"] == "gone" and apps[0]["listing_gone_since"]

    def test_feed_query_still_uses_partial_tiered_index(self, tmp_db):
        # Pull the exact SQL the feed runs, then EXPLAIN it.
        import inspect
        from job_agent import db_manager
        conn = tmp_db._get_thread_conn()
        src = inspect.getsource(db_manager.DBManager.get_unreviewed_jobs)
        sql = src.split('cursor.execute("""', 1)[1].split('""")', 1)[0]
        assert "COALESCE(listing_status, '') <> 'gone'" in sql
        assert " OR listing_status" not in sql
        plan = " ".join(str(r) for r in conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall())
        assert "idx_scrapes_tiered" in plan, plan

    def test_get_gone_links(self, tmp_db):
        self._gone_setup(tmp_db)
        tmp_db.refresh_listing_status()
        assert tmp_db.get_gone_links([self.LINK, self.KEEP, "https://x.com/1", None, ""]) == \
            {self.LINK}
        assert tmp_db.get_gone_links([]) == set()

    def test_get_gone_links_chunks(self, tmp_db):
        self._gone_setup(tmp_db)
        tmp_db.refresh_listing_status()
        links = [f"https://example.com/j/{i}" for i in range(1200)] + [self.LINK]
        assert tmp_db.get_gone_links(links) == {self.LINK}

    def test_migration_is_idempotent(self, tmp_db):
        gone_id, _ = self._gone_setup(tmp_db)
        tmp_db.refresh_listing_status()
        again = DBManager(db_path=tmp_db.db_path)
        try:
            assert again.get_job_details(gone_id)["listing_status"] == "gone"
            assert again.get_listing_status(gone_id)["status"] == "gone"
        finally:
            again.close()

    def test_big_run_is_cheap(self, tmp_db):
        links = [f"https://job-boards.greenhouse.io/big/jobs/{i}" for i in range(10000)]
        import time
        t0 = time.monotonic()
        fid = _fetch(tmp_db, links, token="big")
        assert fid and time.monotonic() - t0 < 5
        assert tmp_db._get_thread_conn().execute(
            "SELECT COUNT(*) FROM board_sightings").fetchone()[0] == 10000


# ── earlier applications at the same company ─────────────────────────────────

class TestCompanyApplications:
    def test_normalized_company_match_with_stage_reach(self, tmp_db):
        old = _save(tmp_db, "https://jobs.lever.co/harborlightsmedia/a", site="lever",
                    company="Harbor Lights Media, Inc.", title="Senior Producer")
        other = _save(tmp_db, "https://jobs.lever.co/copperkettle/b", site="lever",
                      company="Copper Kettle Studios")
        cur = _save(tmp_db, "https://jobs.lever.co/harborlightsmedia/c", site="lever", company="Harbor Lights Media")
        for j in (old, other, cur):
            tmp_db.track_job(j)
        app_old = next(a["app_id"] for a in tmp_db.get_applications() if a["job_id"] == old)
        tmp_db.update_application(app_old, "Applied", "")
        tmp_db.update_application(app_old, "Interview", "")
        tmp_db.update_application(app_old, "Rejected", "")

        rows = tmp_db.get_company_applications("Harbor Lights Media", exclude_job_id=cur)
        assert [r["job_id"] for r in rows] == [old]
        r = rows[0]
        assert r["title"] == "Senior Producer" and r["status"] == "Rejected"
        assert r["highest_stage"] == "Interview"
        assert r["applied_at"] and r["tracked_at"] and r["updated_at"]
        assert set(r) >= {"app_id", "job_id", "title", "company", "status", "tracked_at",
                          "applied_at", "updated_at", "highest_stage", "reached_at"}

        both = tmp_db.get_company_applications("harbor lights media")
        assert {r["job_id"] for r in both} == {old, cur}

    def test_never_applied_has_no_applied_at(self, tmp_db):
        j = _save(tmp_db, "https://jobs.lever.co/harborlightsmedia/a", site="lever")
        tmp_db.track_job(j)
        rows = tmp_db.get_company_applications("Harbor Lights Media")
        assert rows[0]["applied_at"] is None and rows[0]["highest_stage"] == "To Apply"

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_company_never_matches(self, tmp_db, blank):
        j = _save(tmp_db, "https://jobs.lever.co/x/a", site="lever", company="")
        tmp_db.track_job(j)
        assert tmp_db.get_company_applications(blank) == []


# ── posting quality cache ────────────────────────────────────────────────────

class TestPostingQuality:
    def test_save_and_get_roundtrip(self, tmp_db):
        j = _save(tmp_db, "https://jobs.lever.co/harborlightsmedia/a", site="lever")
        assert tmp_db.get_posting_quality(j) is None
        result = {"tools": [{"name": "DaVinci Resolve", "quote": "DaVinci Resolve and Asana"}],
                  "team": None, "first_year_scope": None, "pay": None,
                  "contradictions": [], "boilerplate": "low", "dropped": 1}
        assert tmp_db.save_posting_quality(j, "abc", "claude-sonnet-5", result)
        got = tmp_db.get_posting_quality(j)
        assert got["result"] == result and got["jd_hash"] == "abc"
        assert got["model"] == "claude-sonnet-5" and got["checked_at"]
        assert tmp_db.save_posting_quality(j, "def", "claude-sonnet-5", {"tools": []})
        assert tmp_db.get_posting_quality(j)["jd_hash"] == "def"
        assert _scores(tmp_db, j)[0] == "strong"
