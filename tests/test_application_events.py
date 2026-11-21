"""Outcome history (Phase 27): the application_events ledger, saved document
versions, the sent-CV record, and the API / chat paths that write them.

Before this, a status change overwrote the old status: a card that reached
interviews and was later Rejected left no trace. These tests pin the rules:
- a real status change and its ledger event commit together;
- Interview -> Rejected keeps the reach; an accidental in-and-out does not;
- corrections invalidate, never delete;
- the Kanban PUT and the chat tool share one side-effect path.

Hermetic: temp databases only, spawns mocked, no network, no LLM.
"""
import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from job_agent.db_manager import DBManager


# ── helpers ──────────────────────────────────────────────────────────────────

def _job(db, n=1, **over):
    data = {"title": f"Producer {n}", "company": f"Studio {n}", "location": "Remote",
            "job_url": f"https://example.com/job/{n}", "description": "Make shows. " * 20}
    data.update(over)
    return db.save_job(data)


def _tracked(db, n=1, **over):
    """A job tracked on the Kanban -> (job_id, app_id)."""
    job_id = _job(db, n, **over)
    assert db.track_job(job_id) is True
    app_id = db._get_thread_conn().execute(
        "SELECT id FROM applications WHERE job_id = ?", (job_id,)).fetchone()[0]
    return job_id, app_id


def _age_events(db, app_id, minutes):
    """Pretend every ledger entry of this card was recorded `minutes` ago."""
    conn = db._get_thread_conn()
    conn.execute(
        f"UPDATE application_events SET recorded_at = datetime('now', '-{int(minutes)} minutes') "
        f"WHERE app_id = ?", (app_id,))
    conn.commit()


def _status(db, app_id):
    return db._get_thread_conn().execute(
        "SELECT status FROM applications WHERE id = ?", (app_id,)).fetchone()[0]


def _events(db, app_id, event_type=None):
    events = db.get_application_events(app_id)
    return [e for e in events if event_type is None or e["event_type"] == event_type]


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class _NoThreads:
    """Stand-in for the api module's `threading`: Thread objects never start,
    so no background generation can run from a test. Everything else (Lock,
    Semaphore) is the real thing."""

    def __init__(self):
        self.suppressed = 0

    def __getattr__(self, name):
        return getattr(threading, name)

    def Thread(self, *args, **kwargs):
        def _start():
            self.suppressed += 1
        return SimpleNamespace(start=_start)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """API on a temp DB with both spawns replaced by recorders."""
    from fastapi.testclient import TestClient
    from job_agent import api
    db = DBManager(db_path=str(tmp_path / "ledger.db"))
    monkeypatch.setattr(api, "_db", db)
    api._invalidate_stats_cache()
    spawns = {"intel": [], "prep": []}
    monkeypatch.setattr(api, "_spawn_outreach_intel",
                        lambda app_id, force=False: spawns["intel"].append(app_id) or True)
    monkeypatch.setattr(api, "_spawn_interview_prep",
                        lambda app_id, force=False: spawns["prep"].append(app_id) or True)
    yield SimpleNamespace(client=TestClient(api.app), db=db, api=api, spawns=spawns)
    api._invalidate_stats_cache()
    db.close()


# ── schema ───────────────────────────────────────────────────────────────────

class TestSchema:
    def test_new_tables_and_prep_stage_exist(self, tmp_db):
        conn = tmp_db._get_thread_conn()
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"application_events", "application_artifact_links", "document_versions"} <= tables
        cols = {r[1] for r in conn.execute("PRAGMA table_info(application_events)")}
        assert cols == {"id", "app_id", "job_id", "event_type", "from_status", "to_status",
                        "effective_at", "recorded_at", "source", "note", "corrects_event_id",
                        "title", "company"}
        assert "prep_stage" in {r[1] for r in conn.execute("PRAGMA table_info(applications)")}

    def test_existing_database_gains_the_tables(self, tmp_path):
        """A pre-Phase-27 database: opening it adds the tables, keeps the rows."""
        path = str(tmp_path / "old.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE scrapes (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, "
                     "title TEXT, company TEXT, location TEXT, link TEXT UNIQUE, raw_data TEXT, "
                     "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE applications (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "job_id INTEGER, status TEXT DEFAULT 'To Apply', notes TEXT, "
                     "date_applied TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("INSERT INTO scrapes (title, company, link) VALUES ('P', 'C', 'https://o/1')")
        conn.execute("INSERT INTO applications (job_id, status) VALUES (1, 'Applied')")
        conn.commit()
        conn.close()
        db = DBManager(db_path=path)
        try:
            assert db.get_application_events(1) == []
            assert db.update_application(1, "Interview", None) is True
            assert [e["to_status"] for e in db.get_application_events(1)] == ["Interview"]
        finally:
            db.close()


# ── the ledger itself ────────────────────────────────────────────────────────

class TestLedger:
    def test_track_writes_the_initial_event(self, tmp_db):
        job_id, app_id = _tracked(tmp_db)
        events = _events(tmp_db, app_id)
        assert len(events) == 1
        e = events[0]
        assert (e["event_type"], e["from_status"], e["to_status"]) == ("tracked", None, "To Apply")
        assert (e["title"], e["company"], e["job_id"]) == ("Producer 1", "Studio 1", job_id)
        assert e["effective_at"] and e["valid"] is True
        # tracking the same job again adds nothing
        assert tmp_db.track_job(job_id) is True
        assert len(_events(tmp_db, app_id)) == 1

    def test_status_change_writes_one_event(self, tmp_db):
        _, app_id = _tracked(tmp_db)
        assert tmp_db.update_application(app_id, "Applied", None) is True
        assert _status(tmp_db, app_id) == "Applied"
        e = _events(tmp_db, app_id)[-1]
        assert (e["event_type"], e["from_status"], e["to_status"], e["source"]) == \
            ("status_change", "To Apply", "Applied", "app")

    def test_status_and_event_commit_together(self, tmp_db, monkeypatch):
        """If the ledger write fails, the status change must roll back with it."""
        _, app_id = _tracked(tmp_db)

        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("ledger write failed")

        monkeypatch.setattr(tmp_db, "_insert_application_event", boom)
        assert tmp_db.update_application(app_id, "Applied", "note") is False
        monkeypatch.undo()
        assert _status(tmp_db, app_id) == "To Apply"          # not half-applied
        assert [e["event_type"] for e in _events(tmp_db, app_id)] == ["tracked"]
        assert tmp_db.get_application(app_id)["notes"] != "note"

    def test_same_status_and_notes_only_write_nothing(self, tmp_db):
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "To Apply", "typing notes")   # status resent
        tmp_db.update_application(app_id, None, "more notes")           # notes-only
        assert len(_events(tmp_db, app_id)) == 1
        assert tmp_db.get_application(app_id)["notes"] == "more notes"

    def test_unknown_application_writes_nothing(self, tmp_db):
        tmp_db.update_application(424242, "Applied", None)
        assert tmp_db.get_application_events(424242) == []

    def test_interview_then_rejected_keeps_the_reach(self, tmp_db):
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)
        tmp_db.update_application(app_id, "Interview", None)
        _age_events(tmp_db, app_id, 60 * 24 * 30)            # the interview was a month ago
        tmp_db.update_application(app_id, "Rejected", None)

        assert _status(tmp_db, app_id) == "Rejected"          # the Kanban stays Rejected
        reach = tmp_db.get_stage_reach(app_id)
        assert reach["highest_stage"] == "Interview"
        assert reach["reached_at"]
        assert reach["reached_inferred"] is False
        assert [h["status"] for h in reach["history"]] == ["Applied", "Interview", "Rejected"]
        assert all(h["at"] and h["inferred"] is False for h in reach["history"])
        assert _events(tmp_db, app_id, "correction") == []
        assert tmp_db.count_reached_interview()["confirmed"] == 1

    def test_quick_in_and_out_is_corrected(self, tmp_db):
        """An accidental drag: into Interview and out again within 10 minutes."""
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)
        tmp_db.update_application(app_id, "Interview", None)
        tmp_db.update_application(app_id, "Applied", None)

        reach = tmp_db.get_stage_reach(app_id)
        assert reach["highest_stage"] == "Applied"
        assert "Interview" not in [h["status"] for h in reach["history"]]
        assert tmp_db.count_reached_interview()["confirmed"] == 0
        # audit trail kept: the entry is still there, flagged, with its correction
        entry = next(e for e in _events(tmp_db, app_id) if e["to_status"] == "Interview")
        assert entry["valid"] is False
        corrections = _events(tmp_db, app_id, "correction")
        assert len(corrections) == 1
        assert corrections[0]["corrects_event_id"] == entry["id"]
        assert corrections[0]["source"] == "auto"

    def test_catch_up_interview_then_rejected_quickly_keeps_the_reach(self, tmp_db):
        """Catching up an old card in two quick moves (Interview, then
        Rejected) records a real interview-then-rejection — never a drag."""
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)
        _age_events(tmp_db, app_id, 60 * 24 * 30)
        tmp_db.update_application(app_id, "Interview", None)
        tmp_db.update_application(app_id, "Rejected", None)
        assert _events(tmp_db, app_id, "correction") == []
        reach = tmp_db.get_stage_reach(app_id)
        assert reach["highest_stage"] == "Interview"
        assert [h["status"] for h in reach["history"]] == ["Applied", "Interview", "Rejected"]
        assert tmp_db.count_reached_interview()["confirmed"] == 1

    @pytest.mark.parametrize("dead", ["Rejected", "Ghosted"])
    def test_misselect_undone_on_a_dead_card_is_corrected(self, tmp_db, dead):
        """A Rejected / Ghosted card set to Interview by mistake in the status
        dropdown and set straight back: it never reached an interview."""
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)
        tmp_db.update_application(app_id, dead, None)
        _age_events(tmp_db, app_id, 60 * 24 * 30)            # rejected a month ago
        tmp_db.update_application(app_id, "Interview", None)  # wrong option
        tmp_db.update_application(app_id, dead, None)         # undone seconds later

        reach = tmp_db.get_stage_reach(app_id)
        assert reach["highest_stage"] == "Applied"
        assert [h["status"] for h in reach["history"]] == ["Applied", dead]
        assert tmp_db.count_reached_interview() == {"confirmed": 0, "unconfirmed": 0}
        entry = next(e for e in _events(tmp_db, app_id) if e["to_status"] == "Interview")
        assert entry["valid"] is False
        assert [c["corrects_event_id"] for c in _events(tmp_db, app_id, "correction")] == [entry["id"]]
        assert _status(tmp_db, app_id) == dead

    def test_accepted_misselect_on_a_rejected_card_is_corrected(self, tmp_db):
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)
        tmp_db.update_application(app_id, "Rejected", None)
        _age_events(tmp_db, app_id, 60)
        tmp_db.update_application(app_id, "Accepted", None)
        tmp_db.update_application(app_id, "Rejected", None)
        reach = tmp_db.get_stage_reach(app_id)
        assert reach["highest_stage"] == "Applied"
        assert [h["status"] for h in reach["history"]] == ["Applied", "Rejected"]
        assert tmp_db.count_reached_interview()["confirmed"] == 0

    def test_dead_card_revived_for_real_keeps_the_reach(self, tmp_db):
        """The opposite direction: a Rejected card that really got an
        interview later, and was rejected again after the window, keeps it."""
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)
        tmp_db.update_application(app_id, "Rejected", None)
        tmp_db.update_application(app_id, "Interview", None)
        _age_events(tmp_db, app_id, 11)
        tmp_db.update_application(app_id, "Rejected", None)
        assert _events(tmp_db, app_id, "correction") == []
        assert tmp_db.get_stage_reach(app_id)["highest_stage"] == "Interview"
        assert tmp_db.count_reached_interview()["confirmed"] == 1

    def test_quick_forward_move_keeps_the_earlier_stage(self, tmp_db):
        """Screening Call -> Interview within minutes: both steps stand."""
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)
        tmp_db.update_application(app_id, "Screening Call", None)
        tmp_db.update_application(app_id, "Interview", None)
        tmp_db.update_application(app_id, "Ghosted", None)
        assert _events(tmp_db, app_id, "correction") == []
        assert [h["status"] for h in tmp_db.get_stage_reach(app_id)["history"]] == [
            "Applied", "Screening Call", "Interview", "Ghosted"]

    def test_leaving_after_the_window_is_not_a_correction(self, tmp_db):
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Screening Call", None)
        _age_events(tmp_db, app_id, 11)
        tmp_db.update_application(app_id, "Applied", None)
        assert _events(tmp_db, app_id, "correction") == []
        assert tmp_db.get_stage_reach(app_id)["highest_stage"] == "Screening Call"

    def test_overshoot_is_corrected_but_the_real_stage_stands(self, tmp_db):
        """Picked Final Interview by mistake, fixed to Interview right away."""
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Final Interview", None)
        tmp_db.update_application(app_id, "Interview", None)
        reach = tmp_db.get_stage_reach(app_id)
        assert reach["highest_stage"] == "Interview"
        assert [h["status"] for h in reach["history"]] == ["Interview"]

    def test_explicit_correction_invalidates_without_deleting(self, tmp_db):
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Interview", None)
        _age_events(tmp_db, app_id, 60)
        entry = _events(tmp_db, app_id)[-1]
        before = len(_events(tmp_db, app_id))

        correction_id = tmp_db.correct_application_event(app_id, entry["id"], note="mis-click")
        assert correction_id
        events = _events(tmp_db, app_id)
        assert len(events) == before + 1                      # appended, nothing removed
        assert next(e for e in events if e["id"] == entry["id"])["valid"] is False
        assert tmp_db.get_stage_reach(app_id)["highest_stage"] == "To Apply"
        assert tmp_db.count_reached_interview()["confirmed"] == 0
        assert _status(tmp_db, app_id) == "Interview"          # the card itself is untouched

        # correcting the correction restores the original entry
        assert tmp_db.correct_application_event(app_id, correction_id)
        assert tmp_db.get_stage_reach(app_id)["highest_stage"] == "Interview"
        assert tmp_db.count_reached_interview()["confirmed"] == 1

    def test_correction_must_match_the_application(self, tmp_db):
        _, app_a = _tracked(tmp_db, 1)
        _, app_b = _tracked(tmp_db, 2)
        event_of_a = _events(tmp_db, app_a)[0]["id"]
        assert tmp_db.correct_application_event(app_b, event_of_a) == 0
        assert tmp_db.correct_application_event(app_a, 999999) == 0
        assert tmp_db.record_application_event(app_a, None, "correction") == 0  # needs a target

    def test_record_application_event_copies_title_and_feeds_the_reach(self, tmp_db):
        job_id, app_id = _tracked(tmp_db)
        event_id = tmp_db.record_application_event(
            app_id, job_id, "stage_reach", to_status="Screening Call",
            effective_at="2020-01-14", source="maintainer", note="confirmed by the candidate")
        assert event_id > 0
        e = next(e for e in _events(tmp_db, app_id) if e["id"] == event_id)
        assert (e["title"], e["company"], e["source"]) == ("Producer 1", "Studio 1", "maintainer")
        reach = tmp_db.get_stage_reach(app_id)
        assert (reach["highest_stage"], reach["reached_at"]) == ("Screening Call", "2020-01-14")

    def test_events_survive_deleting_the_application(self, tmp_db):
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Interview", None)
        _age_events(tmp_db, app_id, 60)
        assert tmp_db.delete_application(app_id) is True
        events = _events(tmp_db, app_id)
        assert [e["event_type"] for e in events] == ["tracked", "status_change", "deleted"]
        assert all(e["title"] == "Producer 1" and e["company"] == "Studio 1" for e in events)
        assert tmp_db.count_reached_interview()["confirmed"] == 1

    def test_events_survive_deleting_the_job(self, tmp_db):
        job_id, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Screening Call", None)
        assert tmp_db.delete_job(job_id) is True
        assert tmp_db.get_job_details(job_id) is None
        events = _events(tmp_db, app_id)
        assert len(events) == 3 and events[-1]["event_type"] == "deleted"
        assert events[0]["company"] == "Studio 1"


# ── apply-time snapshot + sent-CV record ─────────────────────────────────────

class TestApplySnapshotAndSentCv:
    def _with_docs(self, db, n=1):
        job_id, app_id = _tracked(
            db, n, job_url=f"https://boards.example.com/acme/{n}", location="Bellevue, WA")
        conn = db._get_thread_conn()
        conn.execute("UPDATE scrapes SET tier='strong', match_level='HIGH_MATCH', confidence=0.9, "
                     "keyword_category='youtube_and_creators', cv_quality_flag='PASSED', "
                     "comp_text='$90K-$110K', employment_type='full_time' WHERE id=?", (job_id,))
        conn.commit()
        version_id = db.record_document_version(
            job_id, "gen-1", "cv/20261003/a.docx", "cl/20261003/a.docx", {"mode": "B0"})
        db.update_job_documents(job_id, cv_path="cv/20261003/a.docx", cl_path="cl/20261003/a.docx")
        return job_id, app_id, version_id

    def test_first_applied_stores_snapshot_and_inferred_link(self, tmp_db):
        job_id, app_id, version_id = self._with_docs(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)

        snap = json.loads(_events(tmp_db, app_id)[-1]["note"])
        assert snap["tier"] == "strong" and snap["match_level"] == "HIGH_MATCH"
        assert snap["confidence"] == 0.9
        assert snap["keyword_category"] == "youtube_and_creators"
        assert snap["cv_quality_flag"] == "PASSED"
        assert snap["cv_path"] == "cv/20261003/a.docx"
        assert snap["document_version_id"] == version_id
        assert snap["comp_text"] == "$90K-$110K" and snap["employment_type"] == "full_time"
        assert snap["location"] == "Bellevue, WA"
        assert snap["source_host"] == "boards.example.com"
        assert snap["posting_length"] == len("Make shows. " * 20)

        link = tmp_db.get_sent_document(app_id)
        assert (link["provenance"], link["answer"]) == ("inferred", None)
        assert (link["version_id"], link["cv_path"]) == (version_id, "cv/20261003/a.docx")

    def test_only_the_first_move_to_applied_is_snapshotted(self, tmp_db):
        _, app_id, _ = self._with_docs(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)
        tmp_db.update_application(app_id, "Ghosted", None)
        tmp_db.update_application(app_id, "Applied", None)        # round-trip
        applied = [e for e in _events(tmp_db, app_id) if e["to_status"] == "Applied"]
        assert len(applied) == 2
        assert applied[0]["note"] and applied[1]["note"] is None
        links = tmp_db._get_thread_conn().execute(
            "SELECT COUNT(*) FROM application_artifact_links WHERE app_id = ?", (app_id,)).fetchone()[0]
        assert links == 1

    def test_misselected_applied_undone_then_real_apply_links_the_cv_she_sent(self, tmp_db):
        """To Apply -> Applied (wrong option) -> To Apply, the CV is
        regenerated, then the candidate really applies: the snapshot and the sent-CV
        link follow the REAL application (v2), not the mis-select (v1)."""
        job_id, app_id, v1 = self._with_docs(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)       # mis-select
        tmp_db.update_application(app_id, "To Apply", None)      # undone
        v2 = tmp_db.record_document_version(job_id, "gen-2", "cv/v2.docx", "cl/v2.docx", {})
        tmp_db.update_job_documents(job_id, cv_path="cv/v2.docx", cl_path="cl/v2.docx")
        tmp_db.update_application(app_id, "Applied", None)       # the real application

        applied = [e for e in _events(tmp_db, app_id) if e["to_status"] == "Applied"]
        assert len(applied) == 2 and applied[1]["note"]
        snap = json.loads(applied[1]["note"])
        assert (snap["cv_path"], snap["document_version_id"]) == ("cv/v2.docx", v2)
        link = tmp_db.get_sent_document(app_id)
        assert (link["version_id"], link["cv_path"], link["provenance"]) == (v2, "cv/v2.docx", "inferred")

        assert tmp_db.set_sent_document(app_id, "as_is") is True
        link = tmp_db.get_sent_document(app_id)
        assert (link["version_id"], link["cv_path"], link["answer"], link["provenance"]) == \
            (v2, "cv/v2.docx", "as_is", "confirmed")
        assert v1 != v2

    def test_card_from_before_the_ledger_gets_no_late_apply_snapshot(self, tmp_db):
        """Applied months ago (no ledger yet), then Ghosted -> Applied after
        the deploy: that is not the candidate's application day — no snapshot, no link."""
        job_id = _job(tmp_db)
        conn = tmp_db._get_thread_conn()
        conn.execute("INSERT INTO applications (job_id, status, date_applied) "
                     "VALUES (?, 'Applied', '2020-01-15')", (job_id,))
        conn.execute("UPDATE scrapes SET cv_path='cv/today.docx' WHERE id=?", (job_id,))
        conn.commit()
        app_id = conn.execute("SELECT id FROM applications WHERE job_id=?", (job_id,)).fetchone()[0]
        tmp_db.update_application(app_id, "Ghosted", None)
        _age_events(tmp_db, app_id, 60)
        tmp_db.update_application(app_id, "Applied", None)
        applied = [e for e in _events(tmp_db, app_id) if e["to_status"] == "Applied"]
        assert len(applied) == 1 and applied[0]["note"] is None
        assert tmp_db.get_sent_document(app_id) is None
        assert tmp_db.get_application(app_id)["date_applied"] == "2020-01-15"

    def test_a_snapshot_read_failure_does_not_block_the_move(self, tmp_db, monkeypatch):
        """The snapshot is analysis data; the candidate's move to Applied must still land."""
        _, app_id = _tracked(tmp_db)

        def broken_snapshot(cursor, job_id):
            cursor.execute("SELECT no_such_column FROM scrapes")   # a real sqlite error

        monkeypatch.setattr(tmp_db, "_apply_snapshot", broken_snapshot)
        assert tmp_db.update_application(app_id, "Applied", None) is True
        assert _status(tmp_db, app_id) == "Applied"
        e = _events(tmp_db, app_id)[-1]
        assert e["to_status"] == "Applied"
        assert json.loads(e["note"])["error"] == "snapshot read failed"

    def test_applied_without_a_cv_links_as_unknown(self, tmp_db):
        _, app_id = _tracked(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)
        link = tmp_db.get_sent_document(app_id)
        assert (link["provenance"], link["cv_path"], link["answer"]) == ("unknown", None, None)

    def test_set_sent_document_keeps_the_apply_time_version(self, tmp_db):
        job_id, app_id, version_id = self._with_docs(tmp_db)
        tmp_db.update_application(app_id, "Applied", None)
        # a regeneration AFTER applying must not change what the candidate is recorded as sending
        tmp_db.record_document_version(job_id, "gen-2", "cv/20261004/b.docx", None, {})
        tmp_db.update_job_documents(job_id, cv_path="cv/20261004/b.docx")

        assert tmp_db.set_sent_document(app_id, "edited") is True
        link = tmp_db.get_sent_document(app_id)
        assert (link["answer"], link["provenance"]) == ("edited", "confirmed")
        assert (link["version_id"], link["cv_path"]) == (version_id, "cv/20261003/a.docx")

        assert tmp_db.set_sent_document(app_id, "own") is True     # the candidate can change it
        assert tmp_db.get_sent_document(app_id)["answer"] == "own"

    def test_set_sent_document_rejects_bad_input(self, tmp_db):
        _, app_id = _tracked(tmp_db)
        assert tmp_db.set_sent_document(app_id, "maybe") is False
        assert tmp_db.set_sent_document(app_id, None) is False           # confirmed needs an answer
        assert tmp_db.set_sent_document(999999, "as_is") is False
        assert tmp_db.get_sent_document(app_id) is None

    def test_card_applied_before_the_ledger_uses_the_cv_on_the_card(self, tmp_db):
        job_id = _job(tmp_db)
        conn = tmp_db._get_thread_conn()
        conn.execute("INSERT INTO applications (job_id, status) VALUES (?, 'Applied')", (job_id,))
        conn.execute("UPDATE scrapes SET cv_path='cv/old.docx' WHERE id=?", (job_id,))
        conn.commit()
        app_id = conn.execute("SELECT id FROM applications WHERE job_id=?", (job_id,)).fetchone()[0]
        assert tmp_db.set_sent_document(app_id, "as_is") is True
        link = tmp_db.get_sent_document(app_id)
        assert (link["answer"], link["cv_path"], link["version_id"]) == ("as_is", "cv/old.docx", None)


# ── saved document versions ──────────────────────────────────────────────────

class TestDocumentVersions:
    def test_versions_are_appended_newest_first(self, tmp_db):
        job_id = _job(tmp_db)
        v1 = tmp_db.record_document_version(job_id, "gen-1", "cv/a.docx", "cl/a.docx",
                                            {"content_mode": "B0", "truth": "passed"})
        v2 = tmp_db.record_document_version(job_id, "gen-2", "cv/b.docx", None, {"n": 2})
        assert 0 < v1 < v2
        versions = tmp_db.get_document_versions(job_id)
        assert [v["id"] for v in versions] == [v2, v1]
        assert versions[1]["meta"] == {"content_mode": "B0", "truth": "passed"}
        assert versions[1]["cv_path"] == "cv/a.docx" and versions[1]["cl_path"] == "cl/a.docx"
        assert versions[0]["generation_id"] == "gen-2" and versions[0]["cl_path"] is None
        assert versions[0]["created_at"]
        assert tmp_db.get_document_versions(job_id + 100) == []

    def test_update_job_documents_sets_only_what_is_given(self, tmp_db):
        job_id = _job(tmp_db)
        assert tmp_db.update_job_documents(job_id, cv_path="cv/a.docx", cl_path="cl/a.docx") is True
        assert tmp_db.update_job_documents(job_id, cv_path="cv/b.docx") is True
        job = tmp_db.get_job_details(job_id)
        assert (job["cv_path"], job["cl_path"]) == ("cv/b.docx", "cl/a.docx")
        assert tmp_db.update_job_documents(job_id, cl_path="cl/b.docx") is True
        job = tmp_db.get_job_details(job_id)
        assert (job["cv_path"], job["cl_path"]) == ("cv/b.docx", "cl/b.docx")
        assert tmp_db.update_job_documents(job_id) is True               # nothing to do
        assert tmp_db.update_job_documents(987654, cv_path="cv/x.docx") is False


# ── API path ─────────────────────────────────────────────────────────────────

class TestApiPath:
    def test_put_writes_event_and_triggers_side_effects(self, env):
        _, app_id = _tracked(env.db)
        r = env.client.put(f"/api/applications/{app_id}", json={"status": "Applied"})
        assert r.status_code == 200
        assert r.json()["intel_spawned"] is True and r.json()["prep_spawned"] is False
        assert env.spawns == {"intel": [app_id], "prep": []}
        e = _events(env.db, app_id)[-1]
        assert (e["to_status"], e["source"]) == ("Applied", "app")

        r = env.client.put(f"/api/applications/{app_id}", json={"status": "Interview"})
        assert r.json()["prep_spawned"] is True
        assert env.spawns == {"intel": [app_id], "prep": [app_id]}

    def test_offer_triggers_prep(self, env):
        _, app_id = _tracked(env.db)
        r = env.client.put(f"/api/applications/{app_id}", json={"status": "Offer"})
        assert r.json()["prep_spawned"] is True
        assert env.spawns["prep"] == [app_id]

    def test_same_status_put_writes_nothing(self, env):
        _, app_id = _tracked(env.db)
        env.client.put(f"/api/applications/{app_id}", json={"status": "Applied"})
        before = len(_events(env.db, app_id))
        # the Kanban resends the status on every notes save
        r = env.client.put(f"/api/applications/{app_id}", json={"status": "Applied", "notes": "x"})
        r2 = env.client.put(f"/api/applications/{app_id}", json={"notes": "y"})
        assert r.status_code == 200 and r2.status_code == 200
        assert r.json()["intel_spawned"] is False and r.json()["prep_spawned"] is False
        assert len(_events(env.db, app_id)) == before
        assert env.spawns == {"intel": [app_id], "prep": []}        # not re-triggered

    def test_failed_save_is_reported_not_swallowed(self, env, monkeypatch):
        _, app_id = _tracked(env.db)
        monkeypatch.setattr(env.db, "update_application", lambda *a, **k: False)
        r = env.client.put(f"/api/applications/{app_id}", json={"status": "Applied"})
        assert r.status_code == 500
        assert env.spawns == {"intel": [], "prep": []}

    def test_applications_rows_carry_the_history(self, env):
        _, app_id = _tracked(env.db)
        env.client.put(f"/api/applications/{app_id}", json={"status": "Applied"})
        env.client.put(f"/api/applications/{app_id}", json={"status": "Interview"})
        _age_events(env.db, app_id, 60)
        env.client.put(f"/api/applications/{app_id}", json={"status": "Rejected"})

        row = env.client.get("/api/applications").json()[0]
        assert row["status"] == "Rejected"
        assert row["highest_stage"] == "Interview"
        assert row["reached_at"] and row["reached_inferred"] is False
        assert [h["status"] for h in row["history"]] == ["Applied", "Interview", "Rejected"]
        assert set(row["history"][0]) == {"status", "at", "inferred"}
        assert row["sent_cv"] is None
        assert row["has_prep"] is False

        env.db.update_application_prep(app_id, "ready", "/tmp/prep.md")
        assert env.client.get("/api/applications").json()[0]["has_prep"] is True

    def test_sent_cv_endpoint(self, env):
        job_id, app_id = _tracked(env.db)
        env.db.update_job_documents(job_id, cv_path="cv/a.docx")
        env.client.put(f"/api/applications/{app_id}", json={"status": "Applied"})

        r = env.client.put(f"/api/applications/{app_id}/sent-cv", json={"answer": "as_is"})
        assert r.status_code == 200 and r.json() == {"app_id": app_id, "sent_cv": "as_is"}
        link = env.db.get_sent_document(app_id)
        assert (link["answer"], link["provenance"], link["cv_path"]) == \
            ("as_is", "confirmed", "cv/a.docx")
        assert env.client.get("/api/applications").json()[0]["sent_cv"] == "as_is"

        # changeable: the latest answer wins
        r = env.client.put(f"/api/applications/{app_id}/sent-cv", json={"answer": "own"})
        assert r.status_code == 200
        assert env.client.get("/api/applications").json()[0]["sent_cv"] == "own"

    def test_sent_cv_endpoint_validates(self, env):
        _, app_id = _tracked(env.db)
        assert env.client.put(f"/api/applications/{app_id}/sent-cv",
                              json={"answer": "kind of"}).status_code == 400
        assert env.client.put(f"/api/applications/{app_id}/sent-cv", json={}).status_code == 422
        assert env.client.put("/api/applications/999999/sent-cv",
                              json={"answer": "as_is"}).status_code == 404
        assert env.db.get_sent_document(app_id) is None

    def test_stats_counts_reached_interview_all_time(self, env):
        _, kept = _tracked(env.db, 1)        # real interview, later rejected
        _, slip = _tracked(env.db, 2)        # accidental drag
        _, plain = _tracked(env.db, 3)       # applied only
        for app_id in (kept, slip, plain):
            env.client.put(f"/api/applications/{app_id}", json={"status": "Applied"})
        env.client.put(f"/api/applications/{kept}", json={"status": "Interview"})
        _age_events(env.db, kept, 60)
        env.client.put(f"/api/applications/{kept}", json={"status": "Rejected"})
        env.client.put(f"/api/applications/{slip}", json={"status": "Screening Call"})
        env.client.put(f"/api/applications/{slip}", json={"status": "Applied"})

        data = env.client.get("/api/stats/overview").json()
        assert data["reached_interview_total"] == 1
        assert data["reached_interview_unconfirmed"] == 0
        # the current-status numbers are unchanged in meaning
        assert data["funnel"]["in_process"] == 0
        assert data["statuses"] == {"Applied": 2, "Rejected": 1}

    def test_stats_counts_cards_from_before_the_ledger(self, env):
        """Right after deploy, before any backfill: cards already sitting at
        an interview stage are facts and count; a Rejected card with only a
        prep sheet does not (nothing proves it yet)."""
        conn = env.db._get_thread_conn()
        for n, status, prep in ((1, "Interview", "2020-01-08 10:00:00"),
                                (2, "Screening Call", "2020-01-15 11:30:00"),
                                (3, "Rejected", "2020-01-01 09:15:00"),
                                (4, "Applied", None)):
            _legacy_app(conn, _job(env.db, n), status, prep=prep)
        conn.commit()
        assert env.db.count_reached_interview() == {"confirmed": 2, "unconfirmed": 0}
        data = env.client.get("/api/stats/overview?refresh=true").json()
        assert (data["reached_interview_total"], data["reached_interview_unconfirmed"]) == (2, 0)

    def test_correct_endpoint_and_events_listing(self, env):
        _, app_id = _tracked(env.db)
        env.client.put(f"/api/applications/{app_id}", json={"status": "Interview"})
        _age_events(env.db, app_id, 60)
        assert env.client.get("/api/stats/overview").json()["reached_interview_total"] == 1

        listing = env.client.get(f"/api/applications/{app_id}/events").json()
        entry = listing["events"][-1]
        assert listing["reach"]["highest_stage"] == "Interview"

        r = env.client.post(f"/api/applications/{app_id}/events/{entry['id']}/correct",
                            json={"note": "mis-click"})
        assert r.status_code == 200
        assert r.json()["reach"]["highest_stage"] == "To Apply"
        # the stats cache was invalidated by the correction
        assert env.client.get("/api/stats/overview").json()["reached_interview_total"] == 0
        after = env.client.get(f"/api/applications/{app_id}/events").json()["events"]
        assert len(after) == len(listing["events"]) + 1           # nothing deleted
        assert after[-1]["event_type"] == "correction" and after[-1]["source"] == "maintainer"

        # no body is fine; a wrong application / event is a 404
        assert env.client.post(
            f"/api/applications/{app_id}/events/{r.json()['correction_event_id']}/correct"
        ).status_code == 200
        assert env.client.post(
            f"/api/applications/{app_id + 50}/events/{entry['id']}/correct").status_code == 404
        assert env.client.post(
            f"/api/applications/{app_id}/events/999999/correct").status_code == 404


# ── chat path ────────────────────────────────────────────────────────────────

class TestChatPath:
    def test_chat_move_writes_event_and_triggers_side_effects(self, env):
        from job_agent import chat_tools as ct
        _, app_id = _tracked(env.db)

        out = ct.update_application_status(env.db, app_id, "Applied", notes="via portal")
        assert out["updated"] is True and out["intel_spawned"] is True
        assert env.spawns == {"intel": [app_id], "prep": []}
        e = _events(env.db, app_id)[-1]
        assert (e["event_type"], e["to_status"], e["source"]) == ("status_change", "Applied", "chat")

        out = ct.update_application_status(env.db, app_id, "Screening Call")
        assert out["prep_spawned"] is True
        assert env.spawns == {"intel": [app_id], "prep": [app_id]}
        assert _events(env.db, app_id)[-1]["source"] == "chat"

        # same status again: no event, no side effect
        before = len(_events(env.db, app_id))
        ct.update_application_status(env.db, app_id, "Screening Call")
        assert len(_events(env.db, app_id)) == before
        assert env.spawns == {"intel": [app_id], "prep": [app_id]}

    def test_api_and_chat_moves_share_one_ledger(self, env):
        from job_agent import chat_tools as ct
        _, app_id = _tracked(env.db)
        env.client.put(f"/api/applications/{app_id}", json={"status": "Applied"})
        ct.update_application_status(env.db, app_id, "Interview")
        moves = [(e["to_status"], e["source"]) for e in _events(env.db, app_id, "status_change")]
        assert moves == [("Applied", "app"), ("Interview", "chat")]

    def test_tool_on_another_database_never_spawns(self, env, tmp_path):
        """Side effects belong to the API's own database. A tool call against
        any other one (a test DB, a script) still records the move but must
        not start a generation."""
        from job_agent import chat_tools as ct
        other = DBManager(db_path=str(tmp_path / "other.db"))
        try:
            _, app_id = _tracked(other)
            out = ct.update_application_status(other, app_id, "Applied")
            assert out["updated"] is True and "intel_spawned" not in out
            assert env.spawns == {"intel": [], "prep": []}
            assert _events(other, app_id)[-1]["source"] == "chat"
        finally:
            other.close()

    def test_chat_track_tags_its_source(self, env):
        from job_agent import chat_tools as ct
        job_id = _job(env.db)
        assert ct.track_job(env.db, job_id)["tracked"] is True
        app_id = env.db.get_applications()[0]["app_id"]
        assert _events(env.db, app_id)[0]["source"] == "chat"

    def test_chat_dismiss_runs_the_location_feedback(self, env):
        from job_agent import chat_tools as ct
        far = _job(env.db, 1, location="Austin, TX")
        out = ct.dismiss_job(env.db, far, reason="wrong_location")
        assert out["dismissed"] is True and out["suppressed_location"] is None   # one vote
        assert env.db.get_job_details(far)["rejection_reason"] == "wrong_location"
        conn = env.db._get_thread_conn()                      # an earlier day's dismissal
        conn.execute("UPDATE location_votes SET voted_at = datetime('now', '-1 day')")
        conn.commit()
        far2 = _job(env.db, 4, location="Austin, TX")
        assert ct.dismiss_job(env.db, far2, reason="wrong_location")["suppressed_location"] == "austin, tx"
        assert "austin, tx" in env.db.get_suppressed_locations()

        # same rules as the HTTP endpoint: no suppression without a location
        # reason, and never for remote
        other = _job(env.db, 2, location="Denver, CO")
        assert ct.dismiss_job(env.db, other, reason="too_junior")["suppressed_location"] is None
        remote = _job(env.db, 3, location="Remote")
        assert ct.dismiss_job(env.db, remote, reason="wrong_location")["suppressed_location"] is None
        assert env.db.get_suppressed_locations() == {"austin, tx"}

    def test_http_dismiss_still_suppresses(self, env):
        far = _job(env.db, 1, location="Austin, TX")
        r = env.client.post(f"/api/jobs/{far}/dismiss", json={"reason": "wrong location"})
        assert r.json()["suppressed_location"] is None                       # one vote
        conn = env.db._get_thread_conn()                      # an earlier day's dismissal
        conn.execute("UPDATE location_votes SET voted_at = datetime('now', '-1 day')")
        conn.commit()
        far2 = _job(env.db, 2, location="Austin, TX")
        r = env.client.post(f"/api/jobs/{far2}/dismiss", json={"reason": "wrong location"})
        assert r.json()["suppressed_location"] == "austin, tx"


# ── interview prep per stage ─────────────────────────────────────────────────

class TestPrepPerStage:
    @pytest.fixture
    def prep_env(self, tmp_path, monkeypatch):
        """Real _spawn_interview_prep, fake generator that accepts stage=."""
        from fastapi.testclient import TestClient
        from job_agent import api
        db = DBManager(db_path=str(tmp_path / "prep.db"))
        monkeypatch.setattr(api, "_db", db)
        calls = []

        def fake_generate(app_id, db_, stage=None):
            calls.append(stage)
            db_.update_application_prep(app_id, "ready", str(tmp_path / "prep.md"))
            return {"app_id": app_id}

        monkeypatch.setattr("job_agent.interview_prep.generate_interview_prep",
                            fake_generate)
        _, app_id = _tracked(db)
        yield SimpleNamespace(client=TestClient(api.app), db=db, api=api,
                              calls=calls, app_id=app_id)
        db.close()

    def _move(self, e, status):
        return e.client.put(f"/api/applications/{e.app_id}", json={"status": status}).json()

    def _built_for(self, e):
        return e.db.get_application(e.app_id)["prep_stage"]

    def test_prep_regenerates_only_for_a_higher_stage(self, prep_env):
        e = prep_env
        assert self._move(e, "Screening Call")["prep_spawned"] is True
        assert _wait_for(lambda: self._built_for(e) == "Screening Call")
        assert e.calls == ["Screening Call"]

        assert self._move(e, "Interview")["prep_spawned"] is True       # higher -> rebuilt
        assert _wait_for(lambda: self._built_for(e) == "Interview")
        assert e.calls == ["Screening Call", "Interview"]

        _age_events(e.db, e.app_id, 60)
        assert self._move(e, "Screening Call")["prep_spawned"] is False  # lower -> kept
        assert self._move(e, "Interview")["prep_spawned"] is False       # round-trip -> kept
        assert self._move(e, "Rejected")["prep_spawned"] is False
        assert e.calls == ["Screening Call", "Interview"]
        assert self._built_for(e) == "Interview"

    def test_offer_builds_a_sheet_for_the_offer_stage(self, prep_env):
        e = prep_env
        assert self._move(e, "Offer")["prep_spawned"] is True
        assert _wait_for(lambda: self._built_for(e) == "Offer")
        assert e.calls == ["Offer"]

    def test_sheet_from_before_prep_stage_counts_as_earliest_stage(self, prep_env):
        e = prep_env
        conn = e.db._get_thread_conn()
        conn.execute("UPDATE applications SET status='Screening Call', prep_status='ready' "
                     "WHERE id=?", (e.app_id,))
        conn.commit()
        assert e.api._spawn_interview_prep(e.app_id) is False            # same stage: keep
        assert self._move(e, "Interview")["prep_spawned"] is True        # higher: rebuild
        assert _wait_for(lambda: self._built_for(e) == "Interview")

    def test_move_up_while_a_sheet_is_building_gets_a_fresh_sheet(self, prep_env, monkeypatch):
        """Screening Call, then Interview a moment later: the second move finds
        the row busy. When the first sheet lands, the higher stage is built."""
        e = prep_env
        release = threading.Event()
        calls = []

        def slow_generate(app_id, db_, stage=None):
            calls.append(stage)
            if len(calls) == 1:
                release.wait(5)
            db_.update_application_prep(app_id, "ready", "/tmp/prep.md")
            return {"app_id": app_id}

        monkeypatch.setattr("job_agent.interview_prep.generate_interview_prep",
                            slow_generate)
        assert self._move(e, "Screening Call")["prep_spawned"] is True
        assert _wait_for(lambda: calls == ["Screening Call"])
        assert self._move(e, "Interview")["prep_spawned"] is False       # row is busy
        release.set()
        assert _wait_for(lambda: self._built_for(e) == "Interview")
        assert calls == ["Screening Call", "Interview"]
        time.sleep(0.1)
        assert calls == ["Screening Call", "Interview"]                  # and it stops there

    def test_generator_without_stage_keyword_still_works(self, prep_env, monkeypatch):
        e = prep_env
        legacy_calls = []

        def legacy_generate(app_id, db_):
            legacy_calls.append(app_id)
            db_.update_application_prep(app_id, "ready", "/tmp/legacy.md")
            return {"app_id": app_id}

        monkeypatch.setattr("job_agent.interview_prep.generate_interview_prep",
                            legacy_generate)
        assert self._move(e, "Interview")["prep_spawned"] is True
        assert _wait_for(lambda: self._built_for(e) == "Interview")
        assert legacy_calls == [e.app_id]                                # called exactly once

    def test_a_real_typeerror_is_not_retried(self, prep_env, monkeypatch):
        e = prep_env
        attempts = []

        def broken_generate(app_id, db_, stage=None):
            attempts.append(stage)
            raise TypeError("unsupported operand")

        monkeypatch.setattr("job_agent.interview_prep.generate_interview_prep",
                            broken_generate)
        assert self._move(e, "Interview")["prep_spawned"] is True
        assert _wait_for(lambda: e.db.get_application(e.app_id)["prep_status"] == "failed")
        assert attempts == ["Interview"]                                 # no second paid call
        assert self._built_for(e) is None

    def _ready_screening_sheet(self, e, tmp_path, on_disk=True):
        """A ready Screening Call sheet built a while ago."""
        sheet = tmp_path / "prep_screening.md"
        if on_disk:
            sheet.write_text("## Screening call sheet\n" + "x" * 600)
        e.db.update_application(e.app_id, "Applied", None)
        e.db.update_application(e.app_id, "Screening Call", None)
        e.db.update_application_prep(e.app_id, "ready", str(sheet))
        e.db.set_application_prep_stage(e.app_id, "Screening Call")
        conn = e.db._get_thread_conn()
        conn.execute("UPDATE applications SET prep_generated_at = '2020-01-15 11:30:00' "
                     "WHERE id = ?", (e.app_id,))
        conn.commit()
        _age_events(e.db, e.app_id, 60)
        return str(sheet)

    def _prep_row(self, e):
        row = e.db.get_application(e.app_id)
        return (row["prep_status"], row["prep_path"], row["prep_stage"], row["prep_generated_at"])

    @pytest.mark.parametrize("how", ["quality_floor", "exception"])
    def test_failed_automatic_rebuild_keeps_the_previous_sheet(self, prep_env, monkeypatch,
                                                               tmp_path, how):
        """Screening Call sheet ready; the card moves up to Interview and the
        automatic rebuild fails: the old sheet stays readable, as it was."""
        e = prep_env
        sheet = self._ready_screening_sheet(e, tmp_path)
        calls = []

        def failing_generate(app_id, db_, stage=None):
            calls.append(stage)
            if how == "exception":
                raise RuntimeError("synthesis crashed")
            db_.update_application_prep(app_id, "failed")
            return {"error": "interview-prep synthesis failed or below quality floor"}

        monkeypatch.setattr("job_agent.interview_prep.generate_interview_prep",
                            failing_generate)
        assert self._move(e, "Interview")["prep_spawned"] is True
        assert _wait_for(lambda: calls == ["Interview"]
                         and e.db.get_application(e.app_id)["prep_status"] != "pending")
        time.sleep(0.1)
        assert self._prep_row(e) == ("ready", sheet, "Screening Call", "2020-01-15 11:30:00")
        # and once the card is Rejected the sheet is still on it
        e.db.update_application(e.app_id, "Rejected", None)
        card = next(a for a in e.db.get_applications() if a["app_id"] == e.app_id)
        assert card["has_prep"] is True and card["highest_stage"] == "Interview"

    def test_failed_rebuild_with_the_old_file_gone_stays_failed(self, prep_env, monkeypatch,
                                                                 tmp_path):
        e = prep_env
        self._ready_screening_sheet(e, tmp_path, on_disk=False)

        def failing_generate(app_id, db_, stage=None):
            db_.update_application_prep(app_id, "failed")
            return {"error": "below quality floor"}

        monkeypatch.setattr("job_agent.interview_prep.generate_interview_prep",
                            failing_generate)
        self._move(e, "Interview")
        assert _wait_for(lambda: e.db.get_application(e.app_id)["prep_status"] == "failed")
        time.sleep(0.1)
        assert e.db.get_application(e.app_id)["prep_status"] == "failed"

    def test_failed_forced_rebuild_is_not_masked(self, prep_env, monkeypatch, tmp_path):
        """'Prep me' is an explicit request: its failure shows as a failure."""
        e = prep_env
        self._ready_screening_sheet(e, tmp_path)

        def failing_generate(app_id, db_, stage=None):
            db_.update_application_prep(app_id, "failed")
            return {"error": "below quality floor"}

        monkeypatch.setattr("job_agent.interview_prep.generate_interview_prep",
                            failing_generate)
        assert e.api._spawn_interview_prep(e.app_id, force=True) is True
        assert _wait_for(lambda: e.db.get_application(e.app_id)["prep_status"] == "failed")
        time.sleep(0.1)
        assert e.db.get_application(e.app_id)["prep_status"] == "failed"

    def test_failed_generation_records_no_stage(self, prep_env, monkeypatch):
        e = prep_env

        def failing_generate(app_id, db_, stage=None):
            db_.update_application_prep(app_id, "failed")
            return {"error": "below quality floor"}

        monkeypatch.setattr("job_agent.interview_prep.generate_interview_prep",
                            failing_generate)
        self._move(e, "Interview")
        assert _wait_for(lambda: e.db.get_application(e.app_id)["prep_status"] == "failed")
        time.sleep(0.1)
        assert self._built_for(e) is None


# ── manual add with no link (an inbound process) ─────────────────────────────

class TestManualAddInbound:
    def test_no_link_job_goes_straight_to_interview_with_history(self, env, monkeypatch):
        """A recruiter reached out directly: there is no posting URL. The
        candidate adds the role by hand and moves it straight to an interview
        stage."""
        # No background document generation from a test (two layers).
        no_threads = _NoThreads()
        monkeypatch.setattr(env.api, "threading", no_threads)
        monkeypatch.setattr("job_agent.docs_builder.save_documents_for_job",
                            lambda *a, **k: (None, None, None, None, None))
        monkeypatch.setattr("job_agent.gpt_filter.load_profile_text", lambda *a, **k: "")

        # No description either: there is no posting to paste.
        r = env.client.post("/api/jobs", json={
            "title": "Senior Producer", "company": "Inbound Co"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        assert no_threads.suppressed >= 1

        row = env.client.get("/api/applications").json()[0]
        assert row["job_id"] == job_id and row["status"] == "To Apply"
        assert row["link"].startswith("manual://")
        app_id = row["app_id"]
        assert [e["event_type"] for e in _events(env.db, app_id)] == ["tracked"]

        r = env.client.put(f"/api/applications/{app_id}", json={"status": "Interview"})
        assert r.status_code == 200 and r.json()["prep_spawned"] is True
        assert env.spawns == {"intel": [], "prep": [app_id]}             # the candidate never "applied"

        row = env.client.get("/api/applications").json()[0]
        assert row["status"] == "Interview" and row["highest_stage"] == "Interview"
        assert [h["status"] for h in row["history"]] == ["Interview"]
        assert row["history"][0]["at"]
        e = _events(env.db, app_id)[-1]
        assert (e["from_status"], e["to_status"], e["company"]) == \
            ("To Apply", "Interview", "Inbound Co")
        assert env.client.get("/api/stats/overview").json()["reached_interview_total"] == 1

        # if the candidate later applies formally, the snapshot knows it was a manual add
        _age_events(env.db, app_id, 60)
        env.client.put(f"/api/applications/{app_id}", json={"status": "Applied"})
        snap = json.loads(_events(env.db, app_id)[-1]["note"])
        assert snap["source_host"] == "manual" and snap["match_level"] == "USER_INPUT"


# ── cards from before the ledger ─────────────────────────────────────────────

def _legacy_app(conn, job_id, status, app_id=None, prep=None):
    """An application row as it existed before the ledger (no events)."""
    conn.execute(
        "INSERT INTO applications (id, job_id, status, date_applied, prep_status, prep_generated_at) "
        "VALUES (?, ?, ?, '2020-01-01', ?, ?)",
        (app_id, job_id, status, "ready" if prep else None, prep))
    return conn.execute("SELECT id FROM applications WHERE job_id = ?", (job_id,)).fetchone()[0]


# ── the candidate's answer on a closed card (the outcome form) ─────────────────────────

def _closed(db, n=1, path=("Applied", "Rejected")):
    """A card moved through `path` on the board (never an accidental drag)."""
    _, app_id = _tracked(db, n)
    for status in path:
        db.update_application(app_id, status, None)
        _age_events(db, app_id, 60)
    return app_id


def _pre_ledger_closed(db, n=1, guess="Screening Call", snapshot="Rejected"):
    """A card from before the ledger: its backfilled snapshot plus, when
    `guess` is set, the prep-sheet proxy for that stage."""
    job_id = _job(db, n)
    conn = db._get_thread_conn()
    app_id = _legacy_app(conn, job_id, "Rejected")
    conn.commit()
    db.record_application_event(app_id, job_id, "snapshot", to_status=snapshot,
                                source="backfill", note='{"snapshot": "ledger_start"}')
    if guess:
        db.record_application_event(app_id, job_id, "stage_reach", to_status=guess,
                                    effective_at="2020-01-01 09:15:00",
                                    source="backfill_prep_proxy",
                                    note="inferred from the prep sheet; stage unconfirmed")
    return app_id


class TestOutcomeAnswer:
    def test_answer_names_a_stage_the_board_skipped(self, tmp_db):
        """Moved Applied -> Rejected, but the candidate had a screening call."""
        app_id = _closed(tmp_db)
        assert tmp_db.set_application_outcome(
            app_id, "Screening Call", "  Test answer:\n reason V ") == "saved"
        card = tmp_db.get_card_outcome(app_id)
        assert card["highest_stage"] == "Screening Call" and card["reached_inferred"] is False
        assert [h["status"] for h in card["history"]] == ["Applied", "Screening Call", "Rejected"]
        assert card["history"][1]["at"] is None            # the candidate gave no date; none is guessed
        assert card["outcome"]["stage"] == "Screening Call"
        assert card["outcome"]["what_happened"] == "Test answer: reason V"
        assert card["outcome"]["at"]
        assert tmp_db.count_reached_interview() == {"confirmed": 1, "unconfirmed": 0}
        assert _status(tmp_db, app_id) == "Rejected"        # the card itself does not move
        # the Kanban rows carry it too
        row = next(a for a in tmp_db.get_applications() if a["app_id"] == app_id)
        assert row["outcome"] == card["outcome"] and row["highest_stage"] == "Screening Call"

    def test_same_stage_writes_only_the_answer(self, tmp_db):
        app_id = _closed(tmp_db, path=("Applied", "Interview", "Rejected"))
        before = len(_events(tmp_db, app_id))
        assert tmp_db.set_application_outcome(app_id, "Interview") == "saved"
        events = _events(tmp_db, app_id)
        assert [e["event_type"] for e in events[before:]] == ["outcome"]
        assert (events[-1]["source"], events[-1]["from_status"]) == ("outcome_form", "Rejected")
        assert json.loads(events[-1]["note"]) == {"stage": "Interview", "what_happened": None}
        assert tmp_db.get_card_outcome(app_id)["outcome"]["what_happened"] is None

    def test_lower_answer_invalidates_and_a_later_answer_restores(self, tmp_db):
        app_id = _closed(tmp_db, path=("Applied", "Interview", "Rejected"))
        interview = next(e for e in _events(tmp_db, app_id) if e["to_status"] == "Interview")
        assert tmp_db.set_application_outcome(app_id, "Applied") == "saved"
        assert tmp_db.get_stage_reach(app_id)["highest_stage"] == "Applied"
        assert tmp_db.count_reached_interview()["confirmed"] == 0
        assert next(e for e in _events(tmp_db, app_id) if e["id"] == interview["id"])["valid"] is False

        assert tmp_db.set_application_outcome(app_id, "Interview") == "saved"
        reach = tmp_db.get_stage_reach(app_id)
        assert reach["highest_stage"] == "Interview"
        assert reach["reached_at"] == interview["effective_at"]   # the real move, date and all
        assert [h["status"] for h in reach["history"]] == ["Applied", "Interview", "Rejected"]
        assert _events(tmp_db, app_id, "stage_reach") == []       # nothing invented
        assert tmp_db.count_reached_interview()["confirmed"] == 1

    def test_a_new_answer_supersedes_her_earlier_one(self, tmp_db):
        app_id = _closed(tmp_db)
        tmp_db.set_application_outcome(app_id, "Screening Call")
        tmp_db.set_application_outcome(app_id, "Interview")
        assert [h["status"] for h in tmp_db.get_stage_reach(app_id)["history"]] == \
            ["Applied", "Interview", "Rejected"]
        tmp_db.set_application_outcome(app_id, "Screening Call")
        reach = tmp_db.get_stage_reach(app_id)
        assert reach["highest_stage"] == "Screening Call"
        assert [h["status"] for h in reach["history"]] == ["Applied", "Screening Call", "Rejected"]
        # the candidate's first entry came back; no third one was added
        reaches = _events(tmp_db, app_id, "stage_reach")
        assert len(reaches) == 2
        assert [e["to_status"] for e in reaches if e["valid"]] == ["Screening Call"]

    def test_a_maintainer_correction_is_never_undone(self, tmp_db):
        app_id = _closed(tmp_db, path=("Applied", "Interview", "Rejected"))
        interview = next(e for e in _events(tmp_db, app_id) if e["to_status"] == "Interview")
        tmp_db.correct_application_event(app_id, interview["id"], source="maintainer",
                                         note="mis-click")
        assert tmp_db.set_application_outcome(app_id, "Interview") == "saved"
        assert next(e for e in _events(tmp_db, app_id) if e["id"] == interview["id"])["valid"] is False
        # the candidate's answer still sets the stage, as their own entry
        reach = tmp_db.get_stage_reach(app_id)
        assert reach["highest_stage"] == "Interview" and reach["reached_inferred"] is False
        assert [e["source"] for e in _events(tmp_db, app_id, "stage_reach")] == ["outcome_form"]

    def test_answer_replaces_the_proxy_guess(self, tmp_db):
        """A pre-ledger card known only from its prep sheet (the proxy guess)."""
        app_id = _pre_ledger_closed(tmp_db)
        assert tmp_db.count_reached_interview() == {"confirmed": 0, "unconfirmed": 1}
        assert tmp_db.get_stage_reach(app_id)["reached_inferred"] is True
        assert tmp_db.set_application_outcome(app_id, "Interview", "Test answer: reason W") == "saved"
        card = tmp_db.get_card_outcome(app_id)
        assert card["highest_stage"] == "Interview" and card["reached_inferred"] is False
        assert card["history"] == [
            {"status": "Interview", "at": None, "inferred": False},
            {"status": "Rejected", "at": None, "inferred": False},
        ]
        assert tmp_db.count_reached_interview() == {"confirmed": 1, "unconfirmed": 0}

    def test_confirming_the_guessed_stage(self, tmp_db):
        app_id = _pre_ledger_closed(tmp_db)
        assert tmp_db.set_application_outcome(app_id, "Screening Call") == "saved"
        card = tmp_db.get_card_outcome(app_id)
        assert card["highest_stage"] == "Screening Call" and card["reached_inferred"] is False
        assert [h["status"] for h in card["history"]] == ["Screening Call", "Rejected"]
        assert tmp_db.count_reached_interview() == {"confirmed": 1, "unconfirmed": 0}

    def test_a_recorded_entry_beats_a_guess_for_the_same_stage(self, tmp_db):
        """No answer needed: the snapshot (a fact) says Interview and the proxy
        guessed the same stage — the card names it, without the guessed date."""
        app_id = _pre_ledger_closed(tmp_db, guess="Interview", snapshot="Interview")
        conn = tmp_db._get_thread_conn()
        conn.execute("UPDATE applications SET status = 'Interview' WHERE id = ?", (app_id,))
        conn.commit()
        tmp_db.update_application(app_id, "Rejected", None)
        reach = tmp_db.get_stage_reach(app_id)
        assert reach["highest_stage"] == "Interview"
        assert reach["reached_inferred"] is False and reach["reached_at"] is None
        assert [(h["status"], h["inferred"]) for h in reach["history"]] == \
            [("Interview", False), ("Rejected", False)]

    def test_only_a_closed_card_takes_an_answer(self, tmp_db):
        _, live = _tracked(tmp_db)
        tmp_db.update_application(live, "Interview", None)
        closed = _closed(tmp_db, 2)
        assert tmp_db.set_application_outcome(live, "Interview") == "not_closed"
        assert tmp_db.set_application_outcome(424242, "Interview") == "not_found"
        assert tmp_db.set_application_outcome(closed, "Rejected") == "bad_stage"
        assert tmp_db.set_application_outcome(closed, "To Apply") == "bad_stage"
        assert _events(tmp_db, live, "outcome") == [] and _events(tmp_db, closed, "outcome") == []
        assert tmp_db.get_card_outcome(live)["outcome"] is None

    def test_an_answer_belongs_to_its_round(self, tmp_db):
        """Re-opened, then closed again: the old answer is history, not the
        answer for the new rejection."""
        app_id = _closed(tmp_db)
        tmp_db.set_application_outcome(app_id, "Applied", "Test answer: outcome A")
        assert tmp_db.get_card_outcome(app_id)["outcome"]["what_happened"] == "Test answer: outcome A"
        tmp_db.update_application(app_id, "Interview", None)
        _age_events(tmp_db, app_id, 60)
        assert tmp_db.get_card_outcome(app_id)["outcome"] is None        # live again
        tmp_db.update_application(app_id, "Rejected", None)
        assert tmp_db.get_card_outcome(app_id)["outcome"] is None        # new round, no answer yet
        tmp_db.set_application_outcome(app_id, "Interview", "Test answer: outcome B")
        out = tmp_db.get_card_outcome(app_id)["outcome"]
        assert (out["stage"], out["what_happened"]) == ("Interview", "Test answer: outcome B")

    def test_ghosted_card_and_a_long_line(self, tmp_db):
        app_id = _closed(tmp_db, path=("Applied", "Ghosted"))
        assert tmp_db.set_application_outcome(app_id, "Applied", "x" * 900) == "saved"
        assert len(tmp_db.get_card_outcome(app_id)["outcome"]["what_happened"]) == 500

    def test_summary_for_the_stats_page(self, tmp_db):
        interview = _closed(tmp_db, 1, path=("Applied", "Interview", "Rejected"))
        screen = _closed(tmp_db, 2)
        ghosted = _closed(tmp_db, 3, path=("Applied", "Ghosted"))
        _closed(tmp_db, 4, path=("Applied",))                       # live: not counted
        guessed = _pre_ledger_closed(tmp_db, 5)
        tmp_db.set_application_outcome(screen, "Screening Call", "Test answer: reason X")
        conn = tmp_db._get_thread_conn()
        conn.execute("UPDATE application_events SET effective_at = datetime('now', '-1 day') "
                     "WHERE app_id = ? AND event_type = 'outcome'", (screen,))
        conn.commit()
        tmp_db.set_application_outcome(ghosted, "Applied", "Test answer: outcome C")
        s = tmp_db.get_outcome_summary()
        assert (s["closed"], s["answered"], s["to_confirm"]) == (4, 2, 1)
        assert s["ended_after"] == {"Interview": 1, "Screening Call": 1}
        assert [n["app_id"] for n in s["notes"]] == [ghosted, screen]      # newest first
        assert s["notes"][1] | {"at": None} == {
            "app_id": screen, "company": "Studio 2", "title": "Producer 2", "status": "Rejected",
            "stage": "Screening Call", "what_happened": "Test answer: reason X", "at": None}
        assert interview not in {n["app_id"] for n in s["notes"]}     # no answer, no line

    def test_candidate_withdrew(self, tmp_db):
        """I Withdrew closes a card like Rejected: the same question and rules,
        and the Stats page counts the ones the candidate ended."""
        app_id = _closed(tmp_db, 1, path=("Applied", "Screening Call", "I Withdrew"))
        assert tmp_db.set_application_outcome(
            app_id, "Screening Call", "Test answer: reason Y") == "saved"
        card = tmp_db.get_card_outcome(app_id)
        assert [h["status"] for h in card["history"]] == ["Applied", "Screening Call", "I Withdrew"]
        assert card["outcome"]["what_happened"] == "Test answer: reason Y"
        _closed(tmp_db, 2)                                          # a rejection
        s = tmp_db.get_outcome_summary()
        assert (s["closed"], s["withdrew"], s["answered"]) == (2, 1, 1)
        assert s["ended_after"] == {"Screening Call": 1}
        assert s["notes"][0]["status"] == "I Withdrew"

    def test_switching_how_it_closed_is_still_one_close(self, tmp_db):
        """Marked Rejected, then the candidate says they withdrew: one close under their
        status and its first date, and the candidate's answer stays with the card."""
        app_id = _closed(tmp_db, 1, path=("Applied", "Interview", "Rejected"))
        tmp_db.set_application_outcome(app_id, "Interview", "Test answer: outcome D")
        closed_at = tmp_db.get_card_outcome(app_id)["history"][-1]["at"]
        tmp_db.update_application(app_id, "I Withdrew", None)
        card = tmp_db.get_card_outcome(app_id)
        assert [h["status"] for h in card["history"]] == ["Applied", "Interview", "I Withdrew"]
        assert card["history"][-1]["at"] == closed_at
        assert (card["outcome"]["stage"], card["outcome"]["what_happened"]) == (
            "Interview", "Test answer: outcome D")
        # reopened and closed again: two closes
        _age_events(tmp_db, app_id, 60)
        tmp_db.update_application(app_id, "Interview", None)
        _age_events(tmp_db, app_id, 60)
        tmp_db.update_application(app_id, "Rejected", None)
        assert [h["status"] for h in tmp_db.get_card_outcome(app_id)["history"]] == [
            "Applied", "Interview", "I Withdrew", "Interview", "Rejected"]

    def test_api_answer_on_a_card_she_withdrew_from(self, env):
        _, app_id = _tracked(env.db)
        for status in ("Applied", "I Withdrew"):
            env.client.put(f"/api/applications/{app_id}", json={"status": status})
        r = env.client.put(f"/api/applications/{app_id}/outcome",
                           json={"stage": "Applied", "what_happened": "Test answer: reason Z"})
        assert r.status_code == 200 and r.json()["outcome"]["stage"] == "Applied"
        assert env.client.get("/api/stats/overview").json()["endings"]["withdrew"] == 1

    def test_api_path(self, env):
        _, app_id = _tracked(env.db)
        for status in ("Applied", "Rejected"):
            env.client.put(f"/api/applications/{app_id}", json={"status": status})
        r = env.client.put(f"/api/applications/{app_id}/outcome",
                           json={"stage": "Interview", "what_happened": "Test answer: outcome E"})
        assert r.status_code == 200
        body = r.json()
        assert body["app_id"] == app_id and body["highest_stage"] == "Interview"
        assert body["outcome"]["what_happened"] == "Test answer: outcome E"
        assert [h["status"] for h in body["history"]] == ["Applied", "Interview", "Rejected"]
        row = env.client.get("/api/applications").json()[0]
        assert row["outcome"] == body["outcome"]
        stats = env.client.get("/api/stats/overview").json()
        assert stats["endings"]["ended_after"] == {"Interview": 1}
        assert stats["endings"]["notes"][0]["what_happened"] == "Test answer: outcome E"
        assert stats["reached_interview_total"] == 1

        bad = [({"stage": "Rejected"}, 400),
               ({"stage": "Applied", "what_happened": "x" * 501}, 400)]
        for payload, code in bad:
            assert env.client.put(f"/api/applications/{app_id}/outcome", json=payload).status_code == code
        assert env.client.put("/api/applications/424242/outcome",
                              json={"stage": "Applied"}).status_code == 404
        env.client.put(f"/api/applications/{app_id}", json={"status": "Interview"})
        assert env.client.put(f"/api/applications/{app_id}/outcome",
                              json={"stage": "Applied"}).status_code == 409
