"""Tests for the assistant tool layer (chat_tools.py)."""
import pytest

from tests.fixtures.persona import FIRST_NAME

from job_agent.db_manager import DBManager
from job_agent import chat_tools as ct


@pytest.fixture
def db(tmp_path):
    return DBManager(db_path=str(tmp_path / "chat_test.db"))


@pytest.fixture
def seeded(db):
    """Three jobs across tiers + one tracked application."""
    j1 = db.save_job_with_relevance(
        {"title": "Senior Video Producer", "company": "Harbor Lights Media", "location": "Bellevue, WA",
         "job_url": "https://n.com/1", "description": "Lead video production end to end."},
        relevance="relevant", match_level="HIGH_MATCH", confidence=0.92,
        session_id="s1", tier="strong",
    )
    j2 = db.save_job_with_relevance(
        {"title": "Content Strategist", "company": "Driftwood Audio", "location": "Remote",
         "job_url": "https://s.com/2", "description": "Content with production duties."},
        relevance="relevant", match_level="WORTH_A_LOOK", confidence=0.5,
        session_id="s1", tier="worth_a_look",
    )
    j3 = db.save_job_with_relevance(
        {"title": "Backend Engineer", "company": "Ledgerline Payments", "location": "Seattle, WA",
         "job_url": "https://st.com/3", "description": "Go and Postgres."},
        relevance="irrelevant", match_level="IRRELEVANT", confidence=0.7,
        session_id="s1",
    )
    db.track_job(j1)
    return {"strong": j1, "worth": j2, "reject": j3}


class TestQueryAndGet:
    def test_query_by_tier(self, db, seeded):
        out = ct.query_jobs(db, tier="strong")
        assert out["count"] == 1
        assert out["jobs"][0]["company"] == "Harbor Lights Media"

    def test_query_by_company_substring(self, db, seeded):
        out = ct.query_jobs(db, company="drift")
        assert out["count"] == 1
        assert out["jobs"][0]["tier"] == "worth_a_look"

    def test_query_sorts_strong_first(self, db, seeded):
        out = ct.query_jobs(db)
        tiers = [j["tier"] for j in out["jobs"]]
        assert tiers[0] == "strong"

    def test_get_job_detail(self, db, seeded):
        out = ct.get_job(db, seeded["strong"])
        assert out["title"] == "Senior Video Producer"
        assert "Lead video production" in out["description"]

    def test_get_job_missing(self, db, seeded):
        assert "error" in ct.get_job(db, 99999)


class TestApplicationsFlow:
    def test_check_already_applied_finds_tracked(self, db, seeded):
        out = ct.check_already_applied(db, "Harbor Lights Media")
        assert out["already_engaged"] is True
        assert out["applications"][0]["title"] == "Senior Video Producer"

    def test_check_already_applied_clean_company(self, db, seeded):
        out = ct.check_already_applied(db, "Kestrel Studios")
        assert out["already_engaged"] is False

    def test_update_application_status(self, db, seeded):
        apps = ct.get_applications(db)["applications"]
        app_id = apps[0]["app_id"]
        out = ct.update_application_status(db, app_id, "Applied", notes="sent via portal")
        assert out["updated"] is True
        refreshed = ct.get_applications(db, status="Applied")
        assert refreshed["count"] == 1

    def test_update_rejects_invalid_status(self, db, seeded):
        assert "error" in ct.update_application_status(db, 1, "Bogus Column")

    def test_track_and_dismiss(self, db, seeded):
        out = ct.track_job(db, seeded["worth"])
        assert out["tracked"] is True
        out2 = ct.dismiss_job(db, seeded["reject"], reason="wrong_industry")
        assert out2["dismissed"] is True
        detail = ct.get_job(db, seeded["reject"])
        assert detail["rejection_reason"] == "wrong_industry"


class TestSafetyAndStats:
    def test_read_document_path_restricted(self, db):
        assert "error" in ct.read_document(db, "/etc/passwd")
        assert "error" in ct.read_document(db, "../secrets.docx")
        assert "error" in ct.read_document(db, "results/x.docx")
        # under cv/ but missing -> file-not-found error, not a crash
        assert "error" in ct.read_document(db, "cv/nope.docx")

    def test_pipeline_stats_shape(self, db, seeded):
        out = ct.get_pipeline_stats(db)
        assert "unreviewed_backlog" in out
        assert "kanban" in out
        assert out["kanban"].get("To Apply") == 1

    def test_execute_tool_dispatch(self, db, seeded):
        out = ct.execute_tool(db, "query_jobs", {"tier": "strong"})
        assert out["count"] == 1
        assert "error" in ct.execute_tool(db, "no_such_tool", {})
        assert "error" in ct.execute_tool(db, "get_job", {"bogus_arg": 1})

    def test_anthropic_specs_complete(self):
        specs = ct.anthropic_tool_specs()
        assert len(specs) == len(ct.TOOLS)
        for s in specs:
            assert set(s) == {"name", "description", "input_schema"}


class TestPowerTools:
    def test_list_stale_applications(self, db, seeded):
        # Make the tracked app stale: Applied 30 days ago
        import datetime as dt
        conn = db._get_thread_conn()
        old_date = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%d")
        conn.execute("UPDATE applications SET status='Applied', date_applied=?", (old_date,))
        conn.commit()
        out = ct.list_stale_applications(db, days=14)
        assert out["count"] == 1
        assert out["stale_applications"][0]["company"] == "Harbor Lights Media"
        assert out["stale_applications"][0]["days_since_applied"] >= 29

    def test_list_stale_excludes_fresh(self, db, seeded):
        import datetime as dt
        conn = db._get_thread_conn()
        conn.execute("UPDATE applications SET status='Applied', date_applied=?",
                     (dt.datetime.now().strftime("%Y-%m-%d"),))
        conn.commit()
        assert ct.list_stale_applications(db, days=14)["count"] == 0

    def test_draft_followup_uses_llm(self, db, seeded, monkeypatch):
        import datetime as dt
        conn = db._get_thread_conn()
        conn.execute("UPDATE applications SET status='Applied', date_applied=?",
                     ((dt.datetime.now() - dt.timedelta(days=20)).strftime("%Y-%m-%d"),))
        conn.commit()
        app_id = ct.get_applications(db)["applications"][0]["app_id"]
        captured = {}

        def fake_llm(prompt, model="", **k):
            captured["prompt"] = prompt
            return "EMAIL: subject...\nLINKEDIN: hi..."
        import job_agent.gpt_filter as gf
        monkeypatch.setattr(gf, "call_llm", fake_llm)
        out = ct.draft_followup(db, app_id)
        assert "EMAIL" in out["drafts"]
        assert "Harbor Lights Media" in captured["prompt"]   # grounded in the tracked app
        assert "VERIFIED METRICS" in captured["prompt"]              # metrics bank present

    def test_draft_outreach_includes_jd(self, db, seeded, monkeypatch):
        captured = {}

        def fake_llm(prompt, model="", **k):
            captured["prompt"] = prompt
            return "CONNECTION NOTE: ...\nINMAIL: ..."
        import job_agent.gpt_filter as gf
        monkeypatch.setattr(gf, "call_llm", fake_llm)
        out = ct.draft_outreach(db, seeded["strong"])
        assert "INMAIL" in out["drafts"]
        assert "Lead video production" in captured["prompt"]  # actual JD text

    def test_apply_kit_bundle(self, db, seeded):
        out = ct.get_apply_kit(db, seeded["strong"])
        assert out["company"] == "Harbor Lights Media"
        assert out["apply_link"] == "https://n.com/1"
        assert out["standard_answers"]["first_name"] == FIRST_NAME
        assert out["already_engaged"] is True   # it's tracked
        assert out["cv_ready"] is False


class TestChatSessions:
    def test_list_sessions_with_preview(self, db):
        db.save_chat_message("user", "Find me podcast jobs", session_id="thread-a")
        db.save_chat_message("assistant", "Here are 3...", session_id="thread-a")
        db.save_chat_message("user", "Apply kit for Kestrel Studios", session_id="thread-b")
        sessions = db.list_chat_sessions()
        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "thread-b"      # most recent first
        assert sessions[0]["preview"].startswith("Apply kit")
        assert sessions[1]["turns"] == 2

    def test_histories_are_isolated(self, db):
        db.save_chat_message("user", "A", session_id="t1")
        db.save_chat_message("user", "B", session_id="t2")
        assert len(db.get_chat_messages(session_id="t1")) == 1
        db.clear_chat_messages(session_id="t1")
        assert db.get_chat_messages(session_id="t1") == []
        assert len(db.get_chat_messages(session_id="t2")) == 1
