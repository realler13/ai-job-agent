"""Contact resolution (Spec M2): Hunter waterfall, verified-only gating."""

import pytest

from job_agent import contact_resolution as cr


@pytest.fixture
def db(tmp_path):
    from job_agent.db_manager import DBManager
    return DBManager(db_path=str(tmp_path / "cr.db"))


class TestResolveEmail:
    def test_disabled_without_key(self, monkeypatch):
        monkeypatch.setattr(cr, "HUNTER_API_KEY", "")
        assert cr.hunter_enabled() is False
        assert cr.resolve_email("Jane Doe", "Acme") is None

    def _mock_hunter(self, monkeypatch, payload, status=200):
        class R:
            status_code = status
            def raise_for_status(self): pass
            def json(self): return payload
        monkeypatch.setattr(cr, "HUNTER_API_KEY", "k")
        monkeypatch.setattr(cr.requests, "get", lambda *a, **k: R())

    def test_valid_verification_is_verified(self, monkeypatch):
        self._mock_hunter(monkeypatch, {"data": {
            "email": "jane@acme.com", "score": 95,
            "verification": {"status": "valid"},
            "sources": [{"uri": "https://acme.com/team"}]}})
        r = cr.resolve_email("Jane Doe", "Acme")
        assert r["email_status"] == "verified"
        assert r["sources"] == ["https://acme.com/team"]

    def test_accept_all_is_pattern_guess_not_verified(self, monkeypatch):
        # accept_all/unknown must NOT unlock email drafts (spec: bounces hurt).
        self._mock_hunter(monkeypatch, {"data": {
            "email": "jane@acme.com", "score": 70,
            "verification": {"status": "accept_all"}, "sources": []}})
        assert cr.resolve_email("Jane Doe", "Acme")["email_status"] == "pattern_guess"

    def test_no_email_returns_none(self, monkeypatch):
        self._mock_hunter(monkeypatch, {"data": {"email": None}})
        assert cr.resolve_email("Jane Doe", "Acme") is None

    def test_rate_limit_returns_none(self, monkeypatch):
        self._mock_hunter(monkeypatch, {}, status=429)
        assert cr.resolve_email("Jane Doe", "Acme") is None


class TestExtractNamedContacts:
    def test_extracts_full_names_only(self, monkeypatch):
        monkeypatch.setattr(
            "job_agent.gpt_filter.call_llm",
            lambda p, model="", **kw: '[{"name":"Jane Doe","title":"Head of Video","role_type":"hiring_manager"},'
                                      '{"name":"Joseph","title":"?","role_type":"peer"}]')
        out = cr.extract_named_contacts([{"content": "Jane Doe leads video."}])
        assert [c["name"] for c in out] == ["Jane Doe"]  # single-word name dropped

    def test_empty_blocks(self):
        assert cr.extract_named_contacts([]) == []

    def test_garbage_llm_output(self, monkeypatch):
        monkeypatch.setattr("job_agent.gpt_filter.call_llm",
                            lambda p, model="", **kw: "no json here")
        assert cr.extract_named_contacts([{"content": "x"}]) == []


class TestResolveContactsForJob:
    def test_stores_contacts_and_caps_lookups(self, db, monkeypatch):
        monkeypatch.setattr(cr, "extract_named_contacts", lambda blocks, company="": [
            {"name": "A One", "title": "T1", "role_type": "recruiter"},
            {"name": "B Two", "title": "T2", "role_type": "hiring_manager"},
            {"name": "C Three", "title": "T3", "role_type": "peer"},
        ])
        calls = []
        monkeypatch.setattr(cr, "HUNTER_API_KEY", "k")
        monkeypatch.setattr(cr, "company_domain", lambda db, c: "acme.com")
        monkeypatch.setattr(cr, "resolve_email", lambda n, c, domain=None: calls.append(n) or {
            "email": f"{n.split()[0].lower()}@x.com", "email_status": "verified",
            "score": 90, "position": "", "sources": []})
        out = cr.resolve_contacts_for_job(db, 1, "Acme", [{"content": "x"}])
        assert len(out) == 3
        assert len(calls) == cr.MAX_LOOKUPS_PER_RUN  # hard cap: 2 credits max
        stored = db.get_contacts_for_job(1)
        assert len(stored) == 3
        assert stored[0]["email_status"] == "verified"  # verified sort first

    def test_verified_only_in_sendable_text(self):
        text = cr.resolved_contacts_text([
            {"name": "A One", "title": "T", "email": "a@x.com", "email_status": "verified",
             "source": "hunter", "role_type": "recruiter"},
            {"name": "B Two", "title": "T", "email": "b@x.com", "email_status": "pattern_guess",
             "source": None, "role_type": "peer"},
            {"name": "C Three", "title": "T", "email": None, "email_status": "unknown",
             "source": None, "role_type": "peer"},
        ])
        assert "VERIFIED email: a@x.com" in text
        assert "b@x.com" not in text          # unverified address never shown
        assert "DO-NOT-EMAIL" in text
