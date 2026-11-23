import inspect
import pytest


class TestPostprocess:
    def test_single_send_email_function(self):
        """Verify only one send_email_with_results exists (no duplicate)."""
        from job_agent import postprocess

        # Count definitions of send_email_with_results
        source = inspect.getsource(postprocess)
        count = source.count("def send_email_with_results(")
        assert count == 1, f"Expected 1 send_email_with_results definition, found {count}"

    def test_send_email_accepts_profile_text(self):
        """The kept version should accept profile_text parameter."""
        from job_agent.postprocess import send_email_with_results
        sig = inspect.signature(send_email_with_results)
        assert "profile_text" in sig.parameters

    def test_send_email_accepts_db_parameter(self):
        """send_email_with_results must accept an optional db parameter."""
        from job_agent.postprocess import send_email_with_results
        sig = inspect.signature(send_email_with_results)
        assert "db" in sig.parameters
        # Default should be None (optional)
        assert sig.parameters["db"].default is None


class TestMdToHtml:
    """Tests for _md_to_html markdown-to-HTML conversion."""

    def test_bold_double_asterisks(self):
        from job_agent.postprocess import _md_to_html
        assert _md_to_html("**hello**") == "<strong>hello</strong>"

    def test_bold_double_underscores(self):
        from job_agent.postprocess import _md_to_html
        assert _md_to_html("__hello__") == "<strong>hello</strong>"

    def test_italic_single_asterisk(self):
        from job_agent.postprocess import _md_to_html
        assert _md_to_html("*hello*") == "<em>hello</em>"

    def test_strip_header_markers(self):
        from job_agent.postprocess import _md_to_html
        assert _md_to_html("## Section Title") == "Section Title"
        assert _md_to_html("# Top") == "Top"

    def test_empty_and_none(self):
        from job_agent.postprocess import _md_to_html
        assert _md_to_html("") == ""
        assert _md_to_html(None) == ""

    def test_mixed_formatting(self):
        from job_agent.postprocess import _md_to_html
        result = _md_to_html("**bold** and *italic*")
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result


class TestTruncateSentences:
    """Tests for _truncate_sentences helper."""

    def test_one_sentence(self):
        from job_agent.postprocess import _truncate_sentences
        text = "This is one sentence. This is two. This is three."
        assert _truncate_sentences(text, 1) == "This is one sentence."

    def test_two_sentences(self):
        from job_agent.postprocess import _truncate_sentences
        text = "First. Second. Third."
        assert _truncate_sentences(text, 2) == "First. Second."

    def test_empty_string(self):
        from job_agent.postprocess import _truncate_sentences
        assert _truncate_sentences("", 2) == ""
        assert _truncate_sentences(None, 2) == ""

    def test_fewer_sentences_than_max(self):
        from job_agent.postprocess import _truncate_sentences
        assert _truncate_sentences("Only one.", 5) == "Only one."


class TestBuildIntelPreview:
    """Tests for _build_intel_preview — the per-job intel section in the email."""

    def test_returns_empty_without_db(self):
        from job_agent.postprocess import _build_intel_preview
        assert _build_intel_preview("Acme", "Producer", None) == ""

    def test_returns_empty_when_no_intel_cached(self):
        from job_agent.postprocess import _build_intel_preview

        class MockDB:
            def get_company_intel(self, key):
                return None

        assert _build_intel_preview("Acme", "Producer", MockDB()) == ""

    def test_includes_company_profile(self):
        from job_agent.postprocess import _build_intel_preview

        class MockDB:
            def get_company_intel(self, key):
                if key == "Acme":
                    return {"company_profile": "Acme is a tech company. They make widgets."}
                return None

        html = _build_intel_preview("Acme", "Producer", MockDB())
        assert "<strong>Company:</strong>" in html
        assert "Acme is a tech company" in html

    def test_includes_salary_intel(self):
        from job_agent.postprocess import _build_intel_preview

        class MockDB:
            def get_company_intel(self, key):
                if key == "__salary_intel__Acme__Producer":
                    return {"company_profile": "Base salary range is $90K-$120K."}
                return None

        html = _build_intel_preview("Acme", "Producer", MockDB())
        assert "<strong>Salary:</strong>" in html
        assert "$90K-$120K" in html

    def test_includes_salary_recommendation(self):
        from job_agent.postprocess import _build_intel_preview

        class MockDB:
            def get_company_intel(self, key):
                if key == "__salary_rec__Acme__Producer":
                    return {"company_profile": "Target $110K base given your experience."}
                return None

        html = _build_intel_preview("Acme", "Producer", MockDB())
        assert "<strong>Strategy:</strong>" in html
        assert "Target $110K" in html

    def test_includes_hiring_intel(self):
        from job_agent.postprocess import _build_intel_preview

        class MockDB:
            def get_company_intel(self, key):
                if key == "__hiring_intel__Acme":
                    return {"company_profile": "VP of Content Jane Smith is the hiring manager. She posted about the role last week."}
                return None

        html = _build_intel_preview("Acme", "Producer", MockDB())
        assert "<strong>Intel:</strong>" in html
        assert "Jane Smith" in html

    def test_all_sections_combined(self):
        from job_agent.postprocess import _build_intel_preview

        class MockDB:
            def get_company_intel(self, key):
                data = {
                    "Acme": {"company_profile": "Acme is a media company."},
                    "__salary_intel__Acme__Producer": {"company_profile": "$100K-$130K base."},
                    "__salary_rec__Acme__Producer": {"company_profile": "Ask for $120K."},
                    "__hiring_intel__Acme": {"company_profile": "John Doe is hiring."},
                }
                return data.get(key)

        html = _build_intel_preview("Acme", "Producer", MockDB())
        assert "<strong>Company:</strong>" in html
        assert "<strong>Salary:</strong>" in html
        assert "<strong>Strategy:</strong>" in html
        assert "<strong>Intel:</strong>" in html
        # Styled container
        assert "border-left:3px solid #6366f1" in html

    def test_intel_preview_converts_markdown(self):
        """Markdown in cached intel should be converted to HTML in the email."""
        from job_agent.postprocess import _build_intel_preview

        class MockDB:
            def get_company_intel(self, key):
                if key == "Acme":
                    return {"company_profile": "**Acme** is a *great* company."}
                return None

        html = _build_intel_preview("Acme", "Producer", MockDB())
        assert "<strong>Acme</strong>" in html
        assert "<em>great</em>" in html


class TestEmailIntelIntegration:
    """Integration tests for intel preview appearing in the email body."""

    def _make_df(self):
        import pandas as pd
        return pd.DataFrame([{
            "title": "Content Producer",
            "company": "Harbor Lights Media",
            "location": "Tacoma, WA",
            "job_url": "https://example.com/job/1",
            "confidence": 0.9,
            "match_level": "HIGH_MATCH",
            "gpt_relevance": "relevant",
            "gpt_reasoning": "Strong match for production skills",
        }])

    def test_email_excludes_intel_preview(self, monkeypatch):
        """Intel previews were REMOVED from the digest (they ballooned every row
        and made the email unreadable on phones) — intel lives in the app and
        the assistant. The email must NOT include cached intel text even when
        the db has it."""
        from job_agent.postprocess import send_email_with_results

        class MockDB:
            def get_company_intel(self, key):
                if key == "Harbor Lights Media":
                    return {"company_profile": "Harbor Lights Media is a regional streaming studio."}
                if key == "__salary_intel__Harbor Lights Media__Content Producer":
                    return {"company_profile": "Content Producers earn $120K-$160K at Harbor Lights Media."}
                return None

        captured = {}
        def fake_smtp(*a, **kw):
            class FakeSMTP:
                def login(self, *a): pass
                def sendmail(self, sender, recips, msg_str):
                    captured["msg"] = msg_str
                def quit(self): pass
            return FakeSMTP()

        monkeypatch.setenv("EMAIL_SENDER", "test@test.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret")
        monkeypatch.setattr("job_agent.postprocess.smtplib.SMTP_SSL", fake_smtp)

        send_email_with_results(
            None, "candidate@test.com", self._make_df(),
            db=MockDB(),
        )

        msg = captured["msg"]
        assert "Harbor Lights Media is a regional streaming studio" not in msg
        assert "$120K-$160K" not in msg

    def test_email_works_without_db(self, monkeypatch):
        """Email should still send fine when db=None (no intel)."""
        from job_agent.postprocess import send_email_with_results

        captured = {}
        def fake_smtp(*a, **kw):
            class FakeSMTP:
                def login(self, *a): pass
                def sendmail(self, sender, recips, msg_str):
                    captured["msg"] = msg_str
                def quit(self): pass
            return FakeSMTP()

        monkeypatch.setenv("EMAIL_SENDER", "test@test.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret")
        monkeypatch.setattr("job_agent.postprocess.smtplib.SMTP_SSL", fake_smtp)

        send_email_with_results(
            None, "candidate@test.com", self._make_df(),
            db=None,
        )

        msg = captured["msg"]
        assert "Content Producer" in msg
        assert "Harbor Lights Media" in msg

    def test_no_intel_for_non_match_jobs(self, monkeypatch):
        """Jobs that are not HIGH_MATCH or POTENTIAL_MATCH should not get intel preview."""
        import pandas as pd
        from job_agent.postprocess import send_email_with_results

        df = pd.DataFrame([{
            "title": "Random Role",
            "company": "Acme",
            "location": "Remote",
            "job_url": "https://example.com/job/2",
            "confidence": 0.3,
            "match_level": "NO_MATCH",
            "gpt_relevance": "relevant",
            "gpt_reasoning": "Low match",
        }])

        intel_calls = []
        class MockDB:
            def get_company_intel(self, key):
                intel_calls.append(key)
                return {"company_profile": "Should not appear."}

        captured = {}
        def fake_smtp(*a, **kw):
            class FakeSMTP:
                def login(self, *a): pass
                def sendmail(self, sender, recips, msg_str):
                    captured["msg"] = msg_str
                def quit(self): pass
            return FakeSMTP()

        monkeypatch.setenv("EMAIL_SENDER", "test@test.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret")
        monkeypatch.setattr("job_agent.postprocess.smtplib.SMTP_SSL", fake_smtp)

        send_email_with_results(None, "candidate@test.com", df, db=MockDB())

        # _build_intel_preview should not have been called for NO_MATCH jobs
        assert len(intel_calls) == 0

    def test_intel_lookup_error_does_not_break_email(self, monkeypatch):
        """If intel DB lookup raises, the email should still send."""
        from job_agent.postprocess import send_email_with_results

        class BrokenDB:
            def get_company_intel(self, key):
                raise RuntimeError("DB exploded")

        captured = {}
        def fake_smtp(*a, **kw):
            class FakeSMTP:
                def login(self, *a): pass
                def sendmail(self, sender, recips, msg_str):
                    captured["msg"] = msg_str
                def quit(self): pass
            return FakeSMTP()

        monkeypatch.setenv("EMAIL_SENDER", "test@test.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret")
        monkeypatch.setattr("job_agent.postprocess.smtplib.SMTP_SSL", fake_smtp)

        # Should not raise
        send_email_with_results(
            None, "candidate@test.com", self._make_df(),
            db=BrokenDB(),
        )

        msg = captured["msg"]
        assert "Content Producer" in msg  # Email still sent


class TestSmtpRetry:
    """_smtp_send: connect timeout + retry/backoff, so one transient connect
    timeout (Errno 60) does not lose the digest (sends used to be one-shot
    with no timeout)."""

    def test_transient_failure_then_success(self, monkeypatch):
        from job_agent.postprocess import _smtp_send
        calls = {"n": 0}

        def fake_smtp(*a, **kw):
            calls["n"] += 1
            assert kw.get("timeout") == 30  # connect timeout must be set
            if calls["n"] == 1:
                raise OSError(60, "Operation timed out")
            class FakeSMTP:
                def login(self, *a): pass
                def sendmail(self, *a): calls["sent"] = True
                def quit(self): pass
            return FakeSMTP()

        monkeypatch.setattr("job_agent.postprocess.smtplib.SMTP_SSL", fake_smtp)
        monkeypatch.setattr("job_agent.postprocess.time.sleep", lambda s: None)

        class Msg:
            def as_string(self): return "m"

        _smtp_send("s@t.com", "pw", ["r@t.com"], Msg())
        assert calls["n"] == 2 and calls.get("sent")

    def test_persistent_failure_raises(self, monkeypatch):
        import pytest
        from job_agent.postprocess import _smtp_send

        def fake_smtp(*a, **kw):
            raise OSError(60, "Operation timed out")

        monkeypatch.setattr("job_agent.postprocess.smtplib.SMTP_SSL", fake_smtp)
        monkeypatch.setattr("job_agent.postprocess.time.sleep", lambda s: None)

        class Msg:
            def as_string(self): return "m"

        with pytest.raises(OSError):
            _smtp_send("s@t.com", "pw", ["r@t.com"], Msg())
