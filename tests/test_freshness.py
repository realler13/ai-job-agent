"""Freshness: recent postings are the ones worth applying to. Strong roles show
up to 7 days, Worth a look up to 3 — by posting date, or the day we first saw
the job when it has none. Stale roles are hidden, never deleted."""
from datetime import date, timedelta

import pandas as pd

from job_agent import postprocess
from job_agent.config import REVIEW_MAX_AGE_DAYS, STALE_POSTING_DAYS
from job_agent.db_manager import DBManager
from job_agent.utils import is_fresh_for_review, role_age_days

TODAY = date(2026, 10, 4)


def _ago(n):
    return (date.today() - timedelta(days=n)).isoformat()


def test_limits():
    assert REVIEW_MAX_AGE_DAYS == {"strong": 7, "worth_a_look": 3}
    assert STALE_POSTING_DAYS == 7


def test_role_age_uses_posting_date_then_first_seen():
    assert role_age_days("2026-10-01", "2026-09-01 10:00:00", TODAY) == 3
    assert role_age_days(None, "2026-09-30 22:00:00", TODAY) == 4          # no posting date
    assert role_age_days("nan", "2026-10-04 01:00:00", TODAY) == 0
    assert role_age_days(None, None, TODAY) is None
    assert role_age_days("2026-10-09", None, TODAY) == 0                   # future -> 0


def test_fresh_by_tier():
    assert is_fresh_for_review("strong", "2026-09-27", None, TODAY)         # 7 days
    assert not is_fresh_for_review("strong", "2026-09-26", None, TODAY)     # 8 days
    assert is_fresh_for_review("worth_a_look", "2026-10-01", None, TODAY)   # 3 days
    assert not is_fresh_for_review("worth_a_look", "2026-09-30", None, TODAY)
    assert is_fresh_for_review("worth_a_look", None, None, TODAY)           # unknown age passes
    assert is_fresh_for_review(None, "2020-01-01", None, TODAY)             # no tier, no limit


def test_review_list_hides_stale_roles(tmp_path):
    db = DBManager(db_path=str(tmp_path / "f.db"))
    try:
        def save(n, tier, posted):
            return db.save_job_with_relevance(
                {"title": f"Producer {n}", "company": f"C{n}", "location": "Seattle, WA",
                 "job_url": f"https://ex/{n}", "description": "d", "date_posted": posted},
                relevance="relevant", match_level="HIGH_MATCH", confidence=0.9, reasoning="fit", tier=tier)
        keep = {save(1, "strong", _ago(6)), save(2, "worth_a_look", _ago(2)), save(3, "worth_a_look", None)}
        save(4, "strong", _ago(9))
        save(5, "worth_a_look", _ago(5))
        assert {r["id"] for r in db.get_unreviewed_jobs()} == keep
    finally:
        db.close()


def test_email_drops_stale_rows_and_marks_apply_today():
    df = pd.DataFrame([
        {"title": "Fresh", "tier": "strong", "date_posted": _ago(0)},
        {"title": "Old strong", "tier": "strong", "date_posted": _ago(9)},
        {"title": "Old worth", "tier": "worth_a_look", "date_posted": _ago(4)},
    ])
    assert list(postprocess._drop_stale_rows(df)["title"]) == ["Fresh"]


def test_email_marks_same_day_roles_apply_today(monkeypatch):
    import email as _email
    captured = {}

    class FakeSMTP:
        def login(self, *a):
            pass

        def sendmail(self, sender, to, msg):
            captured["msg"] = msg

        def quit(self):
            pass
    monkeypatch.setenv("EMAIL_SENDER", "digest@example.invalid")
    monkeypatch.setenv("EMAIL_PASSWORD", "secret")
    monkeypatch.setattr("job_agent.postprocess.smtplib.SMTP_SSL", lambda *a, **k: FakeSMTP())
    monkeypatch.setattr(postprocess, "_is_morning_digest", lambda now: False)
    df = pd.DataFrame([{"title": "Video Producer", "company": "Acme", "location": "Seattle, WA",
                        "job_url": "https://ex/1", "tier": "strong", "match_level": "HIGH_MATCH",
                        "confidence": 0.9, "gpt_reasoning": "fit", "date_posted": _ago(0)}])
    postprocess.send_email_with_results(None, "recipient@example.invalid", df, db=None)
    body = next(p for p in _email.message_from_string(captured["msg"]).walk()
                if p.get_content_type() == "text/html").get_payload(decode=True).decode("ascii")
    assert "new today &middot; apply today" in body or "new today · apply today" in body \
        or "new today &#183; apply today" in body
