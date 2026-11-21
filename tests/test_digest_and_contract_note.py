"""Phase 27 digest changes: follow-ups below the Strong matches, morning digest
only, at most 5, no day-21 line; the contract-pay note on contract / hourly
cards; an ASCII-only HTML body. Hermetic: SMTP and the DB are fakes."""
import email

import pandas as pd
import pytest

from job_agent import postprocess
from job_agent.comp_utils import (CONTRACT_PAY_NOTE, contract_pay_note,
                                          is_contract_pay, is_staffing_agency)

TAIL = ("taxes, benefits and paid time off work differently from a salaried role; "
        "compare take-home, not the annualized figure.")
BASE = "Contract role — " + TAIL
VIA = " (W2 through an agency, or 1099)"
# The digest card in _jobs(): Contract · W2 at $45/hr.
NOTE = "Contract role, paid hourly" + VIA + " — " + TAIL
NOTE_HTML = NOTE.replace("—", "&#8212;")


# ── contract-pay rule ────────────────────────────────────────────────────────
@pytest.mark.parametrize("etype,comp,expected", [
    ("Contract", "$90K/yr", True),
    ("Contract · W2", "$45/hr (≈$94K/yr)", True),
    ("Contract · 1099", None, True),
    ("Temp", None, True),
    ("Full-time/Contract", "$100K/yr", True),
    ("1099", None, True),
    ("Full-time · C2C", None, True),
    ("W2", "$40/hr (≈$83K/yr)", True),            # agency W2 billed hourly
    (None, "$60/hr (≈$125K/yr)", True),           # hourly, nothing says salaried
    ("Full-time", "$60/hr (≈$125K/yr)", False),   # an hourly EMPLOYEE role
    ("Part-time", "$30/hr (≈$62K/yr)", False),
    ("Internship", "$25/hr (≈$52K/yr)", False),
    ("Full-time", "$120K–$150K/yr", False),
    ("Full-time · W2", None, False),
    (None, None, False),
    ("nan", "nan", False),
])
def test_is_contract_pay(etype, comp, expected):
    assert is_contract_pay(etype, comp) is expected
    assert (contract_pay_note(etype, comp) is not None) is expected


# The note is built from what the row supports (Phase 27 review: a fixed
# "paid hourly (W2 through an agency, or 1099)" also sat on cards whose pay
# was yearly, a day rate, or unknown). Cases cover the row shapes the scrapers produce;
# employers are fictional except the staffing agencies (a market list).
@pytest.mark.parametrize("etype,comp,company,note", [
    # no pay stated, plain contract: the base text, no basis, no agency claim
    ("Contract", None, "Quiet Harbor Co", BASE),
    ("Contract", "", None, BASE),
    # yearly pay (a big-company events contract, a direct public-radio temp):
    # never "paid hourly"
    ("Contract", "$140K–$200K/yr", "Harbor Lights Media", BASE),
    ("Temp", "$110K–$140K/yr", "Puget Public Radio", "Temporary role — " + TAIL),
    ("Full-time/Contract", "$57K–$69K/yr", "Saltmarsh Finance LLC.", BASE),
    # day rate
    ("Contract", "$400–$900/day (≈$104K–$234K/yr)", "Bramble Pictures",
     "Contract role, paid a day rate — " + TAIL),
    # hourly contract, no tax form, direct employer
    ("Contract", "$43–$48/hr (≈$89K–$100K/yr)", "Northwind Software",
     "Contract role, paid hourly — " + TAIL),
    # hourly + W2 named
    ("Contract · W2", "$65–$75/hr (≈$135K–$156K/yr)", "Cedar & Pine Studio",
     "Contract role, paid hourly" + VIA + " — " + TAIL),
    ("Contract · 1099", None, None, "Contract role" + VIA + " — " + TAIL),
    ("Full-time · C2C", None, None, "Contract role" + VIA + " — " + TAIL),
    # staffing agency with only W2 named and hourly pay
    ("W2", "$60/hr (≈$125K/yr)", "Aquent (staffing)",
     "Contract role, paid hourly" + VIA + " — " + TAIL),
    # staffing agency, contract, no pay, no tax form named
    ("Contract", None, "Robert Half", "Contract role" + VIA + " — " + TAIL),
    # hourly, nothing says contract or employee: not called a contract
    (None, "$40–$55/hr (≈$83K–$114K/yr)", None, "Hourly role — " + TAIL),
    ("W2", "$40/hr (≈$83K/yr)", "Some Studio", "Hourly role" + VIA + " — " + TAIL),
])
def test_note_says_only_what_is_known(etype, comp, company, note):
    assert contract_pay_note(etype, comp, company) == note


@pytest.mark.parametrize("etype,comp,company", [
    ("Contract", None, "Quiet Harbor Co"),
    ("Contract", "$140K–$200K/yr", "Harbor Lights Media"),
    ("Temp", "$110K–$140K/yr", "Puget Public Radio"),
    ("Contract", "$400–$900/day (≈$104K–$234K/yr)", "Bramble Pictures"),
])
def test_no_hourly_or_agency_claim_without_evidence(etype, comp, company):
    note = contract_pay_note(etype, comp, company)
    assert "paid hourly" not in note
    assert "W2" not in note and "1099" not in note and "agency" not in note


@pytest.mark.parametrize("company,expected", [
    ("Aquent", True), ("Aquent (staffing)", True), ("Apple (via WillHire)", True),
    ("Robert Half", True), ("Insight Global", True), ("Motion Recruitment", True),
    ("Eastridge Workforce Solutions", True), ("CELLA", True),
    ("Harbor Lights Media", False), ("Puget Public Radio", False), ("Bramble Pictures", False),
    ("Cellar Door", False),
    (None, False), ("nan", False),
])
def test_staffing_agency(company, expected):
    assert is_staffing_agency(company) is expected


def test_base_wording():
    assert CONTRACT_PAY_NOTE == BASE


# ── digest harness ───────────────────────────────────────────────────────────
class FakeDB:
    def __init__(self, due):
        self.due = due
        self.asked = None

    def get_followups_due(self, days):
        self.asked = list(days)
        return [f for f in self.due if f["days_since"] in days]

    def get_company_intel(self, key):
        return None


def _send(monkeypatch, df, db=None, morning=True):
    captured = {}

    class FakeSMTP:
        def login(self, *a):
            pass

        def sendmail(self, sender, recips, msg_str):
            captured["msg"] = msg_str

        def quit(self):
            pass

    monkeypatch.setenv("EMAIL_SENDER", "digest@test.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "secret")
    monkeypatch.setattr("job_agent.postprocess.smtplib.SMTP_SSL", lambda *a, **kw: FakeSMTP())
    monkeypatch.setattr(postprocess, "_is_morning_digest", lambda now: morning)
    postprocess.send_email_with_results(None, "candidate@test.com", df, db=db)
    msg = email.message_from_string(captured["msg"])
    part = next(p for p in msg.walk() if p.get_content_type() == "text/html")
    raw = part.get_payload(decode=True)
    return raw.decode("ascii")  # raises if the body is not pure ASCII


def _jobs():
    return pd.DataFrame([
        {"title": "Senior Video Producer", "company": "Studio Ü", "location": "Seattle, WA",
         "job_url": "https://example.com/1", "tier": "strong", "match_level": "HIGH_MATCH",
         "confidence": 0.9, "gpt_reasoning": "Strong production fit",
         "comp_text": "$120K–$150K/yr", "employment_type": "Full-time"},
        {"title": "Studio Producer", "company": "Agency Co", "location": "Bellevue, WA",
         "job_url": "https://example.com/2", "tier": "strong", "match_level": "HIGH_MATCH",
         "confidence": 0.85, "gpt_reasoning": "Contract studio work",
         "comp_text": "$45/hr (≈$94K/yr)", "employment_type": "Contract · W2"},
        {"title": "Content Producer", "company": "Media Co", "location": "Remote",
         "job_url": "https://example.com/3", "tier": "worth_a_look", "match_level": "WORTH_A_LOOK",
         "confidence": 0.5, "gpt_reasoning": "Adjacent", "comp_text": None, "employment_type": None},
    ])


def _due(n_per_day=1):
    out = []
    for day in (2, 7, 14, 21):
        for i in range(n_per_day):
            out.append({"title": f"Role d{day} #{i}", "company": f"Co {day}-{i}",
                        "link": f"https://example.com/a/{day}/{i}", "days_since": day})
    return out


def test_followups_sit_below_strong_and_above_worth(monkeypatch):
    html = _send(monkeypatch, _jobs(), db=FakeDB(_due()))
    strong = html.index("STRONG MATCHES")
    follow = html.index("Follow-ups due today")
    worth = html.index("WORTH A LOOK &mdash;")
    assert strong < html.index("Senior Video Producer") < follow < worth


def test_day_21_is_never_listed_and_labels_are_right(monkeypatch):
    db = FakeDB(_due())
    html = _send(monkeypatch, _jobs(), db=db)
    assert db.asked == [2, 7, 14]
    assert "Day 21" not in html and "Role d21" not in html
    assert "Day 2: LinkedIn touch: connect with the hiring manager/recruiter" in html
    assert "Day 7: Email follow-up: one proof point, ask about timeline" in html
    assert "Day 14: Final nudge, then move on" in html
    assert ">Follow up<" not in html and ": Follow up</span>" not in html


def test_followups_capped_at_five(monkeypatch):
    html = _send(monkeypatch, _jobs(), db=FakeDB(_due(n_per_day=3)))  # 9 due (day 21 excluded)
    assert html.count("Role d") == 5
    assert "&hellip;and 4 more on the Kanban board." in html


def test_followups_only_in_the_morning_digest(monkeypatch):
    html = _send(monkeypatch, _jobs(), db=FakeDB(_due()), morning=False)
    assert "Follow-ups due today" not in html


def test_followups_also_on_a_no_match_morning(monkeypatch):
    html = _send(monkeypatch, pd.DataFrame(columns=["title", "tier"]), db=FakeDB(_due()))
    assert "Nothing new today" in html and "Follow-ups due today" in html


def test_morning_cutoff():
    from datetime import datetime
    assert postprocess._is_morning_digest(datetime(2026, 10, 4, 9, 0))
    assert not postprocess._is_morning_digest(datetime(2026, 10, 4, 15, 0))


def test_contract_note_on_contract_card_only(monkeypatch):
    html = _send(monkeypatch, _jobs(), db=None)
    assert html.count(NOTE_HTML) == 1
    i_contract = html.index("Studio Producer")
    i_note = html.index(NOTE_HTML)
    assert i_note > i_contract
    # The hourly rate the yearly figure was annualized from stays visible.
    assert "$45/hr (&#8776;$94K/yr)" in html
    # The salaried card says nothing about contracts.
    salaried = html[html.index("Senior Video Producer"):i_contract]
    assert "paid hourly" not in salaried


def test_body_is_ascii_html(monkeypatch):
    html = _send(monkeypatch, _jobs(), db=FakeDB(_due()))
    assert html.isascii()
    assert "Studio &#220;" in html  # non-ASCII company name survives as an entity


def test_digest_contract_note_on_yearly_and_unknown_pay(monkeypatch):
    """A contract card with N/A pay must not say 'Contract — paid hourly'.
    Yearly / unknown-pay contract cards get the base note only."""
    df = pd.DataFrame([
        {"title": "AI Video Producer (Contract)", "company": "Quiet Harbor Co",
         "location": "Remote", "job_url": "https://example.com/9", "tier": "strong",
         "match_level": "HIGH_MATCH", "confidence": 0.8, "gpt_reasoning": "Fit",
         "comp_text": None, "employment_type": "Contract"},
        {"title": "Program Manager, Live Events", "company": "Harbor Lights Media",
         "location": "Redmond, WA", "job_url": "https://example.com/10", "tier": "strong",
         "match_level": "HIGH_MATCH", "confidence": 0.8, "gpt_reasoning": "Fit",
         "comp_text": "$140K–$200K/yr", "employment_type": "Contract"},
    ])
    html = _send(monkeypatch, df, db=None)
    base_html = BASE.replace("—", "&#8212;")
    assert html.count(base_html) == 2
    assert "paid hourly" not in html
    assert "W2 through an agency" not in html
