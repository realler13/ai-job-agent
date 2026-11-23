"""Two fixes found by an audit.

1. A dispute about a base-CV role header was charged to the tailored summary,
   which was then thrown away: the judge quotes a role header as its title and
   company / date line joined by " | ", and the shipped text kept them apart.
2. One rate-limit reply ended the whole Netflix (Eightfold) fetch for a run.
Hermetic: no network, no model."""
from types import SimpleNamespace

from job_agent import docs_builder
from job_agent.cv_tailor import quote_in, summary_verdict


# ── 1. summary gate: a quoted role header is base text ────────────────────

EXPERIENCE = [
    {"title": "Segment Producer", "description": ""},                       # title placeholder
    {"title": "Harbor Lights Media | Portland | March 2015 – June 2019",
     "description": "• Produced 40 studio segments a year\n• Managed a $90K production budget"},
]
HEADER = "Segment Producer | Harbor Lights Media | Portland | March 2015 – June 2019"


def _check_with(claims):
    return {"state": "ok", "audit": {"overall_pass": False, "claims": claims}}


def test_the_header_as_the_judge_reads_it_is_shipped_text():
    shipped = docs_builder._shipped_text(EXPERIENCE, "Budgeting, scheduling")
    assert quote_in(HEADER, shipped)
    assert quote_in("Produced 40 studio segments a year", shipped)          # bullets as before
    assert quote_in("Budgeting, scheduling", shipped)


def test_a_disputed_role_header_is_not_charged_to_the_summary():
    """The case that discarded the tailored summary: the judge flags the role
    header (its city is not in the sources) under 'experience'."""
    check = _check_with([{"section": "experience", "role_index": 1, "verdict": "unsupported", "quote": HEADER}])
    shipped = docs_builder._shipped_text(EXPERIENCE, "Budgeting, scheduling")
    assert summary_verdict(check, summary="Hands-on producer.", shipped_text=shipped) is None


def test_an_invented_claim_still_counts_against_the_summary():
    """The opposite direction must keep working: a claim found in neither the
    summary nor the shipped text cannot be located, so the summary pays."""
    check = _check_with([{"section": "experience", "verdict": "unsupported",
                          "quote": "Led a team of 30 editors across three continents"}])
    shipped = docs_builder._shipped_text(EXPERIENCE, "Budgeting, scheduling")
    verdict = summary_verdict(check, summary="Hands-on producer.", shipped_text=shipped)
    assert verdict and "not in the shipped" in verdict


def test_a_claim_filed_under_summary_still_counts():
    check = _check_with([{"section": "summary", "verdict": "unsupported", "quote": "award-winning"}])
    shipped = docs_builder._shipped_text(EXPERIENCE, "")
    assert "summary claim unsupported" in summary_verdict(check, summary="An award-winning producer.",
                                                          shipped_text=shipped)


# ── 2. Netflix: one 429 no longer ends the fetch ──────────────────────────

def _resp(status, positions=None, count=1):
    r = SimpleNamespace(status_code=status)
    r.json = lambda: {"positions": positions or [], "count": count}

    def raise_for_status():
        if status >= 400:
            raise RuntimeError(f"{status} Client Error")
    r.raise_for_status = raise_for_status
    return r


def test_a_rate_limit_reply_is_retried(monkeypatch):
    from job_agent import scraper_eightfold_apple as mod
    position = {"id": 77, "name": "Video Producer", "location": "Remote, United States", "t_create": 1731700000}
    replies = [_resp(429), _resp(200, [position]), _resp(200, [])]
    waits = []
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(mod.time, "sleep", waits.append)
    jobs = mod.scrape_eightfold("netflix.com", "Netflix", ["producer"])
    assert [j["title"] for j in jobs] == ["Video Producer"]
    assert waits == [5]


def test_a_persistent_rate_limit_gives_up_after_three_waits(monkeypatch):
    from job_agent import scraper_eightfold_apple as mod
    waits = []
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _resp(429))
    monkeypatch.setattr(mod.time, "sleep", waits.append)
    assert mod.scrape_eightfold("netflix.com", "Netflix", ["producer"]) == []
    assert waits == [5, 15, 30]
