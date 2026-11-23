"""Shared test setup. Everything here keeps the suite hermetic and fictional.

- The candidate profile is pinned to tests/fixtures/test_profile.yaml (a
  fictional person, see tests/fixtures/README.md) BEFORE anything imports
  job_agent: candidate_profile reads $JOB_AGENT_PROFILE once, at import time,
  and config.py / cv_facts.py build their constants from it.
- Tracing is switched off, so no CascadeFilter() starts a tracing client.
- The Anthropic client is nulled, so an unmocked model call returns "" instead
  of making a paid request.
"""

import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
TEST_PROFILE = TESTS_DIR / "fixtures" / "test_profile.yaml"

# ── Pin the profile and switch tracing off (before the first job_agent import) ──
# Overwritten, not defaulted: a profile someone exports in their shell, or keeps
# at profiles/profile.yaml, must never change what the suite tests.
os.environ["JOB_AGENT_PROFILE"] = str(TEST_PROFILE)
# Kill switch honoured by observability.CascadeTracer.__init__: with tracing
# keys in the environment, every CascadeFilter() would otherwise start a real
# tracing client with background exporter threads.
os.environ["JOB_AGENT_DISABLE_LANGFUSE"] = "1"

sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest  # noqa: E402

from job_agent import candidate_profile as _candidate_profile  # noqa: E402

if _candidate_profile.PROFILE.source.resolve() != TEST_PROFILE.resolve():
    raise RuntimeError(
        f"job_agent was imported before tests/conftest.py pinned the profile: "
        f"the suite would run against {_candidate_profile.PROFILE.source} "
        f"instead of {TEST_PROFILE}."
    )


@pytest.fixture(autouse=True)
def _no_real_anthropic_calls(monkeypatch):
    """HERMETIC-TESTS GUARD. Several modules load a .env file at import time,
    so a real Anthropic key can be live inside pytest, and any unmocked
    call_llm path would make PAID API calls from unit tests (and slow the
    suite down by minutes). Null the client by default: an unmocked call_llm
    returns "" and every gate fails open. Tests that mock call_llm/gpt_filter
    are unaffected. A test that genuinely needs the live API can override this
    fixture explicitly."""
    try:
        monkeypatch.setattr("job_agent.gpt_filter.get_anthropic_client",
                            lambda: None)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _no_real_posting_page_fetches(monkeypatch):
    """HERMETIC-TESTS GUARD. generate_package looks up the
    posting text of a job listed without it, and posting_text reads the job's
    own page for that. Unit tests never reach a job site: the page read
    returns "" unless a test patches it (tests/test_posting_text.py overrides
    this fixture to test the real reader against mocked responses)."""
    try:
        monkeypatch.setattr("job_agent.posting_text.fetch_posting_text", lambda link: "")
    except Exception:
        pass


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary SQLite database for testing."""
    from job_agent.db_manager import DBManager
    db_path = str(tmp_path / "test.db")
    db = DBManager(db_path=db_path)
    yield db
    db.close()


@pytest.fixture
def sample_jobs():
    """5 mock job dicts for testing. Dates are FRESH relative to now: the
    pre-filter drops postings older than STALE_POSTING_DAYS, so hardcoded
    dates would silently empty every pipeline test as time passes.
    Locations: three in the fixture profile's home metro, one remote, one
    out of area."""
    from datetime import datetime, timedelta
    _d = lambda i: (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
    return [
        {
            "title": "Senior Video Producer",
            "company": "TechCorp",
            "location": "Seattle, WA",
            "job_url": "https://example.com/job/1",
            "date_posted": _d(4),
            "description": "Lead video production for marketing team. 5+ years experience required.",
        },
        {
            "title": "Content Producer",
            "company": "MediaHouse",
            "location": "Remote",
            "job_url": "https://example.com/job/2",
            "date_posted": _d(3),
            "description": "Create and manage digital content across platforms.",
        },
        {
            "title": "Social Media Manager",
            "company": "StartupCo",
            "location": "Bellevue, WA",
            "job_url": "https://example.com/job/3",
            "date_posted": _d(2),
            "description": "Manage social media accounts and community engagement.",
        },
        {
            "title": "Creative Director",
            "company": "BigMedia Inc",
            "location": "Redmond, WA",
            "job_url": "https://example.com/job/4",
            "date_posted": _d(1),
            "description": "Lead creative team of 20+. Oversee video and digital content strategy.",
        },
        {
            "title": "Junior Videographer",
            "company": "SmallShop",
            "location": "New York, NY",
            "job_url": "https://example.com/job/5",
            "date_posted": _d(0),
            "description": "Entry-level position. Shoot and edit video content.",
        },
    ]


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch gpt_filter.call_llm to return controlled responses, in order.
    A module that did `from .gpt_filter import call_llm` holds its own
    reference and needs its own patch."""
    responses = []

    def fake_call_llm(prompt, model="test", temperature=0.0, **kw):
        if responses:
            return responses.pop(0)
        return '{"match_level": "IRRELEVANT", "reason": "mock", "rejection_rule": "NONE", "confidence": 0.1}'

    monkeypatch.setattr("job_agent.gpt_filter.call_llm", fake_call_llm)
    return responses
