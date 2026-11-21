"""Hiring health (Phase 27): "is this company really hiring?"

Two parts under test, both hermetic (the search call is always mocked):
  - the free scanner signal read from our own scrape history (temp DB), and
  - dated, sourced public facts — cite-or-omit enforced in code, cached 14 days
    under __hiring_health__{company}, never cached or shown when the check is
    incomplete.
"""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from job_agent import perplexity_utils as pu
from job_agent.perplexity_utils import (
    HIRING_HEALTH_TTL_DAYS,
    _cite_or_omit,
    _hiring_health_query,
    get_hiring_health,
    get_scanner_signal,
)

YEAR = date.today().year
NOW = datetime.now(timezone.utc)          # created_at is UTC text in SQLite
CACHE_KEY = "__hiring_health__Acme"


def _add(db, title, link, days_ago, company="Acme", location="Seattle, WA"):
    """One scrape row first seen `days_ago` days back (created_at is UTC text)."""
    created = (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO scrapes (date, title, company, location, link, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (created[:10], title, company, location, link, created),
        )


def _answer(text, urls=("https://news.test/acme-layoffs",)):
    """What _agent_search returns: prose plus the grounding search results."""
    return {"content": text, "search_results": [{"url": u} for u in urls]}


def _age_cache(db, key, days):
    old = (date.today() - timedelta(days=days)).isoformat()
    with db.transaction() as conn:
        conn.execute("UPDATE company_intel SET last_checked = ? WHERE company_name = ?", (old, key))


GOOD = (
    f"- **Layoffs:** {YEAR}-03-12 — cut about 200 roles in marketing [1]\n"
    f"- **Funding / financial:** No public signal found"
)


# === Scanner signal (our own scrape history, temp DB) ===


class TestScannerSignal:
    def test_counts_listings_and_distinct_titles_in_window(self, tmp_db):
        _add(tmp_db, "Video Producer", "https://www.linkedin.com/jobs/view/1", 5)
        _add(tmp_db, "video  producer", "https://www.indeed.com/viewjob?jk=a", 4)   # same title
        _add(tmp_db, "Content Producer", "https://www.linkedin.com/jobs/view/2", 3)
        _add(tmp_db, "Editor", "https://www.linkedin.com/jobs/view/3", 120)         # outside 90 days
        _add(tmp_db, "Producer", "https://www.linkedin.com/jobs/view/4", 2, company="Other Co")

        out = get_scanner_signal("Acme", tmp_db)
        assert out.startswith("- **Seen by our scanner:** 3 listings from this company in the last 90 days")
        assert "2 distinct titles" in out
        assert "This exact title" not in out            # no title asked for

    def test_company_match_ignores_case_and_padding(self, tmp_db):
        _add(tmp_db, "Producer", "https://x.test/1", 1, company=" ACME ")
        assert "1 listing from" in get_scanner_signal("  acme", tmp_db)
        assert "1 distinct title." in get_scanner_signal("acme", tmp_db)

    def test_page_of_panels_scans_the_window_once(self, tmp_db):
        """The Review page asks for every row's signal at once: one pass over
        the 90-day window, then primary-key reads — never a scan per panel."""
        for n in range(5):
            _add(tmp_db, "Producer", f"https://x.test/{n}", 1, company=f"Co {n}")
        statements = []
        conn = tmp_db._get_thread_conn()
        conn.set_trace_callback(statements.append)
        try:
            for n in range(5):
                assert "1 listing from" in get_scanner_signal(f"Co {n}", tmp_db)
        finally:
            conn.set_trace_callback(None)
        assert sum("SELECT id, company FROM scrapes" in s for s in statements) == 1
        assert sum("WHERE id IN (" in s for s in statements) == 5

    def test_company_index_is_reused_then_refreshed(self, tmp_db):
        _add(tmp_db, "Producer", "https://x.test/1", 1)
        assert "1 listing from" in get_scanner_signal("Acme", tmp_db)
        _add(tmp_db, "Producer", "https://x.test/2", 1)
        assert "1 listing from" in get_scanner_signal("Acme", tmp_db)       # index still fresh
        pu._scanner_index["built_at"] -= pu._SCANNER_INDEX_TTL_SECONDS + 1
        assert "2 listings from" in get_scanner_signal("Acme", tmp_db)      # rebuilt after TTL

    def test_no_history_returns_none(self, tmp_db):
        assert get_scanner_signal("Acme", tmp_db) is None
        assert get_scanner_signal("", tmp_db) is None
        assert get_scanner_signal("Acme", None) is None

    def test_read_failure_returns_none(self):
        broken = MagicMock()
        broken._get_thread_conn.side_effect = RuntimeError("db gone")
        assert get_scanner_signal("Acme", broken) is None

    def test_title_not_seen(self, tmp_db):
        _add(tmp_db, "Editor", "https://x.test/1", 1)
        out = get_scanner_signal("Acme", tmp_db, "Video Producer")
        assert "- **This exact title:** not seen in the last 90 days." in out

    def test_cross_board_copies_are_not_a_relisting(self, tmp_db):
        """One job reaching four boards over three weeks is one job."""
        _add(tmp_db, "Video Producer", "https://careers.acme.test/job/1", 40)
        _add(tmp_db, "Video Producer", "https://www.linkedin.com/jobs/view/1", 39)
        _add(tmp_db, "Video Producer", "https://www.indeed.com/viewjob?jk=a", 30)
        _add(tmp_db, "Video Producer", "https://www.glassdoor.com/job-listing/x?jl=1", 20)

        out = get_scanner_signal("Acme", tmp_db, "Video Producer")
        first = (NOW - timedelta(days=40)).date().isoformat()
        latest = (NOW - timedelta(days=20)).date().isoformat()
        assert f"4 listings on 4 job boards, first seen {first}, latest {latest}" in out
        assert out.endswith("; no re-listing seen.")

    def test_same_board_new_listing_after_gap_is_a_relisting(self, tmp_db):
        _add(tmp_db, "Video Producer", "https://www.linkedin.com/jobs/view/1", 60)
        _add(tmp_db, "Video Producer", "https://www.linkedin.com/jobs/view/2", 46)   # +14 days
        _add(tmp_db, "Video Producer", "https://www.linkedin.com/jobs/view/3", 10)   # +36 days
        out = get_scanner_signal("Acme", tmp_db, "Video Producer")
        assert "re-listed 2 times (a new listing on the same job board and location, 14+ days" in out
        assert "3 listings on 1 job board," in out

    def test_gap_below_threshold_is_not_a_relisting(self, tmp_db):
        _add(tmp_db, "Video Producer", "https://www.linkedin.com/jobs/view/1", 30)
        _add(tmp_db, "Video Producer", "https://www.linkedin.com/jobs/view/2", 17)   # +13 days
        assert "no re-listing seen" in get_scanner_signal("Acme", tmp_db, "Video Producer")

    def test_gap_is_measured_from_the_last_counted_listing(self, tmp_db):
        """Listings 10 days apart: day 0, 10, 20 -> one re-listing (day 20 vs day 0)."""
        for n, ago in enumerate((40, 30, 20)):
            _add(tmp_db, "Video Producer", f"https://www.linkedin.com/jobs/view/{n}", ago)
        assert "re-listed 1 time (" in get_scanner_signal("Acme", tmp_db, "Video Producer")

    def test_other_location_on_same_board_is_not_a_relisting(self, tmp_db):
        """Same title in another city weeks later is a second opening, not a re-list."""
        _add(tmp_db, "Video Producer", "https://www.indeed.com/viewjob?jk=a", 50,
             location="Seattle, WA")
        _add(tmp_db, "Video Producer", "https://www.indeed.com/viewjob?jk=b", 10,
             location="Portland, OR")
        assert "no re-listing seen" in get_scanner_signal("Acme", tmp_db, "Video Producer")


# === Cite-or-omit (deterministic gate on the search answer) ===


class TestCiteOrOmit:
    def test_marker_becomes_link_to_search_result(self):
        out = _cite_or_omit(
            f"- **Layoffs:** {YEAR}-03-12 — cut 200 roles [1]",
            [{"url": "https://news.test/a"}],
        )
        assert out == f"- **Layoffs:** {YEAR}-03-12 — cut 200 roles ([source](https://news.test/a))"

    def test_line_without_source_is_dropped(self):
        out = _cite_or_omit(
            f"- **Layoffs:** {YEAR}-03-12 — cut 200 roles\n"
            f"- **Funding:** {YEAR}-01-05 — raised a Series C [1]",
            [{"url": "https://news.test/a"}],
        )
        assert "cut 200 roles" not in out
        assert "raised a Series C" in out
        assert "_1 line without a dated source omitted._" in out

    def test_marker_out_of_range_is_not_a_source(self):
        assert _cite_or_omit(f"- **Layoffs:** {YEAR}-03-12 — cuts [7]", [{"url": "https://n.test/a"}]) is None

    def test_line_without_recent_date_is_dropped(self):
        results = [{"url": "https://news.test/a"}]
        assert _cite_or_omit("- **Layoffs:** cut 200 roles [1]", results) is None
        assert _cite_or_omit(f"- **Layoffs:** {YEAR - 3}-03-12 — cut 200 roles [1]", results) is None
        months_ago = (date.today() - timedelta(days=200)).isoformat()
        assert _cite_or_omit(f"- **Layoffs:** {months_ago} — cut 200 roles [1]", results) is not None

    def test_year_inside_the_url_is_not_a_date(self):
        out = _cite_or_omit(
            f"- **Layoffs:** cut 200 roles https://news.test/{YEAR}/03/story",
            [{"url": f"https://news.test/{YEAR}/03/story"}],
        )
        assert out is None

    def test_inline_url_must_come_from_a_grounding_site(self):
        results = [{"url": "https://www.news.test/a"}]
        ok = _cite_or_omit(f"- **Funding:** {YEAR}-02-01 — raised $50M — https://news.test/a", results)
        assert ok and "https://news.test/a" in ok
        invented = _cite_or_omit(
            f"- **Funding:** {YEAR}-02-01 — raised $50M — https://made-up.test/x", results)
        assert invented is None

    def test_answer_without_grounding_results_is_incomplete(self):
        """No search results = nothing grounded the answer: not a check, even
        for a dated line with a URL, and not a "no signal" either."""
        assert _cite_or_omit(f"- **Funding:** {YEAR}-02-01 — raised $50M — https://news.test/a", []) is None
        assert _cite_or_omit("- **Layoffs:** No public signal found", None) is None

    def test_full_date_must_be_inside_the_last_12_months(self):
        results = [{"url": "https://news.test/a"}]
        today = date.today()
        recent = (today - timedelta(days=300)).isoformat()
        stale = (today - timedelta(days=500)).isoformat()      # last year, but 16 months back
        assert _cite_or_omit(f"- **Layoffs:** {recent} — cut 200 roles [1]", results) is not None
        assert _cite_or_omit(f"- **Layoffs:** {stale} — cut 200 roles [1]", results) is None

    def test_named_and_month_dates_are_recognised(self):
        results = [{"url": "https://news.test/a"}]
        month = (date.today() - timedelta(days=40)).strftime("%B %d, %Y")
        old_month = (date.today() - timedelta(days=600)).strftime("%b %Y")
        assert _cite_or_omit(f"- **Funding:** {month} — raised a round [1]", results) is not None
        assert _cite_or_omit(f"- **Funding:** {old_month} — raised a round [1]", results) is None
        ym = (date.today() - timedelta(days=60)).strftime("%Y-%m")
        assert _cite_or_omit(f"- **Funding:** {ym} — raised a round [1]", results) is not None

    def test_long_topic_label_before_no_signal_is_kept(self):
        out = _cite_or_omit(
            "- **Leadership changes in the content, production or marketing organization:** "
            "No public signal found",
            [{"url": "https://news.test/a"}],
        )
        assert out.endswith("organization:** No public signal found")

    def test_claim_dressed_as_a_label_needs_a_source(self):
        """A short claim ending in a colon is not a topic label."""
        results = [{"url": "https://news.test/a"}]
        assert _cite_or_omit("- **Layoffs:** cut 500 roles; freezes: No public signal found", results) is None
        assert _cite_or_omit("- **Studio closed, staff let go:** No public signal found", results) is None

    def test_no_signal_line_kept_and_trailing_claim_cut(self):
        out = _cite_or_omit(
            "- **Layoffs:** No public signal found, though rumours suggest cuts are coming [2]",
            [{"url": "https://news.test/a"}],
        )
        assert out == "- **Layoffs:** No public signal found"

    def test_claim_before_the_no_signal_phrase_needs_a_source(self):
        text = ("- **Layoffs:** Rumours of large cuts across every studio team circulated widely, "
                "but No public signal found")
        assert _cite_or_omit(text, [{"url": "https://news.test/a"}]) is None

    def test_nothing_survives_returns_none(self):
        assert _cite_or_omit("Here is a summary of what I think is going on.", []) is None
        assert _cite_or_omit("", []) is None

    def test_source_on_a_sub_bullet_counts_for_its_bullet(self):
        out = _cite_or_omit(
            f"- **Layoffs:** {YEAR}-03-12 — cut 200 roles\n  - Source: https://news.test/a",
            [{"url": "https://news.test/a"}],
        )
        assert "cut 200 roles" in out and "https://news.test/a" in out

    def test_table_rows_become_bullets(self):
        """Format drift: a table answer still yields one readable fact per row."""
        out = _cite_or_omit(
            "| Topic | Date | Fact | Source |\n|---|---|---|---|\n"
            f"| Layoffs | {YEAR}-03-12 | cut 200 roles | [1] |\n"
            "| Funding | | No public signal found | |",
            [{"url": "https://news.test/a"}],
        )
        lines = out.splitlines()
        assert lines[0] == f"- Layoffs — {YEAR}-03-12 — cut 200 roles — ([source](https://news.test/a))"
        assert lines[1] == "- Funding — No public signal found"
        assert "_1 line without a dated source omitted._" in out      # the header row only

    def test_heading_kept_only_above_a_surviving_line(self):
        out = _cite_or_omit(
            f"## Layoffs\n- cut 200 roles\n## Funding\n- {YEAR}-01-05 — raised a Series C [1]",
            [{"url": "https://news.test/a"}],
        )
        assert "## Funding" in out
        assert "## Layoffs" not in out


# === get_hiring_health (search mocked) ===


@patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
class TestGetHiringHealth:
    def test_cache_miss_searches_and_caches(self, tmp_db):
        _add(tmp_db, "Video Producer", "https://www.linkedin.com/jobs/view/1", 3)
        with patch.object(pu, "_agent_search", return_value=_answer(GOOD)) as search:
            out = get_hiring_health("Acme", db=tmp_db, job_title="Video Producer")

        search.assert_called_once()
        assert search.call_args.kwargs["preset"] == "low"
        # Scanner signal is the first lines of the section, then the public facts.
        assert out.startswith("- **Seen by our scanner:** 1 listing")
        assert out.index("This exact title") < out.index("**Public sources**")
        assert "([source](https://news.test/acme-layoffs))" in out
        assert "- **Funding / financial:** No public signal found" in out
        # Cached: public facts only — the scanner signal is recomputed each call.
        row = tmp_db.get_company_intel(CACHE_KEY)
        assert f"**Public sources** (checked {row['last_checked']}):" in out
        assert "cut about 200 roles" in row["company_profile"]
        assert "Seen by our scanner" not in row["company_profile"]

    def test_cache_hit_skips_search(self, tmp_db):
        tmp_db.save_company_intel({"company_name": CACHE_KEY, "company_profile": "- cached fact"})
        with patch.object(pu, "_agent_search") as search:
            out = get_hiring_health("Acme", db=tmp_db)
        search.assert_not_called()
        assert "- cached fact" in out

    def test_ttl_boundary(self, tmp_db):
        tmp_db.save_company_intel({"company_name": CACHE_KEY, "company_profile": "- cached fact"})
        _age_cache(tmp_db, CACHE_KEY, HIRING_HEALTH_TTL_DAYS - 1)
        with patch.object(pu, "_agent_search") as search:
            get_hiring_health("Acme", db=tmp_db)
        search.assert_not_called()

        _age_cache(tmp_db, CACHE_KEY, HIRING_HEALTH_TTL_DAYS)
        aged = tmp_db.get_company_intel(CACHE_KEY)["last_checked"]
        with patch.object(pu, "_agent_search", return_value=_answer(GOOD)) as search:
            out = get_hiring_health("Acme", db=tmp_db)
        search.assert_called_once()
        assert "cut about 200 roles" in out and "cached fact" not in out
        assert tmp_db.get_company_intel(CACHE_KEY)["last_checked"] > aged   # re-stamped

    def test_empty_result_is_not_cached_and_not_called_no_signal(self, tmp_db):
        _add(tmp_db, "Video Producer", "https://www.linkedin.com/jobs/view/1", 3)
        with patch.object(pu, "_agent_search", return_value=None):
            out = get_hiring_health("Acme", db=tmp_db)
        assert tmp_db.get_company_intel(CACHE_KEY) is None
        assert out.startswith("- **Seen by our scanner:**")
        assert "Public-source check did not complete" in out
        assert "No public signal found" not in out

    def test_unsourced_answer_is_not_cached(self, tmp_db):
        unsourced = _answer("The company seems healthy and is probably hiring a lot.")
        with patch.object(pu, "_agent_search", return_value=unsourced):
            out = get_hiring_health("Acme", db=tmp_db)
        assert tmp_db.get_company_intel(CACHE_KEY) is None
        assert "probably hiring" not in out
        assert "Public-source check did not complete" in out

    def test_timeout_falls_back_and_caches_nothing(self, tmp_db):
        """A hung search surfaces as a requests timeout: no cache write, no guess."""
        with patch.object(pu, "_rate_limiter"), \
                patch("job_agent.perplexity_utils.requests.post",
                      side_effect=requests.exceptions.Timeout("read timed out")) as post:
            out = get_hiring_health("Acme", db=tmp_db)
        assert post.call_args.kwargs["timeout"] == pu.PERPLEXITY_TIMEOUT
        assert tmp_db.get_company_intel(CACHE_KEY) is None
        assert "Public-source check did not complete" in out

    def test_hung_search_is_abandoned_at_the_deadline(self, tmp_db, monkeypatch):
        """requests' timeout bounds each socket wait, not the whole call: a
        search that never returns is cut off by the wall-clock deadline, and
        its late answer is never cached."""
        import threading
        import time
        release = threading.Event()

        def _hang(query, preset=None):
            release.wait(5)
            return _answer(GOOD)

        monkeypatch.setattr(pu, "HIRING_HEALTH_SEARCH_DEADLINE_SEC", 0.2)
        try:
            with patch.object(pu, "_agent_search", side_effect=_hang):
                started = time.monotonic()
                out = get_hiring_health("Acme", db=tmp_db)
                elapsed = time.monotonic() - started
        finally:
            release.set()
        assert elapsed < 2
        assert "Public-source check did not complete" in out
        assert tmp_db.get_company_intel(CACHE_KEY) is None

    def test_search_exception_is_contained(self, tmp_db):
        with patch.object(pu, "_agent_search", side_effect=RuntimeError("boom")):
            out = get_hiring_health("Acme", db=tmp_db)
        assert "Public-source check did not complete" in out
        assert tmp_db.get_company_intel(CACHE_KEY) is None

    def test_failed_search_keeps_older_facts(self, tmp_db):
        tmp_db.save_company_intel({"company_name": CACHE_KEY, "company_profile": "- older fact"})
        _age_cache(tmp_db, CACHE_KEY, 40)
        checked = tmp_db.get_company_intel(CACHE_KEY)["last_checked"]
        with patch.object(pu, "_agent_search", return_value=None):
            out = get_hiring_health("Acme", db=tmp_db)
        assert f"**Public sources** (checked {checked}):" in out
        assert "- older fact" in out
        assert tmp_db.get_company_intel(CACHE_KEY)["last_checked"] == checked   # not re-stamped

    def test_cached_only_never_searches(self, tmp_db):
        _add(tmp_db, "Video Producer", "https://www.linkedin.com/jobs/view/1", 3)
        with patch.object(pu, "_agent_search") as search:
            bare = get_hiring_health("Acme", db=tmp_db, cached_only=True)
            tmp_db.save_company_intel({"company_name": CACHE_KEY, "company_profile": "- older fact"})
            _age_cache(tmp_db, CACHE_KEY, 40)
            with_cache = get_hiring_health("Acme", db=tmp_db, cached_only=True)
        search.assert_not_called()
        assert bare.startswith("- **Seen by our scanner:**")
        assert "Public" not in bare                       # no check attempted, no note
        assert "- older fact" in with_cache               # past-TTL facts still shown

    def test_query_carries_the_company_name_only(self, tmp_db):
        with patch.object(pu, "_agent_search", return_value=_answer(GOOD)) as search:
            get_hiring_health("Acme", db=tmp_db, job_title="Senior Video Producer")
        query = search.call_args.args[0]
        assert "Acme" in query
        assert "Producer" not in query
        assert query == _hiring_health_query("Acme")
        for topic in ("Layoffs", "freezes", "Funding", "Open roles", "Leadership changes"):
            assert topic in query
        assert "No public signal found" in query

    def test_without_db_returns_research_only(self):
        with patch.object(pu, "_agent_search", return_value=_answer(GOOD)):
            out = get_hiring_health("Acme")
        assert out.startswith("**Public sources**")

    def test_blank_company_returns_none(self, tmp_db):
        with patch.object(pu, "_agent_search") as search:
            assert get_hiring_health("  ", db=tmp_db) is None
        search.assert_not_called()


def test_disabled_search_still_gives_scanner_signal(tmp_db):
    """No Perplexity key: the free signal still shows, with no failure note."""
    _add(tmp_db, "Video Producer", "https://www.linkedin.com/jobs/view/1", 3)
    with patch.object(pu, "USE_PERPLEXITY", False), patch.object(pu, "_agent_search") as search:
        out = get_hiring_health("Acme", db=tmp_db)
    search.assert_not_called()
    assert out.startswith("- **Seen by our scanner:**")
    assert "did not complete" not in out


# === Company-intel endpoints ===


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from job_agent import api
    from job_agent.db_manager import DBManager
    db = DBManager(db_path=str(tmp_path / "intel_api.db"))
    monkeypatch.setattr(api, "_db", db)
    yield TestClient(api.app), db
    db.close()


def test_get_serves_hiring_health_without_a_search(client):
    c, db = client
    _add(db, "Video Producer", "https://www.linkedin.com/jobs/view/1", 3)
    db.save_company_intel({"company_name": CACHE_KEY, "company_profile": "- cached fact"})
    _age_cache(db, CACHE_KEY, 40)                     # past TTL: a refresh WOULD search
    with patch.object(pu, "USE_PERPLEXITY", True), patch.object(pu, "_agent_search") as search:
        body = c.get("/api/company-intel/Acme", params={"job_title": "Video Producer"}).json()
    search.assert_not_called()                        # opening a panel never spends a call
    assert body["hiring_health"].startswith("- **Seen by our scanner:** 1 listing")
    assert "This exact title" in body["hiring_health"]
    assert "- cached fact" in body["hiring_health"]
    assert body["company_profile"] is None            # nothing else was invented


def test_get_without_history_or_cache_is_null(client):
    c, _db = client
    with patch.object(pu, "USE_PERPLEXITY", True), patch.object(pu, "_agent_search") as search:
        assert c.get("/api/company-intel/Nobody").json()["hiring_health"] is None
    search.assert_not_called()


def test_enrich_returns_hiring_health(client, monkeypatch):
    c, db = client
    from job_agent import gpt_filter
    _add(db, "Video Producer", "https://www.linkedin.com/jobs/view/1", 3)
    monkeypatch.setattr(gpt_filter, "load_profile_text", lambda path: "profile")
    monkeypatch.setattr(pu, "generate_salary_recommendation", lambda **kw: None)
    with patch.object(pu, "USE_PERPLEXITY", True), \
            patch.object(pu, "_raw_search", return_value=None), \
            patch.object(pu, "_agent_search", return_value=_answer(GOOD)) as search:
        r = c.post("/api/company-intel/enrich",
                   json={"company_name": "Acme", "job_title": "Video Producer"})
    assert r.status_code == 200
    body = r.json()
    assert search.call_count == 1                     # hiring health is the only grounded call
    assert "Producer" not in search.call_args.args[0]
    assert body["hiring_health"].startswith("- **Seen by our scanner:**")
    assert "cut about 200 roles" in body["hiring_health"]
    assert "Hiring health" not in " ".join(body["errors"])


# === Batch prefetch before generation ===


def _batch_job(db, monkeypatch):
    """A tracked job that already has a CV (no generation) at a company whose
    profile is cached — the branch where the older intel calls are skipped."""
    from job_agent import api, config
    monkeypatch.setattr(api, "_db", db)
    monkeypatch.setattr(config, "USE_PERPLEXITY", True)
    _add(db, "Video Producer", "https://www.linkedin.com/jobs/view/1", 3)
    with db.transaction() as conn:
        conn.execute("UPDATE scrapes SET cv_path = 'cv/x.docx'")
        job_id = conn.execute("SELECT id FROM scrapes").fetchone()[0]
    db.save_company_intel({"company_name": "Acme", "company_profile": "A company."})
    return api, job_id


def test_batch_prefetch_warms_hiring_health_even_with_cached_profile(tmp_db, monkeypatch):
    api, job_id = _batch_job(tmp_db, monkeypatch)
    with patch.object(pu, "get_hiring_health", return_value="ok") as health:
        result = api._process_batch_job(job_id, "profile")
    health.assert_called_once_with("Acme", db=tmp_db, job_title="Video Producer")
    assert result == {"job_id": job_id, "cv": "exists", "intel": "cached", "error": None}


def test_batch_survives_a_hiring_health_failure(tmp_db, monkeypatch):
    api, job_id = _batch_job(tmp_db, monkeypatch)
    with patch.object(pu, "get_hiring_health", side_effect=RuntimeError("boom")):
        result = api._process_batch_job(job_id, "profile")
    assert result["intel"] == "cached" and result["error"] is None


# === Web wiring (source-level guards; skipped where the web app is absent) ===


def _web_sources():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "web" / "src"
    if not root.is_dir():
        pytest.skip("web app not present")
    return {p: p.read_text(encoding="utf-8") for p in root.rglob("*.ts*")}


def test_chat_ui_is_removed_from_the_web_app():
    """2026-10-03: the chat drawer is gone and nothing calls the chat routes."""
    for path, text in _web_sources().items():
        assert "ChatDrawer" not in text, path
        assert "/api/chat" not in text, path
    assert not any(p.name == "ChatDrawer.tsx" for p in _web_sources())


def test_company_panel_renders_hiring_health():
    sources = {p.name: text for p, text in _web_sources().items()}
    assert "hiring_health" in sources["types.ts"]
    panel = sources["CompanyIntelPanel.tsx"]
    assert "Hiring Health" in panel and "{intel.hiring_health}" in panel
