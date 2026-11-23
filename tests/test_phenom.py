"""Tests for the Phenom career-site scraper (Disney's join.disneycareers.com).

Hermetic: requests.Session is mocked, no network."""

from unittest.mock import MagicMock, patch

import pytest

from job_agent import ats_scraper
from job_agent import scraper_phenom as ph

TOKEN = "https://join.disneycareers.com/global/en"
KEYWORDS = ["Producer", "Video Producer", "Content Manager"]


def _job(job_id, title, posted="2026-03-16T21:42:13.000+0000", **extra):
    job = {
        "jobId": job_id,
        "title": title,
        "location": "Orlando, Florida, United States of America",
        "postedDate": posted,
        "companyName": "Walt Disney Pictures",
    }
    job.update(extra)
    return job


def _page(jobs, total):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"refineSearch": {"status": 200, "hits": len(jobs), "totalHits": total,
                                               "data": {"jobs": jobs}}}
    return resp


def _posting_page(description_html):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.text = (
        '<html><head><script type="application/ld+json">'
        '{"@context":"https://schema.org","@type":"JobPosting","title":"x",'
        f'"datePosted":"2026-03-19","description":"{description_html}"}}'
        "</script></head><body></body></html>"
    )
    return resp


@pytest.fixture
def session():
    """The mocked requests.Session the scraper builds; sleeps are skipped."""
    with patch("job_agent.scraper_phenom.requests.Session") as cls, \
            patch("job_agent.scraper_phenom.time.sleep"):
        sess = MagicMock()
        cls.return_value = sess
        ats_scraper._clear_board_report()
        yield sess
        ats_scraper._clear_board_report()


def test_request_body_asks_for_the_whole_board(session):
    session.post.return_value = _page([_job("1", "Accountant")], 1)
    ph.scrape_phenom(TOKEN, "Disney", KEYWORDS)

    url = session.post.call_args.args[0]
    body = session.post.call_args.kwargs["json"]
    assert url == "https://join.disneycareers.com/widgets"
    assert body["ddoKey"] == "refineSearch"
    assert body["keywords"] == ""
    assert body["from"] == 0 and body["size"] == ph._PAGE_SIZE
    assert body["lang"] == "en_global" and body["country"] == "global"
    assert body["jobs"] is True and body["pageName"] == "search-results"


def test_pages_until_total_and_reports_a_complete_board(session, monkeypatch):
    monkeypatch.setattr(ph, "_PAGE_SIZE", 2)
    session.post.side_effect = [
        _page([_job("1", "Video Producer"), _job("2", "Accountant")], 3),
        _page([_job("3", "Line Cook")], 3),
    ]
    rows = ph.scrape_phenom(TOKEN, "Disney", KEYWORDS, fetch_details=False)

    assert [c.kwargs["json"]["from"] for c in session.post.call_args_list] == [0, 2]
    assert [r["title"] for r in rows] == ["Video Producer"]
    report = ats_scraper._take_board_report()
    assert report["platform"] == "phenom" and report["token"] == TOKEN
    assert report["ok"] and report["complete"] and report["total_count"] == 3
    # Every listed posting, before the keyword filter, linked like job_url.
    assert ("https://join.disneycareers.com/global/en/job/3/Line-Cook", "Line Cook") in report["links"]


def test_a_posting_repeated_across_pages_makes_the_list_incomplete(session, monkeypatch):
    # The API's order shifts between calls: a page boundary can repeat a
    # posting and skip another. Counting the repeat would hide the skip.
    monkeypatch.setattr(ph, "_PAGE_SIZE", 2)
    session.post.side_effect = [
        _page([_job("1", "Video Producer"), _job("2", "Accountant")], 4),
        _page([_job("2", "Accountant"), _job("3", "Line Cook")], 4),
    ]
    rows = ph.scrape_phenom(TOKEN, "Disney", KEYWORDS, fetch_details=False)

    assert len(rows) == 1
    report = ats_scraper._take_board_report()
    assert report["ok"] and not report["complete"]
    assert report["total_count"] == 3


def test_row_shape_date_link_location(session):
    session.post.return_value = _page([
        _job("10000030", "Producer – Example Studio Live (As-Needed)", posted="2026-03-10T23:03:08.000+0000",
             companyName="Example Studio Productions, Inc.",
             location="New York, New York, United States of America",
             multi_location=["New York, NY, USA", "Seattle, WA, USA"]),
    ], 1)
    rows = ph.scrape_phenom(TOKEN, "Disney", KEYWORDS, fetch_details=False)

    assert rows == [{
        "title": "Producer – Example Studio Live (As-Needed)",
        "company": "Disney",
        "location": "New York, NY, USA; Seattle, WA, USA",
        "job_url": "https://join.disneycareers.com/global/en/job/10000030/Producer-Example-Studio-Live-As-Needed",
        "description": "",
        "date_posted": "2026-03-10",
        "site": "phenom",
        "department": "Example Studio Productions, Inc.",
    }]


def test_single_location_keeps_the_full_place_name(session):
    session.post.return_value = _page([_job("7", "Video Producer", multi_location=["Orlando, FL, USA"])], 1)
    rows = ph.scrape_phenom(TOKEN, "Disney", KEYWORDS, fetch_details=False)
    assert rows[0]["location"] == "Orlando, Florida, United States of America"


def test_description_from_posting_page_newest_first_and_capped(session, monkeypatch):
    monkeypatch.setattr(ph, "DETAIL_FETCH_CAP", 2)
    session.post.return_value = _page([
        _job("1", "Video Producer", posted="2026-03-01T00:00:00.000+0000"),
        _job("2", "Content Producer", posted="2026-03-18T00:00:00.000+0000"),
        _job("3", "Podcast Producer", posted="2026-03-10T00:00:00.000+0000"),
    ], 3)
    # The JSON-LD HTML can arrive entity-encoded.
    session.get.return_value = _posting_page("&lt;p&gt;Make &lt;strong&gt;videos&lt;/strong&gt;.&lt;/p&gt;")

    rows = ph.scrape_phenom(TOKEN, "Disney", KEYWORDS)

    fetched = [c.args[0] for c in session.get.call_args_list]
    assert fetched == [
        "https://join.disneycareers.com/global/en/job/2/Content-Producer",
        "https://join.disneycareers.com/global/en/job/3/Podcast-Producer",
    ]
    by_title = {r["title"]: r["description"] for r in rows}
    assert by_title["Content Producer"] == "Make\nvideos\n."
    assert by_title["Podcast Producer"] == "Make\nvideos\n."
    assert by_title["Video Producer"] == ""  # over the cap: list data only


def test_description_fetch_stops_after_repeated_failures(session):
    session.post.return_value = _page(
        [_job(str(i), f"Video Producer {i}", posted=f"2026-03-{10 + i}T00:00:00.000+0000") for i in range(6)], 6)
    session.get.side_effect = Exception("timeout")

    rows = ph.scrape_phenom(TOKEN, "Disney", KEYWORDS)

    assert session.get.call_count == ph.DETAIL_MAX_FAILURES
    assert len(rows) == 6 and all(r["description"] == "" for r in rows)


def test_parse_description_ignores_other_json_ld():
    page = (
        '<script type="application/ld+json">{"@type":"WebPage","description":"site blurb"}</script>'
        '<script type="application/ld+json">not json</script>'
        '<script type="application/ld+json">{"@type":"JobPosting","description":"<p>Edit video.</p>"}</script>'
    )
    assert ph._parse_description(page) == "Edit video."
    assert ph._parse_description("<html></html>") == ""


def test_first_page_error_returns_nothing_and_reports_a_failed_fetch(session):
    session.post.side_effect = Exception("HTTP 503")
    assert ph.scrape_phenom(TOKEN, "Disney", KEYWORDS) == []
    report = ats_scraper._take_board_report()
    assert report["ok"] is False and report["complete"] is False


def test_malformed_answer_is_a_failed_fetch(session):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"error": "bad request"}
    session.post.return_value = resp
    assert ph.scrape_phenom(TOKEN, "Disney", KEYWORDS) == []
    assert ats_scraper._take_board_report()["ok"] is False


def test_later_page_error_keeps_rows_but_list_is_incomplete(session, monkeypatch):
    monkeypatch.setattr(ph, "_PAGE_SIZE", 1)
    session.post.side_effect = [_page([_job("1", "Video Producer")], 2), Exception("HTTP 429")]
    rows = ph.scrape_phenom(TOKEN, "Disney", KEYWORDS, fetch_details=False)
    assert [r["title"] for r in rows] == ["Video Producer"]
    report = ats_scraper._take_board_report()
    assert report["ok"] and not report["complete"]


def test_token_without_country_and_language_is_rejected(session):
    assert ph.scrape_phenom("https://join.disneycareers.com", "Disney", KEYWORDS) == []
    session.post.assert_not_called()


def test_ats_dispatcher_routes_phenom_entries():
    with patch("job_agent.scraper_phenom.scrape_phenom") as fake:
        fake.return_value = [{"title": "Video Producer", "company": "Disney", "location": "Orlando, FL",
                              "job_url": "https://join.disneycareers.com/global/en/job/1/Video-Producer",
                              "description": "", "date_posted": "2026-03-16", "site": "phenom",
                              "department": ""}]
        df = ats_scraper.scrape_ats_boards(
            [{"company": "Disney", "ats": "phenom", "token": TOKEN}], ["Video Producer"])
    fake.assert_called_once_with(TOKEN, "Disney", ["Video Producer"])
    assert df.iloc[0]["site"] == "phenom"
    assert "keyword_category" in df.columns


def test_a_null_title_is_skipped_not_fatal(session):
    # A null title used to raise in _matches_keywords, and with no per-board
    # guard in scrape_ats_boards that stopped every later board.
    session.post.return_value = _page([_job("1", None), _job("2", "Video Producer")], 2)
    rows = ph.scrape_phenom(TOKEN, "Disney", KEYWORDS, fetch_details=False)
    assert [r["title"] for r in rows] == ["Video Producer"]
    report = ats_scraper._take_board_report()
    assert ("https://join.disneycareers.com/global/en/job/1", "") in report["links"]


# --- The pipeline's phase timeout is never swallowed ---

class _ScrapeTimeout(Exception):
    """Stands in for main._ScrapeTimeout (matched by name, as in ats_scraper)."""


def test_phase_timeout_on_the_first_page_ends_the_scan(session):
    session.post.side_effect = _ScrapeTimeout("ATS phase timed out")
    with pytest.raises(_ScrapeTimeout):
        ph.scrape_phenom(TOKEN, "Disney", KEYWORDS)


def test_phase_timeout_on_a_later_page_ends_the_scan(session, monkeypatch):
    monkeypatch.setattr(ph, "_PAGE_SIZE", 1)
    session.post.side_effect = [_page([_job("1", "Video Producer")], 2),
                                _ScrapeTimeout("ATS phase timed out")]
    with pytest.raises(_ScrapeTimeout):
        ph.scrape_phenom(TOKEN, "Disney", KEYWORDS, fetch_details=False)


def test_phase_timeout_during_a_description_fetch_ends_the_scan(session):
    session.post.return_value = _page([_job("1", "Video Producer"), _job("2", "Content Producer")], 2)
    session.get.side_effect = _ScrapeTimeout("ATS phase timed out")
    with pytest.raises(_ScrapeTimeout):
        ph.scrape_phenom(TOKEN, "Disney", KEYWORDS)
    assert session.get.call_count == 1
