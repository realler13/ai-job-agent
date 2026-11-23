"""Workday paging: the first page's total is kept, and a seed that only
repeats an earlier seed's list stops paging.

Hermetic: a fake board answers session.post the way Workday's CXS API does
(the total on the first page only, 0 on later pages)."""

from unittest.mock import MagicMock, patch

import pytest

from job_agent import scraper_workday as wd

BASE = "https://acme.wd5.myworkdayjobs.com/External"
KEYWORDS = ["Video Producer"]


def _postings(prefix, n, start=0):
    return [{"title": f"Analyst {prefix}{i}", "externalPath": f"/job/{prefix}{i}",
             "locationsText": "Seattle, WA", "postedOn": "Posted Today"}
            for i in range(start, start + n)]


def _run(lists: dict, max_pages=25):
    """Scrape with the fake board; lists = {seed: [postings in server order]}.
    Returns (rows, {seed: list requests})."""
    calls: dict = {}

    def post(url, json, timeout):
        seed, offset, limit = json["searchText"], json["offset"], json["limit"]
        calls[seed] = calls.get(seed, 0) + 1
        found = lists.get(seed, [])
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "jobPostings": found[offset:offset + limit],
            # Workday sends the total on the first page only.
            "total": len(found) if offset == 0 else 0,
        }
        return resp

    with patch("job_agent.scraper_workday.requests.Session") as cls, \
            patch("job_agent.scraper_workday.time.sleep"), \
            patch.object(wd, "_MAX_PAGES", max_pages):
        sess = MagicMock()
        sess.post.side_effect = post
        cls.return_value = sess
        rows = wd.scrape_workday(BASE, "Acme", KEYWORDS, fetch_details=False)
    return rows, calls


def test_pages_past_the_second_page_when_later_pages_say_total_0():
    board = _postings("a", 45) + [{"title": "Video Producer", "externalPath": "/job/vp",
                                   "locationsText": "Seattle, WA", "postedOn": "Posted Today"}]
    rows, calls = _run({"producer": board})
    assert calls["producer"] == 3  # 46 postings = 3 pages of 20
    assert [r["title"] for r in rows] == ["Video Producer"]  # on the last page


def test_a_seed_returning_the_whole_board_again_stops_after_one_page():
    # Some boards answer "producer" and "production" both with every posting
    # on the board, in the same order. 1,500 is past the 10-page cap, so
    # "producer" never reaches the list's end; "production" must still stop
    # after its first page.
    board = _postings("s", 1500)
    rows, calls = _run({"producer": board, "production": board, "content": _postings("c", 5)},
                       max_pages=wd._MAX_PAGES)
    assert calls["producer"] == 10
    assert calls["production"] == 1
    assert calls["content"] == 1


def test_once_a_board_repeats_a_list_a_shuffled_repeat_stops_after_its_first_page():
    board = _postings("s", 100)
    shuffled = list(reversed(board))
    shuffled[0] = {"title": "Video Producer", "externalPath": "/job/new",
                   "locationsText": "Seattle, WA", "postedOn": "Posted Today"}
    rows, calls = _run({"producer": board, "production": board, "content": shuffled})
    assert calls["production"] == 1
    assert calls["content"] == 1  # same total as the repeated list: rule 2
    assert [r["title"] for r in rows] == ["Video Producer"]  # its first page still counts


def test_a_different_total_never_stops_early_even_if_page_one_is_all_repeats():
    # A real search: its first page holds postings an earlier seed found,
    # its later pages hold new ones.
    first = _postings("p", 40)
    second = first[:20] + _postings("n", 40)
    second.append({"title": "Video Producer", "externalPath": "/job/vp",
                   "locationsText": "Seattle, WA", "postedOn": "Posted Today"})
    rows, calls = _run({"producer": first, "production": second})
    assert calls["production"] == 4  # 61 postings = 4 pages
    assert [r["title"] for r in rows] == ["Video Producer"]


def test_same_total_but_new_postings_keeps_paging():
    # Two real searches that happen to have the same size.
    rows, calls = _run({"producer": _postings("p", 60), "production": _postings("q", 60)})
    assert calls["producer"] == 3
    assert calls["production"] == 3


def test_page_cap_still_applies():
    rows, calls = _run({"producer": _postings("s", 200)}, max_pages=4)
    assert calls["producer"] == 4


def test_default_page_cap_is_10_per_seed():
    # 25 pages a seed let fuzzy seeds on big boards page ~100 times a board.
    assert wd._MAX_PAGES == 10
    rows, calls = _run({"producer": _postings("s", 400)}, max_pages=wd._MAX_PAGES)
    assert calls["producer"] == 10


def test_a_seed_at_workdays_2000_cap_is_too_broad_and_stops_after_4_pages():
    big = _postings("b", 2000)
    rows, calls = _run({"producer": big, "content": _postings("c", 150)})
    assert calls["producer"] == wd._BROAD_SEED_PAGES == 4
    assert calls["content"] == 8  # a real search keeps paging to its end


def test_board_time_budget_stops_the_board_between_pages(caplog):
    # Each list request "takes" 40 s: the guard stops the board once 150 s
    # are used, mid-seed, and skips the later seeds.
    clock = {"t": 1000.0}

    def monotonic():
        return clock["t"]

    lists = {"producer": _postings("p", 200), "content": _postings("c", 200)}
    calls: dict = {}

    def post(url, json, timeout):
        clock["t"] += 40
        seed, offset, limit = json["searchText"], json["offset"], json["limit"]
        calls[seed] = calls.get(seed, 0) + 1
        found = lists.get(seed, [])
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"jobPostings": found[offset:offset + limit],
                                  "total": len(found) if offset == 0 else 0}
        return resp

    with patch("job_agent.scraper_workday.requests.Session") as cls, \
            patch("job_agent.scraper_workday.time.sleep"), \
            patch("job_agent.scraper_workday.time.monotonic", side_effect=monotonic):
        sess = MagicMock()
        sess.post.side_effect = post
        cls.return_value = sess
        with caplog.at_level("WARNING", logger="job_agent.scraper_workday"):
            wd.scrape_workday(BASE, "Acme", KEYWORDS, fetch_details=False)

    assert calls == {"producer": 4}  # 4 x 40 s = 160 s >= 150 s
    assert "board time budget" in caplog.text


# --- Rule 1 after a list read to its end ---

def test_rule_1_still_applies_after_a_list_read_to_its_end():
    board = _postings("s", 60)
    rows, calls = _run({"producer": board, "production": board})
    assert calls["producer"] == 3
    assert calls["production"] == 1


# --- 429 on a list page: wait Retry-After, retry the page once ---

def _http_error(status, headers=None):
    import requests as _requests
    return _requests.exceptions.HTTPError(
        f"{status} Client Error", response=MagicMock(status_code=status, headers=headers or {}))


def _run_with_errors(lists: dict, errors: dict):
    """Like _run, but errors = {(seed, offset): [status, ...]} answers those
    requests with HTTP errors first, in order. Returns (rows, calls, sleeps)."""
    calls: dict = {}
    pending = {k: list(v) for k, v in errors.items()}

    def post(url, json, timeout):
        seed, offset, limit = json["searchText"], json["offset"], json["limit"]
        calls[seed] = calls.get(seed, 0) + 1
        resp = MagicMock()
        queued = pending.get((seed, offset))
        if queued:
            status, headers = queued.pop(0)
            resp.raise_for_status.side_effect = _http_error(status, headers)
            return resp
        found = lists.get(seed, [])
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"jobPostings": found[offset:offset + limit],
                                  "total": len(found) if offset == 0 else 0}
        return resp

    with patch("job_agent.scraper_workday.requests.Session") as cls, \
            patch("job_agent.scraper_workday.time.sleep") as sleep:
        sess = MagicMock()
        sess.post.side_effect = post
        cls.return_value = sess
        rows = wd.scrape_workday(BASE, "Acme", KEYWORDS, fetch_details=False)
    return rows, calls, [c.args[0] for c in sleep.call_args_list]


def test_a_429_waits_retry_after_and_retries_the_same_page():
    board = _postings("a", 45) + [{"title": "Video Producer", "externalPath": "/job/vp",
                                   "locationsText": "Seattle, WA", "postedOn": "Posted Today"}]
    rows, calls, sleeps = _run_with_errors(
        {"producer": board}, {("producer", 20): [(429, {"Retry-After": "7"})]})
    assert calls["producer"] == 4  # 3 pages + the one retry
    assert 7.0 in sleeps
    assert [r["title"] for r in rows] == ["Video Producer"]  # the seed was not dropped


def test_a_second_429_on_the_same_page_counts_as_the_error():
    rows, calls, sleeps = _run_with_errors(
        {"producer": _postings("a", 45), "content": _postings("c", 5)},
        {("producer", 20): [(429, {}), (429, {})]})
    assert calls["producer"] == 3  # page 1, the 429, one retry; then the seed ends
    assert calls["content"] == 1  # one error is not a dead token


def test_other_http_errors_are_not_retried():
    rows, calls, sleeps = _run_with_errors(
        {"producer": _postings("a", 45)}, {("producer", 0): [(503, {"Retry-After": "2"})]})
    assert calls["producer"] == 1
    assert 2.0 not in sleeps


@pytest.mark.parametrize("headers,wait", [
    ({}, 4), ({"Retry-After": "abc"}, 4), ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, 4),
    ({"Retry-After": "-3"}, 4), ({"Retry-After": "nan"}, 4),
    ({"Retry-After": "120"}, 15), ({"Retry-After": "0"}, 0), ({"Retry-After": "2.5"}, 2.5),
])
def test_retry_after_default_and_cap(headers, wait):
    assert wd._retry_after_seconds(MagicMock(headers=headers)) == wait


# --- The phase timeout is not a failed description ---

class _ScrapeTimeout(Exception):
    """Stands in for main._ScrapeTimeout (matched by name, as in ats_scraper)."""


def test_phase_timeout_in_a_detail_fetch_ends_the_scan():
    sess = MagicMock()
    sess.get.side_effect = _ScrapeTimeout("ATS phase (Workday boards) timed out")
    with pytest.raises(_ScrapeTimeout):
        wd._fetch_job_detail("https://acme.wd5.myworkdayjobs.com", "acme", "External",
                             "/job/1", sess)


def test_other_detail_fetch_errors_still_give_an_empty_detail():
    sess = MagicMock()
    sess.get.side_effect = RuntimeError("boom")
    assert wd._fetch_job_detail("https://acme.wd5.myworkdayjobs.com", "acme", "External",
                                "/job/1", sess) == {}
