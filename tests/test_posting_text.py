"""posting_text: the posting text of jobs whose source lists them without it."""
import ipaddress
import logging
import socket

import pandas as pd
import pytest

from job_agent import posting_text as pt
from tests.fixtures import ssrf_addresses as ADDRESSES

LONG = ("We are hiring a producer to run our weekly video series from pitch to delivery. "
        "You will own budgets, schedules and the edit, and work with hosts and guests. ") * 4


@pytest.fixture
def _no_real_posting_page_fetches():
    """This module tests the real page reader (against mocked responses)."""
    yield


class _Resp:
    def __init__(self, status=200, text="", headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}




@pytest.fixture(autouse=True)
def _dns(monkeypatch):
    """No real DNS in tests: every host is a public address unless listed."""
    def fake(host, port, *args, **kwargs):
        try:
            ip = str(ipaddress.ip_address(host))
        except ValueError:
            ip = ADDRESSES.BY_HOST.get(host, ADDRESSES.PUBLIC)
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port))]
    monkeypatch.setattr(pt.socket, "getaddrinfo", fake)


def _jsonld_page(description):
    body = ('{"@context": "https://schema.org", "@type": "JobPosting", "title": "Producer", '
            f'"description": "{description}"}}')
    return f'<html><head><script type="application/ld+json">{body}</script></head><body><nav>menu</nav></body></html>'


# ── html_to_text ──

def test_html_to_text_puts_blocks_on_lines_and_marks_list_items():
    out = pt.html_to_text("<h3>About</h3><p>We make <strong>shows</strong>.</p><ul><li>Budgets</li><li>Schedules</li></ul>")
    assert out.splitlines() == ["About", "We make shows.", "- Budgets", "- Schedules"]


def test_html_to_text_decodes_entity_escaped_html():
    assert pt.html_to_text("&lt;p&gt;Own the edit.&lt;/p&gt;&lt;p&gt;Ship weekly.&lt;/p&gt;").splitlines() == [
        "Own the edit.", "Ship weekly."]


# ── text_from_html ──

def test_jsonld_job_posting_description_wins():
    page = _jsonld_page("<p>" + LONG + "</p>")
    assert pt.text_from_html(page).startswith("We are hiring a producer")


def test_jsonld_inside_a_graph_is_found():
    page = ('<script type="application/ld+json">{"@graph": [{"@type": "Organization", "name": "X"}, '
            '{"@type": "JobPosting", "description": "<p>Run the series.</p><p>' + LONG + '</p>"}]}</script>')
    assert pt.text_from_html(page).startswith("Run the series.")


def test_without_jsonld_the_largest_block_of_paragraphs_is_used():
    page = ("<html><body><nav><p>Home</p><p>Jobs</p><p>About</p></nav>"
            "<div class='job'><p>Run the series.</p><p>Own the budget.</p><ul><li>Hosts</li></ul></div>"
            "<footer><p>a</p><p>b</p><p>c</p></footer></body></html>")
    assert pt.text_from_html(page).splitlines()[:2] == ["Run the series.", "Own the budget."]


def test_a_page_with_no_posting_gives_nothing():
    assert pt.text_from_html("<html><body><p>Sign in</p></body></html>") == ""
    assert pt.text_from_html("") == ""


# ── ZipRecruiter's React Server Components payload ──

def _js(s):
    return '"' + (s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                  .replace("<", "\\u003c").replace(">", "\\u003e")) + '"'


def _rsc_page(stream_parts):
    scripts = "".join(f"<script>self.__next_f.push([1,{_js(part)}])</script>" for part in stream_parts)
    return f"<html><body><article class='job-card'></article>{scripts}</body></html>"


def test_ziprecruiter_text_follows_the_reference_to_a_text_chunk_split_across_scripts():
    html = "<h3>Job Description</h3><p>Café crew, édit bay, " + LONG + "</p>"
    size = format(len(html.encode("utf-8")), "x")
    stream = '1:["$","div",null,{}]\n2:{"job":{"htmlFullDescription":"$4a","title":"Producer"}}\n4a:T' + size + "," + html + "5:[]\n"
    cut = stream.index("édit") + 2              # split inside the chunk, mid-word
    out = pt.ziprecruiter_text(_rsc_page([stream[:cut], stream[cut:]]))
    assert out.startswith("Job Description\nCafé crew, édit bay, We are hiring")
    assert out.endswith("guests.")                # the byte length stops before "5:[]"


def test_ziprecruiter_text_reads_an_inline_description():
    stream = '2:{"htmlFullDescription":"\\u003cp\\u003eRun the series.\\u003c/p\\u003e"}\n'
    assert pt.ziprecruiter_text(_rsc_page([stream])) == "Run the series."


def test_ziprecruiter_text_without_a_description_is_empty():
    assert pt.ziprecruiter_text(_rsc_page(['2:{"title":"Producer"}\n'])) == ""
    assert pt.ziprecruiter_text("<html></html>") == ""


# ── fetch_posting_text ──

def test_fetch_reads_the_posting_page(monkeypatch):
    monkeypatch.setattr(pt.requests, "get", lambda url, headers, timeout, **kw: _Resp(200, _jsonld_page(LONG)))
    assert pt.fetch_posting_text("https://www.showbizjobs.com/jobs/x").startswith("We are hiring")


@pytest.mark.parametrize("resp", [_Resp(403, "Forbidden"), _Resp(200, "<title>Just a moment...</title>")])
def test_a_blocked_page_gives_nothing(monkeypatch, resp):
    monkeypatch.setattr(pt.requests, "get", lambda url, headers, timeout, **kw: resp)
    assert pt.fetch_posting_text("https://www.glassdoor.com/job-listing/x") == ""


def test_a_failed_request_gives_nothing(monkeypatch):
    def boom(url, headers, timeout, **kw):
        raise pt.requests.ConnectionError("down")
    monkeypatch.setattr(pt.requests, "get", boom)
    assert pt.fetch_posting_text("https://podnews.net/jobs/x") == ""
    assert pt.fetch_posting_text("") == ""


def test_the_phase_timeout_is_not_swallowed(monkeypatch):
    class _ScrapeTimeout(Exception):
        pass
    def alarm(url, headers, timeout, **kw):
        raise _ScrapeTimeout()
    monkeypatch.setattr(pt.requests, "get", alarm)
    with pytest.raises(_ScrapeTimeout):
        pt.fetch_posting_text("https://podnews.net/jobs/x")


@pytest.mark.parametrize("link", [
    "http://127.0.0.1:8502/api/config", "http://localhost:3000/", "http://[::1]:8503/kanban",
    "http://intranet.example/jobs/1", "http://metadata.example/latest/meta-data/",
    "http://mapped.example/x", "http://overlay.example/", "http://0.0.0.0/", "file:///etc/hosts", "ftp://jobs.example/x",
    "javascript:alert(1)", "https://", "not a url"])
def test_links_that_are_not_public_web_addresses_are_never_fetched(monkeypatch, link):
    monkeypatch.setattr(pt.requests, "get", lambda *a, **k: pytest.fail(f"fetched {link}"))
    assert pt.is_public_web_url(link) is False
    assert pt.fetch_posting_text(link) == ""


def test_a_redirect_to_a_private_address_is_refused(monkeypatch):
    calls = []
    def get(url, headers, timeout, **kw):
        calls.append((url, kw.get("allow_redirects")))
        return _Resp(302, "", {"Location": "http://127.0.0.1:8502/api/applications"})
    monkeypatch.setattr(pt.requests, "get", get)
    assert pt.fetch_posting_text("https://podnews.net/jobs/x") == ""
    assert calls == [("https://podnews.net/jobs/x", False)]


def test_public_redirects_are_followed_hop_by_hop(monkeypatch):
    pages = {"https://podnews.net/jobs/x": _Resp(301, "", {"Location": "/jobs/x-producer"}),
             "https://podnews.net/jobs/x-producer": _Resp(200, _jsonld_page(LONG))}
    monkeypatch.setattr(pt.requests, "get", lambda url, headers, timeout, **kw: pages[url])
    assert pt.fetch_posting_text("https://podnews.net/jobs/x").startswith("We are hiring")


def test_a_redirect_loop_stops(monkeypatch):
    monkeypatch.setattr(pt.requests, "get",
                        lambda url, headers, timeout, **kw: _Resp(302, "", {"Location": url + "x"}))
    assert pt.fetch_posting_text("https://podnews.net/jobs/x") == ""


def test_only_ziprecruiter_itself_goes_to_the_stealth_browser(monkeypatch):
    monkeypatch.setattr(pt, "_ziprecruiter_page", lambda link: pytest.fail("not ZipRecruiter"))
    monkeypatch.setattr(pt.requests, "get", lambda url, headers, timeout, **kw: _Resp(404, ""))
    assert pt.fetch_posting_text("https://attacker-ziprecruiter.com/jobs/1") == ""


def test_ziprecruiter_goes_through_its_stealth_browser(monkeypatch):
    seen = []
    monkeypatch.setattr(pt, "_ziprecruiter_page", lambda link: seen.append(link) or LONG)
    monkeypatch.setattr(pt.requests, "get", lambda *a, **k: pytest.fail("no plain request to ZipRecruiter"))
    assert pt.fetch_posting_text("https://www.ziprecruiter.com/jobs//j?lvk=abc") == LONG
    assert seen == ["https://www.ziprecruiter.com/jobs//j?lvk=abc"]


# ── find_text / fill_posting_text with the real database ──

def _save(db, link, title="Producer", company="Acme Studios", description=""):
    return db.save_job({"title": title, "company": company, "location": "Remote", "job_url": link,
                        "site": "test", "description": description}, session_id="s1")


def test_another_copy_of_the_job_supplies_the_text_without_a_request(tmp_db, monkeypatch):
    monkeypatch.setattr(pt, "fetch_posting_text", lambda link: pytest.fail("no page read needed"))
    _save(tmp_db, "https://www.linkedin.com/jobs/view/1", description=LONG)
    thin_id = _save(tmp_db, "https://www.glassdoor.com/job-listing/1")
    job = tmp_db.get_job_details(thin_id)
    assert pt.fill_posting_text(job, tmp_db) == "copy"
    assert job["description"] == LONG
    assert tmp_db.get_job_details(thin_id)["description"] == LONG
    assert tmp_db.get_job_details(thin_id)["description_source"] == "posting_text:copy"


def test_a_copy_is_matched_on_title_and_company_and_never_on_itself(tmp_db, monkeypatch):
    monkeypatch.setattr(pt, "fetch_posting_text", lambda link: "")
    _save(tmp_db, "https://a.example/1", title="Producer", company="Other Co", description=LONG)
    _save(tmp_db, "https://a.example/2", title="Senior Producer", company="Acme Studios", description=LONG)
    _save(tmp_db, "https://a.example/3", title="Producer", company="Acme Studios", description="short")
    thin_id = _save(tmp_db, "https://a.example/4")
    job = tmp_db.get_job_details(thin_id)
    assert pt.find_text(job, tmp_db) == ("", "")


def test_the_posting_page_is_read_when_there_is_no_copy(tmp_db, monkeypatch):
    monkeypatch.setattr(pt, "fetch_posting_text", lambda link: LONG if "podnews" in link else "")
    thin_id = _save(tmp_db, "https://podnews.net/jobs/producer-1")
    job = tmp_db.get_job_details(thin_id)
    assert pt.fill_posting_text(job, tmp_db) == "page"
    assert tmp_db.get_job_details(thin_id)["description"] == LONG


def test_a_job_with_text_is_left_alone(tmp_db, monkeypatch):
    monkeypatch.setattr(pt, "fetch_posting_text", lambda link: pytest.fail("no lookup for a job with text"))
    job = {"id": 1, "title": "Producer", "company": "Acme", "link": "https://x", "description": LONG}
    assert pt.fill_posting_text(job, tmp_db) == ""


def test_nothing_found_changes_nothing(tmp_db, monkeypatch, caplog):
    monkeypatch.setattr(pt, "fetch_posting_text", lambda link: "")
    thin_id = _save(tmp_db, "https://www.glassdoor.com/job-listing/2")
    job = tmp_db.get_job_details(thin_id)
    with caplog.at_level(logging.INFO):
        assert pt.fill_posting_text(job, tmp_db) == ""
    assert "none found" in caplog.text
    assert not tmp_db.get_job_details(thin_id).get("description")


def test_set_posting_text_keeps_the_rest_of_the_row_and_finds_it_by_link(tmp_db):
    job_id = _save(tmp_db, "https://a.example/9")
    assert tmp_db.set_posting_text(LONG, source="posting_text:page", link="https://a.example/9")
    row = tmp_db.get_job_details(job_id)
    assert (row["description"], row["title"], row["site"]) == (LONG, "Producer", "test")
    assert tmp_db.set_posting_text(LONG, link="https://a.example/none") is False


# ── fill_tiered_jobs (the pipeline step) ──

def test_fill_tiered_jobs_fills_only_the_rows_without_text(tmp_db, monkeypatch, caplog):
    pages = {"https://podnews.net/jobs/a": LONG}
    monkeypatch.setattr(pt, "fetch_posting_text", lambda link: pages.get(link, ""))
    for link in ("https://podnews.net/jobs/a", "https://www.glassdoor.com/job-listing/b", "https://x.example/c"):
        _save(tmp_db, link, title=link[-1])
    df = pd.DataFrame([
        {"title": "a", "company": "Acme Studios", "job_url": "https://podnews.net/jobs/a", "description": ""},
        {"title": "b", "company": "Acme Studios", "job_url": "https://www.glassdoor.com/job-listing/b", "description": float("nan")},
        {"title": "c", "company": "Acme Studios", "job_url": "https://x.example/c", "description": LONG},
    ])
    with caplog.at_level(logging.INFO):
        assert pt.fill_tiered_jobs(df, tmp_db, logging.getLogger("t")) == 1
    assert df.loc[0, "description"] == LONG and df.loc[2, "description"] == LONG
    assert tmp_db.get_job_details(tmp_db.get_job_id_by_link("https://podnews.net/jobs/a"))["description"] == LONG
    assert "1 of 2 tiered jobs without text got it (podnews.net 1/1, www.glassdoor.com 0/1)" in caplog.text


def test_a_scraper_id_in_the_dataframe_does_not_stop_the_text_being_saved(tmp_db, monkeypatch):
    monkeypatch.setattr(pt, "fetch_posting_text", lambda link: LONG)
    _save(tmp_db, "https://www.linkedin.com/jobs/view/42", title="x")
    df = pd.DataFrame([{"id": "li-42", "title": "x", "company": "Acme Studios",
                        "job_url": "https://www.linkedin.com/jobs/view/42", "description": ""}])
    assert pt.fill_tiered_jobs(df, tmp_db) == 1
    row_id = tmp_db.get_job_id_by_link("https://www.linkedin.com/jobs/view/42")
    assert tmp_db.get_job_details(row_id)["description"] == LONG


def test_a_ziprecruiter_page_that_never_finishes_is_abandoned(monkeypatch):
    import threading
    from job_agent import ziprecruiter_scraper
    release = threading.Event()
    monkeypatch.setattr(ziprecruiter_scraper, "_fetch_html", lambda link, attempts=2: release.wait(5) and "")
    monkeypatch.setattr(pt, "BROWSER_TIMEOUT_S", 0.2)
    try:
        assert pt._ziprecruiter_page("https://www.ziprecruiter.com/jobs//j?lvk=x") == ""
    finally:
        release.set()


def test_fill_tiered_jobs_respects_its_cap_and_stop(tmp_db, monkeypatch):
    calls = []
    monkeypatch.setattr(pt, "fetch_posting_text", lambda link: calls.append(link) or "")
    df = pd.DataFrame([{"title": f"t{i}", "company": "Acme", "job_url": f"https://x.example/{i}", "description": ""}
                       for i in range(5)])
    pt.fill_tiered_jobs(df, tmp_db, cap=2)
    assert len(calls) == 2
    calls.clear()
    pt.fill_tiered_jobs(df, tmp_db, should_stop=lambda: True)
    assert calls == []
    assert pt.fill_tiered_jobs(pd.DataFrame(), tmp_db) == 0


# ── wiring ──

def test_generate_package_looks_up_the_text_before_writing(monkeypatch):
    from job_agent import docs_builder
    monkeypatch.setattr(pt, "fetch_posting_text", lambda link: LONG)
    seen = {}

    def fake_save(base_cv_path, profile_text, job, **kw):
        seen["description"] = job.get("description")
        raise RuntimeError("stop here")
    monkeypatch.setattr(docs_builder, "save_documents_for_job", fake_save)
    out = docs_builder.generate_package({"id": 7, "title": "Producer", "company": "Acme",
                                         "link": "https://podnews.net/jobs/p", "description": ""},
                                        None, "profile", source="api")
    assert seen["description"] == LONG
    assert out["status"] == "failed"


def test_the_pipeline_fills_text_after_tiering_and_before_documents():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "job_agent" / "main.py").read_text()
    found = src.index('progress_callback(f"Found {len(relevant_df)} relevant jobs!"')
    fill = src.index("fill_tiered_jobs(", found)
    assert found < fill < src.index("_auto_enrich_matched_jobs(enrich_df", found) < src.index("generate_package(", found)
