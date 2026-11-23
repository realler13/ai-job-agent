"""Tests for the ZipRecruiter scraper (Scrapling solve_cloudflare + parsing)."""
import pandas as pd

import job_agent.ziprecruiter_scraper as zr
from tests.fixtures.persona import HOME


# Minimal fixture mirroring the OLD live DOM (before the redesign):
# article[id^='job-card'] with an h2 title and data-testid company/location
# anchors. The first listing is doubled (same id) to exercise dedup.
FIXTURE_HTML = f"""
<html><body>
  <article id="job-card-AAA111">
    <h2>Senior Video Producer</h2>
    <a data-testid="job-card-company" href="/co/Acme">Acme Studios</a>
    <a data-testid="job-card-location" href="/x">{HOME}</a>
  </article>
  <article id="job-card-AAA111">
    <h2>Senior Video Producer</h2>
    <a data-testid="job-card-company">Acme Studios</a>
    <a data-testid="job-card-location">{HOME}</a>
  </article>
  <article id="job-card-BBB222">
    <h2>Podcast Producer</h2>
    <a data-testid="job-card-company">MediaCo</a>
    <a data-testid="job-card-location">Remote</a>
  </article>
</body></html>
"""


class _FakePage:
    def __init__(self, body):
        self.body = body
        self.status = 200


def test_parses_and_dedupes(monkeypatch):
    monkeypatch.setattr(
        zr.StealthyFetcher, "fetch",
        staticmethod(lambda *a, **k: _FakePage(FIXTURE_HTML)),
    )
    df = zr.scrape_ziprecruiter("video producer", HOME, results_wanted=20)
    assert df is not None
    # 3 article cards, but the doubled AAA111 dedups -> 2 unique
    assert len(df) == 2
    titles = set(df["title"])
    assert titles == {"Senior Video Producer", "Podcast Producer"}
    row = df[df["title"] == "Senior Video Producer"].iloc[0]
    assert row["company"] == "Acme Studios"
    assert row["location"] == HOME
    assert row["job_url"] == "https://www.ziprecruiter.com/jobs//j?lvk=AAA111"
    assert row["site"] == "zip_recruiter"


def test_respects_results_wanted(monkeypatch):
    monkeypatch.setattr(
        zr.StealthyFetcher, "fetch",
        staticmethod(lambda *a, **k: _FakePage(FIXTURE_HTML)),
    )
    df = zr.scrape_ziprecruiter("producer", results_wanted=1)
    assert len(df) == 1


def test_blocked_page_returns_none(monkeypatch):
    # Block markers / tiny shell -> None (counts toward the circuit breaker).
    monkeypatch.setattr(
        zr.StealthyFetcher, "fetch",
        staticmethod(lambda *a, **k: _FakePage("<html><body>Access denied</body></html>")),
    )
    assert zr.scrape_ziprecruiter("producer") is None


def test_rendered_empty_page_returns_empty_df(monkeypatch):
    # A real-size rendered page with zero cards = legitimate empty result ->
    # empty DataFrame (resets the breaker), NOT None.
    big_empty = "<html><body>" + "<div>filler</div>" * 4000 + "</body></html>"
    monkeypatch.setattr(
        zr.StealthyFetcher, "fetch",
        staticmethod(lambda *a, **k: _FakePage(big_empty)),
    )
    df = zr.scrape_ziprecruiter("producer")
    assert df is not None and df.empty


def test_fetch_exception_returns_none(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("browser crashed")
    monkeypatch.setattr(zr.StealthyFetcher, "fetch", staticmethod(boom))
    assert zr.scrape_ziprecruiter("producer") is None


# The NEW layout, in a live page's shape (made-up content): the key lives in
# the article's data-testid, the company is a /co/ link, the location a
# /jobs-search link. The Next.js payload carries "forbidden" as an ordinary key —
# a block check over the raw HTML reads that as a block page and empties every
# search. Padding makes it real-page sized (> 50 KB) like the live page.
_NEXT_PAYLOAD = (
    '<script>self.__next_f.push([1,"{\\"notFound\\":\\"$undefined\\",'
    '\\"forbidden\\":\\"$undefined\\"}"])</script>'
)
_PAD = "<div>filler</div>" * 4000


def _new_card(key, title, company, location):
    return f"""
  <li role="listitem" data-testid="job-card-item-{key}">
    <article data-testid="job-card-{key}" class="x1ghz6dp x1717udv">
      <button type="button" aria-label="View {title}" data-testid="job-card-overlay"></button>
      <a href="/co/{company}/Jobs/-in-Seattle,WA?uuid=&amp;radius=25"><span>{company}</span></a>
      <button type="button" data-testid="save-job-card-button" title="Save job for later"></button>
      <h2 data-testid="serp-job-card-title" aria-label="{title}">
        <button type="button" data-testid="job-card-activation"><span>{title}</span></button>
      </h2>
      <a href="/jobs-search?location=Seattle%2CWA&amp;search=Video+Producer" rel="nofollow"><span>{location}</span></a>
      <span>$70K/yr</span>
    </article>
  </li>"""


NEW_LAYOUT_HTML = (
    f"<html><head><title>187 Video Producer Jobs (NOW HIRING) in {HOME}</title>"
    + _NEXT_PAYLOAD + "</head><body><ul>"
    + _new_card("KeyAaaa1111Bbbb2222Cc", "Founding Video Producer", "Brightkettle Studios", HOME)
    + _new_card("KeyAaaa1111Bbbb2222Cc", "Founding Video Producer", "Brightkettle Studios", HOME)
    + _new_card("KeyDddd3333Eeee4444Ff", "Senior Video Producer", "Owlbridge Media", "Bellevue, WA")
    + "</ul>" + _PAD + "</body></html>"
)


def _serve(monkeypatch, html):
    monkeypatch.setattr(
        zr.StealthyFetcher, "fetch",
        staticmethod(lambda *a, **k: _FakePage(html)),
    )


def test_new_layout_parses_and_dedupes(monkeypatch):
    _serve(monkeypatch, NEW_LAYOUT_HTML)
    df = zr.scrape_ziprecruiter("video producer", HOME, results_wanted=20)
    assert df is not None
    assert list(df["title"]) == ["Founding Video Producer", "Senior Video Producer"]
    assert list(df["company"]) == ["Brightkettle Studios", "Owlbridge Media"]
    assert list(df["location"]) == [HOME, "Bellevue, WA"]
    assert list(df["job_url"]) == [
        "https://www.ziprecruiter.com/jobs//j?lvk=KeyAaaa1111Bbbb2222Cc",
        "https://www.ziprecruiter.com/jobs//j?lvk=KeyDddd3333Eeee4444Ff",
    ]
    # Same output shape the pipeline has always saved
    assert list(df.columns) == [
        "site", "title", "company", "location", "job_url", "description", "date_posted",
    ]


def test_forbidden_in_script_payload_is_not_a_block(monkeypatch):
    # A real-size page, no cards, no hit count in the title, "forbidden" only in
    # the script payload: a legitimate empty search, not a block.
    html = ("<html><head><title>Jobs in Nowhere</title>" + _NEXT_PAYLOAD
            + "</head><body>" + _PAD + "</body></html>")
    _serve(monkeypatch, html)
    df = zr.scrape_ziprecruiter("producer")
    assert df is not None and df.empty


def test_visible_challenge_text_on_big_page_is_a_block(monkeypatch):
    # A challenge page padded past the size check still reads as blocked,
    # because the marker is in what the visitor sees.
    html = ("<html><head><title>Just a moment...</title></head><body>"
            + _PAD + "</body></html>")
    _serve(monkeypatch, html)
    assert zr.scrape_ziprecruiter("producer") is None


def test_jobs_in_title_but_no_cards_flags_layout_change(monkeypatch, caplog):
    # The page says it has 187 jobs but no card matches our selector: the layout
    # changed again. Report it as a failure (None) with a plain warning, never a
    # quiet "0 jobs" that resets the breaker forever.
    html = NEW_LAYOUT_HTML.replace("<article", "<section").replace("</article>", "</section>")
    _serve(monkeypatch, html)
    with caplog.at_level("WARNING", logger=zr.logger.name):
        assert zr.scrape_ziprecruiter("producer") is None
    assert "layout changed" in caplog.text


# --- posting dates from the page payload --------------------------------------
# One JobCard object per card in the Next.js payload, quotes escaped as on the
# live page: status.postedAtUtc in Unix seconds ("$n" big-number
# form) and the relative rollingPostedAt.
def _payload_card(key, posted=None, rolling_text=None):
    status = ""
    if posted is not None:
        status = (',\\"status\\":{\\"$typeName\\":\\"jobs.service.public.job.v1.ListingStatus\\",'
                  '\\"isActive\\":true,\\"postedAtUtc\\":{\\"$typeName\\":\\"google.protobuf.Timestamp\\",'
                  f'\\"seconds\\":\\"$n{posted}\\",\\"nanos\\":0}},'
                  '\\"rollingPostedAtUtc\\":{\\"$typeName\\":\\"google.protobuf.Timestamp\\",'
                  f'\\"seconds\\":\\"$n{posted + 30 * 86400}\\",\\"nanos\\":0}}}}')
    rolling = f',\\"rollingPostedAt\\":\\"{rolling_text}\\"' if rolling_text else ""
    return ('{\\"$typeName\\":\\"job_services.job_card.public.v1.JobCard\\",\\"matchId\\":\\"m\\",'
            f'\\"listingKey\\":\\"{key}\\",\\"title\\":\\"T\\"{status}{rolling}}}')


def _page_with_payload(cards, payload_cards):
    payload = ('<script>self.__next_f.push([1,"{\\"jobCards\\":['
               + ",".join(payload_cards) + ']}"])</script>')
    return ("<html><head><title>3 Video Producer Jobs</title>" + payload + "</head><body><ul>"
            + "".join(_new_card(k, t, "Co", HOME) for k, t in cards)
            + "</ul>" + _PAD + "</body></html>")


def test_date_posted_from_payload_original_date_wins(monkeypatch):
    from datetime import datetime, timedelta
    posted = 1780000000  # months back; the payload rolls it forward ("22 days ago")
    html = _page_with_payload(
        [("KEYAAA", "Video Producer"), ("KEYBBB", "Podcast Producer"), ("KEYCCC", "Editor Producer")],
        [_payload_card("KEYAAA", posted=posted, rolling_text="22 days ago"),
         _payload_card("KEYBBB", rolling_text="3 days ago")],
    )
    _serve(monkeypatch, html)
    df = zr.scrape_ziprecruiter("video producer", results_wanted=20)
    dates = dict(zip(df["title"], df["date_posted"]))
    # postedAtUtc beats the rolled-forward relative age
    assert dates["Video Producer"] == datetime.fromtimestamp(posted).strftime("%Y-%m-%d")
    # no postedAtUtc: the relative age is still better than nothing
    assert dates["Podcast Producer"] == (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    # a card the payload does not describe keeps the old "no date"
    assert dates["Editor Producer"] is None


def test_ago_to_date_forms():
    from datetime import datetime, timedelta
    today = datetime.now()
    assert zr._ago_to_date("today") == today.strftime("%Y-%m-%d")
    assert zr._ago_to_date("1 day ago") == (today - timedelta(days=1)).strftime("%Y-%m-%d")
    assert zr._ago_to_date("30+ days ago") == (today - timedelta(days=30)).strftime("%Y-%m-%d")
    assert zr._ago_to_date("") == ""
    assert zr._ago_to_date("recently") == ""
