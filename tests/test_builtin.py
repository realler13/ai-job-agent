"""Tests for scraper_builtin — Built In regional boards HTML scraper.

The sites come from the profile (search.builtin_sites) and the location hints
from its metro terms; the fixture profile is Seattle (tests/fixtures/README.md).
"""
import logging
from unittest.mock import MagicMock, patch

import pytest
import yaml
from bs4 import BeautifulSoup

from job_agent import scraper_builtin
from job_agent.scraper_builtin import (
    location_hints,
    regions_from_sites,
    scrape_builtin,
    _add_descriptions,
    _job_id,
    _parse_card,
    _parse_description,
    _parse_posting_age,
    _parse_posting_page,
    _scrape_region,
)
from tests.fixtures.persona import (
    BUILTIN_SITES,
    PRIMARY_METRO_TERMS,
    SECONDARY_METRO_TERMS,
    raw_profile,
)

SEATTLE = "https://www.builtinseattle.com"
NATIONAL = "https://builtin.com"

# Card structure as Built In serves it (company link, title link, age, location)
CARD_HTML = """
<div class="job-bounded-responsive" data-id="job-card" id="job-card-8882526">
  <div class="col-12 col-lg-7 left-side-tile">
    <div class="left-side-tile-item-2">
      <a data-id="company-title" href="/company/gullwing-pictures"><span>Gullwing Pictures</span></a>
    </div>
    <div class="left-side-tile-item-3">
      <h2>
        <a data-id="job-card-title" href="/job/associate-producer-social-weekends/8882526">
          Associate Producer, Social (Weekends)
        </a>
      </h2>
    </div>
  </div>
  <div class="col-12 col-lg-5">
    <span>Reposted 3 Hours Ago</span>
    <span>Bellevue, WA, Remote</span>
  </div>
</div>
"""

PAGE_WITH_TWO_CARDS = f"""
<html><body>
  <div class="results">
    {CARD_HTML}
    <div class="job-bounded-responsive" data-id="job-card" id="job-card-1111111">
      <div class="left-side-tile-item-2">
        <a data-id="company-title" href="/company/widgetco"><span>WidgetCo</span></a>
      </div>
      <div class="left-side-tile-item-3">
        <h2><a data-id="job-card-title" href="/job/backend-engineer/1111111">Backend Engineer</a></h2>
      </div>
      <span>Posted 2 Days Ago — Seattle</span>
    </div>
  </div>
</body></html>
"""


def _card(html):
    return BeautifulSoup(html, "html.parser").select_one('div[data-id="job-card"]')


def _card_with_location(location):
    return _card(
        '<div data-id="job-card"><a data-id="company-title" href="/c/x">WidgetCo</a>'
        '<h2><a data-id="job-card-title" href="/job/producer/7">Producer</a></h2>'
        f"<span>{location}</span></div>"
    )


def _profile_with(tmp_path, sites):
    from job_agent.candidate_profile import load_profile
    data = raw_profile()
    if sites is None:
        del data["search"]["builtin_sites"]
    else:
        data["search"]["builtin_sites"] = sites
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_profile(path)


# --- regions and hints come from the profile ------------------------------


class TestProfileDrivenRegions:
    def test_regions_are_the_fixture_sites(self):
        assert BUILTIN_SITES == (SEATTLE,)
        assert scraper_builtin.REGIONS == {"seattle": SEATTLE}

    def test_regions_from_sites_keys(self):
        assert regions_from_sites([SEATTLE, NATIONAL, "https://www.builtinnyc.com"]) == {
            "seattle": SEATTLE,
            "builtin": NATIONAL,
            "nyc": "https://www.builtinnyc.com",
        }
        assert regions_from_sites([]) == {}

    def test_profile_sites_are_cleaned(self, tmp_path):
        from job_agent.candidate_profile import builtin_sites
        profile = _profile_with(tmp_path, [SEATTLE + "/", "https://WWW.BuiltinSeattle.com",
                                           NATIONAL])
        assert builtin_sites(profile) == (SEATTLE, NATIONAL)

    def test_missing_key_means_no_sites(self, tmp_path):
        from job_agent.candidate_profile import builtin_sites
        assert builtin_sites(_profile_with(tmp_path, None)) == ()
        assert builtin_sites(_profile_with(tmp_path, [])) == ()

    @pytest.mark.parametrize("sites", [
        "https://www.builtinseattle.com",                 # a string, not a list
        {"seattle": SEATTLE},                              # a mapping
        ["http://www.builtinseattle.com"],                 # not https
        ["www.builtinseattle.com"],                        # no scheme
        ["https://www.example.com"],                       # not a Built In host
        ["https://builtinseattle.example.com"],            # builtin* label on another domain
        ["https://www.builtinseattle.com/jobs"],           # a path, not a base URL
        ["https://www.builtinseattle.com?search=producer"],  # a query
        ["https://www.builtinseattle.com:8443"],           # a port
        [SEATTLE, 42],                                     # one entry not a string
        [""],
    ])
    def test_bad_value_raises(self, tmp_path, sites):
        from job_agent.candidate_profile import builtin_sites
        with pytest.raises(ValueError, match="builtin_sites"):
            builtin_sites(_profile_with(tmp_path, sites))

    def test_hints_follow_the_profile_terms(self):
        expected_towns = []
        for term in PRIMARY_METRO_TERMS + SECONDARY_METRO_TERMS:
            town = term.split(",")[0].strip().title()
            if town not in expected_towns:
                expected_towns.append(town)
        assert scraper_builtin.LOCATION_HINTS == tuple(expected_towns) + ("Remote", "Hybrid")
        # the fixture's stated term "vancouver, wa" contributes its town
        assert "Vancouver" in scraper_builtin.LOCATION_HINTS
        assert not any("," in h for h in scraper_builtin.LOCATION_HINTS)

    def test_location_hints_order_and_dedupe(self):
        assert location_hints(["austin", "round rock", "dublin, oh"], ["austin", "san marcos"]) == (
            "Austin", "Round Rock", "Dublin", "San Marcos", "Remote", "Hybrid")
        assert location_hints([], []) == ("Remote", "Hybrid")
        assert location_hints(["remote", ""], None) == ("Remote", "Hybrid")


# --- _parse_posting_age ----------------------------------------------------


class TestParsePostingAge:
    def test_hours_ago_is_today(self):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        assert _parse_posting_age("Reposted 3 Hours Ago") == today

    def test_days_ago(self):
        from datetime import datetime, timedelta
        expected = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        assert _parse_posting_age("Posted 2 Days Ago") == expected

    def test_today_token(self):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        assert _parse_posting_age("Posted Today") == today

    def test_yesterday_token(self):
        from datetime import datetime, timedelta
        expected = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        assert _parse_posting_age("Posted Yesterday") == expected

    def test_empty(self):
        assert _parse_posting_age("") == ""
        assert _parse_posting_age("random text no date") == ""

    # The live card wording has no "Posted" prefix; a pattern that requires
    # it misses most rows.
    @pytest.mark.parametrize("text,days", [
        ("9 Days Ago", 9), ("An Hour Ago", 0), ("2 Hours Ago", 0), ("A Day Ago", 1),
        ("One Month Ago", 30), ("2 Months Ago", 60), ("Reposted 17 Days Ago", 17),
        ("Reposted An Hour Ago", 0),
    ])
    def test_prefixless_card_wording(self, text, days):
        from datetime import datetime, timedelta
        assert _parse_posting_age(text) == (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


# --- _parse_card -----------------------------------------------------------


class TestParseCard:
    def test_extracts_fields(self):
        job = _parse_card(_card(CARD_HTML), SEATTLE)

        assert job["title"] == "Associate Producer, Social (Weekends)"
        assert job["company"] == "Gullwing Pictures"
        assert job["job_url"] == f"{SEATTLE}/job/associate-producer-social-weekends/8882526"
        assert job["site"] == "builtin"
        # "Bellevue, WA, Remote": the metro town comes before Remote in the hints
        assert job["location"] == "Bellevue"

    def test_first_hint_wins(self):
        # Seattle is the profile's first primary term, so it beats Bellevue
        # wherever each appears in the card text.
        job = _parse_card(_card_with_location("Bellevue, WA or Seattle, WA"), SEATTLE)
        assert job["location"] == "Seattle"
        # a secondary-metro town, and a stated term's town
        assert _parse_card(_card_with_location("Portland, OR"), SEATTLE)["location"] == "Portland"
        assert _parse_card(_card_with_location("Vancouver, WA"), SEATTLE)["location"] == "Vancouver"
        assert _parse_card(_card_with_location("Remote"), SEATTLE)["location"] == "Remote"
        assert _parse_card(_card_with_location("Hybrid"), SEATTLE)["location"] == "Hybrid"
        # a town outside the profile's metros leaves the location empty
        assert _parse_card(_card_with_location("Austin, TX"), SEATTLE)["location"] == ""

    def test_hints_are_read_at_parse_time(self, monkeypatch):
        monkeypatch.setattr(scraper_builtin, "LOCATION_HINTS", ("Austin", "Remote", "Hybrid"))
        assert _parse_card(_card_with_location("Austin, TX"), SEATTLE)["location"] == "Austin"
        assert _parse_card(_card_with_location("Seattle, WA"), SEATTLE)["location"] == ""

    def test_age_comes_from_its_own_span_not_the_snippet(self):
        # Live card shape: the age has its own span; the description snippet
        # can say "today" or "3 days ago" and must not date the job.
        from datetime import datetime, timedelta
        html = """
        <div data-id="job-card">
          <a data-id="company-title" href="/company/gullwing-pictures">Gullwing Pictures</a>
          <h2><a data-id="job-card-title" href="/job/creative-producer/9000712">Creative Producer</a></h2>
          <span x-show="!showSavedTag" class="fs-xs fw-bold">9 Days Ago</span>
          <span>Hybrid</span><span>Seattle, WA, USA</span>
          <div>Join today and ship campaigns 3 days ago planned.</div>
        </div>"""
        card = BeautifulSoup(html, "html.parser").select_one('div[data-id="job-card"]')
        job = _parse_card(card, SEATTLE)
        assert job["date_posted"] == (datetime.now() - timedelta(days=9)).strftime("%Y-%m-%d")

    def test_snippet_word_alone_gives_no_date(self):
        html = """
        <div data-id="job-card">
          <a data-id="company-title" href="/company/x">X</a>
          <h2><a data-id="job-card-title" href="/job/producer/1">Producer</a></h2>
          <div>Apply today to join the team.</div>
        </div>"""
        card = BeautifulSoup(html, "html.parser").select_one('div[data-id="job-card"]')
        assert _parse_card(card, SEATTLE)["date_posted"] == ""

    def test_missing_title_returns_none(self):
        html = '<div data-id="job-card"><a data-id="company-title">X</a></div>'
        assert _parse_card(_card(html), SEATTLE) is None

    def test_missing_company_returns_none(self):
        html = '<div data-id="job-card"><h2><a data-id="job-card-title" href="/j/1">T</a></h2></div>'
        assert _parse_card(_card(html), SEATTLE) is None


# --- _scrape_region + scrape_builtin --------------------------------------


class TestScrapeRegion:
    @patch("job_agent.scraper_builtin.time.sleep", lambda _: None)
    @patch("job_agent.scraper_builtin._session")
    def test_filters_by_title_keyword(self, mock_session):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = PAGE_WITH_TWO_CARDS
        resp.raise_for_status = MagicMock()
        mock_session.get.return_value = resp

        results = _scrape_region("seattle", "producer", ["producer"], max_pages=1)
        # Only "Associate Producer, Social (Weekends)" matches 'producer'
        assert len(results) == 1
        assert "Producer" in results[0]["title"]
        assert results[0]["job_url"].startswith(SEATTLE + "/job/")
        url = mock_session.get.call_args[0][0]
        assert url == f"{SEATTLE}/jobs?search=producer&page=1"

    @patch("job_agent.scraper_builtin.time.sleep", lambda _: None)
    @patch("job_agent.scraper_builtin._session")
    def test_unknown_region_returns_empty(self, mock_session):
        # _scrape_region returns early for unknown regions without HTTP
        results = _scrape_region("nyc", "producer", ["producer"], max_pages=1)
        assert results == []
        mock_session.get.assert_not_called()

    @patch("job_agent.scraper_builtin.time.sleep", lambda _: None)
    @patch("job_agent.scraper_builtin._session")
    def test_fetch_error_breaks_early(self, mock_session):
        mock_session.get.side_effect = Exception("500")
        results = _scrape_region("seattle", "producer", ["producer"], max_pages=3)
        assert results == []
        assert mock_session.get.call_count == 1


class TestScrapeBuiltin:
    @patch("job_agent.scraper_builtin._add_descriptions")
    @patch("job_agent.scraper_builtin._scrape_region")
    def test_merges_two_regions(self, mock_scrape, _mock_desc, monkeypatch):
        monkeypatch.setattr(scraper_builtin, "REGIONS", regions_from_sites([SEATTLE, NATIONAL]))
        seattle_job = {"title": "Producer Seattle", "company": "A", "location": "Seattle",
                       "job_url": f"{SEATTLE}/job/a/1",
                       "description": "", "date_posted": "", "site": "builtin", "department": ""}
        national_job = {"title": "Producer Remote", "company": "B", "location": "Remote",
                        "job_url": f"{NATIONAL}/job/b/2",
                        "description": "", "date_posted": "", "site": "builtin", "department": ""}

        # scrape_builtin queries broad SEEDS (not raw keywords); "producer"
        # is the first seed, so key the mock on (region, seed="producer").
        call_map = {("seattle", "producer"): [seattle_job], ("builtin", "producer"): [national_job]}
        mock_scrape.side_effect = lambda region, seed, kws, **_: call_map.get((region, seed), [])

        jobs = scrape_builtin(["producer"])
        assert len(jobs) == 2
        assert {j["location"] for j in jobs} == {"Seattle", "Remote"}
        regions_called = {c.args[0] for c in mock_scrape.call_args_list}
        assert regions_called == {"seattle", "builtin"}
        assert mock_scrape.call_count == 2 * len(scraper_builtin.BUILTIN_SEARCH_SEEDS)

    @patch("job_agent.scraper_builtin._add_descriptions")
    @patch("job_agent.scraper_builtin._scrape_region")
    def test_fixture_profile_scrapes_its_site_only(self, mock_scrape, _mock_desc):
        mock_scrape.return_value = []
        assert scrape_builtin(["producer"]) == []
        assert {c.args[0] for c in mock_scrape.call_args_list} == {"seattle"}
        assert mock_scrape.call_count == len(scraper_builtin.BUILTIN_SEARCH_SEEDS)

    @patch("job_agent.scraper_builtin._add_descriptions")
    @patch("job_agent.scraper_builtin._scrape_region")
    def test_dedupes_by_url(self, mock_scrape, _mock_desc, monkeypatch):
        monkeypatch.setattr(scraper_builtin, "REGIONS", regions_from_sites([SEATTLE, NATIONAL]))
        # Same URL returned from both regions — should appear once
        dup_job = {"title": "Producer", "company": "X", "location": "Remote",
                   "job_url": f"{SEATTLE}/job/dup/5",
                   "description": "", "date_posted": "", "site": "builtin", "department": ""}
        mock_scrape.return_value = [dup_job]

        jobs = scrape_builtin(["producer"])
        assert len(jobs) == 1
        assert mock_scrape.call_count == 2 * len(scraper_builtin.BUILTIN_SEARCH_SEEDS)

    @patch("job_agent.scraper_builtin.time.sleep", lambda _: None)
    @patch("job_agent.scraper_builtin._session")
    def test_no_sites_means_no_requests(self, mock_session, monkeypatch, tmp_path, caplog):
        from job_agent.candidate_profile import builtin_sites
        monkeypatch.setattr(
            scraper_builtin, "REGIONS",
            regions_from_sites(builtin_sites(_profile_with(tmp_path, None))))
        assert scraper_builtin.REGIONS == {}
        with caplog.at_level(logging.INFO, logger="job_agent.scraper_builtin"):
            assert scrape_builtin(["producer"]) == []
        mock_session.get.assert_not_called()
        skipped = [r for r in caplog.records
                   if r.name == "job_agent.scraper_builtin" and "builtin_sites" in r.getMessage()]
        assert len(skipped) == 1
        assert skipped[0].levelno == logging.INFO

    @patch("job_agent.scraper_builtin.time.sleep", lambda _: None)
    @patch("job_agent.scraper_builtin._session")
    @patch("job_agent.scraper_builtin._scrape_region")
    def test_returned_rows_carry_the_posting_text(self, mock_scrape, mock_session):
        job = {"title": "Video Producer", "company": "Gullwing Pictures", "location": "Seattle",
               "job_url": f"{SEATTLE}/job/video-producer/9000580",
               "description": "", "date_posted": "", "site": "builtin", "department": ""}
        mock_scrape.side_effect = lambda region, seed, kws, **_: [dict(job)] if region == "seattle" else []
        mock_session.get.return_value = _resp(DETAIL_HTML)

        jobs = scrape_builtin(["producer"])
        assert len(jobs) == 1
        assert jobs[0]["description"].startswith("About the role")
        # Same keys as before: the text fills the existing field, nothing new.
        assert set(jobs[0]) == set(job)
        mock_session.get.assert_called_once_with(job["job_url"], timeout=15)


# --- posting page text ------------------------------------------------------

# A Built In posting page in its real shape (made-up content): the JobPosting
# sits in an @graph, its description is HTML inside the JSON string.
DETAIL_HTML = """
<html><head>
<script type="application/ld+json">
  {"@context": "https://schema.org", "@graph": [
    {"@type": "BreadcrumbList", "itemListElement": []},
    {"@type": "JobPosting", "title": "Video Producer",
     "description": "<p><strong>About the role</strong></p><p>Run multi-camera shoots &amp; cut the footage.</p><ul><li>Own the production rhythm</li></ul>",
     "datePosted": "2026-03-19"}
  ]}
</script>
</head><body>
<div id="job-post-body-9000580"><p>About the role</p><p>Body copy</p></div>
</body></html>
"""

BODY_ONLY_HTML = """
<html><body>
<script type="application/ld+json">{"@type": "Organization", "name": "Gullwing Pictures"}</script>
<div id="job-post-body-1"><p>About the role</p><p>Run shoots.</p></div>
</body></html>
"""


def _resp(text, status=200):
    r = MagicMock()
    r.text = text
    r.status_code = status
    r.raise_for_status = MagicMock()
    return r


def _job(job_id, base=SEATTLE):
    return {"title": f"Producer {job_id}", "company": "C", "location": "",
            "job_url": f"{base}/job/producer/{job_id}",
            "description": "", "date_posted": "", "site": "builtin", "department": ""}


class TestParseDescription:
    def test_reads_jobposting_from_ld_json_graph(self):
        text = _parse_description(DETAIL_HTML)
        assert text == ("About the role\nRun multi-camera shoots & cut the footage.\n"
                        "Own the production rhythm")

    def test_falls_back_to_body_div(self):
        assert _parse_description(BODY_ONLY_HTML) == "About the role\nRun shoots."

    def test_entity_encoded_markup(self):
        page = ('<script type="application/ld+json">{"@type": "JobPosting", '
                '"description": "&lt;p&gt;Edit video&lt;/p&gt;"}</script>')
        assert _parse_description(page) == "Edit video"

    def test_bad_json_and_no_body_gives_empty(self):
        page = '<script type="application/ld+json">{not json</script><p>nav</p>'
        assert _parse_description(page) == ""

    def test_page_gives_date_posted(self):
        text, date = _parse_posting_page(DETAIL_HTML)
        assert text.startswith("About the role")
        assert date == "2026-03-19"

    def test_date_without_description_and_iso_datetime(self):
        page = ('<script type="application/ld+json">{"@type": "JobPosting", '
                '"datePosted": "2026-03-13T07:40:39+00:00"}</script>'
                '<div id="job-post-body-9"><p>Body</p></div>')
        assert _parse_posting_page(page) == ("Body", "2026-03-13")

    def test_no_date_gives_empty(self):
        assert _parse_posting_page(BODY_ONLY_HTML)[1] == ""


class TestJobId:
    def test_trailing_id(self):
        assert _job_id(f"{SEATTLE}/job/associate-producer/9000030") == 9000030

    def test_no_id(self):
        assert _job_id(f"{SEATTLE}/jobs") == 0


@patch("job_agent.scraper_builtin.time.sleep", lambda _: None)
class TestAddDescriptions:
    @patch("job_agent.scraper_builtin._session")
    def test_newest_ids_first_within_cap(self, mock_session):
        mock_session.get.return_value = _resp(DETAIL_HTML)
        jobs = [_job(100), _job(300), _job(200)]

        assert _add_descriptions(jobs, cap=2) == 2
        fetched = [c.args[0] for c in mock_session.get.call_args_list]
        assert fetched == [jobs[1]["job_url"], jobs[2]["job_url"]]
        assert jobs[0]["description"] == ""  # oldest, over the cap: list data only
        assert jobs[1]["description"] and jobs[2]["description"]

    @patch("job_agent.scraper_builtin._session")
    def test_same_posting_on_two_sites_fetched_once(self, mock_session):
        mock_session.get.return_value = _resp(DETAIL_HTML)
        jobs = [_job(500, SEATTLE), _job(500, NATIONAL)]

        assert _add_descriptions(jobs) == 2
        assert mock_session.get.call_count == 1
        assert jobs[0]["description"] == jobs[1]["description"] != ""

    @patch("job_agent.scraper_builtin._session")
    def test_persistent_429_stops_fetching(self, mock_session):
        mock_session.get.return_value = _resp("", status=429)
        jobs = [_job(i) for i in range(1, 6)]

        assert _add_descriptions(jobs) == 0
        # 3 retries on the first page, then no other page is tried.
        assert mock_session.get.call_count == 3
        assert all(j["description"] == "" for j in jobs)

    @patch("job_agent.scraper_builtin._session")
    def test_failures_in_a_row_stop_fetching(self, mock_session):
        mock_session.get.side_effect = Exception("timed out")
        jobs = [_job(i) for i in range(1, 11)]

        assert _add_descriptions(jobs) == 0
        assert mock_session.get.call_count == 3

    @patch("job_agent.scraper_builtin._session")
    def test_page_date_replaces_card_age(self, mock_session):
        # The page's plain date beats the card's relative age; a page without
        # one leaves the card's date alone.
        mock_session.get.side_effect = [_resp(DETAIL_HTML), _resp(BODY_ONLY_HTML)]
        jobs = [_job(2), _job(1)]
        jobs[0]["date_posted"] = jobs[1]["date_posted"] = "2026-03-10"

        _add_descriptions(jobs)
        assert jobs[0]["date_posted"] == "2026-03-19"
        assert jobs[1]["date_posted"] == "2026-03-10"

    @patch("job_agent.scraper_builtin._session")
    def test_same_posting_on_two_sites_shares_the_date(self, mock_session):
        mock_session.get.return_value = _resp(DETAIL_HTML)
        jobs = [_job(500, SEATTLE), _job(500, NATIONAL)]

        _add_descriptions(jobs)
        assert [j["date_posted"] for j in jobs] == ["2026-03-19", "2026-03-19"]

    @patch("job_agent.scraper_builtin._session")
    def test_one_bad_page_does_not_stop_the_rest(self, mock_session):
        mock_session.get.side_effect = [Exception("404"), _resp(DETAIL_HTML), _resp(DETAIL_HTML)]
        jobs = [_job(3), _job(2), _job(1)]

        assert _add_descriptions(jobs) == 2
        assert jobs[0]["description"] == ""
        assert jobs[1]["description"] and jobs[2]["description"]
