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
    _scrape_region,
    _parse_card,
    _parse_posting_age,
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
    @patch("job_agent.scraper_builtin._scrape_region")
    def test_merges_two_regions(self, mock_scrape, monkeypatch):
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

    @patch("job_agent.scraper_builtin._scrape_region")
    def test_fixture_profile_scrapes_its_site_only(self, mock_scrape):
        mock_scrape.return_value = []
        assert scrape_builtin(["producer"]) == []
        assert {c.args[0] for c in mock_scrape.call_args_list} == {"seattle"}
        assert mock_scrape.call_count == len(scraper_builtin.BUILTIN_SEARCH_SEEDS)

    @patch("job_agent.scraper_builtin._scrape_region")
    def test_dedupes_by_url(self, mock_scrape, monkeypatch):
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
