"""Tests for company_deriver — recover company name from ATS URLs and title suffixes."""
import math
import pytest

from job_agent.company_deriver import (
    derive_company,
    _clean_existing_company,
    _company_from_ats_url,
    _company_from_title_suffix,
    _prettify_slug,
)


class TestCleanExistingCompany:
    def test_none_returns_none(self):
        assert _clean_existing_company(None) is None

    def test_nan_returns_none(self):
        assert _clean_existing_company(float("nan")) is None

    def test_empty_string_returns_none(self):
        assert _clean_existing_company("") is None
        assert _clean_existing_company("   ") is None

    def test_literal_nan_string_returns_none(self):
        assert _clean_existing_company("nan") is None
        assert _clean_existing_company("None") is None
        assert _clean_existing_company("NaN") is None

    def test_valid_company_returned_trimmed(self):
        assert _clean_existing_company("Apple") == "Apple"
        assert _clean_existing_company("  Apple  ") == "Apple"


class TestPrettifySlug:
    def test_hyphenated_slug(self):
        assert _prettify_slug("new-media") == "New Media"

    def test_brand_with_digits_preserved(self):
        # Key case: a brand word with digits (acme3d) should stay lowercase
        assert _prettify_slug("acme3d-new-media") == "acme3d New Media"

    def test_all_digits_preserved(self):
        assert _prettify_slug("23andme") == "23andme"

    def test_single_word(self):
        assert _prettify_slug("driftwood") == "Driftwood"

    def test_underscore_treated_like_hyphen(self):
        assert _prettify_slug("harbor_lights") == "Harbor Lights"


class TestCompanyFromAtsUrl:
    def test_ashby(self):
        url = "https://jobs.ashbyhq.com/acme3d-new-media/00000000-0000-4000-8000-000000000001"
        assert _company_from_ats_url(url) == "acme3d New Media"

    def test_greenhouse_boards(self):
        url = "https://boards.greenhouse.io/spotify/jobs/12345"
        assert _company_from_ats_url(url) == "Spotify"

    def test_greenhouse_job_boards(self):
        url = "https://job-boards.greenhouse.io/netflix/jobs/12345"
        assert _company_from_ats_url(url) == "Netflix"

    def test_lever(self):
        url = "https://jobs.lever.co/headspace/abc123"
        assert _company_from_ats_url(url) == "Headspace"

    def test_workable(self):
        url = "https://apply.workable.com/some-company/j/abc"
        assert _company_from_ats_url(url) == "Some Company"

    def test_workday_subdomain(self):
        url = "https://goodwillaz.wd1.myworkdayjobs.com/external/job/abc"
        assert _company_from_ats_url(url) == "Goodwillaz"

    def test_recruitee_subdomain(self):
        url = "https://someco.recruitee.com/o/engineer"
        assert _company_from_ats_url(url) == "Someco"

    def test_bamboohr_subdomain(self):
        url = "https://mycompany.bamboohr.com/jobs/view.php?id=42"
        assert _company_from_ats_url(url) == "Mycompany"

    def test_unknown_ats_returns_none(self):
        assert _company_from_ats_url("https://unknown.example.com/jobs/abc") is None

    def test_empty_string(self):
        assert _company_from_ats_url("") is None


class TestCompanyFromTitleSuffix:
    def test_standard_entertainmentcareers_pattern(self):
        assert _company_from_title_suffix("Creative Producer - Harbor Lights Media") == "Harbor Lights Media"

    def test_multi_word_company(self):
        assert _company_from_title_suffix("Video Producer - Bluefin Publishing Group") == "Bluefin Publishing Group"

    def test_multiple_dashes_takes_last(self):
        # "Podcast Producer - Senior - Amazon" → "Amazon"
        assert _company_from_title_suffix("Podcast Producer - Senior - Amazon") == "Amazon"

    def test_work_mode_rejected(self):
        assert _company_from_title_suffix("YouTube Channel Lead - Hybrid") is None
        assert _company_from_title_suffix("Producer - Remote") is None

    def test_employment_type_rejected(self):
        assert _company_from_title_suffix("Producer - Full Time") is None
        assert _company_from_title_suffix("Editor - Contract") is None

    def test_salary_rejected(self):
        assert _company_from_title_suffix("Senior Editor - $140k") is None
        assert _company_from_title_suffix("Producer - Up to $200k") is None

    def test_location_shorthand_rejected(self):
        assert _company_from_title_suffix("Editor - LA") is None
        assert _company_from_title_suffix("Producer - NYC") is None

    def test_no_dash_returns_none(self):
        assert _company_from_title_suffix("Podcast Producer") is None

    def test_empty_title(self):
        assert _company_from_title_suffix("") is None


class TestDeriveCompany:
    def test_existing_company_wins(self):
        # If company is already set, don't override
        job = {"company": "Spotify", "job_url_direct": "https://jobs.ashbyhq.com/netflix/x"}
        assert derive_company(job) == "Spotify"

    def test_falls_back_to_ats_url_when_company_nan(self):
        job = {
            "company": float("nan"),
            "title": "Head of Podcasts",
            "job_url_direct": "https://jobs.ashbyhq.com/acme3d-new-media/0000abcd",
        }
        assert derive_company(job) == "acme3d New Media"

    def test_falls_back_to_title_suffix_when_no_url(self):
        job = {
            "company": None,
            "title": "Creative Producer - Harbor Lights Media",
        }
        assert derive_company(job) == "Harbor Lights Media"

    def test_ats_url_preferred_over_title_suffix(self):
        # Both signals available — ATS URL is more reliable
        job = {
            "company": None,
            "title": "Video Producer - Some Random Suffix",
            "job_url_direct": "https://jobs.ashbyhq.com/real-company/abc",
        }
        assert derive_company(job) == "Real Company"

    def test_all_signals_missing_returns_none(self):
        job = {"company": None, "title": "Video", "job_url_direct": ""}
        assert derive_company(job) is None

    def test_indeed_nan_company_recovered_from_ats_url(self):
        # Regression: a JobSpy Indeed row with a NaN company must recover it from
        # job_url_direct, or batch document generation has no company to use
        job = {
            "title": "Head of Podcasts",
            "company": float("nan"),  # NaN from pandas via JobSpy Indeed
            "site": "indeed",
            "job_url": "https://www.indeed.com/viewjob?jk=0000000000000001",
            "job_url_direct": "https://jobs.ashbyhq.com/acme3d-new-media/00000000-0000-4000-8000-000000000001",
        }
        assert derive_company(job) == "acme3d New Media"

    def test_entertainmentcareers_rss_case(self):
        # Regression: EntertainmentCareers RSS — no company, but title has " - Company"
        job = {
            "title": "Creative Producer - Harbor Lights Media",
            "company": None,
            "site": "entertainmentcareers",
        }
        assert derive_company(job) == "Harbor Lights Media"
