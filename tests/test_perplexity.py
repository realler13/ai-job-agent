"""Tests for Perplexity Sonar API integration (Phase 15) + Salary Advisor (Phase 16)."""

import json
import time
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.fixtures.persona import YEARS

from job_agent.perplexity_utils import (
    PerplexityRateLimiter,
    _raw_search,
    _within_ttl,
    enrich_borderline_job,
    generate_salary_recommendation,
    get_company_profile,
    get_hiring_intel,
    get_salary_intel,
    resolve_careers_page,
)


# === TTL Helper ===


class TestWithinTTL:
    def test_within_ttl_fresh(self):
        """Today's date should be within any positive TTL."""
        assert _within_ttl(date.today().isoformat(), 7) is True

    def test_within_ttl_expired(self):
        """Date 10 days ago should be outside 7-day TTL."""
        old = (date.today() - timedelta(days=10)).isoformat()
        assert _within_ttl(old, 7) is False

    def test_within_ttl_none(self):
        """None date should return False."""
        assert _within_ttl(None, 7) is False

    def test_within_ttl_bad_format(self):
        """Bad date string should return False gracefully."""
        assert _within_ttl("not-a-date", 7) is False

    def test_within_ttl_exact_boundary(self):
        """Date exactly at TTL boundary should still be within TTL (< not <=)."""
        boundary = (date.today() - timedelta(days=6)).isoformat()
        assert _within_ttl(boundary, 7) is True

    def test_within_ttl_at_limit(self):
        """Date exactly TTL days ago should be outside TTL."""
        at_limit = (date.today() - timedelta(days=7)).isoformat()
        assert _within_ttl(at_limit, 7) is False


# === Rate Limiter ===


class TestPerplexityRateLimiter:
    def test_no_wait_under_limit(self):
        """Should not block when under the RPM limit."""
        limiter = PerplexityRateLimiter(rpm=100)
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        assert elapsed < 0.5

    def test_tracks_calls(self):
        """Should track call timestamps."""
        limiter = PerplexityRateLimiter(rpm=100)
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        assert len(limiter._calls) == 2


# === _raw_search ===


class TestRawSearch:
    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", False)
    def test_disabled_returns_none(self):
        """Should return None when Perplexity is disabled."""
        result = _raw_search("test query")
        assert result is None

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils.requests.post")
    @patch("job_agent.perplexity_utils._rate_limiter")
    def test_successful_call(self, mock_limiter, mock_post):
        """Should return message text from an Agent API `output` array."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "output": [
                {"type": "search_results", "results": [{"url": "https://x.test"}]},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "Netflix is a streaming company."}
                ]},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = _raw_search("Tell me about Netflix")
        assert result == "Netflix is a streaming company."
        mock_post.assert_called_once()
        body = mock_post.call_args.kwargs["json"]
        assert body == {"preset": "fast", "input": "Tell me about Netflix"}
        assert mock_post.call_args.args[0].endswith("/v1/agent")

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils.requests.post")
    @patch("job_agent.perplexity_utils._rate_limiter")
    def test_api_error_returns_none(self, mock_limiter, mock_post):
        """Should return None on API error."""
        mock_post.side_effect = Exception("API Error")
        result = _raw_search("test")
        assert result is None

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils.requests.post")
    @patch("job_agent.perplexity_utils._rate_limiter")
    def test_agent_search_returns_grounding(self, mock_limiter, mock_post):
        """_agent_search keeps search_results (cite-or-omit needs them) and
        honors a preset override."""
        from job_agent.perplexity_utils import _agent_search
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "output": [
                {"type": "search_results", "results": [
                    {"url": "https://a.test", "title": "A", "last_updated": "2026-08-01"},
                ]},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "Answer."}
                ]},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        out = _agent_search("who hires here", preset="low")
        assert out["content"] == "Answer."
        assert out["search_results"][0]["url"] == "https://a.test"
        assert mock_post.call_args.kwargs["json"]["preset"] == "low"

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils.requests.post")
    @patch("job_agent.perplexity_utils._rate_limiter")
    def test_empty_output_returns_none(self, mock_limiter, mock_post):
        """A response with no message text is a failure, not an empty string."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"output": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        assert _raw_search("test") is None


# === get_company_profile ===


class TestGetCompanyProfile:
    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", False)
    def test_disabled(self):
        assert get_company_profile("Netflix") is None

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    def test_cache_hit(self):
        """Should return cached profile when within TTL."""
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = {
            "company_profile": "Cached profile",
            "last_checked": date.today().isoformat(),
        }
        result = get_company_profile("Netflix", db=mock_db)
        assert result == "Cached profile"

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils._raw_search")
    def test_cache_miss_fetches(self, mock_search):
        """Should call API on cache miss and save result."""
        mock_search.return_value = "Netflix is a streaming giant."
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = None

        result = get_company_profile("Netflix", db=mock_db)
        assert result == "Netflix is a streaming giant."
        mock_db.save_company_intel.assert_called_once()

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils._raw_search")
    def test_stale_cache_refetches(self, mock_search):
        """Should refetch when cache is expired."""
        mock_search.return_value = "Updated profile"
        mock_db = MagicMock()
        old_date = (date.today() - timedelta(days=60)).isoformat()
        mock_db.get_company_intel.return_value = {
            "company_profile": "Old profile",
            "last_checked": old_date,
        }
        result = get_company_profile("Netflix", db=mock_db)
        assert result == "Updated profile"


# === resolve_careers_page ===


class TestResolveCareersPage:
    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", False)
    def test_disabled(self):
        assert resolve_careers_page("Netflix") is None

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils._raw_search")
    def test_parses_json_from_response(self, mock_search):
        """Should extract JSON from Perplexity response."""
        mock_search.return_value = (
            'The careers page is at: {"careers_url": "https://jobs.netflix.com", "ats_type": "Greenhouse"}'
        )
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = None

        result = resolve_careers_page("Netflix", db=mock_db)
        assert result["careers_url"] == "https://jobs.netflix.com"
        assert result["ats_type"] == "Greenhouse"

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils._raw_search")
    def test_no_json_returns_none(self, mock_search):
        """Should return None when response has no JSON."""
        mock_search.return_value = "I'm not sure about their careers page."
        result = resolve_careers_page("UnknownCorp")
        assert result is None


# === enrich_borderline_job ===


class TestEnrichBorderlineJob:
    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", False)
    def test_disabled(self):
        assert enrich_borderline_job("Producer", "Netflix") is None

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils._raw_search")
    def test_returns_enrichment(self, mock_search):
        """Should return enrichment text."""
        mock_search.return_value = "This is a content production role."
        result = enrich_borderline_job("Content Producer", "Netflix")
        assert "production" in result

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    def test_cache_hit_uses_enrichment_key(self):
        """Should use __enrichment__ prefix for cache key."""
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = {
            "company_profile": "Cached enrichment",
            "last_checked": date.today().isoformat(),
        }
        result = enrich_borderline_job("Producer", "Netflix", db=mock_db)
        assert result == "Cached enrichment"
        mock_db.get_company_intel.assert_called_with("__enrichment__Netflix")

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils._raw_search")
    def test_saves_with_enrichment_key(self, mock_search):
        """Should save cache with __enrichment__ prefix."""
        mock_search.return_value = "This is a production role."
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = None

        enrich_borderline_job("Producer", "Netflix", db=mock_db)
        call_args = mock_db.save_company_intel.call_args[0][0]
        assert call_args["company_name"] == "__enrichment__Netflix"


# === get_hiring_intel ===


class TestGetHiringIntel:
    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", False)
    def test_disabled(self):
        assert get_hiring_intel("Netflix", "Producer") is None

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    def test_cache_hit(self):
        """Should return cached hiring intel when within TTL."""
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = {
            "company_profile": "VP of Content is hiring",
            "last_checked": date.today().isoformat(),
        }
        result = get_hiring_intel("Netflix", "Producer", db=mock_db)
        assert result == "VP of Content is hiring"
        # per company+ROLE key (the old per-company key cross-served roles)
        mock_db.get_company_intel.assert_called_with("__hiring_intel__Netflix__Producer")

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils._raw_search")
    def test_cache_miss_fetches_and_saves(self, mock_search):
        """Should call API and save with __hiring_intel__ prefix."""
        mock_search.return_value = "Found hiring manager Jane Doe."
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = None

        result = get_hiring_intel("Netflix", "Producer", db=mock_db)
        assert "Jane Doe" in result
        call_args = mock_db.save_company_intel.call_args[0][0]
        assert call_args["company_name"] == "__hiring_intel__Netflix__Producer"


# === get_salary_intel ===


class TestGetSalaryIntel:
    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", False)
    def test_disabled(self):
        assert get_salary_intel("Producer", "Netflix", "Seattle") is None

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    def test_cache_hit(self):
        """Should return cached salary intel within TTL."""
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = {
            "company_profile": "$120K-$180K base",
            "last_checked": date.today().isoformat(),
        }
        result = get_salary_intel("Producer", "Netflix", "Seattle", db=mock_db)
        assert result == "$120K-$180K base"
        mock_db.get_company_intel.assert_called_with("__salary_intel__Netflix__Producer")

    @patch("job_agent.perplexity_utils.USE_PERPLEXITY", True)
    @patch("job_agent.perplexity_utils._raw_search")
    def test_cache_miss_fetches(self, mock_search):
        """Should call API on cache miss and save."""
        mock_search.return_value = "Salary range: $130K-$170K"
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = None

        result = get_salary_intel("Producer", "Netflix", "Seattle", db=mock_db)
        assert "$130K" in result
        call_args = mock_db.save_company_intel.call_args[0][0]
        assert call_args["company_name"] == "__salary_intel__Netflix__Producer"


# === generate_salary_recommendation ===


class TestGenerateSalaryRecommendation:
    @patch("job_agent.gpt_filter.call_llm")
    def test_generates_and_caches(self, mock_llm):
        """Should call Opus and cache the result."""
        mock_llm.return_value = "Target $150K-$180K. Strong leverage from a verified production record."
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = None

        job = {"title": "Senior Producer", "company": "Netflix", "location": "LA", "description": "Video production"}
        intel = {"company_profile": "Streaming giant", "salary_intel": "$140K-$180K", "hiring_intel": None}

        result = generate_salary_recommendation(job, intel, profile_text=f"{YEARS} of experience", db=mock_db)
        assert "150K" in result
        mock_llm.assert_called_once()
        # Verify it was cached
        call_args = mock_db.save_company_intel.call_args[0][0]
        assert call_args["company_name"] == "__salary_rec__Netflix__Senior Producer"

    @patch("job_agent.gpt_filter.call_llm")
    def test_cache_hit_skips_llm(self, mock_llm):
        """Should return cached recommendation without calling LLM."""
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = {
            "company_profile": "Cached recommendation: target $160K",
            "last_checked": date.today().isoformat(),
        }

        job = {"title": "Producer", "company": "Netflix", "location": "LA"}
        intel = {}

        result = generate_salary_recommendation(job, intel, db=mock_db)
        assert "160K" in result
        mock_llm.assert_not_called()

    @patch("job_agent.gpt_filter.call_llm")
    def test_llm_failure_returns_none(self, mock_llm):
        """Should return None when LLM call fails."""
        mock_llm.side_effect = Exception("API error")
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = None

        job = {"title": "Producer", "company": "Netflix", "location": "LA"}
        intel = {}

        result = generate_salary_recommendation(job, intel, db=mock_db)
        assert result is None

    @patch("job_agent.gpt_filter.call_llm")
    def test_empty_llm_response_returns_none(self, mock_llm):
        """Should return None when LLM returns empty string."""
        mock_llm.return_value = ""
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = None

        job = {"title": "Producer", "company": "Netflix", "location": "LA"}
        intel = {}

        result = generate_salary_recommendation(job, intel, db=mock_db)
        assert result is None

    @patch("job_agent.gpt_filter.call_llm")
    def test_no_db_still_works(self, mock_llm):
        """Should work without DB (no caching)."""
        mock_llm.return_value = "Target $140K"
        job = {"title": "Producer", "company": "Netflix", "location": "LA"}
        intel = {"company_profile": "Streaming", "salary_intel": "$130K", "hiring_intel": None}

        result = generate_salary_recommendation(job, intel, profile_text="experienced", db=None)
        assert "140K" in result

    @patch("job_agent.gpt_filter.call_llm")
    def test_prompt_includes_metrics_bank(self, mock_llm):
        """Should include METRICS_BANK_TEXT in the prompt."""
        mock_llm.return_value = "Strategy text"
        mock_db = MagicMock()
        mock_db.get_company_intel.return_value = None

        job = {"title": "Producer", "company": "Netflix", "location": "LA"}
        intel = {"company_profile": None, "salary_intel": None, "hiring_intel": None}

        generate_salary_recommendation(job, intel, db=mock_db)
        prompt = mock_llm.call_args[0][0]
        assert "VERIFIED METRICS" in prompt
        from job_agent.config import METRICS_BANK_TEXT
        assert METRICS_BANK_TEXT.splitlines()[1][:40] in prompt  # metrics bank v2 (2026-07-16) wording


# === _auto_enrich_matched_jobs ===


class TestAutoEnrichMatchedJobs:
    @patch("job_agent.config.USE_PERPLEXITY", False)
    @patch("job_agent.config.AUTO_ENRICH_MATCHES", True)
    def test_skipped_when_perplexity_disabled(self):
        """Should skip enrichment when Perplexity is disabled."""
        from job_agent.main import _auto_enrich_matched_jobs
        import logging
        mock_db = MagicMock()
        df = pd.DataFrame([{"title": "Producer", "company": "Netflix", "location": "LA"}])
        # Should return without error
        _auto_enrich_matched_jobs(df, "profile", mock_db, None, logging.getLogger("test"))
        mock_db.get_company_intel.assert_not_called()

    @patch("job_agent.config.USE_PERPLEXITY", True)
    @patch("job_agent.config.AUTO_ENRICH_MATCHES", False)
    def test_skipped_when_auto_enrich_disabled(self):
        """Should skip enrichment when AUTO_ENRICH_MATCHES is False."""
        from job_agent.main import _auto_enrich_matched_jobs
        import logging
        mock_db = MagicMock()
        df = pd.DataFrame([{"title": "Producer", "company": "Netflix", "location": "LA"}])
        _auto_enrich_matched_jobs(df, "profile", mock_db, None, logging.getLogger("test"))
        mock_db.get_company_intel.assert_not_called()

    @patch("job_agent.perplexity_utils.generate_salary_recommendation")
    @patch("job_agent.perplexity_utils.get_salary_intel")
    @patch("job_agent.perplexity_utils.get_hiring_intel")
    @patch("job_agent.perplexity_utils.get_company_profile")
    @patch("job_agent.config.USE_PERPLEXITY", True)
    @patch("job_agent.config.AUTO_ENRICH_MATCHES", True)
    def test_enriches_all_matched_jobs(self, mock_profile, mock_hiring,
                                        mock_salary, mock_sal_rec):
        """Should call all enrichment functions for each matched job."""
        from job_agent.main import _auto_enrich_matched_jobs
        import logging

        mock_profile.return_value = "Company profile"
        mock_hiring.return_value = "Hiring intel"
        mock_salary.return_value = "Salary data"
        mock_sal_rec.return_value = "Target $150K"

        mock_db = MagicMock()
        df = pd.DataFrame([
            {"title": "Producer", "company": "Netflix", "location": "LA"},
            {"title": "PM", "company": "Disney", "location": "Burbank"},
        ])

        progress_calls = []
        def mock_progress(text, pct):
            progress_calls.append(text)

        _auto_enrich_matched_jobs(df, "profile text", mock_db, mock_progress, logging.getLogger("test"))

        assert mock_profile.call_count == 2
        assert mock_hiring.call_count == 2
        assert mock_salary.call_count == 2
        assert mock_sal_rec.call_count == 2
        assert len(progress_calls) >= 2  # At least initial + per-job

    @patch("job_agent.perplexity_utils.generate_salary_recommendation")
    @patch("job_agent.perplexity_utils.get_salary_intel")
    @patch("job_agent.perplexity_utils.get_hiring_intel")
    @patch("job_agent.perplexity_utils.get_company_profile")
    @patch("job_agent.config.USE_PERPLEXITY", True)
    @patch("job_agent.config.AUTO_ENRICH_MATCHES", True)
    def test_continues_on_individual_failure(self, mock_profile, mock_hiring,
                                              mock_salary, mock_sal_rec):
        """Should continue enriching remaining jobs when one fails."""
        from job_agent.main import _auto_enrich_matched_jobs
        import logging

        # First call raises, so all 3 downstream calls fail for job 1
        mock_profile.side_effect = [Exception("API down"), "Disney profile", "Apple profile"]
        mock_hiring.return_value = "Hiring"
        mock_salary.return_value = "Salary"
        mock_sal_rec.return_value = "Rec"

        mock_db = MagicMock()
        df = pd.DataFrame([
            {"title": "P1", "company": "Netflix", "location": "LA"},
            {"title": "P2", "company": "Disney", "location": "Burbank"},
            {"title": "P3", "company": "Apple", "location": "Austin, TX"},
        ])

        _auto_enrich_matched_jobs(df, "profile", mock_db, None, logging.getLogger("test"))

        # The first job failed at get_company_profile, so all 3 profile calls happened
        assert mock_profile.call_count == 3
        # hiring/salary/recommendation only called for the 2 that succeeded at profile step
        assert mock_sal_rec.call_count == 2
