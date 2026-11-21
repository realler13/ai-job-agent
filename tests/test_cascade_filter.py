"""
Tests for the 3-Stage Cascade AI Filtering Pipeline.

Tests cover:
- CascadeStats initialization and summary
- Stage 1: Local pre-filter (binary YES/NO)
- Stage 2: Full grading (batched)
- Stage 3: Judge (borderline POTENTIAL_MATCH)
- Full pipeline integration

IMPORTANT: call_llm is imported in cascade_filter.py as `from .gpt_filter import call_llm`,
so we MUST patch `job_agent.cascade_filter.call_llm` (the importing module's reference).
"""

import json
import pytest
import pandas as pd

from job_agent.cascade_filter import CascadeFilter, CascadeStats, assign_tier
from tests.fixtures.persona import YEARS


# ── Helpers ──────────────────────────────────────────────────────────────────

PROFILE_TEXT = f"Senior Video Producer with {YEARS} in Seattle-area media companies."


def make_sample_df(n=2):
    """Create a sample DataFrame with n jobs for testing."""
    jobs = [
        {
            "title": "Senior Video Producer",
            "company": "Netflix",
            "location": "Seattle, WA",
            "date_posted": "2025-01-15",
            "description": "Looking for a senior video producer to lead content strategy.",
        },
        {
            "title": "Software Engineer",
            "company": "Google",
            "location": "New York, NY",
            "date_posted": "2025-01-16",
            "description": "Build scalable distributed systems in Go and Python.",
        },
        {
            "title": "Content Producer",
            "company": "Meta",
            "location": "Remote",
            "date_posted": "2025-01-17",
            "description": "Create and manage digital content across platforms.",
        },
        {
            "title": "Creative Director",
            "company": "Apple",
            "location": "Redmond, WA",
            "date_posted": "2025-01-18",
            "description": "Lead creative team of 20+. Oversee video and digital content strategy.",
        },
        {
            "title": "Social Media Manager",
            "company": "StartupCo",
            "location": "Bellevue, WA",
            "date_posted": "2025-01-19",
            "description": "Manage social media accounts and community engagement.",
        },
    ]
    return pd.DataFrame(jobs[:n])


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestCascadeStats
# ═══════════════════════════════════════════════════════════════════════════════


class TestCascadeStats:
    def test_initial_values(self):
        """All counters start at 0."""
        stats = CascadeStats()
        assert stats.stage1_total == 0
        assert stats.stage1_passed == 0
        assert stats.stage1_filtered == 0
        assert stats.stage2_total == 0
        assert stats.stage2_high == 0
        assert stats.stage2_potential == 0
        assert stats.stage2_rejected == 0
        assert stats.stage3_total == 0
        assert stats.stage3_upgraded == 0
        assert stats.stage3_confirmed == 0
        assert stats.stage3_downgraded == 0
        assert stats.start_time is None
        assert stats.end_time is None

    def test_summary_format(self):
        """summary() returns dict with stage1/stage2/stage3/elapsed_seconds keys."""
        stats = CascadeStats()
        summary = stats.summary()

        assert "stage1" in summary
        assert "stage2" in summary
        assert "stage3" in summary
        assert "elapsed_seconds" in summary

        # Verify nested structure
        assert "total" in summary["stage1"]
        assert "passed" in summary["stage1"]
        assert "filtered" in summary["stage1"]

        assert "total" in summary["stage2"]
        assert "high_match" in summary["stage2"]
        assert "potential_match" in summary["stage2"]
        assert "rejected" in summary["stage2"]

        assert "total" in summary["stage3"]
        assert "upgraded" in summary["stage3"]
        assert "confirmed" in summary["stage3"]
        assert "downgraded" in summary["stage3"]

        # With no start/end time, elapsed should be 0
        assert summary["elapsed_seconds"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestStage1
# ═══════════════════════════════════════════════════════════════════════════════


class TestStage1:
    def test_stage1_yes_passes(self, monkeypatch):
        """When call_llm returns 'YES', job passes through Stage 1."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "YES",
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(1)
        passed_df, filtered_df = cf.run_stage1(df)

        assert len(passed_df) == 1
        assert len(filtered_df) == 0

    def test_stage1_no_filters(self, monkeypatch):
        """When call_llm returns 'NO', job is filtered out."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "NO",
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(1)
        passed_df, filtered_df = cf.run_stage1(df)

        assert len(passed_df) == 0
        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]["match_level"] == "FILTERED_OUT"
        assert filtered_df.iloc[0]["gpt_relevance"] == "irrelevant"
        assert filtered_df.iloc[0]["cascade_stage"] == 1

    def test_stage1_with_think_tags(self, monkeypatch):
        """Qwen3 model wraps answer in <think> tags; Stage 1 should strip them."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "<think>This is a senior role in the Seattle area, clearly a match.</think>\nYES",
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(1)
        passed_df, filtered_df = cf.run_stage1(df)

        assert len(passed_df) == 1
        assert len(filtered_df) == 0

    def test_stage1_verdict_with_preamble_parses(self, monkeypatch):
        """A framed reply ('Verdict: NO' / 'Yes.') must parse to the verdict —
        startswith('YES') silently dropped any answer with a preamble word
        (sonnet-5 upgrade hardening, 2026-08-13)."""
        for reply, should_pass in [("Verdict: NO", False), ("Yes.", True),
                                   ("The answer is YES", True)]:
            monkeypatch.setattr(
                "job_agent.cascade_filter.call_llm",
                lambda prompt, model="", temperature=0.0, _r=reply, **_kw: _r,
            )
            cf = CascadeFilter(PROFILE_TEXT)
            passed_df, filtered_df = cf.run_stage1(make_sample_df(1))
            assert (len(passed_df) == 1) == should_pass, reply

    def test_stage1_unparseable_fails_open(self, monkeypatch):
        """No YES/NO token at all → job passes through (never lose a job to
        a formatting quirk)."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "This role is unrelated to production.",
        )
        cf = CascadeFilter(PROFILE_TEXT)
        passed_df, filtered_df = cf.run_stage1(make_sample_df(1))
        assert len(passed_df) == 1

    def test_stage1_prompt_selected_by_model_family(self):
        """Claude Stage-1 models get the XML variant; others keep the plain
        prompt (which the gemini fallback also receives)."""
        from job_agent.config import (
            CASCADE_STAGE1_PROMPT, CASCADE_STAGE1_PROMPT_CLAUDE)
        cf_claude = CascadeFilter(PROFILE_TEXT, stage1_model="claude-sonnet-5")
        assert cf_claude.stage1_prompt == CASCADE_STAGE1_PROMPT_CLAUDE
        assert "<output_format>" in cf_claude.stage1_prompt
        cf_gemini = CascadeFilter(PROFILE_TEXT, stage1_model="gemini-2.5-flash")
        assert cf_gemini.stage1_prompt == CASCADE_STAGE1_PROMPT

    def test_stage1_error_passes_through(self, monkeypatch):
        """When call_llm raises an Exception, job passes through (fail-safe)."""
        def failing_llm(prompt, model="", temperature=0.0, **_kw):
            raise ConnectionError("Ollama is down")

        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            failing_llm,
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(1)
        passed_df, filtered_df = cf.run_stage1(df)

        assert len(passed_df) == 1
        assert len(filtered_df) == 0

    def test_stage1_rejection_reason_text(self, monkeypatch):
        """Filtered jobs should have the updated rejection reason."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "NO",
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(1)
        passed_df, filtered_df = cf.run_stage1(df)

        assert len(filtered_df) == 1
        assert "not relevant creative/media role" in filtered_df.iloc[0]["reason"]

    def test_stage1_prompt_includes_podcast_audio(self, monkeypatch):
        """Stage 1 prompt should mention podcast and audio roles."""
        captured_prompts = []

        def capturing_llm(prompt, model="", temperature=0.0, **_kw):
            captured_prompts.append(prompt)
            return "YES"

        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            capturing_llm,
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = pd.DataFrame([{
            "title": "Senior Podcast Producer",
            "company": "PodCo",
            "location": "",
            "date_posted": "2025-01-15",
            "description": "Lead podcast production for a growing media company.",
        }])
        passed_df, filtered_df = cf.run_stage1(df)

        assert len(passed_df) == 1
        # Verify the plausibility-gate prompt still spans podcast/audio/content
        assert any("podcast" in p.lower() for p in captured_prompts)
        assert any("audio" in p.lower() for p in captured_prompts)
        assert any("content" in p.lower() for p in captured_prompts)

    def test_stage1_stats(self, monkeypatch):
        """Verify stats.stage1_passed and stage1_filtered are correctly tracked."""
        call_count = [0]

        def alternating_llm(prompt, model="", temperature=0.0, **_kw):
            call_count[0] += 1
            # First job passes, second is filtered
            if "Senior Video Producer" in prompt:
                return "YES"
            return "NO"

        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            alternating_llm,
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(2)  # Senior Video Producer + Software Engineer
        passed_df, filtered_df = cf.run_stage1(df)

        assert cf.stats.stage1_total == 2
        assert cf.stats.stage1_passed == 1
        assert cf.stats.stage1_filtered == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestStage2
# ═══════════════════════════════════════════════════════════════════════════════


class TestStage2:
    def test_stage2_parses_high_match(self, monkeypatch):
        """Stage 2 correctly parses HIGH MATCH from LLM response."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "Job 1: HIGH MATCH - Perfect senior producer fit at Netflix",
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(1)
        high_df, rejected_df, potential_df = cf.run_stage2(df)

        assert len(high_df) == 1
        assert len(rejected_df) == 0
        assert len(potential_df) == 0
        assert high_df.iloc[0]["match_level"] == "HIGH_MATCH"
        assert high_df.iloc[0]["confidence"] == 0.9
        assert high_df.iloc[0]["cascade_stage"] == 2
        assert high_df.iloc[0]["gpt_relevance"] == "relevant"

    def test_stage2_parses_potential_match(self, monkeypatch):
        """Stage 2 correctly parses POTENTIAL MATCH from LLM response."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "Job 1: POTENTIAL MATCH - Good fit but uncertain seniority level",
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(1)
        high_df, rejected_df, potential_df = cf.run_stage2(df)

        assert len(high_df) == 0
        assert len(rejected_df) == 0
        assert len(potential_df) == 1
        assert potential_df.iloc[0]["match_level"] == "POTENTIAL_MATCH"
        assert potential_df.iloc[0]["confidence"] == 0.6
        assert potential_df.iloc[0]["gpt_relevance"] == "relevant"

    def test_stage2_parses_social_trap(self, monkeypatch):
        """Stage 2 correctly identifies SOCIAL TRAP roles."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "Job 1: SOCIAL TRAP - Community management role disguised as producer",
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(1)
        high_df, rejected_df, potential_df = cf.run_stage2(df)

        assert len(high_df) == 0
        assert len(rejected_df) == 1
        assert len(potential_df) == 0
        assert rejected_df.iloc[0]["match_level"] == "SOCIAL_TRAP"
        assert rejected_df.iloc[0]["rejection_rule"] == "SOCIAL_TRAP"
        assert rejected_df.iloc[0]["gpt_relevance"] == "irrelevant"

    def test_stage2_batching(self, monkeypatch):
        """With 7 jobs and batch_size=3, call_llm should be called 3 times (3+3+1)."""
        call_count = [0]

        def counting_llm(prompt, model="", temperature=0.0, **_kw):
            call_count[0] += 1
            # Return a response that covers however many jobs are in this batch
            lines = []
            for i in range(1, 8):  # More than enough
                lines.append(f"Job {i}: HIGH MATCH - Great fit")
            return "\n".join(lines)

        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            counting_llm,
        )

        cf = CascadeFilter(PROFILE_TEXT, stage2_batch_size=3)
        # Create a 7-job DataFrame
        jobs = []
        for i in range(7):
            jobs.append({
                "title": f"Producer {i}",
                "company": f"Company {i}",
                "location": "Seattle, WA",
                "date_posted": "2025-01-15",
                "description": f"Job description {i}.",
            })
        df = pd.DataFrame(jobs)

        high_df, rejected_df, potential_df = cf.run_stage2(df)

        assert call_count[0] == 3  # ceil(7/3) = 3 batches
        assert len(high_df) == 7  # All should be HIGH_MATCH


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestStage3
# ═══════════════════════════════════════════════════════════════════════════════


class TestStage3:
    def _make_potential_df(self, n=1):
        """Create a DataFrame that looks like Stage 2 POTENTIAL_MATCH output."""
        jobs = []
        for i in range(n):
            jobs.append({
                "title": f"Content Producer {i}",
                "company": f"MediaCo {i}",
                "location": "Seattle, WA",
                "date_posted": "2025-01-15",
                "description": f"Produce content for a growing media team. Role {i}.",
                "match_level": "POTENTIAL_MATCH",
                "confidence": 0.6,
                "reason": "POTENTIAL MATCH - Good fit but uncertain seniority",
                "rejection_rule": "NONE",
                "gpt_relevance": "relevant",
                "gpt_reasoning": "POTENTIAL MATCH - Good fit but uncertain seniority",
                "cascade_stage": 2,
            })
        return pd.DataFrame(jobs)

    def test_stage3_upgrade(self, monkeypatch):
        """Stage 3 upgrades a POTENTIAL_MATCH to HIGH_MATCH when judge says so."""
        response = json.dumps({
            "final_level": "HIGH_MATCH",
            "reason": "Clear strategic fit for senior producer role",
            "apply_priority": 8,
        })
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: response,
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = self._make_potential_df(1)
        result = cf.run_stage3(df)

        assert len(result) == 1
        assert result.iloc[0]["match_level"] == "HIGH_MATCH"
        assert "Stage 3 upgraded" in result.iloc[0]["reason"]
        assert result.iloc[0]["cascade_stage"] == 3
        assert result.iloc[0]["gpt_relevance"] == "relevant"
        assert cf.stats.stage3_upgraded == 1

    def test_stage3_confirm(self, monkeypatch):
        """Stage 3 confirms a POTENTIAL_MATCH as CONFIRMED_POTENTIAL."""
        response = json.dumps({
            "final_level": "CONFIRMED_POTENTIAL",
            "reason": "Worth applying, good culture fit",
            "apply_priority": 6,
        })
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: response,
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = self._make_potential_df(1)
        result = cf.run_stage3(df)

        assert len(result) == 1
        assert result.iloc[0]["match_level"] == "POTENTIAL_MATCH"
        assert "Stage 3 confirmed" in result.iloc[0]["reason"]
        assert result.iloc[0]["cascade_stage"] == 3
        assert cf.stats.stage3_confirmed == 1

    def test_stage3_downgrade_keeps_visible(self, monkeypatch):
        """Recall mode: a legacy 'DOWNGRADED' verdict no longer deletes the job —
        it lands in the worth_a_look tier (still visible) instead of IRRELEVANT."""
        response = json.dumps({
            "final_level": "DOWNGRADED",
            "reason": "Not a strong fit, but adjacent enough to surface",
            "apply_priority": 3,
        })
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: response,
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = self._make_potential_df(1)
        result = cf.run_stage3(df)

        assert len(result) == 1
        assert result.iloc[0]["match_level"] == "WORTH_A_LOOK"
        assert "Stage 3 worth-a-look" in result.iloc[0]["reason"]
        assert result.iloc[0]["gpt_relevance"] == "relevant"
        assert result.iloc[0]["cascade_stage"] == 3

    def test_stage3_worth_a_look(self, monkeypatch):
        """Stage 3 'WORTH_A_LOOK' lands a job in the worth_a_look tier (kept visible)."""
        response = json.dumps({
            "final_level": "WORTH_A_LOOK",
            "reason": "Adjacent stretch role — content-heavy but light on video",
            "apply_priority": 4,
        })
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: response,
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = self._make_potential_df(1)
        result = cf.run_stage3(df)

        assert result.iloc[0]["match_level"] == "WORTH_A_LOOK"
        assert result.iloc[0]["gpt_relevance"] == "relevant"
        assert result.iloc[0]["cascade_stage"] == 3

    def test_stage3_reject_drops_out(self, monkeypatch):
        """Stage 3 'REJECT' (genuinely different job) maps to IRRELEVANT and is dropped."""
        response = json.dumps({
            "final_level": "REJECT",
            "reason": "This is a video-game development producer — a different job",
            "apply_priority": 1,
        })
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: response,
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = self._make_potential_df(1)
        result = cf.run_stage3(df)

        assert len(result) == 1
        assert result.iloc[0]["match_level"] == "IRRELEVANT"
        assert "Stage 3 rejected" in result.iloc[0]["reason"]
        assert result.iloc[0]["gpt_relevance"] == "irrelevant"
        assert cf.stats.stage3_downgraded == 1

    def test_stage3_json_parse_error(self, monkeypatch):
        """When Stage 3 returns no JSON, job keeps POTENTIAL_MATCH (fail-safe)."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "I think this is a good match but let me explain why in detail without any structured format.",
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = self._make_potential_df(1)
        result = cf.run_stage3(df)

        assert len(result) == 1
        assert result.iloc[0]["match_level"] == "POTENTIAL_MATCH"
        assert result.iloc[0]["confidence"] == 0.5
        assert "Stage 3 parse error" in result.iloc[0]["reason"]
        assert result.iloc[0]["cascade_stage"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestFullPipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    def test_full_pipeline_mixed_results(self, monkeypatch):
        """
        Full pipeline with 5 jobs:
        - Stage 1: pass 3 (Senior Video Producer, Content Producer, Creative Director),
                    filter 2 (Software Engineer, Social Media Manager)
        - Stage 2: 1 HIGH (Senior Video Producer) + 1 POTENTIAL (Content Producer) +
                    1 IRRELEVANT (Creative Director)
        - Stage 3: Confirm the POTENTIAL (Content Producer)
        Result: 5 jobs total with correct cascade_stage values.
        """

        def mock_llm_side_effect(prompt, model="", temperature=0.0, **_kw):
            # Stage 1: binary YES/NO based on job title in prompt
            if "YES or NO" in prompt:
                if "Software Engineer" in prompt:
                    return "NO"
                if "Social Media Manager" in prompt:
                    return "NO"
                return "YES"

            # Stage 3: judge prompt (has CANDIDATE PROFILE and JUNIOR REVIEWER)
            if "CANDIDATE PROFILE" in prompt and "JUNIOR REVIEWER" in prompt:
                return json.dumps({
                    "final_level": "CONFIRMED_POTENTIAL",
                    "reason": "Worth applying, decent fit",
                    "apply_priority": 7,
                })

            # Stage 2: batch grading (has jobs_block content)
            # The 3 passed jobs come through in one batch (batch_size=5 by default)
            return (
                "Job 1: HIGH MATCH - Senior Video Producer at Netflix, great fit\n"
                "Job 2: POTENTIAL MATCH - Content Producer is borderline\n"
                "Job 3: IRRELEVANT - Creative Director is too high level"
            )

        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            mock_llm_side_effect,
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(5)
        result = cf.filter_jobs(df)

        assert len(result) == 5

        # Check that we have the expected cascade_stage distribution
        stage_counts = result["cascade_stage"].value_counts().to_dict()
        assert stage_counts.get(1, 0) == 2   # 2 filtered at Stage 1
        assert stage_counts.get(2, 0) == 2   # 1 HIGH + 1 IRRELEVANT decided at Stage 2
        assert stage_counts.get(3, 0) == 1   # 1 POTENTIAL judged at Stage 3

        # Verify stats
        assert cf.stats.stage1_total == 5
        assert cf.stats.stage1_filtered == 2
        assert cf.stats.stage1_passed == 3
        assert cf.stats.stage2_total == 3
        assert cf.stats.stage2_high == 1
        assert cf.stats.stage2_potential == 1
        assert cf.stats.stage2_rejected == 1
        assert cf.stats.stage3_total == 1
        assert cf.stats.stage3_confirmed == 1

        # Check summary returns valid elapsed time
        summary = cf.stats.summary()
        assert summary["elapsed_seconds"] >= 0

    def test_empty_dataframe(self, monkeypatch):
        """Passing an empty DataFrame returns an empty DataFrame without LLM calls."""
        call_count = [0]

        def should_not_be_called(prompt, model="", temperature=0.0, **_kw):
            call_count[0] += 1
            return "YES"

        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            should_not_be_called,
        )

        cf = CascadeFilter(PROFILE_TEXT)
        result = cf.filter_jobs(pd.DataFrame())

        assert len(result) == 0
        assert call_count[0] == 0

    def test_all_filtered_stage1(self, monkeypatch):
        """When all jobs are filtered at Stage 1, no Stage 2/3 calls happen."""
        call_log = []

        def logging_llm(prompt, model="", temperature=0.0, **_kw):
            if "YES or NO" in prompt:
                call_log.append("stage1")
                return "NO"
            else:
                call_log.append("stage2_or_3")
                return "Job 1: IRRELEVANT - should not reach here"

        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            logging_llm,
        )

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(3)
        result = cf.filter_jobs(df)

        # All 3 jobs should be FILTERED_OUT from Stage 1
        assert len(result) == 3
        assert all(result["match_level"] == "FILTERED_OUT")
        assert all(result["cascade_stage"] == 1)

        # Only stage1 calls should have been made, no stage2/3
        assert all(stage == "stage1" for stage in call_log)
        assert len(call_log) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestCrossSourceDedup
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossSourceDedup:
    """Test that cross-source duplicates (same title+company, different URL) are deduped."""

    def test_dedup_removes_cross_source_duplicates(self, monkeypatch):
        """Same title+company from different sources → only first processed by AI."""
        call_count = [0]

        def counting_llm(prompt, model="", temperature=0.0, **_kw):
            call_count[0] += 1
            if "YES or NO" in prompt:
                return "YES"
            return "Job 1: HIGH MATCH - Great fit"

        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            counting_llm,
        )

        df = pd.DataFrame([
            {"title": "Video Producer", "company": "Netflix", "location": "Remote",
             "description": "Lead video production", "date_posted": "2025-01-15",
             "job_url": "https://linkedin.com/jobs/123"},
            {"title": "Video Producer", "company": "Netflix", "location": "Remote",
             "description": "Lead video production", "date_posted": "2025-01-15",
             "job_url": "https://netflix.greenhouse.io/jobs/456"},
        ])

        cf = CascadeFilter(PROFILE_TEXT)
        result = cf.filter_jobs(df)

        assert len(result) == 2  # Both in result, but one is DUPLICATE
        dup_rows = result[result["match_level"] == "DUPLICATE"]
        assert len(dup_rows) == 1
        assert dup_rows.iloc[0]["rejection_rule"] == "DUPLICATE"
        assert dup_rows.iloc[0]["cascade_stage"] == 0

        # Only 1 job should have been sent to Stage 1 (not 2)
        assert call_count[0] >= 1  # At least 1 Stage 1 call
        # The duplicate should NOT have triggered any LLM call

    def test_dedup_case_insensitive(self, monkeypatch):
        """Dedup should be case-insensitive on title and company."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "YES" if "YES or NO" in prompt else "Job 1: HIGH MATCH - good",
        )

        df = pd.DataFrame([
            {"title": "Content Producer", "company": "Meta", "location": "Remote",
             "description": "Create content", "date_posted": "2025-01-15"},
            {"title": "content producer", "company": "META", "location": "Seattle",
             "description": "Create content", "date_posted": "2025-01-15"},
        ])

        cf = CascadeFilter(PROFILE_TEXT)
        result = cf.filter_jobs(df)

        dup_rows = result[result["match_level"] == "DUPLICATE"]
        assert len(dup_rows) == 1

    def test_dedup_preserves_different_jobs(self, monkeypatch):
        """Different title OR company → not deduped."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "YES" if "YES or NO" in prompt else "Job 1: HIGH MATCH - good\nJob 2: HIGH MATCH - also good",
        )

        df = pd.DataFrame([
            {"title": "Video Producer", "company": "Netflix", "location": "Remote",
             "description": "Lead video", "date_posted": "2025-01-15"},
            {"title": "Video Producer", "company": "Disney", "location": "Remote",
             "description": "Lead video", "date_posted": "2025-01-15"},
            {"title": "Audio Producer", "company": "Netflix", "location": "Remote",
             "description": "Lead audio", "date_posted": "2025-01-15"},
        ])

        cf = CascadeFilter(PROFILE_TEXT)
        result = cf.filter_jobs(df)

        dup_rows = result[result["match_level"] == "DUPLICATE"]
        assert len(dup_rows) == 0  # All 3 are distinct, no dedup

    def test_dedup_no_dedup_key_column_leak(self, monkeypatch):
        """The internal _dedup_key column should not appear in the result."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "YES" if "YES or NO" in prompt else "Job 1: HIGH MATCH - good",
        )

        df = pd.DataFrame([
            {"title": "Video Producer", "company": "Netflix", "location": "Remote",
             "description": "Lead video", "date_posted": "2025-01-15"},
            {"title": "Video Producer", "company": "Netflix", "location": "Remote",
             "description": "Lead video", "date_posted": "2025-01-15"},
        ])

        cf = CascadeFilter(PROFILE_TEXT)
        result = cf.filter_jobs(df)

        assert "_dedup_key" not in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TestStageFallbacks
# ═══════════════════════════════════════════════════════════════════════════════


class TestStageFallbacks:
    """Test that when the primary model fails, stage-level fallbacks are tried."""

    def test_stage1_fallback_on_primary_failure(self, monkeypatch):
        """Stage 1: primary model returns empty → fallback model is tried and succeeds."""
        call_log = []

        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            call_log.append(model)
            if model == "ollama/qwen3:8b":
                return ""  # Primary fails (empty response)
            if model == "ollama/gemma3:12b":
                return "YES"  # Fallback succeeds
            return "YES"

        monkeypatch.setattr("job_agent.cascade_filter.call_llm", mock_llm)

        cf = CascadeFilter(
            PROFILE_TEXT,
            stage1_model="ollama/qwen3:8b",
            stage1_fallbacks=["ollama/gemma3:12b", "ollama/qwen3:14b"],
        )
        df = make_sample_df(1)
        passed_df, filtered_df = cf.run_stage1(df)

        # Should have tried primary then first fallback
        assert "ollama/qwen3:8b" in call_log
        assert "ollama/gemma3:12b" in call_log
        # qwen3:14b should NOT have been tried (gemma3 succeeded)
        assert "ollama/qwen3:14b" not in call_log
        assert len(passed_df) == 1  # YES → passed

    def test_stage1_all_fallbacks_fail_passes_through(self, monkeypatch):
        """Stage 1: all models fail → job passes through to Stage 2 (fail-safe)."""
        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            raise ConnectionError(f"Model {model} is down")

        monkeypatch.setattr("job_agent.cascade_filter.call_llm", mock_llm)

        cf = CascadeFilter(
            PROFILE_TEXT,
            stage1_model="ollama/qwen3:8b",
            stage1_fallbacks=["ollama/gemma3:12b"],
        )
        df = make_sample_df(1)
        passed_df, filtered_df = cf.run_stage1(df)

        # On total failure, job passes through (don't lose it)
        assert len(passed_df) == 1
        assert len(filtered_df) == 0

    def test_stage2_fallback_on_primary_failure(self, monkeypatch):
        """Stage 2: primary model returns empty → fallback model succeeds."""
        call_log = []

        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            call_log.append(model)
            # Stage 1 always passes
            if "YES or NO" in prompt:
                return "YES"
            # Stage 2: primary fails, fallback works
            if model == "gemini-2.5-flash":
                return ""  # Empty = failure
            if model == "claude-haiku-4-5-20251001":
                return "Job 1: HIGH MATCH - Great producer fit"
            return ""

        monkeypatch.setattr("job_agent.cascade_filter.call_llm", mock_llm)

        cf = CascadeFilter(
            PROFILE_TEXT,
            stage2_model="gemini-2.5-flash",
            stage2_fallbacks=["claude-haiku-4-5-20251001", "gpt-4.1-mini"],
        )
        df = make_sample_df(1)
        result = cf.filter_jobs(df)

        # Fallback should have been tried
        assert "claude-haiku-4-5-20251001" in call_log
        # gpt-4.1-mini should NOT (claude succeeded)
        assert "gpt-4.1-mini" not in call_log
        # Job should be HIGH_MATCH from the fallback
        high = result[result["match_level"] == "HIGH_MATCH"]
        assert len(high) == 1

    def test_stage3_fallback_on_primary_exception(self, monkeypatch):
        """Stage 3: primary model throws → fallback returns valid JSON."""
        call_log = []

        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            call_log.append(model)
            # Stage 1
            if "YES or NO" in prompt:
                return "YES"
            # Stage 2 → returns POTENTIAL_MATCH to trigger Stage 3
            if "CANDIDATE PROFILE" not in prompt or "JUNIOR REVIEWER" not in prompt:
                return "Job 1: POTENTIAL MATCH - borderline"
            # Stage 3
            if model == "gemini-3-pro-preview":
                raise TimeoutError("Gemini is overloaded")
            if model == "gemini-2.5-pro":
                raise TimeoutError("Also overloaded")
            if model == "claude-sonnet-4-6":
                import json as _json
                return _json.dumps({
                    "final_level": "HIGH_MATCH",
                    "reason": "Claude says great fit",
                    "apply_priority": 9,
                })
            return ""

        monkeypatch.setattr("job_agent.cascade_filter.call_llm", mock_llm)

        cf = CascadeFilter(
            PROFILE_TEXT,
            stage3_model="gemini-3-pro-preview",
            stage3_fallbacks=["gemini-2.5-pro", "claude-sonnet-4-6", "gpt-4.1"],
        )
        df = make_sample_df(1)
        result = cf.filter_jobs(df)

        # Primary + first fallback failed, second fallback (claude) should have worked
        assert "gemini-3-pro-preview" in call_log
        assert "gemini-2.5-pro" in call_log
        assert "claude-sonnet-4-6" in call_log
        assert "gpt-4.1" not in call_log  # Not needed

        high = result[result["match_level"] == "HIGH_MATCH"]
        assert len(high) == 1
        assert "Claude says great fit" in high.iloc[0]["reason"]

    def test_fallback_skips_primary_model_in_list(self, monkeypatch):
        """If the primary model appears in fallback list, it's skipped (no double-try)."""
        call_log = []

        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            call_log.append(model)
            if model == "ollama/qwen3:8b":
                return ""  # Always fails
            return "YES"

        monkeypatch.setattr("job_agent.cascade_filter.call_llm", mock_llm)

        cf = CascadeFilter(
            PROFILE_TEXT,
            stage1_model="ollama/qwen3:8b",
            stage1_fallbacks=["ollama/qwen3:8b", "ollama/gemma3:12b"],
        )
        df = make_sample_df(1)
        cf.run_stage1(df)

        # qwen3:8b should appear only once (primary), not twice
        qwen_calls = [m for m in call_log if m == "ollama/qwen3:8b"]
        assert len(qwen_calls) == 1

    def test_empty_fallback_list(self, monkeypatch):
        """With empty fallback list, failure in primary → pass-through (Stage 1 behavior)."""
        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            return ""  # Primary returns empty

        monkeypatch.setattr("job_agent.cascade_filter.call_llm", mock_llm)

        cf = CascadeFilter(
            PROFILE_TEXT,
            stage1_model="ollama/qwen3:8b",
            stage1_fallbacks=[],
        )
        df = make_sample_df(1)
        passed_df, filtered_df = cf.run_stage1(df)

        # All models exhausted → RuntimeError caught → pass through
        assert len(passed_df) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TestTiering — the worth_a_look tier (recall pass)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTiering:
    def test_assign_tier_mapping(self):
        assert assign_tier("HIGH_MATCH") == "strong"
        assert assign_tier("USER_INPUT") == "strong"
        assert assign_tier("POTENTIAL_MATCH") == "worth_a_look"
        assert assign_tier("WORTH_A_LOOK") == "worth_a_look"
        for reject in ("IRRELEVANT", "SOCIAL_TRAP", "JUNIOR_WEAK", "FILTERED_OUT", "DUPLICATE", None):
            assert assign_tier(reject) is None

    def test_stage2_parses_worth_a_look(self, monkeypatch):
        """Stage 2 'WORTH A LOOK' keeps its verdict WITHOUT a Stage-3 Opus trip
        (cost: judging the deliberately-loose tier almost never changed it).
        It travels in rejected_df (historical name) but stays relevant."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw: "Job 1: WORTH A LOOK - adjacent stretch role",
        )
        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(1)
        high_df, rejected_df, potential_df = cf.run_stage2(df)
        assert len(high_df) == 0
        assert len(potential_df) == 0          # NOT routed to the Opus judge
        assert len(rejected_df) == 1
        assert rejected_df.iloc[0]["match_level"] == "WORTH_A_LOOK"
        assert rejected_df.iloc[0]["gpt_relevance"] == "relevant"  # still surfaced

    def test_filter_jobs_stamps_tier_column(self, monkeypatch):
        """End-to-end: filter_jobs returns a `tier` column; strong/worth_a_look/None
        align with each row's final match_level."""

        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            if "YES or NO" in prompt:
                return "YES"  # everything passes Stage 1
            if "Executive Recruiter" in prompt:  # Stage 2 batch
                return (
                    "Job 1: HIGH MATCH - perfect fit\n"
                    "Job 2: WORTH A LOOK - adjacent\n"
                    "Job 3: IRRELEVANT - wrong field"
                )
            if "final_level" in prompt:  # Stage 3 judge (runs on the WORTH/POTENTIAL row)
                return json.dumps({
                    "final_level": "CONFIRMED_POTENTIAL",
                    "reason": "kept",
                    "apply_priority": 5,
                })
            return "YES"

        monkeypatch.setattr("job_agent.cascade_filter.call_llm", mock_llm)

        cf = CascadeFilter(PROFILE_TEXT)
        df = make_sample_df(3)  # Senior Video Producer, Software Engineer, Content Producer
        result = cf.filter_jobs(df)

        # Stage 1 runs in a ThreadPoolExecutor (as_completed), so row order into
        # Stage 2 is non-deterministic — assert on the tier distribution, not order.
        assert "tier" in result.columns
        counts = result["tier"].fillna("none").value_counts().to_dict()
        assert counts.get("strong") == 1          # the HIGH MATCH row
        assert counts.get("worth_a_look") == 1    # the WORTH A LOOK row (Stage 3-confirmed)
        assert counts.get("none") == 1            # the IRRELEVANT row (stored, untiered)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TestThinDescriptionRescue — Perplexity Stage 1->2 enrichment
# ═══════════════════════════════════════════════════════════════════════════════


class TestThinDescriptionRescue:
    def test_noop_when_perplexity_disabled(self):
        """With Perplexity off (default — no API key), enrichment is a pure no-op."""
        cf = CascadeFilter(PROFILE_TEXT, db=object())
        df = pd.DataFrame([
            {"title": "Producer", "company": "X", "location": "Seattle", "description": "short stub"},
        ])
        out = cf._enrich_thin_descriptions(df)
        assert out.iloc[0]["description"] == "short stub"

    def test_enriches_only_thin_descriptions(self, monkeypatch):
        """When enabled, only descriptions below the threshold get role context."""
        import job_agent.cascade_filter as cfmod
        import job_agent.perplexity_utils as pxmod
        monkeypatch.setattr(cfmod, "USE_PERPLEXITY", True)
        monkeypatch.setattr(
            pxmod, "enrich_job_description",
            lambda title, company, location="", db=None: "ENRICHED CONTEXT",
        )
        cf = CascadeFilter(PROFILE_TEXT, db=object())
        df = pd.DataFrame([
            {"title": "Thin Producer", "company": "X", "location": "Seattle", "description": "stub"},
            {"title": "Fat Producer", "company": "Y", "location": "Seattle", "description": "x" * 300},
        ])
        out = cf._enrich_thin_descriptions(df)
        thin = out[out["title"] == "Thin Producer"].iloc[0]["description"]
        fat = out[out["title"] == "Fat Producer"].iloc[0]["description"]
        assert "ENRICHED CONTEXT" in thin
        assert "ENRICHED CONTEXT" not in fat


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TestStage2Resilience — anchored verdict parsing + retryable STAGE2_PENDING
# ═══════════════════════════════════════════════════════════════════════════════


class TestStage2Resilience:
    def test_reasoning_text_cannot_flip_verdict(self, monkeypatch):
        """Earliest-token parsing: 'IRRELEVANT - ...not a high match...' must NOT
        become HIGH_MATCH (the old anywhere-substring parser flipped it)."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw:
                "Job 1: IRRELEVANT - This is not a high match because the role is demand-gen marketing.",
        )
        cf = CascadeFilter(PROFILE_TEXT)
        high_df, rejected_df, potential_df = cf.run_stage2(make_sample_df(1))
        assert len(high_df) == 0
        assert len(rejected_df) == 1
        assert rejected_df.iloc[0]["match_level"] == "IRRELEVANT"

    def test_batch_error_marks_retryable_not_irrelevant(self, monkeypatch):
        """All-model failure must NOT bury jobs as IRRELEVANT — STAGE2_PENDING
        keeps relevance NULL (ungraded, recoverable by the manual
        process_pending_db_jobs re-grade)."""
        def boom(prompt, model="", temperature=0.0, **_kw):
            raise ConnectionError("network down")
        monkeypatch.setattr("job_agent.cascade_filter.call_llm", boom)

        cf = CascadeFilter(PROFILE_TEXT, stage2_fallbacks=[])
        high_df, rejected_df, potential_df = cf.run_stage2(make_sample_df(2))
        assert len(rejected_df) == 2
        assert set(rejected_df["match_level"]) == {"STAGE2_PENDING"}
        assert rejected_df["gpt_relevance"].isna().all()  # NULL -> retryable

    def test_missing_verdict_line_marks_retryable(self, monkeypatch):
        """A response missing a job's line leaves that job STAGE2_PENDING (after
        one in-run re-grade that skips it again)."""
        calls = []

        def fake(prompt, model="", temperature=0.0, **_kw):
            calls.append(prompt)
            if "Looking for a senior video producer" in prompt:  # job 1's JD
                return "Job 1: HIGH MATCH - great fit"
            return "I could not grade this listing."
        monkeypatch.setattr("job_agent.cascade_filter.call_llm", fake)
        cf = CascadeFilter(PROFILE_TEXT)
        high_df, rejected_df, potential_df = cf.run_stage2(make_sample_df(2))
        assert len(high_df) == 1
        assert len(rejected_df) == 1
        assert rejected_df.iloc[0]["match_level"] == "STAGE2_PENDING"
        assert len(calls) == 2  # the batch + one re-grade of the skipped job

    def test_underscore_variants_parse(self, monkeypatch):
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            lambda prompt, model="", temperature=0.0, **_kw:
                "Job 1: WORTH_A_LOOK - adjacent\nJob 2: JUNIOR/WEAK - too junior",
        )
        cf = CascadeFilter(PROFILE_TEXT)
        high_df, rejected_df, potential_df = cf.run_stage2(make_sample_df(2))
        levels = set(rejected_df["match_level"])  # both bypass the Stage-3 judge
        assert levels == {"WORTH_A_LOOK", "JUNIOR_WEAK"}


class TestStage3Cap:
    def test_overflow_keeps_stage2_verdict_and_skips_judge(self, monkeypatch):
        """Beyond CASCADE_STAGE3_MAX_JOBS, POTENTIAL jobs keep their Stage-2
        verdict (still surfaced as worth_a_look) and never hit the judge."""
        import job_agent.config as cfg
        monkeypatch.setattr(cfg, "CASCADE_STAGE3_MAX_JOBS", 2)
        judge_calls = []

        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            if "YES or NO" in prompt:
                return "YES"
            if "Executive Recruiter" in prompt:
                # all five jobs POTENTIAL
                return "\n".join(f"Job {i}: POTENTIAL MATCH - fine" for i in range(1, 6))
            if "final_level" in prompt:
                judge_calls.append(1)
                return '{"final_level": "CONFIRMED_POTENTIAL", "reason": "ok", "apply_priority": 5}'
            return "YES"

        monkeypatch.setattr("job_agent.cascade_filter.call_llm", mock_llm)
        cf = CascadeFilter(PROFILE_TEXT)
        result = cf.filter_jobs(make_sample_df(5))

        assert len(judge_calls) == 2                      # capped at 2 premium calls
        pot = result[result["match_level"] == "POTENTIAL_MATCH"]
        assert len(pot) == 5                              # nothing lost
        assert set(pot["tier"]) == {"worth_a_look"}       # all still surfaced
        judged = pot[pot["reason"].str.startswith("Stage 3")]
        assert len(judged) == 2                           # only the top slice was judged


class TestSecondaryMetroLocationPolicy:
    """The profile's secondary metro (Portland for the fixture profile) is the
    capped one: on-site roles there never reach the strong tier."""

    def test_secondary_metro_onsite_high_capped_at_worth_a_look(self, monkeypatch):
        """A secondary-metro on-site HIGH match is tier-demoted to worth_a_look;
        a primary-metro HIGH and a remote secondary-metro HIGH stay strong."""
        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            if "YES or NO" in prompt:
                return "YES"
            if "Executive Recruiter" in prompt:
                return "\n".join(f"Job {i}: HIGH MATCH - great" for i in range(1, 4))
            return '{"final_level": "CONFIRMED_POTENTIAL", "reason": "x", "apply_priority": 5}'
        monkeypatch.setattr("job_agent.cascade_filter.call_llm", mock_llm)

        df = pd.DataFrame([
            {"title": "Producer", "company": "A", "location": "Seattle, WA",
             "date_posted": "", "description": "d"},
            {"title": "Producer II", "company": "B", "location": "Beaverton, OR",
             "date_posted": "", "description": "d"},
            {"title": "Producer III", "company": "C", "location": "Portland, OR (Remote)",
             "date_posted": "", "description": "d"},
        ])
        cf = CascadeFilter(PROFILE_TEXT)
        result = cf.filter_jobs(df)
        by_co = {r["company"]: r for _, r in result.iterrows()}
        assert by_co["A"]["tier"] == "strong"                 # primary metro
        assert by_co["B"]["tier"] == "worth_a_look"           # secondary-metro on-site: demoted
        assert "location priority" in by_co["B"]["reason"]
        assert by_co["C"]["tier"] == "strong"                 # remote beats the cap

    def test_secondary_metro_onsite_potential_skips_judge(self, monkeypatch):
        """Secondary-metro on-site POTENTIALs never reach the premium Stage-3 judge."""
        judge_calls = []
        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            if "YES or NO" in prompt:
                return "YES"
            if "Executive Recruiter" in prompt:
                return "Job 1: POTENTIAL MATCH - ok"
            judge_calls.append(1)
            return '{"final_level": "CONFIRMED_POTENTIAL", "reason": "x", "apply_priority": 5}'
        monkeypatch.setattr("job_agent.cascade_filter.call_llm", mock_llm)

        df = pd.DataFrame([{"title": "Producer", "company": "X", "location": "Hillsboro, OR",
                            "date_posted": "", "description": "d"}])
        cf = CascadeFilter(PROFILE_TEXT)
        result = cf.filter_jobs(df)
        assert judge_calls == []                              # no premium call
        assert result.iloc[0]["tier"] == "worth_a_look"       # still surfaced


class TestSeniorityPolicy:
    def _mock(self, stage2, stage3):
        """Cascade LLM stub. Capital 'Executive Recruiter' = Stage-2 grading
        prompt; the lowercase 'executive recruiter' Stage-3 judge falls through."""
        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            if "YES or NO" in prompt:
                return "YES"
            if "Executive Recruiter" in prompt:
                return stage2
            return stage3
        return mock_llm

    def test_confirmed_people_manager_rejected(self, monkeypatch):
        """A POTENTIAL the Opus judge identifies as a confirmed people-manager is
        REJECTED outright — it drops from BOTH tiers (tier None), not capped."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            self._mock("Job 1: POTENTIAL MATCH - maybe",
                       '{"final_level": "REJECT", "reason": "manages a team of 6, hires", "apply_priority": 2}'))
        df = pd.DataFrame([{"title": "Production Manager", "company": "X", "location": "Remote",
                            "date_posted": "", "description": "manage a team of 6, hire, run performance reviews"}])
        result = CascadeFilter(PROFILE_TEXT).filter_jobs(df)
        assert pd.isna(result.iloc[0]["tier"])
        assert result.iloc[0]["match_level"] == "IRRELEVANT"

    def test_ambiguous_leadership_capped_to_worth_a_look(self, monkeypatch):
        """An over-leveled Director with no report evidence stays VISIBLE in the
        lower tier (judge -> WORTH_A_LOOK), never rejected on the title alone."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            self._mock("Job 1: POTENTIAL MATCH - over-leveled",
                       '{"final_level": "WORTH_A_LOOK", "reason": "inflated title, hands-on", "apply_priority": 4}'))
        df = pd.DataFrame([{"title": "Director, Content", "company": "X", "location": "Remote",
                            "date_posted": "", "description": "hands-on producer, owns schedule"}])
        result = CascadeFilter(PROFILE_TEXT).filter_jobs(df)
        assert result.iloc[0]["tier"] == "worth_a_look"

    def test_hands_on_producer_stays_strong(self, monkeypatch):
        """A hands-on Producer role rated HIGH by Stage 2 reaches strong
        untouched — no deterministic seniority pass can demote it."""
        monkeypatch.setattr(
            "job_agent.cascade_filter.call_llm",
            self._mock("Job 1: HIGH MATCH - clean fit",
                       '{"final_level": "HIGH_MATCH", "reason": "x", "apply_priority": 8}'))
        df = pd.DataFrame([{"title": "Producer", "company": "Harbor Lights Media", "location": "Remote",
                            "date_posted": "", "description": "d"}])
        result = CascadeFilter(PROFILE_TEXT).filter_jobs(df)
        assert result.iloc[0]["tier"] == "strong"

    def test_leadership_potential_now_reaches_judge(self, monkeypatch):
        """Leadership-titled POTENTIALs are NO LONGER skipped — the Opus judge
        runs so it can decide reject-vs-cap (the regex skip was removed)."""
        judge_calls = []
        def mock_llm(prompt, model="", temperature=0.0, **_kw):
            if "YES or NO" in prompt:
                return "YES"
            if "Executive Recruiter" in prompt:
                return "Job 1: POTENTIAL MATCH - ok"
            judge_calls.append(1)
            return '{"final_level": "WORTH_A_LOOK", "reason": "x", "apply_priority": 4}'
        monkeypatch.setattr("job_agent.cascade_filter.call_llm", mock_llm)
        df = pd.DataFrame([{"title": "Head of Production", "company": "X",
                            "location": "Remote", "date_posted": "", "description": "d"}])
        cf = CascadeFilter(PROFILE_TEXT)
        result = cf.filter_jobs(df)
        assert judge_calls == [1]
        assert result.iloc[0]["tier"] == "worth_a_look"

    def test_user_input_exempt(self):
        """Manually-submitted jobs are never seniority-touched."""
        cf = CascadeFilter(PROFILE_TEXT)
        df = pd.DataFrame([{
            "title": "Director of Production", "company": "X", "location": "Remote",
            "match_level": "USER_INPUT", "confidence": 1.0, "reason": "manual",
        }])
        out = cf._finalize([df])
        assert out.iloc[0]["tier"] == "strong"

    # --- the policy now lives entirely in the prompts (no deterministic regex) ---
    def test_no_deterministic_seniority_regex(self):
        """The brittle title regex was removed (2026-06-20) — seniority is fully
        judge-arbitrated. Guard against silently re-introducing a backstop."""
        import job_agent.config as cfg
        assert not hasattr(cfg, "is_senior_leadership_title")
        assert not hasattr(cfg, "SENIOR_TITLE_RE")

    def test_stage2_prompts_reject_confirmed_people_managers(self):
        """Both Stage-2 prompts must REJECT (not merely cap) confirmed people-
        management — a WORTH_A_LOOK verdict bypasses Stage 3, so the reject has
        to be possible at Stage 2 too."""
        from job_agent.gpt_filter import (
            DEFAULT_PROMPT_TEMPLATE, STAGE2_PROMPT_CLAUDE)
        for p in (DEFAULT_PROMPT_TEMPLATE, STAGE2_PROMPT_CLAUDE):
            assert "PEOPLE-MANAGEMENT TEST" in p
            assert "direct reports" in p
            assert "REJECT" in p

    def test_stage3_prompts_reject_confirmed_people_managers(self):
        """Both Stage-3 judge prompts gate confirmed people-management to REJECT
        and route ambiguous leadership titles to WORTH_A_LOOK."""
        from job_agent.config import (
            CASCADE_STAGE3_PROMPT, CASCADE_STAGE3_PROMPT_CLAUDE)
        for p in (CASCADE_STAGE3_PROMPT, CASCADE_STAGE3_PROMPT_CLAUDE):
            assert "PEOPLE-MANAGEMENT TEST" in p
            assert "final_level = REJECT" in p
            assert "WORTH_A_LOOK" in p

    def test_all_prompts_reject_no_evidence_directors(self):
        """A Director title with an empty/stub description cannot reach reading
        (c) and must not slide through the general ambiguity branch: with no
        description there is no evidence the role is hands-on."""
        from job_agent.gpt_filter import (
            DEFAULT_PROMPT_TEMPLATE, STAGE2_PROMPT_CLAUDE)
        from job_agent.config import (
            CASCADE_STAGE3_PROMPT, CASCADE_STAGE3_PROMPT_CLAUDE)
        for p in (DEFAULT_PROMPT_TEMPLATE, STAGE2_PROMPT_CLAUDE,
                  CASCADE_STAGE3_PROMPT, CASCADE_STAGE3_PROMPT_CLAUDE):
            assert "CANNOT reach (c)" in p
            assert "does NOT apply to Director titles" in p

    def test_all_prompts_carry_marketing_events_wall(self):
        """Corporate/field-marketing event roles are a different profession —
        every cascade prompt must carry the MARKETING-EVENTS WALL, including
        the explicit clause that adjacent production experience does
        not rescue them."""
        from job_agent.gpt_filter import (
            DEFAULT_PROMPT_TEMPLATE, STAGE2_PROMPT_CLAUDE)
        from job_agent.config import (
            CASCADE_STAGE3_PROMPT, CASCADE_STAGE3_PROMPT_CLAUDE)
        for p in (DEFAULT_PROMPT_TEMPLATE, STAGE2_PROMPT_CLAUDE,
                  CASCADE_STAGE3_PROMPT, CASCADE_STAGE3_PROMPT_CLAUDE):
            assert "MARKETING-EVENTS WALL" in p
            assert "Field Marketing" in p
            assert "Meeting Planner" in p
            assert "Design Program Manager" in p
            assert "does NOT rescue" in p  # softer-neighbor-rule-does-not-apply clause


class TestCompanyScrutinyCaution:
    """Feedback loop: repeatedly-dismissed companies get a judge caution."""

    class _FakeDB:
        def __init__(self, counts):
            self._counts = counts
        def get_dismissed_company_counts(self, min_count=2):
            return {k: v for k, v in self._counts.items() if v >= min_count}

    def test_caution_for_repeatedly_dismissed_company(self):
        cf = CascadeFilter(PROFILE_TEXT, db=self._FakeDB({"acme media": 3}))
        note = cf._company_caution("Acme Media")
        assert "CAUTION" in note and "3" in note

    def test_no_caution_for_clean_company(self):
        cf = CascadeFilter(PROFILE_TEXT, db=self._FakeDB({"acme media": 3}))
        assert cf._company_caution("Fresh Studio") == ""

    def test_no_caution_without_db(self):
        cf = CascadeFilter(PROFILE_TEXT)
        assert cf._company_caution("Acme Media") == ""
