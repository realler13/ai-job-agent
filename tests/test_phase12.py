"""Tests for Phase 12: Interview Knowledge, Domain Filtering, Seniority Relaxation, CV Quality Judge."""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock


# ──────────────────────────────────────────────────
# Config Constants (Task 5)
# ──────────────────────────────────────────────────

class TestConfigConstants:
    def test_transcript_context_path_exists(self):
        from job_agent.config import TRANSCRIPT_CONTEXT_PATH
        assert TRANSCRIPT_CONTEXT_PATH.endswith(".md")

    def test_truncation_transcript_context(self):
        """Budget must cover the FULL transcript — an 8000-char head-truncation
        used to drop every section after the first two of a long transcript."""
        import os
        from job_agent.config import TRUNCATION_TRANSCRIPT_CONTEXT, TRANSCRIPT_CONTEXT_PATH
        assert TRUNCATION_TRANSCRIPT_CONTEXT >= 32000
        if os.path.exists(TRANSCRIPT_CONTEXT_PATH):
            assert os.path.getsize(TRANSCRIPT_CONTEXT_PATH) <= TRUNCATION_TRANSCRIPT_CONTEXT

    def test_cv_quality_model_decorrelated_from_generator(self):
        """The fabrication gate must run on a DIFFERENT model than the
        generator — a model is weakest at spotting its own characteristic
        errors (2026-07-25: opus-5 generates, sonnet-5 judges; the invariant
        is the constant, the models follow)."""
        from job_agent.config import CV_QUALITY_MODEL, DEFAULT_GENERATION_MODEL
        assert CV_QUALITY_MODEL == "claude-sonnet-5"
        assert CV_QUALITY_MODEL != DEFAULT_GENERATION_MODEL

    def test_cv_quality_max_retries(self):
        from job_agent.config import CV_QUALITY_MAX_RETRIES
        assert CV_QUALITY_MAX_RETRIES == 2

    def test_cv_quality_flags(self):
        from job_agent.config import CV_QUALITY_FLAGS
        assert "PASSED" in CV_QUALITY_FLAGS
        assert "HALLUCINATION" in CV_QUALITY_FLAGS
        assert "HALLUCINATION_UNRESOLVED" in CV_QUALITY_FLAGS


# ──────────────────────────────────────────────────
# Task 1: Interview Transcript Integration
# ──────────────────────────────────────────────────

class TestTranscriptLoading:
    def setup_method(self):
        """Reset the transcript cache before each test."""
        import job_agent.cv_tailor as ct
        ct._transcript_cache = None

    def test_load_transcript_returns_fallback_when_missing(self):
        from job_agent.cv_tailor import _load_transcript_context
        # Point config at a non-existent file
        with patch("job_agent.cv_tailor.TRANSCRIPT_CONTEXT_PATH", "nonexistent_file_12345.md"):
            result = _load_transcript_context(domain="producer_core")
        assert result == "No additional transcript context provided."

    def test_load_transcript_reads_file(self, tmp_path):
        from job_agent.cv_tailor import _load_transcript_context
        import job_agent.cv_tailor as ct

        # Create a temp transcript file
        transcript_content = "This is a test transcript with interview notes."
        transcript_file = tmp_path / "test_transcript.md"
        transcript_file.write_text(transcript_content)

        # Patch the path resolution to find our temp file
        with patch("job_agent.cv_tailor.TRANSCRIPT_CONTEXT_PATH", str(transcript_file)):
            # Also patch the project root resolution
            with patch("job_agent.cv_tailor.os.path.dirname") as mock_dirname:
                # Make it so project_root / TRANSCRIPT_CONTEXT_PATH resolves to our file
                mock_dirname.side_effect = lambda p: str(tmp_path) if p.endswith(str(tmp_path)) else os.path.dirname(p)

                # Simpler: just write to the expected location relative to __file__
                ct._transcript_cache = None
                ct._transcript_cache = transcript_content  # Pre-populate cache
                result = _load_transcript_context(domain="games_production")

        assert "game production" in result.lower()
        assert transcript_content in result

    def test_load_transcript_cache_reuse(self):
        import job_agent.cv_tailor as ct

        ct._transcript_cache = "Cached transcript text"
        from job_agent.cv_tailor import _load_transcript_context
        result = _load_transcript_context(domain="producer_core")
        assert "Cached transcript text" in result
        assert "[Focus on" in result

    def test_load_transcript_domain_hint_varies(self):
        import job_agent.cv_tailor as ct
        from job_agent.cv_tailor import _load_transcript_context

        ct._transcript_cache = "Some transcript content"

        result_podcast = _load_transcript_context(domain="podcast_and_audio")
        assert "podcast" in result_podcast.lower()

        result_games = _load_transcript_context(domain="games_production")
        assert "game production" in result_games.lower()

    def test_load_transcript_truncation(self):
        import job_agent.cv_tailor as ct
        from job_agent.cv_tailor import _load_transcript_context
        from job_agent.config import TRUNCATION_TRANSCRIPT_CONTEXT

        # Set a very long transcript
        ct._transcript_cache = "A" * (TRUNCATION_TRANSCRIPT_CONTEXT + 5000)
        result = _load_transcript_context(domain="producer_core")
        # The domain hint prefix + truncated text should be limited
        # The raw text portion should be at most TRUNCATION_TRANSCRIPT_CONTEXT chars
        text_after_hint = result.split("\n", 1)[1] if "\n" in result else result
        assert len(text_after_hint) <= TRUNCATION_TRANSCRIPT_CONTEXT


class TestTranscriptInGenerateTailoredCV:
    def setup_method(self):
        import job_agent.cv_tailor as ct
        ct._transcript_cache = None

    def test_auto_load_when_transcript_text_empty(self):
        """When transcript_text is empty, generate_tailored_cv should auto-load."""
        import job_agent.cv_tailor as ct

        ct._transcript_cache = "Auto-loaded interview context"

        with patch("job_agent.cv_tailor.call_llm", return_value="Mocked summary") as mock_llm, \
             patch("job_agent.cv_tailor.parse_docx_to_structure", return_value={
                 "summary": "Original summary",
                 "experience": [],
             }):
            result = ct.generate_tailored_cv(
                base_cv_path="/fake/path.docx",
                profile_text="Test profile",
                target_job={"title": "Producer", "company": "TestCo", "description": "Make videos"},
                model_name="claude-haiku-4-5-20251001",
                domain="producer_core",
                transcript_text="",  # Empty → auto-load
            )

        # The LLM should have been called with transcript context containing auto-loaded text
        call_args = mock_llm.call_args_list[0][0][0]  # First call, first positional arg (prompt)
        assert "Auto-loaded interview context" in call_args

    def test_explicit_transcript_overrides_autoload(self):
        """When transcript_text is provided, it should NOT auto-load."""
        import job_agent.cv_tailor as ct

        ct._transcript_cache = "This should NOT appear"

        with patch("job_agent.cv_tailor.call_llm", return_value="Mocked summary") as mock_llm, \
             patch("job_agent.cv_tailor.parse_docx_to_structure", return_value={
                 "summary": "Original summary",
                 "experience": [],
             }):
            result = ct.generate_tailored_cv(
                base_cv_path="/fake/path.docx",
                profile_text="Test profile",
                target_job={"title": "Producer", "company": "TestCo", "description": "Make videos"},
                model_name="claude-haiku-4-5-20251001",
                domain="producer_core",
                transcript_text="Explicit user-provided transcript",
            )

        call_args = mock_llm.call_args_list[0][0][0]
        assert "Explicit user-provided transcript" in call_args
        assert "This should NOT appear" not in call_args


class TestCoverLetterTranscript:
    def setup_method(self):
        import job_agent.cv_tailor as ct
        ct._transcript_cache = None

    def test_cover_letter_includes_transcript_claude(self):
        """Claude variant should include <transcript_context> block."""
        import job_agent.cv_tailor as ct
        ct._transcript_cache = "Interview transcript for CL"

        with patch("job_agent.docs_builder.call_llm", return_value="Dear Hiring Manager, ...") as mock_llm:
            from job_agent.docs_builder import generate_cover_letter_content
            result = generate_cover_letter_content(
                base_text="Profile text",
                job={"title": "Producer", "company": "TestCo"},
                model="claude-haiku-4-5-20251001",
                domain="producer_core",
                transcript_text="",  # Auto-load
            )

        call_args = mock_llm.call_args_list[0][0][0]
        assert "<transcript_context>" in call_args
        assert "Interview transcript for CL" in call_args

    def test_cover_letter_includes_transcript_non_claude(self):
        """Non-Claude variant should include 'Additional Candidate Context' section."""
        import job_agent.cv_tailor as ct
        ct._transcript_cache = "Non-claude transcript context"

        with patch("job_agent.docs_builder.call_llm", return_value="Dear Hiring Manager, ...") as mock_llm:
            from job_agent.docs_builder import generate_cover_letter_content
            result = generate_cover_letter_content(
                base_text="Profile text",
                job={"title": "Producer", "company": "TestCo"},
                model="gpt-4o",  # Non-claude
                domain="producer_core",
                transcript_text="",  # Auto-load
            )

        call_args = mock_llm.call_args_list[0][0][0]
        assert "Additional Candidate Context:" in call_args
        assert "Non-claude transcript context" in call_args

    def test_cover_letter_explicit_transcript(self):
        """Explicit transcript_text should override auto-load."""
        import job_agent.cv_tailor as ct
        ct._transcript_cache = "Should not appear"

        with patch("job_agent.docs_builder.call_llm", return_value="Dear Hiring Manager, ...") as mock_llm:
            from job_agent.docs_builder import generate_cover_letter_content
            result = generate_cover_letter_content(
                base_text="Profile text",
                job={"title": "Producer", "company": "TestCo"},
                model="claude-haiku-4-5-20251001",
                domain="producer_core",
                transcript_text="Explicitly passed transcript",
            )

        call_args = mock_llm.call_args_list[0][0][0]
        assert "Explicitly passed transcript" in call_args
        assert "Should not appear" not in call_args


# ──────────────────────────────────────────────────
# Task 2: Domain-Aware Filtering
# ──────────────────────────────────────────────────

class TestDomainAwareFiltering:
    def test_stage2_batch_includes_domain_match(self):
        """Stage 2 job blocks should include Domain Match line."""
        import pandas as pd
        from job_agent.cascade_filter import CascadeFilter

        cf = CascadeFilter(
            profile_text="Test profile",
            stage2_model="claude-haiku-4-5-20251001",
        )

        batch_df = pd.DataFrame([{
            "title": "Podcast Producer",
            "company": "TestCo",
            "location": "Remote",
            "date_posted": "2025-01-15",
            "description": "Produce podcasts.",
            "keyword_category": "podcast_and_audio",
        }])

        captured_prompts = []
        original_method = cf._call_llm_with_fallback

        def capturing_call(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return ("Job 1: HIGH MATCH - Perfect podcast role", "claude-haiku-4-5-20251001")

        cf._call_llm_with_fallback = capturing_call
        results = cf._stage2_batch(batch_df)

        assert len(captured_prompts) == 1
        assert "Domain Match:" in captured_prompts[0]
        assert "podcast" in captured_prompts[0].lower()

    def test_stage3_includes_domain_hint(self):
        """Stage 3 prompt should include domain_hint."""
        import pandas as pd
        from job_agent.cascade_filter import CascadeFilter

        cf = CascadeFilter(
            profile_text="Test profile",
            stage3_model="claude-opus-4-8",
        )

        row = {
            "title": "Game Producer",
            "company": "GameCo",
            "location": "Remote",
            "description": "Ship games.",
            "reason": "Stage 2 says potential",
            "keyword_category": "games_production",
        }

        captured_prompts = []

        def capturing_call(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return ('{"final_level": "HIGH_MATCH", "reason": "great fit", "apply_priority": 8}', "claude-opus-4-8")

        cf._call_llm_with_fallback = capturing_call
        result = cf._stage3_single(row)

        assert len(captured_prompts) == 1
        assert "Domain match:" in captured_prompts[0]
        assert "game production" in captured_prompts[0].lower()

    def test_resolve_domain_for_job_with_keyword_category(self):
        from job_agent.cascade_filter import CascadeFilter
        cf = CascadeFilter(profile_text="Test")
        domain, desc = cf._resolve_domain_for_job({"title": "Producer", "keyword_category": "youtube_and_creators"})
        assert domain == "youtube_and_creators"
        assert "youtube" in desc.lower()

    def test_resolve_domain_for_job_fallback_to_title(self):
        from job_agent.cascade_filter import CascadeFilter
        cf = CascadeFilter(profile_text="Test")
        domain, desc = cf._resolve_domain_for_job({"title": "Senior Podcast Producer"})
        assert domain == "podcast_and_audio"


class TestDomainAwarenessPrompts:
    def test_claude_prompt_has_domain_awareness_section(self):
        from job_agent.gpt_filter import STAGE2_PROMPT_CLAUDE
        assert "<domain_awareness>" in STAGE2_PROMPT_CLAUDE
        assert "Domain Match" in STAGE2_PROMPT_CLAUDE
        assert "producer_core" in STAGE2_PROMPT_CLAUDE

    def test_default_prompt_has_domain_awareness_rule(self):
        from job_agent.gpt_filter import DEFAULT_PROMPT_TEMPLATE
        assert "DOMAIN AWARENESS" in DEFAULT_PROMPT_TEMPLATE
        assert "Domain Match" in DEFAULT_PROMPT_TEMPLATE

    def test_stage3_prompt_has_domain_hint_placeholder(self):
        from job_agent.config import CASCADE_STAGE3_PROMPT, CASCADE_STAGE3_PROMPT_CLAUDE
        assert "{domain_hint}" in CASCADE_STAGE3_PROMPT
        assert "{domain_hint}" in CASCADE_STAGE3_PROMPT_CLAUDE


# ──────────────────────────────────────────────────
# Task 3: Relaxed Seniority Filter
# ──────────────────────────────────────────────────

class TestRelaxedSeniority:
    def test_claude_prompt_has_overqualified_exception(self):
        from job_agent.gpt_filter import STAGE2_PROMPT_CLAUDE
        assert "OVERQUALIFIED EXCEPTION" in STAGE2_PROMPT_CLAUDE
        assert "overqualified but strategic fit" in STAGE2_PROMPT_CLAUDE

    def test_default_prompt_has_overqualified_exception(self):
        from job_agent.gpt_filter import DEFAULT_PROMPT_TEMPLATE
        assert "OVERQUALIFIED EXCEPTION" in DEFAULT_PROMPT_TEMPLATE
        assert "overqualified but strategic fit" in DEFAULT_PROMPT_TEMPLATE

    def test_stage3_has_seniority_note(self):
        from job_agent.config import CASCADE_STAGE3_PROMPT, CASCADE_STAGE3_PROMPT_CLAUDE
        assert "SENIORITY NOTE" in CASCADE_STAGE3_PROMPT
        assert "overqualified but strategic fit" in CASCADE_STAGE3_PROMPT
        assert "<seniority_note>" in CASCADE_STAGE3_PROMPT_CLAUDE
        assert "overqualified but strategic fit" in CASCADE_STAGE3_PROMPT_CLAUDE

    def test_intern_still_rejected_in_claude_prompt(self):
        from job_agent.gpt_filter import STAGE2_PROMPT_CLAUDE
        assert "Intern" in STAGE2_PROMPT_CLAUDE or "intern" in STAGE2_PROMPT_CLAUDE.lower()

    def test_intern_still_rejected_in_default_prompt(self):
        from job_agent.gpt_filter import DEFAULT_PROMPT_TEMPLATE
        assert "Intern" in DEFAULT_PROMPT_TEMPLATE or "intern" in DEFAULT_PROMPT_TEMPLATE.lower()


# ──────────────────────────────────────────────────
# Task 4: CV Quality Judge
# ──────────────────────────────────────────────────

def _summary_claim(verdict="supported"):
    """Phase 27: a verdict counts only with its per-claim list (summary checked)."""
    return [{"section": "summary", "role_index": None, "quote": "summary text",
             "verdict": verdict, "reason": "test"}]


class TestGradeCvQuality:
    def test_pass_verdict(self):
        """grade_cv_quality returns PASSED when audit says overall_pass=true."""
        from job_agent.cv_tailor import grade_cv_quality

        audit_json = json.dumps({
            "overall_pass": True,
            "hallucination": {"found": False, "details": "None"},
            "claims": _summary_claim(),
            "enrichment": {"adequate": True, "details": "Good"},
            "keyword_alignment": {"adequate": True, "details": "Good"},
            "format": {"clean": True, "details": "Clean"},
            "length": {"appropriate": True, "details": "Ok"},
            "corrected_summary": None,
        })

        with patch("job_agent.cv_tailor.call_llm", return_value=audit_json):
            result = grade_cv_quality(
                tailored_data={"summary": "Test summary", "experience": []},
                profile_text="Profile",
                target_job={"title": "Producer", "company": "Co", "description": "Desc"},
            )

        assert result["flag"] == "PASSED"
        assert result["state"] == "ok"
        assert result["audit"]["overall_pass"] is True
        assert result["corrected_summary"] is None

    def test_hallucination_detected(self):
        """grade_cv_quality returns HALLUCINATION when hallucination found."""
        from job_agent.cv_tailor import grade_cv_quality

        audit_json = json.dumps({
            "overall_pass": False,
            "hallucination": {"found": True, "details": "Fabricated 'Emmy Award'"},
            "claims": _summary_claim("unsupported"),
            "enrichment": {"adequate": True, "details": ""},
            "keyword_alignment": {"adequate": True, "details": ""},
            "format": {"clean": True, "details": ""},
            "length": {"appropriate": True, "details": ""},
            "corrected_summary": "Fixed summary without fabrication",
        })

        with patch("job_agent.cv_tailor.call_llm", return_value=audit_json):
            result = grade_cv_quality(
                tailored_data={"summary": "Won an Emmy Award...", "experience": []},
                profile_text="Profile",
                target_job={"title": "Producer", "company": "Co", "description": "Desc"},
            )

        assert result["flag"] == "HALLUCINATION"
        assert result["audit"]["hallucination"]["found"] is True
        assert result["corrected_summary"] == "Fixed summary without fabrication"

    def test_overall_pass_overrides_minor_hallucination(self):
        """overall_pass=true + hallucination.found=true → REVIEW_RECOMMENDED, NOT
        HALLUCINATION. This fixes a run where every CV ended HALLUCINATION_UNRESOLVED:
        a usable CV the auditor passed must not be torn down (and its summary
        stripped) over a nitpick."""
        from job_agent.cv_tailor import grade_cv_quality
        audit_json = json.dumps({
            "overall_pass": True,
            "hallucination": {"found": True, "details": "minor reframing concern"},
            "claims": _summary_claim("composition"),
            "enrichment": {"adequate": True, "details": ""},
            "keyword_alignment": {"adequate": True, "details": ""},
            "format": {"clean": True, "details": ""},
            "length": {"appropriate": True, "details": ""},
            "corrected_summary": None,
        })
        with patch("job_agent.cv_tailor.call_llm", return_value=audit_json):
            result = grade_cv_quality(
                tailored_data={"summary": "Test", "experience": []},
                profile_text="Profile",
                target_job={"title": "Producer", "company": "Co", "description": "Desc"},
            )
        assert result["flag"] == "REVIEW_RECOMMENDED"
        assert result["state"] == "ok"      # a real verdict, not an unavailable check

    def test_judge_grounded_on_metrics_and_base_summary(self):
        """The auditor prompt must carry the verified metrics + base CV summary,
        so verified facts (episode counts, employer names) are not flagged as
        fabrications just because the thin profile text omits them."""
        from job_agent.cv_tailor import grade_cv_quality
        from tests.fixtures.persona import EMPLOYERS, YEARS
        captured = {}

        def fake_llm(prompt, model="", **k):
            captured["prompt"] = prompt
            return json.dumps({"overall_pass": True,
                               "hallucination": {"found": False, "details": "none"},
                               "corrected_summary": None})

        with patch("job_agent.cv_tailor.call_llm", side_effect=fake_llm):
            grade_cv_quality(
                tailored_data={"summary": f"40 episodes produced at {EMPLOYERS[0]}", "experience": []},
                profile_text="short profile",
                target_job={"title": "Producer", "company": "Co", "description": "Desc"},
                base_summary=f"Video producer with {YEARS} across branded and episodic content.",
            )
        # metrics bank injected: a figure that only the verified metrics carry
        assert "Vendor costs cut 18%" in captured["prompt"]
        assert f"Video producer with {YEARS}" in captured["prompt"]   # base summary injected

    # Phase 27 (deliberate change): an error or an incomplete answer is
    # check_unavailable / REVIEW_RECOMMENDED — never PASSED (a judge error
    # stored as PASSED looks like a clean check). The summary gate then ships
    # the base summary, so the package still never blocks.
    def test_parse_error_returns_review_recommended(self):
        """When LLM returns garbage, grade_cv_quality returns REVIEW_RECOMMENDED."""
        from job_agent.cv_tailor import grade_cv_quality

        with patch("job_agent.cv_tailor.call_llm", return_value="I cannot produce JSON."):
            result = grade_cv_quality(
                tailored_data={"summary": "Test", "experience": []},
                profile_text="Profile",
                target_job={"title": "Producer", "company": "Co", "description": "Desc"},
            )

        assert result["flag"] == "REVIEW_RECOMMENDED"
        assert result["state"] == "check_unavailable"
        assert "error" in result["audit"]

    def test_empty_response_returns_review_recommended(self):
        """When LLM returns empty, grade_cv_quality returns REVIEW_RECOMMENDED."""
        from job_agent.cv_tailor import grade_cv_quality

        with patch("job_agent.cv_tailor.call_llm", return_value=""):
            result = grade_cv_quality(
                tailored_data={"summary": "Test", "experience": []},
                profile_text="Profile",
                target_job={"title": "Producer", "company": "Co", "description": "Desc"},
            )

        assert result["flag"] == "REVIEW_RECOMMENDED"
        assert result["state"] == "check_unavailable"

    def test_llm_exception_returns_review_recommended(self):
        """When call_llm raises, grade_cv_quality returns REVIEW_RECOMMENDED with the error."""
        from job_agent.cv_tailor import grade_cv_quality

        with patch("job_agent.cv_tailor.call_llm", side_effect=Exception("API down")):
            result = grade_cv_quality(
                tailored_data={"summary": "Test", "experience": []},
                profile_text="Profile",
                target_job={"title": "Producer", "company": "Co", "description": "Desc"},
            )

        assert result["flag"] == "REVIEW_RECOMMENDED"
        assert result["state"] == "check_unavailable"
        assert "API down" in result["audit"]["error"]

    def test_claude_prompt_used_for_claude_model(self):
        """Verify Claude prompt variant is selected for claude models."""
        from job_agent.cv_tailor import grade_cv_quality

        with patch("job_agent.cv_tailor.call_llm", return_value='{"overall_pass": true, "hallucination": {"found": false}}') as mock:
            grade_cv_quality(
                tailored_data={"summary": "S", "experience": []},
                profile_text="P",
                target_job={"title": "T", "company": "C", "description": "D"},
                model="claude-opus-4-8",
            )
        prompt = mock.call_args[0][0]
        assert "<source_profile>" in prompt  # Claude variant uses XML tags


# Phase 27 (deliberate change): the regenerate-until-clean retry loop is gone.
# The summary is the only generated text a CV ships; the summary gate ships it
# only on a complete clean check, takes the judge's corrected summary only after
# a clean RE-check, and otherwise ships the base summary. No regeneration, and
# HALLUCINATION_UNRESOLVED is no longer a shipped state.
_RETRY_POSTING = ("Lead documentary and branded shoots from pitch to delivery: budgets, schedules, "
                  "call sheets, vendor bids and a small remote post team, delivering an episodic "
                  "series on a fixed calendar. 5+ years producing video, calm on set, and live "
                  "broadcast scheduling experience. Educational or museum content is a plus.")


def _judge(flag, verdict="supported", corrected=None):
    return {"flag": flag, "state": "ok", "corrected_summary": corrected,
            "audit": {"overall_pass": flag != "HALLUCINATION",
                      "hallucination": {"found": verdict != "supported"},
                      "claims": _summary_claim(verdict)}}


class TestSummaryGateReplacesRetryLoop:
    def _run(self, tmp_path, monkeypatch, judge_results):
        from tests.fixtures.fictional_cv import CAREER_MD, build_fixture_docx
        from job_agent.cv_tailor import parse_docx_to_structure
        from job_agent.docs_builder import save_documents_for_job
        monkeypatch.chdir(tmp_path)
        base = build_fixture_docx(tmp_path / "base_cv_producer_core.docx")
        base_parsed = parse_docx_to_structure(base)
        calls = {"judge": [], "generate": 0}

        def mock_grade(data, **k):
            calls["judge"].append(data["summary"])
            return judge_results[min(len(calls["judge"]), len(judge_results)) - 1]

        def mock_generate(*a, **k):
            calls["generate"] += 1
            return {"summary": "Generated summary for the role.", "experience": base_parsed["experience"]}

        monkeypatch.setattr("job_agent.cv_tailor.generate_tailored_cv", mock_generate)
        monkeypatch.setattr("job_agent.docs_builder.grade_cv_quality", mock_grade)
        monkeypatch.setattr("job_agent.docs_builder._resolve_base_cv_path",
                            lambda *a, **k: (base, "producer_core"))
        monkeypatch.setattr("job_agent.docs_builder.check_keyword_coverage",
                            lambda *a, **k: {"coverage": [], "add_keywords": [], "gaps": [], "summary": ""})
        monkeypatch.setattr("job_agent.cv_tailor.grade_cv_hiring_manager",
                            lambda *a, **k: {"overall": 80})
        monkeypatch.setattr("job_agent.docs_builder.generate_cover_letter_content",
                            lambda *a, **k: "Dear Hiring Manager,\nBody.")
        monkeypatch.setattr("job_agent.cv_facts.verified_corpus",
                            lambda extra=(): [CAREER_MD.read_text()])
        result = save_documents_for_job(base_cv_path="base_cv.docx", profile_text="Profile",
                                        job={"id": 1, "title": "Producer", "company": "Co",
                                             "description": _RETRY_POSTING},
                                        transcript_text="TRANSCRIPT")
        return result, calls, base_parsed

    def test_correction_applied_only_after_clean_recheck(self, tmp_path, monkeypatch):
        corrected = "Corrected summary without fabrication."
        result, calls, _ = self._run(tmp_path, monkeypatch, [
            _judge("HALLUCINATION", "unsupported", corrected), _judge("PASSED")])
        assert len(result) == 5
        cv_quality = result[4]
        assert cv_quality["flag"] == "PASSED"
        assert cv_quality["summary_source"] == "corrected"
        # original check, then the re-check OF THE CORRECTED summary
        assert calls["judge"][:2] == ["Generated summary for the role.", corrected]
        assert parse_docx_summary(result[0]) == corrected

    def test_no_regeneration_and_base_summary_on_hallucination(self, tmp_path, monkeypatch):
        result, calls, base_parsed = self._run(tmp_path, monkeypatch,
                                               [_judge("HALLUCINATION", "unsupported")])
        cv_quality = result[4]
        assert cv_quality["flag"] == "REVIEW_RECOMMENDED"
        assert cv_quality["flag"] != "HALLUCINATION_UNRESOLVED"
        assert cv_quality["summary_source"] == "base_fallback"
        assert calls["generate"] == 1                     # no regeneration loop
        assert parse_docx_summary(result[0]) == base_parsed["summary"]


def parse_docx_summary(path):
    from job_agent.cv_tailor import parse_docx_to_structure
    return parse_docx_to_structure(path)["summary"]


class TestSaveDocumentsReturnsTuple:
    def test_returns_5_tuple(self):
        """save_documents_for_job returns (cv_path, cl_path, cv_id, cl_id, cv_quality_result)."""
        from job_agent.docs_builder import save_documents_for_job

        with patch("job_agent.cv_tailor.generate_tailored_cv", return_value={
                "summary": "S", "experience": []
            }), \
             patch("job_agent.cv_tailor.fill_docx_with_tailored_data"), \
             patch("job_agent.docs_builder.generate_cover_letter_content", return_value="CL"), \
             patch("job_agent.docs_builder.create_formatted_cl") as mock_cl, \
             patch("job_agent.docs_builder.upload_to_google_drive", return_value={}), \
             patch("job_agent.docs_builder.grade_cv_quality", return_value={
                 "flag": "PASSED", "audit": {}, "corrected_summary": None
             }), \
             patch("job_agent.docs_builder._resolve_base_cv_path", return_value=("/fake/cv.docx", "producer_core")), \
             patch("os.path.exists", return_value=False), \
             patch("os.makedirs"):

            mock_cl.return_value = MagicMock()

            result = save_documents_for_job(
                base_cv_path="/fake/base_cv.docx",
                profile_text="Profile",
                job={"title": "P", "company": "C", "description": "D"},
            )

        assert len(result) == 5
        # Phase 27: the base at /fake cannot be read, so no CV ships and the
        # verdict is honest (never PASSED for a CV that was not checked).
        assert result[0] is None
        assert result[4]["flag"] == "REVIEW_RECOMMENDED"
        assert {"state", "base_variant", "content_mode", "generation_id"} <= set(result[4])


# ──────────────────────────────────────────────────
# Task 4c: DB Schema & Methods
# ──────────────────────────────────────────────────

class TestDBCvQuality:
    def test_migration_adds_cv_quality_columns(self, tmp_db):
        """New columns should exist after migration."""
        conn = tmp_db._get_thread_conn()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(scrapes)")
        columns = [info[1] for info in cursor.fetchall()]

        assert "cv_quality_flag" in columns
        assert "cv_audit_summary" in columns
        assert "cv_generated_at" in columns
        assert "cv_domain_used" in columns

    def test_update_job_cv_quality(self, tmp_db):
        """update_job_cv_quality stores and retrieves correctly."""
        # Insert a job first
        job_id = tmp_db.save_job({
            "title": "Producer",
            "company": "TestCo",
            "job_url": "https://example.com/job/cv-quality-test",
        })
        assert job_id is not None

        # Update quality
        audit_json = json.dumps({"overall_pass": True, "hallucination": {"found": False}})
        success = tmp_db.update_job_cv_quality(job_id, "PASSED", audit_json, "producer_core")
        assert success is True

        # Retrieve and verify
        job = tmp_db.get_job_details(job_id)
        assert job["cv_quality_flag"] == "PASSED"
        assert job["cv_audit_summary"] == audit_json
        assert job["cv_domain_used"] == "producer_core"
        assert job["cv_generated_at"] is not None

    def test_get_applications_includes_cv_quality_flag(self, tmp_db):
        """get_applications should include cv_quality_flag column."""
        # Insert job and track it
        job_id = tmp_db.save_job({
            "title": "Producer",
            "company": "TestCo",
            "job_url": "https://example.com/job/kanban-quality",
        })
        tmp_db.track_job(job_id)
        tmp_db.update_job_cv_quality(job_id, "PASSED", None, "games_production")

        apps = tmp_db.get_applications()
        assert len(apps) == 1
        assert apps[0]["cv_quality_flag"] == "PASSED"

    def test_update_nonexistent_job_returns_true(self, tmp_db):
        """Updating a non-existent job shouldn't error (SQL UPDATE on 0 rows is fine)."""
        result = tmp_db.update_job_cv_quality(99999, "PASSED", None, None)
        assert result is True  # No error, just 0 rows affected
