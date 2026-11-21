"""Tests for domain-based CV variant system (Phase 11)."""

import os
import pytest
from unittest.mock import patch, MagicMock

from tests.fixtures.persona import EMPLOYERS, YEARS


# === resolve_cv_domain() tests ===

class TestResolveCvDomain:
    """Test keyword-to-domain resolution logic."""

    def test_exact_keyword_match_producer_core(self):
        from job_agent.config import resolve_cv_domain
        assert resolve_cv_domain(matched_keyword="Producer") == "producer_core"
        assert resolve_cv_domain(matched_keyword="Video Producer") == "producer_core"
        assert resolve_cv_domain(matched_keyword="Branded Content Producer") == "producer_core"

    def test_exact_keyword_match_live(self):
        from job_agent.config import resolve_cv_domain
        assert resolve_cv_domain(matched_keyword="Live Producer") == "live_and_experiential"
        assert resolve_cv_domain(matched_keyword="Event Producer") == "live_and_experiential"

    def test_exact_keyword_match_ops(self):
        from job_agent.config import resolve_cv_domain
        assert resolve_cv_domain(matched_keyword="Production Manager") == "production_ops_and_management"
        assert resolve_cv_domain(matched_keyword="Studio Operations") == "production_ops_and_management"

    def test_exact_keyword_match_project(self):
        from job_agent.config import resolve_cv_domain
        assert resolve_cv_domain(matched_keyword="Creative Project Manager") == "project_and_program_management"
        assert resolve_cv_domain(matched_keyword="Campaign Producer") == "project_and_program_management"

    def test_exact_keyword_match_podcast(self):
        from job_agent.config import resolve_cv_domain
        assert resolve_cv_domain(matched_keyword="Podcast Producer") == "podcast_and_audio"
        assert resolve_cv_domain(matched_keyword="Audio Producer") == "podcast_and_audio"

    def test_exact_keyword_match_youtube(self):
        from job_agent.config import resolve_cv_domain
        assert resolve_cv_domain(matched_keyword="YouTube Producer") == "youtube_and_creators"
        assert resolve_cv_domain(matched_keyword="Creator Partnerships Manager") == "youtube_and_creators"

    def test_exact_keyword_match_games(self):
        from job_agent.config import resolve_cv_domain
        assert resolve_cv_domain(matched_keyword="Game Producer") == "games_production"
        assert resolve_cv_domain(matched_keyword="Development Producer") == "games_production"

    def test_case_insensitive_keyword(self):
        from job_agent.config import resolve_cv_domain
        assert resolve_cv_domain(matched_keyword="podcast producer") == "podcast_and_audio"
        assert resolve_cv_domain(matched_keyword="GAME PRODUCER") == "games_production"

    def test_title_fuzzy_match(self):
        """When no keyword provided, match against job title."""
        from job_agent.config import resolve_cv_domain
        assert resolve_cv_domain(job_title="Senior Podcast Producer at Harbor Lights Media") == "podcast_and_audio"
        assert resolve_cv_domain(job_title="YouTube Channel Manager") == "youtube_and_creators"
        assert resolve_cv_domain(job_title="Game Producer - AAA Title") == "games_production"

    def test_title_longest_match_wins(self):
        """More specific keyword should win over generic 'Producer'."""
        from job_agent.config import resolve_cv_domain
        # "Podcast Producer" (podcast_and_audio) should win over "Producer" (producer_core)
        assert resolve_cv_domain(job_title="Podcast Producer") == "podcast_and_audio"
        # "Live Producer" should match live_and_experiential
        assert resolve_cv_domain(job_title="Live Producer") == "live_and_experiential"

    def test_fallback_to_producer_core(self):
        """Unknown titles fall back to producer_core."""
        from job_agent.config import resolve_cv_domain
        assert resolve_cv_domain(job_title="Director of Operations") == "producer_core"
        assert resolve_cv_domain() == "producer_core"
        assert resolve_cv_domain(job_title="", matched_keyword=None) == "producer_core"

    def test_keyword_takes_precedence_over_title(self):
        """If keyword matches, use it even if title suggests different domain."""
        from job_agent.config import resolve_cv_domain
        # Title says "game" but keyword says "podcast"
        result = resolve_cv_domain(
            job_title="Game Producer",
            matched_keyword="Podcast Producer"
        )
        assert result == "podcast_and_audio"


# === DOMAIN_CV_MAP completeness ===

class TestDomainCvMap:
    """Verify DOMAIN_CV_MAP covers all keyword categories."""

    def test_all_categories_have_cv_files(self):
        from job_agent.config import KEYWORDS_BY_CATEGORY, DOMAIN_CV_MAP
        for category in KEYWORDS_BY_CATEGORY:
            assert category in DOMAIN_CV_MAP, f"Missing CV file for category: {category}"

    def test_all_map_values_are_docx(self):
        from job_agent.config import DOMAIN_CV_MAP
        for domain, filename in DOMAIN_CV_MAP.items():
            assert filename.endswith(".docx"), f"Expected .docx for {domain}, got {filename}"

    def test_domain_descriptions_complete(self):
        from job_agent.config import DOMAIN_CV_MAP, DOMAIN_DESCRIPTIONS
        for domain in DOMAIN_CV_MAP:
            assert domain in DOMAIN_DESCRIPTIONS, f"Missing description for domain: {domain}"


# === KEYWORD_TO_CATEGORY reverse lookup ===

class TestKeywordToCategory:
    """Verify the reverse-lookup dict."""

    def test_all_keywords_mapped(self):
        from job_agent.config import KEYWORDS, KEYWORD_TO_CATEGORY
        for kw in KEYWORDS:
            assert kw.lower() in KEYWORD_TO_CATEGORY, f"Keyword not in reverse map: {kw}"

    def test_no_empty_categories(self):
        from job_agent.config import KEYWORD_TO_CATEGORY, KEYWORDS_BY_CATEGORY
        for cat in KEYWORDS_BY_CATEGORY:
            # At least one keyword should map to this category
            found = any(v == cat for v in KEYWORD_TO_CATEGORY.values())
            assert found, f"No keywords map to category: {cat}"


# === _resolve_base_cv_path tests ===

class TestResolveBaseCvPath:
    """Test the domain CV file resolver in docs_builder."""

    def test_resolves_to_domain_file(self, tmp_path):
        from job_agent.docs_builder import _resolve_base_cv_path

        # Create a fake domain CV file
        domain_cv = tmp_path / "base_cv_podcast_audio.docx"
        domain_cv.write_text("fake")
        generic_cv = tmp_path / "base_cv.docx"
        generic_cv.write_text("fake")

        job = {"title": "Podcast Producer", "company": "PodCo"}
        resolved, domain = _resolve_base_cv_path(str(generic_cv), job)

        assert domain == "podcast_and_audio"
        assert resolved == str(domain_cv)

    def test_falls_back_to_generic_when_domain_file_missing(self, tmp_path):
        from job_agent.docs_builder import _resolve_base_cv_path

        generic_cv = tmp_path / "base_cv.docx"
        generic_cv.write_text("fake")

        job = {"title": "Game Producer", "company": "GameCo"}
        resolved, domain = _resolve_base_cv_path(str(generic_cv), job)

        assert domain == "games_production"
        assert resolved == str(generic_cv)

    def test_uses_keyword_category_from_job(self, tmp_path):
        from job_agent.docs_builder import _resolve_base_cv_path

        domain_cv = tmp_path / "base_cv_youtube_creators.docx"
        domain_cv.write_text("fake")
        generic_cv = tmp_path / "base_cv.docx"
        generic_cv.write_text("fake")

        job = {
            "title": "Senior Manager",  # title alone wouldn't match youtube
            "keyword_category": "youtube_and_creators",
        }
        resolved, domain = _resolve_base_cv_path(str(generic_cv), job)

        assert domain == "youtube_and_creators"
        assert resolved == str(domain_cv)


# === CV Tailor domain-aware prompts ===

class TestCvTailorDomainPrompts:
    """Test that domain context is injected into prompts."""

    def test_build_domain_context(self):
        from job_agent.cv_tailor import _build_domain_context
        ctx = _build_domain_context("games_production")
        assert "games_production" in ctx
        assert "game production" in ctx

    def test_build_domain_context_unknown_domain(self):
        from job_agent.cv_tailor import _build_domain_context
        ctx = _build_domain_context("nonexistent_domain")
        assert "general production" in ctx

    def test_summary_prompt_has_domain_placeholder(self):
        from job_agent.cv_tailor import PROMPT_SUMMARY_TEMPLATE, PROMPT_SUMMARY_TEMPLATE_CLAUDE
        assert "{domain_context}" in PROMPT_SUMMARY_TEMPLATE
        assert "{domain_context}" in PROMPT_SUMMARY_TEMPLATE_CLAUDE

    def test_experience_prompt_has_domain_placeholder(self):
        from job_agent.cv_tailor import PROMPT_EXPERIENCE_TEMPLATE, PROMPT_EXPERIENCE_TEMPLATE_CLAUDE
        assert "{domain_context}" in PROMPT_EXPERIENCE_TEMPLATE
        assert "{domain_context}" in PROMPT_EXPERIENCE_TEMPLATE_CLAUDE

    def test_summary_prompt_has_transcript_placeholder(self):
        from job_agent.cv_tailor import PROMPT_SUMMARY_TEMPLATE, PROMPT_SUMMARY_TEMPLATE_CLAUDE
        assert "{transcript_context}" in PROMPT_SUMMARY_TEMPLATE
        assert "{transcript_context}" in PROMPT_SUMMARY_TEMPLATE_CLAUDE

    def test_experience_prompt_has_transcript_placeholder(self):
        from job_agent.cv_tailor import PROMPT_EXPERIENCE_TEMPLATE, PROMPT_EXPERIENCE_TEMPLATE_CLAUDE
        assert "{transcript_context}" in PROMPT_EXPERIENCE_TEMPLATE
        assert "{transcript_context}" in PROMPT_EXPERIENCE_TEMPLATE_CLAUDE

    def test_prompts_have_anti_hallucination_guardrails(self):
        from job_agent.cv_tailor import (
            PROMPT_SUMMARY_TEMPLATE, PROMPT_SUMMARY_TEMPLATE_CLAUDE,
            PROMPT_EXPERIENCE_TEMPLATE, PROMPT_EXPERIENCE_TEMPLATE_CLAUDE,
        )
        for prompt in [PROMPT_SUMMARY_TEMPLATE, PROMPT_SUMMARY_TEMPLATE_CLAUDE,
                       PROMPT_EXPERIENCE_TEMPLATE, PROMPT_EXPERIENCE_TEMPLATE_CLAUDE]:
            assert "fabricate" in prompt.lower() or "invent" in prompt.lower()


# === generate_tailored_cv domain parameter ===

class TestGenerateTailoredCvDomain:
    """Test that generate_tailored_cv passes domain through correctly."""

    def test_accepts_domain_parameter(self, monkeypatch, tmp_path):
        from job_agent.cv_tailor import generate_tailored_cv

        # Create a minimal DOCX
        from docx import Document
        doc = Document()
        doc.add_paragraph("PROFESSIONAL SUMMARY")
        doc.add_paragraph(f"A producer with {YEARS} of experience.")
        doc.add_paragraph("EXPERIENCE")
        doc.add_paragraph("Content Producer")
        p = doc.add_paragraph("Managed video production pipeline", style="List Bullet")
        doc.add_paragraph("SKILLS")
        cv_path = str(tmp_path / "test_cv.docx")
        doc.save(cv_path)

        calls = []
        def mock_call_llm(prompt, model="test", temperature=0.0):
            calls.append(prompt)
            return "Tailored text for the role."

        monkeypatch.setattr("job_agent.cv_tailor.call_llm", mock_call_llm)

        result = generate_tailored_cv(
            cv_path,
            "Profile text here",
            {"title": "Game Producer", "company": "GameCo", "description": "Ship games"},
            model_name="claude-opus-4-8",
            domain="games_production",
        )

        assert result is not None
        assert len(calls) >= 1
        # Check that domain context was injected into prompts
        assert "games_production" in calls[0]

    def test_transcript_text_injected(self, monkeypatch, tmp_path):
        from job_agent.cv_tailor import generate_tailored_cv
        from docx import Document

        doc = Document()
        doc.add_paragraph("PROFESSIONAL SUMMARY")
        doc.add_paragraph("A producer.")
        doc.add_paragraph("EXPERIENCE")
        doc.add_paragraph("SKILLS")
        cv_path = str(tmp_path / "test_cv.docx")
        doc.save(cv_path)

        calls = []
        def mock_call_llm(prompt, model="test", temperature=0.0):
            calls.append(prompt)
            return "Tailored."

        monkeypatch.setattr("job_agent.cv_tailor.call_llm", mock_call_llm)

        generate_tailored_cv(
            cv_path, "Profile", {"title": "Producer", "description": "Produce"},
            transcript_text=f"I ran a 9-person remote team at {EMPLOYERS[1]} for three years.",
        )

        assert len(calls) >= 1
        assert f"ran a 9-person remote team at {EMPLOYERS[1]}" in calls[0]


# === DB keyword_category column ===

class TestDBKeywordCategory:
    """Verify keyword_category column exists and works."""

    def test_migration_adds_keyword_category_column(self, tmp_db):
        conn = tmp_db._get_thread_conn()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(scrapes)")
        columns = [info[1] for info in cursor.fetchall()]
        assert "keyword_category" in columns

    def test_save_job_with_keyword_category(self, tmp_db):
        job = {
            "title": "Podcast Producer",
            "company": "PodCo",
            "job_url": "https://podco.com/jobs/1",
        }
        job_id = tmp_db.save_job(job, keyword_category="podcast_and_audio")
        assert job_id is not None

        details = tmp_db.get_job_details(job_id)
        assert details["keyword_category"] == "podcast_and_audio"

    def test_save_job_with_relevance_stores_keyword_category(self, tmp_db):
        job = {
            "title": "Game Producer",
            "company": "GameCo",
            "job_url": "https://gameco.com/jobs/1",
        }
        job_id = tmp_db.save_job_with_relevance(
            job,
            relevance="relevant",
            match_level="HIGH_MATCH",
            confidence=0.9,
            keyword_category="games_production",
        )
        details = tmp_db.get_job_details(job_id)
        assert details["keyword_category"] == "games_production"
