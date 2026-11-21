"""CV framework revamp: source bank, metrics contract, domain bases,
hiring-manager grader — tested against sample job postings (fictional
companies, invented posting text).

Patching notes: docs_builder imports the HM functions lazily inside the gate
block (`from .cv_tailor import grade_cv_hiring_manager, ...`) — patch them at
`job_agent.cv_tailor.*`. The conftest hermetic guard nulls the
Anthropic client, so any unmocked call_llm returns "" (gates fail open).
"""

import glob
import re
import json
from unittest.mock import patch, MagicMock

import pytest

from job_agent import cv_tailor as t
from job_agent.config import (METRICS_BANK_TEXT, METRICS_BANK, DOMAIN_CV_MAP,
                                      resolve_cv_domain)
from tests.fixtures.persona import EMPLOYERS, YEARS


# ── Sample postings (fictional companies; the text is invented) ─────────────
SAMPLE_ROLES = {
    "podcast": {
        "title": "Podcast Producer", "company": "Harbor Lights Media",
        "description": ("Produce a weekly customer-story interview podcast: book guests, prepare "
                        "question outlines, run remote and in-studio recordings, manage freelance "
                        "audio editors and deliver every episode on a fixed release calendar. "
                        "You also cut short video clips of each episode for social channels, "
                        "keep the guest pipeline a month ahead and track listener numbers."),
    },
    "live_event": {
        "title": "Live Event Producer", "company": "Northwind Stageworks",
        "description": ("Producer for a regional trade fair: plan the floor layout, book exhibitors "
                        "and catering, coordinate load-in crews and venue staff, track the budget "
                        "and keep the three-day show running to schedule."),
    },
    "branded_video": {
        "title": "Branded Video Producer", "company": "Brightkettle Studios",
        "description": ("Turn product briefs into short branded videos for social feeds: write "
                        "treatments, cast and schedule shoots, supervise color and sound finishing, "
                        "and report on completion rates after each campaign goes live."),
    },
}


class TestRoleRouting:
    def test_domains_resolve_sensibly(self):
        assert resolve_cv_domain(job_title="Podcast Producer") in DOMAIN_CV_MAP
        assert resolve_cv_domain(job_title="YouTube Producer") == "youtube_and_creators"
        # live-event role routes via title fuzzy or falls back to a valid domain
        assert resolve_cv_domain(job_title="Live Event Producer") in DOMAIN_CV_MAP

    @pytest.mark.parametrize("role", list(SAMPLE_ROLES.values()), ids=list(SAMPLE_ROLES))
    def test_consolidated_prompt_carries_new_contract(self, role):
        p = t.PROMPT_CV_CONSOLIDATED_CLAUDE.format(
            domain_context="ctx", job_title=role["title"], company=role["company"],
            jd_text=role["description"], full_profile="profile",
            source_bank="THE-BANK", transcript_context="stories",
            original_summary="sum", roles_block="<role index='0'>x</role>",
            metrics_bank=METRICS_BANK_TEXT, anti_slop_rules="rules")
        assert "career_source_bank" in p and "THE-BANK" in p
        assert "THIS BLOCK WINS" in p                    # precedence rule
        assert "MIX FROM THE SOURCE BANK" in p           # bank lines may replace weak bullets
        # canonical tenure: the rule points at the canonical figure, and the
        # verified metrics that carry it are in the prompt
        assert "canonical years figure" in p
        assert YEARS in p
        assert "PAST role" in p or "past tense" in p.lower()  # ended-role status
        assert role["description"][:60] in p             # the posting is present


class TestSourceBank:
    def test_loads_the_bank_file_and_caches_it(self, tmp_path, monkeypatch):
        import job_agent.config as cfg
        bank_text = ("# CAREER SOURCE BANK (fictional)\n"
                     f"Canonical years: {YEARS}.\n"
                     f"{EMPLOYERS[0]}: Producer, March 2023 – January 2026 (past role).\n")
        bank = tmp_path / "source_bank.md"
        bank.write_text(bank_text, encoding="utf-8")
        t._source_bank_cache = None
        monkeypatch.setattr(cfg, "SOURCE_BANK_PATH", str(bank))
        try:
            loaded = t._load_source_bank()
            assert loaded == bank_text
            bank.unlink()                          # cached: the file is not read again
            assert t._load_source_bank() == bank_text
        finally:
            t._source_bank_cache = None            # reset for other tests

    def test_long_bank_is_truncated_to_the_prompt_budget(self, tmp_path, monkeypatch):
        import job_agent.config as cfg
        bank = tmp_path / "source_bank.md"
        bank.write_text("x" * (cfg.TRUNCATION_SOURCE_BANK + 500), encoding="utf-8")
        t._source_bank_cache = None
        monkeypatch.setattr(cfg, "SOURCE_BANK_PATH", str(bank))
        try:
            assert len(t._load_source_bank()) == cfg.TRUNCATION_SOURCE_BANK
        finally:
            t._source_bank_cache = None

    def test_missing_file_degrades(self, monkeypatch):
        t._source_bank_cache = None
        import job_agent.config as cfg
        monkeypatch.setattr(cfg, "SOURCE_BANK_PATH", "nonexistent-bank.md")
        bank = t._load_source_bank()
        assert "unavailable" in bank
        t._source_bank_cache = None        # reset for other tests


class TestMetricsV2:
    def test_canonical_tenure(self):
        """A profile must pin ONE canonical tenure figure. Generators quote it
        verbatim; without the pin they drift between '9 years' and 'over a
        decade' across documents."""
        assert "CANONICAL years figure" in METRICS_BANK_TEXT
        assert re.search(r"\d+\+? years", METRICS_BANK_TEXT)

    def test_past_role_rule(self):
        """Past roles must be pinned as past, or the CV writes them in present
        tense and claims the candidate is still employed there."""
        assert "PAST role" in METRICS_BANK_TEXT
        assert "EMPLOYMENT STATUS RULE" in METRICS_BANK_TEXT
        # no ATTRIBUTION uses the stale open-ended range (the ban-rule text
        # legitimately quotes the forbidden string, so exclude rule lines)
        attrib_lines = [l for l in METRICS_BANK_TEXT.split("\n") if "RULE" not in l]
        assert not any(re.search(r"\b20\d\d\s*[–-]\s*present\b", l, re.I) for l in attrib_lines)

    def test_attributions_carry_employer_and_period(self):
        """Every metric needs an owner and a date, so a generator cannot move
        an achievement from one employer to another."""
        assert re.search(r"\(.*\d{4}", METRICS_BANK_TEXT) or "–" in METRICS_BANK_TEXT

    def test_metrics_block_is_non_trivial(self):
        """A profile with an empty metrics bank would let the generator invent
        numbers freely — the anti-hallucination contract depends on content."""
        assert len(METRICS_BANK_TEXT.splitlines()) > 5


class TestDomainBases:
    # A deployment's base CVs are its candidate's own documents and are
    # gitignored, so these only run where they are present in the current
    # directory. The checks are the profile-independent contract.
    BASES = sorted(glob.glob("base_cv_*.docx"))

    @pytest.mark.skipif(not BASES, reason="no base CVs present (private artefact)")
    def test_one_base_per_domain(self):
        from job_agent.config import DOMAIN_CV_MAP
        assert len(self.BASES) == len(DOMAIN_CV_MAP)

    @pytest.mark.parametrize("path", BASES)
    def test_structure_and_truth(self, path):
        import docx
        text = "\n".join(p.text for p in docx.Document(path).paragraphs)
        parsed = t.parse_docx_to_structure(path)
        bullet_roles = [r for r in parsed["experience"] if r.get("description")]
        assert bullet_roles, f"{path}: no role with bullets"
        assert parsed["summary"]
        for banned in ("Present", "currently", "&amp;"):
            assert banned not in text, f"{path}: {banned!r}"

    @pytest.mark.parametrize("path", BASES)
    def test_fits_two_pages(self, path):
        assert t.estimate_docx_pages(path) <= 2.0


class TestHMGrader:
    GOOD = json.dumps({
        "ideal_candidate": "x", "scores": {"relevance": 80, "evidence": 70,
        "ats_alignment": 75, "differentiation": 60, "clarity": 85},
        "score_notes": {}, "red_flags": [], "top_fixes": ["sharpen summary line 1"],
        "overall": 74, "verdict": "maybe"})

    TD = {"summary": f"Video producer, {YEARS}.", "experience": [
        {"title": "Producer", "company": EMPLOYERS[0], "description": "b1\nb2\nb3", "duration": "2023-2026"}]}

    def test_parses_grader_json(self):
        with patch("job_agent.cv_tailor.call_llm", return_value=self.GOOD):
            hm = t.grade_cv_hiring_manager(self.TD, SAMPLE_ROLES["podcast"])
        assert hm["overall"] == 74 and hm["verdict"] == "maybe"
        assert hm["top_fixes"]

    def test_fail_open_on_garbage(self):
        with patch("job_agent.cv_tailor.call_llm", return_value="no json"):
            hm = t.grade_cv_hiring_manager(self.TD, SAMPLE_ROLES["podcast"])
        assert hm["overall"] is None and "error" in hm

    def test_fail_open_on_exception(self):
        with patch("job_agent.cv_tailor.call_llm", side_effect=RuntimeError("x")):
            hm = t.grade_cv_hiring_manager(self.TD, SAMPLE_ROLES["podcast"])
        assert hm["overall"] is None

    def test_revise_applies_and_respects_floors(self):
        revised = {"summary": "Sharper.", "experience": [
            {"title": "Producer", "bullets": ["B1 sharp", "B2 sharp", "B3 sharp"]}]}
        with patch("job_agent.cv_tailor.call_llm", return_value=json.dumps(revised)):
            out = t.revise_cv_with_hm_feedback(self.TD, SAMPLE_ROLES["podcast"], ["fix"])
        assert out["summary"] == "Sharper."
        assert out["experience"][0]["description"] == "B1 sharp\nB2 sharp\nB3 sharp"
        # gutting trim rejected — keeps originals
        gutted = {"experience": [{"title": "Producer", "bullets": ["only"]}]}
        with patch("job_agent.cv_tailor.call_llm", return_value=json.dumps(gutted)):
            out2 = t.revise_cv_with_hm_feedback(self.TD, SAMPLE_ROLES["podcast"], ["fix"])
        assert out2["experience"][0]["description"] == "b1\nb2\nb3"

    def test_no_fixes_noop(self):
        assert t.revise_cv_with_hm_feedback(self.TD, SAMPLE_ROLES["podcast"], []) is None


class TestHMIntegrationInDocsBuilder:
    """Phase 27 (deliberate change): the HM grade is a DIAGNOSTIC on the final
    text (shipped summary + shipped bullets + skills). The revision pass no
    longer touches what ships, whatever the grade — it is never called."""

    def _run(self, tmp_path, monkeypatch, hm):
        from tests.fixtures.fictional_cv import CAREER_MD, build_fixture_docx
        from job_agent.docs_builder import save_documents_for_job
        monkeypatch.chdir(tmp_path)
        base = build_fixture_docx(tmp_path / "base_cv_producer_core.docx")
        parsed = t.parse_docx_to_structure(base)
        seen = {}
        revise = MagicMock(return_value={"summary": "S2", "experience": []})

        def mock_hm(data, job, **k):
            seen["data"], seen["kw"] = data, k
            return hm

        clean = {"flag": "PASSED", "state": "ok", "corrected_summary": None,
                 "audit": {"overall_pass": True, "hallucination": {"found": False},
                           "claims": [{"section": "summary", "role_index": None, "quote": "S",
                                       "verdict": "supported", "reason": "r"}]}}
        monkeypatch.setattr("job_agent.cv_tailor.generate_tailored_cv",
                            lambda *a, **k: {"summary": "Producer for documentary shoots.",
                                             "experience": parsed["experience"]})
        monkeypatch.setattr("job_agent.docs_builder.grade_cv_quality", lambda *a, **k: clean)
        monkeypatch.setattr("job_agent.docs_builder.check_keyword_coverage",
                            lambda *a, **k: {"coverage": [], "add_keywords": [], "gaps": [], "summary": ""})
        monkeypatch.setattr("job_agent.cv_tailor.grade_cv_hiring_manager", mock_hm)
        monkeypatch.setattr("job_agent.cv_tailor.revise_cv_with_hm_feedback", revise)
        monkeypatch.setattr("job_agent.docs_builder.generate_cover_letter_content",
                            lambda *a, **k: "Dear Hiring Manager,\nBody.")
        monkeypatch.setattr("job_agent.docs_builder._resolve_base_cv_path",
                            lambda *a, **k: (base, "producer_core"))
        monkeypatch.setattr("job_agent.cv_facts.verified_corpus",
                            lambda extra=(): [CAREER_MD.read_text()])
        result = save_documents_for_job("base_cv.docx", "P", {
            "id": 1, "title": "Documentary Producer", "company": "C",
            "description": SAMPLE_ROLES["podcast"]["description"]}, transcript_text="T")
        return result, seen, revise, parsed

    def test_low_grade_is_recorded_never_revised(self, tmp_path, monkeypatch):
        low = {"overall": 55, "verdict": "maybe", "top_fixes": ["fix A"], "scores": {}}
        result, seen, revise, parsed = self._run(tmp_path, monkeypatch, low)
        q = result[4]
        assert q["audit"]["hm_grade"]["overall"] == 55
        assert q["audit"]["hm_below_threshold"] is True
        assert "hm_grade_after" not in q["audit"]
        revise.assert_not_called()
        # graded on the FINAL text, with the SKILLS section and today's date
        assert seen["data"]["experience"] == parsed["experience"]
        assert "budgeting" in seen["kw"]["skills_text"] and seen["kw"]["evaluation_date"]
        assert t.parse_docx_to_structure(result[0])["experience"] == parsed["experience"]

    def test_high_grade_no_revision(self, tmp_path, monkeypatch):
        high = {"overall": 90, "verdict": "interview", "top_fixes": [], "scores": {}}
        result, _, revise, _ = self._run(tmp_path, monkeypatch, high)
        assert result[4]["audit"]["hm_grade"]["overall"] == 90
        assert "hm_grade_after" not in result[4]["audit"]
        assert "hm_below_threshold" not in result[4]["audit"]
        revise.assert_not_called()
