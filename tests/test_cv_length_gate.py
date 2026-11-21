"""CV 2-page length gate (deterministic, no PDF render).

The estimator reproduces the rendered page height from the .docx content against
the fixed template geometry. Calibrated on real files (all genuine 2-page CVs
land under the trim ceiling; true overflow exceeds it).

Patching note: trim_cv_to_length uses the module-level cv_tailor.call_llm, so
patch `job_agent.cv_tailor.call_llm`.
"""

import glob
import json
from unittest.mock import patch

import pytest

from job_agent import cv_tailor as t
from job_agent.config import CV_PAGE_CEILING
from tests.fixtures.fictional_cv import build_fixture_docx
from tests.fixtures.persona import EMPLOYERS

STUDIO, COLLECTIVE = EMPLOYERS[0], EMPLOYERS[1]


class TestEstimator:
    def test_fixture_bases_estimate_under_the_ceiling(self, tmp_path):
        """Both base layouts — the domain base AND the legacy base_cv.docx
        (the earlier blind spot) — must estimate BELOW the trim ceiling, or the
        gate nuisance-trims a CV that already fits."""
        bases = [build_fixture_docx(tmp_path / "base_cv_producer_core.docx"),
                 build_fixture_docx(tmp_path / "base_cv.docx", legacy=True)]
        for f in bases:
            pages = t.estimate_docx_pages(f)
            assert 0.5 < pages <= CV_PAGE_CEILING, f"{f} estimated {pages}"

    def test_deployment_base_cvs_estimate_two_pages(self):
        """A deployment's own base templates (gitignored; present only where a
        profile has been set up, run from the repo root) render to 2 pages, so
        all must estimate BELOW the trim ceiling."""
        bases = [f for f in glob.glob("base_cv*.docx") if "~$" not in f]
        if not bases:
            pytest.skip("no deployment base CVs in the current directory (private artefact)")
        for f in bases:
            pages = t.estimate_docx_pages(f)
            assert 1.3 < pages <= CV_PAGE_CEILING, f"{f} estimated {pages}"

    def test_generated_cvs_under_ceiling(self):
        """Sample ACROSS the full history (not just the newest few) — the
        rendered ground truth is that all generated CVs are <=2 pages, so NONE
        may exceed the ceiling (this is where the 15% false-trim bug hid)."""
        # Phase 27: each generation has its own directory (cv/cv_DATE/j<id>_<HHMMSS>/).
        # A checkout with no generated-CV history at all (a git worktree:
        # cv/ is gitignored and lives only in the live checkout) has nothing
        # to sample; where cv/ exists the broad-history floor still applies.
        if not glob.glob("cv/cv_*"):
            pytest.skip("no generated-CV history in this checkout (cv/ is data, not code)")
        gen = [f for f in glob.glob("cv/cv_*/*.docx") + glob.glob("cv/cv_*/*/*.docx") if "~$" not in f]
        assert len(gen) > 20, "expected a broad CV history to sample"
        vals = [t.estimate_docx_pages(f) for f in gen]
        over = [f for f, v in zip(gen, vals) if v > CV_PAGE_CEILING]
        assert not over, f"{len(over)} genuine 2-page CVs over ceiling: {over[:3]}"

    def test_overflow_docx_exceeds_ceiling(self, tmp_path):
        """A base CV bloated with many extra bullets must estimate OVER the
        ceiling — proving the gate fires on genuine 3-page overflow."""
        from docx import Document
        base = build_fixture_docx(tmp_path / "base_cv_producer_core.docx")
        doc = Document(base)
        for i in range(60):
            doc.add_paragraph(
                "This is a deliberately long extra bullet line number "
                f"{i} added to push the document well past two pages of content.",
                style="List Bullet")
        out = tmp_path / "bloated.docx"
        doc.save(str(out))
        assert t.estimate_docx_pages(str(out)) > CV_PAGE_CEILING

    def test_missing_file_fails_open(self):
        # never block delivery on an estimation bug
        assert t.estimate_docx_pages("/nonexistent/x.docx") == 1.5

    def test_font_resolution(self, tmp_path):
        from docx import Document
        from docx.shared import Pt
        doc = Document()
        p = doc.add_paragraph()
        r = p.add_run("header")
        r.font.size = Pt(12)
        assert t._paragraph_font_pt(p) == 12.0
        p2 = doc.add_paragraph("body with no explicit size")
        assert t._paragraph_font_pt(p2) == 11.0  # docDefault fallback


class TestJsonHelpers:
    def test_roundtrip_preserves_untouched_fields(self):
        td = {"summary": "S", "experience": [
            {"title": "Producer", "company": STUDIO, "description": "b1\nb2\nb3\nb4", "dates": "2024"}]}
        j = json.loads(t._tailored_data_to_json(td))
        assert j["experience"][0]["bullets"] == ["b1", "b2", "b3", "b4"]
        # top role: revised keeps >=3 bullets (respects floor)
        revised = {"summary": "S2", "experience": [{"title": "Producer", "bullets": ["b1", "b2", "b3"]}]}
        out = t._apply_revised_json(td, revised)
        assert out["summary"] == "S2"
        assert out["experience"][0]["description"] == "b1\nb2\nb3"
        assert out["experience"][0]["dates"] == "2024"  # preserved

    def test_apply_ignores_extra_roles(self):
        td = {"summary": "S", "experience": [{"title": "A", "company": "X",
              "description": "a1\na2\na3"}]}
        revised = {"experience": [{"title": "A", "bullets": ["y1", "y2", "y3"]},
                                  {"title": "GHOST", "bullets": ["z"]}]}
        out = t._apply_revised_json(td, revised)
        assert len(out["experience"]) == 1  # no phantom role added
        assert out["experience"][0]["description"] == "y1\ny2\ny3"

    def test_matches_by_title_not_index(self):
        """A REORDERED revised set must not put one employer's bullets under
        another's heading (the attribution-corruption bug)."""
        td = {"summary": "S", "experience": [
            {"title": "Producer", "company": STUDIO, "description": "p1\np2\np3"},
            {"title": "Sr Producer", "company": COLLECTIVE, "description": "q1\nq2"}]}
        # LLM returns them in the OPPOSITE order
        revised = {"experience": [
            {"title": "Sr Producer", "bullets": ["NEW q1", "NEW q2"]},
            {"title": "Producer", "bullets": ["NEW p1", "NEW p2", "NEW p3"]}]}
        out = t._apply_revised_json(td, revised)
        # Producer (the first studio) must get the Producer bullets, NOT Sr Producer's
        assert out["experience"][0]["company"] == STUDIO
        assert out["experience"][0]["description"] == "NEW p1\nNEW p2\nNEW p3"
        assert out["experience"][1]["company"] == COLLECTIVE
        assert out["experience"][1]["description"] == "NEW q1\nNEW q2"

    def test_dropped_role_keeps_original(self):
        """If the LLM omits a role, that role keeps its ORIGINAL bullets — it
        must never inherit another role's content."""
        td = {"summary": "S", "experience": [
            {"title": "Producer", "company": STUDIO, "description": "p1\np2\np3"},
            {"title": "Coordinator", "company": "Freelance", "description": "c1\nc2"}]}
        revised = {"experience": [{"title": "Producer", "bullets": ["NEW p1", "NEW p2", "NEW p3"]}]}
        out = t._apply_revised_json(td, revised)
        assert out["experience"][1]["company"] == "Freelance"
        assert out["experience"][1]["description"] == "c1\nc2"  # untouched original

    def test_floor_enforced_top_role(self):
        """A trim below the floor (top role <3) keeps the original bullets."""
        td = {"summary": "S", "experience": [
            {"title": "Producer", "company": STUDIO, "description": "p1\np2\np3\np4"}]}
        revised = {"experience": [{"title": "Producer", "bullets": ["only one"]}]}
        out = t._apply_revised_json(td, revised)
        assert out["experience"][0]["description"] == "p1\np2\np3\np4"  # original kept

    def test_floor_enforced_other_role(self):
        td = {"summary": "S", "experience": [
            {"title": "Producer", "company": STUDIO, "description": "p1\np2\np3"},
            {"title": "Coordinator", "company": "Freelance", "description": "c1\nc2\nc3"}]}
        revised = {"experience": [
            {"title": "Producer", "bullets": ["p1", "p2", "p3"]},
            {"title": "Coordinator", "bullets": ["only one"]}]}  # <2 → floor
        out = t._apply_revised_json(td, revised)
        assert out["experience"][1]["description"] == "c1\nc2\nc3"  # original kept

    def test_parse_brace_in_string(self):
        ans = 'ok {"summary": "cut costs {30%}", "experience": []} trailing'
        assert t._parse_llm_json_object(ans)["summary"] == "cut costs {30%}"

    def test_parse_no_json(self):
        assert t._parse_llm_json_object("no json here") is None
        assert t._parse_llm_json_object("") is None


class TestTrim:
    def test_trim_returns_revised(self):
        td = {"summary": "S", "experience": [
            {"title": "Producer", "company": STUDIO, "description": "b1\nb2\nb3\nb4\nb5"}]}
        trimmed = {"summary": "S", "experience": [{"title": "Producer", "bullets": ["b1", "b2", "b3"]}]}
        with patch("job_agent.cv_tailor.call_llm", return_value=json.dumps(trimmed)):
            out = t.trim_cv_to_length(td, {"title": "P", "company": "Harbor Lights Media", "description": "d"}, 2.3)
        assert out["experience"][0]["description"] == "b1\nb2\nb3"

    def test_trim_bad_json_returns_none(self):
        td = {"summary": "S", "experience": []}
        with patch("job_agent.cv_tailor.call_llm", return_value="sorry, no"):
            assert t.trim_cv_to_length(td, {"title": "P"}, 2.3) is None
