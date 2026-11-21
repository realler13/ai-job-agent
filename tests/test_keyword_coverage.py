"""ATS keyword coverage — grounded classification, honesty rule (never stuffs).

Patch `job_agent.cv_tailor.call_llm` (module-level import).
"""

import json
from unittest.mock import patch

from job_agent import cv_tailor as t
from tests.fixtures.persona import EMPLOYERS, YEARS


TD = {"summary": f"Video producer, {YEARS}.", "experience": [
    {"title": "Producer", "company": EMPLOYERS[1],
     "description": "Delivered 25 branded shorts\nRan edit review cycles"}]}
JOB = {"title": "Video Producer", "company": "Harbor Lights Media",
       "description": "Need edit supervision, review cycles, budgeting, After Effects."}


def _coverage_json(rows, add=None, gaps=None):
    return json.dumps({
        "coverage": rows,
        "add_keywords": add if add is not None else [],
        "gaps": gaps if gaps is not None else [],
        "summary": "test",
    })


class TestCoverage:
    def test_parses_and_classifies(self):
        rows = [
            {"keyword": "review cycles", "priority": "required", "status": "covered", "note": "in bullet"},
            {"keyword": "budgeting", "priority": "required", "status": "missing_have_it", "note": "has it"},
            {"keyword": "After Effects", "priority": "preferred", "status": "missing_gap", "note": "gap"},
        ]
        with patch("job_agent.cv_tailor.call_llm",
                   return_value=_coverage_json(rows, add=["budgeting"], gaps=["After Effects"])):
            cov = t.check_keyword_coverage(TD, JOB)
        assert len(cov["coverage"]) == 3
        assert cov["add_keywords"] == ["budgeting"]
        assert cov["gaps"] == ["After Effects"]

    def test_honesty_gap_never_added(self):
        """A model that wrongly lists a genuine GAP in add_keywords must be
        overruled — the defensive filter only lets have-it terms through."""
        rows = [
            {"keyword": "After Effects", "priority": "required", "status": "missing_gap", "note": "gap"},
            {"keyword": "budgeting", "priority": "required", "status": "missing_have_it", "note": "has"},
        ]
        # model erroneously tries to stuff the gap term too
        with patch("job_agent.cv_tailor.call_llm",
                   return_value=_coverage_json(rows, add=["budgeting", "After Effects"])):
            cov = t.check_keyword_coverage(TD, JOB)
        assert "After Effects" not in cov["add_keywords"]  # gap stripped
        assert cov["add_keywords"] == ["budgeting"]

    def test_add_falls_back_to_have_it_when_model_omits(self):
        # model marks a have-it row but forgets to populate add_keywords
        rows = [{"keyword": "budgeting", "priority": "required", "status": "missing_have_it", "note": "has"}]
        with patch("job_agent.cv_tailor.call_llm", return_value=_coverage_json(rows, add=[])):
            cov = t.check_keyword_coverage(TD, JOB)
        assert cov["add_keywords"] == ["budgeting"]

    def test_fail_open_on_garbage(self):
        with patch("job_agent.cv_tailor.call_llm", return_value="not json at all"):
            cov = t.check_keyword_coverage(TD, JOB)
        assert cov["coverage"] == [] and cov["add_keywords"] == []

    def test_fail_open_on_exception(self):
        with patch("job_agent.cv_tailor.call_llm", side_effect=RuntimeError("boom")):
            cov = t.check_keyword_coverage(TD, JOB)
        assert cov["add_keywords"] == [] and "error" in cov["summary"]


class TestIncorporate:
    def test_weaves_and_returns_revised(self):
        revised = {"summary": f"Video producer, {YEARS}.", "experience": [
            {"title": "Producer", "bullets": ["Delivered 25 branded shorts on budget",
                                               "Ran edit review cycles with edit supervision"]}]}
        with patch("job_agent.cv_tailor.call_llm", return_value=json.dumps(revised)):
            out = t.incorporate_keywords(TD, ["budgeting", "edit supervision"], JOB)
        assert "budget" in out["experience"][0]["description"]

    def test_no_keywords_is_noop(self):
        assert t.incorporate_keywords(TD, [], JOB) is None

    def test_bad_json_returns_none(self):
        with patch("job_agent.cv_tailor.call_llm", return_value="nope"):
            assert t.incorporate_keywords(TD, ["budgeting"], JOB) is None
