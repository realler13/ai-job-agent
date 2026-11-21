"""Tests for single-call (consolidated) CV generation + its fallback."""
import job_agent.cv_tailor as cv


def _base_profile(n_roles=2):
    roles = [
        {"title": "Producer @ A", "description": "- old A1\n- old A2"},
        {"title": "Coordinator @ B", "description": "- old B1"},
    ][:n_roles]
    return {"summary": "Original summary.", "experience": roles}


def _patch_common(monkeypatch, profile=None):
    monkeypatch.setattr(cv, "CV_SINGLE_CALL_GENERATION", True)
    monkeypatch.setattr(cv, "_load_transcript_context", lambda domain="producer_core": "")
    monkeypatch.setattr(cv, "parse_docx_to_structure", lambda p: profile or _base_profile())


JOB = {"title": "Senior Producer", "company": "X", "description": "JD text here"}


def test_consolidated_single_call(monkeypatch):
    """One JSON call produces the full tailored_data; role dicts are preserved."""
    _patch_common(monkeypatch)
    calls = []

    def fake_llm(prompt, model="", **k):
        calls.append(prompt)
        return (
            '{"summary": "New tailored summary.", "roles": ['
            '{"index": 0, "bullets": "- new A1\\n- new A2"},'
            '{"index": 1, "bullets": "- new B1"}]}'
        )
    monkeypatch.setattr(cv, "call_llm", fake_llm)

    data = cv.generate_tailored_cv("base.docx", "profile text", JOB, model_name="claude-opus-4-8")
    assert len(calls) == 1                                   # ONE call, not 3
    assert data["summary"] == "New tailored summary."
    assert data["experience"][0]["description"] == "- new A1\n- new A2"
    assert data["experience"][1]["description"] == "- new B1"
    assert data["experience"][0]["title"] == "Producer @ A"  # original role keys preserved


def test_consolidated_tolerates_code_fences(monkeypatch):
    """A ```json-fenced / prose-wrapped response still parses (raw_decode)."""
    _patch_common(monkeypatch, profile=_base_profile(1))
    monkeypatch.setattr(
        cv, "call_llm",
        lambda prompt, model="", **k: 'Here you go:\n```json\n{"summary": "S2", "roles": [{"index": 0, "bullets": "- x"}]}\n```',
    )
    data = cv.generate_tailored_cv("base.docx", "p", JOB, model_name="claude-opus-4-8")
    assert data["summary"] == "S2"
    assert data["experience"][0]["description"] == "- x"


def test_falls_back_on_bad_json(monkeypatch):
    """Malformed consolidated response → fall back to per-section calls."""
    _patch_common(monkeypatch, profile=_base_profile(1))
    calls = []

    def fake_llm(prompt, model="", **k):
        calls.append(prompt)
        if "JSON object" in prompt:        # the consolidated prompt
            return "sorry, not json at all"
        return "- rewritten bullet"        # summary + per-role fallback calls
    monkeypatch.setattr(cv, "call_llm", fake_llm)

    data = cv.generate_tailored_cv("base.docx", "p", JOB, model_name="claude-opus-4-8")
    assert len(calls) >= 2                 # consolidated attempt + fallback section calls
    assert data["experience"][0]["description"] == "- rewritten bullet"


def test_falls_back_when_role_missing(monkeypatch):
    """Consolidated response missing a requested role index → fall back."""
    _patch_common(monkeypatch, profile=_base_profile(2))

    def fake_llm(prompt, model="", **k):
        if "JSON object" in prompt:
            return '{"summary": "S", "roles": [{"index": 0, "bullets": "- only A"}]}'  # missing index 1
        return "- fallback bullet"
    monkeypatch.setattr(cv, "call_llm", fake_llm)

    data = cv.generate_tailored_cv("base.docx", "p", JOB, model_name="claude-opus-4-8")
    assert data["experience"][0]["description"] == "- fallback bullet"
    assert data["experience"][1]["description"] == "- fallback bullet"
