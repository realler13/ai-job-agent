import json
import pytest
from job_agent.two_stage_matcher import parse_json_strict


class TestParseJsonStrict:
    def test_valid_json(self):
        result = parse_json_strict('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_preamble(self):
        result = parse_json_strict('Here is the JSON:\n{"key": "value"}\nDone.')
        assert result == {"key": "value"}

    def test_nested_json(self):
        text = '{"outer": {"inner": true}, "list": [1, 2]}'
        result = parse_json_strict(text)
        assert result["outer"]["inner"] is True

    def test_no_json_block(self):
        with pytest.raises(ValueError, match="No JSON block found"):
            parse_json_strict("no json here")

    def test_invalid_json(self):
        with pytest.raises(ValueError):
            parse_json_strict("{invalid json}")

    def test_empty_string(self):
        with pytest.raises(ValueError):
            parse_json_strict("")

    def test_markdown_wrapped_json(self):
        text = '```json\n{"match_level": "HIGH_MATCH"}\n```'
        result = parse_json_strict(text)
        assert result["match_level"] == "HIGH_MATCH"


class TestJobMatcherIntegration:
    def test_match_jobs_with_mock_llm(self, monkeypatch):
        """Test that JobMatcher produces correct columns with mocked LLM."""
        import pandas as pd

        call_count = [0]

        def mock_call_llm(prompt, model="test", temperature=0.0):
            call_count[0] += 1
            # Return different responses based on call count
            if call_count[0] == 1:
                # Candidate highlights
                return json.dumps({
                    "candidate_id": "TEST_CAND",
                    "primary_role": "PRODUCER",
                    "seniority": "SENIOR",
                    "location_preference": "HOME_METRO",
                    "industries": ["Tech"],
                    "core_strengths": ["video production"],
                    "keywords": ["video", "producer"],
                })
            elif call_count[0] % 2 == 0:
                # Job extraction (even calls)
                return json.dumps({
                    "job_id": "JOB_1",
                    "role_type": "PRODUCER",
                    "seniority": "SENIOR",
                    "location_type": "HOME_METRO",
                    "must_haves": ["video production"],
                    "nice_to_haves": ["team management"],
                    "keywords": ["video"],
                    "trap_flags": {
                        "social_trap": False,
                        "junior_trap": False,
                        "location_mismatch_possible": False,
                    },
                })
            else:
                # Job decision (odd calls after first)
                return json.dumps({
                    "match_level": "HIGH_MATCH",
                    "reason": "Perfect fit for senior video producer",
                    "rejection_rule": "NONE",
                    "confidence": 0.92,
                })

        monkeypatch.setattr("job_agent.job_matcher.call_llm", mock_call_llm)

        from job_agent.job_matcher import JobMatcher

        df = pd.DataFrame([{
            "title": "Senior Video Producer",
            "company": "TechCorp",
            "location": "Seattle, WA",
            "description": "Lead video production.",
        }])

        matcher = JobMatcher("Test profile text", "test-model")
        result = matcher.match_jobs(df)

        assert "match_level" in result.columns
        assert "confidence" in result.columns
        assert "gpt_relevance" in result.columns
        assert result.iloc[0]["match_level"] == "HIGH_MATCH"
        assert result.iloc[0]["confidence"] == 0.92
        assert result.iloc[0]["gpt_relevance"] == "relevant"

    def test_match_jobs_error_recovery(self, monkeypatch):
        """Test that failed jobs get ERROR status instead of being silently marked irrelevant."""
        import pandas as pd

        call_count = [0]

        def mock_call_llm(prompt, model="test", temperature=0.0):
            call_count[0] += 1
            if call_count[0] <= 3:
                # First few calls are candidate highlights (including retries)
                return json.dumps({
                    "candidate_id": "TEST_CAND",
                    "primary_role": "PRODUCER",
                    "seniority": "SENIOR",
                    "location_preference": "HOME_METRO",
                    "industries": [],
                    "core_strengths": [],
                    "keywords": [],
                })
            # All subsequent calls (job extraction/decision) return empty to simulate failure
            return ""

        monkeypatch.setattr("job_agent.job_matcher.call_llm", mock_call_llm)

        from job_agent.job_matcher import JobMatcher

        df = pd.DataFrame([{
            "title": "Test Job",
            "company": "TestCo",
            "location": "Remote",
            "description": "A test.",
        }])

        matcher = JobMatcher("Test profile", "test-model")
        result = matcher.match_jobs(df)

        assert result.iloc[0]["match_level"] == "ERROR"
        assert result.iloc[0]["confidence"] == 0.0
