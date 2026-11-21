"""Tests for the LangFuse cascade tracer — the fail-safe contract."""

from job_agent.observability import (
    CascadeTracer, _board, _job_key, _posting_id)


class TestFailSafe:
    def test_disabled_tracer_is_noop(self):
        """conftest sets JOB_AGENT_DISABLE_LANGFUSE — every method must no-op
        without touching the network or raising."""
        t = CascadeTracer()
        assert not t.enabled
        job = {"title": "Producer", "company": "X", "link": "https://x.test/1"}
        t.log_stage1(job, prompt="p", completion="YES", model="m",
                     usage=None, cost=None, passed=True)
        t.log_stage2_batch([job], prompt="p", completion="c", model="m",
                           usage=None, cost=None, results=[{}])
        t.log_stage3(job, prompt="p", completion="c", model="m",
                     usage=None, cost=None, result={})
        t.finish(None)

    def test_bad_keys_disable_not_raise(self, monkeypatch):
        """Broken config must disable tracing, never raise at construction."""
        monkeypatch.delenv("JOB_AGENT_DISABLE_LANGFUSE", raising=False)
        monkeypatch.setattr("job_agent.config.LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setattr("job_agent.config.LANGFUSE_SECRET_KEY", "")
        t = CascadeTracer()
        assert not t.enabled


class TestIdentity:
    def test_trace_identity_fields(self):
        job = {"title": "Producer", "company": "X",
               "link": "https://boards.greenhouse.io/x/jobs/1", "site": "Greenhouse"}
        assert _board(job) == "greenhouse"
        assert _job_key(job) == "https://boards.greenhouse.io/x/jobs/1"
        assert len(_posting_id(job)) == 12
        # No link -> stable title|company key
        assert _job_key({"title": "T", "company": "C"}) == "T|C"
        assert _board({}) == "unknown"


class TestUsageSplit:
    def test_even_share_remainder_to_first(self):
        u = {"input_tokens": 10, "output_tokens": 7}
        shares = [CascadeTracer._split_usage(u, 3, i) for i in range(3)]
        assert shares[0] == {"input_tokens": 4, "output_tokens": 3}
        assert shares[1] == shares[2] == {"input_tokens": 3, "output_tokens": 2}
        # Run totals must sum back exactly
        assert sum(s["input_tokens"] for s in shares) == 10
        assert sum(s["output_tokens"] for s in shares) == 7

    def test_single_job_passthrough(self):
        u = {"input_tokens": 5}
        assert CascadeTracer._split_usage(u, 1, 0) is u
        assert CascadeTracer._split_usage(None, 3, 0) is None
