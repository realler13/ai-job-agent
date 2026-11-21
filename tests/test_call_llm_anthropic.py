"""The Anthropic branch of call_llm — the path every Claude call takes.

Pins the request shape each model generation demands. Opus 5.5 (2026-09-26)
made three things load-bearing: effort is the only thinking control and its
default dropped from "high" to "medium"; sampling params are rejected; and
responses begin with thinking blocks, with refusals signalled by stop_reason.
"""
import logging
from types import SimpleNamespace

import pytest

from job_agent import gpt_filter as gf


class _FakeStream:
    def __init__(self, reply):
        self._reply = reply

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._reply


def _reply(content=None, stop_reason="end_turn", stop_details=None, tin=11, tout=7):
    return SimpleNamespace(
        content=content if content is not None else [SimpleNamespace(type="text", text="OK")],
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=tin, output_tokens=tout),
    )


class _FakeClient:
    """Records every messages.create / messages.stream call and returns a
    scripted reply — one for all models, or per model via `by_model`.
    `transport` says which one the LAST request used; `models` lists every
    model called, in order, so fallback routing is observable."""

    def __init__(self, content=None, stop_reason="end_turn", stop_details=None, by_model=None):
        self.sent = None
        self.transport = None
        self.models = []
        self._default = _reply(content, stop_reason, stop_details)
        self._by_model = by_model or {}
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    def _pick(self, kwargs):
        self.sent = kwargs
        self.models.append(kwargs["model"])
        return self._by_model.get(kwargs["model"], self._default)

    def _create(self, **kwargs):
        self.transport = "create"
        return self._pick(kwargs)

    def _stream(self, **kwargs):
        self.transport = "stream"
        return _FakeStream(self._pick(kwargs))


@pytest.fixture
def fake(monkeypatch):
    """Replace the (conftest-nulled) Anthropic client with a recording fake."""
    def install(**kw):
        client = _FakeClient(**kw)
        monkeypatch.setattr(gf, "get_anthropic_client", lambda: client)
        return client
    return install


class TestEffortPin:
    def test_opus_5_5_gets_pinned_effort_when_caller_omits_it(self, fake):
        """Opus 5.5 defaults to effort 'medium' where Opus 5 defaulted to
        'high'. Unpinned, every generation call would silently drop a level —
        and generation runs ABOVE the old level, at xhigh."""
        client = fake()
        gf.call_llm("hi", model="claude-opus-5-5")
        assert client.sent["output_config"] == {"effort": "xhigh"}

    def test_explicit_effort_overrides_the_pin(self, fake):
        client = fake()
        gf.call_llm("hi", model="claude-opus-5-5", effort="medium")
        assert client.sent["output_config"] == {"effort": "medium"}

    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-fable-5-1", "claude-opus-5"])
    def test_models_without_a_pin_send_no_output_config(self, fake, model):
        """Their API defaults are what the pipeline was tuned on — sending
        nothing keeps them exactly as they were."""
        client = fake()
        gf.call_llm("hi", model=model)
        assert "output_config" not in client.sent

    def test_the_judge_runs_at_its_pinned_level(self, fake):
        """Fable 5.2's default is 'high' today. The pin keeps the judge at the
        level the replay measured if that default ever moves."""
        client = fake()
        gf.call_llm("hi", model="claude-fable-5-2")
        assert client.sent["output_config"] == {"effort": "high"}

    def test_every_pinned_model_is_priced_and_selectable(self):
        """A pin for a model the cost table does not know would book its spend
        at $0 and hide it from the digest's cost line."""
        from job_agent.cascade_filter import MODEL_PRICING
        from job_agent.config import AVAILABLE_MODELS, MODEL_EFFORT
        for model, effort in MODEL_EFFORT.items():
            assert effort in ("low", "medium", "high", "xhigh", "max")
            assert model in MODEL_PRICING, model
            assert model in AVAILABLE_MODELS, model


class TestRequestShape:
    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-2", "claude-fable-5-1",
                                       "claude-sonnet-5"])
    def test_no_sampling_params_or_thinking_field_on_current_models(self, fake, model):
        """Non-default temperature/top_p/top_k and a disabled thinking field
        are rejected with a 400 on the current generation."""
        client = fake()
        gf.call_llm("hi", model=model, temperature=0.7)
        for rejected in ("temperature", "top_p", "top_k", "thinking"):
            assert rejected not in client.sent, (model, rejected)

    def test_legacy_model_still_receives_temperature(self, fake):
        client = fake()
        gf.call_llm("hi", model="claude-haiku-4-5-20251001", temperature=0.3)
        assert client.sent["temperature"] == 0.3

    @pytest.mark.parametrize("model,effort,budget,transport", [
        ("claude-fable-5-1", None, 32768, "create"),       # always-on thinking
        ("claude-fable-5-2", None, 32768, "create"),       # the judge: pinned high, not streamed
        ("claude-opus-5-5", None, 65536, "stream"),        # pinned xhigh
        ("claude-opus-5-5", "max", 65536, "stream"),
        ("claude-opus-5-5", "high", 16384, "create"),      # never truncated at high
        ("claude-sonnet-5", None, 16384, "create"),
        ("claude-haiku-4-5-20251001", None, 8192, "create"),
    ])
    def test_budget_and_transport(self, fake, model, effort, budget, transport):
        """max_tokens caps thinking + answer together. At xhigh/max the
        migration guide says start at 64K — and a budget that large must
        stream, or the 300s client timeout cuts the call off mid-thought and
        the SDK retries it twice."""
        client = fake()
        gf.call_llm("hi", model=model, effort=effort)
        assert client.sent["max_tokens"] == budget
        assert client.transport == transport

    def test_streamed_reply_is_parsed_like_a_created_one(self, fake):
        fake(content=[
            SimpleNamespace(type="thinking", thinking="", signature="sig"),
            SimpleNamespace(type="text", text="streamed answer"),
        ])
        usage = {}
        assert gf.call_llm("hi", model="claude-opus-5-5", usage_out=usage) == "streamed answer"
        assert usage == {"input_tokens": 11, "output_tokens": 7}


class TestResponseHandling:
    def test_thinking_blocks_first_are_skipped(self, fake):
        """Responses begin with thinking blocks; reading content[0].text
        silently killed every call once (2026-07). Only text blocks count."""
        fake(content=[
            SimpleNamespace(type="thinking", thinking="", signature="sig"),
            SimpleNamespace(type="text", text="the answer"),
        ])
        assert gf.call_llm("hi", model="claude-opus-5-5") == "the answer"


class TestRefusal:
    """Opus 5.5's classifiers cover more categories than Opus 5's (bio,
    reasoning_extraction on top of cyber). The first version of this handling
    (2026-09-26) was caught by an adversarial audit: it logged pre-output
    refusals as a generic EMPTY response, returned partial text on a mid-answer
    refusal, and claimed to "fall back" when nothing did."""

    REFUSE = dict(stop_reason="refusal", stop_details=SimpleNamespace(category="bio"))

    def test_pre_output_refusal_is_logged_as_a_refusal(self, fake, caplog):
        """Classifiers usually decline before any output: empty content."""
        fake(by_model={"claude-opus-5-5": _reply(content=[], **self.REFUSE),
                       "claude-opus-5": _reply(content=[], **self.REFUSE),
                       "claude-sonnet-5": _reply(content=[], **self.REFUSE)})
        with caplog.at_level(logging.WARNING):
            assert gf.call_llm("hi", model="claude-opus-5-5") == ""
        assert "stop_reason=refusal (category=bio)" in caplog.text
        assert "EMPTY response" not in caplog.text

    def test_refusal_falls_back_down_the_chain(self, fake):
        from job_agent.config import FALLBACK_CHAIN
        client = fake(by_model={
            "claude-opus-5-5": _reply(content=[], **self.REFUSE),
            "claude-opus-5": _reply(content=[SimpleNamespace(type="text", text="from opus-5")]),
        })
        assert gf.call_llm("hi", model="claude-opus-5-5") == "from opus-5"
        assert client.models == ["claude-opus-5-5", FALLBACK_CHAIN["claude-opus-5-5"][0]]

    def test_mid_answer_refusal_never_returns_partial_text(self, fake):
        """A half-written cover letter must not be saved as ready."""
        partial = [SimpleNamespace(type="text", text="Dear Hiring Manager, I am excit")]
        fake(by_model={m: _reply(content=partial, **self.REFUSE)
                       for m in ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5")})
        assert gf.call_llm("hi", model="claude-opus-5-5") == ""

    def test_reasoning_extraction_is_not_retried(self, fake):
        """Anthropic's guidance: these declines are not retried on another model."""
        client = fake(by_model={"claude-opus-5-5": _reply(
            content=[], stop_reason="refusal",
            stop_details=SimpleNamespace(category="reasoning_extraction"))})
        assert gf.call_llm("hi", model="claude-opus-5-5") == ""
        assert client.models == ["claude-opus-5-5"]

    def test_fallback_is_one_level_deep_despite_chain_cycles(self, fake):
        """FALLBACK_CHAIN has cycles (sonnet-5 -> opus-5-5 -> sonnet-5). With
        every model refusing, the walk must stop instead of recursing."""
        from job_agent.config import FALLBACK_CHAIN
        client = fake(content=[], **self.REFUSE)
        assert gf.call_llm("hi", model="claude-sonnet-5") == ""
        assert client.models == ["claude-sonnet-5", *FALLBACK_CHAIN["claude-sonnet-5"]]

    def test_every_attempt_is_billed(self, fake):
        """The refused call burned tokens too; cost accounting must see them."""
        fake(by_model={
            "claude-opus-5-5": _reply(content=[], tin=100, tout=40, **self.REFUSE),
            "claude-opus-5": _reply(tin=100, tout=60),
        })
        usage = {}
        gf.call_llm("hi", model="claude-opus-5-5", usage_out=usage)
        assert usage == {"input_tokens": 200, "output_tokens": 100}


class TestUsage:

    def test_real_usage_reported_to_caller(self, fake):
        fake()
        usage = {}
        gf.call_llm("hi", model="claude-opus-5-5", usage_out=usage)
        assert usage == {"input_tokens": 11, "output_tokens": 7}


class TestFallbackConfig:
    def test_every_claude_fallback_is_priced(self):
        """Cost accounting prices whichever model actually served a call; an
        unpriced fallback books real spend as $0."""
        from job_agent.cascade_filter import MODEL_PRICING
        from job_agent import config as c
        chains = [c.CASCADE_STAGE1_FALLBACKS, c.CASCADE_STAGE2_FALLBACKS,
                  c.CASCADE_STAGE3_FALLBACKS, *c.FALLBACK_CHAIN.values()]
        named = {m for chain in chains for m in chain if m.startswith("claude-")}
        named |= {k for k in c.FALLBACK_CHAIN if k.startswith("claude-")}
        missing = sorted(m for m in named if m not in MODEL_PRICING)
        assert not missing, missing

    def test_stage3_fallback_never_repeats_the_primary(self):
        from job_agent import config as c
        assert c.CASCADE_STAGE3_MODEL not in c.CASCADE_STAGE3_FALLBACKS

    def test_the_judge_is_fable_52_backed_first_by_fable_51(self):
        """Moved 2026-11-16 after a judge replay. Fable 5.1 draws
        from a separate rate-limit pool, so it answers first when 5.2 cannot;
        a refusal goes down FALLBACK_CHAIN instead (same classifiers in 5.1)."""
        from job_agent import config as c
        from job_agent.cascade_filter import _cost_for_model
        assert c.CASCADE_STAGE3_MODEL == "claude-fable-5-2"
        assert c.CASCADE_STAGE3_FALLBACKS[0] == "claude-fable-5-1"
        assert c.MODEL_EFFORT.get(c.CASCADE_STAGE3_MODEL) == "high"
        assert c.FALLBACK_CHAIN["claude-fable-5-2"][0].startswith("claude-opus")
        assert _cost_for_model("claude-fable-5-2", 1_000_000, 1_000_000) == pytest.approx(48.0)

    def test_generation_model_is_current_opus(self):
        from job_agent.config import DEFAULT_GENERATION_MODEL, MODEL_EFFORT
        assert DEFAULT_GENERATION_MODEL == "claude-opus-5-5"
        assert MODEL_EFFORT.get(DEFAULT_GENERATION_MODEL) == "xhigh"
