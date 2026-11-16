"""
The in-app Assistant — a tool-using Claude agent over the whole app.

`stream_chat_response()` is an async generator yielding SSE-ready event dicts:
  {"type": "text",        "text": "..."}            — streamed answer tokens
  {"type": "tool_start",  "name": "...", "input": {...}}
  {"type": "tool_result", "name": "...", "summary": "..."}
  {"type": "done",        "full_text": "..."}
  {"type": "error",       "message": "..."}

Design notes:
- Model: CHAT_MODEL (claude-sonnet-5) with adaptive thinking (no sampling
  params — the 5-series rejects them). Streaming so the UI renders tokens as they arrive.
- Tool loop: assistant content blocks (INCLUDING thinking blocks) are passed
  back verbatim between tool rounds — required for tool use with adaptive
  thinking. Across user turns we persist plain text only.
- Tools execute via chat_tools.execute_tool (never raises). The slow one
  (generate_cv) runs in a worker thread so the event loop keeps streaming.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncIterator, Optional

from .chat_tools import anthropic_tool_specs, execute_tool
from .config import METRICS_BANK_TEXT

logger = logging.getLogger(__name__)

# Sonnet for the CHAT loop: orchestration + summarization is squarely in
# Sonnet's wheelhouse at 40% of Opus pricing — and the chat's cost problem was
# never model IQ, it was context volume (see caching + trims below). The
# drafting tools the agent calls still use the premium generation model.
# This loop streams text_delta only, ignores thinking deltas, and preserves
# all assistant blocks (incl. thinking) between tool rounds.
CHAT_MODEL = "claude-sonnet-5"
MAX_TOOL_ROUNDS = 8          # hard stop against tool loops
MAX_TOKENS_PER_TURN = 4096
HISTORY_TURNS = 12           # replayed turns (was 30 — history compounds per round)
HISTORY_MSG_CHARS = 4000     # cap per replayed message (tables/drafts are huge)
TOOL_RESULT_CHARS = 8000     # cap per tool result fed back (was 30000)

_async_client = None


def _get_client():
    """Lazy AsyncAnthropic client (None if SDK/key unavailable)."""
    global _async_client
    if _async_client is None:
        try:
            from anthropic import AsyncAnthropic
            _async_client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY
        except Exception as e:
            logger.error(f"Chat agent: Anthropic client unavailable: {e}")
            return None
    return _async_client


SYSTEM_PROMPT_TEMPLATE = """You are the candidate's Assistant — the in-app copilot of "AI Job Agent", a job-search pipeline that scrapes company job boards + job sites, AI-filters roles into tiers, tailors CVs/cover letters, and tracks applications on a Kanban board.

You work for the candidate. The candidate's verified career metrics (the only career facts you may state):
{metrics_bank}

<app_model>
- Jobs live in tiers: "strong" (clear fit — top priority), "worth_a_look" (adjacent/stretch), or rejected (tier null). Each job has match_level, confidence, AI reasoning, and may have a generated CV/cover letter (cv_path/cl_path point under cv/ and cl/).
- The Kanban (applications) tracks statuses: To Apply → Applied → Screening Call → Interview → Final Interview → Offer/Accepted, plus Rejected, Ghosted and I Withdrew (the candidate ended it).
- "Reviewed" means the candidate has triaged the job (tracked or dismissed it).
- LOCATION PRIORITY: follow the location order and caps configured in the candidate profile. The tiers already encode them.
- SENIORITY: the tiers already encode the level policy written in the candidate profile. Roles the pipeline capped at worth_a_look are never primary targets; roles tiered strong are the priority.
- The pipeline runs automatically on a schedule; the review queue is where new matches wait.
</app_model>

<live_search>
You are an ACTIVE search participant, not just a database reader. Live tools:
- live_search_jobs — fresh LinkedIn/Indeed/Google search right now (free, ~10-30s). Default when the candidate wants "anything new" beyond the DB.
- search_company_board — query one company's ATS board live ("what's open at <company> today").
- search_google_jobs_live — reaches boards we can't scrape (large corporate career sites, boutique studios). BUDGETED: 250/month — one targeted query, never sweeps; prefer live_search_jobs first.
- get_warm_paths — imported networks at a company; a referral beats cold outreach. A contact from someone else's imported network → ask that person for an intro, never message the contact directly.
- find_role_contacts — hiring manager/recruiter research with citations; only VERIFIED emails are sendable.
Flow for "find me X": DB first (query_jobs — instant), then live_search_jobs if the candidate wants fresher, then search_google_jobs_live only for named targets we can't otherwise reach. Offer save=true when live results look promising so they enter the pipeline.
</live_search>

<how_to_work>
- Use tools to ground EVERY factual claim about jobs, applications, or stats — never guess IDs, counts, or statuses.
- Before recommending the candidate apply somewhere, run check_already_applied for the company.
- When the candidate asks "what should I do today": get_pipeline_stats first, then surface the freshest unreviewed strong matches (query_jobs tier=strong reviewed=false), and flag stale 'Applied' items needing follow-up.
- generate_cv is slow (~1-2 min) and uses the premium model — confirm before running it on more than one job, and never regenerate documents that already exist unless the candidate explicitly asks.
- Destructive-ish actions (dismiss_job, moving Kanban cards) — do them when asked clearly; if ambiguous, ask one short clarifying question first.
- Be concise and concrete: job titles, companies, links. Tables for lists of 3+. The candidate's time is the scarce resource — lead with the action, not the explanation.
- You may give career advice (positioning, follow-up timing, which tier deserves attention) grounded in the verified metrics above.
</how_to_work>

<power_moves>
These are your signature workflows — offer them proactively when relevant:
1. FOLLOW-UP SESSION: list_stale_applications -> pick the most promising 3-5 (recognizable companies, recent-ish, strong original fit) -> draft_followup for each -> present drafts for the candidate to copy/send. Never send anything yourself; you only draft.
2. OUTREACH: for a strong match the candidate is excited about, draft_outreach (connection note + InMail) referencing the actual JD. Suggest this especially for fresh postings (<7 days) where early contact matters most.
3. APPLY KIT: when the candidate is about to apply, get_apply_kit — give the link, posting age, doc paths, the standard form answers, and flag any earlier contact with the company. If no CV exists yet, offer generate_cv first.
4. DAILY BRIEFING: get_pipeline_stats + freshest unreviewed strong matches + stale-Applied count -> one compact action plan: apply-to (top 2-3), follow-up-on (top 2-3), and what to ignore.
5. DOCUMENTS: when a job has cv_url/cl_url, ALWAYS present them as markdown download links, e.g. [Download CV](/api/jobs/123/cv) — they download right in the chat. Documents attach to the job, so once a job is tracked they appear on its Kanban card automatically; offer track_job if it isn't tracked yet.
6. INTERVIEW PREP: when the candidate has a Screening Call / Interview / Final Interview on the Kanban, or asks to prep for a company, interview_prep(company, title, job_id if it's in the pipeline) — likely questions, the candidate's strongest REAL stories mapped to them (grounded in the candidate's verified transcript, never invented), a company angle, and smart questions to ask back. Cards at those stages already auto-generate a prep sheet on their Kanban panel; use this tool for on-demand prep or to talk through answers with the candidate. After generating, offer to roleplay: ask one question, wait for the answer, give brief feedback and suggest which verified story fits best.
The drafting tools cost a few cents each — batch sensibly (3-5 drafts per session, never the whole stale queue at once) and say roughly what a batch costs when a big one is requested.
</power_moves>

Today's date: {today}."""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        metrics_bank=METRICS_BANK_TEXT,
        today=datetime.now().strftime("%A, %B %d, %Y"),
    )


def _history_to_messages(history: list[dict]) -> list[dict]:
    """Replay persisted plain-text turns, coercing to alternating roles."""
    msgs: list[dict] = []
    for m in history[-HISTORY_TURNS:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        content = content[:HISTORY_MSG_CHARS]
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + content  # merge consecutive same-role
        else:
            msgs.append({"role": role, "content": content})
    # The API requires the first message to be from the user
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def _cached_system() -> list:
    """System prompt as a cacheable block (static across turns and rounds)."""
    return [{"type": "text", "text": build_system_prompt(),
             "cache_control": {"type": "ephemeral"}}]


def _cached_tools() -> list:
    """Tool specs with a cache breakpoint on the last one (static)."""
    specs = anthropic_tool_specs()
    if specs:
        specs[-1] = {**specs[-1], "cache_control": {"type": "ephemeral"}}
    return specs


def _with_cache_breakpoint(messages: list) -> list:
    """Mark the last user message's final block as a cache breakpoint, so the
    next round/turn re-reads the whole conversation prefix from cache (~10% of
    input price) instead of re-billing it in full — THE fix for tool-loop turns
    whose context compounds every round."""
    if not messages:
        return messages
    out = list(messages)
    last = out[-1]
    if not (isinstance(last, dict) and last.get("role") == "user"):
        return out
    content = last.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        content = list(content)
        content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
        out[-1] = {**last, "content": content}
    return out


def _summarize_tool_result(result: dict) -> str:
    """One-line summary of a tool result for the UI event stream."""
    if not isinstance(result, dict):
        return str(result)[:120]
    if "error" in result:
        return f"error: {result['error']}"[:160]
    if "count" in result:
        return f"{result['count']} result(s)"
    if "status" in result:
        return str(result["status"])
    return "ok"


async def stream_chat_response(
    db, user_message: str, session_id: str = "candidate",
) -> AsyncIterator[dict]:
    """Run one user turn through the tool-use loop, yielding stream events."""
    client = _get_client()
    if client is None:
        yield {"type": "error", "message": "Anthropic client unavailable (check ANTHROPIC_API_KEY)."}
        return

    db.save_chat_message("user", user_message, session_id=session_id)
    messages = _history_to_messages(db.get_chat_messages(session_id=session_id))
    if not messages or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": user_message})

    full_text_parts: list[str] = []
    try:
        for _round in range(MAX_TOOL_ROUNDS):
            async with client.messages.stream(
                model=CHAT_MODEL,
                max_tokens=MAX_TOKENS_PER_TURN,
                system=_cached_system(),
                thinking={"type": "adaptive"},
                tools=_cached_tools(),
                messages=_with_cache_breakpoint(messages),
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
                        full_text_parts.append(event.delta.text)
                        yield {"type": "text", "text": event.delta.text}
                    elif event.type == "content_block_start" and getattr(event.content_block, "type", "") == "tool_use":
                        yield {"type": "tool_start", "name": event.content_block.name, "input": {}}
                final = await stream.get_final_message()

            u = getattr(final, "usage", None)
            if u:
                logger.info(
                    f"chat round: in={getattr(u, 'input_tokens', '?')} "
                    f"cache_write={getattr(u, 'cache_creation_input_tokens', 0)} "
                    f"cache_read={getattr(u, 'cache_read_input_tokens', 0)} "
                    f"out={getattr(u, 'output_tokens', '?')}"
                )

            # Preserve ALL assistant blocks (incl. thinking) between tool rounds
            messages.append({"role": "assistant", "content": final.content})

            if final.stop_reason != "tool_use":
                break

            tool_results = []
            for block in final.content:
                if getattr(block, "type", "") != "tool_use":
                    continue
                result = await asyncio.to_thread(execute_tool, db, block.name, dict(block.input or {}))
                yield {"type": "tool_result", "name": block.name,
                       "summary": _summarize_tool_result(result)}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str)[:TOOL_RESULT_CHARS],
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            yield {"type": "text", "text": "\n\n[Stopped: too many tool rounds — ask me to continue.]"}

        full_text = "".join(full_text_parts).strip()
        if full_text:
            db.save_chat_message("assistant", full_text, session_id=session_id)
        yield {"type": "done", "full_text": full_text}
    except Exception as e:
        logger.error(f"Chat agent error: {e}")
        yield {"type": "error", "message": str(e)[:300]}
